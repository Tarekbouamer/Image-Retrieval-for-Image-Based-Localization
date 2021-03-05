import argparse
from os import makedirs, path
import shutil
import time
from collections import OrderedDict

import numpy as np
from tqdm import tqdm


import torch
import torch.optim as optim
import torch.utils.data as data
import torch.distributed as distributed

import tensorboardX as tensorboard

# configuration
from cirtorch.configuration import load_config, config_to_string, DEFAULTS as DEFAULT_CONFIGS

# backbones
import cirtorch.backbones as models
from cirtorch.backbones.url import model_urls, model_urls_cvut
from cirtorch.backbones.util import load_state_dict_from_url, init_weights

# dataset
from cirtorch.datasets.localFeatures import MegaDepthDataset, ISSTransform, DistributedARBatchSampler, hpatches, iss_collate_fn
from cirtorch.datasets.localFeatures import HPacthes, ISSTestTransform, HP_INPUTS

# data augmentation
from cirtorch.datasets.augmentation import RandomAugmentation

# modules
from cirtorch.modules.fpn import FPN, FPNBody
from cirtorch.modules.utils import OUTPUT_DIM
from cirtorch.modules.heads.local_head import localHead

# algos
from cirtorch.algos.LF_algo import localFeatureLoss, localFeatureAlgo

# models
from cirtorch.models.LF_net import localNet, NETWORK_TRAIN_INPUTS

# utils
from cirtorch.utils.download import download_train, download_test
from cirtorch.utils.misc import config_to_string, scheduler_from_config, norm_act_from_config, freeze_params, \
    all_reduce_losses, NORM_LAYERS, OTHER_LAYERS

from cirtorch.utils.general import htime
from cirtorch.utils.options import test_datasets_names
from cirtorch.utils import logging
from cirtorch.utils.snapshot import save_snapshot, resume_from_snapshot, pre_train_from_snapshots
from cirtorch.utils.meters import AverageMeter
from cirtorch.utils.parallel import DistributedDataParallel, PackedSequence

from cirtorch.utils.evaluation.HPatchesEval import run_descriptor_evaluation

LAYERS = NORM_LAYERS + OTHER_LAYERS


def log_debug(msg, *args, **kwargs):
    if distributed.get_rank() == 0:
        logging.get_logger().debug(msg, *args, **kwargs)


def log_info(msg, *args, **kwargs):
    if distributed.get_rank() == 0:
        logging.get_logger().info(msg, *args, **kwargs)


def make_parser():
    # ArgumentParser
    parser = argparse.ArgumentParser(description='PyTorch CNN local features description')

    # Export directory, training and val datasets, test datasets
    parser.add_argument("--local_rank", type=int)

    parser.add_argument('--directory', metavar='EXPORT_DIR',
                        help='destination where trained network should be saved')

    parser.add_argument('--data', metavar='EXPORT_DIR',
                        help='destination where trained network should be saved')

    parser.add_argument("--config", metavar="FILE", type=str, help="Path to configuration file",
                        default='./cirtorch/configuration/defaults/local_config.ini')

    parser.add_argument("--eval", action="store_true", help="Do a single validation run")

    parser.add_argument('--resume', metavar='FILENAME', type=str,
                        help='name of the latest checkpoint (default: None)')

    parser.add_argument("--pre_train", metavar="FILE", type=str, nargs="*",
                        help="Start from the given pre-trained snapshots, overwriting each with the next one in the list. "
                             "Snapshots can be given in the format '{module_name}:{path}', where '{module_name} is one of "
                             "'body', 'rpn_head', 'roi_head' or 'sem_head'. In that case only that part of the network "
                             "will be loaded from the snapshot")

    parser.add_argument('--test-datasets', '-td', metavar='DATASETS')

    args = parser.parse_args()

    for arg in vars(args):
        print(' {}\t  {}'.format(arg, getattr(args, arg)))

    print('\n ')

    return parser


def make_dir(config, directory):
    # Create export dir name if it doesnt exist in your experiment folder
    extension = "{}".format(config["dataloader"].get("training_dataset"))
    extension += "_{}".format(config["body"].get("arch"))

    if config["fpn"].getboolean("fpn"):
        extension += "_FPN"

    extension += "_{}_gamma{:.2f}_em{:.2f}_cm{:.2f}".format(config["local"].get("loss"),
                                                config["local"].getfloat("gamma"),
                                                config["local"].getfloat("epipolar_margin"),
                                                config["local"].getfloat("cyclic_margin"))

    extension += "_{}_lr{:.1e}_wd{:.1e}".format(config["optimizer"].get("type"),
                                                config["optimizer"].getfloat("lr"),
                                                config["optimizer"].getfloat("weight_decay"))

    extension += "_max_num{}".format(config["dataloader"].getint("max_per_scene"))
    extension += "_bsize{}_uevery{}_imsize{}".format(config["dataloader"].getint("train_batch_size"),
                                                     config["dataloader"].getint("update_every"),
                                                     config["dataloader"].getint("train_longest_max_size"))

    directory = path.join(directory, extension)

    if not path.exists(directory):
        log_debug("Create experiment path  from %s", directory)
        makedirs(directory)

    return directory


def make_config(args):
    log_info("Loading configuration from %s", args.config)

    config = load_config(args.config, DEFAULT_CONFIGS["base"])

    log_info("\n%s", config_to_string(config))

    return config


def make_dataloader(args, config, rank=None, world_size=None):
    general_config = config["general"]
    data_config = config["dataloader"]

    # Data Loader
    log_debug("Creating dataloaders for dataset in %s", args.data)

    # Training dataloader

    train_tf = ISSTransform(shortest_size=data_config.getint("train_shortest_size"),
                            longest_max_size=data_config.getint("train_longest_max_size"),
                            random_flip=data_config.getboolean("random_flip"),
                            random_scale=data_config.getstruct("random_scale"))

    train_db = MegaDepthDataset(root_dir=args.data,
                                name=data_config.get("training_dataset"),
                                split='train',
                                transform=train_tf,
                                max_per_scene=data_config.getint("max_per_scene"),
                                num_kpt=data_config.getint("num_kpt"),
                                kpt_type=data_config.get("kpt_type"),
                                prune_kpt=data_config.getboolean("prune_kpt"))

    train_sampler = DistributedARBatchSampler(train_db,
                                              batch_size=data_config.getint("train_batch_size"),
                                              num_replicas=world_size,
                                              rank=rank,
                                              drop_last=True)

    train_dl = data.DataLoader(train_db,
                               batch_sampler=train_sampler,
                               collate_fn=iss_collate_fn,
                               pin_memory=True,
                               num_workers=data_config.getint("num_workers"),
                               shuffle=False)

    # Validation dataloader

    val_tf = ISSTransform(shortest_size=data_config.getint("train_shortest_size"),
                          longest_max_size=data_config.getint("train_longest_max_size"),
                          random_flip=data_config.getboolean("random_flip"),
                          random_scale=data_config.getstruct("random_scale"))

    val_db = MegaDepthDataset(root_dir=args.data,
                              name=data_config.get("training_dataset"),
                              split='val',
                              transform=val_tf,
                              max_per_scene=data_config.getint("max_per_scene"),
                              num_kpt=data_config.getint("num_kpt"),
                              kpt_type=data_config.get("kpt_type"),
                              prune_kpt=data_config.getboolean("prune_kpt"))

    val_sampler = DistributedARBatchSampler(val_db,
                                            batch_size=data_config.getint("train_batch_size"),
                                            num_replicas=world_size,
                                            rank=rank,
                                            drop_last=True)

    val_dl = data.DataLoader(val_db,
                             batch_sampler=val_sampler,
                             collate_fn=iss_collate_fn,
                             pin_memory=True,
                             num_workers=data_config.getint("num_workers"),
                             shuffle=False)

    return train_dl, val_dl


def make_model(args, config):
    
    # parse params with default values
    body_config = config["body"]
    fpn_config = config["fpn"]
    local_config = config["local"]
    data_config = config["dataloader"]

    # BN + activation
    norm_act_static, norm_act_dynamic = norm_act_from_config(body_config)

    # Create backbone
    log_debug("Creating backbone model %s", body_config.get("arch"))

    body_fn = models.__dict__[body_config.get("arch")]
    body_params = body_config.getstruct("body_params") if body_config.get("body_params") else {}
    body = body_fn(norm_act=norm_act_static, config=body_config, **body_params)

    if body_config.getboolean("pretrained"):
        arch = body_config.get("arch")

        # vgg with bn or without
        if body_config.get("arch").startswith("vgg"):
            if body_config["normalization_mode"] != 'off':
                arch = body_config.get("arch") + '_bn'

        # Download pre trained model
        log_debug("Downloading pre - trained model weights ")

        if body_config.get("source_url") == "cvut":
            if body_config.get("arch") not in model_urls_cvut:
                raise ValueError(" body arch not found in cvut witch  source_url = pytorch")
            log_info("Downloading from m ", model_urls_cvut[arch])
            state_dict = load_state_dict_from_url(model_urls_cvut[arch], progress=True)

        elif body_config.get("source_url") == "pytorch":
            if body_config.get("arch") not in model_urls:
                raise ValueError(" body arch not found in pytorch ")
            state_dict = load_state_dict_from_url(model_urls[arch], progress=True)

        else:
            raise ValueError(" try source_url = cvut  or pytorch  ")

        # Convert model to unified format and save it
        converted_model = body.convert(state_dict)
        folder = args.directory + "/image_net"

        if not path.exists(folder):
            log_debug("Create path to save pretrained backbones: %s ", folder)
            makedirs(folder)

        body_path = folder + "/" + arch + ".pth"
        log_debug("Saving pretrained backbones in : %s ", body_path)
        torch.save(converted_model, body_path)

        # Load  converted weights to model
        body.load_state_dict(torch.load(body_path, map_location="cpu"))

        # Freeze modules in backbone
        for n, m in body.named_modules():
            for mod_id in body_config.getstruct("num_frozen"):
                if ("mod%d" % mod_id) in n:
                    freeze_params(m)

    else:
        log_info("Initialize body to train from scratch")
        init_weights(body, body_config)

    # Feature pyramids
    if fpn_config.getboolean("fpn"):
        
        # Create FPN
        body_channels = body_config.getstruct("out_channels")

        fpn_inputs = fpn_config.getstruct("inputs")

        fpn = FPN([body_channels[inp] for inp in fpn_inputs],
                  fpn_config.getint("out_channels"),
                  fpn_config.getint("extra_scales"),
                  norm_act_static,
                  fpn_config.get("interpolation"))

        body = FPNBody(body, fpn, fpn_inputs)

        output_dim = fpn_config.getint("out_channels")
    else:
        output_dim = OUTPUT_DIM[body_config.get("arch")]


    # Create Local features
    local_loss = localFeatureLoss(name=local_config.get("loss"),
                                  gamma=local_config.getfloat("gamma"),
                                  epipolar_margin=local_config.getfloat("epipolar_margin"),
                                  cyclic_margin=local_config.getfloat("cyclic_margin"))

    local_algo = localFeatureAlgo(loss=local_loss,
                                  min_level=local_config.getint("fpn_min_level"),
                                  fpn_levels=local_config.getint("fpn_levels"))

    local_head = localHead(dim=output_dim,
                           embedding_size=local_config.getint("embedding_size"))
    
    if local_config.getint("embedding_size"):
        output_dim = local_config.getint("embedding_size")
    
    # Data augmentation
    augment = RandomAugmentation(rgb_mean=data_config.getstruct("rgb_mean"),
                                 rgb_std=data_config.getstruct("rgb_std"))

    # Create a generic Local features network
    net = localNet(body, local_algo, local_head, augment=augment)

    return net, output_dim


def make_optimizer(model, config, epoch_length):

    body_config = config["body"]

    optimizer_config = config["optimizer"]
    scheduler_config = config["scheduler"]

    # Base learning rate and weight decay
    LR = optimizer_config.getfloat("lr")
    WEIGHT_DECAY = optimizer_config.getfloat("weight_decay")

    # Tunning learning rate and weight decay
    lr_coefs = optimizer_config.getstruct("lr_coefs")
    weight_decay_coefs = optimizer_config.getstruct("weight_decay_coefs")

    # Separate classifier parameters from all others
    net_params = []
    local_head_params = []
    
    for k, v in model.named_parameters():
        if k.find("local_head") != -1:
            if v.requires_grad:
                local_head_params.append(v)
        else:
            if v.requires_grad:
                net_params.append(v)

    # Set-up optimizer hyper-parameters
    parameters = [
        {
            "params": net_params,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY
        },
        {
            "params": local_head_params,
            "lr": LR * lr_coefs["local_head"],
            "weight_decay": WEIGHT_DECAY * weight_decay_coefs["local_head"]
        }
    ]

    # Select optimizer
    if optimizer_config.get("type") == 'SGD':
        optimizer = optim.SGD(parameters,
                              weight_decay=optimizer_config.getfloat("weight_decay"),
                              nesterov=optimizer_config.getboolean("nesterov"))
    elif optimizer_config.get("type") == 'Adam':
        optimizer = optim.Adam(parameters)
    else:
        raise KeyError("unrecognized optimizer {}".format(optimizer_config["type"]))

    # Set scheduler
    scheduler = scheduler_from_config(scheduler_config, optimizer, epoch_length)

    assert scheduler_config.get("update_mode") in ("batch", "epoch")
    batch_update = scheduler_config.get("update_mode") == "batch"
    total_epochs = scheduler_config.getint("epochs")

    return optimizer, scheduler, parameters, batch_update, total_epochs


def train(model, config, dataloader, optimizer, scheduler, meters, **varargs):

    # Create tuples for training
    data_config = config["dataloader"]

    # Switch to train mode
    model.train()
    dataloader.batch_sampler.set_epoch(varargs["epoch"])
    optimizer.zero_grad()
    global_step = varargs["global_step"]
    loss_weights = varargs["loss_weights"]

    data_time_meter     = AverageMeter((), meters["loss"].momentum)
    batch_time_meter    = AverageMeter((), meters["loss"].momentum)

    data_time = time.time()

    for it, batch in enumerate(dataloader):
        #Upload batch
        batch = {k: batch[k].cuda(device=varargs["device"], non_blocking=True) for k in NETWORK_TRAIN_INPUTS}

        # Measure data loading time
        data_time_meter.update(torch.tensor(time.time() - data_time))

        # Update scheduler
        global_step += 1
        if varargs["batch_update"]:
            scheduler.step(global_step)

        batch_time = time.time()

        # Run network
        losses, _ = model(**batch, do_loss=True, do_augmentaton=True)
        
        distributed.barrier()

        losses = OrderedDict((k, v.mean()) for k, v in losses.items())
        losses["loss"] = sum(w * l for w, l in zip(loss_weights, losses.values()))

        losses["loss"].backward()

        optimizer.step()
        optimizer.zero_grad()


        # Gather from all workers
        losses = all_reduce_losses(losses)

        # Update meters
        with torch.no_grad():
            for loss_name, loss_value in losses.items():
                meters[loss_name].update(loss_value.cpu())

        batch_time_meter.update(torch.tensor(time.time() - batch_time))

        # Clean-up
        del batch, losses

        # Log
        if varargs["summary"] is not None and (it + 1) % varargs["log_interval"] == 0:
            logging.iteration(
                varargs["summary"], "train", global_step,
                varargs["epoch"] + 1, varargs["num_epochs"],
                it + 1, len(dataloader),
                OrderedDict([
                    ("lr_body", scheduler.get_lr()[0] * 1e6),
                    ("lr_local", scheduler.get_lr()[1] * 1e6),
                    ("loss", meters["loss"]),
                    ("local_loss", meters["local_loss"]),
                    ("data_time", data_time_meter),
                    ("batch_time", batch_time_meter)
                ])
            )

        data_time = time.time()

    return global_step


def validate(model, config, dataloader, **varargs):

    # create tuples for validation
    data_config = config["dataloader"]

    # Switch to eval mode
    model.eval()
    dataloader.batch_sampler.set_epoch(varargs["epoch"])

    loss_weights = varargs["loss_weights"]

    loss_meter = AverageMeter(())
    data_time_meter = AverageMeter(())
    batch_time_meter = AverageMeter(())

    data_time = time.time()

    for it, batch in enumerate(dataloader):
        with torch.no_grad():

            #Upload batch
            batch = {k: batch[k].cuda(device=varargs["device"], non_blocking=True) for k in NETWORK_TRAIN_INPUTS}

            data_time_meter.update(torch.tensor(time.time() - data_time))

            batch_time = time.time()

            # Run network
            losses, _ = model(**batch, do_loss=True, do_prediction=True, do_augmentation=True)

            losses = OrderedDict((k, v.mean()) for k, v in losses.items())
            losses = all_reduce_losses(losses)
            loss = sum(w * l for w, l in zip(loss_weights, losses.values()))

            # Update meters
            loss_meter.update(loss.cpu())
            batch_time_meter.update(torch.tensor(time.time() - batch_time))

            del loss, losses, batch

        # Log batch
        if varargs["summary"] is not None and (it + 1) % varargs["log_interval"] == 0:
            logging.iteration(
                None, "val", varargs["global_step"],
                varargs["epoch"] + 1, varargs["num_epochs"],
                it + 1, len(dataloader),
                OrderedDict([
                    ("loss", loss_meter),
                    ("data_time", data_time_meter),
                    ("batch_time", batch_time_meter)]
                    )
            )

        data_time = time.time()

    return loss_meter.mean


def test(args, config, model, rank=None, world_size=None, **varargs):

    log_debug('Evaluating network on test datasets...')

    # Eval mode
    model.eval()
    data_config = config["dataloader"]

    # Average score
    avg_score = 0.0

    # Evaluate on test datasets
    list_datasets = data_config.getstruct("test_datasets")

    for dataset in list_datasets:

        start = time.time()
        Avg_F_score = 0.
        
        log_debug('{%s}: Loading Dataset', dataset)

        for split in {"i", "v"}:

            test_tf = ISSTestTransform(shortest_size=data_config.getint("test_shortest_size"),
                                        longest_max_size=data_config.getint("test_longest_max_size"))

            test_db = HPacthes(root_dir=args.data,
                            name=dataset,
                            split=split,
                            transform=test_tf,
                            num_kpt=data_config.getint("num_kpt"),
                            kpt_type=data_config.get("kpt_type"),
                            prune_kpt=data_config.getboolean("prune_kpt"))

            test_sampler = DistributedARBatchSampler(data_source=test_db,
                                                    batch_size=data_config.getint("test_batch_size"),
                                                    num_replicas=world_size,
                                                    rank=rank,
                                                    drop_last=True,
                                                    shuffle=False)

            test_dl = torch.utils.data.DataLoader(test_db,
                                                batch_sampler=test_sampler,
                                                collate_fn=iss_collate_fn,
                                                pin_memory=True,
                                                num_workers=data_config.getstruct("num_workers"),
                                                shuffle=False)
            
            log_debug('{%s}: Feature Extraction %s...', dataset, split)

            for it, batch in tqdm(enumerate(test_dl), total=len(test_dl)):
                with torch.no_grad():
                    
                    # Upload batch
                    img1_path, img2_path = batch["img1_path"][0], batch["img2_path"][0]

                    batch = {k: batch[k].cuda(device=varargs["device"], non_blocking=True) for k in HP_INPUTS}

                    # Run Image 1
                    _, pred1 = model(img=batch["img1"], kpts=batch["kpts1"], do_prediction=True, do_augmentaton=True)
                    distributed.barrier()

                    kpts1, _ = batch["kpts1"].contiguous
                    pred1, _ = pred1["local_pred"].contiguous

                    with open(img1_path + ".npz", 'wb') as output_file:
                        np.savez(output_file, 
                                keypoints=kpts1.squeeze(0).cpu().numpy(),
                                descriptors=pred1.squeeze(0).cpu().numpy())

                    # Run Image 2
                    _, pred2 = model(img=batch["img2"], kpts=batch["kpts2"], do_prediction=True, do_augmentaton=True)
                    distributed.barrier()

                    kpts2, _ = batch["kpts2"].contiguous
                    pred2, _ = pred2["local_pred"].contiguous

                    with open(img2_path + ".npz", 'wb') as output_file:
                        np.savez(output_file, 
                                keypoints=kpts2.squeeze(0).cpu().numpy(),
                                descriptors=pred2.squeeze(0).cpu().numpy())

            log_debug('{%s}: Run Evalution %s...', dataset, split)

            # run evaluation
            config = {'num_kp': 1000,   
                    'correctness_threshold': 3, 
                    'max_mma_threshold': 10}

            H_estimation, Precision, Recall, MMA= run_descriptor_evaluation (config, test_dl ) 
            
            # compute F score 
            F_score = 2*((Precision * Recall)/(Precision + Recall))
            
            Avg_F_score += 0.5 * F_score
            
            np.set_printoptions(precision=3)

            log_info('{%s}: H_estimation_%s = %s', dataset , split, format(H_estimation, '.3f') )
            log_info('{%s}: Precision_%s    = %s', dataset , split, format(Precision, '.3f')    )
            log_info('{%s}: Recall_%s       = %s', dataset , split, format(Recall, '.3f')       )
            log_info('{%s}: MMA_%s          = %s', dataset , split, MMA                         )
            log_info('{%s}: F_score_%s      = %s', dataset , split, format(F_score, '.3f')      )

        log_info('{%s}: Running time     = %s', dataset , htime(time.time()-start)   )
        log_info('{%s}: Avg_F_score      = %s', dataset , format(Avg_F_score, '.3f') )

    return Avg_F_score


def save_checkpoint(state, is_best, directory):
    filename = path.join(directory, 'model_epoch%d.pth.tar' % state['epoch'])
    torch.save(state, filename)
    if is_best:
        filename_best = path.join(directory, 'model_best.pth.tar')
        shutil.copyfile(filename, filename_best)


def main(args):

    # Initialize multi-processing
    distributed.init_process_group(backend='nccl', init_method='env://')
    device_id, device = args.local_rank, torch.device(args.local_rank)
    rank, world_size = distributed.get_rank(), distributed.get_world_size()
    torch.cuda.set_device(device_id)

    # Load configuration
    config = make_config(args)

    # Experiment Path
    exp_dir = make_dir(config, args.directory)

    # Initialize logging
    if rank == 0:
        logging.init(exp_dir, "training" if not args.eval else "eval")
        summary = tensorboard.SummaryWriter(args.directory)
    else:
        summary = None

    body_config = config["body"]
    optimizer_config = config["optimizer"]

    # Load data
    train_dataloader, val_dataloader = make_dataloader(args, config, rank, world_size)

    # Initialize model
    if body_config.getboolean("pretrained"):
        log_debug("Use pre-trained model %s", body_config.get("arch"))
    else:
        log_debug("Initialize model to train from scratch %s". body_config.get("arch"))

    # Load model
    model, output_dim = make_model(args, config)

    # Resume / Pre_Train
    if args.resume:
        assert not args.pre_train, "resume and pre_train are mutually exclusive"
        log_debug("Loading snapshot from %s", args.resume)
        snapshot = resume_from_snapshot(model, args.resume, ["body", "local_head"])
    elif args.pre_train:
        assert not args.resume, "resume and pre_train are mutually exclusive"
        log_debug("Loading pre-trained model from %s", args.pre_train)
        pre_train_from_snapshots(model, args.pre_train, ["body", "local_head"])
    else:
        #assert not args.eval, "--resume is needed in eval mode"
        snapshot = None

    # Init GPU stuff
    torch.backends.cudnn.benchmark = config["general"].getboolean("cudnn_benchmark")
    model = DistributedDataParallel(model.cuda(device), device_ids=[device_id], output_device=device_id,
                                    find_unused_parameters=True)

    # Create optimizer & scheduler
    optimizer, scheduler, parameters, batch_update, total_epochs = make_optimizer(model, config, epoch_length=len(train_dataloader))
    if args.resume:
        optimizer.load_state_dict(snapshot["state_dict"]["optimizer"])

    # Training loop
    momentum = 1. - 1. / len(train_dataloader)
    meters = {
        "loss": AverageMeter((), momentum),
        "local_loss": AverageMeter((), momentum)
    }

    if args.resume:
        start_epoch = snapshot["training_meta"]["epoch"] + 1
        best_score = snapshot["training_meta"]["best_score"]
        global_step = snapshot["training_meta"]["global_step"]

        for name, meter in meters.items():
            meter.load_state_dict(snapshot["state_dict"][name + "_meter"])
        del snapshot
    else:
        start_epoch = 0
        best_score = {
            "val": 1000.0,
            "test": 0.0,
        }
        global_step = 0

    # Optional: evaluation only:
    if args.eval:
        log_info("Evaluation epoch %d", start_epoch - 1)

        test(args, config, model, rank=rank, world_size=world_size,
             output_dim=output_dim,
             device=device)

        log_info("Evaluation Done ..... ")

        exit(0)
    
    for epoch in range(start_epoch, total_epochs):

        log_info("Starting epoch %d", epoch + 1)

        if not batch_update:
            scheduler.step(epoch)

        score = {}
        
        # Run training
        global_step = train(model, config, train_dataloader, optimizer, scheduler, meters,
                            summary=summary,
                            batch_update=batch_update,
                            log_interval=config["general"].getint("log_interval"),
                            epoch=epoch,
                            num_epochs=total_epochs,
                            global_step=global_step,
                            output_dim=output_dim,
                            world_size=world_size,
                            rank=rank,
                            device=device,
                            loss_weights=optimizer_config.getstruct("loss_weights"))

        # Save snapshot (only on rank 0)
        if rank == 0:
            snapshot_file = path.join(exp_dir, "model_{}.pth.tar".format(epoch))

            log_debug("Saving snapshot to %s", snapshot_file)

            meters_out_dict = {k + "_meter": v.state_dict() for k, v in meters.items()}

            save_snapshot(snapshot_file, config, epoch, 0, best_score, global_step,
                          body=model.module.body.state_dict(),
                          local_head=model.module.local_head.state_dict(),
                          optimizer=optimizer.state_dict(),
                          **meters_out_dict)

        # Run validation
        if (epoch + 1) % config["general"].getint("val_interval") == 0:
            log_info("Validating epoch %d", epoch + 1)

            score['val'] = validate(model, config, val_dataloader,
                                    summary=summary,
                                    batch_update=batch_update,
                                    log_interval=config["general"].getint("log_interval"),
                                    epoch=epoch,
                                    num_epochs=total_epochs,
                                    global_step=global_step,
                                    output_dim=output_dim,
                                    world_size=world_size,
                                    rank=rank,
                                    device=device,
                                    loss_weights=optimizer_config.getstruct("loss_weights"))

        # Run Test
        if (epoch + 1) % config["general"].getint("test_interval") == 0:
            log_info("Testing epoch %d", epoch + 1)

            score['test'] = test(args, config, model, rank=rank, world_size=world_size,
                                 output_dim=output_dim,
                                 device=device)

            # Update the score on the last saved snapshot
            if rank == 0:
                snapshot = torch.load(snapshot_file, map_location="cpu")
                snapshot["training_meta"]["last_score"] = score
                torch.save(snapshot, snapshot_file)
                del snapshot

            if score['test'] > best_score['test']:
                best_score = score
                if rank == 0:
                    shutil.copy(snapshot_file, path.join(exp_dir, "test_model_best.pth.tar"))


if __name__ == '__main__':

    parser = make_parser()

    main(parser.parse_args())

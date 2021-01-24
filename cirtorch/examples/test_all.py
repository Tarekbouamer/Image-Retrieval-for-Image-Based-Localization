import argparse
from os import path, makedirs, path
import shutil
import time
from glob import glob

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
from cirtorch.datasets.tuples_dataset import TuplesDataset, NETWORK_INPUTS
from cirtorch.datasets.tuples_transform import TuplesTransform
from cirtorch.datasets.tuples_sampler import TuplesDistributedARBatchSampler
from cirtorch.datasets.generic import ISSDataset, ISSTestTransform, ParisOxfordTestDataset, DistributedARBatchSampler, \
    INPUTS
from cirtorch.datasets.augmentation import RandomAugmentation
from cirtorch.datasets.misc import iss_collate_fn

# modules
from cirtorch.modules.fpn import FPN, FPNBody
from cirtorch.modules.utils import OUTPUT_DIM
from cirtorch.modules.heads.ir_head import ImageRetrievalHead

# algos
from cirtorch.algos.image_retrieval import ImageRetrievalAlgo, ImageRetrievalLoss

# models
from cirtorch.models.imageretrievalnet import ImageRetrievalNet

# utils
from cirtorch.utils.download import download_train, download_test
from cirtorch.utils.misc import config_to_string, scheduler_from_config, norm_act_from_config, freeze_params, \
    all_reduce_losses, NORM_LAYERS, OTHER_LAYERS
from cirtorch.utils.evaluate import compute_map_and_print
from cirtorch.utils.general import htime
from cirtorch.utils.options import test_datasets_names
from cirtorch.utils import logging
from cirtorch.utils.snapshot import save_snapshot, resume_from_snapshot, pre_train_from_snapshots
from cirtorch.utils.meters import AverageMeter
from cirtorch.utils.parallel import DistributedDataParallel, PackedSequence

min_loss = float('inf')

LAYERS = NORM_LAYERS + OTHER_LAYERS


def log_debug(msg, *args, **kwargs):
    if distributed.get_rank() == 0:
        logging.get_logger().debug(msg, *args, **kwargs)


def log_info(msg, *args, **kwargs):
    if distributed.get_rank() == 0:
        logging.get_logger().info(msg, *args, **kwargs)


def make_parser():
    # ArgumentParser
    parser = argparse.ArgumentParser(description='PyTorch CNN Image Retrieval Training')

    # Export directory, training and val datasets, test datasets
    parser.add_argument("--local_rank", type=int)

    parser.add_argument('--directory', metavar='EXPORT_DIR',
                        help='destination where trained network should be saved')

    parser.add_argument('--data', metavar='EXPORT_DIR',
                        help='destination where trained network should be saved')

    parser.add_argument("--config", metavar="FILE", type=str, help="Path to configuration file",
                        default='./cirtorch/configuration/defaults/base.ini')

    parser.add_argument("--eval", action="store_true", help="Do a single validation run")

    parser.add_argument('--resume', metavar='FILENAME', type=str,
                        help='name of the latest checkpoint (default: None)')

    parser.add_argument("--pre_train", metavar="FILE", type=str, nargs="*",
                        help="Start from the given pre-trained snapshots, overwriting each with the next one in the list. "
                             "Snapshots can be given in the format '{module_name}:{path}', where '{module_name} is one of "
                             "'body', 'rpn_head', 'roi_head' or 'sem_head'. In that case only that part of the network "
                             "will be loaded from the snapshot")

    parser.add_argument('--test-datasets', '-td', metavar='DATASETS', default='roxford5k,rparis6k',
                        help='comma separated list of test datasets: ' + ' | '.join(
                            test_datasets_names) + ' (default: roxford5k,rparis6k)')

    args = parser.parse_args()

    for arg in vars(args):
        print(' {}\t  {}'.format(arg, getattr(args, arg)))

    print('\n ')

    return parser


def make_config(args):
    log_info("Loading configuration from %s", args.config)

    config = load_config(args.config, DEFAULT_CONFIGS["base"])

    log_info("\n%s", config_to_string(config))

    return config


def make_model(args, config):
    # parse params with default values

    body_config = config["body"]
    fpn_config = config["fpn"]
    ir_config = config["ir"]
    data_config = config["dataloader"]

    # get output dimensionality size
    net_modules = []

    # BN + activation
    norm_act_static, norm_act_dynamic = norm_act_from_config(body_config)

    # Create backbone
    log_debug("Creating backbone model %s", body_config.get("arch"))

    body_fn = models.__dict__[body_config.get("arch")]
    body_params = body_config.getstruct("body_params") if body_config.get("body_params") else {}
    body = body_fn(norm_act=norm_act_static, config=body_config, **body_params)

    net_modules.append("body")

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
            for mod_id in range(1, body_config.getint("num_frozen") + 1):
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
        fpn_outputs = fpn_config.getstruct("outputs")

        fpn = FPN([body_channels[inp] for inp in fpn_inputs],
                  fpn_config.getint("out_channels"),
                  fpn_config.getint("extra_scales"),
                  norm_act_static,
                  fpn_config.get("interpolation"))

        body = FPNBody(body, fpn, fpn_inputs)

        output_dim = fpn_config.getint("out_channels")
    else:
        output_dim = OUTPUT_DIM[body_config.get("arch")]

    # Create Image Retrieval
    ret_loss = ImageRetrievalLoss(name=ir_config.get("loss"),
                                  sigma=ir_config.getfloat("loss_margin"))

    ret_algo = ImageRetrievalAlgo(loss=ret_loss,
                                  min_level=ir_config.getint("fpn_min_level"),
                                  fpn_levels=ir_config.getint("fpn_levels")
                                  )

    ret_head = ImageRetrievalHead(pooling=ir_config.getstruct("pooling"),
                                  normal=ir_config.getstruct("normal"),
                                  dim=output_dim)
    # Data augmentation

    augment = RandomAugmentation(rgb_mean=data_config.getstruct("rgb_mean"),
                                 rgb_std=data_config.getstruct("rgb_std")
                                 )

    # Create a generic image retrieval network
    net = ImageRetrievalNet(body,
                            ret_algo,
                            ret_head,
                            augment=augment)

    return net, net_modules, output_dim


def test(args, config, model, rank=None, world_size=None, **varargs):
    log_debug('Evaluating network on test datasets...')

    # Eval mode
    model.eval()
    data_config = config["dataloader"]

    # Average score
    avg_score = 0.0

    # Evaluate on test datasets
    list_datasets = data_config.getstruct("test_datasets")

    if data_config.get("multi_scale"):
        scales = eval(data_config.get("multi_scale"))
    else:
        scales = [1]

    for dataset in list_datasets:

        start = time.time()

        log_debug('{%s}: Loading Dataset', dataset)

        # Prepare database
        db = ParisOxfordTestDataset(root_dir=path.join(args.data, 'test', dataset),
                                    name=dataset)

        batch_size = data_config.getint("test_batch_size")

        with torch.no_grad():
            """ Paris and Oxford are :
                    1 - resized to a ratio of desired max size, after bbx cropping 
                    2 - normalized after that
                    3 - not flipped and not scaled (!! important for evaluation)

            """
            # Prepare query loader
            log_debug('{%s}: Extracting descriptors for query images', dataset)

            query_tf = ISSTestTransform(shortest_size=data_config.getint("test_shortest_size"),
                                        longest_max_size=data_config.getint("test_longest_max_size"),
                                        random_scale=data_config.getstruct("random_scale"))

            query_data = ISSDataset(root_dir='',
                                    name="query",
                                    images=db['query_names'],
                                    bbx=db['query_bbx'],
                                    transform=query_tf)

            query_sampler = DistributedARBatchSampler(data_source=query_data,
                                                      batch_size=data_config.getint("test_batch_size"),
                                                      num_replicas=world_size,
                                                      rank=rank,
                                                      drop_last=True,
                                                      shuffle=False)

            query_dl = torch.utils.data.DataLoader(query_data,
                                                   batch_sampler=query_sampler,
                                                   collate_fn=iss_collate_fn,
                                                   pin_memory=True,
                                                   num_workers=data_config.getstruct("num_workers"),
                                                   shuffle=False)

            # Extract query vectors
            qvecs = torch.zeros(varargs["output_dim"], len(query_data)).cuda()

            for it, batch in tqdm(enumerate(query_dl), total=len(query_dl)):
                # Upload batch
                batch = {k: batch[k].cuda(device=varargs["device"], non_blocking=True) for k in INPUTS}

                _, pred = model(**batch, scales=scales, do_prediction=True, do_augmentaton=False)

                distributed.barrier()

                qvecs[:, it * batch_size: (it + 1) * batch_size] = pred["ret_pred"]

                del pred

            # Prepare negative database data loader
            log_debug('{%s}: Extracting descriptors for database images', dataset)

            database_tf = ISSTestTransform(shortest_size=data_config.getint("test_shortest_size"),
                                           longest_max_size=data_config.getint("test_longest_max_size"),
                                           random_scale=data_config.getstruct("random_scale"))

            database_data = ISSDataset(root_dir='',
                                       name="database",
                                       images=db['img_names'],
                                       transform=database_tf)

            database_sampler = DistributedARBatchSampler(data_source=database_data,
                                                         batch_size=data_config.getint("test_batch_size"),
                                                         num_replicas=world_size,
                                                         rank=rank,
                                                         drop_last=True,
                                                         shuffle=False)

            database_dl = torch.utils.data.DataLoader(database_data,
                                                      batch_sampler=database_sampler,
                                                      collate_fn=iss_collate_fn,
                                                      pin_memory=True,
                                                      num_workers=data_config.getstruct("num_workers"),
                                                      shuffle=False)

            # Extract negative pool vectors
            database_vecs = torch.zeros(varargs["output_dim"], len(database_data)).cuda()

            for it, batch in tqdm(enumerate(database_dl), total=len(database_dl)):
                # Upload batch

                batch = {k: batch[k].cuda(device=varargs["device"], non_blocking=True) for k in INPUTS}

                _, pred = model(**batch, scales=scales, do_prediction=True, do_augmentaton=False)

                distributed.barrier()

                database_vecs[:, it * batch_size: (it + 1) * batch_size] = pred["ret_pred"]

                del pred

        # Compute dot product scores and ranks on GPU
        # scores = torch.mm(database_vecs.t(), qvecs)
        # scores, scores_indices = torch.sort(-scores, dim=0, descending=False)

        # convert to numpy
        qvecs = qvecs.cpu().numpy()
        database_vecs = database_vecs.cpu().numpy()

        # search, rank, and print
        scores = np.dot(database_vecs.T, qvecs)
        ranks = np.argsort(-scores, axis=0)

        score = compute_map_and_print(dataset, ranks, db['gnd'], log_info)
        log_info('{%s}: Running time = %s', dataset, htime(time.time() - start))

        avg_score += 0.5 * score["mAP"]

    # As Evaluation metrics
    log_info('Average score = %s', avg_score)

    return avg_score


def main(args):
    global min_loss

    # Initialize multi-processing
    distributed.init_process_group(backend='nccl', init_method='env://')
    device_id, device = args.local_rank, torch.device(args.local_rank)
    rank, world_size = distributed.get_rank(), distributed.get_world_size()
    torch.cuda.set_device(device_id)

    # Load configuration
    config = make_config(args)

    # Initialize logging
    if rank == 0:
        logging.init(args.resume, "testing")
    else:
        summary = None

    # Load model
    model, _, output_dim = make_model(args, config)

    # Init GPU stuff
    torch.backends.cudnn.benchmark = config["general"].getboolean("cudnn_benchmark")
    model = model.cuda(device)

    # Resume / Pre_Train
    log_info("Loading snapshots from %s", args.resume)

    models = sorted(glob(args.resume + '/model*.pth.tar'), reverse=True)

    print(models)
    for model_i in models:

        snapshot = resume_from_snapshot(model, model_i, ["body", "ret_head"])

        epoch = snapshot["training_meta"]["epoch"]

        log_info("Evaluation epoch %d", epoch )

        test(args, config, model,
             rank=rank,
             world_size=world_size,
             output_dim=output_dim,
             device=device)

    log_info("Evaluation Done ..... ")


if __name__ == '__main__':
    parser = make_parser()

    main(parser.parse_args())
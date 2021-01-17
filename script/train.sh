export DB_ROOT=/media/torres/ssd_1tb/ImageRetrieval/data/datasets/

clear
python3 setup.py install

clear

N_GPUS=1
DATA_DIR='./data/'
EXPERIMENT='./experiments/'

Resume='./experiments/retrieval-SfM-120k_resnet50_triplet_m0.70_GeM_L2N_Adam_lr1.0e-06_wd5.0e-04_nnum2_bsize1_uevery10_imsize1024/model_last.pth.tar'


python3 -m torch.distributed.launch --nproc_per_node=$N_GPUS ./cirtorch/examples/train.py \
      --directory $EXPERIMENT \
      --data $DATA_DIR \
      --local_rank 0 \

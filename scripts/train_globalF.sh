export DB_ROOT=/media/torres/ssd_1tb/ImageRetrieval/data/datasets/

clear
python3 setup.py install

clear

N_GPUS=1
DATA_DIR='./data/'
EXPERIMENT='./experiments/'

Resume='./experiments/'
python3 -m torch.distributed.launch --nproc_per_node=$N_GPUS ./scripts/train_globalF.py \
      --directory $EXPERIMENT \
      --data $DATA_DIR \
      --local_rank 0 \

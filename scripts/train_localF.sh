

clear
python3 setup.py install

clear

N_GPUS=1
DATA_DIR='/media/torres/ssd_1tb/'
EXPERIMENT='./experiments/'

Resume='./experiments/'
python3 -m torch.distributed.launch --nproc_per_node=$N_GPUS ./scripts/train_localF.py \
      --directory $EXPERIMENT \
      --data $DATA_DIR \
      --local_rank 0 \
      #--eval
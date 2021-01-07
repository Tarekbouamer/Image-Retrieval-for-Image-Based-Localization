export DB_ROOT=/media/torres/ssd_1tb/ImageRetrieval/data/datasets/

python3 -m cirtorch.examples.test --gpu-id '0' \
    --network-path 'retrievalSfM120k-resnet101-gem' \
    --datasets 'paris6k' \
    --whitening 'retrieval-SfM-120k' \
    --multiscale '[1, 1/2**(1/2), 1/2]' \

# python3 -m cirtorch.examples.test --gpu-id '0' \
#     --network-path 'retrievalSfM120k-resnet101-gem' \
#     --datasets 'oxford5k,paris6k,roxford5k,rparis6k' \
#     --whitening 'retrieval-SfM-120k' \
#     --multiscale '[1, 1/2**(1/2), 1/2]' \

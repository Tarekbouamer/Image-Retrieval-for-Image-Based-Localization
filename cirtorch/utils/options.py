import torchvision.models as models

training_dataset_names = ['retrieval-SfM-120k']
test_datasets_names = ['oxford5k', 'paris6k', 'roxford5k', 'rparis6k']
test_whiten_names = ['retrieval-SfM-30k', 'retrieval-SfM-120k']


pool_names = ['mac', 'spoc', 'gem', 'gemmp']
loss_names = ['contrastive', 'triplet']
optimizer_names = ['sgd', 'adam']
interpolation_names = ['nearest', 'bilinear']

model_names = sorted(name for name in models.__dict__
                     if name.islower()
                     and not name.startswith("__")
                     and callable(models.__dict__[name])
                     )
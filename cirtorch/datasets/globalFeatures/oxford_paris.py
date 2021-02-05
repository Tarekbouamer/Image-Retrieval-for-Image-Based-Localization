import pickle
from os import path


TEST_DATASETS = ['oxford5k', 'paris6k', 'roxford5k', 'rparis6k']
_EXTENSIONS = ["*.jpg", "*.jpeg", "*.png"]


def ParisOxfordTestDataset(root_dir, name=None):

    if name not in TEST_DATASETS:
        raise ValueError('Unknown dataset: {}!'.format(name))

    # Loading imlist, qimlist, and gnd, in cfg as a dict
    pkl_path = path.join(root_dir,
                         'gnd_{}.pkl'.format(name))

    with open(pkl_path, 'rb') as f:
        db = pickle.load(f)

    db['pkl_path'] = pkl_path
    db['data_path'] = path.join(root_dir)
    db['images_path'] = path.join(db['data_path'], 'jpg')
    db['_ext'] = '.jpg'

    db['n_img'] = len(db['imlist'])
    db['n_query'] = len(db['qimlist'])

    db['img_names'] = [path.join(db['images_path'], item + db['_ext']) for item in db['imlist']]
    db['query_names'] = [path.join(db['images_path'], item + db['_ext']) for item in db['qimlist']]
    db['query_bbx'] = [db['gnd'][item]['bbx'] for item in range(db['n_query'])]

    db['dataset'] = name
    return db
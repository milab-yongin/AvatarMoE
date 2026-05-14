from .zjumocap import ZJUMoCapDataset
from .refined_zjumocap import RefinedZJUMoCapDataset
from .people_snapshot import PeopleSnapshotDataset

def load_dataset(cfg, split='train'):
    dataset_dict = {
        'zjumocap': ZJUMoCapDataset,
        'zjumocap_refine': RefinedZJUMoCapDataset,
        'people_snapshot': PeopleSnapshotDataset,
    }
    return dataset_dict[cfg.name](cfg, split)

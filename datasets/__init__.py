from torch.utils.data import Dataset

from datasets import sdf_dataset

# Add new datasets here
datasets = {
    "sdf_k_env_relative": sdf_dataset.RelativeSDFKEnvDataset,
}


def dataset_factory(dataloader_id, opt) -> Dataset:
    """
    Returns a dataset initialized with opt, if the dataloader/dataset_name is recognized and defined above.
    """
    if dataloader_id not in datasets.keys():
        raise ValueError("Dataset [%s] not recognized." % dataloader_id)
    return datasets.get(dataloader_id)(opt)

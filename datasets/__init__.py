import os

from torch.utils.data import Dataset

from datasets import sdf_dataset

# Add new datasets here
datasets = {
    "sdf_k_env_relative": sdf_dataset.RelativeSDFKEnvDataset,
}


def dataset_factory(dataset_name, opt) -> Dataset:
    """
    Returns a dataset initialized with opt, if the dataset_name is recognized and defined above.
    """
    if dataset_name not in datasets.keys():
        raise ValueError("Dataset [%s] not recognized." % dataset_name)
    return datasets.get(dataset_name)(opt)


def defensive_programming_opt_input_checks(opt):
    """
    Checks if the opt object has some necessary attributes for the dataset to work.
    :param opt: The opt object to check for
    """
    if not hasattr(opt, "dataroot"):
        raise ValueError("Dataroot not specified")
    if not os.path.exists(opt.data_dir):
        raise ValueError("Dataroot does not exist")

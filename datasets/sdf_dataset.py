import abc
import os

import numpy as np
from torch.utils.data import Dataset

from datasets.positional_encoding import point_encoder_fabric
from datasets.sdf_bacon_mesh import MeshSDF
from utils import defensive_programming_opt_input_checks_datasets


class ABSSDFDataset(Dataset, metaclass=abc.ABCMeta):
    def __init__(self, opt):
        defensive_programming_opt_input_checks_datasets(opt)
        self.opt = opt
        self.data_dir = os.path.join(opt.dataroot, opt.dataset_name)

    def __len__(self):
        """
        To define the length of one epoch, we define the dataset length as the number of meshes in the dataset times the
        number one mesh should be sampled per epoch.
        :return:
        """
        return self.size * self.opt.num_samples_per_mesh_per_epoch

    @abc.abstractmethod
    def __getitem__(self, idx):
        pass


class RelativeSDFKEnvDataset(ABSSDFDataset):
    """
    This dataset is for training a model to predict the SDF value of a point relative to its nearest neighbors.
    It is parameterized by
    - the number of nearest neighbors to consider (kdtree_num_samples)
    - the used positional encoding (positional_encoding) (currently only no_encode is supported)
    """

    def __init__(self, opt):
        super(RelativeSDFKEnvDataset, self).__init__(opt)
        self.files = os.listdir(self.data_dir)
        self.meshes = [
            MeshSDF(
                os.path.join(self.data_dir, path),
                num_samples=opt.num_samples_per_mesh_per_epoch,
                num_closest_points=opt.kdtree_num_samples,
            )
            for path in self.files
        ]
        self.positional_encoder = point_encoder_fabric(opt)
        self.size = len(self.files)
        print("Dataset loaded.")

    def __getitem__(self, idx):
        """
        Gets the next mesh and a sampled point from it to train on.
        As regression target, the SDF value of the point is returned by the same measure as in bacon.
        :param idx:
        :return: relative: relative coordinates of the nearest neighbors to the point, sdf: the SDF value of the point
        """
        mesh = self.meshes[idx % self.size]
        point, sdf, nn, _ = mesh.single_sample_plus()
        relative = nn - point
        return relative.flatten().astype("float32"), sdf.astype("float32")


class RelativeSDFKEnvConvoDataset(ABSSDFDataset):
    """
    This dataset is for training a model to predict the SDF value of a point relative to its nearest neighbors.
    It includes normals to build a 2d structure to run a convo on
    It is parameterized by
    - the number of nearest neighbors to consider (kdtree_num_samples)
    - the used positional encoding (positional_encoding) (currently only no_encode is supported)
    """

    def __init__(self, opt):
        super(RelativeSDFKEnvConvoDataset, self).__init__(opt)
        self.files = os.listdir(self.data_dir)
        self.meshes = [
            MeshSDF(
                os.path.join(self.data_dir, path),
                num_samples=opt.num_samples_per_mesh_per_epoch,
                num_closest_points=opt.kdtree_num_samples,
            )
            for path in self.files
        ]
        self.positional_encoder = point_encoder_fabric(opt)
        self.size = len(self.files)
        print("Dataset loaded.")

    def __getitem__(self, idx):
        """
        Gets the next mesh and a sampled point from it to train on.
        As regression target, the SDF value of the point is returned by the same measure as in bacon.
        :param idx:
        :return: relative: relative coordinates of the nearest neighbors to the point, sdf: the SDF value of the point
        """
        idx = idx % self.size
        mesh = self.meshes[idx]
        point, sdf, nn, normals = mesh.single_sample_plus()
        relative = nn - point
        sdf = sdf.astype("float32")
        concat = np.concatenate((relative, normals), axis=1).T.astype("float32")
        return concat, sdf

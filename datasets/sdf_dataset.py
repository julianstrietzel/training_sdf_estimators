import abc
import os

from torch.utils.data import Dataset

from datasets import defensive_programming_opt_input_checks
from datasets.positional_encoding import point_encoder_fabric
from datasets.sdf_bacon_mesh import MeshSDF


class ABSSDFDataset(Dataset, metaclass=abc.ABCMeta):
    def __init__(self, opt):
        defensive_programming_opt_input_checks(opt)
        self.opt = opt
        self.data_root = opt.dataroot

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
                num_closest_points=opt.kdtree_num_samples,
            )
            for path in self.files
        ]
        self.positional_encoder = point_encoder_fabric(opt)
        self.size = len(self.files)

    def __getitem__(self, idx):
        """
        Gets the next mesh and a sampled point from it to train on.
        As regression target, the SDF value of the point is returned by the same measure as in bacon.
        :param idx:
        :return: relative: relative coordinates of the nearest neighbors to the point, sdf: the SDF value of the point
        """
        idx = idx % self.size
        mesh = self.meshes[idx]
        point, sdf, nn = mesh.single_sample_plus_nearest_neighbors()
        # positional_encoded_point = self.positional_encoder.forward(
        #    torch.from_numpy(np.expand_dims(point, 0))
        # )[0, ..., np.newaxis]
        # if positional_encoded_point.shape != (3, 1):
        #    raise ValueError(
        #        "positional_encoded_point.shape != (3, 1)\nThis is for debugging at the first time only: Remove afterwards!"
        #    )

        # subtract point from nearest neighbors to get relative coordinates so that we do not need any point_encoding
        # nn is shape (num_closest_points, 3) and point is shape (3)
        relative = (
            nn - point
        )  # Is the following necessary? .reshape((1, 3)).repeat(nn.shape[0], axis=0)
        return relative, sdf

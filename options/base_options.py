import argparse
import os
from datetime import datetime

import torch


class BaseOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        self.initialized = False
        self.is_train = None

    def initialize(self):
        # data params
        self.parser.add_argument(
            "--dataroot",
            default="./data",
            required=False,
            help="path to data root should have subfolder for each dataset especially the one called in param dataset_name",
        )
        self.parser.add_argument(
            "--dataset_name",
            default="simple_shapes",
            help="name of the dataset to use being the folder in data_root that contains the dataset",
        )
        self.parser.add_argument(
            "--dataloader",
            default="sdf_k_env_relative",
            help="The kind of dataprovider for training",
        )

        # network params
        self.parser.add_argument(
            "--batch_size", type=int, default=512, help="input batch size"
        )
        self.parser.add_argument(
            "--model_name",
            type=str,
            default="simplest_regression_model",
            help="model to use",
        )

        # sizes of hidden layers in the network
        self.parser.add_argument(
            "--hlayer_sizes",
            type=list,
            default=[64, 128, 128, 128, 128, 64, 32, 16],
            help="output_size of each hidden layer",
        )

        # general params
        self.parser.add_argument(
            "--num_threads", default=3, type=int, help="# threads for loading data"
        )
        self.parser.add_argument(
            "--gpu_ids",
            type=str,
            default="0",
            help="gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU",
        )
        self.parser.add_argument(
            "--name",
            type=str,
            default="debug",
            help="name of the experiment. It decides where to store samples and models",
        )
        self.parser.add_argument(
            "--checkpoints_dir",
            type=str,
            default="./checkpoints",
            help="models are saved here",
        )
        self.parser.add_argument("--seed", type=int, help="if specified, uses seed")
        # sdf_regression arguments
        self.parser.add_argument(
            "--point_encode",
            type=str,
            default="no_encode",
            help="point encoding method from no_encode or positional_encoding_3d",
        )
        self.parser.add_argument(
            "--kdtree_num_samples",
            type=int,
            default=10,
            help="number of samples for kdtree closest point search for sdf approximation",
        )
        self.parser.add_argument(
            "--num_samples_per_mesh_per_epoch",
            type=int,
            default=2 ** 16,
            help="number of samples per mesh per epoch to be constant for comparability "
            + "but for introducing something that epochs can refer to",
        )
        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt, unknown = self.parser.parse_known_args()
        self.opt.is_train = self.is_train  # train or test

        str_ids = self.opt.gpu_ids.split(",")
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)
        # set gpu ids
        if len(self.opt.gpu_ids) > 0 and torch.cuda.is_available():
            torch.cuda.set_device(self.opt.gpu_ids[0])

        args = vars(self.opt)

        if self.opt.seed is not None:
            import numpy as np
            import random

            torch.manual_seed(self.opt.seed)
            np.random.seed(self.opt.seed)
            random.seed(self.opt.seed)

        self.opt.name = (
            datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            + "_"
            + self.opt.name
            + "_"
            + self.opt.model_name
            + "_"
            + self.opt.dataset_name
        )

        if self.is_train:
            print("------------ Options -------------")
            for k, v in sorted(args.items()):
                print("%s: %s" % (str(k), str(v)))
            print("-------------- End ----------------")

            # save to the disk
            expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
            self.opt.expr_dir = expr_dir
            if not os.path.exists(expr_dir):
                os.makedirs(expr_dir)
            file_name = os.path.join(expr_dir, "opt.txt")
            with open(file_name, "wt") as opt_file:
                opt_file.write(self.opt.name + "\n")
                opt_file.write("------------ Options -------------\n")
                for k, v in sorted(args.items()):
                    opt_file.write("%s: %s\n" % (str(k), str(v)))
                opt_file.write("-------------- End ----------------\n")
        return self.opt

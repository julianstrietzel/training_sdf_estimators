from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        #               self.parser.add_argument(
        #             "--print_freq",
        #             type=int,
        #             default=10,
        #             help="frequency of showing training results on console",
        #         )
        #         self.parser.add_argument(
        #             "--save_latest_freq",
        #             type=int,
        #             default=250,
        #             help="frequency of saving the latest results",
        #         )
        self.parser.add_argument(
            "--save_epoch_freq",
            type=int,
            default=1,
            help="frequency of saving checkpoints at the end of epochs",
        )
        #         self.parser.add_argument(
        #             "--run_test_freq",
        #             type=int,
        #             default=1,
        #             help="frequency of running test in training script",
        #         )
        #         self.parser.add_argument(
        #             "--continue_train",
        #             action="store_true",
        #             help="continue training: load the latest model",
        #         )
        #         self.parser.add_argument(
        #             "--epoch_count",
        #             type=int,
        #             default=1,
        #             help="the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...",
        #         )
        #         self.parser.add_argument(
        #             "--phase", type=str, default="train", help="train, val, test, etc"
        #         )
        #         self.parser.add_argument(
        #             "--which_epoch",
        #             type=str,
        #             default="latest",
        #             help="which epoch to load? set to latest to use latest cached model",
        #         )
        #         self.parser.add_argument(
        #             "--niter", type=int, default=100, help="# of iter at starting learning rate"
        #         )
        self.parser.add_argument(
            "--optimizer", type=str, default="adam", help="optimizer: adam | sgd"
        )

        self.parser.add_argument(
            "--lr", type=float, default=0.0002, help="initial learning rate for adam"
        )

        self.parser.add_argument(
            "--epochs", type=int, default=100, help="number of epochs to train for"
        )

        self.is_train = True

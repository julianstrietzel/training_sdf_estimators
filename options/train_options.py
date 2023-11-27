from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument(
            "--save_epoch_freq",
            type=int,
            default=25,
            help="frequency of saving checkpoints at the end of epochs",
        )
        self.parser.add_argument(
            "--init_type",
            type=str,
            default="normal",
            help="network initialization [normal|xavier|kaiming|orthogonal]",
        )
        self.parser.add_argument(
            "--init_gain",
            type=float,
            default=0.02,
            help="scaling factor for normal, xavier and orthogonal.",
        )
        # regression loss
        self.parser.add_argument(
            "--loss", type=str, default="mse", help="loss function type: mse | mae",
        )
        self.parser.add_argument(
            "--lr_scheduler",
            type=str,
            default="plateau",
            help="learning rate scheduler: plateau | exponential | step",
        )
        # gamma
        self.parser.add_argument(
            "--lr_decay_iters",
            type=int,
            default=20,
            help="multiply by a gamma every lr_decay_iters iterations",
        )
        self.parser.add_argument(
            "--lr_decay_gamma",
            type=float,
            default=0.5,
            help="gamma multiplier for lr decay",
        )
        # patience
        self.parser.add_argument(
            "--lr_decay_patience",
            type=int,
            default=5,
            help="patience for lr decay scheduler",
        )
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

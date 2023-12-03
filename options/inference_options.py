from .base_options import BaseOptions


class InferenceOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument(
            "--load_dir", type=str, default="./checkpoints/", help="saves results here."
        )
        self.parser.add_argument(
            "--model_id",
            type=str,
            default="2023-11-30_10-43-18_full_fully_connected_simplest_regression_model_simple_shapes",
            help="which model to load?",
        )
        self.parser.add_argument(
            "--which_epoch",
            type=str,
            default="final",
            help="which epoch to load? set to latest to use latest cached model",
        )
        self.is_train = False
        self.is_inference = True

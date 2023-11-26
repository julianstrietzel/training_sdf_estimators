from torch import nn

import abs_regression_model

models = {
    "simplest_regression_model": abs_regression_model.SimpleRegressionModel,
}


def model_factory(model_name, opt) -> nn.Module:
    """
    Returns a model class given a model name.
    """

    if model_name not in models.keys():
        raise ValueError("Model [%s] not recognized." % model_name)
    return models.get(model_name)(opt)

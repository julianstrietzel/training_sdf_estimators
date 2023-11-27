from models import regression_models

models = {
    "simplest_regression_model": regression_models.SimpleRegressionModel,
}


def model_factory(model_name, opt) -> regression_models.AbsRegressionModel:
    """
    Returns a model class given a model name.
    """

    if model_name not in models.keys():
        raise ValueError("Model [%s] not recognized." % model_name)
    return models.get(model_name)(opt)

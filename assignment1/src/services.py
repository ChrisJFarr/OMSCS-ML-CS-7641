import hydra
from src.datamodules import DataParent
from src.models import ModelParent


def test_data_loader(config):
    # Load a data source
    # Count examples over splits, report class balance on each split
    datamodule: DataParent = hydra.utils.instantiate(config.experiments.datamodule)
    for (x_train, y_train), (x_test, y_test) in datamodule.make_loader():
        train_target_avg = y_train.mean()
        test_target_avg = y_test.mean()
        print("avg train target: %.2f " % train_target_avg, "avg test target: %.2f" % test_target_avg)
    return


def test_training_loop(config):
    # TEMP, swap with evaluation
    from sklearn.metrics import accuracy_score
    # Create data loader
    datamodule: DataParent = hydra.utils.instantiate(config.experiments.datamodule)
    # Create model object
    model: ModelParent = hydra.utils.instantiate(config.experiments.model)
    # For each cv fold, train and test, report accuracy
    for (x_train, y_train), (x_test, y_test) in datamodule.make_loader():
        model.fit(x_train, y_train)
        train_prediction = model.predict(x_train)
        test_prediction = model.predict(x_test)
        print(
            "train accuracy: %.2f" % accuracy_score(y_train, train_prediction), 
            "test accuracy: %.2f" % accuracy_score(y_test, test_prediction)
            )
    return


def run(config):
    # Top level of running an experiment
    # Create data loader
    # Create evaluation
    # Run training loop for each CV fold and log evaluation results
    pass
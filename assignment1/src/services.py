import hydra
from sklearn.metrics import accuracy_score


from src.datamodules import DataParent
from src.models import ModelParent

# TODO Move tests to test folder/file
# https://github.com/facebookresearch/hydra/blob/main/examples/advanced/hydra_app_example/tests/test_example.py


def test_data_loader(config):
    # Load a data source
    # Count examples over splits, report class balance on each split
    datamodule: DataParent = hydra.utils.instantiate(config.experiments.datamodule)
    train_generator, test_data = datamodule.make_loader()
    for (x_train, y_train), (x_test, y_test) in train_generator:
        train_target_avg = y_train.mean()
        test_target_avg = y_test.mean()
        print("avg train target: %.2f " % train_target_avg, "avg test target: %.2f" % test_target_avg)
    return


def test_training_loop(config):
    # Create model object
    model: ModelParent = hydra.utils.instantiate(config.experiments.model)
    # Create data loader
    datamodule: DataParent = hydra.utils.instantiate(config.experiments.datamodule)
    # Split train/test
    train_generator, test_data = datamodule.make_loader()
    # For each cv fold, train and test, report accuracy
    for (x_train, y_train), (x_test, y_test) in train_generator:
        model.fit(x_train, y_train)
        train_prediction = model.predict(x_train)
        test_prediction = model.predict(x_test)
        print(
            "train accuracy: %.2f" % accuracy_score(y_train, train_prediction), 
            "test accuracy: %.2f" % accuracy_score(y_test, test_prediction)
            )
    return


def test_learning_curve(config):
    pass

def test_iterative_plot(config):
    pass

def test_validation_curve(config):
    pass

def test_train_test_performance(config):
    pass


def run(config):
    # Top level of running an experiment
    # Create data loader
    # Create evaluation
    # Run training loop for each CV fold and log evaluation results
    pass
import hydra
from sklearn.metrics import accuracy_score
from time import perf_counter


from src.datamodules import DataParent
from src.models import ModelParent
from src.evaluation import Assignment1Evaluation

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
    start_time = perf_counter()
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
    end_time = perf_counter()
    print("Total execution time: %.1f" % (end_time - start_time))
    return


def lc(config):  # shorthand: learning curve
    start_time = perf_counter()
    # Initialize model
    model: ModelParent = hydra.utils.instantiate(config.experiments.model)
    # Initialize data
    datamodule: DataParent = hydra.utils.instantiate(config.experiments.datamodule)
    # Initialize evaluation
    evaluation = Assignment1Evaluation(model=model, datamodule=datamodule, config=config)
    # Test
    evaluation.generate_learning_curve()
    end_time = perf_counter()
    print("Total execution time: %.1f" % (end_time - start_time))
    return

def ip(config):  # shorthand: iterative plot
    start_time = perf_counter()
    # Initialize model
    model: ModelParent = hydra.utils.instantiate(config.experiments.model)
    # Initialize data
    datamodule: DataParent = hydra.utils.instantiate(config.experiments.datamodule)
    # Initialize evaluation
    evaluation = Assignment1Evaluation(model=model, datamodule=datamodule, config=config)
    # Test
    evaluation.generate_iterative_plot()
    end_time = perf_counter()
    print("Total execution time: %.1f" % (end_time - start_time))
    pass

def vc(config):  # shorthand: validation curve
    start_time = perf_counter()
    # Initialize model
    model: ModelParent = hydra.utils.instantiate(config.experiments.model)
    # Initialize data
    datamodule: DataParent = hydra.utils.instantiate(config.experiments.datamodule)
    # Initialize evaluation
    evaluation = Assignment1Evaluation(model=model, datamodule=datamodule, config=config)
    # Test
    evaluation.generate_validation_curve()
    end_time = perf_counter()
    print("Total execution time: %.1f" % (end_time - start_time))
    return

def test_train_test_performance(config):
    start_time = perf_counter()
    end_time = perf_counter()
    print("Total execution time: %.1f" % (end_time - start_time))
    pass

def run(config):
    start_time = perf_counter()
    # Top level of running an experiment
    # Create data loader
    # Create evaluation
    # Run training loop for each CV fold and log evaluation results
    end_time = perf_counter()
    print("Total execution time: %.1f" % (end_time - start_time))
    pass
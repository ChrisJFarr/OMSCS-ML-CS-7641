import hydra
from sklearn.metrics import accuracy_score
from time import perf_counter


from src.datamodules import DataParent
from src.models import ModelParent
from src.evaluation import Assignment3Evaluation

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

def get_evaluation(config):
    # Initialize model
    model: ModelParent = hydra.utils.instantiate(config.experiments.model)
    # Initialize data
    datamodule: DataParent = hydra.utils.instantiate(config.experiments.datamodule)
    # Initialize evaluation
    evaluation = Assignment3Evaluation(model=model, datamodule=datamodule, config=config)
    return evaluation

def lc(config):  # shorthand: learning curve
    print("Generating learning curve...")
    start_time = perf_counter()
    evaluation = get_evaluation(config)
    evaluation.generate_learning_curve()
    end_time = perf_counter()
    print("Learning curve completed in %.1f seconds" % (end_time - start_time))
    return

def ip(config):  # shorthand: iterative plot
    print("Generating iterative plot...")
    start_time = perf_counter()
    evaluation = get_evaluation(config)
    evaluation.generate_iterative_plot()
    end_time = perf_counter()
    print("Iterative plot completed in %.1f seconds" % (end_time - start_time))
    pass

def vc(config):  # shorthand: validation curve
    print("Generating validation curve...")
    start_time = perf_counter()
    evaluation = get_evaluation(config)
    evaluation.generate_validation_curve_data()
    end_time = perf_counter()
    print("Validation curve completed in %.1f seconds" % (end_time - start_time))
    return

def cv(config): # shorthand: cross-validation
    print("Performing cross-validation...")
    start_time = perf_counter()
    evaluation = get_evaluation(config)
    evaluation.cv_score()
    end_time = perf_counter()
    print("Cross-validation completed in %.1f seconds" % (end_time - start_time))
    return

def full(config): # shorthand: full train test evaluation
    evaluation = get_evaluation(config)
    # Run test
    evaluation.evaluate_train_test_performance()
    return

def sp(config):  # shorthand: silhouette-plot
    print("Generating silhouette plot...")
    start_time = perf_counter()
    evaluation = get_evaluation(config)
    evaluation.silhouette_plot()
    end_time = perf_counter()
    print("Silhouette plot completed in %.1f seconds" % (end_time - start_time))

def sp_auc(config): # shorthand: silhouette-plot+auc
    print("Generating silhouette+auc plot...")
    start_time = perf_counter()
    evaluation = get_evaluation(config)
    evaluation.silhouette_plot_with_auc()
    end_time = perf_counter()
    print("silhouette+auc plot completed in %.1f seconds" % (end_time - start_time))

def tsne(config): # shorthand: tsne plot
    print("Generating tsne plot...")
    start_time = perf_counter()
    evaluation = get_evaluation(config)
    evaluation.tsne_plot()
    end_time = perf_counter()
    print("Tsne plot completed in %.1f seconds" % (end_time - start_time))

def feat_sel(config):  # Short for: feature selection plot
    print("Generating feature-selection plot...")
    start_time = perf_counter()
    evaluation = get_evaluation(config)
    evaluation.feature_selection()
    end_time = perf_counter()
    print("Feature-selection plot completed in %.1f seconds" % (end_time - start_time))

def feat_sel_auc(config):
    print("Generating feature-selection+auc plot...")
    start_time = perf_counter()
    evaluation = get_evaluation(config)
    evaluation.feature_selection_with_auc()
    end_time = perf_counter()
    print("Feature-selection+auc plot completed in %.1f seconds" % (end_time - start_time))

def feat_imp(config):  # Short for: random-cluster-select-feature-importance
    print("Generating feature-importance plot...")
    start_time = perf_counter()
    evaluation = get_evaluation(config)
    evaluation.random_repeated_cluster_feature_importance()
    end_time = perf_counter()
    print("Feature-importance plot completed in %.1f seconds" % (end_time - start_time))

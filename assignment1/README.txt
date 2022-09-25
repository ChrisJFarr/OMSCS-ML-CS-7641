# Assignment 1
**GID**: cfarr31

##################
## Instructions ##
##################

CLI through hydra. 

Example usage: (see details below)
    * Evaluate decision tree classifier on dataset 1 and compute validation curve plots
        Run `python run.py +func=vc experiments=tree-1`
    * Evaluate neural-network classifier on dataset 2 and compute full test performance with timers
        Run `python run.py +func=full experiments=nn-2`

Step 1: Install assignment1/requirements.txt into python version 3.8+
    Run `pip install -r requirements.txt`
Step 2: Run experiment as above using the below options while in the `assignment1` folder
    * `experiments` options: "[model]-[dataset]"
        * tree-1
        * tree-2
        * knn-1
        * knn-2
        * boost-1
        * boost-2
        * nn-1
        * nn-2
        * svm-1
        * svm-2
    * `+func` options:
        * vc: validation curve
        * lc: learning curve
        * gc: grid search (not compatible with nn-1 or nn-2)
        * ip: iterative plot
        * cv: cross validation, return cv test and train
        * full: full train and test evaluation with timers
        optional tests
        * test_data_loader
        * test_training_loop


Data Downloads

1. https://www.kaggle.com/datasets/mathchi/diabetes-data-set
    * store in path: data/diabetes.csv
2. https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset]
    * store in path: data/diabetes_012_health_indicators_BRFSS2015.csv

Implementation Notes
* Code is primarily stored in the src and eda folders while experiment
    parameters are found in the configs folder.
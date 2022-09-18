# Assignment 1
**GID**: cfarr31


# Writeup TODO's

1. README.txt containing instructions for running your code

2. Analysis pdf titled cfarr31-analysis.pdf
* You'll have to explain why they (data) are interesting, use them in later assignments, and come to really care about them.
* we need to be able to get to your code and your data, arrange for an URL of some sort (I should provide my github link to code in README.txt)
    * https://github.com/ChrisJFarr/OMSCS-ML-CS-7641/tree/main/assignment1
    * 1: https://www.kaggle.com/datasets/mathchi/diabetes-data-set
    * 2: https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset
a description of your classification problems, and why you feel that they are interesting. 
    * Think hard about this. 
    * To be at all interesting the problems should be non-trivial on the one hand, but capable of admitting comparisons 
        and analysis of the various algorithms on the other. 
    * Avoid the mistake of working on the largest most complicated and messy dataset you can find. 
    * The key is to be interesting and clear, no points for hairy and complex.
the training and testing error rates you obtained running the various learning algorithms on your problems. 
    * At the very least you should include graphs that show performance on both training and test data as a 
        function of training size (note that this implies that you need to design a classification problem 
        that has more than a trivial amount of data) and--for the algorithms that are iterative--training times/iterations. 
    * Both of these kinds of graphs are referred to as learning curves, BTW.
analyses of your results. 
    * Why did you get the results you did? 
    * Compare and contrast the different algorithms. 
    * What sort of changes might you make to each of those algorithms to improve performance? 
    * How fast were they in terms of wall clock time? Iterations? 
    * Would cross validation help (and if it would, why didn't you implement it?)? 
    * How much performance was due to the problems you chose? 
    * How about the values you choose for learning rates, stopping criteria, pruning methods, and 
        so forth (and why doesn't your analysis show results for the different values you chose? 
    * Please do look at more than one. 
    * And please make sure you understand it, it only counts if the results are meaningful)? 
    * Which algorithm performed best? 
    * How do you define best? 
    * Be creative and think of as many questions you can, and as many answers as you can but a lot of the questions boil down to: why... WHY WHY WHY?
For the sanity of your graders, please keep your analysis as short as possible while still covering the requirements of the assignment: 
    * to facilitate this sanity, analysis writeup is limited to 12 pages.


Office Hours
Required components
1. Generate results using experiments
    * learning curve plots (some notion of a learning curve)
        * y axis: performance metric
        * x axis: percentage of training data
    * iterative plot, plot loss curve
        * nn: plot loss for each iteration:
            * y axis loss
            * x axis iteration/epoch
    * validation curve
        * y-axis: performance metric
        * x-axis: varying hyperparameter values
        * plot performance across hyper parameter tuning
    * wall-clock time
        * fit times during training
2. structure analysis around these results
    * choose any evaluation metric but introduce early and justify
        * talk about class balance
        * only plot what i analyze
    * why is your dataset interesting from a ML perspective
        * for mine: differences in the features, types, size, and similarity in problem
    * compare and contrast various algorithms
    * compare and contrast across datasets
    * why did something do well or poorly
    * for learning curve
        * bias/variance tradeoff AKA overfit-underfit AKA how well are you generalizing?
    * iterative plot
        * useful for debugging issues during training
        * data anomalies, weights that are zeroing out
        * evaluating learning rate
        * stability during training
    * validation curve
        * isolate tuning to a specific hyperparameter
        * 2 per algorith per dataset
        * dt: pruning
        * nn: hidden layers
        * ensemble: n weak learners
        * SVM: kernel types
        * KNN: varying K


# Writeup: to be transferred to pdf

Describe datasets:
* ds1
    * target: predicting if diabetes is present
    * moderately imbalanced target
    * has 9 features
    * has 768 examples
    * all patients are female, >21 yrs old, and of the same heritage
    * includes only medical biomarkers
    * note(it is unclear how prediabetes is handled)

* ds2
    * target: predicting if diabetes is present
    * highly imbalanced target
    * has 21 features
    * has 253,680 examples
    * includes health history, demographics, lifestyle, and self-reported vital information
    * No missing information
    * Binary, ordinal, and continuous variables
    * note(prediabetes is grouped with diabetes in the predicted target)

Note: The features in ds2 are more numerous as well as more broad, however many also have lower specificity than ds1.

Describe validation approach:
* ds1
    * Smaller so I needed to stretch further with random repeats
    * With 768 examples, I performed k=5 fold cross-validation with 100 random repeats
        * With no repeats the model performance is more optimistic.
        * After around 100 repeats, increasing the repeats results in no noticeable performance or behavior changes
* ds2
    * With more examples, k=5 fold validation is time-consuming especially for the neural network. In order to simplify
        experiments and maintain maneable runtime I chose not to repeat.
    * k=5 cross-validation without repeat

Analysis

Why its an interesting problem?
* Learn about the impact of self-reported vs clinical data sources. (experiment 3)
    * risks: 
        * the number of examples are vastly different between datasets, more examples make it easier to learn general rules / features
        * the homogenity of the clinical dataset could also assist with less data, solution: filter ds2 down to narrow demographics 
        * the differences in class balance could make it harder for the imbalanced dataset
    * step 1: align datasets by size, class balance and example homogenity
        * sample ds2 to match ds1 size (repeat resizing sampling multiple times and average experiment metrics)
    * step 2: run experiments and produce visuals
    * Analysis
        * compare performance between datasets
        * compare regularization requirements between datasets
        * discuss the differences and point to possible data characteristics
            * My hypothesis: clinical-data sources require less data to be accurate and requires less regularization
            
            * Why might this be true?
                * Less noise in the data when collected by clinical sources vs self-reporting
            * Why might this not be true?
                * There are many unknowns about the patient data. A possible driver of noise could be when the target is considered true relative to 
                    when the patient state was captured. Something else could be behind the noise.

Misc questions: (these are questions I should be able to answer after running the 4 experiments on all of the models)
    Provide a hypothesis for each question 
* when matching dataset sizes, which dataset can obtain higher performance without feature engineering?
* which dataset generalizes to its test set the best? Is this the same for each model?
    * hypothesis: The dataset with more examples I believe is more likely to generalize well to it's test set for all models.
* does the class imbalance hurt the ability to fit ds2 or does the larger number of samples make up for it?
    * hypothesis: 
* does one dataset require more regularization (in general lower complexity) than the other on a consistent basis?
    * which needs more regularization? Could this correlate with the quality of signal? I would expect lower quality to need more and be lower performing
* what can we learn about a potentially larger ds1 size and how it might perform based on experimenting with various ds2 sizes?
* how might the population homogenity of ds1 give it an advantage? could this be exploited in ds2?

Unbaked thoughts
* Which dataset requires more complexity? (opposite of regularization)
* Talk about bias variance tradeoff
* Which dataset requires more bias
* Which dataset requires more variance

Analysis topics:
* bias variance tradeoff
* model complexity
* dataset size


Run 3 experiments. 
{model-name}-1: Optimize performance for ds1 using grid-search, produce plots
{model-name}-2: Shrink ds2 through repeated random selection, optimize using grid-search, produce plots
    * produce ds3 from ds2:
        * balance the classes, so randomly sample for each class and match ds1 balance exactly
        * filter to a narrower demographic to match ds1 (all female)
    * (use n seeds for n repeats and % of data parameter)
{model-name}-3: Optimize performance for ds2 using grid-search, produce plots


Experiment 1 Steps (do this for each model/dataset combination)
1. Starting from default parameters, perform validation curve analysis on the selected parameters
    a. report top test performance and associated train performance
    b. report train and test performance for each metric
    c. call anything interesting out
2. Produce grid-search parameter space to cover from defaults to beyond optimal values discovered in step 1
    a. repeat experiment to increase granularity when narrowing in on optimal parameter values
    a. report parameters used for final search space
    b. report best performance and analyze if any improvement was discovered
3. Using the best parameters from the grid-search build the iterative plot
    a. does it appear that more data is helpful? Estimate how much more data would be useful
4. Once all experiments are complete, return and run the training and test the full dataset with the held-out test set
    a. how does the performance compare to what was estimated?
    b. why might it be different and could that risk maybe have been mitigated using knowledge only from the train set?
        Or what else would have been needed to make a better estimate?



### SVM Analysis

validation curve parameters for experiments 1, 2 and 3
evaluation:
  validation_curve:
    # run for defaults
    kernel:
      start: 0
      stop: 4
      step: 1
    # run for poly kernel
    C:
      start: 0.001
      stop: 1.01
      step: .01
    coef0:
      start: 0.0
      stop: 1.01
      step: .05
    degree:
      start: 1
      stop: 7
      step: 1

# Data preprocessing
* I tested min-max scaling and l2 normalizing as preprocessing functions for
    the input data and min-max scaling performed significantly better. I will use
    scaling for the remaining experiments.


############
Experiment 1
#############

## Perform validation curve analysis

First step is to select which kernels might perform best by using only default parameters.
Top performers were:
    * rbf with test AUC .849 (train .887)
    * poly with an AUC less than rbf and greater than linear, also >.8, and train AUC > rbf. This tells me that it is fitting the data really well
        and there might be potential for higher performance than rbf when tuned.
    * then linear with and AUC less than poly but also > .8. This kernel also had the least amount of overfitting with the train score very close
        to the test score visually.
Due to the close to optimal test performance of poly as well as the optimal train performance, I decided to explore this kernel further.

set kernel to poly for remaining experiments

* C:
    * best test .849, train .882
    * param=0.141
* coef0:
    * best test .844, train .898
    * param=0.05
* degree:
    * best test .851, train .879
    * param=2

## Perform grid search

Round 1
Best parameters
{'model__C': 0.9510000000000001,
 'model__coef0': 0.15000000000000002,
 'model__degree': 2}
'Best performance: 0.849'
Grid search completed in 175.6 seconds
Round 2
Best parameters
{'model__C': 0.9700000000000002,
 'model__coef0': 0.17500000000000002,
 'model__degree': 2}
'Best performance: 0.849'
Grid search completed in 162.5 seconds
Round 3
Best parameters
{'model__C': 0.9, 'model__coef0': 0.19999999999999996, 'model__degree': 2}
'Best performance: 0.849'
Grid search completed in 117.2 seconds

## Compute learning curve
Generating learning curve...
Learning curve completed in 157.2 seconds

A blip at 17% reached peak performance of 86.9 test AUC. However, the general trend of using more data
is an increase in test score and a decrease in overfitting.

#############
Experiment 2
############


## Generate validation curves

First analyzing impacts of different kernels on the dataset
* poly
    * highest performing
    * .795 test AUC, .814 train AUC
* linear, rbf, and sigmoid performed similarly
* sigmoid overfit the least

poly performed the best and is the same kernel that was used in experiment 1
so I will use poly for the remaining experiments



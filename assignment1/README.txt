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



Decision tree

validation curve parameters for experiments 1, 2 and 3
max_depth:
    start: 2
    stop: 10
    step: 1
ccp_alpha:
    start: 0.0
    stop: 0.051
    step: 0.001
min_samples_split:
    start: 2
    stop: 500
    step: 25
min_samples_leaf:
    start: 1
    stop: 500
    step: 25

#################
Tree Experiment 1:
#################

## Validation Curve
* Starting from default parameters, performed validation curve analysis on the following parameters:
    * Top performance is test AUC=.81
        * achieved with all defaults and setting min_samples_leaf to 26
            * train AUC .879 test AUC=.812
        * achieved with all defaults and setting min_samples_split to 102
            * train AUC .862 test AUC=.810 (slightly less overfitting) 
    * max-depth
        * train .854 and test .793 with max-depth=3
        * this reduced overfitting at cost of some performance compared to top metrics
    * ccp_alpha
        * train .848 and test .794 with ccp_alpha=0.01
        * similar effect of max-depth on test performance (when viewing the curve) but inverse
            impact on the training
Validation curve completed in 34.3 seconds

## Grid-Search

grid-search parameters
* Performed grid-search cv from defaults and through max-performing values discovered above


Best parameters
Round 1 of grid-search
{'ccp_alpha': 0.0,
 'max_depth': 5,
 'min_samples_leaf': 16,
 'min_samples_split': 92}
'Best performance: 0.814'
Grid search completed in 284.4 seconds
Round 2 of grid-search
{'ccp_alpha': 0.0,
 'max_depth': 6,
 'min_samples_leaf': 20,
 'min_samples_split': 90}
'Best performance: 0.814'
Grid search completed in 120.1 seconds
Round 3 of grid-search
Best parameters
{'ccp_alpha': 0.0004,
 'max_depth': 6,
 'min_samples_leaf': 17,
 'min_samples_split': 91}
'Best performance: 0.815'
Grid search completed in 529.3 seconds

## Learning Curve

With the best model parameters, compute the learning curve.
Learning curve completed in 7.1 seconds
* Analysis: Clearly there is a trend which shows in general more data leads to higher performance. A dataset
    several times the one used would likely produce better results. Performance further improves when using 94%
    of the data, likely indicating there are a few harder examples that happen to get removed until more  than 94%
    of the data is used at the random seed used.

#############
Tree Experiment 2:
#############


With a larger dataset I will have to reduce the parameter search space.
Could any insights from experiment 1 help?

## Validation Curve

* Starting from default parameters, performed validation curve analysis on the following parameters:
    * ccp_alpha
        * Best test auc: .80 train auc .858
        * best param value 0.006
    * max-depth
        * best test auc: .792 train auc: .875
        * best param value 4
    * min_samples_leaf
        * best test auc: .806 train auc: .864
        * best param value 26
    * min_samples_split
        * best test auc: .789 train auc: .829
        * best param value 127
Notes: the shapes of the validation curves are very similar to ds1 and the optimal values are similar or the same for each parameter.
    * noteable difference is max-depth is higher for the dataset that contains more and less-specific features

Validation curve completed in 38.9 seconds

## Grid Search

Round 1
{'ccp_alpha': 0.0,
 'max_depth': 8,
 'min_samples_leaf': 36,
 'min_samples_split': 177}
'Best performance: 0.813'
Grid search completed in 837.1 seconds
(gatech) chrisfarr@Chris-Farr-MacBook-Pro assignment1 % 
Round 2
{'ccp_alpha': 0.0,
 'max_depth': 8,
 'min_samples_leaf': 44,
 'min_samples_split': 178}
'Best performance: 0.813'
Grid search completed in 535.6 seconds

* after multiple experiments found no improvements to the 3rd digit
* seeing very similar top performance in the small-ds2 compared to ds1
* The additional features in exchange for lower granularity gives no improvement
* When considering the small ds2 is self-reported vs clinically captured it performs
    very similarly. However this may be because the increased number of features helps, the
    evidence of this is the larger max-depth required for the small ds2 indicating more features
    are leveraged

## Learning Curve

Learning curve completed in 39.0 seconds
* Again the optimal performance is reached prior to 100% of data, but the trend
    that more data is better is clear.

###############
Experiment 3
#############

# Version idea 1: Use the optimal parameters on the small ds2 and perform validation curve
 and learning rate analysis directly (no grid-searching the large dataset)

# Version idea 2: Repeat full experiment on large dataset but with reduced number of parameters
    searched

# Hybrid idea: Do version 1, then pick 2 features to search based on results






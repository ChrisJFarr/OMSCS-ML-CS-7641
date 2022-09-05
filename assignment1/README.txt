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

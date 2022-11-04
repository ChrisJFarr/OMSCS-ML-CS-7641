You are to implement (or find the code for) six algorithms. 
The first two are clustering algorithms:

k-means clustering
Expectation Maximization (sklearn.mixture.GaussianMixture)

You can choose your own measures of distance/similarity. Naturally, 
you'll have to justify your choices, but you're practiced at that sort of thing by now.

The last four algorithms are dimensionality reduction algorithms:

PCA
ICA (https://scikit-learn.org/stable/modules/decomposition.html#ica)
Randomized Projections (https://scikit-learn.org/stable/modules/classes.html#module-sklearn.random_projection)
Any other feature selection algorithm you desire

You are to run a number of experiments. Come up with at least two datasets. If 
you'd like (and it makes a lot of sense in this case) you can use the ones you 
used in the first assignment.

1. Run the clustering algorithms on the datasets and describe what you see.
    * intertia and elbow-plot
    * without looking at labels decide on the number of clusters
    * analyze how to decide how many clusters to use
    * discuss the number of features and compare datasets and the optimal clusters


2. Apply the dimensionality reduction algorithms to the two datasets and describe what you see.
    * For PCA, ICA, and random projections: try to use tsne plot
    * tsne
        * https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
        * https://scikit-learn.org/stable/modules/manifold.html#t-sne
        * color by cluster, and shape by target + and - ?
    * random projections
        * https://scikit-learn.org/stable/modules/random_projection.html#random-projection
    * what to do for feature selection?
        * interesting how many features are selected based
        * plot: y-axis is count of features over a range for a parameter
        * use tree-based. since also analyzing entropy in step 1


3. Reproduce your clustering experiments, but on the data after you've run dimensionality 
reduction on it. Yes, that’s 16 combinations of datasets, dimensionality reduction, 
and clustering method. You should look at all of them, but focus on the more interesting 
findings in your report.
    * this produces 16 plots (times how many plots per experiment)
        * one: how do you determine clusters
            * i can just use inertia
            * choose in an unsupervised manor
            * 
        * two how you perform evaluation somehow (just look into skearn for any way to evaluate)
            * can I look at entropy? entropy plot  
                * compute entropy at each k-step and plot on separate axis
                * (my hypothesis is that it will continue to improve entropy)
                * Does entropy improve and if so, does it steadily improve or jump around?
                * ultimately want to know which method is better for choosing k
                * computing entropy: compute positive class average and join as a prediction prob
    * you don't have to analyze every combination

4. Apply the dimensionality reduction algorithms to one of your datasets from assignment #1 
(if you've reused the datasets from assignment #1 to do experiments 1-3 above then you've 
already done this) and rerun your neural network learner on the newly projected data.
    * can use any kind of reduction for features
    * I should probably try to maximize model performance
    * experiments: pca, ica, rand-proj, feature-select with ds1
    * Use plots from assignment 1:
        * validation curve (how do parameter changes impact performance)
        * learning plot (how does the projection help or hurt overfitting?)
    

5. Apply the clustering algorithms to the same dataset to which you just applied the 
dimensionality reduction algorithms (you've probably already done this), treating 
the clusters as if they were new features. In other words, treat the clustering 
algorithms as if they were dimensionality reduction algorithms. Again, rerun your 
neural network learner on the newly projected data.
    * also going to run feature selection, so... are these selected? If not, what happens when 
        loosen restrictions do they ever improve performance?
    * if time, also wanted to try smaller groups of features and fitting multiple k-means
    * add random pairwise clusters, feature select small portion of them to reduce dim
    * run dim reduction + produce random pairwise clusters
    * decide on global k using elbow plot below
    * plot: expand on elbow-plot with multi-cluster plotting (black line for inertia for each)
        

Experiments

1. clustering
    * k-means ds1 & ds2
    * gmm ds1 & ds2
2. dimensionality reduction
    * 


Question:
Is the elbow method the best for SL?
    Is there an increase in performance around the elbow?
    Or is the best performance away from the elbow?

Plot: inertia? https://stackoverflow.com/questions/41540751/sklearn-kmeans-equivalent-of-elbow-method


Clustering: Try doing silhouette_score and elbow method.

Consider for clustering Pair plot? https://seaborn.pydata.org/generated/seaborn.pairplot.html
    Could just pick a few top featueres

TSNE for dim reduction analysis

Include some sort of artifact that you'd include in assignment 1 (this is done)
Plot that shows impact of dropping a feature on a metric

Mention this: No free lunch theorem: you have to try different algorithms, no single model works everywhere.




Experiment 5

Using PCA -> random-column-clustering -> lasso-feature-selection
    * I would expect the earlier principal components to be selected more often
    * summarize how often each component is selected-by-lasso when in random-column-clustering
    * preliminary results show that the middle components are favored by the model-selected-random-clustering
Without PCA, random-column-cluster feature importance is not as close to uniform as PCA, instead
the distribution looks more normal with some features with far less representation.
Compare selection distribution between non-pca features and pca features where n-components=n-features
Then compare the selection distribution between pca with fewer components
Analyze how the feature selection is robust to pca projections

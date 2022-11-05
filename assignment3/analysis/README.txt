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


Config Checklist

General Analysis
* Having a smaller similar dataset was
    very helpful in streamlining this analysis. I was able to test and understand
    each algorithm quickly on the smaller dataset before making a run on the 
    large dataset, which sometimes took almost an hour for the experiment to process
    saving me time by reducing the number of iterations needed for the larger.

Experiment 1

kmeans: ds1; python run.py +func=sp experiments=km-1
Analysis
The best silhouette_score is for 2 clusters. Does this mean
that ds1 doesn't have good naturally defined clusters? A case
could be made for roughly 4-5 clusters as the silhouette_score
doesn't drop off flat until it goes from 7 to about 20, then
additional clusters don't hurt or help the silhouette_score.

em: ds1; python run.py +func=sp experiments=em-1
Analysis
Compared to KM, the silhouette_score for 2 clusters is worse 
when using EM. Otherwise the behavior as cluster count increases
is very similar. KM appears to be better at creating more
defined clusters than EM for ds1.

kmeans: ds2; nohup python run.py +func=sp experiments=km-2 &
Analysis
Opposite of the ds1 results, after 20 clusters the silhouette_score reaches
a new high and this continues till at least 32 clusters. It seems that
with more clusters in ds2 it leads to better defined clusters while
for ds1 fewer were better defined. Why? Maybe the types of features
are contributing to this. ds1 has more continuous features and ds2 has
more binary features. Perhaps it is easier to cluster the latter?
Silhouette plot completed in 3537.1 seconds

em: ds2; nohup python run.py +func=sp experiments=em-2 &
Analysis
Similar line shape between KM and EM for ds2, although due
to runtime I wasn't able to test as many clusters and in this
case the best silhouette score found was for only 2 clusters. Perhaps
similar to KM if more clusters were tested after >25 clusters the
silhouette score would reach a new high. Based on this visual though
it also seems there are more diminishing returns when given more 
clusters than witnessed in KM.


Silhouette plot completed in 3186.4 seconds

Experiment 2

pca: ds1; python run.py +func=tsne experiments=pca-1
Analysis
TSNE plot allows a max of 3 components.
Using tsne to visualize on 2 components some clear patterns emerge which
seem to generally separate negative and positive classes. interesting
that this is able to occur without any knowledge of the targets. Consdiering
that the inputs here are clinical and related to diabetes it is not surprising
that tsne can loosley separate the classes. There are also many instances of
a positive example placed near groups of negative. Similar to how model performance
can't be perfect, here, there are also challenges with separating examples.

ica: ds1; python run.py +func=tsne experiments=ica-1
Analysis
The outputs of ica for ds1 are almost a 180 degree rotation of pca, indicating somewhat
similar results, which is interesting. I applied a 180 degree rotation to the tsne plot for
ICA to illustrate this property. When comparing the rotated ica to the pca for ds1
nearly all of the structure looks similar except for an additional tail reconnecting 
the lower left area to the lower middle of the structure. This tail in PCA
connects to the other side but is still visibly present. From this I would conclude
that only a handful of examples are treated differently and even for those there is 
some consistency in how they are mapped to the projected space between PCA and ICA.
If I were to use a projection in an SL model, based on this analysis it would be worth
trying both methods because there may be an important exception to how some examples are
projected.

proj: ds1; python run.py +func=tsne experiments=proj-1
Analysis
Testing both 2 and 3 components for the tsne plot, 2 appears to provide
more structure as well as more separation of the positive and negative classes.
Of course, to avoid making decisions based on the label I would choose 2 because it
provides more structure, although I think I biased myself by looking at the class
separation first now no matter what I decide it will be in an effort to unbias myself
which is impossible now.

pca: ds2; python run.py +func=tsne experiments=pca-2
Analysis
After the first iteration its hard to see the label colors because of the
density of the examples. But it is also interesting that so much is filled in
looking like a blob. Could it be that the diversity of the ds2 results in a 
more filled out structure where the homogenity of ds1 results in a more sparse
structure?


ica: ds2; python run.py +func=tsne experiments=ica-2
Analysis

proj: ds2; python run.py +func=tsne experiments=proj-2
Analysis

feat: ds1; python run.py +func=feat1 experiments=feat-1
Analysis
My goal here is to understand the behavior of the feature selection
algorithm as I change the Lasso Alpha parameter, producing a validation
curve in a way, but instead of analyzing model performance I am just
looking at how many features are selected.
There is a steep drop from 0.0 to 0.02 in which it goes from all 8 features down
to only a single feature. Perhaps looking for longer plateaus is a natural
and semi-supervised approach to selecting how many features to use. A plateau
means there is a gap between the decision to drop the next feature. Similar in
concept to the elbow method, but instead of diminishing returns for inertia it would
be diminishing returns for increased regularization on the number of features selected.
If I were to use that approach here I would probably stop just before Alpha=0.01 and 
use the 5 features selected at that point.


feat: ds2; python run.py +func=feat2 experiments=feat-2
Analysis


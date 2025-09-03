# BreastCancer_BiomarkerDiscovery

Wanted to see how far you can get with a small set of quantitative features and a binary outcome — basically to practice ML, data exploration, visualization, and PCA.

*What I did:*

**1 - Data Loading and Exploration**

The #data_loading_exploration.py is all about getting familiar with the dataset. First, I load the Breast Cancer dataset and inspect it. Basic stats, and checking for missing values.
Then, pick our features and targets depending on whether we’re doing classification or regression. For regression, we focus on predicting HOMA using insulin, leptin, resistin, and MCP-1.
Features are standardized so everything is on the same scale, which makes models like Ridge or ANN training way smoother. I also did a little correlation check to see which features are highly correlated with the target and which are redundant. Basically, this script makes sure we know the data before doing anything fancy.

**2 – Classification Models**

Here, I tackle classification problems (e.g., predicting breast cancer class). I build a few models:

Baseline classifier with DummyClassifier to see what “random guessing” looks like.

Linear models, like logistic regression with L2 regularization.

ANNs, using PyTorch with a hidden layer and activation functions like ReLU or Tanh.

I use k-fold cross-validation to estimate generalization error for each model. Finally, I do statistical comparison between models (paired t-tests) to check if the ANN or linear model actually beats the baseline in a meaningful way. I also plot learning curves for ANNs to see training progress.

**3 – Regression Models**

This script is predicting continuous values, in our case HOMA. The workflow is:

Baseline regression using DummyRegressor (just the mean of the training set).

Linear regression with Ridge and hyperparameter tuning for lambda using nested CV. I pick the lambda that gives the lowest validation error and train a final model.

ANN regression using PyTorch. The network has one hidden layer (3 units is a good start) and trains with MSE loss.

After training, we compare models by computing MSE on test data and use paired t-tests to see if the differences are statistically significant. We also plot predictions vs true values for a visual check.

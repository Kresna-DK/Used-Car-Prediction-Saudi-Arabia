# Used-Car-Prediction-Saudi-Arabia

In this machine learning project, the model is trying to recommend the price of the cars that will be listed in the syarah.com website.

The first step is data checking and data cleaning if necessary. In this dataset, the data is clean, but have some redundant columns and rows, which should be dropped before modeling the data.

Afterwards finding the correlation between the features and the target, which is the price. In this case, there are no strong correlations between the numerical features and the price.

Then the data is split into training and testing, where several models are tested to find out which has the lowest error score overall. After the test, it is shown that XGBoost Regression has the lowest overall error score, this model can be used as a benchmark.

After finding out which model should be used, it is important to know which parameters will give the best result, which is achieved by doing random cross value test. When the test is run again using the best parameters, it is compared to the benchmark.

With the best parameters, the value overall has decreased, but only by a small amount. Therefore it is better to use the best parameters when running this model.

Another way to increase the efficiency of the model is by looking at which feature has the most significant impact on the target. After finding out the "least significant" feature, it can be dropped when doing the next modelling. To make sure that the feature has no impact, several sample testing could be done, like in this project the Kruskal-Wallis and Chi-Squared test.

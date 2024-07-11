# Other

### Training and Test Splits

We can use a model to accurately fit to a set of data, with a 100% accuracy.

The issue with this is that the model would likely do a horrible job at modelling unseen data. There'd be a lack of generalization. You've essentially overfit.

A means to mitigate this is to **test** a model on unseen data, prior to inference.

To do so, you'd split your dataset into training and test sets. 

The Training set would then be used to learn the optimal parameters, while the test set, being held as unseen data, would be used to evaluate the model using a variety of metrics (i.e., F1-Score, ROC-AUC, Accuracy, MSE,$R^2$, etc).

You want to make sure you avoid any type of data leakage, make sure your test data is always independent from the training data.

TLDR

- Training Data is used to fit the model
- Test data is used to measure performance, by predicting the label with a model, comparing the label with the real value, and then measuring the error (MSE, MAE, etc)

You can use `sklearn.model_selection.train_test_split` or `sklearn.model_selection.ShuffleSplit` to split a dataset or `sklearn.model_selection.StratifiedShuffleSplit`

### Cross Validation


In cross-validation, you make different splits of the data, each varying in training and test splits. 

Each time a split is made, the training and testing datasets for that specific split are unique, meaning that the test set does not overlap with the training set for that split. 

However, across different splits, the same data points will appear in different training and test sets, ensuring that every data point is used for both training and testing across the entire cross-validation process.

A different model is trained and tested amongst all k-folds of training data, each time evaluating the loss for training and testing seperately.

Then accross all splits, we average the individual test score (loss and accuracy and f1 + other metrics) over all folds, in order to get a more broader view of accuracy and loss.

*This is done to ensure that a model didn't get lucky on a specific training split, which can be used as comparable results accross different models.*

If you only have 1 split of train-test data, it's likely that the model will overfit on that specific split and have a near perfect training error if a model is increasingly complex.

But during cross validation, given that the model is evaluated on different folds of train/test splits, overfitting will generally not be an option and the cross-validation error can increase as the issue of your model getting a 'lucky' testing score diminshes since you're validating accross multiple folds.


#### Different Cross Validation Approaches

 
# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
# import
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import autosklearn
import scipy
import seaborn as sns
import sklearn
from sklearn import preprocessing
from sklearn.metrics import make_scorer, recall_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.feature_selection import SelectFromModel

# print('autosklearn: %s' % autosklearn.__version__)
# print('autosklearn: %s' % scipy.__version__)

from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score

seed = 42


def data_visualization(df):
    # data visualization
    df.head(10)
    df.info()
    df.describe()
    df.sample(5)
    xx = df['target'].value_counts()
    print(xx)
    # sns.barplot(x='index', y='target', data=xx.reset_index())
    # Because of this imbalanceness, recall would be a better measurement for the performance.


def missing_values(df):
    # Missing values
    # get the number of missing data points per column
    missing_values_count = df.isnull().sum()

    # look at the # of missing points in the first ten columns
    missing_values_count[0:10]

    # how many total missing values do we have?
    total_cells = np.product(df.shape)
    total_missing = missing_values_count.sum()

    # percent of data that is missing
    print((total_missing / total_cells) * 100, "%")

    # Fill NaNs
    # df = df.fillna(0)
    df_miss = df.fillna(method='bfill', axis=0).fillna(0)  # bfill and remaining ones with 0
    print(df_miss.head(10))
    return df_miss


# Normalization is useful when your data has varying scales and the algorithm you are using
# does not make assumptions about the distribution of your data, such as k-nearest neighbors and artificial neural networks.
def normalization(X):
    scaler = preprocessing.MinMaxScaler()
    col = X.columns
    d = scaler.fit_transform(X)
    scaled_X = pd.DataFrame(d, columns=col)  # transform back to dataframe
    scaled_X.head()
    print(scaled_X.head(10))
    return scaled_X
    # from sklearn.preprocessing import StandardScaler
    # scaler = StandardScaler()
    # scaled_X = scaler.fit_transform(X)
    # return scaled_X


def feature_selection(df_miss, scaled_X, y):
    # feature importance method
    model = ExtraTreesClassifier()
    model.fit(scaled_X, y)

    print(model.feature_importances_)  # use inbuilt class feature_importances of tree based classifiers
    # plot graph of feature importances for better visualization
    feat_importances = pd.Series(model.feature_importances_, index=scaled_X.columns)
    feat_importances.nlargest(50).plot(kind='barh')
    plt.show()
    print(feat_importances)

    # target values
    X_feat = df_miss[feat_importances.nlargest(50).index]
    return X_feat


def random_forest_classifier(scaled_X, y):
    X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.3, random_state=seed)

    # feature selection ridge;
    sel_ = SelectFromModel(LogisticRegression(C=1, penalty='l1', solver='liblinear'))
    sel_.fit(X_train, np.ravel(y_train, order='C'))
    sel_.get_support()
    X_train = pd.DataFrame(X_train)
    selected_feat = X_train.columns[(sel_.get_support())]
    print('total features: {}'.format((X_train.shape[1])))
    print('selected features: {}'.format(len(selected_feat)))
    print('features with coefficients shrank to zero: {}'.format(np.sum(sel_.estimator_.coef_ == 0)))
    # removed_feats = X_train.columns[(sel_.estimator_.coef_ == 0).ravel().tolist()]
    X_train_sel = sel_.transform(X_train)
    X_test_sel = sel_.transform(X_test)
    print(X_train_sel.shape, X_test_sel.shape)

    # Instantiate model with 1000 decision trees
    rf = RandomForestClassifier(n_estimators=1000, random_state=seed)

    # Train the model on training data
    rf.fit(X_train_sel, np.ravel(y_train, order='C'))
    print(X_test_sel)
    y_pred = rf.predict(X_test_sel)
    print(y_test)
    print(y_pred)
    return rf, y_test, y_pred, X_train_sel, y_train


def evaluate_model(rf, y_test, y_pred, X_train, y_train):  # evaluate the model
    scoring = make_scorer(recall_score, average='micro')
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=seed)
    results = cross_val_score(rf,
                              X=X_train,
                              y=y_train,
                              cv=cv,
                              scoring=scoring)
    print(results)
    print('Recall mean: %.3f (%.3f)' % (np.mean(results), np.std(results)))

    acc = accuracy_score(y_test, y_pred)
    print("Accuracy: %.3f" % acc)
    recall = recall_score(y_test, y_pred, average='micro')
    print("Recall: %.3f" % recall)


if __name__ == '__main__':
    df = pd.read_csv('train_data.csv')
    data_visualization(df)
    df_miss = missing_values(df)
    X = df_miss.iloc[:, :-1]
    y = df_miss.iloc[:, len(df.columns) - 1]  # y should not be normalized
    print(y)
    scaled_X = normalization(X)
    # X_feat = feature_selection(df_miss, scaled_X, y)
    rf, y_test, y_pred, X_train, y_train = random_forest_classifier(scaled_X, y)
    evaluate_model(rf, y_test, y_pred, X_train, y_train)

    # parameter optimization > check notebook (onder Random Forest Classifier)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

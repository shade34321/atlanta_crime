import pandas as pd
import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split

classifiers = [
OneVsRestClassifier(SVC(kernel = 'rbf'), n_jobs=-1),
RandomForestClassifier(),
DecisionTreeClassifier(),
]


if __name__ == '__main__':

    database = pd.read_csv('preprocessed_data.csv')

    X_all = database.drop(['crime'], 1)
    y_all = database['crime']
    X_all = X_all.fillna(-1)

    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, random_state = 2)

    for clf in classifiers:
        print("Predicting type of crime ...")
        print("Classifer : ", clf)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_train)
        # print("Training accuracy :", sum(y_train == y_pred) / float(len(y_pred)))
        y_pred = clf.predict(X_test)
        print("Test accuracy :", sum(y_test == y_pred) / float(len(y_pred)))
        print("Precision score :", precision_score(y_test, y_pred, average='micro'))
        print("recall_score: ", recall_score(y_test, y_pred, average='weighted'))
        print("f1_score : ", f1_score(y_test, y_pred, average='micro'))

    arr = (database.groupby(['neighborhood']))['victims'].unique()
    count = (database.groupby(['neighborhood']))['victims'].nunique()

    for i in range(0, database.neighborhood.nunique()):
        arr[i] = sum(arr[i]) / count[i]
        database.loc[database.neighborhood == i, 'victims'] = arr[i]

    X_all = database.drop(['victims'], 1)
    y_all = database.victims
    X_all = X_all.fillna(-1)

    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, random_state = 2)

    clf = LinearRegression()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_train)
    y_pred = clf.predict(X_test)
    print("Predicting number of victims ...")
    print("Regression model : ", clf)
    print("Mean squared error : ", (np.mean(y_pred - y_test) ** 2))

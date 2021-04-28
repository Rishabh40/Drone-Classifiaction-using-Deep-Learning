import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm,datasets
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

DATA_PATH = "data_1.json"


def load_data(data_path):

    with open(data_path, "r") as fp:
        data = json.load(fp)

    X = np.array(data["mfcc"])
    y = np.array(data["labels"])
    return X, y


def prepare_datasets():

    # load data
    X, y = load_data(DATA_PATH)

    # create train and test split
    X_train, X_test, y_train, y_test  = train_test_split(X, y, random_state=0, test_size=0.25)

    return X_train, X_test, y_train, y_test



if __name__ == "__main__":

    # get train, test splits
    X_train, X_test, y_train, y_test = prepare_datasets()

    # reshaping data so it fits with svm dimention 2
    nsamples, nx, ny = X_train.shape
    X_train = X_train.reshape((nsamples, nx * ny))
    nsamples, nx, ny = X_test.shape
    X_test = X_test.reshape((nsamples, nx * ny))

    # train model
    clf=svm.SVC(kernel='linear',C=1).fit(X_train,y_train)

    # testing
    classifier_prediction=clf.predict(X_test)
    print(accuracy_score(y_test,classifier_prediction)*100)
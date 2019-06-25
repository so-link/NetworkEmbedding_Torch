#coding:utf-8

import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

import tqdm

from openne.classify import Classifier

def openne_transform_label(labels):
    X = []
    Y = []
    for node, label in labels.items():
        X.append(node)
        Y.append(label)
    return X, Y


def openne_node_classification(embeddings, labels, clf_ratio=0.5, clf=None):
    if clf is None:
        clf = LogisticRegression()
    classifier = Classifier(vectors=embeddings, clf=clf)
    X, Y = openne_transform_label(labels)
    return classifier.split_train_evaluate(X, Y, clf_ratio)

def node_classification(embeddings, labels, clf_ratio=0.5, clf=None):
    if clf is None:
        clf = LogisticRegression()
    X = []
    Y = []
    for node,label in labels.items():
        X.append(embeddings[node])
        Y.append(label)
    
    X = np.array(X)
    Y = np.array(Y)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1-clf_ratio, random_state=233)

    classifier = OneVsRestClassifier(clf)
    classifier.fit(X_train, Y_train)

    Y_pred = classifier.predict(X_test)

    return {
        'accuracy': accuracy_score(Y_test, Y_pred), 
        'macro_f1': f1_score(Y_test, Y_pred, average='macro'), 
        'micro_f1': f1_score(Y_test, Y_pred, average='micro'),
    }

def evaluate(embeddings, labels, clf_ratio=0.5, classifier=None, seed=233):
    if classifier is None:
        classifier = LogisticRegression()
    macro_f1 = []
    micro_f1 = []
    state = np.random.get_state()
    np.random.seed(seed)
    result = openne_node_classification(embeddings, labels, clf_ratio=clf_ratio, clf=classifier)
    macro_f1.append(result['macro_f1'])
    micro_f1.append(result['micro_f1'])
    np.random.set_state(state)
    return {
        'macro_f1': result['macro_f1'],
        'micro_f1': result['micro_f1'],
    }
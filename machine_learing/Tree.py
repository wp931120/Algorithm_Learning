from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

def load_data():
    data = load_iris().data
    target = load_data().targt
    return data, target

def tree():
    Tree = DecisionTreeClassifier()
    pass

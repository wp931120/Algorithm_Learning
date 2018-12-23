from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score,f1_score


class Trees:

    def __init__(self, model_type="decisiontree"):
        self.train_data, self.test_data, self.train_label, self.test_label = train_test_split(load_iris().data,
                                                                                              load_iris().target,
                                                                                              test_size=0.3,
                                                                                              random_state=11)
        self.tree = self.choose_tree(model_type)
        # self.score = self.score()

    def choose_tree(self, model_type):
        if model_type == "decisiontree":
            tree = DecisionTreeClassifier()
        elif model_type == "randomforest":
            tree = RandomForestClassifier()
        elif model_type == "gbdt":
            tree = GradientBoostingClassifier()
        elif model_type == "extrat":
            tree = ExtraTreesClassifier()
        else:
            tree = DecisionTreeClassifier()
        return tree

    def train(self):
        self.tree.fit(self.train_data, self.train_label)

    def acc_score(self):
        predict = self.tree.predict(self.test_data)
        return accuracy_score(predict, self.test_label)

    def f1_score(self):
        predict = self.tree.predict(self.test_data)
        return f1_score(predict, self.test_label)

if __name__ == "__main__":
    tree = Trees()
    tree.train()
    print(tree.acc_score())
    tree1 = Trees("randomforest" )
    tree1.train()
    print(tree1.acc_score())
    tree2 = Trees("gbdt")
    tree2.train()
    print(tree2.acc_score())
    tree3 = Trees("extrat")
    tree3.train()
    print(tree3.acc_score())




from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def load_data():
    data_x = load_boston().data
    data_y = load_boston().target
    return data_x, data_y


def plot(target_):
    plt.scatter([i for i in range(len(target_))], target_)
    plt.ylabel("price")
    plt.show()


class Linear:
    """
    A linear model class conclude some linear models
    """

    def __init__(self, data_in, target_in):
        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(data_in, target_in)

    def lr(self):
        """
        lr
        loss = |wx + b - y_label|^2
        :return:
        """
        lr_ = LinearRegression()
        lr_.fit(self.train_x, self.train_y)
        return lr_

    def ridge(self, l2=0.2):
        """
        add the l2 regularization to lr loss
        Ridge_loss = |wx + b - y_label|^2 + alpha*w^2
        :param l2:
        :return:
        """
        ridge_ = Ridge(alpha=l2)
        ridge_.fit(self.train_x, self.train_y)
        return ridge_

    def lasso(self, l1=0.5):
        """
         add the l1 regularization to lr loss
        Lasso_loss = |wx + b - y_label|^2 + alpha*|w|
        :param l1:
        :return:
        """
        lasso_ = Lasso(alpha=l1)
        lasso_.fit(self.train_x, self.train_y)
        return lasso_

    def elastic(self, l2=0.5, l1=0.5):
        """
         add the l1 regularization and l2 regularization to lr loss
        Elastic_loss = |wx + b - y_label|^2 + alpha*w^2 + l1_ratio*|w|
        :param l2:
        :param l1:
        :return:
        """
        elastic_ = ElasticNet(alpha=l2, l1_ratio=l1)
        elastic_.fit(self.train_x, self.train_y)
        return elastic_

    def metric(self, model):
        predict = model.predict(self.test_x)
        mse = mean_squared_error(predict, self.test_y)
        return mse

    @staticmethod
    def predict(model, test):
        predict = model.predict(test)
        return predict


if __name__ == "__main__":
    data, target = load_data()
    plot(target)
    Linear_models = Linear(data, target)
    print("加上正则：")
    lr = Linear_models.lr()
    rg = Linear_models.ridge()
    la = Linear_models.lasso()
    el = Linear_models.elastic()
    print("lr的mse损失为{}".format(Linear_models.metric(lr)))
    print("ridge的mse损失为{}".format(Linear_models.metric(rg)))
    print("lasso的mse损失为{}".format(Linear_models.metric(la)))
    print("elastic的mse损失为{}".format(Linear_models.metric(el)))
    print("正则项系数都设为0之后：")
    lr = Linear_models.lr()
    rg = Linear_models.ridge(0)
    la = Linear_models.lasso(0)
    el = Linear_models.elastic(0, 0)
    print("lr的mse损失为{}".format(Linear_models.metric(lr)))
    print("ridge的mse损失为{}".format(Linear_models.metric(rg)))
    print("lasso的mse损失为{}".format(Linear_models.metric(la)))
    print("elastic的mse损失为{}".format(Linear_models.metric(el)))

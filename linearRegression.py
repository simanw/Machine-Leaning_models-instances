from numpy import random, matrix, linalg
import numpy as np
from matplotlib import pyplot

class DataGenerator(object):
    """
    Generate training set and testing set with different distributions
    """

    def __init__(self):
        pass

    def uniform(self, scale=(0,0), size=None):
        """
        Draw data from uniform distribution
        :param scale: tuple
            scale[0] is low; scale[1] is high.
        :param size: int or tuple
            Output shape
        """
        x = random.uniform(scale[0], scale[1], size)

        return x

    def guassian(self, miu=0.0, sigma=0.0, size=None):
        """
        Draw data from guassian distribution
        :param miu: float
            Mean of the distribution
        :param sigma: float
            Standard deviation of the distribution
        :param size: int or tuple
            Output shape
        """
        x = random.normal(miu, sigma, size)
        return x

    def label_generator(self, x, w, bias=None):
        """
        This is an experiment for linear regression. Thus, we generate the labels
        using a linear function.
        :param x: ndarray
            feature vectors
        :param w: ndarray
            real weights
        :param bias: ndarray
            bias, following a specific distribution
        :return: ndarray
        """
        y = np.dot(x, w) + bias
        return y


class Predictor(object):

    def __init__(self, test_x=None, learnt_weights=None):
        self.test_x = test_x
        self.learnt_weights = learnt_weights
        pass

    def predict(self):
        x = self.test_x
        w = self.learnt_weights
        return np.dot(w.T, x)


class Evaluation(object):
    """

    """
    def __init__(self):
        pass

    def MSE(self, real=None, predicted=None):
        return np.sum((real - predicted) ** 2) / real.size

class ModelLearner(object):

    def __init__(self, train_x=None, train_y=None):
        self.train_x = train_x
        self.train_y = train_y
        pass

    def closed_form_solution(self):
        """
        W = (X^TX)^-1X^TY
        :return: learnt weight
        """
        x = self.train_x
        y = self.train_y
        temp = linalg.inv(x.T.dot(x))
        w = temp.dot(x.T.dot(y))
        return w

    def SGD(self, MSE, terminates=10000, lr=0.01, sample_size=0):
        w = np.zeros(self.train_x.shape[1])
        evaluate = Evaluation()

        for i in range(terminates):
            if i % 100 == 0:
                predicted_y = np.dot(w.T, self.train_x)
                curr_MSE = evaluate.MSE(self.train_y, predicted_y)
                MSE.append(curr_MSE)

            index = random.randint(1,sample_size)
            x_i = self.train_x[index, :]
            w = w + lr * (self.train_y - np.dot(w.T, x_i)).dot(x_i)

class Experiment(object):

    def __init__(self, miu=0.0, sigma=0.0, range=(-1, 1), sample_size=0):
        data_generator = DataGenerator()

        real_w = np.array([2, -1])
        bias = data_generator.guassian(miu, sigma, sample_size)
        x_n = data_generator.uniform(range, (sample_size, 2))
        x_0 = np.ones((sample_size, 1))
        self.x = np.concatenate((x_n, x_0), axis=1)
        self.y = data_generator.label_generator(x_n, real_w, bias)


    def closed_form_solution(self):

        ks = [10, 100, 1000, 10000]

        for k in ks:
            train_x = self.x[0:k, :]
            test_x = self.x[10000:, :]
            train_y = self.y[0:k]
            test_y = self.y[10000:]
            model_learner = ModelLearner(train_x, train_y)
            learnt_w = model_learner.closed_form_solution()

            predictor1 = Predictor(test_x, learnt_w)
            predicted_test_y = predictor1.predict()
            predictor2 = Predictor(train_x, learnt_w)
            predicted_train_y = predictor2.predict()


            evaluate = Evaluation()
            train_MSE = evaluate.MSE(train_y, predicted_train_y)
            test_MSE = evaluate.MSE(test_y, predicted_test_y)

            print("k = ", k, train_MSE, test_MSE)

    def stochastic_gradient_descent(self):

        learning_rates = [0.1, 0.01, 0.001]
        MSE_history = []
        train_x = self.x
        train_y = self.y
        model_learner = ModelLearner(train_x, train_y)

        for lr in learning_rates:
            model_learner.SGD(MSE_history, 10000, lr,10000)
        print(MSE_history)
        #pyplot.plot(learning_rates, MSE_history)



print("hhe")
experiment_A = Experiment(0, 0.1, (-1, 1), 20000)
experiment_A.closed_form_solution()

#experiment_B = Experiment(0, 1, 10000)
#experiment_B.stochastic_gradient_descent()
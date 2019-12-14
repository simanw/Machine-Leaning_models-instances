import numpy as np


class StochasticGradientDescent(object):
    """
    Gradient descent is one of the most widely used optimizaiton algorithm.
    It can be used in minimize loss functions or maximize objective functions.

    SGD repeatedly runs through the training set, and each time we encounter
    a training example, we update the parameters theta according to the gradient
    of the error with respect to that single training example only.
    """

    def __init__(
            self, lr=0.01, momentum=0.0, lr_scheduler=None
    ):
        """
        :param lr: float
            Learning rate of SGD. Default is 0.01.
        :param momentum: float in range [0, 1]
            The fraction of the previous update to add to the current update. If 0, no momentum is applied.
        :param lr_schedule:
            The learning rate scheduler. If None, use a constant learning rate that equals to 'lr

        """


from dataclasses import dataclass
from enum import auto
from enum import Enum
from typing import Dict
from typing import Type

import numpy as np


HUBERLOSS_DELTA = 1.35


@dataclass
class LearningRate:
    lambda_: float = 1e-3
    s0: float = 1
    p: float = 0.5

    iteration: int = 0

    def __call__(self):
        """
        Calculate learning rate according to lambda (s0/(s0 + t))^p formula
        """
        self.iteration += 1
        return self.lambda_ * (self.s0 / (self.s0 + self.iteration)) ** self.p


class LossFunction(Enum):
    MSE = auto()
    MAE = auto()
    LogCosh = auto()
    Huber = auto()


class BaseDescent:
    """
    A base class and templates for all functions
    """

    def __init__(self, dimension: int, lambda_: float = 1e-3, loss_function: LossFunction = LossFunction.MSE):
        """
        :param dimension: feature space dimension
        :param lambda_: learning rate parameter
        :param loss_function: optimized loss function
        """
        self.w: np.ndarray = np.random.rand(dimension)
        self.lr: LearningRate = LearningRate(lambda_=lambda_)
        self.loss_function: LossFunction = loss_function

    def step(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.update_weights(self.calc_gradient(x, y))

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        """
        Template for update_weights function
        Update weights with respect to gradient
        :param gradient: gradient
        :return: weight difference (w_{k + 1} - w_k): np.ndarray
        """
        pass

    def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Template for calc_gradient function
        Calculate gradient of loss function with respect to weights
        :param x: features array
        :param y: targets array
        :return: gradient: np.ndarray
        """
        pass

    def calc_loss(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate loss for x and y with our weights
        :param x: features array
        :param y: targets array
        :return: loss: float
        """
        loss = x @ self.w - y
        if self.loss_function is LossFunction.MSE:
            return np.dot(loss, loss) / y.shape[0]
        elif self.loss_function is LossFunction.LogCosh:
            return np.log(np.cosh(loss)).mean()
        elif self.loss_function is LossFunction.MAE:
            return np.linalg.norm(loss, ord=1) / y.shape[0]
        elif self.loss_function is LossFunction.Huber:
            return np.where(
                np.abs(loss) < HUBERLOSS_DELTA,
                (loss * loss) / 2,
                HUBERLOSS_DELTA * np.abs(loss) - HUBERLOSS_DELTA ** 2 / 2
            ).mean()

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Calculate predictions for x
        :param x: features array
        :return: prediction: np.ndarray
        """
        return x @ self.w


class VanillaGradientDescent(BaseDescent):
    """
    Full gradient descent class
    """

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        """
        Updates weights and returns difference
        :return: weight difference (w_{k + 1} - w_k): np.ndarray
        """
        step = -self.lr() * gradient
        self.w += step
        return step

    def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Calculates loss gradient
        :return: loss gradient
        """
        loss = x @ self.w - y
        if self.loss_function is LossFunction.MSE:
            return 2 * x.T @ loss / y.shape[0]
        elif self.loss_function is LossFunction.LogCosh:
            return x.T @ np.tanh(loss)
        elif self.loss_function is LossFunction.MAE:
            return x.T @ np.sign(loss) / y.shape[0]
        elif self.loss_function is LossFunction.Huber:
            return x.T @ np.where(
                np.abs(loss) < HUBERLOSS_DELTA,
                loss,
                HUBERLOSS_DELTA * np.sign(loss)
            ) / y.shape[0]


class StochasticDescent(VanillaGradientDescent):
    """
    Stochastic gradient descent class
    """

    def __init__(self, dimension: int, lambda_: float = 1e-3, batch_size: int = 50,
                 loss_function: LossFunction = LossFunction.MSE):
        """
        :param batch_size: batch size (int)
        """
        super().__init__(dimension, lambda_, loss_function)
        self.batch_size = batch_size

    def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Calculates gradient as the mean of gradients for random batch
        :return: mean of gradients for samples in batch
        """
        batch_indexes = np.random.randint(0, y.shape[0], self.batch_size)
        x_batch, y_batch = x[batch_indexes], y[batch_indexes]
        return super().calc_gradient(x_batch, y_batch)


class MomentumDescent(VanillaGradientDescent):
    """
    Momentum gradient descent class
    """

    def __init__(self, dimension: int, lambda_: float = 1e-3, loss_function: LossFunction = LossFunction.MSE):
        super().__init__(dimension, lambda_, loss_function)
        self.alpha: float = 0.9

        self.h: np.ndarray = np.zeros(dimension)

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        """
        Updates momentum, then updates weights with calculated momentum
        :return: momentum = weight difference (w_{k + 1} - w_k): np.ndarray
        """
        self.h = self.alpha * self.h + self.lr() * gradient
        self.w -= self.h
        return -self.h


class Adam(VanillaGradientDescent):
    """
    Adaptive Moment Estimation gradient descent class
    """

    def __init__(self, dimension: int, lambda_: float = 1e-3, loss_function: LossFunction = LossFunction.MSE):
        super().__init__(dimension, lambda_, loss_function)
        self.eps: float = 1e-8

        self.m: np.ndarray = np.zeros(dimension)
        self.v: np.ndarray = np.zeros(dimension)

        self.beta_1: float = 0.9
        self.beta_2: float = 0.999

        self.iteration: int = 0

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        """
        Updates mean and variance, then weights
        :return: weight difference (w_{k + 1} - w_k): np.ndarray
        """
        self.iteration += 1
        np.add(self.beta_1 * self.m, (1 - self.beta_1) * gradient, out=self.m)
        np.add(self.beta_2 * self.v, (1 - self.beta_2) * (gradient ** 2), out=self.v)

        m_norm = self.m / (1 - np.power(self.beta_1, self.iteration))
        v_norm = self.v / (1 - np.power(self.beta_2, self.iteration))

        step = -self.lr() * m_norm / np.sqrt(v_norm + self.eps)
        self.w += step
        return step


class AdaMax(VanillaGradientDescent):
    """
    Adaptive Moment Estimation gradient descent class on the infinity norm
    """

    def __init__(self, dimension: int, lambda_: float = 1e-3, loss_function: LossFunction = LossFunction.MSE):
        super().__init__(dimension, lambda_, loss_function)
        self.m: np.ndarray = np.zeros(dimension)
        self.u: float = 0.0

        self.beta_1: float = 0.9
        self.beta_2: float = 0.999

        self.iteration: int = 0

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        """
        Updates mean and variance, then weights
        :return: weight difference (w_{k + 1} - w_k): np.ndarray
        """
        self.iteration += 1
        np.add(self.beta_1 * self.m, (1 - self.beta_1) * gradient, out=self.m)
        self.u = max(self.beta_2 * self.u, np.linalg.norm(gradient, ord=1))

        m_norm = self.m / (1 - np.power(self.beta_1, self.iteration))

        step = -self.lr() * m_norm / self.u
        self.w += step
        return step


class BaseDescentReg(BaseDescent):
    """
    A base class with regularization
    """

    def __init__(self, *args, mu: float = 0, **kwargs):
        """
        :param mu: regularization coefficient (float)
        """
        super().__init__(*args, **kwargs)
        self.mu = mu

    def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Calculate gradient of loss function and L2 regularization with respect to weights
        """
        l2_gradient: np.ndarray = self.w.copy()
        l2_gradient[-1] = 0
        return super().calc_gradient(x, y) + l2_gradient * self.mu


class VanillaGradientDescentReg(BaseDescentReg, VanillaGradientDescent):
    """
    Full gradient descent with regularization class
    """


class StochasticDescentReg(BaseDescentReg, StochasticDescent):
    """
    Stochastic gradient descent with regularization class
    """


class MomentumDescentReg(BaseDescentReg, MomentumDescent):
    """
    Momentum gradient descent with regularization class
    """


class AdamReg(BaseDescentReg, Adam):
    """
    Adaptive gradient algorithm with regularization class
    """


class AdaMaxReg(BaseDescentReg, AdaMax):
    """
    Adaptive gradient algorithm on the infinity norm with regularization class
    """


def get_descent(descent_config: dict) -> BaseDescent:
    descent_name = descent_config.get('descent_name', 'full')
    regularized = descent_config.get('regularized', False)

    descent_mapping: Dict[str, Type[BaseDescent]] = {
        'full': VanillaGradientDescent if not regularized else VanillaGradientDescentReg,
        'stochastic': StochasticDescent if not regularized else StochasticDescentReg,
        'momentum': MomentumDescent if not regularized else MomentumDescentReg,
        'adam': Adam if not regularized else AdamReg,
        'adamax': AdaMax if not regularized else AdaMaxReg
    }

    if descent_name not in descent_mapping:
        raise ValueError(f'Incorrect descent name: {descent_name}, use one of these: {descent_mapping.keys()}')

    descent_class = descent_mapping[descent_name]

    return descent_class(**descent_config.get('kwargs', {}))

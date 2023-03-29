from __future__ import annotations

from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeRegressor


EPS = 1e-7


def score(clf, x, y):
    return roc_auc_score(y == 1, clf.predict_proba(x)[:, 1])


class Boosting:

    def __init__(
            self,
            base_model_params: dict = None,
            n_estimators: int = 10,
            learning_rate: float = 0.1,
            subsample: float = 0.3,
            early_stopping_rounds: int = None,
            plot: bool = False,
    ):
        self.base_model_class = DecisionTreeRegressor
        self.base_model_params: dict = {} if base_model_params is None else base_model_params

        self.n_estimators: int = n_estimators

        self.models: list = []
        self.gammas: list = []

        self.learning_rate: float = learning_rate
        self.subsample: float = subsample

        self.early_stopping_rounds: int = early_stopping_rounds
        if early_stopping_rounds is not None:
            self.validation_loss = np.full(self.early_stopping_rounds, np.inf)

        self.plot: bool = plot

        self.history = defaultdict(list)
        self.feature_importances = None

        self.sigmoid = lambda x: 1 / (1 + np.exp(-x))
        self.loss_fn = lambda y, z: -np.log(self.sigmoid(y * z)).mean()
        self.loss_derivative = lambda y, z: -y * self.sigmoid(-y * z)

    def fit_new_base_model(self, x, y, predictions):
        boot_ind = np.random.choice(
            x.shape[0],
            size=int(self.subsample * x.shape[0]),
            replace=True
        )

        x_boot, y_boot, pred_boot = x[boot_ind], y[boot_ind], predictions[boot_ind]
        # s_boot = np.array(list(map(lambda p: self.loss_derivative(p[0], p[1]), zip(y_boot, pred_boot))))
        s_boot = self.loss_derivative(y_boot, pred_boot)
        base_model = self.base_model_class(**self.base_model_params)
        base_model.fit(x_boot, -s_boot)

        self.gammas.append(self.find_optimal_gamma(y, predictions, base_model.predict(x)))
        self.models.append(base_model)

    def fit(self, x_train, y_train, x_valid=None, y_valid=None):
        """
        :param x_train: features array (train set)
        :param y_train: targets array (train set)
        :param x_valid: features array (validation set)
        :param y_valid: targets array (validation set)
        """
        train_predictions = np.zeros(y_train.shape[0])

        for i in range(self.n_estimators):
            self.fit_new_base_model(x_train, y_train, train_predictions)

            train_predictions = self.models[-1].predict(x_train)
            valid_predictions = self.models[-1].predict(x_valid) if x_valid is not None else None

            self.history[i] = [
                sum(map(lambda p: self.loss_fn(p[0], p[1]), zip(y_train, train_predictions))),
                sum(map(lambda p: self.loss_fn(p[0], p[1]), zip(y_valid, valid_predictions)))
                if x_valid is not None else None
            ]

            if self.early_stopping_rounds is not None:
                self.validation_loss[i] = self.score(x_valid, y_valid)
                if i == self.early_stopping_rounds and \
                        np.abs(self.validation_loss[i], self.validation_loss[i - 1]) < EPS:
                    break

        if self.plot:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            axes[0][0].plot(range(len(self.history)), map(lambda r: r[0], self.history))
            axes[0][0].set_title('Train losses')
            axes[0][1].plot(range(len(self.history)), map(lambda r: r[1], self.history))
            axes[0][1].set_title('Valid losses')

            plt.show()

    def predict_proba(self, x):
        predict_probas = np.zeros(x.shape[0])
        lr = 1.0

        for gamma, model in zip(self.gammas, self.models):
            predict_probas += lr * gamma * model.predict(x)
            lr *= self.learning_rate

        predict_probas = np.array(list(map(self.sigmoid, predict_probas)))
        predict_probas = np.column_stack((1 - predict_probas, predict_probas))
        return predict_probas

    def predict(self, x):
        predicts = np.zeros(x.shape[0])
        lr = 1.0
        for gamma, model in zip(self.gammas, self.models):
            predicts += lr * gamma * model.predict(x)
            lr *= self.learning_rate
        return np.sign(predicts)

    def find_optimal_gamma(self, y, old_predictions, new_predictions) -> float:
        gammas = np.linspace(start=0, stop=1, num=100)
        losses = [self.loss_fn(y, old_predictions + gamma * new_predictions) for gamma in gammas]
        return gammas[np.argmin(losses)]

    def score(self, x, y):
        return score(self, x, y)

    def get_params(self, deep=False):
        return {
            'n_estimators': self.n_estimators,
            'learning_rate': self.learning_rate,
            'subsample': self.subsample
        }

    @property
    def feature_importances_(self):
        if self.feature_importances is None:
            feature_importances = np.zeros(self.models[0].feature_importances_.shape)
            for model in self.models:
                feature_importances += model.feature_importances_

            feature_importances /= np.sum(feature_importances)
            assert np.all(feature_importances >= 0), feature_importances

            self.feature_importances = feature_importances

        return self.feature_importances

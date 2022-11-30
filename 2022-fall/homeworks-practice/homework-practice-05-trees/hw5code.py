import numpy as np
from collections import Counter


def find_best_split(feature_vector, target_vector, min_samples_leaf=None):
    """
    Под критерием Джини здесь подразумевается следующая функция:
    $$Q(R) = -\frac {|R_l|}{|R|}H(R_l) -\frac {|R_r|}{|R|}H(R_r)$$,
    $R$ — множество объектов, $R_l$ и $R_r$ — объекты, попавшие в левое и правое поддерево,
     $H(R) = 1-p_1^2-p_0^2$, $p_1$, $p_0$ — доля объектов класса 1 и 0 соответственно.
sor
    Указания:
    * Пороги, приводящие к попаданию в одно из поддеревьев пустого множества объектов, не рассматриваются.
    * В качестве порогов, нужно брать среднее двух сосдених (при сортировке) значений признака
    * Поведение функции в случае константного признака может быть любым.
    * При одинаковых приростах Джини нужно выбирать минимальный сплит.
    * За наличие в функции циклов балл будет снижен. Векторизуйте! :)

    :param feature_vector: вещественнозначный вектор значений признака
    :param target_vector: вектор классов объектов,  len(feature_vector) == len(target_vector)

    :return thresholds: отсортированный по возрастанию вектор со всеми возможными порогами, по которым объекты можно
     разделить на две различные подвыборки, или поддерева
    :return ginis: вектор со значениями критерия Джини для каждого из порогов в thresholds len(ginis) == len(thresholds)
    :return threshold_best: оптимальный порог (число)
    :return gini_best: оптимальное значение критерия Джини (число)
    """
    # ╰( ͡° ͜ʖ ͡° )つ──☆*:・ﾟ
    if not isinstance(feature_vector, np.ndarray):
        # print(f'features type: {type(feature_vector)}')
        feature_vector = feature_vector.to_numpy()
    if not isinstance(target_vector, np.ndarray):
        # print(f'targets type: {type(target_vector)}')
        target_vector = target_vector.to_numpy()

    R = np.argsort(feature_vector)
    features = feature_vector[R]
    targets = target_vector[R]
    N = len(features)

    features_unique, unique_ind = np.unique(features, return_index=True)
    thresholds = 0.5 * (features_unique[1:] + features_unique[:-1])

    Rl = unique_ind[1:]

    if min_samples_leaf:
        relevant_ind = np.where((Rl >= min_samples_leaf) & (Rl <= N - min_samples_leaf))
        Rl = Rl[relevant_ind]
        thresholds = thresholds[relevant_ind]

    targets_cumsum = targets.cumsum()

    p1l = targets_cumsum[Rl - 1]
    p1r = targets_cumsum[-1] - p1l
    p1l = p1l / Rl
    p1r = p1r / (N - Rl)

    Hl, Hr = (1 - p1 ** 2 - (1 - p1) ** 2 for p1 in [p1l, p1r])
    Q = -(Rl / N * Hl) - ((N - Rl) / N * Hr)

    try:
        best_ind = np.nanargmax(Q)
    except ValueError:
        return None, None, None, None

    best_threshold, best_gini = thresholds[best_ind], Q[best_ind]
    return thresholds, Q, best_threshold, best_gini


class DecisionTree:
    def __init__(self, feature_types, feature_shape_on_fit=False,
                 max_depth=None, min_samples_split=None, min_samples_leaf=None):

        if np.any(list(map(lambda x: x != "real" and x != "categorical", feature_types))):
            raise ValueError("There is unknown feature type")
        if feature_shape_on_fit and len(feature_types) > 1:
            raise ValueError("Only 1 type of features is supported with unknown shape")

        self._tree = {}
        self._feature_types = feature_types
        self._feature_shape_on_fit = feature_shape_on_fit
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf

        if self._max_depth:
            self._tree["depth"] = 0

    def _fit_node(self, sub_X, sub_y, node):
        if self._feature_shape_on_fit:
            self._feature_types = [self._feature_types[0] for _ in range(sub_X.shape[1])]

        if np.all(sub_y == sub_y[0]):                         # != -> ==
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return

        if self._max_depth and node["depth"] >= self._max_depth \
                or self._min_samples_split and sub_X.shape[0] < self._min_samples_split \
                or self._min_samples_leaf and sub_X.shape[0] < 2 * self._min_samples_leaf:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        feature_best, threshold_best, gini_best, split = None, None, None, None
        for feature in range(sub_X.shape[1]):              # range(1, sub_X.shape[1]) -> range(sub_X.shape[1])
            feature_type = self._feature_types[feature]
            categories_map = {}

            if feature_type == "real":
                feature_vector = sub_X[:, feature]
            elif feature_type == "categorical":
                counts = Counter(sub_X[:, feature])
                clicks = Counter(sub_X[sub_y == 1, feature])
                ratio = {}
                for key, current_count in counts.items():
                    if key in clicks:
                        current_click = clicks[key]
                    else:
                        current_click = 0
                    ratio[key] = current_click / current_count  # swapped frac

                sorted_categories = list(map(lambda x: x[0], sorted(ratio.items(), key=lambda x: x[1])))
                # x[1] -> x[0]
                categories_map = dict(zip(sorted_categories, list(range(len(sorted_categories)))))

                feature_vector = np.array(list(map(lambda x: categories_map[x], sub_X[:, feature])))
                # added cast to list
            else:
                raise ValueError

            # if len(feature_vector) == 3:
            #     continue
            # swapped to ->
            if np.all(feature_vector == feature_vector[0]):
                continue

            _, _, threshold, gini = find_best_split(feature_vector, sub_y, self._min_samples_leaf)
            if threshold is not None:
                if gini_best is None or gini > gini_best:
                    feature_best = feature
                    gini_best = gini
                    split = feature_vector < threshold

                    if feature_type == "real":
                        threshold_best = threshold
                    elif feature_type == "categorical":
                        threshold_best = list(map(lambda x: x[0],
                                                  filter(lambda x: x[1] < threshold, categories_map.items())))
                    else:
                        raise ValueError

        if feature_best is None:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]  # added [0][0]
            return

        node["type"] = "nonterminal"

        node["feature_split"] = feature_best
        if self._feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        elif self._feature_types[feature_best] == "categorical":
            node["categories_split"] = threshold_best
        else:
            raise ValueError

        node["left_child"], node["right_child"] = {}, {}

        if self._max_depth:
            node["left_child"]["depth"] = node["depth"] + 1
            node["right_child"]["depth"] = node["depth"] + 1

        self._fit_node(sub_X[split], sub_y[split], node["left_child"])
        self._fit_node(sub_X[np.logical_not(split)], sub_y[np.logical_not(split)], node["right_child"])
        # added np.logical_not for sub_y

    def _predict_node(self, x, node):
        if node["type"] == "terminal":
            return node["class"]

        feature_split = node["feature_split"]
        feature_type = self._feature_types[feature_split]

        if feature_type == "real":
            if x[feature_split] < node["threshold"]:
                return self._predict_node(x, node["left_child"])
            return self._predict_node(x, node["right_child"])

        if x[feature_split] in node["categories_split"]:
            return self._predict_node(x, node["left_child"])
        return self._predict_node(x, node["right_child"])

    def fit(self, X, y):
        self._fit_node(X, y, self._tree)

    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)

    def get_params(self, deep=False):
        return {
            'feature_types': self._feature_types,
            'feature_shape_on_fit': self._feature_shape_on_fit,
            'max_depth': self._max_depth,
            'min_samples_split': self._min_samples_split,
            'min_samples_leaf': self._min_samples_leaf
        }

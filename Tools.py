import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from sklearn.linear_model import Ridge
from DataClass import XYClass


def check_orthonormality(a_mat: np.array) -> bool:
    """
    Check if the column of a_mat are orthonormal
    :param a_mat: array of mat_a
    :return: True if orthonormal, False if not
    """
    res = True
    d, f = a_mat.shape
    err = pd.DataFrame(a_mat.T @ a_mat - np.eye(f)).abs()

    if any(err.unstack() > 1e-6):
        res = False

    return res


def gram_schmidt(a_mat: np.array) -> np.array:
    """
    Orthonormalize a_mat using the gram_schmidt method
    :param a_mat: array of a_mat
    :return: a_mat orthonormalized
    """
    return np.linalg.qr(a_mat)[0]


def metric(xy_class: XYClass, a_mat: np.array, beta: np.array) -> float:
    """
    Compute the score of model (a_mat, beta) on the dataset of the XYClass xy_class
    :param xy_class: class XYClass containing X and Y
    :param a_mat: matrix A
    :param beta: beta vector
    :return: score
    """
    assert a_mat.shape == (250, 10), 'A not good shape'
    assert (beta.shape == (10,)) or (beta.shape == (10, 1)), 'beta not good shape'

    if not check_orthonormality(a_mat):
        return -1.0

    x_df = xy_class.x_df.copy()
    y_df = xy_class.y_df.copy()

    assert x_df.index.names == ['date', 'stocksID']
    assert y_df.index.names == ['date', 'stocksID']

    y_pred = (x_df @ a_mat @ beta).unstack().T
    y_true = y_df.unstack().T

    y_true = y_true.div(np.sqrt((y_true ** 2).sum()), 1)
    y_pred = y_pred.div(np.sqrt((y_pred ** 2).sum()), 1)

    mean_overlap = (y_true * y_pred).sum().mean()

    return round(mean_overlap, 10)


def parameters_transform(a_mat: np.array, beta: np.array, d: int = 250, f: int = 10) -> np.array:
    """
    Transform A and beta to the good shape for submission
    :param a_mat: matrix A
    :param beta: vector beta
    :param d: time depth
    :param f: number of factors
    :return: output for submission
    """
    if a_mat.shape != (d, f):
        print('A has not the good shape')
        return

    if beta.shape[0] != f:
        print('beta has not the good shape')
        return

    output = np.hstack((np.hstack([a_mat.T, beta.reshape((f, 1))])).T)

    return output


class OrthonormalityKernelConstraint(tf.keras.constraints.Constraint):
    """
    Class to add an orthonormality constrain on neural network layer
    """

    def __init__(self):
        pass

    def __call__(self, w):
        w = tfp.math.gram_schmidt(w)
        return w


def fit_beta(xy_class: XYClass, a_mat: np.array) -> np.array:
    """
    Fit beta with linear regression using the X and Y fo xy_class
    :param xy_class: XYClass containing X and Y
    :param a_mat: matrix A
    :return: beta
    """
    x_df = xy_class.x_df
    y_df = xy_class.y_df

    predictors = x_df @ a_mat
    beta = np.linalg.inv(predictors.T @ predictors) @ predictors.T @ y_df

    return beta.to_numpy()


def fit_beta_ridge(xy_class, a_mat, alpha):
    """
    Fit beta using the X and Y fo xy_class with Ridge linear regression of parameter alpha
    :param xy_class: XYClass containing X and Y
    :param a_mat: matrix A
    :param alpha: parameter of the Ridge regression
    :return: beta
    """
    x = xy_class.x_tab @ a_mat
    y = xy_class.y_tab
    model = Ridge(alpha=alpha, fit_intercept=False).fit(x, y)
    beta = model.coef_
    return beta


def random_mat_a(d: int = 250, f: int = 10) -> np.array:
    """
    generate f random orthonormal vectors of size d
    :param d: vector size
    :param f: number of vectors
    :return: matrix with columns the orthonormal vectors
    """
    mat_m = np.random.randn(d, f)
    random_stiefel = gram_schmidt(mat_m)  # Apply Gram-Schmidt algorithm to the columns of mat_m

    return random_stiefel

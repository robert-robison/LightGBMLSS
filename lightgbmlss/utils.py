from typing import Callable, Union
import torch
from torch.nn.functional import softplus, gumbel_softmax, softmax
import lightgbm as lgb
import numpy as np
import pandas as pd


def nan_to_num(predt: torch.tensor) -> torch.tensor:
    """
    Replace nan, inf and -inf with the mean of predt.

    Arguments
    ---------
    predt: torch.tensor
        Predicted values.

    Returns
    -------
    predt: torch.tensor
        Predicted values.
    """
    predt = torch.nan_to_num(predt,
                             nan=float(torch.nanmean(predt)),
                             posinf=float(torch.nanmean(predt)),
                             neginf=float(torch.nanmean(predt))
                             )

    return predt


def identity_fn(predt: torch.tensor) -> torch.tensor:
    """
    Identity mapping of predt.

    Arguments
    ---------
    predt: torch.tensor
        Predicted values.

    Returns
    -------
    predt: torch.tensor
        Predicted values.
    """
    predt = nan_to_num(predt) + torch.tensor(0, dtype=predt.dtype)

    return predt


def exp_fn(predt: torch.tensor) -> torch.tensor:
    """
    Exponential function used to ensure predt is strictly positive.

    Arguments
    ---------
    predt: torch.tensor
        Predicted values.

    Returns
    -------
    predt: torch.tensor
        Predicted values.
    """
    predt = torch.exp(nan_to_num(predt)) + torch.tensor(1e-06, dtype=predt.dtype)

    return predt


def exp_fn_df(predt: torch.tensor) -> torch.tensor:
    """
    Exponential function used for Student-T distribution.

    Arguments
    ---------
    predt: torch.tensor
        Predicted values.

    Returns
    -------
    predt: torch.tensor
        Predicted values.
    """
    predt = torch.exp(nan_to_num(predt)) + torch.tensor(1e-06, dtype=predt.dtype)

    return predt + torch.tensor(2.0, dtype=predt.dtype)


def softplus_fn(predt: torch.tensor) -> torch.tensor:
    """
    Softplus function used to ensure predt is strictly positive.

    Arguments
    ---------
    predt: torch.tensor
        Predicted values.

    Returns
    -------
    predt: torch.tensor
        Predicted values.
    """
    predt = softplus(nan_to_num(predt)) + torch.tensor(1e-06, dtype=predt.dtype)

    return predt


def softplus_fn_df(predt: torch.tensor) -> torch.tensor:
    """
    Softplus function used for Student-T distribution.

    Arguments
    ---------
    predt: torch.tensor
        Predicted values.

    Returns
    -------
    predt: torch.tensor
        Predicted values.
    """
    predt = softplus(nan_to_num(predt)) + torch.tensor(1e-06, dtype=predt.dtype)

    return predt + torch.tensor(2.0, dtype=predt.dtype)


def sigmoid_fn(predt: torch.tensor) -> torch.tensor:
    """
    Function used to ensure predt are scaled to (0,1).

    Arguments
    ---------
    predt: torch.tensor
        Predicted values.

    Returns
    -------
    predt: torch.tensor
        Predicted values.
    """
    predt = torch.sigmoid(nan_to_num(predt)) + torch.tensor(1e-06, dtype=predt.dtype)
    predt = torch.clamp(predt, 1e-03, 1-1e-03)

    return predt


def relu_fn(predt: torch.tensor) -> torch.tensor:
    """
    Function used to ensure predt are scaled to max(0, predt).

    Arguments
    ---------
    predt: torch.tensor
        Predicted values.

    Returns
    -------
    predt: torch.tensor
        Predicted values.
    """
    predt = torch.relu(nan_to_num(predt)) + torch.tensor(1e-06, dtype=predt.dtype)

    return predt


def softmax_fn(predt: torch.tensor) -> torch.tensor:
    """
    Softmax function used to ensure predt is adding to one.


    Arguments
    ---------
    predt: torch.tensor
        Predicted values.

    Returns
    -------
    predt: torch.tensor
        Predicted values.
    """
    predt = softmax(nan_to_num(predt), dim=1) + torch.tensor(0, dtype=predt.dtype)

    return predt


def gumbel_softmax_fn(predt: torch.tensor,
                      tau: float = 1.0
                      ) -> torch.tensor:
    """
    Gumbel-softmax function used to ensure predt is adding to one.

    The Gumbel-softmax distribution is a continuous distribution over the simplex, which can be thought of as a "soft"
    version of a categorical distribution. It’s a way to draw samples from a categorical distribution in a
    differentiable way. The motivation behind using the Gumbel-Softmax is to make the discrete sampling process of
    categorical variables differentiable, which is useful in gradient-based optimization problems. To sample from a
    Gumbel-Softmax distribution, one would use the Gumbel-max trick: add a Gumbel noise to logits and apply the softmax.
    Formally, given a vector z, the Gumbel-softmax function s(z,tau)_i for a component i at temperature tau is
    defined as:

        s(z,tau)_i = frac{e^{(z_i + g_i) / tau}}{sum_{j=1}^M e^{(z_j + g_j) / tau}}

    where g_i is a sample from the Gumbel(0, 1) distribution. The parameter tau (temperature) controls the sharpness
    of the output distribution. As tau approaches 0, the mixing probabilities become more discrete, and as tau
    approaches infty, the mixing probabilities become more uniform. For more information we refer to

        Jang, E., Gu, Shixiang and Poole, B. "Categorical Reparameterization with Gumbel-Softmax", ICLR, 2017.

    Arguments
    ---------
    predt: torch.tensor
        Predicted values.
    tau: float, non-negative scalar temperature.
        Temperature parameter for the Gumbel-softmax distribution. As tau -> 0, the output becomes more discrete, and as
        tau -> inf, the output becomes more uniform.

    Returns
    -------
    predt: torch.tensor
        Predicted values.
    """
    torch.manual_seed(123)
    predt = gumbel_softmax(nan_to_num(predt), tau=tau, dim=1) + torch.tensor(0, dtype=predt.dtype)


    return predt


def reshape_scale_tril(predt: torch.tensor, response_fn: Callable = exp_fn) -> torch.tensor:
    """
    Reshape the scale_tril parameter to be a valid lower triangular matrix.

    Arguments
    ---------
    predt: torch.tensor
        Predicted values.
    response_fn: Callable
        Response function used to ensure predt is strictly positive.

    Returns
    -------
    predt: torch.tensor
        Predicted values.
    """
    n, m = predt.shape

    # Calculate the number of dimensions
    n_dim = int((-1 + np.sqrt(1 + 8 * m)) / 2)
    
    # Initialize the final tensor
    final_tensor = torch.zeros(n, n_dim, n_dim, dtype=predt.dtype)

    # Create indices for the lower triangle including diagonal
    tril_indices = torch.tril_indices(row=n_dim, col=n_dim, offset=0)

    # Use advanced indexing to fill the lower triangle and diagonal in one operation
    final_tensor[:, tril_indices[0], tril_indices[1]] = predt

    # Create diagonal indices
    diag_indices = torch.arange(n_dim)

    # Use these indices to access or modify the diagonals of all five 3x3 matrices
    final_tensor[:, diag_indices, diag_indices] = response_fn(final_tensor[:, diag_indices, diag_indices])

    return final_tensor


def create_mv_dataset(data: Union[pd.DataFrame, np.ndarray],
                      labels: Union[pd.DataFrame, np.ndarray]
                      ) -> lgb.Dataset:
    """
    Creates a lightgbm Dataset object for multivariate data.

    Arguments
    ---------
    data: pandas.DataFrame or numpy.ndarray
        Input dataset with shape (N, M)
    labels: pandas.DataFrame or numpy.ndarray
        Test labels with shape (N, D)

    Returns
    -------
    dataset: lightgbm.Dataset
        LightGBM Dataset object
    """

    # Convert to numpy arrays if they're pandas DataFrames
    column_names = None
    if isinstance(data, pd.DataFrame):
        column_names = data.columns.tolist()
        data = data.values

    if isinstance(labels, pd.DataFrame):
        labels = labels.values

    # Ravel the labels
    labels_raveled = labels.ravel(order="F")

    # Get the number of label columns
    D = labels.shape[1]

    # Repeat the data D times
    data_repeated = np.tile(data, (D, 1))

    # Create and return the lightgbm Dataset
    data_params = {"data": data_repeated, "label": labels_raveled}
    if column_names:
        data_params["feature_name"] = column_names
    return lgb.Dataset(**data_params)

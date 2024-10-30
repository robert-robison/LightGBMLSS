import torch
from torch.autograd import grad as autograd
from torch.optim import LBFGS
from torch.optim.lr_scheduler import ReduceLROnPlateau

import lightgbm as lgb
import numpy as np
import pandas as pd
from tqdm import tqdm

from typing import Any, Dict, Optional, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import warnings


class DistributionClass:
    """
    Generic class that contains general functions for univariate distributions.

    Arguments
    ---------
    distribution: torch.distributions.Distribution
        PyTorch Distribution class.
    univariate: bool
        Whether the distribution is univariate or multivariate.
    discrete: bool
        Whether the support of the distribution is discrete or continuous.
    n_dist_param: int
        Number of distributional parameters.
    stabilization: str
        Stabilization method.
    param_dict: Dict[str, Any]
        Dictionary that maps distributional parameters to their response scale.
    distribution_arg_names: List
        List of distributional parameter names.
    loss_fn: str
        Loss function. Options are "nll" (negative log-likelihood) or "crps" (continuous ranked probability score).
        Note that if "crps" is used, the Hessian is set to 1, as the current CRPS version is not twice differentiable.
        Hence, using the CRPS disregards any variation in the curvature of the loss function.
    tau: List
        List of expectiles. Only used for Expectile distributon.
    penalize_crossing: bool
        Whether to include a penalty term to discourage crossing of expectiles. Only used for Expectile distribution.
    param_dims: Dict[str, int]
        Dictionary that maps distributional parameters to their dimensions.
    n_dim: int
        Number of dimensions of the distribution.
    """
    def __init__(self,
                 distribution: torch.distributions.Distribution = None,
                 univariate: bool = True,
                 discrete: bool = False,
                 n_dist_param: int = None,
                 stabilization: str = "None",
                 param_dict: Dict[str, Any] = None,
                 distribution_arg_names: List = None,
                 loss_fn: str = "nll",
                 tau: Optional[List[torch.Tensor]] = None,
                 penalize_crossing: bool = False,
                 param_dims: Dict[str, int] = None,
                 n_dim: int = 1,
                 ):

        self.distribution = distribution
        self.univariate = univariate
        self.discrete = discrete
        self.n_dist_param = n_dist_param
        self.stabilization = stabilization
        self.param_dict = param_dict
        self.distribution_arg_names = distribution_arg_names
        self.loss_fn = loss_fn
        self.tau = tau
        self.penalize_crossing = penalize_crossing
        self.param_dims = param_dims
        self.n_dim = n_dim

    def objective_fn(self, predt: np.ndarray, data: lgb.Dataset) -> Tuple[np.ndarray, np.ndarray]:

        """
        Function to estimate gradients and hessians of distributional parameters.

        Arguments
        ---------
        predt: np.ndarray
            Predicted values.
        data: lgb.DMatrix
            Data used for training.

        Returns
        -------
        grad: np.ndarray
            Gradient.
        hess: np.ndarray
            Hessian.
        """
        # Target
        target = torch.tensor(data.get_label().reshape(-1, 1))

        # Weights
        if data.weight is None:
            # Use 1 as weight if no weights are specified
            weights = torch.ones_like(target, dtype=target.dtype).numpy()
        else:
            weights = data.get_weight().reshape(-1, 1)

        # Reduce weights length if not univariate
        if not self.univariate:
            weights = weights[:weights.shape[0] // self.n_dim]

        # Start values (needed to replace NaNs in predt)
        start_values = data.get_init_score().reshape(-1, self.n_dist_param, order="F")[0, :].tolist()

        # Calculate gradients and hessians
        predt, loss = self.get_params_loss(predt, target, start_values, requires_grad=True)
        grad, hess = self.compute_gradients_and_hessians(loss, predt, weights)

        # Duplicate gradients and hessians if not univariate
        if not self.univariate:
            nobs = target.shape[0] // self.n_dim

            grad_ = np.zeros(nobs * self.n_dim * self.n_dist_param)
            for i in range(self.n_dist_param):
                grad_[i * nobs * self.n_dim : (i + 1) * nobs * self.n_dim] = np.tile(
                    grad[i * nobs : (i + 1) * nobs], self.n_dim
                )
            grad = grad_

            hess_ = np.zeros(nobs * self.n_dim * self.n_dist_param)
            for i in range(self.n_dist_param):
                hess_[i * nobs * self.n_dim : (i + 1) * nobs * self.n_dim] = np.tile(
                    hess[i * nobs : (i + 1) * nobs], self.n_dim
                )
            hess = hess_

        return grad, hess

    def metric_fn(self, predt: np.ndarray, data: lgb.Dataset) -> Tuple[str, float, bool]:
        """
        Function that evaluates the predictions using the negative log-likelihood.

        Arguments
        ---------
        predt: np.ndarray
            Predicted values.
        data: lgb.Dataset
            Data used for training.

        Returns
        -------
        name: str
            Name of the evaluation metric.
        nll: float
            Negative log-likelihood.
        is_higher_better: bool
            Whether a higher value of the metric is better or not.
        """
        # Target
        target = torch.tensor(data.get_label().reshape(-1, 1))
        n_obs = target.shape[0]

        # Start values (needed to replace NaNs in predt)
        start_values = data.get_init_score().reshape(-1, self.n_dist_param)[0, :].tolist()

        # Calculate loss
        is_higher_better = False
        _, loss = self.get_params_loss(predt, target, start_values, requires_grad=False)

        return self.loss_fn, loss / n_obs, is_higher_better

    def loss_fn_start_values(self,
                             params: torch.Tensor,
                             target: torch.Tensor) -> torch.Tensor:
        """
        Function that calculates the loss for a given set of distributional parameters. Only used for calculating
        the loss for the start values.

        Parameter
        ---------
        params: torch.Tensor
            Distributional parameters.
        target: torch.Tensor
            Target values.

        Returns
        -------
        loss: torch.Tensor
            Loss value.
        """
        # Transform parameters to response scale
        if self.univariate:
            params = [
                response_fn(params[i].reshape(-1, 1)) for i, response_fn in enumerate(self.param_dict.values())
            ]
        else:
            # Ensure params is a 2-D tensor
            if isinstance(params, list):
                params = torch.stack(params)
            if len(params.shape) == 1:
                params = params.reshape(1, -1)
            
            # Extract distribution arguments, which can be multiple parameters
            param_list = []
            idx = 0
            for param in self.param_dict:
                n_dim = self.param_dims[param]
                param_list.append(self.param_dict[param](params[:, idx:idx + n_dim]))
                idx += n_dim
            params = param_list

        if self.univariate:
            # Replace NaNs and infinity values with 0.5
            nan_inf_idx = torch.isnan(torch.stack(params)) | torch.isinf(torch.stack(params))
            params = torch.where(nan_inf_idx, torch.tensor(0.5), torch.stack(params))
        else:
            # Replace NaNs and infinity values with 0.5 for each parameter
            for i in range(len(params)):
                nan_inf_idx = torch.isnan(params[i]) | torch.isinf(params[i])
                params[i] = torch.where(nan_inf_idx, torch.tensor(0.5, device=params[i].device, dtype=params[i].dtype), params[i])

        # Specify Distribution and Loss
        if self.tau is None:
            dist_kwargs = dict(zip(self.distribution_arg_names, params))
            dist = self.distribution(**dist_kwargs)
            loss = -torch.nansum(dist.log_prob(target))
        else:
            dist = self.distribution(params, self.penalize_crossing)
            loss = -torch.nansum(dist.log_prob(target, self.tau))

        return loss

    def calculate_start_values(self,
                               target: np.ndarray,
                               max_iter: int = 50
                               ) -> Tuple[float, np.ndarray]:
        """
        Function that calculates the starting values for each distributional parameter.

        Arguments
        ---------
        target: np.ndarray
            Data from which starting values are calculated.
        max_iter: int
            Maximum number of iterations.

        Returns
        -------
        loss: float
            Loss value.
        start_values: np.ndarray
            Starting values for each distributional parameter.
        """
        # Convert target to torch.tensor
        target = torch.tensor(target.reshape(-1, self.n_dim, order="F"))

        # Initialize parameters
        params = [torch.tensor(0.5, requires_grad=True) for _ in range(self.n_dist_param)]

        # Specify optimizer
        optimizer = LBFGS(params, lr=0.1, max_iter=np.min([int(max_iter / 4), 20]), line_search_fn="strong_wolfe")

        # Define learning rate scheduler
        lr_scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)

        # Define closure
        def closure():
            optimizer.zero_grad()
            loss = self.loss_fn_start_values(params, target)
            loss.backward()
            return loss

        # Optimize parameters
        loss_vals = []
        for epoch in range(max_iter):
            loss = optimizer.step(closure)
            lr_scheduler.step(loss)
            loss_vals.append(loss.item())

        # Get final loss
        loss = np.array(loss_vals[-1])

        # Get start values
        start_values = np.array([params[i].detach() for i in range(self.n_dist_param)])

        # Replace any remaining NaNs or infinity values with 0.5
        start_values = np.nan_to_num(start_values, nan=0.5, posinf=0.5, neginf=0.5)

        return loss, start_values

    def get_params_loss(self,
                        predt: np.ndarray,
                        target: torch.Tensor,
                        start_values: List[float],
                        requires_grad: bool = False,
                        ) -> Tuple[List[torch.Tensor], np.ndarray]:
        """
        Function that returns the predicted parameters and the loss.

        Arguments
        ---------
        predt: np.ndarray
            Predicted values.
        target: torch.Tensor
            Target values.
        start_values: List
            Starting values for each distributional parameter.
        requires_grad: bool
            Whether to add to the computational graph or not.

        Returns
        -------
        predt: torch.Tensor
            Predicted parameters.
        loss: torch.Tensor
            Loss value.
        """
        # Predicted Parameters
        predt = predt.reshape(-1, self.n_dist_param, order="F")

        # Replace NaNs and infinity values with unconditional start values
        nan_inf_mask = np.isnan(predt) | np.isinf(predt)
        predt[nan_inf_mask] = np.take(start_values, np.where(nan_inf_mask)[1])

        # Convert to torch.tensor
        predt_torch = [
            torch.tensor(predt[:, i].reshape(-1, 1), requires_grad=requires_grad) for i in range(self.n_dist_param)
        ]
        if self.univariate:
            # Predicted Parameters transformed to response scale
            predt_transformed = [
                response_fn(predt_torch[i]) for i, response_fn in enumerate(self.param_dict.values())
            ]
        else:
            # Reduce predt_torch to size of target
            predt_torch = [
                predt_torch[i][:predt.shape[0] // self.n_dim] for i in range(self.n_dist_param)
            ]

            # Convert to torch.tensors
            predt_list = []
            idx = 0
            for n_dim in self.param_dims.values():
                curr_param = torch.cat([predt_torch[i] for i in range(idx, idx + n_dim)], axis=1)
                predt_list.append(curr_param)
                # predt_list.append(torch.tensor(predt[:, idx:idx + n_dim], requires_grad=requires_grad))
                idx += n_dim

            # Reshape target, imposing order="F"
            target = target.reshape(self.n_dim, -1).T

            # Predicted Parameters transformed to response scale
            predt_transformed = [
                response_fn(predt_list[i]) for i, response_fn in enumerate(self.param_dict.values())
            ]

        # Specify Distribution and Loss
        if self.tau is None:
            dist_kwargs = dict(zip(self.distribution_arg_names, predt_transformed))
            dist_fit = self.distribution(**dist_kwargs)
            if self.loss_fn == "nll":
                loss = -torch.nansum(dist_fit.log_prob(target))
            elif self.loss_fn == "crps":
                torch.manual_seed(123)
                dist_samples = dist_fit.rsample((30,)).squeeze(-1)
                loss = torch.nansum(self.crps_score(target, dist_samples))
            else:
                raise ValueError("Invalid loss function. Please select 'nll' or 'crps'.")
        else:
            dist_fit = self.distribution(predt_transformed, self.penalize_crossing)
            loss = -torch.nansum(dist_fit.log_prob(target, self.tau))

        return predt_torch, loss

    def draw_samples(self,
                     predt_params: List,
                     n_samples: int = 1000,
                     seed: int = 123
                     ) -> pd.DataFrame:
        """
        Function that draws n_samples from a predicted distribution.

        Arguments
        ---------
        predt_params: List
            List with predicted distributional parameters.
        n_samples: int
            Number of sample to draw from predicted response distribution.
        seed: int
            Manual seed.

        Returns
        -------
        pred_dist: pd.DataFrame
            DataFrame with n_samples drawn from predicted response distribution.

        """
        torch.manual_seed(seed)

        if self.tau is None:
            dist_kwargs = {arg_name: param for arg_name, param in zip(self.distribution_arg_names, predt_params)}
            dist_pred = self.distribution(**dist_kwargs)
            dist_samples = dist_pred.sample((n_samples,)).squeeze().detach().numpy().T
            if self.univariate:
                dist_samples = pd.DataFrame(dist_samples)
                dist_samples.columns = [str("y_sample") + str(i) for i in range(dist_samples.shape[1])]
            else:
                dist_samples = dist_samples.reshape(-1, n_samples)
                dist_samples = pd.DataFrame(dist_samples)
                dist_samples.columns = [str("y_sample") + str(i) for i in range(dist_samples.shape[1])]
                dist_samples["dim_num"] = np.repeat(np.arange(self.n_dim), len(dist_samples) // self.n_dim)
        else:
            dist_samples = None

        if self.discrete:
            dist_samples = dist_samples.astype(int)

        return dist_samples

    def predict_dist(self,
                     booster: lgb.Booster,
                     data: pd.DataFrame,
                     start_values: np.ndarray,
                     pred_type: str = "parameters",
                     n_samples: int = 1000,
                     quantiles: list = [0.1, 0.5, 0.9],
                     seed: str = 123
                     ) -> pd.DataFrame:
        """
        Function that predicts from the trained model.

        Arguments
        ---------
        booster : lgb.Booster
            Trained model.
        data : pd.DataFrame
            Data to predict from.
        start_values : np.ndarray.
            Starting values for each distributional parameter.
        pred_type : str
            Type of prediction:
            - "samples" draws n_samples from the predicted distribution.
            - "quantiles" calculates the quantiles from the predicted distribution.
            - "parameters" returns the predicted distributional parameters.
            - "expectiles" returns the predicted expectiles.
        n_samples : int
            Number of samples to draw from the predicted distribution.
        quantiles : List[float]
            List of quantiles to calculate from the predicted distribution.
        seed : int
            Seed for random number generator used to draw samples from the predicted distribution.

        Returns
        -------
        pred : pd.DataFrame
            Predictions.
        """

        predt = torch.tensor(
            booster.predict(data, raw_score=True),
            dtype=torch.float32
        ).reshape(-1, self.n_dist_param)

        # Set init_score as starting point for each distributional parameter.
        init_score_pred = torch.tensor(
            np.ones(shape=(data.shape[0], 1))*start_values,
            dtype=torch.float32
        )

        # The predictions don't include the init_score specified in creating the train data.
        # Hence, it needs to be added manually with the corresponding transform for each distributional parameter.
        if self.univariate:
            dist_params_list = [
                response_fun(
                    predt[:, i].reshape(-1, 1) + init_score_pred[:, i].reshape(-1, 1))
                for i, (dist_param, response_fun) in enumerate(self.param_dict.items())
            ]
            dist_params_predt = np.concatenate([arr.numpy() for arr in dist_params_list], axis=1)
            dist_params_predt = pd.DataFrame(dist_params_predt)
            dist_params_predt.columns = self.param_dict.keys()
        else:
            # Extract distribution arguments, which can be multiple parameters
            dist_params_list = []
            idx = 0
            for param in self.param_dict:
                n_dim = self.param_dims[param]
                response_fn = self.param_dict[param]
                curr_predt = predt[:, idx:idx + n_dim] + init_score_pred[:, idx:idx + n_dim]
                dist_params_list.append(response_fn(curr_predt))
                idx += n_dim
            dist_params_predt = np.concatenate([arr.reshape(arr.shape[0], -1) for arr in dist_params_list], axis=1)
            dist_params_predt = pd.DataFrame(dist_params_predt)
            columns = []
            for i, param in enumerate(self.param_dict):
                columns.extend([f"{param}_{j}" for j in range(np.prod(dist_params_list[i].shape[1:]))])
            dist_params_predt.columns = columns

        # Draw samples from predicted response distribution
        pred_samples_df = self.draw_samples(predt_params=dist_params_list,
                                            n_samples=n_samples,
                                            seed=seed)

        if pred_type == "parameters":
            return dist_params_predt

        elif pred_type == "expectiles":
            return dist_params_predt

        elif pred_type == "samples":
            return pred_samples_df

        elif pred_type == "quantiles":
            # Calculate quantiles from predicted response distribution
            sample_cols = [col for col in pred_samples_df if col.startswith("y_sample")]
            other_cols = [col for col in pred_samples_df if col not in sample_cols]
            pred_quant_df = pred_samples_df[sample_cols].quantile(quantiles, axis=1).T
            pred_quant_df.columns = [str("quant_") + str(quantiles[i]) for i in range(len(quantiles))]
            if self.discrete:
                pred_quant_df = pred_quant_df.astype(int)
            pred_quant_df = pd.concat([pred_samples_df[other_cols], pred_quant_df], axis=1)
            return pred_quant_df

    def compute_gradients_and_hessians(self,
                                       loss: torch.tensor,
                                       predt: torch.tensor,
                                       weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

        """
        Calculates gradients and hessians.

        Output gradients and hessians have shape (n_samples*n_outputs, 1).

        Arguments:
        ---------
        loss: torch.Tensor
            Loss.
        predt: torch.Tensor
            List of predicted parameters.
        weights: np.ndarray
            Weights.

        Returns:
        -------
        grad: torch.Tensor
            Gradients.
        hess: torch.Tensor
            Hessians.
        """
        if self.loss_fn == "nll":
            # Gradient and Hessian
            grad = autograd(loss, inputs=predt, create_graph=True)
            hess = [autograd(grad[i].nansum(), inputs=predt[i], retain_graph=True)[0] for i in range(len(grad))]
        elif self.loss_fn == "crps":
            # Gradient and Hessian
            grad = autograd(loss, inputs=predt, create_graph=True)
            hess = [torch.ones_like(grad[i]) for i in range(len(grad))]

            # # Approximation of Hessian
            # step_size = 1e-6
            # predt_upper = [
            #     response_fn(predt[i] + step_size).reshape(-1, 1) for i, response_fn in
            #     enumerate(self.param_dict.values())
            # ]
            # dist_kwargs_upper = dict(zip(self.distribution_arg_names, predt_upper))
            # dist_fit_upper = self.distribution(**dist_kwargs_upper)
            # dist_samples_upper = dist_fit_upper.rsample((30,)).squeeze(-1)
            # loss_upper = torch.nansum(self.crps_score(self.target, dist_samples_upper))
            #
            # predt_lower = [
            #     response_fn(predt[i] - step_size).reshape(-1, 1) for i, response_fn in
            #     enumerate(self.param_dict.values())
            # ]
            # dist_kwargs_lower = dict(zip(self.distribution_arg_names, predt_lower))
            # dist_fit_lower = self.distribution(**dist_kwargs_lower)
            # dist_samples_lower = dist_fit_lower.rsample((30,)).squeeze(-1)
            # loss_lower = torch.nansum(self.crps_score(self.target, dist_samples_lower))
            #
            # grad_upper = autograd(loss_upper, inputs=predt_upper)
            # grad_lower = autograd(loss_lower, inputs=predt_lower)
            # hess = [(grad_upper[i] - grad_lower[i]) / (2 * step_size) for i in range(len(grad))]

        # Stabilization of Derivatives
        if self.stabilization != "None":
            grad = [self.stabilize_derivative(grad[i], type=self.stabilization) for i in range(len(grad))]
            hess = [self.stabilize_derivative(hess[i], type=self.stabilization) for i in range(len(hess))]

        # Reshape
        grad = torch.cat(grad, axis=1).detach().numpy()
        hess = torch.cat(hess, axis=1).detach().numpy()

        # Weighting
        grad *= weights
        hess *= weights

        # Reshape
        grad = grad.ravel(order="F")
        hess = hess.ravel(order="F")

        return grad, hess

    def stabilize_derivative(self, input_der: torch.Tensor, type: str = "MAD") -> torch.Tensor:
        """
        Function that stabilizes Gradients and Hessians.

        As LightGBMLSS updates the parameter estimates by optimizing Gradients and Hessians, it is important
        that these are comparable in magnitude for all distributional parameters. Due to imbalances regarding the ranges,
        the estimation might become unstable so that it does not converge (or converge very slowly) to the optimal solution.
        Another way to improve convergence might be to standardize the response variable. This is especially useful if the
        range of the response differs strongly from the range of the Gradients and Hessians. Both, the stabilization and
        the standardization of the response are not always advised but need to be carefully considered.
        Source: https://github.com/boost-R/gamboostLSS/blob/7792951d2984f289ed7e530befa42a2a4cb04d1d/R/helpers.R#L173

        Parameters
        ----------
        input_der : torch.Tensor
            Input derivative, either Gradient or Hessian.
        type: str
            Stabilization method. Can be either "None", "MAD" or "L2".

        Returns
        -------
        stab_der : torch.Tensor
            Stabilized Gradient or Hessian.
        """

        if type == "MAD":
            input_der = torch.nan_to_num(input_der, nan=float(torch.nanmean(input_der)))
            div = torch.nanmedian(torch.abs(input_der - torch.nanmedian(input_der)))
            div = torch.where(div < torch.tensor(1e-04), torch.tensor(1e-04), div)
            stab_der = input_der / div

        if type == "L2":
            input_der = torch.nan_to_num(input_der, nan=float(torch.nanmean(input_der)))
            div = torch.sqrt(torch.nanmean(input_der.pow(2)))
            div = torch.where(div < torch.tensor(1e-04), torch.tensor(1e-04), div)
            div = torch.where(div > torch.tensor(10000.0), torch.tensor(10000.0), div)
            stab_der = input_der / div

        if type == "None":
            stab_der = torch.nan_to_num(input_der, nan=float(torch.nanmean(input_der)))

        return stab_der

    def crps_score(self, y: torch.tensor, yhat_dist: torch.tensor) -> torch.tensor:
        """
        Function that calculates the Continuous Ranked Probability Score (CRPS) for a given set of predicted samples.

        Parameters
        ----------
        y: torch.Tensor
            Response variable of shape (n_observations,1).
        yhat_dist: torch.Tensor
            Predicted samples of shape (n_samples, n_observations).

        Returns
        -------
        crps: torch.Tensor
            CRPS score.

        References
        ----------
        Gneiting, Tilmann & Raftery, Adrian. (2007). Strictly Proper Scoring Rules, Prediction, and Estimation.
        Journal of the American Statistical Association. 102. 359-378.

        Source
        ------
        https://github.com/elephaint/pgbm/blob/main/pgbm/torch/pgbm_dist.py#L549
        """
        # Get the number of observations
        n_samples = yhat_dist.shape[0]

        # Sort the forecasts in ascending order
        yhat_dist_sorted, _ = torch.sort(yhat_dist, 0)

        # Create temporary tensors
        y_cdf = torch.zeros_like(y)
        yhat_cdf = torch.zeros_like(y)
        yhat_prev = torch.zeros_like(y)
        crps = torch.zeros_like(y)

        # Loop over the predicted samples generated per observation
        for yhat in yhat_dist_sorted:
            yhat = yhat.reshape(-1, 1)
            flag = (y_cdf == 0) * (y < yhat)
            crps += flag * ((y - yhat_prev) * yhat_cdf ** 2)
            crps += flag * ((yhat - y) * (yhat_cdf - 1) ** 2)
            crps += (~flag) * ((yhat - yhat_prev) * (yhat_cdf - y_cdf) ** 2)
            y_cdf += flag
            yhat_cdf += 1 / n_samples
            yhat_prev = yhat

        # In case y_cdf == 0 after the loop
        flag = (y_cdf == 0)
        crps += flag * (y - yhat)

        return crps

    def dist_select(self,
                    target: np.ndarray,
                    candidate_distributions: List,
                    max_iter: int = 100,
                    plot: bool = False,
                    figure_size: tuple = (10, 5),
                    ) -> pd.DataFrame:
        """
        Function that selects the most suitable distribution among the candidate_distributions for the target variable,
        based on the NegLogLikelihood (lower is better).

        Parameters
        ----------
        target: np.ndarray
            Response variable.
        candidate_distributions: List
            List of candidate distributions.
        max_iter: int
            Maximum number of iterations for the optimization.
        plot: bool
            If True, a density plot of the actual and fitted distribution is created.
        figure_size: tuple
            Figure size of the density plot.

        Returns
        -------
        fit_df: pd.DataFrame
            Dataframe with the loss values of the fitted candidate distributions.
        """
        dist_list = []
        total_iterations = len(candidate_distributions)
        with tqdm(total=total_iterations, desc="Fitting candidate distributions") as pbar:
            for i in range(len(candidate_distributions)):
                dist_name = candidate_distributions[i].__name__.split(".")[2]
                pbar.set_description(f"Fitting {dist_name} distribution")
                dist_sel = getattr(candidate_distributions[i], dist_name)()
                try:
                    loss, params = dist_sel.calculate_start_values(target=target.reshape(-1, 1), max_iter=max_iter)
                    fit_df = pd.DataFrame.from_dict(
                        {self.loss_fn: loss.reshape(-1,),
                         "distribution": str(dist_name),
                         "params": [params]
                         }
                    )
                except Exception as e:
                    warnings.warn(f"Error fitting {dist_name} distribution: {str(e)}")
                    fit_df = pd.DataFrame(
                        {self.loss_fn: np.nan,
                         "distribution": str(dist_name),
                         "params": [np.nan] * self.n_dist_param
                         }
                    )
                dist_list.append(fit_df)
                pbar.update(1)
            pbar.set_description("Fitting of candidate distributions completed")
            fit_df = pd.concat(dist_list).sort_values(by=self.loss_fn, ascending=True)
            fit_df["rank"] = fit_df[self.loss_fn].rank().astype(int)
            fit_df.set_index(fit_df["rank"], inplace=True)

        if plot:
            # Select best distribution
            best_dist = fit_df[fit_df["rank"] == 1].reset_index(drop=True)
            for dist in candidate_distributions:
                if dist.__name__.split(".")[2] == best_dist["distribution"].values[0]:
                    best_dist_sel = dist
                    break
            best_dist_sel = getattr(best_dist_sel, best_dist["distribution"].values[0])()
            params = torch.tensor(best_dist["params"][0]).reshape(-1, best_dist_sel.n_dist_param)

            # Transform parameters to the response scale and draw samples
            fitted_params = np.concatenate(
                [
                    response_fun(params[:, i].reshape(-1, 1)).numpy()
                    for i, (dist_param, response_fun) in enumerate(best_dist_sel.param_dict.items())
                ],
                axis=1,
            )
            fitted_params = pd.DataFrame(fitted_params, columns=best_dist_sel.param_dict.keys())
            n_samples = np.max([10000, target.shape[0]])
            n_samples = np.where(n_samples > 500000, 100000, n_samples)
            dist_samples = best_dist_sel.draw_samples(fitted_params,
                                                      n_samples=n_samples,
                                                      seed=123).values

            # Plot actual and fitted distribution
            plt.figure(figsize=figure_size)
            sns.kdeplot(target.reshape(-1, ), label="Actual")
            sns.kdeplot(dist_samples.reshape(-1, ), label=f"Best-Fit: {best_dist['distribution'].values[0]}")
            plt.legend()
            plt.title("Actual vs. Best-Fit Density", fontweight="bold", fontsize=16)
            plt.show()

        fit_df.drop(columns=["rank", "params"], inplace=True)

        return fit_df

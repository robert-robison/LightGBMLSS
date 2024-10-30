import numpy as np
from lightgbmlss.model import LightGBMLSS
from lightgbmlss.utils import create_mv_dataset
from lightgbmlss.distributions.Gaussian import MultivariateGaussian
from udeorbsim.utils import DATA_PATH
import pandas as pd


import torch
from torch.distributions.multivariate_normal import MultivariateNormal



df = pd.read_csv(
    DATA_PATH / "results/residuals_data/cs2_2017-01-01_2017-01-10_120step_120int.csv",
    parse_dates=["pred_dt", "target_dt"],
)

# Calculate Euclidean error from residual columns
df["error"] = np.sqrt(df["resid_x"] ** 2 + df["resid_y"] ** 2 + df["resid_z"] ** 2)

# Create id column that increments for each pred_dt
df["id"] = (df["pred_dt"] != df["pred_dt"].shift()).cumsum()

# Create training and validation sets split on id
valid_frac = 0.2
m = df.horizon > 0

dtrain = create_mv_dataset(
    df.loc[m, ["horizon"]],
    df.loc[m, ["resid_x", "resid_y", "resid_z"]],
)
# Create lightgbm lss model and train
params = {
    "lambda_l1": 1.0143946488324415e-05,
    "lambda_l2": 31.285811917119002,
    "num_leaves": 247,
    "min_child_samples": 23,
    "eta": 0.04131105489168698,
    # "eta": 1e-10,

}
lgblss = LightGBMLSS(MultivariateGaussian(n_dim=3))
lgblss.train(params={**params, "num_iterations": 100}, train_set=dtrain)


horizons = df.horizon.unique().reshape(-1, 1)

preds = lgblss.predict(horizons)

dist = MultivariateNormal(
    loc=torch.tensor(preds.iloc[:, :3].to_numpy()),
    scale_tril=torch.tensor(preds.iloc[:, 3:].to_numpy().reshape(-1, 3, 3, order="C")),
)
loss = -torch.nansum(dist.log_prob(torch.tensor(df.loc[m, ["resid_x", "resid_y", "resid_z"]].to_numpy())))

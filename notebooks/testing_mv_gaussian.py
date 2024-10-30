import numpy as np
from lightgbmlss.model import LightGBMLSS
from lightgbmlss.utils import create_mv_dataset
from lightgbmlss.distributions.Gaussian import MultivariateGaussian

import torch
from torch.distributions.multivariate_normal import MultivariateNormal


# Generate input data
np.random.seed(42)
n_samples = 1000
x = np.linspace(0, 10, n_samples)


# Create correlated noise with increasing magnitude
def scale_cov_matrix(x, base_cov, scale_factor=0.1):
    return base_cov * (1 + scale_factor * x)


cov_matrix = np.array([[0.25, 0.15, 0.1], [0.15, 0.49, 0.2], [0.1, 0.2, 0.36]])

correlated_noise = np.array(
    [
        np.random.multivariate_normal(
            mean=[0, 0, 0], cov=scale_cov_matrix(xi, cov_matrix)
        )
        for xi in x
    ]
)

# Generate three dependent variables with smooth, non-monotonic relationships and correlated noise
y1_ = 2 * np.sin(x) + 0.5 * x
y1 = y1_ + correlated_noise[:, 0]
y2_ = 3 * np.cos(0.5 * x) + 0.3 * x**2
y2 = y2_ + correlated_noise[:, 1]
y3_ = 1.5 * np.sin(0.7 * x) * np.cos(0.3 * x) + 0.2 * x
y3 = y3_ + correlated_noise[:, 2]
# Combine the data
data = np.column_stack((x, y1, y2, y3))
real_data = np.column_stack((x, y1_, y2_, y3_))

# Create lightgbm dataset
dtrain = create_mv_dataset(
    data[:, 0].reshape(-1, 1),
    data[:, 1:],
)

# Create lightgbm lss model and train
lgblss = LightGBMLSS(MultivariateGaussian(n_dim=3, response_fn="exp"))
lgblss.train(params={"learning_rate": 0.1, "num_iterations": 100, "num_leaves": 4}, train_set=dtrain)

# Make predictions
preds = lgblss.predict(data[:, 0].reshape(-1, 1))

dist = MultivariateNormal(
    loc=torch.tensor(preds.iloc[:, :3].to_numpy()),
    scale_tril=torch.tensor(preds.iloc[:, 3:].to_numpy().reshape(-1, 3, 3, order="C")),
)
loss = -torch.nansum(dist.log_prob(torch.tensor(data[:, 1:])))

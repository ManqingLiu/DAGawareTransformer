import numpy as np
import pandas as pd
from scipy.stats import bernoulli, norm

# Set the seed for reproducibility
np.random.seed(253)

# Number of samples
N = 100

# Covariance matrix for U
Sigma = 0.5 * (0.3 * np.ones((5, 5)) + 0.7 * np.eye(5))

# Generate unobserved confounders U
U = np.random.multivariate_normal(np.zeros(5), Sigma, N)

# Generate covariate matrix X
X = np.zeros((N, 10))
for i in range(10):
    W_ui = np.random.normal(0, 0.1, 5)
    p_Xi = 1 / (1 + np.exp(-(U @ W_ui)))
    X[:, i] = bernoulli.rvs(p_Xi)

# Generate Wa
Wa = np.random.normal(0, 0.22, 15)  # 5 for U and 10 for X

# Generate treatment A
p_A = 1 / (1 + np.exp(-(np.hstack([U, X]) @ Wa)))
A = bernoulli.rvs(p_A)

# Generate outcome variable Y
Wu = np.random.normal(0, 0.05, 5)  # halved from 0.1 to 0.05
Wx = np.random.normal(2.5, 0.22, 10)  # halved from 5 to 2.5
Wa_y = np.random.normal(3, 0.11, 1)  # halved from 0.22 to 0.11
p_Y = 1 / (1 + np.exp(-(-15+U @ Wu + X @ Wx + A * Wa_y)))
p_Y1 = 1 / (1 + np.exp(-(-15+U @ Wu + X @ Wx + 1 * Wa_y)))
p_Y0 = 1 / (1 + np.exp(-(-15+U @ Wu + X @ Wx + 0 * Wa_y)))
ATE = p_Y1.mean() - p_Y0.mean()
print(f"ATE:{ATE}")
Y = bernoulli.rvs(p_Y)

# Create a DataFrame
df = pd.DataFrame(np.hstack([A[:, None], X, Y[:, None]]),
                  columns=['A'] + [f'X{i+1}' for i in range(10)] + ['Y'])

# print percentage of Y = 1
print(df['A'].mean())
print(df['Y'].mean())

# save the data to a CSV file in data/unmeasured_confounding/data_U.csv
df.to_csv('data/unmeasured_confounding/data_U_n100.csv', index=False)

print(df.columns)


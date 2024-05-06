import numpy as np
import pandas as pd
from scipy.stats import bernoulli, norm

# Set the seed for reproducibility
np.random.seed(253)

N = 100

# generate 1 u from uniform[0,1]
U = np.random.uniform(0, 1, N)

# summarise distribution of U
print(pd.Series(U).describe())

# generate 1 binary x which depends on u of sample size N

p_X = 1 / (1 + np.exp(-(U - 0.5)))
X = bernoulli.rvs(p_X)

# generate 1 binary a which depends on x and u
p_A = 1 / (1 + np.exp(-(X + U - 1)))
A = bernoulli.rvs(p_A)

# generate 1 binary y which depends on a, x and u
p_Y = 1 / (1 + np.exp(-(A + X + U - 1)))
p_Y1 = 1 / (1 + np.exp(-(1 + X + U - 1)))
p_Y0 = 1 / (1 + np.exp(-(0 + X + U - 1)))
ATE = p_Y1.mean() - p_Y0.mean()
print(f"ATE:{ATE}")
Y = bernoulli.rvs(p_Y)

# create data frame of size 100*3 including A, X, Y
df = pd.DataFrame(np.hstack([A[:, None], X[:, None], Y[:, None]]),
                  columns=['A', 'X', 'Y'])

# print percentage of A = 1
print(df['A'].mean())
# print percentage of Y = 1
print(df['Y'].mean())

df.to_csv('data/unmeasured_confounding/data_U_uniform_simple.csv', index=False)

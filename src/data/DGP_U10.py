import numpy as np
from scipy.special import expit
import pandas as pd

# Setting the random seed for reproducibility
np.random.seed(42)

# Number of samples
n = 2000

# obsesrved confounder L
L = np.random.binomial(1, 0.05, n)
# one umeasured confounder U - age
U = np.random.normal(44, 12.2, n)
# probability of treatment
A_prob = expit(-2.57 +  0.53*L + 0.3*U)
A = np.random.binomial(1, A_prob, n)

# Outcome
Y = -0.3 + 0.5*A + 0.1*L + 0.4*U

# Sample 10 U_hat following uniform[0,1]
# U_hat = np.random.uniform(0,1, n)

# List to store DataFrames
# dfs = []

# Base DataFrame
df = pd.DataFrame({
    'l': L,
    't': A,
    'u': U,
    'y': Y
})


# Now `dfs` contains 10 DataFrames with each having a different U_hat
# Example: To access the first DataFrame with U_hat_1, you can do
print(df.head())
print(df.describe(include='all'))

import os
subfolder = 'data/unmeasured_confounding'
# Specify the subfolder and file name
file_name = f'df.csv'
path = os.path.join(subfolder, file_name)

# Save the DataFrame
df.to_csv(path, index=False)



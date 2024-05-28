import numpy as np
import pandas as pd


# Set the random seed for reproducibility
np.random.seed(42)

N = 10000

# simulate U from a normal distribution
U = np.random.normal(0, 1, N)

# simulate binary A from a bernoulli, the probability of A=1 is sigmoid(U)
A = np.random.binomial(1, 1 / (1 + np.exp(-U)))

# simulate binary M, the probability depends on A
M = np.random.binomial(1, 1 / (1 + np.exp(-A)))

# simulate continuous Y depends on M and U
Y = 2 * M - 0.3 * U

# create a dataframe of A, M, Y
data = pd.DataFrame({'A': A, 'M': M, 'Y': Y})

print(data.head)

# save data to a csv file in data/raw/front-door
data.to_csv('data/raw/front-door/front_door_data.csv', index=False)


# get the true ATE
M_A1 = np.random.binomial(1, 1 / (1 + np.exp(-1)), N)
Y_A1 = 2*M_A1 - 0.3*U

M_A0 = np.random.binomial(1, 1 / (1 + np.exp(-0)), N)
Y_A0 = 2*M_A0 - 0.3*U

ATE_true = np.mean(Y_A1) - np.mean(Y_A0)
print("True ATE:", ATE_true)  # 0.467
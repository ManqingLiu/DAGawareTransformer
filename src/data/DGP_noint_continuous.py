## Summary
## generate data with N = 50,000 where
## A is 1/0, Y is continuous
## Ls:
## See below
### simulation based on the NHEFS data
import numpy as np
import pandas as pd
from scipy.stats import logistic
from itertools import product
from scipy.special import expit
from src.models.utils import find_appropriate_betas_for_quartiles


# Setting the random seed for reproducibility
np.random.seed(42)

# Number of samples
n = 50000

# Confounders (10 in total)
## Sex -> Bernoulli[0.51]
## Age -> Normal[44, 12.2] (range 25-74)
## Race -> Bernoulli[0.13]
## Asthma -> Bernoulli[0.05]
## Smoke Intensity -> Normal[20, 11.8] (range 1-80)
## education -> Normal[2.7, 1.19](range 1-5)
## smokeyrs -> Normal[25, 12.2] (range 1-64)
## active -> Normal[0.652, 0.65](range 0-2)
## exercise -> Normal[1.2, 0.74] (range 0-2)
## wt71 -> Normal[71.1, 15.7] (range 36-170)
sex = np.random.binomial(1, 0.51, n)
age = np.clip(np.random.normal(44, 12.2, n), 25, 74).astype(int)
race = np.random.binomial(1, 0.13, n)
asthma = np.random.binomial(1, 0.05, n)
smoke_intensity = np.clip(np.random.normal(20, 11.8, n), 1, 80).astype(int)
education =  np.clip(np.random.normal(2.7, 1.19, n), 1, 5).astype(int)
smokeyrs = np.clip(np.random.normal(25, 12.2, n), 1, 64).astype(int)
active =  np.clip(np.random.normal(0.652, 0.65, n), 0, 2).astype(int)
exercise =  np.clip(np.random.normal(1.2, 0.74, n), 0, 2).astype(int)
wt71 =  np.clip(np.random.normal(71.1, 15.7, n), 36, 170)

# Creating the DataFrame
df = pd.DataFrame({
    'intercept': 1,  # Adding an intercept column
    'sex': sex,
    'age': age,
    'race': race,
    'asthma': asthma,
    'smoke_intensity': smoke_intensity,
    'education': education,
    'smokeyrs': smokeyrs,
    'active': active,
    'exercise': exercise,
    'wt71': wt71
})


# betas for treatment
betas_A = {'intercept': -2.57, 'sex': -0.53, 'race': -0.77, 'age': 0.047,
           'education': 0.116, 'smoke_intensity': -0.027, 'smokeyrs': -0.028,
           'exercise': 0.183, 'active': 0.085, 'wt71': 0.00587}

# Calculate the linear combination of the variables with their coefficients
linear_combination = sum(betas_A[key] * df[key] for key in betas_A)

# Apply the logistic function to get probabilities
treatment_probability = expit(linear_combination)

# Assign treatment based on whether the random number is less than the propensity score
# df['treatment'] = np.random.binomial(1, propensity_scores)
df['treatment'] = np.random.binomial(1, p=treatment_probability)

# sanity check
quantiles = [0.25, 0.5, 0.75]  # You can modify this list to include the quantiles you're interested in
quantile_values = treatment_probability.quantile(quantiles)

print("Quantiles of Treatment Probability:")
print(quantile_values)

'''
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming df['treatment_probability'] contains the probabilities
# Set style
sns.set(style="whitegrid")

# Create a figure and a set of subplots
fig, ax = plt.subplots(figsize=(10, 6))

# Plot histograms
sns.histplot(data=df, x='treatment_probability', hue='treatment',
             element="step", stat="density", common_norm=False, ax=ax)

# Overlay KDE plots
sns.kdeplot(data=df, x='treatment_probability', hue='treatment',
            common_norm=False, fill=True, alpha=0.2, ax=ax)

# Add titles and labels
plt.title('Propensity Score Distribution by Treatment Group')
plt.xlabel('Propensity Score')
plt.ylabel('Density')

plt.savefig('reports/figures/propscore_hist_nointeraction.png')
'''

print(df.columns.tolist())
# ['sex', 'age', 'smoke_intensity', 'asthma', 'intercept', 'treatment', 'treatment_probability']

cols = df.columns.tolist()  # Get a list of all columns
cols = [cols[0]] + [cols[-1]] + cols[1:-1]  # Move the last column name to the second
df = df[cols]  # Reindex the DataFrame with the new column order

print(df.columns.tolist())


betas_Y = {'intercept': 16.76, 'treatment': 3.348, 'sex': -1.3173, 'race': 0.5616, 'age': -0.2068,
           'education': 0.0342, 'smoke_intensity': 0.0241, 'smokeyrs': 0.0512,
           'exercise': 0.2320, 'active': -0.5907, 'wt71': -0.1002}

df['outcome'] = sum(betas_Y[key] * df[key] for key in betas_Y)
# Calculating statistics
min_val = df['outcome'].min()
max_val = df['outcome'].max()
mean_val = df['outcome'].mean()
median_val = df['outcome'].median()
std_dev = df['outcome'].std()

print("Minimum:", min_val)
print("Maximum:", max_val)
print("Mean:", mean_val)
print("Median:", median_val)
print("Standard Deviation:", std_dev)

'''
betas_C = {'intercept': -22, 'treatment': 0.4, 'sex': 0.2, 'age': 0.3,
           'smoke_intensity': 0.3, 'asthma': 0.2}

# Calculate the linear combination of the variables with their coefficients
linear_combination_C = sum(betas_C[key] * df_sub[key] for key in betas_C)

# Apply the logistic function to get probabilities of censoring
prob_C = expit(linear_combination_C)

df['censor'] = np.random.binomial(1, prob_C)

# Check the proportion of '1's in the censored
prop_ones_C = np.mean(df['censor'])
print("Proportion of censored:", prop_ones_C)  # 0.1788

# update Y where Y is randomly generated for those who were censored
df['outcome'] = np.where(df['censor'] == 1, np.nan, df['outcome'])
'''

# generate hat variables
np.random.seed(50)
df['sex_hat'] = np.random.binomial(1, 0.51, n)
df['age_hat'] = np.clip(np.random.normal(44, 12.2, n), 25, 74).astype(int)
df['race_hat'] = np.random.binomial(1, 0.13, n)
df['asthma_hat'] = np.random.binomial(1, 0.05, n)
df['smoke_intensity_hat'] = np.clip(np.random.normal(20, 11.8, n), 1, 80).astype(int)
df['education_hat'] =  np.clip(np.random.normal(2.7, 1.19, n), 1, 5).astype(int)
df['smokeyrs_hat'] = np.clip(np.random.normal(25, 12.2, n), 1, 64).astype(int)
df['active_hat'] =  np.clip(np.random.normal(0.652, 0.65, n), 0, 2).astype(int)
df['exercise_hat'] =  np.clip(np.random.normal(1.2, 0.74, n), 0, 2).astype(int)
df['wt71_hat'] =  np.clip(np.random.normal(71.1, 15.7, n), 36, 170)
df['treatment_hat'] = np.random.binomial(1, 0.5, n)
df['outcome_hat']  = np.random.normal(3, 3, n)

# Number of bins for discretization
num_bins = 20

continuous_features = ['age', 'age_hat', 'smoke_intensity', 'smoke_intensity_hat',
                       'education', 'education_hat', 'smokeyrs', 'smokeyrs_hat',
                       'active', 'active_hat','exercise', 'exercise_hat',
                       'wt71','wt71_hat','outcome','outcome_hat']

# Define and apply discretization
for feature in continuous_features:
    bins = np.linspace(df[feature].min(), df[feature].max(), num_bins)
    df[f'{feature}_binned'] = np.digitize(df[feature], bins, right=True)

# Print all column names of the DataFrame
# print(df.columns.tolist())

df = df.drop(columns=['intercept'])

print(df.columns.tolist())

file_name = f'data/raw/simulation{n}_noint_continuous.csv'
path = file_name
# Save the DataFrame
df.to_csv(file_name, index=False)

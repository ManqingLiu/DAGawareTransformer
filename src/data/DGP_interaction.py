import numpy as np
import pandas as pd
from scipy.stats import logistic
from itertools import product
from scipy.special import expit

from itertools import combinations

# Setting the random seed for reproducibility
np.random.seed(42)

# Number of samples
n = 5000

# Generating the confounders
sex = np.random.binomial(1, 0.65, n)
age = np.random.uniform(25, 65, n).astype(int)
smoke_intensity = np.clip(np.random.normal(15, 5, n), 1, 30).astype(int)
asthma = np.random.binomial(1, 0.2, n)

# Creating the DataFrame
df = pd.DataFrame({
    'sex': sex,
    'age': age,
    'smoke_intensity': smoke_intensity,
    'asthma': asthma,
    'intercept': 1  # Adding an intercept column
})

# Define the confounders
confounders = ['sex', 'age', 'smoke_intensity', 'asthma']

# Add two-way interaction terms to the DataFrame
for combo in combinations(confounders, 2):
    interaction_term = f'{combo[0]}_{combo[1]}'
    df[interaction_term] = df[combo[0]] * df[combo[1]]



# Example of adding a two-way interaction term
# df['sex_age'] = df['sex'] * df['age']

def find_appropriate_betas_for_quartiles(df, beta_ranges, lower_quartile=0.2, upper_quartile=0.8, tolerance=0.05, verbose=False):
    """
    Find coefficients for logistic regression such that the 25th and 75th quartiles
    of the propensity score are close to specified values within a tolerance.

    :param df: DataFrame containing the independent variables and interaction terms.
    :param beta_ranges: Dictionary with keys as column names and values as ranges of beta coefficients.
    :param lower_quartile: Target value for the 25th quartile.
    :param upper_quartile: Target value for the 75th quartile.
    :param tolerance: Allowed deviation from the target quartile values.
    :param verbose: If True, prints intermediate results.
    :return: Dictionary with the appropriate beta coefficients, or None if no suitable coefficients are found.
    """
    best_betas = None
    best_quartile_diff = float('inf')

    # Generating all combinations of beta coefficients
    keys, values = zip(*beta_ranges.items())
    beta_combinations = [dict(zip(keys, v)) for v in product(*values)]

    for betas in beta_combinations:
        # Calculating the linear part of the logistic regression model
        linear_part = sum(betas.get(key, 0) * df.get(key, 0) for key in betas)

        # Calculating the propensity scores using expit
        propensity_scores = expit(linear_part)

        # Checking the 25th and 75th quartiles
        quartiles = np.percentile(propensity_scores, [25, 75])

        # Calculate the difference from the target quartiles
        quartile_diff = abs(quartiles[0] - lower_quartile) + abs(quartiles[1] - upper_quartile)

        # Update best betas if this is the closest we have found so far
        if quartile_diff < best_quartile_diff:
            best_quartile_diff = quartile_diff
            best_betas = betas

            if verbose:
                print(f"Betas: {betas}, Quartiles: {quartiles}, Quartile Diff: {quartile_diff}")

        # Check if the quartile_diff is within the specified tolerance
        if quartile_diff <= tolerance:
            return betas

    return best_betas  # Return the best betas found, even if they don't meet the tolerance


# Specifying the ranges for each coefficient
beta_ranges = {
    'intercept': np.arange(-5, 0, 1),
    'sex': np.arange(-0.5, 0.5, 0.1),
    'age': np.arange(-0.1, 0.1, 0.02),
    'smoke_intensity': np.arange(-0.3, 0.3, 0.05),
    'asthma': np.arange(-1.0, 1.0, 0.2),
    'sex_age': np.arange(-1.0, 1.0, 0.5),  # Range for the sex_age interaction
    'sex_smoke_intensity': np.arange(-0.1, 0.1, 0.1),
    'sex_asthma': np.arange(-0.1, 0.1, 0.1),
    'age_smoke_intensity': np.arange(-0.1, 0.1, 0.1),
    'age_asthma': np.arange(-0.1, 0.1, 0.1),
    'smoke_intensity_asthma': np.arange(-0.1, 0.1, 0.1)
}


# Example usage with your beta_ranges and dataframe
# appropriate_betas = find_appropriate_betas_for_quartiles(df, beta_ranges, verbose=True)
# print("Appropriate Betas:", appropriate_betas)

# Your chosen betas
betas = {'intercept': -5, 'sex': -0.5, 'age': 0.08,
         'smoke_intensity': 0.25, 'asthma': -1.0,
         'sex_age': 0.0, 'sex_smoke_intensity': -0.1, 'sex_asthma': -0.1,
         'age_smoke_intensity': 0.0, 'age_asthma': -0.1, 'smoke_intensity_asthma': -0.1}

# Calculate the linear combination of the variables with their coefficients
linear_combination = sum(betas[key] * df[key] for key in betas)

# Apply the logistic function to get probabilities
propensity_scores = expit(linear_combination)

# Assign treatment based on whether the random number is less than the propensity score
df['treatment'] = (propensity_scores > 0.5).astype(int)
df['treatment_probability'] = propensity_scores

# sanity check
quantiles = [0.25, 0.5, 0.75]  # You can modify this list to include the quantiles you're interested in
quantile_values = df['treatment_probability'].quantile(quantiles)

print("Quantiles of Treatment Probability:")
print(quantile_values)


import matplotlib.pyplot as plt
import seaborn as sns

# Assuming df['treatment_probability'] contains the probabilities
sns.histplot(df['treatment_probability'], bins=20, kde=True)
plt.xlabel('Treatment Probability')
plt.ylabel('Frequency')
plt.title('Histogram of Treatment Probability')
plt.savefig('figures/propscore_hist.png')

print(df.columns.tolist())
#['sex', 'age', 'smoke_intensity', 'asthma', 'intercept', 'sex_age',
# 'sex_smoke_intensity', 'sex_asthma', 'age_smoke_intensity',
# 'age_asthma', 'smoke_intensity_asthma', 'treatment', 'treatment_probability']

# Desired order
new_order = ['intercept', 'treatment', 'sex', 'age', 'smoke_intensity', 'asthma',
             'sex_age', 'sex_smoke_intensity', 'sex_asthma', 'age_smoke_intensity',
             'age_asthma', 'smoke_intensity_asthma']

# Reorder the columns
df_sub = df[new_order]

betas_Y = {'intercept': -35, 'treatment': 2.5, 'sex': -0.1, 'age': 0.1,
         'smoke_intensity': 0.25, 'asthma': 0.2,
         'sex_age': 0.02, 'sex_smoke_intensity': -0.1, 'sex_asthma': 0.02,
         'age_smoke_intensity': 0.02, 'age_asthma': 0.01, 'smoke_intensity_asthma': 0.1}

# Calculate the linear combination of the variables with their coefficients
linear_combination = sum(betas_Y[key] * df_sub[key] for key in betas_Y)

# Apply the logistic function to get probabilities
prob_Y = expit(linear_combination)


betas_C = {'intercept': -40, 'treatment': 0.7, 'sex': -0.2, 'age': 0.3,
         'smoke_intensity': 0.1, 'asthma': 0.02,
         'sex_age': 0.03, 'sex_smoke_intensity': -0.05, 'sex_asthma': 0.06,
         'age_smoke_intensity': 0.02, 'age_asthma': 0.01, 'smoke_intensity_asthma': 0.6}

# Calculate the linear combination of the variables with their coefficients
linear_combination_C = sum(betas_C[key] * df_sub[key] for key in betas_C)

# Apply the logistic function to get probabilities of censoring
prob_C = expit(linear_combination_C)

df['censor'] = np.random.binomial(1, prob_C)

# Check the proportion of censoring
prop_ones_C = np.mean(df['censor'])
print("Proportion of censored:", prop_ones_C)  # 0.1596

# Assign treatment based on whether the random number is less than the propensity score
df['outcome'] = np.random.binomial(1, prob_Y)

# Check the proportion of '1's in the outcome
prop_ones = np.mean(df['outcome'])
print("Proportion of 1s in the outcome:", prop_ones)  # 0.063

df_A0 = df_sub.copy()
df_A0['treatment'] = 0
prob_Y_A0 = expit(sum(betas_Y[key] * df_A0[key] for key in betas_Y))

df_A1 = df_sub.copy()
df_A1['treatment'] = 1
prob_Y_A1 = expit(sum(betas_Y[key] * df_A1[key] for key in betas_Y))

print("True ATE:", prob_Y_A1.mean()-prob_Y_A0.mean())  ## 0.0306


# update Y where Y is randomly generated for those who were censored
df['outcome'] = np.where(df['censor'] == 1, np.nan, df['outcome'])


# generate hat variables
np.random.seed(50)
df['outcome_hat'] = np.random.binomial(1, 0.5, n)
df['treatment_hat'] = np.random.binomial(1, 0.5, n)
df['censor_hat'] = np.random.binomial(1, 0.5, n)
df['sex_hat'] = np.random.binomial(1, 0.5, n)
df['age_hat'] = np.random.uniform(25, 65, n).astype(int)
df['smoke_intensity_hat'] = np.clip(np.random.uniform(15, 5, n), 1, 30).astype(int)
df['asthma_hat'] = np.random.binomial(1, 0.5, n)

# Number of bins for discretization
num_bins = 20

# Define and apply discretization
for feature in ['age', 'age_hat', 'smoke_intensity', 'smoke_intensity_hat']:
    bins = np.linspace(df[feature].min(), df[feature].max(), num_bins)
    df[f'{feature}_discretized'] = np.digitize(df[feature], bins, right=True)


# Print all column names of the DataFrame
print(df.columns.tolist())


df = df.drop(columns=['intercept', 'sex_age', 'sex_smoke_intensity',
 'sex_asthma', 'age_smoke_intensity', 'age_asthma', 'smoke_intensity_asthma'])
df['treatment_probability'] = propensity_scores


# Specify the subfolder and file name
import os
subfolder = 'data'
file_name = f'simulation{n}_interaction_censored.csv'
#file_name = f'simulation{n}_interaction.csv'
path = os.path.join(subfolder, file_name)

# Save the DataFrame
df.to_csv(path, index=False)

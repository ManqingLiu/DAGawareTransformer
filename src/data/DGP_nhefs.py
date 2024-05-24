import pandas as pd
import numpy as np

# Load the dataset from the Excel file
file_path = 'data/nhefs/nhefs.xlsx'  # Change this to your actual file path
data = pd.read_excel(file_path)
# Select necessary columns and include 'wt82_71'
selected_columns = ['qsmk', 'sex', 'wt82', 'race', 'age', 'education', 'smokeintensity', 'smokeyrs', 'exercise',
                    'active', 'hbp', 'wt82_71']
selected_data = data[selected_columns]

# Prepare to generate samples
num_samples = 100
num_rows = selected_data.shape[0]
bootstrap_samples = []

# Generate samples
for _ in range(num_samples):
    sample = pd.DataFrame()
    for column in selected_columns:
        if selected_data[column].dtype == 'object' or len(selected_data[column].unique()) > 2:
            # Continuous variables: Sample from a Gaussian distribution
            mean, std = selected_data[column].mean(), selected_data[column].std(ddof=0)  # use population std deviation
            sample[column] = np.random.normal(mean, std, num_rows)
        else:
            # Binary variables: Sample from a Bernoulli distribution
            p = selected_data[column].mean()
            sample[column] = np.random.binomial(1, p, num_rows)

    # Calculate ATE within each sample
    y1 = sample.copy()
    y0 = sample.copy()
    y1['qsmk'] = 1
    y0['qsmk'] = 0

    # Calculate mean outcomes for each counterfactual scenario
    mean_y1 = y1['wt82_71'].mean()
    mean_y0 = y0['wt82_71'].mean()
    sample_ate = mean_y1 - mean_y0

    # Add ATE_true as a column to the sample
    sample['ATE_true'] = sample_ate

    bootstrap_samples.append(sample)

# Optionally, you can check or save the first sample to see the results
print(bootstrap_samples[0].head())


# Assuming `bootstrap_samples` is your list of DataFrame samples
for index, sample in enumerate(bootstrap_samples):
    file_name = f'data/nhefs/sample{index}/nhefs_sample{index}.csv'  # Generates the file name
    sample.to_csv(file_name, index=False)  # Saves the file without the index column

## Import dataset in csv as pandas dataframe
import pandas as pd
import numpy as np

# set print options to include all columns
pd.set_option('display.max_columns', 100)

'''
# Load the dataset
df = pd.read_csv('causal-predictive-analysis.csv')

# Filter the dataframe where dataset equals 'twins'
psid = df[df['dataset'] == 'twins']
# Group the filtered dataframe by 'meta-estimator' and summarize 'ate_rmse'
ate_rmse_summary = psid.groupby('meta-estimator')['ate_rmse'].agg(['mean', 'std'])

# Display the summary
print(ate_rmse_summary)



psid_outcome = psid[psid['meta-estimator'].isin(['standardization', 'stratified_standardization'])]

# adjust the code below to sort by min value of 'ate_rmse'
ate_rmse_outcome = psid_outcome.groupby(['meta-estimator', 'outcome_model'])['ate_rmse'].agg(['min'])
ate_rmse_outcome = ate_rmse_outcome.sort_values(by='min')
print(ate_rmse_outcome)

psid_treatment = psid[psid['meta-estimator'].isin(['ipw','ipw_stabilized'])]
ate_rmse_treatment = psid_treatment.groupby(['meta-estimator', 'prop_score_model'])['ate_rmse'].agg(['min'])
ate_rmse_treatment = ate_rmse_treatment.sort_values(by='min')
print(ate_rmse_treatment)

# print unit value of first column which is categorical
#print(analysis['dataset'].unique()) #['lalonde_psid' 'lalonde_cps' 'twins']

## Dataset: lalonde_cps, laonde_psid, twins
## outcome of laonde_cps and laonde_psid is continuous
## outcome of twins is binary
'''


# import the lalonde_cps_sample0 in csv as pandas dataframe
twins = pd.read_csv('data/realcause_datasets/twins_sample0.csv')

# get summary statistics of every column the dataframe
# print(twins.describe(include='all'))

df = twins
# Identify columns with dots in their names
columns_with_dots = [col for col in df.columns if '.' in col]

# Drop these columns from the DataFrame
df = df.drop(columns=columns_with_dots)

one_hot_columns = [f'gestatcat{i}' for i in range(1, 11)]

df['GESTAT10'] = df[one_hot_columns].idxmax(axis=1)

# Extract the category number from the column name and convert it to the original GESTAT10 values
df['GESTAT10'] = df['GESTAT10'].str.extract('(\d+)').astype(int) - 1

# Drop these columns from the DataFrame
df = df.drop(columns=one_hot_columns)

columns = ['GESTAT10'] + [col for col in df.columns if col != 'GESTAT10']
df = df[columns]

# get summary statistics of every column the dataframe
print(df.describe(include='all'))
print(df.columns)

'''
## summary of lalonde cps dataset
# continuous: age, education, re74, re75
# binary: black, hispanic, married, nodegree
# outcome: y -> continuous
import matplotlib.pyplot as plt
print(lalonde_cps[['re74']].describe())
lalonde_cps['re74'].hist()
plt.savefig('lalonde_cps_re74_hist.png')


# print column names of the dataframe
print(lalonde_cps.columns)
num_columns_lalonde_cps = lalonde_cps.shape[1]
print("Number of columns of cps data:", num_columns_lalonde_cps)
# get summary statistics of every column the dataframe
# get summary statistics of y, y0, y1, ite
print(lalonde_cps[['t','y', 'y0', 'y1', 'ite']].describe(include='all'))
print(lalonde_cps['t'].value_counts())
print(lalonde_cps['t'].value_counts(normalize=True))
print(lalonde_cps['y'].value_counts())
print(lalonde_cps['y'].value_counts(normalize=True))
# plot histogram of y0 and y1
import matplotlib.pyplot as plt
lalonde_cps['y'].hist()
plt.savefig('lalonde_cps_y_hist.png')
lalonde_cps['y0'].hist()
plt.savefig('lalonde_cps_y0_hist.png')
lalonde_cps['y1'].hist()
plt.savefig('lalonde_cps_y1_hist.png')

# Load the lalonde_psid dataset
lalonde_psid = load_realcause_dataset('lalonde_psid', 69)

# print column names of the dataframe
print(lalonde_psid.columns)
num_columns_lalonde_psid = lalonde_psid.shape[1]
print("Number of columns of psid data:", num_columns_lalonde_psid)
# get summary statistics of every column the dataframe
print(lalonde_psid[['t','y', 'y0', 'y1', 'ite']].describe(include='all'))
print(lalonde_psid['t'].value_counts())
print(lalonde_psid['t'].value_counts(normalize=True))

# Load the twins dataset
twins = load_realcause_dataset('twins', 69)

# print column names of the dataframe
print(twins.columns)
# get summary statistics of every column the dataframe
print(twins[['t','y', 'y0', 'y1', 'ite']].describe(include='all'))
# count number of columns in twins dataset
num_columns_twins = twins.shape[1]
print("Number of columns of twins data:", num_columns_twins)
# get counts and proportion of y in twins dataset
print(twins['y'].value_counts())
print(twins['y'].value_counts(normalize=True))
print(twins['t'].value_counts())
print(twins['t'].value_counts(normalize=True))
'''
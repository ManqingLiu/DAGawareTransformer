## Import dataset in csv as pandas dataframe
import pandas as pd
import numpy as np
# Load the dataset
df = pd.read_csv('causal-predictive-analysis.csv')

# Filter the dataframe where dataset equals 'lalonde_cps'
cps = df[df['dataset'] == 'lalonde_cps']
# Group the filtered dataframe by 'meta-estimator' and summarize 'ate_rmse'
ate_rmse_summary = cps.groupby('meta-estimator')['ate_rmse'].agg(['mean', 'std'])

# Display the summary
print(ate_rmse_summary)

# set print options to include all columns
pd.set_option('display.max_columns', 20)

cps_outcome = cps[cps['meta-estimator'].isin(['standardization', 'stratified_standardization'])]

# adjust the code below to sort by min value of 'ate_rmse'
ate_rmse_outcome = cps_outcome.groupby(['meta-estimator', 'outcome_model'])['ate_rmse'].agg(['min'])
ate_rmse_outcome = ate_rmse_outcome.sort_values(by='min')
print(ate_rmse_outcome)

cps_treatment = cps[cps['meta-estimator'].isin(['ipw','ipw_stabilized'])]
ate_rmse_treatment = cps_treatment.groupby(['meta-estimator', 'prop_score_model'])['ate_rmse'].agg(['min'])
ate_rmse_treatment = ate_rmse_treatment.sort_values(by='min')
print(ate_rmse_treatment)

# print unit value of first column which is categorical
#print(analysis['dataset'].unique()) #['lalonde_psid' 'lalonde_cps' 'twins']

## Dataset: lalonde_cps, laonde_psid, twins
## outcome of laonde_cps and laonde_psid is continuous
## outcome of twins is binary


'''
# import the lalonde_cps_sample0 in csv as pandas dataframe
lalonde_cps = pd.read_csv('data/realcause_datasets/lalonde_cps_sample0.csv')

# get summary statistics of every column the dataframe
print(lalonde_cps.describe(include='all'))

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
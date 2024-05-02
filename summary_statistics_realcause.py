## Import dataset in csv as pandas dataframe
import pandas as pd
import numpy as np

# set print options to include all columns
pd.set_option('display.max_columns', 100)


# Load the dataset
df = pd.read_csv('causal-predictive-analysis.csv')

# Filter the dataframe where dataset equals 'twins'
cps = df[df['dataset'] == 'lalonde_cps']
# print column names
print(cps.columns)
# add a column of relative bias where it is ate_bias devided by -7028.2493
cps['relative_bias'] = cps['ate_bias'] / -7028.2493
# Group the filtered dataframe by 'meta-estimator' and summarize 'ate_rmse'
ate_bias_summary = cps.groupby('meta-estimator')['relative_bias'].agg(['mean', 'std'])

# Display the summary
print(ate_bias_summary)



cps_outcome =cps[cps['meta-estimator'].isin(['standardization', 'stratified_standardization'])]

# adjust the code below to sort by min value of 'ate_rmse'
ate_bias_outcome = cps_outcome.groupby(['meta-estimator', 'outcome_model'])['relative_bias'].agg(['min'])
ate_bias_outcome = ate_bias_outcome.sort_values(by='min')
print(ate_bias_outcome)

# get the smallest abosulte value of relative bias
min_relative_bias = cps['relative_bias'].abs().min()
print(min_relative_bias)

# get the smallest absolute value and the corresponding outcome model of relative bias
min_relative_bias_outcome = cps_outcome.loc[cps_outcome['relative_bias'].abs().idxmin(),
['dataset', 'meta-estimator', 'outcome_model', 'params_outcome_model','relative_bias']]
print(min_relative_bias_outcome)



cps_treatment = cps[cps['meta-estimator'].isin(['ipw','ipw_stabilized'])]
ate_bias_treatment = cps_treatment.groupby(['meta-estimator', 'prop_score_model'])['relative_bias'].agg(['min'])
ate_bias_treatment = ate_bias_treatment.sort_values(by='min')
print(ate_bias_treatment)

# get the smallest absolute value and the corresponding treatment model of relative bias
# show dataset, meta_estimator, treatment_model, params_treatment_model in the output

min_relative_bias_treatment = cps_treatment.loc[cps_treatment['relative_bias'].abs().idxmin(),
['dataset', 'meta-estimator', 'prop_score_model', 'params_prop_score_model','relative_bias']]
print(min_relative_bias_treatment)




# Filter the dataframe where dataset equals 'twins'
cps = df[df['dataset'] == 'lalonde_cps']
# print column names
print(cps.columns)
# add a column of relative bias where it is ate_bias devided by -7028.2493
cps['relative_bias'] = cps['ate_bias'] / -7028.2493
# Group the filtered dataframe by 'meta-estimator' and summarize 'ate_rmse'
ate_bias_summary = cps.groupby('meta-estimator')['relative_bias'].agg(['mean', 'std'])

# Display the summary
print(ate_bias_summary)




# Filter the dataframe where dataset equals 'twins'
psid = df[df['dataset'] == 'lalonde_psid']

# add a column of relative bias where it is ate_bias devided by -7028.2493
psid['relative_bias'] = psid['ate_bias'] / -13346.9993

psid_outcome =psid[psid['meta-estimator'].isin(['standardization'])]
min_relative_bias_outcome = psid_outcome.loc[psid_outcome['relative_bias'].abs().idxmin(),
['dataset', 'meta-estimator', 'outcome_model', 'params_outcome_model','relative_bias']]
print(min_relative_bias_outcome)


psid_treatment = psid[psid['meta-estimator'].isin(['ipw','ipw_stabilized'])]
ate_bias_treatment = psid_treatment.groupby(['meta-estimator', 'prop_score_model'])['relative_bias'].agg(['min'])
ate_bias_treatment = ate_bias_treatment.sort_values(by='min')
print(ate_bias_treatment)

# get the smallest absolute value and the corresponding treatment model of relative bias
# show dataset, meta_estimator, treatment_model, params_treatment_model in the output

min_relative_bias_treatment = psid_treatment.loc[psid_treatment['relative_bias'].abs().idxmin(),
['dataset', 'meta-estimator', 'prop_score_model', 'params_prop_score_model','relative_bias']]
print(min_relative_bias_treatment)



# Filter the dataframe where dataset equals 'twins'
twins = df[df['dataset'] == 'twins']

# add a column of relative bias where it is ate_bias devided by -7028.2493
twins['relative_bias'] = twins['ate_bias'] / -0.06934245660881175


twins_treatment = twins[twins['meta-estimator'].isin(['ipw','ipw_stabilized'])]

# get the smallest absolute value and the corresponding treatment model of relative bias
# show dataset, meta_estimator, treatment_model, params_treatment_model in the output

min_relative_bias_treatment = twins_treatment.loc[twins_treatment['relative_bias'].abs().idxmin(),
['dataset', 'meta-estimator', 'prop_score_model', 'params_prop_score_model','relative_bias']]
print(min_relative_bias_treatment)




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
import os
import pandas as pd
import numpy as np
from scipy import stats
from sklearn import preprocessing

RANDOM_SEED = 42

os.chdir("../../../data/right_heart_catheterization")
df = pd.read_csv("rhc_raw.csv")

# Subdivide the columns into their data types/variable category
columns = ['swang1', 'pafi1', 'paco21', 'ph1', 'hema1', 't3d30',
           'dthdte', 'sadmdte', 'cat1', 'cat2', 'ca', 'dschdte',
           'lstctdte', 'death', 'cardiohx', 'chfhx', 'dementhx', 'psychhx',
           'chrpulhx', 'renalhx', 'liverhx', 'gibledhx', 'malighx', 'immunhx',
           'transhx', 'amihx', 'age', 'sex', 'edu', 'surv2md1', 'das2d3pc',
           'dth30', 'aps1', 'scoma1', 'meanbp1', 'wblc1', 'hrt1', 'resp1',
           'temp1', 'alb1', 'bili1', 'crea1', 'sod1', 'pot1',
           'wtkilo1', 'dnr1', 'ninsclas', 'resp',
           'card', 'neuro', 'gastr', 'renal', 'meta', 'hema', 'seps', 'trauma',
           'ortho', 'adld3p', 'urin1', 'race', 'income', 'ptid']

# Key proximal variables, per Miao et al. https://arxiv.org/pdf/2009.10982.pdf
A_name = ['swang1']
Z_names = ['pafi1', 'paco21']
W_names = ['ph1', 'hema1']
Y_name = ['t3d30']

# Variables outside of A, W, Z, and Y
X_or_U_names = ['dthdte', 'sadmdte', 'cat1', 'cat2', 'ca', 'dschdte',
                'lstctdte', 'death', 'cardiohx', 'chfhx', 'dementhx', 'psychhx',
                'chrpulhx', 'renalhx', 'liverhx', 'gibledhx', 'malighx', 'immunhx',
                'transhx', 'amihx', 'age', 'sex', 'edu', 'surv2md1', 'das2d3pc',
                'dth30', 'aps1', 'scoma1', 'meanbp1', 'wblc1', 'hrt1', 'resp1',
                'temp1', 'alb1', 'bili1', 'crea1', 'sod1', 'pot1',
                'wtkilo1', 'dnr1', 'ninsclas', 'resp',
                'card', 'neuro', 'gastr', 'renal', 'meta', 'hema', 'seps', 'trauma',
                'ortho', 'adld3p', 'urin1', 'race', 'income', 'ptid']

df['cat2'] = df['cat2'].fillna(value='missing')
categorical_X_names = ['cat1', 'cat2', 'ca', 'scoma1', 'ninsclas', 'race', 'income']

# No missingness in the binary variables!
binary_X_names = ['cardiohx', 'chfhx', 'dementhx', 'psychhx', 'chrpulhx', 'renalhx',
                  'liverhx', 'gibledhx', 'malighx', 'immunhx', 'transhx', 'amihx',
                  'sex', 'dnr1', 'resp',
                  'card', 'neuro', 'gastr', 'renal', 'meta', 'hema', 'seps', 'trauma',
                  'ortho']

# Create some extra binary indicators for continuous variables that have missingness
df['meanbp1_iszero_binary'] = df.meanbp1 == 0
df['urin1_isna_binary'] = df.urin1.isna()
df['hrt1_iszero_binary'] = df.hrt1 == 0
df['resp1_iszero_binary'] = df.resp1 == 0
df['wtkilo1_iszero_binary'] = df.wtkilo1 == 0
extra_binary_indicators = ['urin1_isna_binary', 'meanbp1_iszero_binary', 'hrt1_iszero_binary', 'resp1_iszero_binary',
                           'wtkilo1_iszero_binary']
binary_X_names += extra_binary_indicators

# urin1 is the only continuous variable with NaNs. Fill with 0.
df['urin1'] = df['urin1'].fillna(value=0)
continuous_X_names = ['age', 'edu', 'surv2md1', 'das2d3pc', 'aps1', 'wblc1', 'temp1', 'alb1', 'bili1',
                      'crea1', 'sod1', 'pot1', 'meanbp1', 'urin1', 'hrt1', 'resp1', 'wtkilo1']

left_out_names = ['dthdte', 'sadmdte', 'dschdte', 'lstctdte', 'death', 'dth30', 'adld3p', 'ptid']

# Encode binary and categorical variables as 0-indexed levels (no leakage if we do this before splitting)
for col in A_name + binary_X_names + categorical_X_names:
    le = preprocessing.LabelEncoder()
    le.fit(df[col])
    df[col] = le.transform(df[col])

# Split data into 80%:10%:10% partitions.
# 80% for train, 10% for validation/model selection, and 10% for computing the ATE at test time
np.random.seed(RANDOM_SEED)
train, validate, test = np.split(df.sample(frac=1, random_state=RANDOM_SEED), [int(.8*len(df)), int(.9*len(df))])

# Perform standardization and log transforms for continuous variables
train_summary_stats = []
for split_name, split in zip(["train", "val", "test"], [train, validate, test]):

    for col in Z_names + W_names + continuous_X_names:  # TODO: should I normalize Y? Should I normalize at all?
        if split_name == 'train':
            if col in ['crea1', 'bili1']:  # log transform before standardizing
                mean = np.log(split[col].mean())
                std = np.log(split[col].std())
                split[col] = (np.log(split[col]) - mean) / std

            mean = split[col].mean()
            std = split[col].std()
            split[col] = (split[col] - split[col].mean()) / split[col].std()

            train_summary_stats.append({'variable': col, 'split': split_name, 'mean': mean, 'std': std})

        else:
            mean = train_summary_stats_df[train_summary_stats_df.variable == col]['mean'].values
            std = train_summary_stats_df[train_summary_stats_df.variable == col]['std'].values
            split[col] = (split[col] - mean) / std

    train_summary_stats_df = pd.DataFrame(train_summary_stats)
    split.drop(columns=left_out_names, inplace=True)
    split.to_csv(f"rhc_{split_name}.csv", index=False)

train_summary_stats_df.to_csv(f"rhc_train_summarystats.csv", index=False)

# Significance testing for the potential X features

# Categorical: Chi-squared test for exposure relationship, Kruskal-Wallis H test for outcome relationship.
# Binary variables: Mann Whitney U test for outcome relationship, Chi-squared for exposure relationship.
# Continuous variables: Mann Whitney U test for exposure relationship, Spearman correlation for outcome relationship.

cat_data = []
for index, cat_col in enumerate(categorical_X_names):

    # test the association with the binary treatment variable using a Chi-squared test of independence
    contingency = pd.crosstab(df.swang1, df[cat_col])
    chi2, A_pval, dof, _ = stats.chi2_contingency(contingency)

    # test the association with the quasi-continuous outcome variable using Kruskal-Wallis H-test
    num_levels = len(np.unique(df[cat_col]))
    Y_pval = stats.kruskal(*[df[df[cat_col] == i].t3d30 for i in range(num_levels)])[1]

    cat_data.append({'variable': cat_col, 'pval_with_A': A_pval, 'pval_with_Y': Y_pval, 'var_type': 'categorical',
                     'A_test': 'chi_squared', 'Y_test': 'kruskal_wallis'})

bin_data = []
for index, bin_col in enumerate(binary_X_names):

    # test the association with the binary treatment variable using a Chi-squared test of independence
    contingency = pd.crosstab(df.swang1, df[bin_col])
    chi2, A_pval, dof, _ = stats.chi2_contingency(contingency)

    # test the association with the quasi-continuous outcome variable using Mann Whitney U test
    u_stat, Y_pval = stats.mannwhitneyu(df[df[bin_col] == 1].t3d30, df[df[bin_col] == 0].t3d30)

    bin_data.append({'variable': bin_col, 'pval_with_A': A_pval, 'pval_with_Y': Y_pval, 'var_type': 'binary',
                     'A_test': 'chi_squared', 'Y_test': 'mann_whitney_u'})

continuous_data = []
for index, continuous_col in enumerate(continuous_X_names):
    # Test the association with the binary exposure variable using the Mann Whitney U test
    u_stat, A_pval = stats.mannwhitneyu(df[df.swang1 == 0][continuous_col],
                                        df[df.swang1 == 1][continuous_col])

    # Test the association with the quasi-continuous outcome variable using Spearman correlation
    corr, Y_pval = stats.spearmanr(df[continuous_col], df.t3d30)

    continuous_data.append(
        {'variable': continuous_col, 'pval_with_A': A_pval, 'pval_with_Y': Y_pval, 'var_type': 'continuous',
         'A_test': 'mann_whitney_u', 'Y_test': 'spearman_correlation'})

cat_df = pd.DataFrame(cat_data)
bin_df = pd.DataFrame(bin_data)
cont_df = pd.DataFrame(continuous_data)

significance_tests_df = pd.concat([cont_df, bin_df, cat_df], ignore_index=True)
significance_tests_df = significance_tests_df.sort_values(by=['pval_with_A', 'pval_with_Y'],
                                                          ascending=False).reset_index(drop=True)
significance_tests_df.to_csv("rhc_Xfeature_significance.csv", index=False)

# Drop variables based on insufficient statistical significance
threshold = 0.05 / (2 * len(significance_tests_df))

X_significant = significance_tests_df[
    (significance_tests_df.pval_with_A < threshold) & (significance_tests_df.pval_with_Y < threshold)].variable

X_all = significance_tests_df.variable

X_significant.to_csv("RHC_X_significantfeatures_list.csv", index=False)
X_all.to_csv("RHC_X_allfeatures_list.csv", index=False)

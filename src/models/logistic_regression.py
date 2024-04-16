### Write code to run a logistic regression where outcome is t, and covariates are all other variables in the dataset
### except y, and predict t using a validation set (split 5/5)
from torch.optim import AdamW
from src.models.DAG_aware_transformer import *
from src.models.utils import *
from src.data.data_preprocess import DataProcessor
from config import *

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


set_seed()

##### Part I: data pre-processing #####
dataset_type = 'twins'  # Can be 'cps' or 'psid', depending on what you want to use

# Use the variable in the file path
# dataframe = pd.read_csv(f'data/realcause_datasets/{dataset_type}_sample{N_SAMPLE}.csv')
# Use the variable in the file path
# Use the variable in the file path
dataframe = pd.read_csv(f'data/realcause_datasets/{dataset_type}_sample{N_SAMPLE}.csv')

ATE_true = dataframe['y1'].mean() - dataframe['y0'].mean()
print("true ATE:", ATE_true)

df = dataframe.iloc[:, :-3].copy()


# Assuming df is your DataFrame and it's already defined
X = df.drop(columns=['t', 'y'])  # Use all columns except 't' and 'y' as features
y = df['t']  # 't' is the target variable

# Split the dataset into training set and validation set
# Here, 50% of the data goes to the validation set because of the test_size=0.5 parameter
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.5, random_state=42)

# Create a Logistic Regression model
# Increase the number of iterations (max_iter) if the model does not converge
# remove y in the model
model = LogisticRegression(max_iter=1000, penalty='l2', C=1)

#model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)

# Predict 't' for the validation set
# prediction should be in probability
# Predict 't' for the validation set in terms of probabilities
y_val_pred_proba = model.predict_proba(X_val)

# y_val_pred_proba[:, 1] will give you the probabilities for class 1
print(y_val_pred_proba[:, 1])

# Assign y_val_pred to the 'pred_t' column in the validation set
X_val['pred_t'] = y_val_pred_proba[:, 1]
X_val['t'] = y_val


# Assuming df is your DataFrame and it's already defined
df = dataframe.iloc[:, :-3].copy()
X2 = df.drop(columns=['t'])  # Use all columns except 't' and 'y' as features
y2 = df['y']  # 'y' is the target variable

# Split the dataset into training set and validation set
# Here, 50% of the data goes to the validation set because of the test_size=0.5 parameter
X2_train, X2_val, y2_train, y2_val = train_test_split(X2, y2, test_size=0.5, random_state=42)
X_val['y'] = y2_val


# Assuming 't' and 'y' columns exist in the combined_df or are added from the original dataset
ATE_IPTW = IPTW_stabilized(X_val['t'], X_val['y'], X_val['pred_t'])
print(f"Predicted ATE from IPTW: {ATE_IPTW:.4f}")
rmse_IPTW = rmse(ATE_IPTW, ATE_true)
print(f"RMSE from IPTW: {rmse_IPTW:.4f}")
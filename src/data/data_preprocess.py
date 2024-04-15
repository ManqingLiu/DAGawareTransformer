import pandas as pd
import numpy as np
import torch
from src.models.utils import generate_dag_edges
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import KBinsDiscretizer
from config import SEED_VALUE

from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return idx, self.data[idx]

class DataProcessor:
    def __init__(self, df):
        self.df = df
        self.feature_names = None
        self.kbins_models = {}

    def detect_variable_type(self, col):
        # Detect if the variable is continuous or binary
        unique_values = self.df[col].unique()
        if len(unique_values) <= 2:
            return 'binary'
        else:
            return 'continuous'


    def standardize_column(self, col):
        """
        Standardizes a column by subtracting the mean and dividing by the standard deviation.

        Parameters:
        - column: Pandas Series representing the column to be standardized.

        Returns:
        - Standardized Pandas Series.
        """
        if self.detect_variable_type(col) == 'continuous':
            column = self.df[col]
            standardized_column = (column - column.mean()) / column.std()
            return standardized_column
        else:
            # Return the column unchanged if it is not continuous
            return self.df[col]

    def sample_variables(self):
        for col in self.df.columns:
            var_type = self.detect_variable_type(col)
            if var_type == 'continuous':
                if col == 'u':  # Check if the column name is 'u'
                    self.df[f'{col}_hat'] = np.random.uniform(0, 1, self.df.shape[0])
                else:
                    mean, std = self.df[col].mean(), self.df[col].std()
                    self.df[f'{col}_hat'] = np.random.normal(mean, std, self.df.shape[0])
                    # Ensure generated values are within the min and max of the original column
                    self.df[f'{col}_hat'] = np.clip(self.df[f'{col}_hat'], self.df[col].min(), self.df[col].max())
            elif var_type == 'binary':
                prob = self.df[col].mean()
                self.df[f'{col}_hat'] = np.random.binomial(1, prob, self.df.shape[0])

    def bin_continuous_variables(self, num_bins):
        # Initialize the KBinsDiscretizer
        kbins = KBinsDiscretizer(n_bins=num_bins, encode='ordinal', strategy='uniform')

        # Identify continuous variables
        continuous_vars = [col for col in self.df.columns if self.detect_variable_type(col) == 'continuous']

        # Temporary DataFrame to hold the binned columns
        new_cols_df = pd.DataFrame(index=self.df.index)

        # Apply KBinsDiscretizer to each continuous variable
        for col in continuous_vars:
            # Reshape data for KBinsDiscretizer
            data_reshaped = self.df[col].values.reshape(-1, 1)
            # Fit and transform the data, then save it in the temporary DataFrame
            new_cols_df[f'{col}_bin'] = kbins.fit_transform(data_reshaped).astype(int)
            self.kbins_models[col] = kbins  # Save the model for later use

            # Repeat for '_hat' version if exists
            if f'{col}_hat' in self.df:
                data_reshaped_hat = self.df[f'{col}_hat'].values.reshape(-1, 1)
                new_cols_df[f'{col}_hat_bin'] = kbins.fit_transform(data_reshaped_hat).astype(int)

        # Concatenate the original DataFrame with the new columns all at once
        self.df = pd.concat([self.df, new_cols_df], axis=1)
    '''
    def bin_continuous_variables(self, num_bins):
        # Initialize the KBinsDiscretizer
        kbins = KBinsDiscretizer(n_bins=num_bins, encode='ordinal', strategy='uniform')

        # Identify continuous variables
        continuous_vars = [col for col in self.df.columns if self.detect_variable_type(col) == 'continuous']

        # Apply KBinsDiscretizer to each continuous variable
        for col in continuous_vars:
            # Reshape data for KBinsDiscretizer
            data_reshaped = self.df[col].values.reshape(-1, 1)
            self.df[f'{col}_bin'] = kbins.fit_transform(data_reshaped).astype(int)
            self.kbins_models[col] = kbins  # Save the model for later use

            # Fit and transform the data
            self.df[f'{col}_bin'] = kbins.fit_transform(data_reshaped).astype(int)

            # Repeat for '_hat' version if exists
            if f'{col}_hat' in self.df:
                data_reshaped_hat = self.df[f'{col}_hat'].values.reshape(-1, 1)
                self.df[f'{col}_hat_bin'] = kbins.fit_transform(data_reshaped_hat).astype(int)
        '''

    '''
    def bin_to_original(self, binned_data, feature_name):
        kbins_model = self.kbins_models.get(feature_name)
        if not kbins_model:
            raise ValueError(f"No KBinsDiscretizer model found for feature '{feature_name}'.")

        # Ensure binned_data is an array of integers
        binned_data = np.array(binned_data).astype(int)

        # Retrieve the bin edges for the feature
        bin_edges = kbins_model.bin_edges_[0]

        # Calculate bin centers from bin edges for the 'uniform' strategy
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Map each binned index to its corresponding bin center
        try:
            original_values = np.array([bin_centers[bin_idx] for bin_idx in binned_data])
        except IndexError as e:
            print(f"IndexError: {e}. This may indicate that binned_data contains invalid bin indices.")
            raise

        return original_values
    '''

    def bin_to_original(self, binned_data, feature_name):
        kbins_model = self.kbins_models.get(feature_name)
        if not kbins_model:
            raise ValueError(f"No KBinsDiscretizer model found for feature '{feature_name}'.")

        # Ensure binned_data is an array of integers
        binned_data = np.array(binned_data).astype(int)

        # Retrieve the bin edges for the feature
        bin_edges = kbins_model.bin_edges_[0]

        # Calculate bin centers from bin edges for the 'uniform' strategy
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Map each binned index to its corresponding bin center
        original_values = []
        for bin_idx in binned_data:
            if bin_idx < len(bin_centers):
                original_values.append(bin_centers[bin_idx])
            else:
                original_values.append(np.nan)

        return np.array(original_values)



    def get_feature_names(self):
        binary_features = []
        continuous_features = []
        for col in self.df.columns:
            var_type = self.detect_variable_type(col)
            if var_type == 'binary':
                binary_features.append(col)
            elif var_type == 'continuous':
                continuous_features.append(col)
        return binary_features, continuous_features

    '''
    def create_tensor(self):
        # Separate lists for original binary variables and their '_hat' versions
        binary_cols = [col for col in self.df.columns if
                       self.detect_variable_type(col) == 'binary' and '_hat' not in col]
        binary_hat_cols = [col for col in self.df.columns if
                           col.endswith('_hat') and self.detect_variable_type(col.replace('_hat', '')) == 'binary']

        # Lists for binned continuous variables and their '_hat_bin' versions
        continuous_bin_cols = [col for col in self.df.columns if col.endswith('_bin') and 'hat' not in col]
        continuous_hat_bin_cols = [col for col in self.df.columns if col.endswith('_hat_bin')]

        # Ordering columns: binary, binary_hat, continuous_bin, continuous_hat_bin
        ordered_cols = []
        for col in binary_cols:
            ordered_cols.append(col)
            hat_col = f'{col}_hat'
            if hat_col in binary_hat_cols:
                ordered_cols.append(hat_col)

        for bin_col in continuous_bin_cols:
            hat_bin_col = bin_col.replace('_bin', '_hat_bin')
            if hat_bin_col in continuous_hat_bin_cols:
                ordered_cols += [bin_col, hat_bin_col]

        tensor_data = torch.tensor(self.df[ordered_cols].values, dtype=torch.long)
        return tensor_data, ordered_cols
    '''
    '''
    def generate_dimensions(self):
        # Count binary variables (including '_hat' versions)
        binary_vars = len([col for col in self.df.columns if self.detect_variable_type(col) == 'binary'])
        binary_dims = [2] * (binary_vars)

        # Initialize list for continuous dimensions
        continuous_dims = []

        # Iterate over continuous variables to find the actual number of bins
        for col in self.df.columns:
            if col.endswith('_bin'):
                # Determine the number of unique bin values for the variable
                num_unique_bins = self.df[col].nunique()

                # Append the number of bins to continuous_dims twice (for the variable and its '_hat' counterpart)
                continuous_dims.append(num_unique_bins)

        return binary_dims, continuous_dims
    '''

    def generate_dimensions(self):
        # Count binary variables (including '_hat' versions)
        binary_vars = len([col for col in self.df.columns if self.detect_variable_type(col) == 'binary'])
        binary_dims = [2] * binary_vars

        # Find the maximum number of unique bins across all continuous variables
        max_unique_bins = max([self.df[col].nunique() for col in self.df.columns if col.endswith('_bin')], default=0)

        # The length of continuous_dims should be equal to the number of features ending with _bin,
        # with each element set to the maximum number of unique bins found
        continuous_dims = [max_unique_bins] * len([col for col in self.df.columns if col.endswith('_bin')])

        return binary_dims, continuous_dims

    def create_tensor(self):
        # Separate lists for original binary variables and their '_hat' versions
        binary_cols = [col for col in self.df.columns if
                       self.detect_variable_type(col) == 'binary' and '_hat' not in col]
        binary_hat_cols = [col for col in self.df.columns if
                           col.endswith('_hat') and self.detect_variable_type(col.replace('_hat', '')) == 'binary']

        # Lists for binned continuous variables and their '_hat_bin' versions
        continuous_bin_cols = [col for col in self.df.columns if col.endswith('_bin') and 'hat' not in col]
        continuous_hat_bin_cols = [col for col in self.df.columns if col.endswith('_hat_bin')]

        # Ordering columns: binary, binary_hat, continuous_bin, continuous_hat_bin
        ordered_cols = []
        for col in binary_cols:
            ordered_cols.append(col)
            hat_col = f'{col}_hat'
            if hat_col in binary_hat_cols:
                ordered_cols.append(hat_col)

        for bin_col in continuous_bin_cols:
            hat_bin_col = bin_col.replace('_bin', '_hat_bin')
            if hat_bin_col in continuous_hat_bin_cols:
                ordered_cols += [bin_col, hat_bin_col]

        # Remove 'u_bin' from ordered_cols but keep everything else, including 'u_hat_bin'
        ordered_cols = [col for col in ordered_cols if col != 'u_bin']

        # Ensure 'u_hat_bin' is moved to the last position if it exists
        if 'u_hat_bin' in ordered_cols:
            ordered_cols.remove('u_hat_bin')  # Remove 'u_hat_bin' from its current position
            #ordered_cols.append('u_hat_bin')  # Add 'u_hat_bin' to the end

        tensor_data = torch.tensor(self.df[ordered_cols].values, dtype=torch.long)
        return tensor_data, ordered_cols

    def split_data_loaders(self, tensor_data, batch_size, test_size, random_state,feature_names):
        """
        Splits the tensor data into training, validation, and test DataLoader objects.

        Parameters:
        - tensor_data: The tensor containing all the data.
        - batch_size: Batch size for the DataLoader objects.
        - test_size: Proportion of the dataset to include in the test split.
        - val_size: Proportion of the test dataset to include in the validation split.
        - random_state: Controls the shuffling applied to the data before applying the split.

        Returns:
        - train_loader: DataLoader for the training set.
        - val_loader: DataLoader for the validation set.
        - test_loader: DataLoader for the testing set.
        """
        # Split the data into training, validation, and testing sets
        train_data, val_data = train_test_split(tensor_data, test_size=test_size, random_state=random_state)
        #val_data, test_data = train_test_split(temp_data, test_size=test_size, random_state=random_state)
        t_index = feature_names.index('t')
        val_data_A1 = val_data.clone()
        val_data_A1[:, t_index] = 1
        val_data_A0 = val_data.clone()
        val_data_A0[:, t_index] = 0
        train_data_A1 = train_data.clone()
        train_data_A1[:, t_index] = 1
        train_data_A0 = train_data.clone()
        train_data_A0[:, t_index] = 0

        '''
        # Convert subsets to TensorDataset
        train_dataset = TensorDataset(train_data)
        val_dataset = TensorDataset(val_data)
        val_dataset_A1 = TensorDataset(val_data_A1)
        val_dataset_A0 = TensorDataset(val_data_A0)
        train_dataset_A1= TensorDataset(train_data_A1)
        train_dataset_A0 = TensorDataset(train_data_A0)
        #test_dataset = TensorDataset(test_data)
        '''

        # Create DataLoader objects
        # train_dataloader = DataLoader(CustomDataset(training_data, training_labels), batch_size=64, shuffle=True)
        train_loader = DataLoader(CustomDataset(train_data), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(CustomDataset(val_data), batch_size=batch_size, shuffle=True)
        val_loader_A1 = DataLoader(CustomDataset(val_data_A1), batch_size=batch_size, shuffle=True)
        val_loader_A0 = DataLoader(CustomDataset(val_data_A0), batch_size=batch_size, shuffle=True)
        train_loader_A1 = DataLoader(CustomDataset(train_data_A1), batch_size=batch_size, shuffle=True)
        train_loader_A0 = DataLoader(CustomDataset(train_data_A0), batch_size=batch_size, shuffle=True)

        return (train_loader, train_data, val_loader, val_data,
                val_loader_A1, val_data_A1, val_loader_A0, val_data_A0,
                train_loader_A1, train_data_A1, train_loader_A0, train_data_A0)

def create_counterfactual_df(df, t=None):
    modified_df = df.copy()
    if t is not None and 't' in modified_df.columns:
        modified_df['t'] = t
    return modified_df



if __name__ == "__main__":
    dataframe = pd.read_csv('data/realcause_datasets/twins_sample0.csv').head(100)
    # remove last 3 columns of the dataframe
    dataframe = dataframe.iloc[:, :-3]
    num_bins = 15
    processor = DataProcessor(dataframe)
    processor.sample_variables()
    processor.bin_continuous_variables(num_bins)
    tensor, feature_names = processor.create_tensor()
    binary_dims, continuous_dims = processor.generate_dimensions()
    binary_features, _ = processor.get_feature_names()  # Get binary and continuous feature names
    pd.set_option('display.max_columns', 20)
    print("Tensor:\n", tensor)
    print("Tensor Dimensions:", tensor.shape)
    print("Feature names:", feature_names)
    print("Binary features:", binary_features)
    print("Binary dimensions:", binary_dims)
    print("Continuous dimensions:", continuous_dims)
    edges = generate_dag_edges(feature_names)
    print(edges)
    # Split data and create DataLoaders
    train_loader, train_data, val_loader, val_data, \
        val_loader_A1, val_data_A1, val_loader_A0, val_data_A0, \
        train_loader_A1, train_data_A1, train_loader_A0, train_data_A0 = (
        processor.split_data_loaders(tensor, batch_size=32, test_size=0.5, random_state=SEED_VALUE,
                                     feature_names=feature_names))

    # print size of train_loader
    print(len(train_loader))
    print(len(train_loader.dataset))
    print(len(val_loader))
    print(len(val_loader.dataset))


    # Assuming num_features is defined
    num_features = len(feature_names)   # Example, adjust based on your actual number of features

    # Initialize storage for min and max values for each feature
    min_values = torch.full((num_features,), float('inf'))
    max_values = torch.full((num_features,), float('-inf'))

    # Loop through all batches in the DataLoader
    for idx, features in val_loader:  # Assuming the dataset only contains features, no labels
        for i in range(num_features):
            # Extract the i-th feature across all samples in the batch
            feature_i = features[:, i]

            # Update the min and max values for the i-th feature
            min_values[i] = min(min_values[i], feature_i.min())
            max_values[i] = max(max_values[i], feature_i.max())

    # Print the min and max values for each feature
    for i in range(num_features):
        print(f"Feature {i + 1}: Min = {min_values[i]}, Max = {max_values[i]}")

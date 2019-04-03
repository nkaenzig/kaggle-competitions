import pandas as pd
import numpy as np


def get_nan_features_with_count(df):
    nan_features_to_count_dict = dict()
    for col_name in df.columns:
        nr_nans = (df[col_name].isnull().values == True).sum()
        if nr_nans > 0:
            nan_features_to_count_dict[col_name] = nr_nans
    return nan_features_to_count_dict


def load_features_and_labels_from_csv(csv_path, label_name, shuffle=False, train_fraction=None):
    if train_fraction is float:
        df = pd.read_csv(csv_path)

        if shuffle:
            df = df.sample(frac=1).reset_index(drop=True)

        train_size = int(train_fraction * len(df))
        x_train_df = df.iloc[:train_size]
        x_test_df = df.iloc[train_size:].reset_index(drop=True)

        y_train_df = x_train_df.pop(label_name)
        y_test_df = x_test_df.pop(label_name).reset_index(drop=True)

        return x_train_df, y_train_df, x_test_df, y_test_df
    else:
        x_df = pd.read_csv(csv_path)

        if shuffle:
            x_df = x_df.sample(frac=1).reset_index(drop=True)

        y_df = x_df.pop(label_name)

        return x_df, y_df


def split_into_train_and_test_sets(df, train_fraction, shuffle=False):
    if shuffle:
        df = df.sample(frac=1).reset_index(drop=True)
    
    train_size = int(train_fraction * len(df))

    df_train = df.iloc[:train_size]
    df_test = df.iloc[train_size:].reset_index(drop=True)

    return df_train, df_test

def impute_missing_values(df, impute_method):
    cat_colnames = df.select_dtypes(include=['object']).columns
    num_colnames = df.select_dtypes(exclude=['object']).columns
    if impute_method == 'mm':
        df[num_colnames] = df[num_colnames].fillna(df[num_colnames].mean())
        # df[num_colnames] = df[num_colnames].fillna(df[num_colnames].median())

        # | calculate modes of the categorical features 
        # | note: can't use x_train_cat_df.mode(axis=1) directly for fillna, because 
        # | some modes are NaN (x_train_cat_df.fillna(x_train_cat_df.mode(axis=1), inplace=True))
        col_mode_dict = dict()
        for col in cat_colnames:
            isnull = df[col].isnull()
            if isnull.all():
                df.drop(col)    # drop column if all values are NaN
            if isnull.any():
                mode = df[col].dropna().mode()
                col_mode_dict[col] = int(mode)

        df[cat_colnames] = df[cat_colnames].fillna(col_mode_dict)
    
    elif impute_method == 'drop':
        # | drop all columns where more than 90% of the values are N/A
        df = df.dropna(axis=1, thresh=int(0.9*df.shape[0]))
        df = df.dropna()
        
    return df


def get_categorical_feature_names(df):
    return df.select_dtypes(include=['object']).columns


def create_categorical_feature_mapping(df, keep_nan=True):
    # | returns a dictionary that maps nominal values of categorical features to integer values
    # categorical_df = df.select_dtypes(include=['object'])
    categorical_features = get_categorical_feature_names(df)
    cat_mapping_dict = dict()
    for feature_name in categorical_features:
        nominal_to_code_dict = dict()
        df[feature_name] = df[feature_name].astype('category')
        for nominal, code in zip(df[feature_name].unique(), df[feature_name].cat.codes.unique()):
            if keep_nan and code == -1:
                code = None
            nominal_to_code_dict[nominal] = code
        
        cat_mapping_dict[feature_name] = nominal_to_code_dict

    return cat_mapping_dict
    

def categorical_to_numerical(df, cat_mapping_dict=None):
    if not cat_mapping_dict:
        cat_mapping_dict = create_categorical_feature_mapping(df)
        return df.replace(cat_mapping_dict), cat_mapping_dict
    else:
        return df.replace(cat_mapping_dict)


def one_hot_encoding(df):
    pd.get_dummies(df, columns=["body_style", "drive_wheels"], prefix=["body", "drive"]).head()

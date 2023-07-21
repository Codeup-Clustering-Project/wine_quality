import pandas as pd
import numpy as np
import os
from typing import Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler



####################################### ACQUIRE #########################################


def save_original_data(df:pd.DataFrame, folder:str = "origainal_data", file_name:str = "original_data") -> str:
    """
    df: dataframe to save
    folder: folder name to save to
    file_name: file name to be assigned to the data
    """
    if not os.path.exists(f"./{folder}"):
        os.makedirs(folder)
        df.to_csv(f"./{folder}/00_{file_name}.csv", mode="w")
        return f"File {file_name} saved"
    else:
        df.to_csv(f"./{folder}/00_{file_name}.csv", mode="w")
        return f"File {file_name} saved"

def get_wine_data():
    '''This function gets the two seperate datasets from Data.World, concats the two,
        adds columns to designate between wine color, and then saves it into a csv file.'''
    
    # pull the data from Data.World
    red_df = pd.read_csv('https://query.data.world/s/onvxunlnuu2ksqr6shkrmh7xbuqox2?dws=00000')
    white_df = pd.read_csv('https://query.data.world/s/npk7bzrgjmxoovtbh4pelcgmwrh5dt?dws=00000')

    # add column for wine color
    red_df["wine_clr"] = "red"
    white_df["wine_clr"] = "white"

    # concat the two dfs
    wine = pd.concat([red_df, white_df])

    # save data
    save_original_data(wine,file_name="wine_original_data")

    # load data from the original data file
    wine = pd.read_csv("./origainal_data/00_wine_original_data.csv", index_col=0)
    wine = wine.reset_index(drop=True) 

    return wine


###################################### PREPARE #########################################

def prep_wine(wine):
    '''This function renames the columns, removes duplicate rows, creates dummy variables,
       makes a copy of the original data, add dummies to df. '''

    # rename the columns
    new_cols = wine.columns.str.strip().str.replace(" ", "_").str.lower()

    wine[new_cols] = wine
    wine = wine[new_cols]

    # remove the duplocated rows
    wine = wine.drop_duplicates(keep="first")

    # create dummie variables
    dummies = pd.get_dummies(wine.wine_clr)

    # clean dummie column names
    dummies_col = dummies.columns.str.replace(" ", "_").str.lower()

    # make a copy of my original data frame to keep integrity of data
    original_clean_wine = wine.copy()

    # add dummies to my data frame
    wine[dummies_col] = dummies

    return wine

def split_data_(df: pd.DataFrame, test_size: float =.2, validate_size: float =.2, stratify_col: str =None, random_state: int=95) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    '''
    parameters:
        df: pandas dataframe you wish to split
        test_size: size of your test dataset
        validate_size: size of your validation dataset
        stratify_col: the column to do the stratification on
        random_state: random seed for the data

    return:
        train, validate, test DataFrames
    '''
    # no stratification
    if stratify_col == None:
        # split test data
        train_validate, test = train_test_split(df, 
                                                test_size=test_size, 
                                                random_state=random_state)
        # split validate data
        train, validate = train_test_split(train_validate, 
                                           test_size=validate_size/(1-test_size),
                                            random_state=random_state)
    # stratify split
    else:
        # split test data
        train_validate, test = train_test_split(df,
                                                test_size=test_size,
                                                random_state=random_state, 
                                                stratify=df[stratify_col])
        # split validate data
        train, validate = train_test_split(train_validate, 
                                           test_size=validate_size/(1-test_size),
                                           random_state=random_state, 
                                           stratify=train_validate[stratify_col])
    return train, validate, test


def split_wine(wine):
    # split the data into training, validation and testing sets
    train, validate, test = split_data_(df=wine,
                    test_size=0.2, 
                     validate_size=0.2,
                    stratify_col= "quality",
                     random_state=95)

    return train, validate, test

def save_split_data_(original_df:pd.DataFrame,encoded_scaled_df: pd.DataFrame, train:pd.DataFrame, validate:pd.DataFrame, test:pd.DataFrame, folder_path: str = "./00_project_data",
                     test_size:float = 0.2,stratify_col:str=None, random_state: int=95 ) -> str:
    """
    parameters:
        encoded_df: full project dataframe that contains the (encoded columns or scalling)
        train: training data set that has been split from the original
        validate: validation data set that has been split from the original
        test: testing data set that has been split from the original
        folder_path: folder path where to save the data sets

        Only apply to spliting the original_df in inside this function
            --> test_size:float = 0.2,stratify_col:str=None, random_state: int=95
    return:
        string to show succes of saving the data
    """
    # split original clearn no dumies data frame
    org_train_df, org_val_df, org_test_df = split_data_(df=original_df, test_size=test_size, stratify_col=stratify_col, random_state=random_state)


    # create new folder if folder don't aready exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        # save the dataframe with dummies in a csv for easy access
        original_df.to_csv(f"./{folder_path}/00_original_clean_no_dummies.csv", mode="w")

        # save the dataframe with dummies in a csv for easy access
        org_train_df.to_csv(f"./{folder_path}/01_original_clean_no_dummies_train.csv", mode="w")

        # save the dataframe with dummies in a csv for easy access
        encoded_scaled_df.to_csv(f"./{folder_path}/1-0_encoded_data.csv", mode="w")

        # save training data
        train.to_csv(f"./{folder_path}/1-1_training_data.csv", mode="w")

        # save validate
        validate.to_csv(f"./{folder_path}/1-2_validation_data.csv", mode="w")

        # Save test
        test.to_csv(f"./{folder_path}/1-3_testing_data.csv", mode="w")

    else:
        # save the dataframe with dummies in a csv for easy access
        original_df.to_csv(f"./{folder_path}/00_original_clean_no_dummies.csv", mode="w")

        # save the dataframe with dummies in a csv for easy access
        org_train_df.to_csv(f"./{folder_path}/01_original_clean_no_dummies_train.csv", mode="w")

        # save the dataframe with dummies in a csv for easy access
        encoded_scaled_df.to_csv(f"./{folder_path}/1-0_encoded_data.csv", mode="w")

        # save training data
        train.to_csv(f"./{folder_path}/1-1_training_data.csv", mode="w")

        # save validate
        validate.to_csv(f"./{folder_path}/1-2_validation_data.csv", mode="w")

        # Save test
        test.to_csv(f"./{folder_path}/1-3_testing_data.csv", mode="w")

    return "SIX data sets saved as .csv"



def scale_wine(train, validate, test):
    '''This function scales all of the features and returns the dataframes.'''

    # separate features to scale from the target
    feature_columns = train.iloc[:,:-4].columns # get the columns

    # repeat for three datasets
    x_features_train = train[feature_columns]
    x_features_validate = validate[feature_columns]
    x_features_test = test[feature_columns]

    # build a scaling object
    scaler = MinMaxScaler()

    # fit the scaler
    x_features_train = scaler.fit_transform(X=x_features_train)

    # transform the validate and test using the minMax object
    x_features_validate = scaler.transform(X=x_features_validate)
    x_features_test = scaler.transform(X=x_features_test)

    # New variable mames to add to data
    new_scale_col = []
    for i in feature_columns:
        new_scale_col.append(f"{i}_scaled")

    # convert to dataframes
    x_train_scaled = pd.DataFrame(x_features_train, columns=new_scale_col)
    x_validate_scaled = pd.DataFrame(x_features_validate, columns=new_scale_col)
    x_test_scaled = pd.DataFrame(x_features_test, columns=new_scale_col)

    # Reset the index before adding the two data frames together
    unscaled_columns = train.iloc[:,-4:].reset_index(drop=True)
    x_train_scaled.reset_index(drop=True, inplace=True)

    # concate the the original columns to this data set
    train = pd.concat([x_train_scaled, unscaled_columns], axis=1, ignore_index=False)


    # Reset the index for validate
    unscaled_columns_validate = validate.iloc[:,-4:].reset_index(drop=True)
    x_validate_scaled.reset_index(drop=True, inplace=True)

    # test
    unscaled_columns_test = test.iloc[:,-4:].reset_index(drop=True)
    x_test_scaled.reset_index(drop=True, inplace=True)

    # concate the the original columns to this data set
    validate = pd.concat([x_validate_scaled, unscaled_columns_validate], axis=1, ignore_index=False)
    test = pd.concat([x_test_scaled, unscaled_columns_test], axis=1, ignore_index=False)

    return train, validate, test



###################################### WRANGLE #########################################

def wrangle_wine_explore():

    '''This function acquires the data, prepares the data, and returns the train, validate, test dataframes.'''

    wine = get_wine_data()

    wine = prep_wine(wine)

    train, validate, test = split_wine(wine)

    return train, validate, test

def wrangle_wine_model():

    '''This function acquires the data, prepares the data, and returns the train, validate, test dataframes encoded and scaled for modeling.'''
    
    wine = get_wine_data()

    wine = prep_wine(wine)

    train, validate, test = split_wine(wine)
    
    train, validate, test = scale_wine(train, validate, test)

    return train, validate, test

    
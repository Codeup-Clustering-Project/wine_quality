# For funtion annotations
from binascii import a2b_qp
from typing import Union
from typing import Tuple

# data manipulation
import pandas as pd
import numpy as np

# data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# data separation/transformation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# system manipulation
import os
import sys
sys.path.append("./util_")
import acquire_

# other
import env
import warnings
warnings.filterwarnings("ignore")

#############################################################################################

# set a default them for all my visuals
sns.set_theme(style="whitegrid")

def wrangle_zillow() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    return the prepared 2017 single family clearn data in 5 different dta frames:
        train, validate, test all come from zillow_encoded_scaled and zillow is the original data with no scaling.
    """
    # sql query
    query = """
    SELECT bedroomcnt, 
            bathroomcnt,
            calculatedfinishedsquarefeet,
            taxvaluedollarcnt,
            yearbuilt,
            taxamount,
            fips
    FROM properties_2017
    WHERE propertylandusetypeid = 261 -- Single family home
    """

    # get existing csv data from the util directory
    zillow = acquire_.get_existing_csv_file_(fileName ="zillow_single_family")

   # rename dataframe columns
    zillow = zillow.rename(columns={"bedroomcnt":"bedrooms",
                        "bathroomcnt":"bathrooms",
                        "calculatedfinishedsquarefeet":"sqr_feet",
                        "taxvaluedollarcnt":"tax_value",
                        "yearbuilt":"year_built",
                        "taxamount":"tax_amount",
                        "fips":"county"})

    # drop all nulls in the dataframe
    zillow = zillow.dropna()

    # convert data type from float to int
    zillow.bedrooms = zillow.bedrooms.astype(int)
    zillow.year_built = zillow.year_built.astype(int)

    # remove the duplocated rows
    zillow = zillow.drop_duplicates(keep="first")

    # remove outliers
    zillow = zillow[zillow.bedrooms <= 7]
    zillow = zillow[zillow.bathrooms <= 7]
    zillow = zillow[zillow.year_built >= 1900]
    zillow = zillow[zillow.sqr_feet <= 5000]
    zillow = zillow[zillow.tax_amount <= 20000]

    # Rename the unique values in fips to county names
    zillow.county = zillow.county.astype(str).str.replace("6037.0","Los Angeles").str.replace("6059.0","Orange").str.replace("6111.0","Ventura")

    # create dummie variables
    dummies = pd.get_dummies(zillow.county)

    # clean dummie column names
    dummies_col = dummies.columns.str.replace(" ", "_").str.lower()

    # make a copy of my original data frame
    zillow_encoded_scaled = zillow.copy()

    # add dummies to my data frame
    zillow_encoded_scaled[dummies_col] = dummies

    # split the data into training, validation and testing sets
    train, validate, test = split_data_(df=zillow_encoded_scaled,
                        test_size=0.2, 
                        validate_size=0.2,
                        stratify_col= "county",
                        random_state=95)
    
    # scalable features
    features_to_scale = train[['bedrooms','bathrooms','sqr_feet','year_built','tax_amount']]

    # build a scaling object
    scaler = MinMaxScaler()

    # Note that we only call .fit with the training data,
    # but we use .transform to apply the scaling to all the data splits.
    # y Target values: this is not being changed in any way(no predictions are being made)
    x_train_scaled = scaler.fit_transform(X=features_to_scale,)

    # transfrom the validate and test using the minMax object
    x_val_scaled = scaler.transform(X=validate[features_to_scale.columns])
    x_test_scaled = scaler.transform(X=test[features_to_scale.columns])


    # New _ariable mames to add to data
    new_scale_col = []
    for i in features_to_scale.columns:
        new_scale_col.append(f"{i}_scaled")

    # convert to dataframe
    x_train_scaled = pd.DataFrame(x_train_scaled)
    x_val_scaled = pd.DataFrame(x_val_scaled)
    x_test_scaled = pd.DataFrame(x_test_scaled)

    # add new column names back to the data frame
    x_train_scaled[new_scale_col] = x_train_scaled
    x_val_scaled[new_scale_col] = x_val_scaled
    x_test_scaled[new_scale_col] = x_test_scaled

    # remove redundent columns from new dataframe
    x_train_scaled = x_train_scaled[new_scale_col]
    x_val_scaled = x_val_scaled[new_scale_col]
    x_test_scaled = x_test_scaled[new_scale_col]

    # train[x_train_scaled.columns] = x_train_scaled
    train = train.reset_index(drop=True)
    validate = validate.reset_index(drop=True)
    test = test.reset_index(drop=True)

    # Concatenate the DataFrames horizontally
    train = pd.concat([train, x_train_scaled], axis=1, verify_integrity=True)
    validate = pd.concat([validate, x_val_scaled], axis=1, verify_integrity=True)
    test = pd.concat([test, x_test_scaled], axis=1, verify_integrity=True)

    # drop the original columns
    train = train.drop(columns=features_to_scale)
    validate = validate.drop(columns=features_to_scale)
    test = test.drop(columns=features_to_scale)

    # save created data frames into csv
    save_split_data_(original_df=zillow,
                            encoded_scaled_df=zillow_encoded_scaled, 
                            train=train, 
                            validate=validate, 
                            test=test,
                            test_size=0.2,
                            stratify_col= "county",
                            random_state=95)

    return zillow, zillow_encoded_scaled, train, validate, test


#---------------------------------------------------------------
# Save visuals
def save_visuals_(fig: plt.figure ,viz_name:str= "unamed_viz", folder_name:int= 0, ) -> str:
    """
    Goal: Save a single visual into the project visual folder
    parameters:
        fig: seaborn visual figure to be saved
        viz_name: name of the visual to save
        folder_name: interger (0-7)represanting the section you are on in the pipeline
            0: all other (defealt)
            1: univariate stats
            2: bivariate stats
            3: multivariate stats
            4: stats test
            5: modeling
            6: final report
            7: presantation
    return:
        message to user on save status
    """
    project_visuals = "./00_project_visuals"
    folder_selection = {
        0: "00_non_specific_viz",
        1: "01_univariate_stats_viz",
        2: "02_bivariate_stats_viz",
        3: "03_multivariate_stats_viz",
        4: "04_stats_test_viz",
        5: "05_modeling_viz",
        6: "06_final_report_viz",
        7: "07_presantation"
    }

    # return error if user input for folder selection is not found
    if folder_name not in list(folder_selection.keys()):
        return f"{folder_name} is not a valid option for a folder name."
    # when folder location is found in selections
    else:
        # Specify the path to the directory where you want to save the figure
        folder_name = folder_selection[folder_name]
        directory_path = f'{project_visuals}/{folder_name}/'

        # Create the full file path by combining the directory path and the desired file name
        file_path = os.path.join(directory_path, f'{viz_name}.png')

        if os.path.exists(project_visuals): # check if the main viz folder exists
            if not os.path.exists(directory_path): # check if the folder name already exists
                os.makedirs(directory_path)
                # Save the figure to the specified file path
                fig.canvas.print_figure(file_path)

            else:
                # Save the figure to the specified file path
                fig.canvas.print_figure(file_path)
        else:
            # create both the project vis folder and the specific section folder
            os.makedirs(project_visuals)
            os.makedirs(directory_path)

            # Save the figure to the specified file path
            fig.canvas.print_figure(file_path)
    
    return f"Visual successfully saved in folder: {folder_name}"

# -----------------------------------------------------------------
# Save the splited data into separate csv files
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


# -----------------------------------------------------------------
# Split the data into train, validate and train
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
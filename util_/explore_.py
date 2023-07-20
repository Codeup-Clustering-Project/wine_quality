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
import itertools
import os
import sys
sys.path.append("./util_")
import prepare_
import explore_

# other
import env
import warnings
warnings.filterwarnings("ignore")

# set the random seed
np.random.seed(95)

# set a default them for all my visuals
sns.set_theme(style="whitegrid")
###############################################################
# This data is already been split and save
# This is only training data
train = pd.read_csv("./00_project_data/01_original_clean_no_dummies_train.csv", index_col=0)
train = train.reset_index(drop=True)

# find only columns with low count of categories
low_category_cols = train.nunique()[train.nunique() < 1000].index
low_category_cols

def plot_categorical_and_continuous_univariate():
    """
    Return: categorical and continuous univariate visuals.
    """
    # plot all the low category columns to see the distributions
    for col in low_category_cols[:-1]:
        print(col.upper())
        print("count of unique:",train[col].nunique())
        print(train[col].value_counts().sort_values())

        # plot
        fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(17,4))
        sns.countplot(data= train, x=col, ax=ax[0])
        sns.boxplot(data= train, x=col, ax=ax[1])
        sns.violinplot(data= train, x=col, ax=ax[2])
        plt.tight_layout()

        # save visual to file path
        explore_.save_visuals(fig=fig, viz_name=col, folder_name= 1)

        plt.show()

    plt.figure(figsize = (5,3))
    sns.countplot(data= train, x="county")

    fig = plt.gcf()

    # save visual to file path
    explore_.save_visuals(fig=fig, viz_name="county_univarate", folder_name= 1)

    # # Get a sample of the continious columns to fro ploting
    train_continious = train[["sqr_feet", "tax_value", "tax_amount"]]

    # plot all the low category columns to see the distributions
    for col in train_continious.columns:
        print(col.upper())
        print("count of unique:",train[col].nunique())
        # plot
        fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(17,4))
        sns.boxplot(data= train, x=col, ax=ax[0])
        sns.violinplot(data= train, x=col, ax=ax[1])
        sns.kdeplot(train, x=col, ax=ax[2])
        plt.tight_layout()
        
        # save visual to file path
        explore_.save_visuals(fig=fig, viz_name=col, folder_name= 1)

        plt.show()

def plot_categorical_and_continuous_bivariate():
    """
    return: categorical and continuous bivariate visuals.
    """

    # separeate discrete from continuous variables
    continuous_col = []
    categorical_col = []
    target = "tax_value"

    for col in train.columns:
        if col == target:
            pass
        elif train[col].dtype == "O":
            categorical_col.append(col)

        else:
            if len(train[col].unique()) < 20: #making anything with less than 4 unique values a catergorical value
                categorical_col.append(col)
            else:
                continuous_col.append(col)
                
    # Get a sample of the categorical columns for ploting
    train_continious = train[categorical_col].sample(100_000)

    # pairs of comninmations
    categorical_comb = list(itertools.product(categorical_col, ["tax_value"]))
    
    # Get a sample of the continious columns to fro ploting
    train_full_sample = train[train.columns].sample(100_000)

    # plot all the low category columns to see the distributions
    for col in categorical_comb:
        print(col[0].upper(), "VS", col[1].upper())
        # plot
        fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(17,4))
        sns.barplot(data= train, x=col[0] , y=col[1], ax= ax[0])
        sns.boxplot(data= train, x=col[0] , y=col[1], ax= ax[1])
        sns.stripplot(data= train, x=col[0] , y=col[1], ax= ax[2])
        plt.tight_layout()
        
        # save visual to file path
        explore_.save_visuals(fig=fig, viz_name=f"{col[0]}_vs_{col[1]}", folder_name= 2)
        plt.show()

    # Get a sample of the categorical columns for ploting
    train_continious = train[continuous_col].sample(50_000)

    # pairs of comninmations
    continuous_comb = list(itertools.product(continuous_col, ["tax_value"]))
    
    # plot all the low category columns to see the distributions

    for col in continuous_comb:
        print(col[0].upper(), "VS", col[1].upper())
        # plots
        fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(18,5))   

        sns.lineplot(data= train, x=col[0] , y=col[1], ax= ax[0])
        sns.scatterplot(data= train, x=col[0] , y=col[1], ax= ax[1])
        sns.kdeplot(data= train, x=col[0] , y=col[1], ax= ax[2])
        
        plt.tight_layout()
        
        # save visual to file path
        explore_.save_visuals(fig=fig, viz_name=f"{col[0]}_vs_{col[1]}", folder_name= 2)
        plt.show()


def plot_categorical_and_continuous_bivariate():
    """
    return: categorical and continuous milty variate visuals.
    """
    # get sample from training data
    train_full_sample = train[train.columns].sample(100_000)

    # plot
    grid = sns.PairGrid(train_full_sample, diag_sharey=False, hue="county")
    grid.map_upper(sns.scatterplot)
    grid.map_lower(sns.kdeplot)
    grid.map_diag(sns.kdeplot)

    # Get the current figure
    fig = plt.gcf()

    # save visual to file path
    explore_.save_visuals(fig=fig, viz_name= "pairplot_by_county", folder_name= 3)
    plt.show()

    for cat in categorical_col[:-1]:
        # figure
        fig, ax = plt.subplots(nrows=1, ncols=len(continuous_col), figsize=(18,5))
        
        for i in range(len(continuous_col)):
            # plot
            sns.lineplot(data= train_full_sample, x= cat, y= continuous_col[i], hue="county", ax=ax[i])
            plt.tight_layout()
            
        # save visual
        explore_.save_visuals(fig=fig, viz_name=f"{cat}_vs_{continuous_col[i]}_by_county", folder_name= 3)
        plt.show
        

##############################################################
#---------------------------------------------------------------
# Save visuals
def save_visuals(fig: plt.figure ,viz_name:str= "unamed_viz", folder_name:int= 0, ) -> str:
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

#---------------------------------------------------------------
# Used to verify p value for a corelations test
def verify_alpha_(p_value, alpha=0.05) -> None:
    """
    Goal: Test if we are acepting or rejecting the null
    """
    # oompare p-value to alpha
    if p_value < alpha:
        print("We have enough evidence to reject the null")
    else:
        print("we fail to reject the null at this time")
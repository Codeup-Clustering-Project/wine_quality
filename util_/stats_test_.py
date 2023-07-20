# data manipulation
import pandas as pd
import numpy as np

# data visualization
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

# data separation/transformation
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE # Recursive Feature Elimination¶

# statistics testing
import scipy.stats as stats

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

###################################################################
# This data is already been split and save
# This is only training data
train = pd.read_csv("./00_project_data/01_original_clean_no_dummies_train.csv", index_col=0)
train = train.reset_index(drop=True)

def mean_tax_value_vs_sqr_feet():
    """
    Return: visual to answer the following question
        Does the mean tax value differ significantly between properties with different square footage?
    """
    print("Null_Hyp: The mean tax value differ significantly between properties with different square footage")
    print("")
    print("Alt_Hyp: The mean tax value does not differ significantly between properties with different square footage")

    # Define the bin edges
    bin_edges = [500, 1000, 2000, 3000, 4000]

    # Create a new column with binned values
    train["sqr_teed_bin"] = pd.cut(train.sqr_feet, bins=bin_edges)

    # Group the data by the bin column and compute the mean value
    grouped_sqr_ft_by_mean_tax_value = train.groupby('sqr_teed_bin').tax_value.mean()
    grouped_sqr_ft_by_total_sqr_feet = train.groupby('sqr_teed_bin').sqr_feet.sum()

    # Plot the means
    fig, ax = plt.subplots(1,2,figsize = (10,4))
    grouped_sqr_ft_by_mean_tax_value.plot(kind='bar', ax=ax[0], color=['steelblue'] * (len(grouped_sqr_ft_by_mean_tax_value) - 1) + ['red'])
    ax[0].set_title('Mean Tax Value by Square Feet Bins')
    ax[0].set_xlabel('Square Feet Bins')
    ax[0].set_ylabel('Mean Tax Value')
    ax[0].set_xticklabels([f'{bins.left}-{bins.right}' for bins in grouped_sqr_ft_by_mean_tax_value.index], rotation=45)

    # plot the counts
    grouped_sqr_ft_by_total_sqr_feet.plot(kind="bar", ax=ax[1], color=['steelblue'] * (len(grouped_sqr_ft_by_total_sqr_feet) - 1) + ['red']) 
    ax[1].set_title('Total Square Feet by Square Feet Bins')
    ax[1].set_xlabel('Square Feet Bins')
    ax[1].set_ylabel('Total Square Feet')
    ax[1].set_xticklabels([f'{bins.left}-{bins.right}' for bins in grouped_sqr_ft_by_total_sqr_feet.index], rotation=45)

    # Change y-axis format to normal numbers
    ax[0].yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax[0].ticklabel_format(style='plain', axis='y')
    ax[1].yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax[1].ticklabel_format(style='plain', axis='y')

    plt.tight_layout()
    plt.show()


def mean_tax_value_vs_bed_bath_rooms():
    """
    Return: visual to answer the following question
        Is there a significant difference in the mean tax value between properties with low numbers of bedrooms or bathrooms against the properties with high numbers of bedrooms or bathrooms
    """
    print("Null_Hyp: There is a significant difference in the mean tax value between properties with low numbers of bedrooms or bathrooms against the properties with high numbers of bedrooms or bathrooms")
    print("")
    print("Alt_Hyp: There is no a significant difference in the mean tax value between properties with low numbers of bedrooms or bathrooms against the properties with high numbers of bedrooms or bathrooms")

    # Group the data by the bin column and compute the mean value
    grouped_bedrooms_by_mean_tax_value = train.groupby('bedrooms').tax_value.mean()
    grouped_bedrooms_by_total_bedroom_count = train.groupby('bedrooms').bedrooms.count()

    # Plot the values
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    # Bar plot for mean tax value
    grouped_bedrooms_by_mean_tax_value.plot(kind='bar', ax=ax[0], color=['steelblue'] * (len(grouped_bedrooms_by_mean_tax_value) - 2) + ['red', 'red'])
    ax[0].set_title('Mean Tax Value by Bedrooms')
    ax[0].set_xlabel('Bedrooms')
    ax[0].set_ylabel('Mean Tax Value')
    ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=0)

    # Bar plot for total bedroom count
    grouped_bedrooms_by_total_bedroom_count.plot(kind='bar', ax=ax[1], color=['steelblue'] * (len(grouped_bedrooms_by_total_bedroom_count) - 2) + ['red', 'red'])
    ax[1].set_title('Total Bedroom Count')
    ax[1].set_xlabel('Bedrooms')
    ax[1].set_ylabel('Total Bedroom Count')
    ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=0)

    plt.tight_layout()
    plt.show()

    # Group the data by the bin column and compute the mean value
    grouped_bathrooms_by_mean_tax_value = train.groupby('bathrooms').tax_value.mean()
    grouped_bathrooms_by_total_bedroom_count = train.groupby('bathrooms').bathrooms.count()

    # Plot the values
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))

    # Bar plot for mean tax value
    grouped_bathrooms_by_mean_tax_value.plot(kind='bar', ax=ax[0], color=['steelblue'] * (len(grouped_bathrooms_by_mean_tax_value.unique()) - 3) + ['red', 'red','red']) 
    ax[0].set_title('Mean Tax Value by Bathrooms')
    ax[0].set_xlabel('Bathrooms')
    ax[0].set_ylabel('Mean Tax Value')
    ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=0)

    # Bar plot for total bedroom count
    grouped_bathrooms_by_total_bedroom_count.plot(kind='bar', ax=ax[1], color=['steelblue'] * (len(grouped_bathrooms_by_total_bedroom_count) - 3) + ['red', 'red','red']) 
    ax[1].set_title('Total Bedroom Count')
    ax[1].set_xlabel('Bathrooms')
    ax[1].set_ylabel('Total Bedroom Count')
    ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=0)

    plt.tight_layout()
    plt.show()

def mean_tax_value_vs_county():
    """
    Return: visual to answer the following question
        Does the mean tax value differ significantly between properties in Los Angeles County compared to the other counties?
    """
    print("Null_Hyp: The mean tax value differ significantly between properties in Los Angeles County compared to the other counties?")
    print("")
    print("Alt_Hyp: The mean tax value does not differ significantly between properties in Los Angeles County compared to the other counties?")


    # Create a new column with binned values
    train['county_bin'] = train['county'].replace(['Ventura', 'Orange'], 'Other')

    # Group the data by the bin column and compute the mean value
    grouped_county_by_mean_tax_value = train.groupby('county_bin').tax_value.mean()
    grouped_county_by_total_property_count = train.groupby('county_bin')['county'].count()

    # Plot the values
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    # Bar plot for mean tax value
    colors_mean_tax_value = ['darkorange'] * (len(grouped_county_by_mean_tax_value) - 1) + ['red']
    grouped_county_by_mean_tax_value.plot(kind='bar', ax=ax[0], color=colors_mean_tax_value)
    ax[0].set_title('Mean Tax Value by County')
    ax[0].set_xlabel('County')
    ax[0].set_ylabel('Mean Tax Value')
    ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=0)

    # Bar plot for total property count
    colors_total_property_count = ['darkorange'] * (len(grouped_county_by_total_property_count) - 1) + ['red']
    grouped_county_by_total_property_count.plot(kind='bar', ax=ax[1], color=colors_total_property_count)
    ax[1].set_title('Total Property Count by County')
    ax[1].set_xlabel('County')
    ax[1].set_ylabel('Total Property Count')
    ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=0)

    plt.tight_layout()
    plt.show()

def mean_tax_value_vs_built_year():
    """
    Return: visual to answer the following question
        Is there a significant difference in the mean tax value between properties built in different years?
    """
    # Define the bin edges
    bin_edges = [1910, 1920, 1940, 1960, 1980, 2000, 2018]

    # Create a new column with binned values
    train["year_built_bin"] = pd.cut(train.year_built, bins=bin_edges)

    # Group the data by the bin column and compute the mean value
    grouped_year_by_mean_tax_value = train.groupby('year_built_bin').tax_value.mean()
    grouped_year_by_total_property_count = train.groupby('year_built_bin').year_built.count()

    # Plot the values
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))

    # Bar plot for mean tax value
    grouped_year_by_mean_tax_value.plot(kind='bar', ax=ax[0], color=['steelblue'] * (len(bin_edges) - 3) + ['red', 'red'])
    ax[0].set_title('Mean Tax Value by Year Built')
    ax[0].set_xlabel('Year Built Bins')
    ax[0].set_ylabel('Mean Tax Value')
    ax[0].set_xticklabels([f'{bins.left}-{bins.right}' for bins in grouped_year_by_mean_tax_value.index], rotation=45)

    # Bar plot for total property count
    grouped_year_by_total_property_count.plot(kind='bar', ax=ax[1], color=['steelblue'] * (len(bin_edges) - 3) + ['red', 'red'])
    ax[1].set_title('Total Property Count by Year Built')
    ax[1].set_xlabel('Year Built Bins')
    ax[1].set_ylabel('Total Property Count')
    ax[1].set_xticklabels([f'{bins.left}-{bins.right}' for bins in grouped_year_by_total_property_count.index], rotation=45)

    plt.tight_layout()
    plt.show()

###################################################################
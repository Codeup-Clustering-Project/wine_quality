# data manipulation
import pandas as pd
import numpy as np

# data visualization
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

# data separation/transformation
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE # Recursive Feature Elimination¶

# modeling
from sklearn.cluster import KMeans
import sklearn.preprocessing

from itertools import combinations
import math

import wrangle as w

###################################### EXPLORE #########################################

def feature_selection(train_scaled):
    '''This function separates the features from the target, makes and fits an RFS model,
        and returns a dataframe of the top 5 features.'''

    # separate feature columns
    feature_cols = train_scaled.columns[~train_scaled.columns.isin(["quality", "wine_clr"])]

    # separate features from target
    x_features = train_scaled[feature_cols]
    y_target = train_scaled["quality"]

    # make a model object to use in RFE process.
    linear_model = LinearRegression()

    # MAKE the RFE object
    rfe = RFE(linear_model, n_features_to_select=1)

    # FIT the RFE object to the training data
    rfe.fit(x_features, y_target)

    # ordered from most important to least important
    #rfe.ranking_

    # get a dataframe of the top 5 columns orderd by importance
    x_feature_selected = x_features.iloc[:, rfe.ranking_[0:5] - 1] # -1 beacuse rfe rank starts at 1

    return x_feature_selected

def volatile_cluster_model(x_feature_selected, train_scaled):
    '''This function'''

    # Get all continous columns for cobinations and leave the encoded columns
    columns_to_combine = x_feature_selected.columns[~x_feature_selected.columns.isin(["white"])]

    feature_combinations = list(combinations(columns_to_combine, 2))

    # set the max number ok of k to loop through
    # 5% of the data
    iter = math.ceil(len(train_scaled) * 0.005)

    model_centers = []
    model_inertia = []

    for k in range(1,iter + 1):
        # ceate model object
        kmean = KMeans(n_clusters= k) #

        # fit model object
        kmean.fit(train_scaled[list(feature_combinations[0])])

        # make predictions
        label = kmean.predict(train_scaled[list(feature_combinations[0])])
    
        # add predictions to the original dataframe
        train_scaled[f"clusters_{k}"] = label

        # view ceters
        model_centers.append(kmean.cluster_centers_)
    
        model_inertia.append(kmean.inertia_)

    # creat a dataframe of the best model k
    ceters = pd.DataFrame(model_centers[2], columns=list(feature_combinations[0]))

    return feature_combinations, ceters

###################################### VISUALIZE #############################################

def wine_color_quality_viz(train):
    '''This function visualizes the boxplot between red and white wines and quality score.'''

    sns.boxplot(x=train.wine_clr, y= train.quality, palette = "rocket")
   
    plt.ylabel("Quality")
    plt.xlabel("Wine Color")
   
    plt.title("Average Quality Score is the Same for Red or White Wine")
    plt.show()

def density_alcohol_viz(train):
    '''This function returns the scatterplot between density and alcohol'''

    sns.scatterplot(train, x ='density', y ='alcohol', hue='wine_clr', palette = "rocket")

    plt.ylabel("Alcohol Level")
    plt.xlabel("Density")
    plt.legend(title="Wine Color")

    plt.title("Strong Negative Relationship Between Density and Alcohol Levels")

    plt.show()


def volatileacid_density_viz(train):
    '''This function returns the scatterplot between volatile acidity and density'''

    sns.scatterplot(train, y ='volatile_acidity', x ='density', hue='wine_clr', palette = "rocket")

    plt.xlabel("Density")
    plt.ylabel("Volatile Acidity")
    plt.legend(title="Wine Color")

    plt.title("Good Case for Clustering with Volatile Acidity and Density")

    plt.show()

def dense_vol_centroids(train_scaled, feature_combinations, ceters):
    # vusialize the centroids
    plt.figure(figsize=(6,4))

    sns.scatterplot(data=train_scaled, x=feature_combinations[0][0], y=feature_combinations[0][1], hue="clusters_3")

    ceters.plot.scatter(x=feature_combinations[0][0], y=feature_combinations[0][1], c='orange', marker='x', s=100, label='centroid', ax=plt.gca())

    plt.title(f"{feature_combinations[0][0]} vs {feature_combinations[0][1]} cetroids")

def volatile_sulfur_viz(train):
    sns.scatterplot(train, x ='volatile_acidity', y ='free_sulfur_dioxide', hue='wine_clr', palette = "rocket")

    plt.ylabel("Free Sulfur Dioxide")
    plt.xlabel("Volatile Acidity")
    plt.legend(title="Wine Color")

    plt.title("Weak Relationship Between Volatile Acidity and Free Sulphur Dioxides")

    plt.show()


###################################### MODELING #############################################

def modeling_prep_and_baseline(train_scaled, val_scaled, test_scaled):
    '''This function bins the target, adds clusters, creates dummies for clusters,
        separates the features from the target, and calculates baseline. '''

        # separate low quality from high quality wine
    train_scaled['quality_bin'] = train_scaled.quality.astype(str).str.replace(r'\b[3-5]\b', '0',regex=True).str.replace(r'\b[6-9]\b', '1',regex=True).astype(int)
    val_scaled['quality_bin'] = val_scaled.quality.astype(str).str.replace(r'\b[3-5]\b', '0',regex=True).str.replace(r'\b[6-9]\b', '1',regex=True).astype(int)
    test_scaled['quality_bin'] = test_scaled.quality.astype(str).str.replace(r'\b[3-5]\b', '0',regex=True).str.replace(r'\b[6-9]\b', '1',regex=True).astype(int)


    # give the cluster valide names
    train_scaled.clusters_3 = train_scaled.clusters_3.astype(str).str.replace(
        "0","density and volatile acid (high, low)").str.replace(
        "1", "density and volatile acid (low, low)").str.replace(
        "2","density and volatile acid (high, high)")

    # apply to validate
    # give the cluster valide names
    val_scaled.clusters_3 = val_scaled.clusters_3.astype(str).str.replace(
        "0","density and volatile acid (high, low)").str.replace(
        "1", "density and volatile acid (low, low)").str.replace(
        "2","density and volatile acid (high, high)")

    # apply to validate
    # give the cluster valide names
    test_scaled.clusters_3 = test_scaled.clusters_3.astype(str).str.replace(
        "0","density and volatile acid (high, low)").str.replace(
        "1", "density and volatile acid (low, low)").str.replace(
        "2","density and volatile acid (high, high)")

    # get cluster dummies
    cluster_dummies = pd.get_dummies(train_scaled.clusters_3)
    val_cluster_dummies = pd.get_dummies(val_scaled.clusters_3)
    test_cluster_dummies = pd.get_dummies(test_scaled.clusters_3)

    # new cleaned column names
    cluster_col = cluster_dummies.columns.str.replace(" ","_").str.lower()

    # add the dummies to the dataframe
    train_scaled[cluster_col] = cluster_dummies
    val_scaled[cluster_col] = val_cluster_dummies
    test_scaled[cluster_col] = test_cluster_dummies

    # assign features and labels for the model
    xtrain = train_scaled[np.append(cluster_col, ["white", "free_sulfur_dioxide_scaled", 
                                              "alcohol_scaled", "density_scaled", "volatile_acidity_scaled"])]
    ytrain = train_scaled.quality_bin

    # validate
    xval = val_scaled[np.append(cluster_col, ["white","free_sulfur_dioxide_scaled", 
                                          "alcohol_scaled", "density_scaled", "volatile_acidity_scaled"])]
    yval = val_scaled.quality_bin

    # test
    xtest = test_scaled[np.append(cluster_col, ["white","free_sulfur_dioxide_scaled", 
                                          "alcohol_scaled", "density_scaled", "volatile_acidity_scaled"])]
    ytest = test_scaled.quality_bin

    # calculate and add bseline to the training data
    train_scaled["baseline"] = int(ytrain.mode())

    # baseline score
    baseline =accuracy_score( ytrain, train_scaled.baseline)

    print(f'Baseline Accuracy : {baseline}')

    return train_scaled, val_scaled, test_scaled
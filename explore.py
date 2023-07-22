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

# modeling
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay

# for knn
from sklearn.neighbors import KNeighborsClassifier

# for decision tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree

# for random forest
from sklearn.ensemble import RandomForestClassifier

# for logistic regression
from sklearn.linear_model import LogisticRegression

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
    columns_to_combine = x_feature_selected.columns[~x_feature_selected.columns.isin(["white", "clusters_3", "clusters_2"])]

    feature_combinations = list(combinations(columns_to_combine, 2))

    # Create dataframe to save mode predictions
    model_df = pd.DataFrame({"index_col":np.arange(len(train_scaled))})
    model_df2 = pd.DataFrame({"index_col":np.arange(len(train_scaled))})

    # set the max number ok of k to loop through
    # 5% of the data
    iter = math.ceil(len(train_scaled) * 0.005)

    model_centers = []
    model_inertia = []
    model_centers2 = []
    model_inertia2 = []

    for k in range(1,iter + 1):
        # ceate model object
        kmean = KMeans(n_clusters= k)
        kmean2 = KMeans(n_clusters= k)

        # fit model object
        kmean.fit(train_scaled[list(feature_combinations[0])])
        kmean2.fit(train_scaled[list(feature_combinations[2])])

        # make predictions
        label = kmean.predict(train_scaled[list(feature_combinations[0])])
        label2 = kmean2.predict(train_scaled[list(feature_combinations[2])])
    
        # add predictions to the original dataframe
        model_df[f"clusters_{k}"] = label
        model_df2[f"clusters_{k}"] = label2

        # view ceters
        model_centers.append(kmean.cluster_centers_)
        model_centers2.append(kmean2.cluster_centers_)
    
        model_inertia.append(kmean.inertia_)
        model_inertia2.append(kmean2.inertia_)


    # creat a dataframe of the best model k
    ceters = pd.DataFrame(model_centers[2], columns=list(feature_combinations[0]))

    # creat a dataframe of the best model k
    ceters2 = pd.DataFrame(model_centers2[1], columns=list(feature_combinations[2]))

    return feature_combinations, ceters, ceters2, model_df, model_df2

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

def dense_vol_centroids(train_scaled, feature_combinations, ceters, model_df):
    # vusialize the centroids
    plt.figure(figsize=(6,4))

    sns.scatterplot(data=train_scaled, x=feature_combinations[0][0], y=feature_combinations[0][1], hue=model_df.clusters_3)

    ceters.plot.scatter(x=feature_combinations[0][0], y=feature_combinations[0][1], c='orange', marker='x', s=400, label='centroid', ax=plt.gca())

    plt.title(f"{feature_combinations[0][0]} vs {feature_combinations[0][1]} cetroids")

    plt.show()

def volatile_sulfur_viz(train):
    sns.scatterplot(train, x ='volatile_acidity', y ='free_sulfur_dioxide', hue='wine_clr', palette = "rocket")

    plt.ylabel("Free Sulfur Dioxide")
    plt.xlabel("Volatile Acidity")
    plt.legend(title="Wine Color")

    plt.title("Weak Relationship Between Volatile Acidity and Free Sulphur Dioxides")

    plt.show()

def density_alc_clusters(train_scaled, feature_combinations, ceters2, model_df2):
    '''This function visualizes density/ alcohol clusters.'''

    # vusialize the centroids
    plt.figure(figsize=(6,4))
    
    sns.scatterplot(data=train_scaled, x=feature_combinations[2][0], y=feature_combinations[2][1], hue=model_df2.clusters_3)
    
    ceters2.plot.scatter(x=feature_combinations[2][0], y=feature_combinations[2][1], c='white', marker='x', s=400, label='centroid', ax=plt.gca())
    
    plt.title(f"{feature_combinations[2][0]} vs {feature_combinations[2][1]} cetroids")
    
    plt.show()


###################################### MODELING #############################################

def modeling_prep(train_scaled, val_scaled, test_scaled, feature_combinations, model_df, model_df2):
    '''This function bins the target, adds clusters, creates dummies for clusters,
        separates the features from the target, and calculates baseline. '''

   # ceate model object
    kmean = KMeans(n_clusters= 3)
    kmea_cls_2 = KMeans(n_clusters= 3)

    # fit model object
    kmean.fit(train_scaled[list(feature_combinations[0])])
    kmea_cls_2.fit(train_scaled[list(feature_combinations[2])])

    # make predictions
    val_label = kmean.predict(val_scaled[list(feature_combinations[0])])
    test_label = kmean.predict(test_scaled[list(feature_combinations[0])])

    val_label2 = kmea_cls_2.predict(val_scaled[list(feature_combinations[2])])
    test_label2 = kmea_cls_2.predict(test_scaled[list(feature_combinations[2])])

    train_scaled["dens_valAcid_cluster"] = model_df.clusters_3
    val_scaled["dens_valAcid_cluster"] = val_label
    test_scaled["dens_valAcid_cluster"] = test_label

    train_scaled["dens_alc_cluster"] = model_df2.clusters_3
    val_scaled["dens_alc_cluster"] = val_label2
    test_scaled["dens_alc_cluster"] = test_label2

    # separate low quality from high quality wine
    train_scaled['quality_bin'] = train_scaled.quality.astype(str).str.replace(r'\b[3-5]\b', '0',regex=True).str.replace(r'\b[6-9]\b', '1',regex=True).astype(int)
    val_scaled['quality_bin'] = val_scaled.quality.astype(str).str.replace(r'\b[3-5]\b', '0',regex=True).str.replace(r'\b[6-9]\b', '1',regex=True).astype(int)
    test_scaled['quality_bin'] = test_scaled.quality.astype(str).str.replace(r'\b[3-5]\b', '0',regex=True).str.replace(r'\b[6-9]\b', '1',regex=True).astype(int)

    # give the cluster valide names
    train_scaled.dens_valAcid_cluster = train_scaled.dens_valAcid_cluster.astype(str).str.replace(
        "0","density and volatile acid (high, low)").str.replace(
        "1", "density and volatile acid (low, low)").str.replace(
        "2","density and volatile acid (high, high)")

    # apply to validate
    # give the cluster valide names
    val_scaled.dens_valAcid_cluster = train_scaled.dens_valAcid_cluster.astype(str).str.replace(
        "0","density and volatile acid (high, low)").str.replace(
        "1", "density and volatile acid (low, low)").str.replace(
        "2","density and volatile acid (high, high)")

    # apply to validate
    # give the cluster valide names
    test_scaled.dens_valAcid_cluster = train_scaled.dens_valAcid_cluster.astype(str).str.replace(
        "0","density and volatile acid (high, low)").str.replace(
        "1", "density and volatile acid (low, low)").str.replace(
        "2","density and volatile acid (high, high)")


        # high density and high alcohol
    # give the cluster valide names
    train_scaled.dens_alc_cluster = train_scaled.dens_alc_cluster.astype(str).str.replace(
        "0","density and alcohol (high, high)").str.replace(
        "1", "density and alcohol (high, low)").str.replace(
        "2","density and alcohol (mid, mid)")

    # apply to validate
    # give the cluster valide names
    val_scaled.dens_alc_cluster = train_scaled.dens_alc_cluster.astype(str).str.replace(
        "0","density and alcohol (high, high)").str.replace(
        "1", "density and alcohol (high, low)").str.replace(
        "2","density and alcohol (mid, mid)")

    # apply to validate
    # give the cluster valide names
    test_scaled.dens_alc_cluster = train_scaled.dens_alc_cluster.astype(str).str.replace(
        "0","density and alcohol (high, high)").str.replace(
        "1", "density and alcohol (high, low)").str.replace(
        "2","density and alcohol (mid, mid)")

        # get cluster dummies
    cluster_dummies = pd.get_dummies(train_scaled[["dens_valAcid_cluster","dens_alc_cluster"]])
    val_cluster_dummies = pd.get_dummies(val_scaled[["dens_valAcid_cluster","dens_alc_cluster"]])
    test_cluster_dummies = pd.get_dummies(test_scaled[["dens_valAcid_cluster","dens_alc_cluster"]])

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

    return train_scaled, val_scaled, test_scaled, xtrain, ytrain, xval, yval, xtest, ytest

def model_baseline(train_scaled, ytrain):
    '''This function returns the baseline for modeling'''

    # calculate and add bseline to the training data
    train_scaled["baseline"] = int(ytrain.mode())

    # baseline score
    baseline =accuracy_score( ytrain, train_scaled.baseline)

    return train_scaled, baseline

def knn_model(xtrain, ytrain, xval, yval, baseline):

    # the maximun number of neighbors the model should look at
    # in my case it can only look at 1% of the data
    k_neighbors = math.ceil(len(xtrain) * 0.01)

    # the final result metric
    metrics = []

    for k in range(1, k_neighbors + 1):
        # create a knn object
        #                          n_neighborsint(default=5) 
        knn = KNeighborsClassifier(n_neighbors=k, weights='uniform', p=2)
        #                                                        p=1 uses the manhattan distance

        # fit training data to the object
        knn = knn.fit(xtrain, ytrain)
    
        #USE the thing
        train_score= knn.score(xtrain, ytrain)
        validate_score = knn.score(xval, yval)
    
        # create a dictionary of scores
        output = {
            "k": k,
            "train_score": train_score,
            "validate_score": validate_score,
            "difference": train_score - validate_score,
            "train_baseline_diff": baseline - train_score,
            "baseline_accuracy": baseline,
        }
    
        metrics.append(output)

    # get the result as a dataframe
    knn_model_df = pd.DataFrame(metrics)

    return knn_model_df

def decTree_model(xtrain, ytrain, xval, yval,baseline):

    metrics = []
    for d in range(1,11):
    #      create tree object
        treeClf = DecisionTreeClassifier(max_depth= d, random_state=95)
    
        # fit model
        treeClf = treeClf.fit(xtrain, ytrain)
    
        # train accurecy score
        train_score = treeClf.score(xtrain, ytrain)
        validate_score = treeClf.score(xval, yval)
    
        # create a dictionary of scores
        output = {
            "depth": d,
            "train_score": train_score,
            "validate_score": validate_score,
            "difference": train_score - validate_score,
            "train_baseline_diff": baseline - train_score,
            "baseline_accuracy": baseline,
        }
    
        metrics.append(output)

    decTree_model_df = pd.DataFrame(metrics)

    return decTree_model_df

def randFor_model(xtrain, ytrain, xval, yval,baseline):

    metrics = []

    for trees in range(2,20):

        # create ramdom tree object
        randFor = RandomForestClassifier(n_estimators= 100, min_samples_leaf= trees, max_depth = trees, random_state=95 )
    
        # fit the model
        randFor = randFor.fit(xtrain, ytrain)
    
        # get accuracy scores
        train_score = randFor.score(xtrain, ytrain)
        validate_score = randFor.score(xval, yval)
    
        # create a dictionary of scores
        output = {
            "trees": trees,
            "train_score": train_score,
            "validate_score": validate_score,
            "difference": train_score - validate_score,
            "train_baseline_diff": baseline - train_score,
            "baseline_accuracy": baseline,
        }
    
        metrics.append(output)

    randFor_model_df = pd.DataFrame(metrics)

    return randFor_model_df

def logReg_model(xtrain, ytrain, xval, yval,baseline):

    metrics = []

    for c in np.arange(0.0001,0.1, 0.01):
    
        # create ramdom tree object
        logReg = LogisticRegression(C=c, random_state=95, max_iter= 1000)
    
        # fit the model
        logReg = logReg.fit(xtrain, ytrain)
    
        # get accuracy scores
        train_score = logReg.score(xtrain, ytrain)
        validate_score = logReg.score(xval, yval)
    
        # create a dictionary of scores
        output = {
            "c": c,
            "train_score": train_score,
            "validate_score": validate_score,
            "difference": train_score - validate_score,
            "train_baseline_diff": baseline - train_score,
            "baseline_accuracy": baseline,
        }
    
        metrics.append(output)

    # get the result as a dataframe
    logReg_model_df = pd.DataFrame(metrics)

    return logReg_model_df


def test_decTree(xtrain, ytrain, xval, yval, xtest, ytest, baseline):

    metrics = []

    # create tree object
    treeClf = DecisionTreeClassifier(max_depth= 3, random_state=95)

    # fit model
    treeClf = treeClf.fit(xtrain, ytrain)

    # predict train
    ypred = treeClf.predict(xtrain)

    # train accurecy score
    train_score = treeClf.score(xtrain, ytrain)
    validate_score = treeClf.score(xval, yval)
    test_score = treeClf.score(xtest, ytest)

    # create a dictionary of scores
    output = {
        "depth": 3,
        "train_score": train_score,
        "validate_score": validate_score,
        "test_score": test_score,
        "train_test_diff": train_score - test_score,
        "test_baseline_diff": baseline - test_score,
        "baseline_accuracy": baseline,
    }
    metrics.append(output)

    # get the result as a dataframe
    decTree_model_df = pd.DataFrame(metrics)

    return decTree_model_df
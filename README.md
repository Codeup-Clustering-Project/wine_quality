# California Wine Quality Project

# Project Description
This project aims to build a model using a variety of features to predict the quality score of wines.

# Project Goal
  * Determine what features in the dataset are related to wine quality score.
  * Use these findings to create a model to predict the wine quality score.
  * Share findings on the model with the data science team.

# Initial Thoughts
My initial thoughts are that property value will be dependent on the counties that they are located in since the home value is the tax assessed value. 

# The Plan
  * Acquire data from Data.World
    * Original dataset combined from two separate dataframes into one
    * 6497 rows and 13 columns in final dataframe
    
  * Prepare data
    * Rename columns
    * No null values in the data  
    * Remove 1177 duplicated rows
    * Create dummie varaibles (wine_clr)
    * Split data into train, validate, and test. (60/20/20 split)
    * Scale all columns except the target (quality) and the encoded (wine_clr)
      
  * Explore the data
    * Use Recursive Feature Elimination to determine best features for wine quality
    * Answer the questions:
        * Does the average quality score differ between red or white wines?
        * Is there a relationship between volatile acidity and density?
        * What does clustering show us about this correlation?
        * Is there a relationship between density and alcohol level?
        * Is there a relationship between volatile acidity and free sulfur dioxide?
        
  * Develop a model to predict wine quality score
    * Use accuracy as my evaluation metric.
    * Baseline will be the mode of quality score.
    * Target variable was binned to improve model performance.
        * Low Score Wine: Quality Score of 5 and Lower
        * High Score Wine: Quality Score of 6 and Higher
   
  * Make conclusions.

# Data Dictionary
|**Feature**|**Description**|
|:-----------|:---------------|
|Fixed Acidity |The total concentration of non-volatile acids in wine, contributing to its overall tartness and structure.|
|Volatile Acidity | The presence of volatile acids in wine, which in excessive amounts can lead to undesirable vinegar-like off-flavors.|
|Citric Acid |A specific type of acid naturally occurring in grapes and sometimes used in winemaking to enhance acidity levels|
|Residual Sugar | The amount of sugar remaining in the wine after fermentation, influencing its perceived sweetness. |
|Chlorides| Salts of hydrochloric acid that can impact the taste and mouthfeel of wine in small quantities.|
|Free Sulfur Dioxide|The form of sulfur dioxide that is not bound to other compounds, used as a preservative in winemaking to prevent oxidation and microbial spoilage. |
|Total Sulfur Dioxide|The combined measurement of free and bound sulfur dioxide in wine, representing the overall concentration of this preservative. |
|Density| The mass of the wine relative to the volume, which can help determine the alcohol content in some cases.|
|pH|A measure of the wine's acidity or alkalinity, affecting its stability and microbial activity. |
|Sulphate| A form of sulfur that occurs naturally in grapes and can impact the fermentation process and wine stability. |
|Alcohol| The ethyl alcohol content in the wine, resulting from the fermentation of sugars by yeast.|
|Quality| A subjective assessment of a wine's overall excellence, taking into account various sensory attributes, balance, and complexity.|

# Steps to Reproduce
  * Clone this repo
  * Acquire the data
  * Put the data in the same file as cloned repo
  * Run the final_report notebook

# Conclusions
  * 

# Next Steps
   * 

# Recommendations
  * 
  
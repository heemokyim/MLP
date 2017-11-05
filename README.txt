PROGRAM INFORMATION:
Zillow Logerror Prediction with MLP in Python 3

Authors: Viyaleta Peterson, Muhammad Wasim Ud Din, Karan Manoj Bhosaale
University of Bridgeport, CPCS 552 - Data Mining, Pr. Jeongkyu Lee, Fall 2017

To Implement:
1. Save all the components into a single folder on your local machine
2. Run the analysis.py file

Description:
This program provides a module to implement MLP on a Zillow
Kaggle competition dataset to predict the quality of Zestimate algorithm.
The quality of algorithm is provided by the logerror which
represents the logarithmic error between the Zestimate real estate value
prediction and the actual sales price of the real estate.
The competition and original dataset may be found here:
https://www.kaggle.com/c/zillow-prize-1

Components:
> MLP.py: python module with classes and methods which allow construction of a
multilayer perceptron architecture with specified hyperparameters, training,
testing, saving and loading the network, and displaying verbose and visual results.
> implementation.py: implementation of MLP module on the merged Kaggle dataset
> merged_data.csv*: merged dataset representing a join between two datasets from
the original competition website: properties_2016.csv and train_2016_v2.csv

Python Dependencies:
time, csv, pickle, numpy, math, matplotlib, PIL, random, sklearn, pandas
(the libraries above may have additional dependencies)



"""
The following represents the code used to merge the original datasets.
Python 3.6.
"""

import pandas as pd
logerror = pd.read_csv('train_2016_v2.csv')
temp_list = []
chunksize = 10**5
df = pd.DataFrame(columns=['parcelid', 'logerror', 'transactiondate', 'airconditioningtypeid', 'architecturalstyletypeid', 'basementsqft', 'bathroomcnt', 'bedroomcnt', 'buildingclasstypeid', 'buildingqualitytypeid', 'calculatedbathnbr', 'decktypeid', 'finishedfloor1squarefeet', 'calculatedfinishedsquarefeet', 'finishedsquarefeet12', 'finishedsquarefeet13', 'finishedsquarefeet15', 'finishedsquarefeet50', 'finishedsquarefeet6', 'fips', 'fireplacecnt', 'fullbathcnt','garagecarcnt','garagetotalsqft', 'hashottuborspa', 'heatingorsystemtypeid', 'latitude', 'longitude', 'lotsizesquarefeet', 'poolcnt', 'poolsizesum', 'pooltypeid10', 'pooltypeid2', 'pooltypeid7', 'propertycountylandusecode', 'propertylandusetypeid', 'propertyzoningdesc', 'rawcensustractandblock', 'regionidcity', 'regionidcounty', 'regionidneighborhood', 'regionidzip', 'roomcnt', 'storytypeid', 'threequarterbathnbr', 'typeconstructiontypeid', 'unitcnt', 'yardbuildingsqft17', 'yardbuildingsqft26','yearbuilt', 'numberofstories', 'fireplaceflag', 'structuretaxvaluedollarcnt', 'taxvaluedollarcnt', 'assessmentyear', 'landtaxvaluedollarcnt', 'taxamount', 'taxdelinquencyflag', 'taxdelinquencyyear', 'censustractandblock'])
for chunk in pd.read_csv('properties_2016.csv', chunksize=chunksize, low_memory=False):
    df_temp = pd.merge(logerror, chunk, on='parcelid', how='inner')
    df = df.append(df_temp)
df.to_csv('merged_data.csv', sep=',')

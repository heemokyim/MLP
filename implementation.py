"""
PROGRAM INFORMATION: Zillow Logerror Prediction with MLP

Authors:        Viyaleta Peterson
                Muhammad Wasim Ud Din
                Karan Manoj Bhosaale

Institution:    University of Bridgeport
                CPCS 552 - Data Mining
                Pr. Jeongkyu Lee
                Fall 2017

Description:    This program implements the MLP library on the Kaggle Zillow
                Competition dataset in order to predict the logerror provided
                various parameters describing a real estate property. Logerror
                is calculated as the log of an error between Zillow's Zestimate
                house prediction value and the actual sales price of the house
                given a set of properties in transactions for 2016.
                This implementation performs the following:
                    1. Data Load: load data from csv file into dataframe
                    2. Preprocessing:
                        a. Extract output as list and drop from dataframe
                        b. Drop attributes with > 50 percent missing values
                        c. Drop attributes with > 50 correlation
                        d. Perform ordinal encoding and max-min on categorical attributes
                        e. Perform max-min on numerical attributes
                        f. Split data into training (80%) and testing (20%) set
                        g. Min-max the output and save parameters
                    3. Training:
                        a. Set up neural network and specify learning rate
                        b. Initialize the neural network (default uniform random
                           weights and bias in range [0, 1])
                        c. Train the network on training dataset with
                           specified hyperparameters
                        d. Save the network architecture
                    4. Testing:
                        a. Use the network architecture to test the testing dataset
                        b. Convert predicted values via reverse max-min with
                           previously saved max-min parameters
                        c. Calculate the Mean Absolute Error (MAE)
                           NOTE: MAE is used by Zillow on Kaggle to evaluate results
                        d. Plot predicted vs. actual values for visual evaluation

Notes:          - The traing and testing datasets are preprocessed all at once
                  (this might cause some bias in the results)
                - This implementation produces a text file with saved network
                  parameters and information. This network can be loaded for a
                  later implementation with MLP.load('my_network') method.
                - Implementation of MOMENTUM is in the works for the future version


Dependencies:   MLP: self-developed library to train and test a MLP network
                sklearn.model_selection: for splitting training and testing data
                pandas: to load the data and do several preprocessing steps
                numpy: to work with arrays of data and access math fucntions
                    NOTE: This implementation requires numpy+mkl wheel
                matplotlib: to plot the predicted vs. actual output

"""
from MLP import Network, Layer
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
print('> Libraries successfully loaded.')



'''DATA LOAD'''
df = pd.read_csv('merged_data.csv')
print('> Data Successfully Loaded.')



'''PREPROCESSING'''
# Extracting logerror into output y list
y = df['logerror'].tolist()
# Dropping attributes (attributes with > 50% missing values or attributes which are too correlated among each other)
df = df.drop([  #drop attributes with > 50% missing values
    'i', 'parcelid', 'logerror',
    'transactiondate',
    'buildingclasstypeid',
    'finishedsquarefeet13',
    'storytypeid',
    'basementsqft',
    'yardbuildingsqft26',
    'fireplaceflag',
    'architecturalstyletypeid',
    'typeconstructiontypeid',
    'finishedsquarefeet6',
    'decktypeid',
    'poolsizesum',
    'pooltypeid10',
    'pooltypeid2',
    'taxdelinquencyyear',
    'taxdelinquencyflag',
    'hashottuborspa',
    'yardbuildingsqft17',
    'finishedsquarefeet15',
    'finishedsquarefeet50',
    'finishedfloor1squarefeet',
    'fireplacecnt',
    'threequarterbathnbr',
    'pooltypeid7',
    'poolcnt',
    'numberofstories',
    'airconditioningtypeid',
    'garagetotalsqft',
    'garagecarcnt',
    'regionidneighborhood'], axis=1)
df = df.drop([   #drop highly correlated columns
    'bathroomcnt',
    'bedroomcnt',
    'calculatedfinishedsquarefeet',
    'finishedsquarefeet12',
    'fullbathcnt',
    'latitude',
    'longitude',
    'rawcensustractandblock',
    'regionidcounty',
    'roomcnt',
    'taxvaluedollarcnt',
    'landtaxvaluedollarcnt',
    'taxamount',
    'censustractandblock',
    'assessmentyear',
    'propertycountylandusecode',
    'propertyzoningdesc'
    ], axis=1)

# Remaining attribute columns
cats = ['buildingqualitytypeid', 'fips', 'heatingorsystemtypeid', 'propertylandusetypeid']  #categorical attributes
nums = ['calculatedbathnbr', 'lotsizesquarefeet', 'unitcnt', 'yearbuilt', 'structuretaxvaluedollarcnt', 'regionidcity', 'regionidzip']  #numerical attributes

# ORDINAL ENCODING for Categorical Attributes for equally spaced orders in range [0,1]
for a in cats:
    df[a] = df[a].fillna(df[a].median())                        #replace missing with median
    n_unique = df[a].nunique()                                  #count unqiue values
    for i, u in enumerate(df[a].unique()):
        df[a] = df[a].replace(to_replace=u, value=i/n_unique)  #replace categories with number in [0,1], evenly spaced between all categories

# MIN-MAX SCALING for Numerical Attributes
for a in nums:
    df[a] = df[a].fillna(df[a].median())  #replace missing with median
    df[a] = (df[a] - min(df[a])) / (max(df[a]) - min(df[a]))  #min-max scaling
X = df.values

# TRAIN-TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)

# MIN-MAX Output SCALING
min_y, max_y = min(y_train), max(y_train)  #keeping track of min and max to unscale the predicted output
y_train = [(y-min_y)/(max_y-min_y) for y in y_train]
print('> Data Preprocessed and Scaled')



'''NEURAL NETWORK'''
# Architecture, activation functions, and learning rate setup
net = Network(
        architecture = [Layer(11),
                        Layer(11, 'tanh'),
                        Layer(1, 'sigmoid')],
        l_rate = 0.1 )
print('> Network setup.')
net.initialize()  #network initilization
print('> Network initialized.')

# Training 
trial_error = net.train(X_train, y_train, epochs=50, verbose=True, plot=True, decay=0.1, decay_rate=10)
print('> Training complete.')

# Saving the network to a file for future use
net.save('my_network')



'''TESTING'''
y_predicted = net.predict(X_test)
y_predicted = [y[0]*(max_y-min_y)+min_y for y in y_predicted]  #rescale the output using the original configuration

# Calculate Testing Set Mean Absolute Error
mae = np.sum([abs(y_predicted[i] - y_test[i]) for i in range(len(y_test))]) / len(y_test)
print('> Test MSE:', mae)

# Scatter Plot of the actual and predicted values
plt.scatter(y_test, y_predicted, s=2)
plt.plot( [min(y_test),max(y_test)],[min(y_test),max(y_test)] , c='blue', lw=0.5)
plt.title('Testing Results')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.show()

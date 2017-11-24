"""
PROGRAM INFORMATION: Zillow Logerror Prediction with BRANE (BRanching Artificial Neural Ensemble)

Motivation:     BRANE architecture supports mutliple representations of same data to be trained using multiple sensor
                multilayer perceptrons' outputs as inputs of a decision multilayer perceptron. Backpropagation is
                performed on individual MLPs but not through the entire architecture. For example,
                two representations of data may involve numerical datapoints and binary flags if the
                datapoints are odd numbers. Another example of such binary flags would identify if the
                datapoints are missing in the original dataset and have been filled with mean, median, or 0 values.
                This program uses the latter approach on a dataset which has a large number of missing values.
                Alternatively, two different datasets with same outcome value can be used in BRANE (for example,
                a picture of an object and other data pertaining to it (like weight, for example); this
                may be especially useful in medical diagnostic imaging applications).

Authors:        Viyaleta Peterson
                Muhammad Wasim Ud Din
                Karan Manoj Bhosaale

Institution:    University of Bridgeport
                CPCS 552 - Data Mining
                Pr. Jeongkyu Lee
                Fall 2017

Description:    This program implements the BRANE architecture to predict the
                logerror provided by Zillow in their Kaggle Zestimate competition,
                where the logerror represents the logarithmic error between the
                actual sale price of a house and Zillow's Zestimate.
                The dataset features 2016 transactions in a region of Southern
                California. Many data rows are missing values. Logerror exhibits
                Gaussian distribution.

                Program Contents:
                1. Data Load:
                    a. load data from merged_data.py csv
                    b. split data into training and testing
                2. Preprocessing:
                    a. drop attributes that have > 50 percent missing data
                    b. drop attributes that are highly correlated
                    c. generate a dataframe of flags where 1 represents the
                       missing value and 0 represents the non-missing value
                    d. for categorical attributes:
                        i. perform ordinal encoding such that values are spaced
                           equally in [0,1] range
                    e. for numerical attributes:
                        i. scale the values using min-max in [0,1] range
                    f. turn dataframes into lists
                    g. create a lambda function which will scale and unscale the
                       output for pre and post processing
                       * the output will be pre and post processed right before
                         and after the backpropagation
                3. Network Setup:
                    a. Generate an instance of Network object for
                        i.   numerical sensor MLP with sigmoid activation function
                             for hidden and output layers with learning rate 0.001
                             with 11 input, 11 hidden, and 1 output neurons
                        ii.  binary sensor MLP with sigmoid activation function
                             for hidden and output layers with learning rate 0.001
                             with 11 input, 4 hidden, and 1 output neurons
                        iii. decision MLP with linear activation function with
                             learning rate 0.001 and 2 input, 0 hidden, and 1
                             output neurons
                    b.  Initialize the network instances
                4. Training:
                    a. Configure the number of epochs
                    b. Iterate through epochs and records
                    c. For each record in each epoch:
                        i.   Feed forward numerical sensor MLP using numerical dataset
                        ii.  Feed forward binary sensor MLP using binary dataset
                        iii. Feed forward decision MLP using the outputs of sensor MLPs
                        iv.  Calculate the error between the output and min-maxed expected value
                        v.   Scale the expected value to use for backpropagation
                        vi.  Backpropagate the numerical sensor MLP
                        vii. Backpropagate the binary sensor MLP
                        viii.Backpropagate the decision MLP
                5. Testing:
                    a. For each data row in testing dataset:
                        i.   Feed forward the numerical sensor MLP
                        ii.  Feed forward the binary sensor MLP
                        iii. Feed forward the decision MLP
                        iv.  Calculate the testing error between the expected and
                             predicted values
                    * Alternatively, this process can be performed on the dataset
                      where the expected values are not provided

Dependencies:   MLP: self-developed library to train and test the network
                pandas: to load the data and do several preprocessing steps
                numpy: to work with arrays of data and access math fucntions
                    NOTE: This implementation requires numpy+mkl wheel
                matplotlib: to plot the predicted vs. actual output

"""
from MLP import Network, Layer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')


"""DATA LOAD"""
# Load data from csv
df = pd.read_csv('merged_data.csv')
# Train-test split 67/33
training = np.random.rand(len(df)) < 0.67 #generate random numbers representing 67% of total data
df_train = df[training]  #include training from original dataset
df_test = df[~training]  #exclude training from original dataset
y_train = df_train['logerror'].tolist()  #split outputs into lists
y_test = df_test['logerror'].tolist()


'''PREPROCESS'''
# Drop attributes (> 50% missing or highly correlated), predetermined via analysis
df_train = df_train.drop([  #drop attributes with > 50% missing values
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
df_train = df_train.drop([   #drop highly correlated columns
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
df_test = df_test.drop([  #drop attributes with > 50% missing values
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
df_test = df_test.drop([   #drop highly correlated columns
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

# Generate a missing data flags dataset
df_train_missing = df_train.isnull().astype('int')
df_test_missing = df_test.isnull().astype('int')

# Remaining attribute columns
cats = ['buildingqualitytypeid', 'fips', 'heatingorsystemtypeid', 'propertylandusetypeid']  #categorical attributes
nums = ['calculatedbathnbr', 'lotsizesquarefeet', 'unitcnt', 'yearbuilt', 'structuretaxvaluedollarcnt', 'regionidcity', 'regionidzip']  #numerical attributes

# ORDINAL ENCODING for Categorical Attributes for equally spaced orders in range [0,1]
for a in cats:
    df_train[a] = df_train[a].fillna(df_train[a].median())      #replace missing with median
    df_test[a] = df_test[a].fillna(df_test[a].median())
    n_unique_train = df_train[a].nunique()                      #count unqiue values
    n_unique_test = df_test[a].nunique()
    for i, u in enumerate(df_train[a].unique()):
        df_train[a] = df_train[a].replace(to_replace=u, value=i/n_unique_train)  #replace categories with number in [0,1], evenly spaced between all categories
    for i, u in enumerate(df_test[a].unique()):
        df_test[a] = df_test[a].replace(to_replace=u, value=i/n_unique_test)  #replace categories with number in [0,1], evenly spaced between all categories

# MIN-MAX SCALING for Numerical Attributes
for a in nums:
    df_train[a] = df_train[a].fillna(df_train[a].median())  #replace missing with median
    df_test[a] = df_test[a].fillna(df_test[a].median())
    df_train[a] = (df_train[a] - min(df_train[a])) / (max(df_train[a]) - min(df_train[a]))  #min-max scaling
    df_test[a] = (df_test[a] - min(df_test[a])) / (max(df_test[a]) - min(df_test[a]))

# Turn training and testing datasets into lists
X_train_numerical = df_train.values
X_train_binary = df_train_missing.values
X_test_numerical = df_test.values
X_test_binary = df_test_missing.values

# Helper functions for pre and post processing the output
min_y, max_y = min(y_train), max(y_train)
min_max_y = lambda a: (a-min_y) / (max_y-min_y)
reverse_y = lambda a: a * (max_y - min_y) + min_y


""" NETWORK SETUP """
# Create the network models: 2 sensor, 1 decision
# Note: existing architecture may be loaded using the load method from MLP library
numerical_sensor = Network(architecture=[Layer(11), Layer(11, 'sigmoid'), Layer(1, 'sigmoid')], l_rate=0.001) #Numerical Sensor
binary_sensor = Network(architecture=[Layer(11), Layer(4, 'sigmoid'), Layer(1, 'sigmoid')], l_rate=0.001) #Binary Sensor
decision = Network(architecture=[Layer(2), Layer(1, 'linear')], l_rate=0.001)  #Decision MLP

# Initialize the networks (this generates the initial weights)
numerical_sensor.initialize()
binary_sensor.initialize()
decision.initialize()

# Uncomment below for a small batch test
# X_train_numerical = X_train_numerical[:600]
# X_train_binary = X_train_binary[:600]
# y_train = y_train[:600]
# X_test_numerical = X_test_numerical[:300]
# X_test_binary = X_test_binary[:300]
# y_test = y_test[:300]


""" TRAINING """
# Initialize training epoch and errors
epochs = 100
MSE_train = []  #mean squared error for training

# Iterate through epochs
for e in range(epochs):
    epoch_mse_train = 0
    # Iterate through data records
    for i in range(len(y_train)):
        # 1. Feed the numerical sensor MLP
        numerical_sensor.architecture[0].values = X_train_numerical[i]
        numerical_sensor.forward_propagate()
        # 2. Feed the binary sensor MLP
        binary_sensor.architecture[0].values = X_train_binary[i]
        binary_sensor.forward_propagate()
        # 3. Feed the decision MLP
        decision.architecture[0].values = [numerical_sensor.architecture[-1].values[0], binary_sensor.architecture[-1].values[0]]
        decision.forward_propagate()

        # Get the error - uses the min-maxed y value
        epoch_mse_train += (y_train[i] - reverse_y(decision.architecture[-1].values[0]))**2 / 2

        # Prepare output for backpropagation by min-maxing it
        scaled_y_train = min_max_y(y_train[i])

        # 1. Back propagate numerical sensor MLP
        numerical_sensor.backward_propagate([scaled_y_train])
        # 2. Back propagate binary sensor MLP
        binary_sensor.backward_propagate([scaled_y_train])
        # 3. Back propagate decision MLP
        decision.backward_propagate([scaled_y_train])

    MSE_train.append(epoch_mse_train / len(y_train))  #add the error to the list of errors
    print('EPOCH: {}, MSE: {}'.format(e, MSE_train[-1]))

# Uncomment the following three lines to save the network architecture.
# numerical_sensor.save(name='numerical_sensor_arch')
# binary_sensor.save(name='binary_sensor_arch')
# decision.save(name='decision_arch')


""" TESTING """
y_predicted = []
MSE_test = []
for i in range(len(y_test)):
    # 1. Feed the numerical sensor MLP
    numerical_sensor.architecture[0].values = X_test_numerical[i]
    numerical_sensor.forward_propagate()

    # 2. Feed the binary sensor MLP
    binary_sensor.architecture[0].values = X_test_binary[i]
    binary_sensor.forward_propagate()

    # 3. Feed the decision MLP
    decision.architecture[0].values = [numerical_sensor.architecture[-1].values[0], binary_sensor.architecture[-1].values[0]]
    decision.forward_propagate()

    # Get the error
    y_predicted.append(reverse_y(numerical_sensor.architecture[-1].values[0]))
    MSE_test.append((y_test[i] - reverse_y(numerical_sensor.architecture[-1].values[0]))**2 / 2)

print('TEST MSE: {}'.format(np.mean(MSE_test)))

# Display testing results in a scatter plot
plt.scatter(y_test, y_predicted, s=5)
plt.plot( [min(y_test),max(y_test)],[min(y_test),max(y_test)] , c='blue', lw=0.5) #line of reference
plt.ylabel('Predicted Output')
plt.xlabel('Actual Output')
plt.title('BRANE Results for {} records'.format(len(y_test)))
plt.show()

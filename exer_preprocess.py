import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from termcolor import colored, cprint
#################################################################
# Steps:
# 1. Read a dataset from excel, csv file.
# 2. Check how many NaN values are there in the dataset.
# 3. Simplest way is the delete those records from the dataset
# 4. Incase any columns has too much variance, do log normalization of 
#   that column. It is mainly representing the column in logerithmic 
#   term.
# 5. Incase column values are too big, we should scale down the values too.
#   Or if different features have different unit, then scaling helps.
# 6. Split the data into feature data and target data
# 7. Split the data info training and test data - Use random state and stratify for better result
# 8. Fit the training data into the model.
# 9. Now use the test data to find the scores of the model.
#################################################################
print (colored("######### Find the BMI for all dogs and add in the dataset ########", "cyan", attrs=['bold']))
print (colored("########################", "cyan", attrs=['bold']))
print (colored("Read data from json file", "cyan", attrs=['bold']))
print (colored("########################", "cyan", attrs=['bold']))
hiking = pd.read_json("hiking.json")
print (hiking.head())

wine = pd.read_csv("wine.csv")
print (wine.head())

ufo_data = pd.read_csv("ufo_data.csv")
print (ufo_data.head())

print (colored("#################################################", "cyan", attrs=['bold']))
print (colored("Find the columns and datatype info of the dataset", "cyan", attrs=['bold']))
print (colored("#################################################", "cyan", attrs=['bold']))
print (hiking.info())
print (wine.info())
print (ufo_data.info())

print (colored("##########################################", "cyan", attrs=['bold']))
print (colored("Find the statistics summary of the dataset", "cyan", attrs=['bold']))
print (colored("##########################################", "cyan", attrs=['bold']))
print (wine.describe())

print (colored("##########################################", "cyan", attrs=['bold']))
print (colored("Drop the data which is having NULL Value  ", "cyan", attrs=['bold']))
print (colored("##########################################", "cyan", attrs=['bold']))
print (colored("########## Original Data #################", "cyan", attrs=['bold']))
print (ufo_data.info())
print (colored("########## Modified Data #################", "cyan", attrs=['bold']))
modified_ufo_data = ufo_data.dropna()
print (modified_ufo_data.info())

print (colored("###############################################", "cyan", attrs=['bold']))
print (colored("Drop the whole column of data having NULL Value", "cyan", attrs=['bold']))
print (colored("###############################################", "cyan", attrs=['bold']))
print (colored("########## Original Data #################", "cyan", attrs=['bold']))
print (hiking.info())
print (colored("########## Modified Data #################", "cyan", attrs=['bold']))
modified_hiking = hiking.drop("lat",axis=1)
print (modified_hiking.info())

print (colored("##################################################", "cyan", attrs=['bold']))
print (colored("## Find the data which is having NULL value     ##", "cyan", attrs=['bold']))
print (colored("##################################################", "cyan", attrs=['bold']))

print (colored("######### In Hiking dataset ######################", "cyan", attrs=['bold']))
print (hiking.isna().sum())
print (colored("########## In Wine dataset #######################", "cyan", attrs=['bold']))
print (wine.isna().sum())
print (colored("########## In UFO dataset ########################", "cyan", attrs=['bold']))
print (ufo_data.isna().sum())

print (colored("########## Modified Data in Hiking Dataset #################", "cyan", attrs=['bold']))
print ("## Original Data ##")
print (hiking)
print (colored("## After deleting the rows having NA value in Difficulty column ##", "cyan", attrs=['bold']))
modified_hiking = hiking.dropna(subset=["Difficulty"])
print (modified_hiking)

print (colored("## After deleting the rows having NA value in Difficulty column ##", "cyan", attrs=['bold']))
print (colored("## Original Data ##", "cyan", attrs=['bold']))
print (ufo_data)
print (colored("## Only keep rows with at least 10 non-NaN values##", "cyan", attrs=['bold']))
print (ufo_data.dropna(thresh=10))

print (colored("#################################", "cyan", attrs=['bold']))
print (colored("## Read the data from datafile ##", "cyan", attrs=['bold']))
print (colored("#################################", "cyan", attrs=['bold']))
volunteer = pd.read_csv("vol_data.csv")
print (volunteer.head())

print (colored("##########################################", "cyan", attrs=['bold']))
print (colored("## How many missing data in the dataset ##", "cyan", attrs=['bold']))
print (colored("##########################################", "cyan", attrs=['bold']))
print (volunteer.isna().sum())

print (colored("#######################################################################################", "cyan", attrs=['bold']))
print (colored("## Drop the Latitude and Longitude columns from volunteer, storing as volunteer_cols ##", "cyan", attrs=['bold']))
print (colored("#######################################################################################", "cyan", attrs=['bold']))
volunteer_cols = volunteer.drop(["Latitude","Longitude"], axis=1)
print (volunteer_cols)

print (colored("################################################################################################################################################", "cyan", attrs=['bold']))
print (colored("## Subset volunteer_cols by dropping rows containing missing values in the category_desc, and store in a new variable called volunteer_subset ##", "cyan", attrs=['bold']))
print (colored("################################################################################################################################################", "cyan", attrs=['bold']))
volunteer_subset = volunteer_cols.dropna(subset=["category_desc"])
print (volunteer_subset)

print (colored("##################################", "cyan", attrs=['bold']))
print (colored("## Find the datatype of dataset ##", "cyan", attrs=['bold']))
print (colored("##################################", "cyan", attrs=['bold']))
print (volunteer.info())

print (volunteer.dtypes)

print (colored("#######################################################", "red", attrs=['bold']))
print (colored("## Drop the whole row is  category_desc value is NaN ##", "red", attrs=['bold']))
print (colored("#######################################################", "red", attrs=['bold']))
print (volunteer["category_desc"].value_counts())

print (colored("###########################################################", "red", attrs=['bold']))
print (colored("## Find the different catagories of value in this cloumn ##", "red", attrs=['bold']))
print (colored("###########################################################", "red", attrs=['bold']))
volunteer.dropna(subset=["category_desc"],inplace=True)

print (colored("########################################################", "red", attrs=['bold']))
print (colored("## Split the data between target and feature variable ##", "red", attrs=['bold']))
print (colored("########################################################", "red", attrs=['bold']))
y = volunteer[["category_desc"]]
print (y)
X = volunteer.drop(["category_desc"], axis=1)
print (X)

print (colored("####################################################", "red", attrs=['bold']))
print (colored("## Split the data between train and test variable ##", "red", attrs=['bold']))
print (colored("####################################################", "red", attrs=['bold']))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, stratify = y)

print (colored("##################################", "red", attrs=['bold']))
print (colored("## Target values from test data ##", "red", attrs=['bold']))
print (colored("##################################", "red", attrs=['bold']))
print (y_test["category_desc"].value_counts())

print (colored("###################################", "red", attrs=['bold']))
print (colored("## Target values from train data ##", "red", attrs=['bold']))
print (colored("###################################", "red", attrs=['bold']))
print (y_train["category_desc"].value_counts())

print (colored("########################################################", "blue", attrs=['bold']))
print (colored("## Split the data between target and feature variable ##", "blue", attrs=['bold']))
print (colored("########################################################", "blue", attrs=['bold']))
X = wine.drop(["Type"],axis=1)
y = wine[["Type"]]
print (X)
print (y)

print (colored("####################################################", "blue", attrs=['bold']))
print (colored("## Split the data between train and test variable ##", "blue", attrs=['bold']))
print (colored("####################################################", "blue", attrs=['bold']))
X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=42, stratify=y)

print (colored("#######################", "blue", attrs=['bold']))
print (colored("## Test feature data ##", "blue", attrs=['bold']))
print (colored("#######################", "blue", attrs=['bold']))
print (X_test)
print (colored("#######################", "blue", attrs=['bold']))
print (colored("## Test target data  ##", "blue", attrs=['bold']))
print (colored("#######################", "blue", attrs=['bold']))
print (y_test)
print (colored("#######################", "blue", attrs=['bold']))
print (colored("## Train feature data ##", "blue", attrs=['bold']))
print (colored("#######################", "blue", attrs=['bold']))
print (X_train)
print (colored("#######################", "blue", attrs=['bold']))
print (colored("## Train target data ##", "blue", attrs=['bold']))
print (colored("#######################", "blue", attrs=['bold']))
print (y_train)

print (colored("####################################", "blue", attrs=['bold']))
print (colored("## Instantiate the KNN model      ##", "blue", attrs=['bold']))
print (colored("## Fit training data in the model ##", "blue", attrs=['bold']))
print (colored("## Find the score from that model ##", "blue", attrs=['bold']))
print (colored("####################################", "blue", attrs=['bold']))
knn = KNeighborsClassifier()
knn.fit(X_train,np.ravel(y_train))
print(colored("Without log normalization, model score {}". format(knn.score(X_test,y_test)), "red", attrs=['bold']))

print (colored("###########################################", "blue", attrs=['bold']))
print (colored("## Find the variance of feature variable ##", "blue", attrs=['bold']))
print (colored("###########################################", "blue", attrs=['bold']))
print (X.var())

print (colored("#############################", "blue", attrs=['bold']))
print (colored("## After log normalization ##", "blue", attrs=['bold']))
print (colored("#############################", "blue", attrs=['bold']))
X["Proline"] = np.log(X["Proline"])
print (X.head())

print (colored("####################################################", "blue", attrs=['bold']))
print (colored("## Split the data between train and test variable ##", "blue", attrs=['bold']))
print (colored("####################################################", "blue", attrs=['bold']))
X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=42, stratify=y)

print (colored("####################################", "blue", attrs=['bold']))
print (colored("## Instantiate the KNN model      ##", "blue", attrs=['bold']))
print (colored("## Fit training data in the model ##", "blue", attrs=['bold']))
print (colored("## Find the score from that model ##", "blue", attrs=['bold']))
print (colored("####################################", "blue", attrs=['bold']))
knn = KNeighborsClassifier()
knn.fit(X_train,np.ravel(y_train))
print(colored("Without log normalization, model score {}". format(knn.score(X_test,y_test)), "red", attrs=['bold']))

print (colored("############################################", "blue", attrs=['bold']))
print (colored("## Find standard deviation of all columns ##", "blue", attrs=['bold']))
print (colored("############################################", "blue", attrs=['bold']))
print (wine.std())
print (wine[["Ash", "Alcalinity of ash", "Magnesium"]].agg('max'))

print (colored("##############################################################", "blue", attrs=['bold']))
print (colored("## Scale data                                               ##", "blue", attrs=['bold']))
print (colored("## Mainly to scale down the data                            ##", "blue", attrs=['bold']))
print (colored("## This reduces the variance of every column significantly  ##", "blue", attrs=['bold']))
print (colored("##############################################################", "blue", attrs=['bold']))
scalar = StandardScaler()
wine_subset = wine[["Ash", "Alcalinity of ash", "Magnesium"]]
wine_scaled = pd.DataFrame(scalar.fit_transform(wine_subset), columns=wine_subset.columns)
print (wine_scaled)

print (colored("##################################", "blue", attrs=['bold']))
print (colored("## Variance after sscaling data ##", "blue", attrs=['bold']))
print (colored("##################################", "blue", attrs=['bold']))
print (wine_scaled.var())


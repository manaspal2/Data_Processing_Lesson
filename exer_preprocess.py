import pandas as pd
import numpy as np
from termcolor import colored, cprint

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




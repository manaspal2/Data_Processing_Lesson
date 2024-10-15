import re
import sys
import pandas as pd
import numpy as np
from termcolor import colored, cprint
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

print (colored("#################################", "green", attrs=['bold']))
print (colored(" Step 1: Read data from json file", "green", attrs=['bold']))
print (colored("#################################", "green", attrs=['bold']))
ufo = pd.read_csv("ufo_data.csv")
print (ufo.head())
print (ufo.info())
print (colored(" Shape of the dataframe", "green", attrs=['bold']))
print (ufo.shape)

print (colored("############################################", "green", attrs=['bold']))
print (colored(" Step 2: Find how many NaN Value per column", "green", attrs=['bold']))
print (colored("############################################", "green", attrs=['bold']))
print (ufo.isna().sum())

print (colored("############################################", "green", attrs=['bold']))
print (colored(" Step 3: Drop the rows having NaN values", "green", attrs=['bold']))
print (colored("############################################", "green", attrs=['bold']))
ufo.dropna(inplace=True)
# Few strategies to test here. Some of values of the seconds columns are 0.
# 1. Those are replaced with NaN value
ufo['seconds'] = ufo['seconds'].replace({'0':np.nan, 0:np.nan})
ufo.dropna(inplace=True)
# 2. Those are replaced with mean value
# print ("Mean of column seconds:", ufo['seconds'].mean())
# ufo['seconds'] = ufo['seconds'].replace(0, ufo['seconds'].mean())
print (ufo.head())
print (ufo.info())
print (colored(" Shape of the dataframe", "green", attrs=['bold']))
print (ufo.shape)

print (colored("############################################", "green", attrs=['bold']))
print (colored(" Step 4: Convert the date to datetime format", "green", attrs=['bold']))
print (colored("############################################", "green", attrs=['bold']))
ufo['date'] = pd.to_datetime(ufo['date'])
print (ufo.head())
print (ufo.info())

print (colored("###################################################", "green", attrs=['bold']))
print (colored(" Step 5: Convert the length_of_time to integer     ", "green", attrs=['bold']))
print (colored("###################################################", "green", attrs=['bold']))
def return_minutes(time_string):
    num = re.search('\d+', time_string)
    if num is not None:
        print (num.group(0))
        return int(num.group(0))
ufo.loc[:, 'minutes'] = ufo['length_of_time'].apply(return_minutes)
print(ufo[["length_of_time", "minutes"]].head())
# Some of the minutes values are NaN as a result need to drop some more rows
print (ufo.isna().sum())
ufo.dropna(inplace=True)
#ufo.to_csv('ufo_mod_Info.csv')

# Few strategies to test here. Some of values of the seconds columns are 0.
# 1. Those are replaced with NaN value
ufo['minutes'] = ufo['minutes'].replace({'0':np.nan, 0:np.nan})
ufo.dropna(inplace=True)
# 2. Those are replaced with mean value
# print ("Mean of column seconds:", ufo['seconds'].mean())
# ufo['seconds'] = ufo['seconds'].replace(0, ufo['seconds'].mean())

print (colored("###################################################", "green", attrs=['bold']))
print (colored("## Step 6: Variance of feature variables ##", "green", attrs=['bold']))
print (colored("###################################################", "green", attrs=['bold']))
print (ufo['seconds'].var(), ufo['minutes'].var())

ufo["seconds_log"] = np.log(ufo["seconds"])
print (ufo[['seconds', 'seconds_log']])
print (ufo['seconds_log'].var())
print (colored("###################################################", "green", attrs=['bold']))
ufo["minutes_log"] = np.log(ufo["minutes"])
print (ufo[['minutes', 'minutes_log']])
print (ufo['minutes_log'].var())

print (colored("###################################################", "green", attrs=['bold']))
print (colored(" Step 7: Check the unique values from two columns ", "green", attrs=['bold']))
print (colored("###################################################", "green", attrs=['bold']))
# Find the unique values of country type
print(ufo['country'].unique())
ufo["country_enc"] = ufo["country"].apply(lambda val:1 if val == 'us' else 0)
print(ufo['country_enc'].unique())
# Find the unique values of column type
print(ufo['type'].unique())
type_set = pd.get_dummies(ufo["type"])
ufo = pd.concat([ufo, type_set], axis=1)
print (ufo.head())

# Extract the month, year from the date column
ufo["month"] = ufo["date"].dt.month
ufo["year"] = ufo["date"].dt.year
print(ufo[["date", "month", "year"]].head())

print (colored("############################################", "green", attrs=['bold']))
print (colored("## Instantiate the tfidf vectorizer object  ", "green", attrs=['bold']))
print (colored("############################################", "green", attrs=['bold']))
vec = TfidfVectorizer()
desc_tfidf = vec.fit_transform(ufo["desc"])
print(desc_tfidf.shape)
print(type(desc_tfidf))

print(type(desc_tfidf))
vocab = {v:k for k,v in vec.vocabulary_.items()}
#print (vocab)

print (colored("####################################", "green", attrs=['bold']))
print (colored("## Make a list of features to drop  ", "green", attrs=['bold']))
print (colored("####################################", "green", attrs=['bold']))
to_drop = ["city", "country", "lat", "long", "state", "date", "recorded", 
           "seconds", "minutes", "desc", "length_of_time"]
ufo = ufo.drop(to_drop, axis=1)
print (ufo.head())

######################################################################
# FOLLOWING PART OF THE CODE IS PROBABLY INCORRECT, NEED CLARIFICATION
# df1 = pd.DataFrame(desc_tfidf.toarray())
# df2 = ufo
# newDf = pd.concat([df1, df2], axis = 1)
# print (newDf)
# newDf.to_csv("ufo_mod_info.csv")

# y = newDf["country_enc"]
# X = newDf.drop(["country_enc", "type"], axis=1)

# X_train, X_test, y_train, y_test = train_test_split(newDf, 
#                                                     y, 
#                                                     stratify=y, 
#                                                     random_state=42)

# print (colored("###############################", "green", attrs=['bold']))
# print (colored("## Fit nb to the training sets ", "green", attrs=['bold']))
# print (colored("###############################", "green", attrs=['bold']))
# nb = GaussianNB()
# nb.fit(X_train, y_train)

# print (colored("###########################################", "green", attrs=['bold']))
# print (colored("## Print the score of nb on the test sets  ", "green", attrs=['bold']))
# print (colored("###########################################", "green", attrs=['bold']))
# print (nb.score(X_test, y_test))

#np.set_printoptions(threshold=sys.maxsize)
#combined_data = np.hstack((desc_tfidf.toarray(), ufo))
#print (combined_data)
#print (type(combined_data))

#df = pd.DataFrame(combined_data)
#print (df.head())
################################################################

# Find the top 4 words as the last parameter from vectorized texts
def return_weights(vocab, original_vocab, vector, vector_index, top_n):
    zipped = dict(zip(vector[vector_index].indices, vector[vector_index].data))
    # Transform that zipped dict into a series
    zipped_series = pd.Series({vocab[i]:zipped[i] for i in vector[vector_index].indices})
    # Sort the series to pull out the top n weighted words
    zipped_index = zipped_series.sort_values(ascending=False)[:top_n].index
    return [original_vocab[i] for i in zipped_index]

def words_to_filter(vocab, original_vocab, vector, top_n):
    filter_list = []
    for i in range(0, vector.shape[0]): 
        # Call the return_weights function and extend filter_list
        filtered = return_weights(vocab, original_vocab, vector, i, top_n)
        filter_list.extend(filtered)
    
    # Return the list in a set, so we don't get duplicate word indices
    return set(filter_list)
filtered_words = words_to_filter(vocab, vec.vocabulary_, desc_tfidf, 4)
#print (filtered_words)

print (colored("#############################################", "green", attrs=['bold']))
print (colored("## Split the dataset into feature and target ", "green", attrs=['bold']))
print (colored("#############################################", "green", attrs=['bold']))
X = ufo.drop(["country_enc", "type"], axis=1)
print (X)
X.to_csv("ufo_mod_info.csv")
y = ufo["country_enc"]
print (y)

print (colored("#############################################", "green", attrs=['bold']))
print (colored("## Split the dataset into train and test     ", "green", attrs=['bold']))
print (colored("#############################################", "green", attrs=['bold']))
X_train, X_test, y_train, y_test = train_test_split(ufo.drop(["country_enc", "type"], axis=1), 
        y, 
        random_state=42,
        stratify=y)
print (X_train.head())
print (X_test.head())
print (y_train.head())
print (y_test.head())

print (colored("################################", "green", attrs=['bold']))
print (colored("## Fit knn to the training sets ", "green", attrs=['bold']))
print (colored("################################", "green", attrs=['bold']))
knn = KNeighborsClassifier()
knn.fit(X_train,y_train)

print (colored("####################################", "green", attrs=['bold']))
print (colored("## Find the score from test dataset ", "green", attrs=['bold']))
print (colored("####################################", "green", attrs=['bold']))
print(knn.score(X_test, y_test))

print (colored("######################################################################", "green", attrs=['bold']))
print (colored("## Use the list of filtered words we created to filter the text vector", "green", attrs=['bold']))
print (colored("######################################################################", "green", attrs=['bold']))
filtered_text = desc_tfidf[:, list(filtered_words)]
#print (filtered_text)

print (colored("####################################", "green", attrs=['bold']))
print (colored("## Split the X and y sets           ", "green", attrs=['bold']))
print (colored("####################################", "green", attrs=['bold']))
X_train, X_test, y_train, y_test = train_test_split(filtered_text.toarray(), 
                                                    y, 
                                                    stratify=y, 
                                                    random_state=42)

print (colored("###############################", "green", attrs=['bold']))
print (colored("## Fit nb to the training sets ", "green", attrs=['bold']))
print (colored("###############################", "green", attrs=['bold']))
nb = GaussianNB()
nb.fit(X_train, y_train)

print (colored("###########################################", "green", attrs=['bold']))
print (colored("## Print the score of nb on the test sets  ", "green", attrs=['bold']))
print (colored("###########################################", "green", attrs=['bold']))
print (nb.score(X_test, y_test))
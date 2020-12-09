#!/usr/bin/env python
# coding: utf-8

# # Starbucks Capstone Challenge
# 
# ### Introduction
# 
# This data set contains simulated data that mimics customer behavior on the Starbucks rewards mobile app. Once every few days, Starbucks sends out an offer to users of the mobile app. An offer can be merely an advertisement for a drink or an actual offer such as a discount or BOGO (buy one get one free). Some users might not receive any offer during certain weeks. 
# 
# Not all users receive the same offer, and that is the challenge to solve with this data set.
# 
# Your task is to combine transaction, demographic and offer data to determine which demographic groups respond best to which offer type. This data set is a simplified version of the real Starbucks app because the underlying simulator only has one product whereas Starbucks actually sells dozens of products.
# 
# Every offer has a validity period before the offer expires. As an example, a BOGO offer might be valid for only 5 days. You'll see in the data set that informational offers have a validity period even though these ads are merely providing information about a product; for example, if an informational offer has 7 days of validity, you can assume the customer is feeling the influence of the offer for 7 days after receiving the advertisement.
# 
# You'll be given transactional data showing user purchases made on the app including the timestamp of purchase and the amount of money spent on a purchase. This transactional data also has a record for each offer that a user receives as well as a record for when a user actually views the offer. There are also records for when a user completes an offer. 
# 
# Keep in mind as well that someone using the app might make a purchase through the app without having received an offer or seen an offer.
# 
# ### Example
# 
# To give an example, a user could receive a discount offer buy 10 dollars get 2 off on Monday. The offer is valid for 10 days from receipt. If the customer accumulates at least 10 dollars in purchases during the validity period, the customer completes the offer.
# 
# However, there are a few things to watch out for in this data set. Customers do not opt into the offers that they receive; in other words, a user can receive an offer, never actually view the offer, and still complete the offer. For example, a user might receive the "buy 10 dollars get 2 dollars off offer", but the user never opens the offer during the 10 day validity period. The customer spends 15 dollars during those ten days. There will be an offer completion record in the data set; however, the customer was not influenced by the offer because the customer never viewed the offer.
# 
# ### Cleaning
# 
# This makes data cleaning especially important and tricky.
# 
# You'll also want to take into account that some demographic groups will make purchases even if they don't receive an offer. From a business perspective, if a customer is going to make a 10 dollar purchase without an offer anyway, you wouldn't want to send a buy 10 dollars get 2 dollars off offer. You'll want to try to assess what a certain demographic group will buy when not receiving any offers.
# 
# ### Final Advice
# 
# Because this is a capstone project, you are free to analyze the data any way you see fit. For example, you could build a machine learning model that predicts how much someone will spend based on demographics and offer type. Or you could build a model that predicts whether or not someone will respond to an offer. Or, you don't need to build a machine learning model at all. You could develop a set of heuristics that determine what offer you should send to each customer (i.e., 75 percent of women customers who were 35 years old responded to offer A vs 40 percent from the same demographic to offer B, so send offer A).

# # Data Sets
# 
# The data is contained in three files:
# 
# * portfolio.json - containing offer ids and meta data about each offer (duration, type, etc.)
# * profile.json - demographic data for each customer
# * transcript.json - records for transactions, offers received, offers viewed, and offers completed
# 
# Here is the schema and explanation of each variable in the files:
# 
# **portfolio.json**
# * id (string) - offer id
# * offer_type (string) - type of offer ie BOGO, discount, informational
# * difficulty (int) - minimum required spend to complete an offer
# * reward (int) - reward given for completing an offer
# * duration (int) - time for offer to be open, in days
# * channels (list of strings)
# 
# **profile.json**
# * age (int) - age of the customer 
# * became_member_on (int) - date when customer created an app account
# * gender (str) - gender of the customer (note some entries contain 'O' for other rather than M or F)
# * id (str) - customer id
# * income (float) - customer's income
# 
# **transcript.json**
# * event (str) - record description (ie transaction, offer received, offer viewed, etc.)
# * person (str) - customer id
# * time (int) - time in hours since start of test. The data begins at time t=0
# * value - (dict of strings) - either an offer id or transaction amount depending on the record
# 
# **Note:** If you are using the workspace, you will need to go to the terminal and run the command `conda update pandas` before reading in the files. This is because the version of pandas in the workspace cannot read in the transcript.json file correctly, but the newest version of pandas can. You can access the termnal from the orange icon in the top left of this notebook.  
# 
# You can see how to access the terminal and how the install works using the two images below.  First you need to access the terminal:
# 
# <img src="pic1.png"/>
# 
# Then you will want to run the above command:
# 
# <img src="pic2.png"/>
# 
# Finally, when you enter back into the notebook (use the jupyter icon again), you should be able to run the below cell without any errors.

# In[1]:


import sys
get_ipython().system('{sys.executable} -m pip install progressbar')


# In[3]:


import re
import math
import json
import progressbar
import pandas as pd
import numpy as np
import seaborn as sns

from matplotlib import pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import fbeta_score, accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')

get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


# read in the json files
portfolio = pd.read_json('data/portfolio.json', orient='records', lines=True)
profile = pd.read_json('data/profile.json', orient='records', lines=True)
transcript = pd.read_json('data/transcript.json', orient='records', lines=True)


# In[5]:


portfolio.shape, profile.shape, transcript.shape


# # 1. Access Portfolio data
# 

# In[6]:


portfolio.shape


# In[7]:


portfolio.info()


# In[8]:


portfolio.describe()


# In[9]:


ax = portfolio["offer_type"].value_counts().plot.bar(
    figsize=(5,5),
    fontsize=14,

)

ax.set_title("offer types", fontsize=20)
ax.set_xlabel("Offers", fontsize=15)
sns.despine(bottom=True, left=True)


# We can see from above that the portfolio table consists of information about the offers provided. It has channels through which the specific offers are given i.e email, web etc. 
# 
# histogram containing a distribution of 3 offer types totalling in 10 entries.

# # 2. Access Profile data

# In[10]:


profile.head(10)


# In[11]:


profile.shape


# In[12]:


profile.describe(include="all")


# In[13]:


#check for null values 
profile.isnull().sum()


# In[14]:


profile[profile['age']== 118].age.count()


# In[15]:


#We can drop lines with age = 118

profile[profile['age']== 118].drop(['became_member_on' ,'id'], axis=1)


# In[16]:


#check for age above 80 and less than 118 

profile[(profile['age'] > 80) & (profile['age'] < 118)]


# Applications usage is less for people with age > 80 we can assume that based on lines retreived.

# In[17]:


#Creating Subplots for distribution based on Gender,Age and Income
sns.set_style('darkgrid')
fig,ax= plt.subplots(1,3,sharex=False, sharey=False,figsize=(12,5))
fig.tight_layout()

# GENDER BASED
profile.gender.value_counts().plot.bar(ax=ax[0],fontsize=10) 
ax[0].set_title("Gender Wise", fontsize=15,color='blue')
ax[0].set_xlabel("Gender", fontsize=10)
sns.despine(bottom=True, left=True)


# AGE BASED
profile.age.plot.hist(ax=ax[1],fontsize=10,edgecolor='black') 
ax[1].set_title("Age  Wise", fontsize=15,color='blue')
ax[1].set_xlabel("Age", fontsize=10)
sns.despine(bottom=True, left=True)

# INCOME BASED
profile.income.plot.hist(ax=ax[2],fontsize=10,edgecolor='black',range=(20000, 120000)) 
ax[2].set_title("Income Wise", fontsize=15,color='blue')
ax[2].set_xlabel("Income", fontsize=10)
sns.despine(bottom=True, left=True)

plt.show()


# Above is the distribution of data on age, income and gender basis.
# 
# Figures clearly shows that males utilize app more than females and age group from 50-70 is the highest, evenn income ramge from 60-80k are highest in using ap when compared to others.
# 
# Graphs may look slight differenr after cleaning the datasets and clearing null values.

# # Access Transcript data

# In[18]:


transcript.head(10)


# In[19]:


transcript.shape


# In[20]:


#check for null values
transcript.isnull().sum()


# In[21]:


# to understand different values in field value from transcript table.

transcript.value.value_counts()


# In[ ]:





# #  Data Cleaning
# 
# - Cleaning Portfolio

# In[22]:


# portfolio: rename id col name to offer_id.

portfolio.rename(columns={'difficulty':'offer_difficulty' , 'id':'offer_id', 'duration':'offer_duration', 'reward': 'offer_reward'}, inplace=True)


# In[23]:


portfolio.columns


# In[24]:


portfolio.head()


# # Cleaning Profile
# 
# - Renaming some columns for better readability .
# - imputing null values with mean and mode to retain them
# - identify age outlier and remove data from the set
# - classify ages into bins.

# In[25]:


# renaming columns
profile.rename(columns={'id':'customer_id' , 'income':'customer_income'}, inplace=True)


# In[26]:


profile.columns


# In[27]:


def processed_profile(profile_df):
    """ 
    Cleans the profile_df data frame like replacing null values and creates bins groups for ages 
        
    Parameters
    ----------   
    profile_df: Input data frame
    
    Returns
    -------
    profile_df: output data frame after cleansing
    """
    
    
    #deal with null values
    #replace 118 age values with NaN so to replace them easily with mean age
    profile_df.replace(118, np.nan , inplace=True)
    
    #replace NaN age values with mean age
    profile_df['age'] = profile_df['age'].fillna(profile_df['age'].mean())
    
    #replace missing income values with mean income
    profile_df['customer_income'] = profile_df['customer_income'].fillna(profile_df['customer_income'].mean())
    
    #replace missing gender values with most frequent gender
    mode = profile_df['gender'].mode()[0]
    profile_df['gender'] = profile_df['gender'].fillna(mode)
    
    #remove outliers
    profile_df = profile_df[profile_df['age'] <= 80]
    profile_df['age'] = profile_df['age'].astype(int)
    
      
    #add Age_group column to set the age ranges
    profile_df.loc[(profile_df.age < 20) , 'Age_group'] = 'Under 20'
    profile_df.loc[(profile_df.age >= 20) & (profile_df.age <= 45) , 'Age_group'] = '20-45'
    profile_df.loc[(profile_df.age >= 46) & (profile_df.age <= 60) , 'Age_group'] = '46-60'
    profile_df.loc[(profile_df.age >= 61) , 'Age_group'] = '61-80'
    profile_df.drop('age',axis=1,inplace=True)
    
    return profile_df


# In[28]:


processed_profile = processed_profile(profile)


# In[29]:


processed_profile.head()


# In[30]:


#Re- check for missing values and null values , I have faced issues in downward steps with these so checking again.

processed_profile.isnull().sum()


# In[31]:


processed_profile.shape


# # Cleaning transcript
# 
# - Rename cols
# - Explore Value field to create columns from them again

# In[32]:


# renaming columns to match with other dataframes

transcript.rename(columns={'person':'customer_id'}, inplace=True)


# In[33]:


transcript.columns


# In[34]:


def processed_transcript(df):
    """
    Cleans the Transcript table by splitting value fileds and replacing nan values, drop extra columns
    PARAMETERS:
        transcript dataframe
    
    RETURNS:
        Cleaned transcript  dataframe
    
    """
    
    #expand the dictionary to coulmns (reward, amount, offre id) from value field
    df['offer_id'] = df['value'].apply(lambda x: x.get('offer_id'))
    df['offer id'] = df['value'].apply(lambda x: x.get('offer id'))
    df['reward'] = df['value'].apply(lambda x: x.get('reward'))
    df['amount'] = df['value'].apply(lambda x: x.get('amount'))
    
    #move 'offer id' values into 'offer_id'
    df['offer_id'] = df.apply(lambda x : x['offer id'] if x['offer_id'] == None else x['offer_id'], axis=1)
    
    #drop 'offer id' column 
    df.drop(['offer id' , 'value'] , axis=1, inplace=True)
    
    #replace nan
    df.fillna(0 , inplace=True)
    
    return df


# In[35]:


processed_transcript = processed_transcript(transcript)


# In[36]:


processed_transcript.head()


# In[37]:


processed_transcript.shape


# # Merge data frame for further analysis

# In[38]:


def combine_data(portfolio,profile,transcript):
    """
    Merge data into single dataframe for analysis
       
    Parameters
    ---------- 
    portfolio :  portfolio data frame
    profile :  profile data frame
    transcript :  transcript data frame
      
    Returns
    -------
    returns merged data frame
    
    """
    
    # merge dataframes portfolio and transcript on offer_id's
    final_df = pd.merge(portfolio, transcript, on='offer_id')
    
    # merge dataframes final and profile on customer ids
    final_df = pd.merge(final_df, profile, on='customer_id')
    
    return final_df


# In[39]:


final_df = combine_data(portfolio, processed_profile, processed_transcript)


# In[40]:


final_df.head()


# In[41]:


final_df.shape


# In[42]:


final_df.info()


# In[ ]:





# # We are trying to answer basic questions like to understand what events should be utilized while building model
# 
# - average Income
# - Frequent offer used by customers
# - what age group of males and females are doing better
# - how different age groups respond to the offers

# # Average income

# In[43]:


final_df['customer_income'].mean()


# the average income of customers is 65924 as of avaiable data after cleaning al null values and merging into single dataframe.
# 

# In[44]:


sns.distplot(final_df['customer_income'], bins=50, hist_kws={'alpha': 0.4});


# # Frequently used offers by customers.
# 
# - BOGO is the most frequent offer availed by most of the users from availabe data and then followed by discounts

# In[45]:


final_df['offer_type'].value_counts().plot.bar(title='Types of offers')


# # How different genders respond to the offers.
# 
# There are three types of responses corresponding to each offer
# - offer received
# - offer viewed
# - offer completed

# In[46]:


plot_gender = final_df[final_df['gender'] != 'O']


# In[47]:


plt.figure(figsize=(12, 4))
sns.countplot(x= "event", hue= "gender", data=plot_gender)
plt.title('Gender distribution by offer responses')
plt.ylabel('Count')
plt.xlabel('Event')
plt.legend(title='Gender')


# we can clearly say that most number of males receive offers but offer completion is seen almost equal in both males and females.
# 
# which means we can understand that irrespective of offer being received and offer viewed , offer will be completed.
# 
# This will be key point while building model, we can consider all lines with different responses than only going for 'offer completion'

# #  Different age groups repond to offers

# In[49]:


plt.figure(figsize=(15, 5))
sns.countplot(x= "Age_group", hue= "event", data=plot_gender)
sns.set(style="darkgrid")
plt.title('Age distribution')
plt.ylabel('Count')
plt.xlabel('Age Range')
plt.legend(title='Event')


# Here we can just look at the offer completion rate across differnt age groups to understand that we can send offers to all customers irrespective if they have previously received and viewed an offer.
# 
# which means while building model we can cosnider all types of events make predictions which will add weightage to the features.

# # Build Model to predict Responses.

# In[50]:


final_df.head()


# In[51]:


final_df.info()


# - one-hot encode categorical data such as gender, offer type, channel and age groups.
# - Encode the 'event' data to numerical values to cinsider all values while buidling model.
#     offer received ---> 1
#     offer viewed ---> 2
#     offer completed ---> 3
# - Encode offer id and customer id.
# - Drop column 'became_member_on' and add separate columns for month and year.
# - Scale and normalize numerical data for few columns.
# 

# In[52]:


def process_final_data(df):
    """
    Cleans merged data to prepare for building model without any string values in the columns.
    
    Parameters
    ----------
    Input df: input merged data frame
    
    Returns
    -------
    output df: cleaned data frame as per requirements
       
    """
    #process the categorical variables by giving them one-hot encode values.
    categorical_cols = ['offer_type', 'gender', 'Age_group']
    df = pd.get_dummies(df, columns = categorical_cols)
    
    #process channels column and drop when done 
    df = df.drop('channels', 1).join(df.channels.str.join('|').str.get_dummies())
    
    #process became_member_on column to change datatype of became_member_on and create month_member, year_member based it
    df['became_member_on'] = df['became_member_on'].apply(lambda x: pd.to_datetime(str(x), format='%Y%m%d'))
    
    #add new columns for month & year
    df['month_member'] = df['became_member_on'].apply(lambda x: x.day)
    df['year_member'] = df['became_member_on'].apply(lambda x: x.year)
    
    #drop became_member_on column once requiered fields are created.
    df.drop('became_member_on',axis=1, inplace=True)    
    
    #process offer_id column to zip to dict
    offerids = df['offer_id'].unique().tolist()
    o_mapping = dict( zip(offerids,range(len(offerids))) )
    df.replace({'offer_id': o_mapping},inplace=True)
    
    #process customer_id column
    cusids = df['customer_id'].unique().tolist()
    c_mapping = dict( zip(cusids,range(len(cusids))) )
    df.replace({'customer_id': c_mapping},inplace=True)
    
    #encode 'event' data to numerical values according
    df['event'] = df['event'].map({'offer received':1, 'offer viewed':2, 'offer completed':3})
    
    return df


# In[53]:


final_data = process_final_data(final_df)


# In[54]:


final_data.info()


# In[55]:


#process numerical variables
#initialize a MinMaxScaler, then apply it to the features
scaler = MinMaxScaler() # default=(0, 1)

numericals = ['customer_income', 'offer_difficulty', 'offer_duration', 'offer_reward', 'time', 'reward', 'amount']
final_data[numericals] = scaler.fit_transform(final_data[numericals])


# In[56]:


final_data.head(5)


# In[57]:


final_data.columns


# In[58]:


final_data.event.value_counts()


# # Train and Test Data
# 

# In[59]:


features = final_data.drop('event', axis=1)
target = final_data['event']


# In[60]:


X_train, X_test, y_train, y_test = train_test_split(features, target, test_size = 0.2, random_state = 0)

print("Training set: {} rows".format(X_train.shape[0]))
print("Testing set: {} rows".format(X_test.shape[0]))


# In[ ]:





# # Metrics
# 
# We will consider the F1 score as the model metric to assess the quality of the approach and determine which model gives the best results. The traditional or balanced F-score (F1 score) is the harmonic mean of precision and recall, where an F1 score reaches its best value at 100 and worst at 0.
# 
# F1-score is used when the False Negatives and False Positives are crucial, here in our case we need to avoid situation
# - where they miss to send out offers to customer who actually completed previously (False Negative)
# - Also, who didnt actually complete offer but are offered now (False Positive)
# 
# F1 score is reliable metric in cases of imbalanced dataset and traditional classification problems.

# In[61]:


def train_test_data(clf):
    """
    Return train and test F1 score along with the models considered
       
    Parameters
    --------
    clf: estimator instance
    
    Returns
    --------
    train_f1: train data F1 score
    test_f1: test data F1 score
    name: model name
       
    """
    train_pred =  (clf.fit(X_train, y_train)).predict(X_train)
    test_pred = (clf.fit(X_train, y_train)).predict(X_test)
    
    train_f1 =  accuracy_score(y_train, train_pred)*100
    test_f1= fbeta_score(y_test, test_pred, beta = 0.5, average='micro' )*100
    
    name_model = clf.__class__.__name__
    
    return train_f1, test_f1, name_model


# In[62]:


clf_KNN = KNeighborsClassifier(n_neighbors = 5)


# train and test f1 score results are stored into bewlo varables.
K_train_f1, K_test_f1, K_model = train_test_data(clf_KNN)

#loading accuracy and f1score into variables.
knn = {'Model': [K_model], 'train F1 score':[K_train_f1], 'test F1 score': [K_test_f1]}

pd.DataFrame(knn)


# In[63]:


clf_dt = DecisionTreeClassifier(random_state = 10)

dt_train_f1, dt_test_f1, dt_model = train_test_data(clf_dt)

dt = {'Model': [dt_model], 'train F1 score':[dt_train_f1], 'test F1 score': [dt_test_f1]}

pd.DataFrame(dt)


# Optimizing model using RandomForestClassifier and am making use of GridSearchCV to identify best parameters, through which best accuracy can be achevived.
# 
# Through which offers can be sent to customers who would only respond

# In[64]:


#  Initialize the classifier
clf_ran = RandomForestClassifier(random_state =42, n_estimators=20)

# Create the parameters list you wish to tune, using a dictionary if needed.
parameters = {'max_features':['auto', 'sqrt'], 'min_samples_leaf':[2,4,6,8,10], 'min_samples_split':[2,4,6,8,10]} 

#  Make an fbeta_score scoring object using make_scorer()
beta= 0.01

#  Perform grid search on the classifier using 'scorer' as the scoring method using GridSearchCV()
grid_obj = GridSearchCV(clf_ran, parameters, cv=3, scoring='f1_macro')

# TODO: Fit the grid search object to the training data and find the optimal parameters using fit()
grid_fit = grid_obj.fit(X_train, y_train)

# Get the estimator
best_clf = grid_fit.best_estimator_

# Make predictions using the unoptimized and model
train_pred =  (clf_ran.fit(X_train, y_train)).predict(X_train)
predictions = (clf_ran.fit(X_train, y_train)).predict(X_test)
best_predictions = best_clf.predict(X_test)


ran_model = clf_ran.__class__.__name__
ran_train_f1 =  accuracy_score(y_train, train_pred)*100
ran_test_f1= fbeta_score(y_test, best_predictions, beta = beta, average='macro' )*100

# Report the before-and-afterscores
print("Unoptimized model\n------")
print("Accuracy score on testing data: {:.4f}".format(accuracy_score(y_test, predictions)))
print("F-score on testing data: {:.4f}".format(fbeta_score(y_test, predictions, average='macro', beta = beta)))
print("\nOptimized Model\n------")
print("Final accuracy score on the testing data: {:.4f}".format(accuracy_score(y_test, best_predictions)))
print("Final F-score on the testing data: {:.4f}".format(fbeta_score(y_test, best_predictions, average='macro', beta = beta)))


# In[ ]:





# #  Exploratort data Analysis

# In[86]:


dis_col = {'color': [ '#3CAEA3','#ED553B']}

EDA = sns.FacetGrid(plot_gender, row='event', col='gender', hue_kws=dis_col, hue='gender', size=5)

EDA.map(plt.hist, 'offer_type')
plt.show()


# we can clearly see here offer completion rate is low though offer is received or not, so we may need additinal features and more data to predict user behavior in responding to the offers.
# 
# The males represent 62.7% of the data and use the Starbucks app more than the females. Specifically, both males & females in the age group 46-60 use app the most. Discount offers are more preferred by the customers. Also, there are less number of customers who actually complete the offer as compared to the ones who just view & ignore it.
# 
# We can look at the basic questions answered above to understand how demographics effect customers in copletimng an offer.
# 

# # Model Metrics comparisons

# In[66]:


model_comp = {'Model name': [ K_model, dt_model, ran_model], 
              'train F1 score': [K_train_f1 , dt_train_f1, ran_train_f1] ,
           'test F1 score': [K_test_f1 , dt_test_f1, ran_test_f1] }
          
pd.DataFrame(model_comp)


# The test data set is used to evaluate the model. Both the models are better than the KNN. The best score is created by the DecisionTreeClassifier model, as its validate F1 score is 84.79, which is much higher than the benchmark.
# 
# The RandomForestClassifier model scores good as well compared to the DecisionTree, with a test F1 score of 84.30. Our problem to solve is not that sensitive which requires very high F1 score, so the scores are good & sufficient and can be used for the classification purpose to predict whether a customer will respond to an offer.

# # Conclusions

# The problem statement chosen was to build a model that predicts whether a customer will respond to an offer.
# 
# Strategy for solving this problem in following steps.
# 
# 1. Combined datasets portfolio, customer profile, and transaction data.
# 2. Assessed the accuracy and F1-score of model on all given events.
# 3. Compared the performance of KNeighborsClassifier and DecisionTree with optimized RandomForest by fine tuning the parameters to achieve better results.
# 
# Analysis suggests that RandomForest model has better testing accuracy of 0.79 and F1-score 0.84 and is not over-fitting the data. However, RandomForest performance can be improved by analyzing more features with more data coming-in. With current data demographics are limited to age, gender and income with which we were able to achieve F1-score of 0.84 and i'm confident that with additional features weightage score can be improved.
# 
# Initially there was no clue on how to start and what features to assess but once the datasets are cleaned and combined together into one single dataset and by answering basic questions on how demographics are responding for the offer It is clear on what features to target. With more data, the classification models will produce better accuracy and F1-score results.
# 

# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# Importing required libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import OneHotEncoder
import joblib


# Exploring the dataset

# In[2]:


#Merge the dataset
# Concatenate the datasets vertically
merged_df = pd.read_csv('datasets/merged_dataset.csv')
merged_df.columns


# In[3]:


#Explore the merged dataset
merged_df.shape


# In[4]:


merged_df.info()


# In[5]:


merged_df.head(5)


# In[6]:


#check for the towns
merged_df['Town'].nunique()


# In[7]:


# Sort the DataFrame by the 'town' column in alphabetical order
merged_df = merged_df.sort_values('Town')
merged_df.head(5)


# In[8]:


unique_towns = merged_df['Town'].unique()
print(unique_towns)


# In[9]:


# Replace 'North', and 'South' from all the values in the 'town' column
merged_df['Town'] = merged_df['Town'].str.replace(' North', '').str.replace(' South', '')
# Replace 'East' and 'West' from all the values in the 'town' column except Westlands
merged_df['Town'] = merged_df['Town'].apply(lambda x: x if x == ' Westlands' else x.replace(' East', '').replace(' West', ''))
# Replace 'Road' and 'Town' from all the values in the 'town' column
merged_df['Town'] = merged_df['Town'].str.replace(' Road', '').str.replace(' Town', '')
unique_towns = merged_df['Town'].unique()
# Replace 'Constituency' and 'Central' from all the values in the 'town' column
merged_df['Town'] = merged_df['Town'].str.replace(' Constituency', '').str.replace(' Central', '')
unique_towns = merged_df['Town'].unique()
# Replace 'CBD' and 'Central' from all the values in the 'town' column
merged_df['Town'] = merged_df['Town'].str.replace(' CBD', '').str.replace(' Central', '')
# Replace 'space' at the beginning from all the values in the 'town' column
merged_df['Town'] = merged_df['Town'].str.lstrip()
unique_towns = merged_df['Town'].unique()
merged_df = merged_df.sort_values('Town')
print(unique_towns)


# In[10]:


missing_mask = merged_df['sq_mtrs'].isna()

for index, row in merged_df[missing_mask].iterrows():
    town = row['Town']
    bedrooms = row['Bedrooms']
    bathrooms = row['Bathrooms']
    similar_rows = merged_df[(merged_df['Town'] == town) & (merged_df['Bedrooms'] == bedrooms) & (merged_df['Bathrooms'] == bathrooms)]
    
    if len(similar_rows) == 0:
        similar_rows = merged_df[(merged_df['Bedrooms'] == bedrooms) & (merged_df['Bathrooms'] == bathrooms)]
    
    if len(similar_rows) > 0:
        merged_df.at[index, 'sq_mtrs'] = similar_rows['sq_mtrs'].mean()

# Check for missing values
missing_mask = merged_df['sq_mtrs'].isna()
if missing_mask.any():
    print("Rows with missing values in 'sq_mtrs' column:")
    print(merged_df[missing_mask])
else:
    print("All missing values in 'sq_mtrs' column have been filled")


# In[11]:


# Save the merged dataset
merged_df.to_csv('../merged_dataset.csv', index=False)


# # Visualising the data
# 
# 
# 

# In[12]:


#Lets first check the distribution of the Price column
sns.displot(merged_df['Price'])


# In[13]:


#scatterplot for price  and Bedrooms colored by Bathrooms
sns.scatterplot(x='Bedrooms',y='Price',data=merged_df,hue='Bathrooms')


# In[14]:


#we can see that the price column is right skewed
#lets check the houses that are outliers (200000 and above))
merged_df[merged_df['Price']>=200000]


# ## Conclusion
# The houses with more Bedrooms and Bathrooms are more expensive
# 

# In[15]:


merged_df.info()


# In[16]:


#generate a pairplot on price, Bedrooms, Bathrooms and sq_mtrs
sns.pairplot(merged_df[['Price','sq_mtrs','Bedrooms','Bathrooms','Town']])


# In[17]:


#check for the towns
merged_df['Town'].nunique()


# In[18]:


#The houses with more Bedrooms and bathrooms are more expensive
#Lets check the scatterplot of the sq_mtrs column with price
sns.scatterplot(x='sq_mtrs',y='Price',data=merged_df)


# In[19]:


#grouby the towns and get the mean price,plot it
merged_df.groupby('Town')['Price'].mean().sort_values(ascending=False).plot(kind='bar')


# In[20]:


plt.figure(figsize=(10,6))
sns.scatterplot(x='Bedrooms',y='Price',data=merged_df,hue='Town')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


# In[21]:


plt.figure(figsize=(10,6))
sns.scatterplot(x='Bathrooms',y='Price',data=merged_df,hue='Town')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


# In[22]:


#Lets check the houses with the most bedrooms
merged_df[merged_df['Bedrooms']==merged_df['Bedrooms'].max()]


# # Set features and labels

# In[23]:


X = merged_df[["Bedrooms","sq_mtrs", "Bathrooms","Town"]]
y = merged_df[["Price"]]


# # Define the model and train it

# In[24]:


# Extract the numerical features and categorical feature
X_num = merged_df[["sq_mtrs","Bedrooms", "Bathrooms"]]
X_cat = merged_df[["Town"]]


# In[25]:


# Create an instance of the OneHotEncoder class and fit it to the categorical feature
ohe = OneHotEncoder()
ohe.fit(X_cat)


# In[26]:


# Transform the categorical feature using the fitted OneHotEncoder
X_cat_encoded = ohe.transform(X_cat).toarray()


# In[27]:


# Combine the numerical and encoded categorical features
X = np.concatenate((X_num, X_cat_encoded), axis=1)
y = merged_df[["Price"]]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)

# Train the regression model using the training data
clf = DecisionTreeRegressor()
clf.fit(X_train, y_train)


# # Prediction and accuracy

# In[28]:


#Predictions using the testing set 
y_pred = clf.predict(X_test)

# #Example of few predictions
# print("Making predictions for the following 5 houses:")
# print(X.head())
# print("The predictions are")
# five_pred= clf.predict(X_test)[:5]

# five_pred


# In[29]:


print(X_test.shape)
print(y_test.shape)
print(y_pred.shape)


# In[30]:


# print("Making predictions for the following 5 houses:")
# print(X.head())
# print("The predictions are")
# str(y_test[:5])


# In[31]:


# #Checking the accuracy of the model using MSE,MAE and R-squared error

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

print('Mean squared error: ', mean_squared_error(y_test, y_pred))
print("Root Mean Squared error: ", np.sqrt(mean_squared_error(y_test, y_pred)))
print('Mean absolute error: ', mean_absolute_error(y_test, y_pred))
print('R-squared score: ', r2_score(y_test, y_pred))
print('Mean: ', np.mean(y_test))


# In[32]:


# trying new model
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Always scale the input. The most convenient way is to use a pipeline.
clf = make_pipeline(StandardScaler(),
                     SGDRegressor(max_iter=1000, tol=1e-3, loss="squared_error"))


# In[33]:


#Checking the accuracy of a model
clf.fit(X_train, y_train)
forestPred = clf.predict(X_test)
forestScores = clf.score(X_test, y_test)
forestScores


# In[34]:


mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
rmse


# In[35]:


#using a scatter plot to visualize how well the model is perfoming
plt.scatter(y_test, y_pred)
plt.xlabel('Actual values')
plt.ylabel('Predicted values')
plt.title('Actual vs Predicted values')
plt.show()


# In[36]:


# from sklearn.model_selection import cross_val_score
#  #Cross-validate the model
#  #Perform cross-validation on the model
# scores = cross_val_score(clf, X, y, cv=5, scoring='neg_mean_squared_error')
# rmse_scores = np.sqrt(-scores)

#  # Display the cross-validation scores
# print('Cross-Validation Scores:', rmse_scores)
# print('Mean:', rmse_scores.mean())
# print('Standard deviation:', rmse_scores.std())


# In[37]:


joblib.dump(clf, '../the_model.joblib')


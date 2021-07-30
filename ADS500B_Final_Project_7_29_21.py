#!/usr/bin/env python
# coding: utf-8

# In[45]:


import numpy as np
import pandas as pd
import matplotlib as mpl 
import matplotlib.pyplot as plt 
import seaborn as sns
import scipy
from sklearn.linear_model import LinearRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[46]:


#dataset and information regarding each notation
#This dataset contains house sale prices for King County, which includes Seattle. It includes homes sold between May 2014 and May 2015.

#id :a notation for a house

#date: Date house was sold

#price: Price is prediction target

#bedrooms: Number of Bedrooms/House

#bathrooms: Number of bathrooms/bedrooms

#sqft_living: square footage of the home

#sqft_lot: square footage of the lot

#floors :Total floors (levels) in house

#waterfront :House which has a view to a waterfront

#view: Has been viewed

#condition :How good the condition is Overall

#grade: overall grade given to the housing unit, based on King County grading system

#sqft_above :square footage of house apart from basement

#sqft_basement: square footage of the basement

#yr_built :Built Year

#yr_renovated :Year when house was renovated

#zipcode:zip code

#lat: Latitude coordinate

#long: Longitude coordinate

#sqft_living15 :Living room area in 2015(implies-- some renovations) This might or might not have affected the lotsize area

#sqft_lot15 :lotSize area in 2015(implies-- some renovations)


# In[3]:


## Use Pandas read function
## Remember to change filepath to your machine
house_df = pd.read_csv('E:\\ADS 500_B\\house_sales.csv')

# Making sure the file was read properly


# In[5]:


house_df.head()
# Returns the first n rows for the object based on the position


# In[6]:


house_df.shape
#Checks the dimensions of the house_df data frame


# In[7]:


#Swaps the format of the date to a a
house_df['date'] = pd.to_datetime(house_df['date'])


# In[8]:


# Returns the first 5 rows for the object based on the position
house_df.head()


# In[11]:


house_median = house_df.median()
house_median.head


# In[13]:


house_df = house_df.fillna(value = house_median)

#checking results
house_df.isnull().sum()


# In[14]:


#physically changing the data types

house_df.dtypes


# In[15]:


#change data types 

house_df = house_df.astype({"id": str, "lat": str, "floors": 'category',"waterfront": 'category',"view": 'category',"condition": 'category', "grade": 'category', 'zipcode' : str, 'long' : str })
#df = df.astype({"a": int, "b": complex})


# In[16]:


house_df.dtypes


# In[17]:


house_df.head()


# In[18]:


house_df.boxplot(column = ['price'])


# In[19]:


house_df.hist(column = ['price'])


# In[20]:


plt.scatter(house_df['price'],house_df['sqft_living'])


# In[21]:


corr_coef = np.corrcoef(house_df['price'],house_df['sqft_living'])
corr_coef


# In[22]:


plt.scatter(house_df['price'],house_df['bedrooms'])


# In[23]:


houseprice_mean = house_df.groupby('zipcode')['price'].mean()


# In[24]:


houseprice_mean.head()


# In[37]:


my_plot = houseprice_mean.plot(kind = 'bar',figsize =(15,5))

#plots a bar graph based on zipcode 


# In[38]:


print(house_df.corr())


# In[39]:


#Uses a seaborn heat map based ont the correlation of the house dataframe


corr = house_df.corr()

ax = sns.heatmap(
    corr, 
    vmin = -1, vmax=1, center=0,
    cmap = sns.color_palette("viridis", as_cmap = True),
    square = True
)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45 ,horizontalalignment='right'
);


# In[40]:


with sns.plotting_context("notebook", font_scale = 2.5):
    g = sns.pairplot(house_df, hue = 'bedrooms', palette = 'tab20', size = 6)
    g.set(xticklabels = []);


# In[41]:



#Excludes data that do not have much correlation, saves both computing power and prevents overfitting when making a mulitple regression value.

with sns.plotting_context("notebook", font_scale = 3):
    g = sns.pairplot(house_df[['price', 'sqft_lot', 'sqft_above', 'sqft_living', 'bedrooms', 'bathrooms', 'sqft_living15']], hue = 'bedrooms', palette = 'tab20c', size = 5)
    g.set(xticklabels = []);


# In[42]:


#Defining data to work with. The inputs (regressors, x) and output (predictor,y) should be arrays

house_df.drop(['id', 'date', 'lat', 'zipcode', 'long', 'yr_renovated', 'yr_built','waterfront','view'], axis = 1, inplace = True)


# In[43]:


house_df.columns

house_df.head()


# In[47]:


#check for VIF(Variance inflation factor) to detect multicollinearity in regression analysis. Check between highest correlation with price and sqft living, then check VIF for sqft_living and sqft_above.

X = house_df

#VIF dataframe
#https://www.geeksforgeeks.org/detecting-multicollinearity-with-vif-python/

vif_data = pd.DataFrame()
vif_data["feeature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i)
                    for i in range(len(X.columns))]
print(vif_data)


#The higher the VIF value, would indicate a higher correlation. 
#As sqft_living and sqft_above are pretty much similar since they take
#the same mount of square footage. only difference is that the sqft_above takes
#into account the footageof the basement. Research papers consider a VIF >10 as an 
#indicator of multicollinearity


# In[ ]:





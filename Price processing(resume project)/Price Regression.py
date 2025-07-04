'''
# Avocado Data Analysis

## Business Understanding

The aim of this project is to answer the following four questions:
    1. Which region are the lowest and highest prices of Avocado?
    2. What is the highest region of avocado production?
    3. What is the average avocado prices in each year?
    4. What is the average avocado volume in each year?
    
 ## Data Understanding

The [Avocado dataset](https://www.kaggle.com/neuromusic/avocado-prices) was been used in this project.

This dataset contains 13 columns:
    1. Date - The date of the observation
    2. AveragePrice: the average price of a single avocado
    3. Total Volume: Total number of avocados sold
    4. Total Bags: Total number  o bags
    5. Small Bags: Total number of Small bags
    6. Large Bags: Total number of Large bags
    7. XLarge Bags: Total number of XLarge bags
    8. type: conventional or organic
    9. year: the year
    10. region: the city or region of the observation
    11. 4046: Total number of avocados with PLU 4046 sold
    12. 4225: Total number of avocados with PLU 4225 sold
    13. 4770: Total number of avocados with PLU 4770 sold

'''
    
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

#load the file
df=pd.read_csv(r'D:\Naresh IT foundation\Python project\Price processing(resume project)\avocado.csv')    


 # Explore the Data  
df.info()

df.head()

#missing value check

df.isnull().sum() 

#checking for columns
df.columns

#dropping Unnecessary coloumns

df= df.drop(['Unnamed: 0', 'Date', '4046', '4225','4770'],axis=1)

# get top five value 

df.head()

def get_average(df,column):
    """
    Description: This function to return the average value of the column 

    Arguments:
        df: the DataFrame. 
        column: the selected column. 
    Returns:
        column's average 
    """
    return sum(df[column])/len(df)

def get_avarge_between_two_columns(df1,column1,column2):
    """
    Description: This function calculate the average between two columns in the dataset

    Arguments:
        df: the DataFrame. 
        column1:the first column. 
        column2:the second column.
    Returns:
        Sorted data for relation between column1 and column2
    """
    
List=list(df[column1].unique())
average=[]

if i in List:
    






























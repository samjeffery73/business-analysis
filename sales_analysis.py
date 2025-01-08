'''
Date: 01/07/2025

Developed by: Sam Jeffery

For more information on this project please refer to the README.md file on the GitHub repository.

A lot of code will be commented out, since it is for charts and display. It only needed to be run once, and the resulting charts were placed in PNG files.

To view my report, charts, please view the pictures and the document file that are also in the repository.

From the dataset on Kaggle, there are a few questions that are imposed that are interesting and worth exploring.

    1. How does customer age and gender influence their purchasing behavior?
    2. Are there discernible patterns in sales across different time periods?
    3. Which product categories hold the highest appeal among customers?
    4. What are the relationships between age, spending, and product preferences?
    5. How do customers adapt their shopping habits during seasonal trends?
    6. Are there distinct purchasing behaviors based on the number of items bought per transaction?
    7. What insights can be gleaned from the distribution of product prices within each category?

'''

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv(r"retail_sales_dataset.csv")

#print(df.shape)
'''
Result:

(1000, 9)



'''

#print(df.head())

'''
Result:

   Transaction ID        Date Customer ID  Gender  Age Product Category  Quantity  Price per Unit  Total Amount
0               1  2023-11-24     CUST001    Male   34           Beauty         3              50           150
1               2  2023-02-27     CUST002  Female   26         Clothing         2             500          1000
2               3  2023-01-13     CUST003    Male   50      Electronics         1              30            30
3               4  2023-05-21     CUST004    Male   37         Clothing         1             500           500
4               5  2023-05-06     CUST005    Male   30           Beauty         2              50           100

'''

# Remove spaces and put all columns to lowercase.

df.columns = df.columns.str.lower().str.replace(' ', '_')


#print(df.info())

'''
Result:

RangeIndex: 1000 entries, 0 to 999
Data columns (total 9 columns):
 #   Column            Non-Null Count  Dtype
---  ------            --------------  -----
 0   transaction_id    1000 non-null   int64
 1   date              1000 non-null   object
 2   customer_id       1000 non-null   object
 3   gender            1000 non-null   object
 4   age               1000 non-null   int64
 5   product_category  1000 non-null   object
 6   quantity          1000 non-null   int64
 7   price_per_unit    1000 non-null   int64
 8   total_amount      1000 non-null   int64
dtypes: int64(5), object(4)

'''

# After looking at the above result, there are a few object dtypes that need to be changed.
# Specifically: date, gender, and category.

df['date'] = pd.to_datetime(df['date'])
df['gender'] = df['gender'].astype('category')
df['product_category'] = df['product_category'].astype('category')


# Transaction ID isn't really important.
df.drop('transaction_id', axis=1, inplace=True)


#print(df.describe())

'''
Result:

                                date         age     quantity  price_per_unit  total_amount
count                           1000  1000.00000  1000.000000     1000.000000   1000.000000
mean   2023-07-03 00:25:55.200000256    41.39200     2.514000      179.890000    456.000000
min              2023-01-01 00:00:00    18.00000     1.000000       25.000000     25.000000
25%              2023-04-08 00:00:00    29.00000     1.000000       30.000000     60.000000
50%              2023-06-29 12:00:00    42.00000     3.000000       50.000000    135.000000
75%              2023-10-04 00:00:00    53.00000     4.000000      300.000000    900.000000
max              2024-01-01 00:00:00    64.00000     4.000000      500.000000   2000.000000
std                              NaN    13.68143     1.132734      189.681356    559.997632


'''




#print(df['gender'].value_counts())

'''
Result:

gender
Female    510
Male      490
Name: count, dtype: int64

 '''


#df['gender'].value_counts().plot(kind='pie', autopct='%1.1f%%')
# plt.show()

'''
The first graph, 01_gender_counts.png shows that 51%  (510) of the customers are female, 
while 49% (490) of the customers are male.



'''


#print(df['product_category'].value_counts())

'''
Result:

product_category
Clothing       351
Electronics    342
Beauty         307

'''

# df['product_category'].value_counts().plot(kind='bar', color='blue')
# plt.title("Product Category Frequency")
# plt.xticks(rotation = 0)
# plt.show()

'''
The second graph just displays the counts of product_category. With Clothing being the most popular.

'''


print(df['age'].describe())

'''
Result:

count    1000.00000
mean       41.39200
std        13.68143
min        18.00000
25%        29.00000
50%        42.00000
75%        53.00000
max        64.00000
Name: age, dtype: float64



Looking at age, we can see that the mean age is 41 years old, with a stdev of 13.

Our customers range from 18 to 64.


'''

print(df['age'].skew())

'''
Result:

-0.04881245380328967

There is almost no skew. This means that data is roughly symmetric.

'''

df['age'].plot(kind='hist')
plt.xlabel('Age')
plt.show()












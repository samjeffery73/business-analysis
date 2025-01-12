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


'''
 Starting with our EDA, I will be doing exploring and cleaning the data.


'''


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

'''
After looking at the above result, there are a few object dtypes that need to be changed.
Specifically: date, gender, and category.
'''


df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)
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

'''

Start of Analysis. Univariate Analysis.

Related graphs:
01, 02, 03, 04, 05

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

# print(df['age'].skew())

'''
Result:

-0.04881245380328967

There is almost no skew. This means that data is roughly symmetric.

'''

# df['age'].plot(kind='hist')
# plt.xlabel('Age')
# plt.show()





# print(df['quantity'].describe())

'''
Result:

count    1000.000000
mean        2.514000
std         1.132734
min         1.000000
25%         1.000000
50%         3.000000
75%         4.000000
max         4.000000
Name: quantity, dtype: float64

We can see the result of the quantity distribution, with the mean quantity purchased being 2.5 with a stdev of 1.13.

'''


# df['quantity'].value_counts().plot(kind='bar')
# plt.show()


# print(df['price_per_unit'].describe())

'''
Result:


count    1000.000000
mean      179.890000
std       189.681356
min        25.000000
25%        30.000000
50%        50.000000
75%       300.000000
max       500.000000


'''

# sns.histplot(df, x='price_per_unit')
# plt.show()


# print(df['total_amount'].describe())


'''
Result:

count    1000.000000
mean      456.000000
std       559.997632
min        25.000000
25%        60.000000
50%       135.000000
75%       900.000000
max      2000.000000

'''


# sns.histplot(df['total_amount'])
# plt.title('Total Amount')
# plt.show()



'''
Moving on to Bivariate Analysis -- Numeric vs Numeric

Related plots:

06, 07, 08

'''

# sns.scatterplot(df, x='age', y='total_amount')
# plt.title('Age vs Total Amount')
# plt.show()

# print(df['age'].corr(df['total_amount']))

'''
Result:

-0.060568023883045656


There seems too be a weak negative linear correlation. But let's explore further.

'''


# print(df['age'].corr(df['total_amount'], method='spearman'))

'''
Result:

-0.037864028718041294

There is still a weak negative correlation, so this isn't really worth looking in to much further.

'''


# print(df['price_per_unit'].corr(df['total_amount']))

'''
Result:

0.8519248403554038

Checkiing the correlation of Price per unit and total amount, we can see that it is pretty strong. As PPU is increased, the total is also increased,
which is to be expected.

To verify, lets check with the graph, 07.

'''

# sns.scatterplot(df, x='price_per_unit', y='total_amount')
# plt.show()




# print(df['quantity'].corr(df['total_amount']))

'''
Result:


0.37370705412140603

This is less of a linear correlation than previous, PPU and total amount, but still positive and worth investigating.

'''

# sns.scatterplot(df, x='quantity', y='total_amount')
# plt.show()



'''

Bivariate Analysis Continued: Categorical vs Categorical


'''

# print(pd.crosstab(df['gender'], df['product_category']))

'''

product_category  Beauty  Clothing  Electronics
gender
Female               166       174          170
Male                 141       177          172



By reviewing, we can see the difference in preferences in relation to Gender. First, female have a pretty even distribution, with the ranks being Clothing, Electronics, Beauty.

Males ranks are Clothng, Electronics, and Beauty, with a stronger difference in the max, Clothing, and the min, Beauty.

'''


# print(pd.crosstab(df['gender'], df['quantity']))


'''
quantity    1    2    3    4
gender
Female    125  120  127  138
Male      128  123  114  125



Doing another simple review, we can see that the most frequent quantity of items bought by females was 4.

The most frequent quantity of items bought for men was 1, with 4 and 2 following closely.

'''


# print(pd.crosstab(df['product_category'], df['quantity']))


'''

quantity           1   2   3   4
product_category
Beauty            74  75  85  73
Clothing          88  80  86  97
Electronics       91  88  70  93


From items that are bought, the most frequent item bought at high quantity is clothing, followed by electronics.

The item bought most at low quantity is electronics.

'''


print(pd.crosstab(df['gender'], df['price_per_unit']))


'''

price_per_unit  25   30   50   300  500
gender
Female          115   92   98  106   99
Male             95   91  113   91  100

Both genders show a balanced variety of price per unit.
Females typically by more items in the 25, and 300 range, with men performing better in the 50 range. 

Both are comparable in 30 and 500.

'''




'''
Numeric vs Categorical Variables.

Related Graphs:

09, 10

'''


# sns.set_theme(style="darkgrid")
# my_pal = {"Male": "g", "Female": "b"}
# sns.boxplot(df, x='gender', y='total_amount', palette=my_pal)
# plt.title("Gender vs Total Amount")
# plt.show()


# sns.set_theme(style="darkgrid")
# my_pal = {"Beauty": "g", "Clothing": "b", "Electronics":"m"}
# sns.boxplot(df, x='product_category', y='total_amount', palette=my_pal)
# plt.title("Product Category vs Total Amount")
# # plt.show()



'''
Moving on to Multivariate analysis

'''

# sns.boxplot(df, x='product_category', y='total_amount', hue='gender')
# plt.show()


sns.boxplot(df, x='price_per_unit', y='total_amount', hue='gender')
plt.show()

# print(df.groupby(['price_per_unit','gender'])['total_amount'].describe())


'''
Result:

                       count         mean         std    min     25%     50%     75%     max
price_per_unit gender
25             Female  115.0    63.043478   28.165361   25.0    50.0    50.0   100.0   100.0
               Male     95.0    61.052632   28.179655   25.0    37.5    50.0    87.5   100.0
30             Female   92.0    74.347826   33.718432   30.0    30.0    75.0    90.0   120.0
               Male     91.0    71.538462   34.283883   30.0    30.0    60.0    90.0   120.0
50             Female   98.0   131.122449   58.992969   50.0   100.0   150.0   200.0   200.0
               Male    113.0   122.566372   57.473266   50.0    50.0   100.0   200.0   200.0
300            Female  106.0   786.792453  340.049148  300.0   600.0   900.0  1200.0  1200.0
               Male     91.0   791.208791  333.216097  300.0   600.0   900.0  1200.0  1200.0
500            Female   99.0  1237.373737  554.864547  500.0  1000.0  1000.0  1500.0  2000.0
               Male    100.0  1250.000000  570.751764  500.0   500.0  1500.0  1625.0  2000.0



'''

'''
Time Series Analysis

Checking the Total Amount spent over Months and quarterly.

'''


monthly = df['total_amount'].resample('M').mean()
monthly.plot()
plt.show()



quarterly = df['total_amount'].resample('Q').mean()
quarterly.plot()
plt.show()



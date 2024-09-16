import pandas as pd
import numpy as np

'''
######## PSET 1 #########
series = pd.Series([10, 20, 30, 40, 50], index = ['a', 'b', 'c', 'd', 'e'])

series_c = series.loc['c']
series_a_e = series.loc[['a', 'e']]
series_last_3 = series.iloc[-3:]
series_shape = series.shape

#print(f'@ index "C"\n{series_c}\n')
#print(f'@ index "A" & "E"\n{series_a_e}\n')
#print(f'Last 3 vals of series:\n{series_last_3}\n')
#print(f'Series shape:\n{series_shape}')

######## PSET 2 #########

alice = pd.Series(['alice', 24, 'new york'])
bob = pd.Series(['bob', 30, 'san francisco'])
carol = pd.Series(['carol', 22, 'los angeles'])
david = pd.Series(['david', 35, 'chicago'])
eve = pd.Series(['eve', 29, 'miami'])

df = pd.DataFrame(data = (alice, bob, carol, david, eve))
df.columns = ['name', 'age', 'city']

#print(df)

####### PSET 3 ########

data = {
    'name': ['alice', 'bob', 'carol', 'david', 'eve'] ,
    'age': [24, 30, 22, 25, 29],
    'city': ['new york', 'san francisco', 'los angeles', 'chicago', 'miami'] 
}    
    
df = pd.DataFrame(data = data)
df.columns = ['name', 'age', 'city']

mask = df['age'] > 25
#print(df[mask], '\n\n')

mask = (df['city'] == 'new york') | (df['city'] == 'miami')
# or 
mask1 = df['city'].isin(['new york', 'miami'])

#print(df[mask])

last2 = df.tail(2)
#print(last2)

######## PSET 4 #########

grades = [90, 85, 88, 92, 75]
index = ['Alice', 'Bob', 'Carol', 'David', 'Eve']

s1 = pd.Series(grades, index = index)
df['grades'] = s1.to_numpy()
#print(df)
df.sort_values(by = 'grades', axis = 0, ascending = False, inplace = True)
#print(df)
'''

######## ROUND 2 #######

# Filtering and sorting

'''
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'Age': [25, 30, 35, 40, 45],
    'Score': [85, 90, 95, 80, 75]
}

df = pd.DataFrame(data)
older_30 = df.query('Age > 30') # greater than 30 in age.
sorted_score = df.sort_values(by = 'Score', axis = 0) # sorting by score in descending order
sorted_age = df.sort_values(by = 'Age', axis = 0, ascending = True) # sorting by age in ascending order

df.index = pd.date_range(start='2023-01-01', periods=5, freq='D')[::-1]
df.sort_index(inplace= True) # sorted by date in descending order

data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'Age': [25, np.nan, 35, np.nan, 45],
    'Score': [85, 90, np.nan, 80, 75]
}

df = pd.DataFrame(data)
df_nan = df.loc[:, 'Age'].isna() # true of the value is nan, else is false.

median = df['Age'].median()
new_df = df.fillna({'Age': median}) # fills nan values in Age with the median age of all people.

dropped_df = df.dropna() # drops rows with nan values
'''

######## ROUND 3 #########
'''
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'Alice', 'Eve'],
    'Age': [25, 30, 35, 25, 45],
    'Score': [85, 90, 95, 85, 75]
}

df = pd.DataFrame(data)

#print(f"Original DF:\n\n{df}\n")
#print(f"Dropped DF:\n\n{df.drop_duplicates()}\n")
#print(f"Dropped Names DF:\n\n{df.drop_duplicates('Name')}")


data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'Score': [85, 90, 95, 80, 75]
}
df = pd.DataFrame(data)
df.replace(to_replace = 95, value = 100, inplace=True)
#print(f'Original DF:\n\n{df}\n')
#print(f'Replaced DF:\n\n{df}\n')

data = {
    'Name': ['Alice Smith', 'Bob Brown', 'Charlie Johnson'],
    'Occupation': ['Engineer', 'Artist', 'Doctor']
}
df = pd.DataFrame(data)

df_names = list(df["Name"])
first_names = [name.split()[0] for name in df_names] # gets first names
#print(first_names)
df.loc[:, 'Occupation'] = df['Occupation'].str.upper()
#print(df)

data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'Age': [25, 30, 30, 40, 45],
    'Score': [85, 90, 85, 80, 75]
}
df = pd.DataFrame(data)

df = df.query('Age == 30 or Age == 45')
df.loc[:, 'Score'] = df.loc[:, 'Score'].replace(to_replace = 85, value = 95)

data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve', 'Frank'],
    'Age': [25, 30, 35, 40, 45, 50],
    'Score': [85, 90, 95, 80, 75, 60]
}
df = pd.DataFrame(data, index=pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-04']))
df = df[~df.index.duplicated()]
df = df.loc['2023-01-02':'2023-01-04']

#df.to_excel('data/test.xlsx')
#df = pd.read_excel('data/test.xlsx')
#print(df)
'''

'''

np.random.seed(42)

import pandas as pd
import numpy as np

np.random.seed(42)

data = np.random.randint(1, 101, size=(6, 5))  
df = pd.DataFrame(data, columns=['A', 'B', 'C', 'D', 'E'])

print(df.dtypes)
print()
print(df.astype(float).dtypes)
print()
print(df.astype(str).dtypes)
print()
print(df.astype('int32').dtypes)
print()
print(df.astype({'A': 'float32'}).dtypes) # convert dtypes of only specific columns in the dataframe

'''

data = pd.Series(['Setosa', 'Veriscolor', 'Virginica', 'Virginica'])
print(pd.get_dummies(data))
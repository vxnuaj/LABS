import pandas as pd
import pandas as pd

######## 1 #########

series = pd.Series(data = [10, 20, 30, 40, 50], index = ['a', 'b', 'c', 'd', 'e'])
series_c_loc = series.loc['c']
series_c_iloc = series.iloc[2]
series_first_3 = series.head(3)
series_last_2 = series.tail(2)

'''
print(f"Original Series:\n\n{series}\n")
print(f"'C' via Series.loc:\n{series_c_loc}\n")
print(f"'C' via Series.iloc:\n{series_c_iloc}\n")
print(f"First 3 values:\n{series_first_3}\n")
print(f"Last 2 values:\n{series_last_2}\n")
'''

######## 2 #########

data = {'Name': ['Alice', 'Bob', 'Charlie'], 'Age': [25, 30, 35], 'City': ['NY', 'SF', 'LA']}
df = pd.DataFrame(data = data)
df_row_2 = df.iloc[1, :]

'''
print(f"Original Data:\n{data}\n")
print(f"Original Dataframe:\n{df}\n")
print(f"Row 2:\n{df_row_2}\n")
print(f"Shape of Row 2:\n{df_row_2.shape}") # (3, ), it is now a series not a DataFrame
'''

######## 3 #########

df = pd.DataFrame({
    'A': [1, 2, 3, 4, 5],
    'B': [10, 20, 30, 40, 50],
    'C': [100, 200, 300, 400, 500]
})

df_greater_3 = df.query("A > 3")
# df_greater_3 = df[df.loc[:, 'A'] > 3] # CAN ALSO BE DONE USING PURE DataFrame.loc[]
df_first_2_rows = df.iloc[0:2, :]
sorted_df = df.sort_values('B', ascending = False)
reset_df = df.sort_index()

'''
print(f"Original DataFrame:\n{df}\n")
print(f"Rows where values in column A is greater than 3:\n{df_greater_3}\n")
print(f"First 2 rows:\n{df_first_2_rows}\n")
print(f"Sorted DataFrame by column B in descending order:\n{sorted_df}\n")
print(f"Resetted DataFrame:\n{reset_df}\n")
'''

######## 4 #########

'''
df = pd.DataFrame({
    'A': [1, 2, None, 4],
    'B': [None, 20, 30, 40],
    'C': [100, None, 300, None]
})

df_filled = df.fillna({'A': 0})
df_dropped = df_filled.dropna(subset = 'B')

print(f"Original DataFrame:\n{df}\n")
print(f"DataFrame with Column 'A' filled with 0s:\n{df_filled}\n")
print(f"DataFrame with rows removed based on NaN values in 'B':\n{df_dropped}\n")
'''

####### 5 #########

df = pd.DataFrame({
    'A': [1, 2, 2, 4],
    'B': [10, 20, 20, 40]
})

df_duplicates = df[df.duplicated()]
df_unique = df[~df.duplicated()]
df_removed = df.drop_duplicates()


'''
print(f"Original DF:\n{df}\n")
print(f"Duplicated Rows:\n{df_duplicates}\n")
print(f"Unique Rows:\n{df_unique}\n")
print(f"DataFrame with Dropped Duplicates:\n{df_removed}\n")
'''

####### 6 #########

df = pd.DataFrame({
    'Names': ['Alice', 'Bob', 'Charlie'],
    'City': ['New York', 'San Francisco', 'Los Angeles']
})

'''
print(f"Original DF:\n{df}\n")
df['City'] = df['City'].str.lower()
print(f"Lower City DF:\n{df}\n")
df['City'] = df['City'].str.replace(" ", "_")
print(f"Underscore City DF:\n{df}\n")
'''

####### 7 #########

'''
data = pd.read_csv('data/iris.csv')
data.to_csv('data/output.csv')
'''

####### 8 #########

df = pd.DataFrame({
    'ID': ['1', '2', '3'],
    'Value': [100.5, 200.3, 300.1]
})

# Convert the 'ID' column to integer
df['ID'] = df.loc[:, 'ID'].astype(int)

# Check the dtype of the 'ID' column
print(f"ID column dtype after conversion: {df['ID'].dtype}")


df = pd.DataFrame({
    'ID': ['1', '2', '3'],
    'Value': [100.5, 200.3, 300.1]
})

'''
print(f"Original DF:\n{df}\n")
df.loc[:, 'ID'] = df.loc[:, 'ID'].astype(int)
print(f"INT DF:\n{df}\n")
'''

####### 8 #########

df = pd.DataFrame({
    'Animal': ['Dog', 'Cat', 'Dog', 'Bird']
})

df_encoded = pd.get_dummies(data = df, dtype = int)

#print(f"Encoded DF: {df_encoded}")


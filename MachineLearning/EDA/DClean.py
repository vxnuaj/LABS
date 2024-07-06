import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

data = pd.read_csv('data/outliers.csv')

'''
sns.boxplot(data, orient='h')
plt.show()
'''

# PRINTING OUTLIERS
# outliers for 'value' in outliers.csv
qr25, qr50, qr75 = np.percentile(data.iloc[:, 1], [25, 50, 75])
iqr = qr75 - qr25
outlier_min = qr25 - 1.5 * iqr
outlier_max = qr75 + 1.5 * iqr

'''print(outlier_min, qr25, qr50, qr75, outlier_max)
print(f"Min Outliers: {[x for x in data['Value'] if x < outlier_min]}")
print(f"Max Outleirs: {[x for x in data['Value'] if x > outlier_max]}")
'''
# outliers for 'value2' in outliers.csv 

q25, q50, q75 = np.percentile(data.iloc[:, 2], [25, 50, 75])
iqr = q75 - q25
outlier_min = qr25 - 1.5 * iqr
outliers_max = qr75 + 1.5 * iqr

print(outlier_min, q25, q50, q75, outlier_max)
print(f"Min Outliers: {[x for x in data['Value2'] if x < outlier_min]} ")
print(f"Max Outliers: {[x for x in data['Value2'] if x > outlier_max]}")


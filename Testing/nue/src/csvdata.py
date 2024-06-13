import numpy as np
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)

# Number of samples
n = 2000

# Generate random features
feature1 = np.random.uniform(low=0, high=10, size=n)
feature2 = np.random.uniform(low=0, high=10, size=n)
feature3 = np.random.uniform(low=0, high=10, size=n)
feature4 = np.random.uniform(low=0, high=10, size=n)

# Generate target variable as a linear combination of features with some noise
target = 2.5 * feature1 - 1.3 * feature2 + 0.7 * feature3 - .5 * feature4 + np.random.normal(loc=0, scale=2, size=n)

# Create a DataFrame
data = {
    'Feature1': feature1,
    'Feature2': feature2,
    'Feature3': feature3,
    'Feature4': feature4,
    'Target': target
}
df = pd.DataFrame(data)

# Save DataFrame to CSV
df.to_csv('linear_regression_dataset2.csv', index=False)

# Display the first few rows of the dataset
print(df.head())

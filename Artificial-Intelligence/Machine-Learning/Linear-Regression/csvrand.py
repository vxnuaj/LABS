import numpy as np
import pandas as pd

# Number of data points
num_points = 200

# Generate random data
np.random.seed(0)
X = np.random.rand(num_points) * 100  # Random x values
noise = np.random.randn(num_points) * 10  # Random noise
y = 3 * X + 2 + noise  # Linear relationship y = 3X + 2 + noise

# Create a DataFrame
data = pd.DataFrame({'X': X, 'y': y})

# Save DataFrame to a CSV file
data.to_csv('random_linear_regression_data.csv', index=False)
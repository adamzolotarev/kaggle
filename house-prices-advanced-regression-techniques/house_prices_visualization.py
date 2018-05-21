import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


main_file_path = '~/.kaggle/competitions/house-prices-advanced-regression-techniques/train.csv'
dataset = pd.read_csv(main_file_path)
dataset = dataset.dropna(axis=1)

print(dataset['SalePrice'])

# Delete Outliers:
dataset.sort_values(by='GrLivArea', ascending=False)[:2]
print(dataset.sort_values(by='GrLivArea', ascending=False)[:2])
# dataset = dataset.drop(dataset[dataset['Id'] == 1299].index)
# dataset = dataset.drop(dataset[dataset['Id'] == 524].index)

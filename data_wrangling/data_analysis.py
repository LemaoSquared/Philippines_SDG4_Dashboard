# # Author: Waken Cean C. Maclang
# Date: December 19, 2025
# Course: Applied Data Science
# Task: Learning Evidence

# data_analysis.py
#     The python file used to explore the 4 uncleaned and 1 merged & cleaned dataset

# Works with Python 3.10.0+

import numpy as np
import pandas as pd
import polars as pl

MAIN_PATH = 'data_wrangling\\'
dataset = []
for path in ['dataset1_completion_rate.csv', 'dataset2_cohort_survival_rate.csv', 'dataset3_gender_parity_index.csv', 'dataset4_participation_rate.csv']:
    dataset.append(pd.read_csv(
        MAIN_PATH + path,
        sep=";",
        skiprows=2,              
    ))

data1, data2, data3, data4 = dataset
data1.head()
(data1 == '..').sum()
(data1 == '...').sum()
type(data1.iloc[0, 4])

for data in [data1, data2, data3, data4]:
    print(data.columns)
    data = data.replace('..', 0)
    data = data.replace('...', 0)

    cols = data.columns[4:]
    data[cols] = data[cols].apply(lambda x: pd.to_numeric(x))

data.head()

main_dataset = pd.read_csv(MAIN_PATH + 'Cleaned_Philippines_Education_Statistics.csv')
main_dataset.columns
np.unique(main_dataset['Geolocation'])

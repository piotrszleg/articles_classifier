import pandas as pd

fields = pd.read_csv('data/y_fields_filtered.csv')

result = {}

for field in fields:
    result[field] = list(pd.get_dummies(fields[field]).columns)

print(result)
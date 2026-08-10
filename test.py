import pandas as pd

data = pd.read_csv('test_probe_table.csv')

print(data[['nearest_in_O__distance','nearest_any__distance','ego__v_e']].describe())


import pandas as pd

chosen_teams = ['-', 'Connecticut Sun', 'New England Patriots', 'Detroit Lions', 'Pittsburgh Steelers', 'Boston Red Sox', 'Chicago Cubs', 'New York Yankees', 'University of Alabama', 'University of Arkansas', 'Seattle Mariners', 'Dallas Cowboys', 'New York Giants', 'Kansas City Chiefs', 'Texas Rangers', 'Los Angeles Dodgers', 'Arizona Diamondbacks', 'Miami Dolphins', 'Green Bay Packers', 'Chicago White Sox']

ys = pd.read_csv('data/y_fields.csv')

for i, row in ys.iterrows():
    if row["Team"] not in chosen_teams:
        ys.at[i,"Team"] = "-"

ys.to_csv("data/y_fields_filtered.csv", index=False)
import pandas as pd
import matplotlib.pyplot as plt

cut = 20

ys = pd.read_csv('data/y_fields.csv')
c = ys["Team"].value_counts()
other = c.size - cut
c = c.head(cut)
print("Columns:", list(c.index))
print("Number of classes is", c.size+1)
print("Smallest count is", c[-1])
print("Mean is", c.drop("-").mean())
c["other"] = other
c.plot(kind="bar")# kind="pie")
plt.show()

"""
Chosen result is:

Columns: 
['-', 'Connecticut Sun', 'New England Patriots', 'Detroit Lions', 'Pittsburgh Steelers', 'Boston Red Sox', 'Chicago Cubs', 'New York Yankees', 'University of Alabama', 'University of Arkansas', 'Seattle Mariners', 'Dallas Cowboys', 'New York Giants', 'Kansas City Chiefs', 'Texas Rangers', 'Los Angeles Dodgers', 'Arizona Diamondbacks', 'Miami Dolphins', 'Green Bay Packers', 'Chicago White Sox']Number of classes is 21
Smallest count is 42
Mean is 54.89473684210526
"""
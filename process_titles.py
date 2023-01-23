from process_text import process_text
import pandas as pd

x = pd.read_csv('data/x.csv')

x["title"] = x.apply(lambda row: process_text(row["title"]), 1)

x.to_csv("data/x_processed.csv", index=False)
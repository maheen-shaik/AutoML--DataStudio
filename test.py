import pandas as pd
from cleaner import DataCleaner

df = pd.read_csv("your_dataset.csv")

cleaner = DataCleaner(df)
cleaned_df = cleaner.clean()

print(cleaner.report())
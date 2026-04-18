import pandas as pd
import glob
import os

folder_path = "./merge/"

# get all CSV files in the folder
csv_files = glob.glob(os.path.join(folder_path, "*.csv"))

# read and combine
df = pd.concat((pd.read_csv(file) for file in csv_files), ignore_index=True)

# optional cleanup
df = df.drop_duplicates()

# save result
df.to_csv(os.path.join(folder_path, "teams_final.csv"), index=False)

print(f"Merged {len(csv_files)} files successfully.")
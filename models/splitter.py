import pandas as pd

# Path to the dataset
file_path = 'original_cleaned_nyc_taxi_data_2018.csv'  # Replace with the actual file path
output_paths = [
    'dataset_part1.csv',
    'dataset_part2.csv',
    'dataset_part3.csv',
    'dataset_part4.csv'
]

# Read the dataset
print("Reading the dataset...")
df = pd.read_csv(file_path)

# Split into four parts
print("Splitting the dataset...")
num_rows = len(df)
split1 = num_rows // 4
split2 = 2 * split1
split3 = 3 * split1

df_part1 = df.iloc[:split1]
df_part2 = df.iloc[split1:split2]
df_part3 = df.iloc[split2:split3]
df_part4 = df.iloc[split3:]

# Save each part
print("Saving the splits...")
df_part1.to_csv(output_paths[0], index=False)
df_part2.to_csv(output_paths[1], index=False)
df_part3.to_csv(output_paths[2], index=False)
df_part4.to_csv(output_paths[3], index=False)

print("Split and save completed!")

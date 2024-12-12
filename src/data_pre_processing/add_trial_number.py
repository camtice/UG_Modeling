import pandas as pd

# Read the CSV file
df = pd.read_csv('data/ground/UG_raw_data.csv')

# Group by ID and assign trial numbers within each group
df['trial_number'] = df.groupby('ID').cumcount() + 1

# Reorder columns to make 'trial_number' the third column
cols = df.columns.tolist()
cols.insert(2, cols.pop(cols.index('trial_number')))  # Move 'trial_number' to the third position
df = df[cols]

# Save the modified dataframe
df.to_csv('data/ground/UG_raw_data_with_trials.csv', index=False)
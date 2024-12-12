import pandas as pd

# Read the raw data
df = pd.read_csv('UG_raw_data_with_trials.csv')

# Filter for role 1 trials
role1_df = df[df['trial_role'] == 1].copy()

# Function to save grouped data
def save_grouped_data(df, group_size, filename):
    grouped_df = df.groupby('ID').head(group_size)
    grouped_df.to_csv(filename, index=False)

# Save first 5, 10, 20, and 30 responses for each participant
save_grouped_data(role1_df, 5, 'UG_first5_role1.csv')
save_grouped_data(role1_df, 10, 'UG_first10_role1.csv')
save_grouped_data(role1_df, 20, 'UG_first20_role1.csv')
save_grouped_data(role1_df, 30, 'UG_first30_role1.csv')

# Create a dataset with all role 1 responses
all_role1_df = role1_df.copy()

# Save to new CSV file
all_role1_df.to_csv('UG_all_role1.csv', index=False)
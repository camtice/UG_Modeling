import pandas as pd

# Load the CSV file
data = pd.read_csv('ground/UG_raw_data_with_trials.csv')

# Split the data into two groups
control_group = data[data['group'] == 'Control']
cocaine_group = data[data['group'] == 'Cocaine']

# Save each group to a separate CSV file
control_group.to_csv('ground/UG_control_group.csv', index=False)
cocaine_group.to_csv('ground/UG_cocaine_group.csv', index=False)

print("Data has been split and saved into 'UG_control_group.csv' and 'UG_cocaine_group.csv'.")
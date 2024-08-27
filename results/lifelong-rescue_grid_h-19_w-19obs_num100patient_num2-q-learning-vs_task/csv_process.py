import pandas as pd

input_file_name = 'SR_LLRL_result.csv'

# remove the 'result':
truncatad_file_name = input_file_name[:-4-6]
output_file_name = truncatad_file_name + '_transformed.csv'
# Load the original CSV data
df = pd.read_csv(input_file_name)

# Pivot the data
df_pivot = df.pivot(index='task', columns='episode', values='return')

# Reset index to remove the task labels and drop the index entirely
df_pivot.reset_index(drop=True, inplace=True)

# Save the transformed data to a new CSV without headers and index
df_pivot.to_csv(output_file_name, index=False, header=False)

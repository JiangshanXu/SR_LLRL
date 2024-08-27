import pandas as pd
import matplotlib.pyplot as plt
import glob

# Function to calculate success rates
def calculate_success_rates(df):
    success_rates = {}
    for col in df.columns[1:-1]:
        success_rates[col] = df[col].value_counts(normalize=True).get(True, 0) * 100
    return success_rates

# Read the CSV files and calculate success rates
files = ['four_room_task_success.csv', 'lava_task_success.csv', 'maze_task_success.csv']
# add directory to files:
dir = './results'
files = [dir + '/' + file for file in files]
env_names = ['Four Room', 'Lava', 'Maze']
success_rates = {}

for file, env_name in zip(files, env_names):
    df = pd.read_csv(file)
    success_rates[env_name] = calculate_success_rates(df)

# Convert the success rates to a DataFrame for easier plotting
success_df = pd.DataFrame(success_rates).T

# Plotting
ax = success_df.plot(kind='bar', figsize=(10, 6), width=0.8)
ax.set_xlabel('Environment')
ax.set_ylabel('Success Rate (%)')
ax.set_title('Algorithm Success Rates in Different Environments')
ax.legend(title='Algorithm')
plt.xticks(rotation=0)
plt.tight_layout()

# Show the plot
# plt.show()

# save:
plt.savefig('success_rates.png')

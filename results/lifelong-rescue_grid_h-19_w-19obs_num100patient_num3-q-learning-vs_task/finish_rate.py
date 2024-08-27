import pandas as pd

# List of CSV files for the five algorithms
csv_files = [
    'SR_dyna_result.csv',
    'SR_Sarsa(5)_result.csv',
    'sarsa_lambda2_SR_result.csv',
    'SR_sarsa_λ_result.csv',
    'SR_LLRL_result.csv'
]

# Total number of tasks
total_tasks = 40

finish_tasks_list = []
finish_tasks_rate_list = []
# Iterate over each CSV file
for i, csv_file in enumerate(csv_files, 1):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(csv_file)

    df = df[df['episode'] <= 200]
    # Group by 'task' and check if any 'return' is non-zero
    finished_tasks = df.groupby('task')['return'].apply(lambda x: any(x > 0)).sum()

    # Calculate the task finish rate
    finish_rate = finished_tasks / total_tasks
    finish_tasks_list.append(finished_tasks)
    finish_tasks_rate_list.append(finish_rate)

    # Print the results for the current algorithm
    print(f"Algorithm {i}:")
    print(f"  Finished Tasks: {finished_tasks}")
    print(f"  Task Finish Rate: {finish_rate:.2%}\n")

print('done')
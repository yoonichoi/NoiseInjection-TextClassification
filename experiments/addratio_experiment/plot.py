import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def read_and_split_datasets(file_path):
    full_df = pd.read_csv(file_path)
    dataset_names = full_df.iloc[:, 0].unique()
    datasets = {name: full_df[full_df.iloc[:, 0] == name] for name in dataset_names}
    return datasets

def convert_to_numeric(df):
    return df.apply(pd.to_numeric, errors='coerce')

file_names = [
    "experiments/addratio_experiment/outputs/initial/result_0.csv", 
    "experiments/addratio_experiment/outputs/initial/result_1.csv", 
    "experiments/addratio_experiment/outputs/initial/result_2.csv", 
    "experiments/addratio_experiment/outputs/initial/result_3.csv", 
    "experiments/addratio_experiment/outputs/initial/result_4.csv"
]

all_datasets = [read_and_split_datasets(file_name) for file_name in file_names]

average_datasets = {}
for dataset_name in all_datasets[0].keys():
    numeric_datasets = [convert_to_numeric(datasets[dataset_name]) for datasets in all_datasets]
    average_datasets[dataset_name] = pd.concat(numeric_datasets).groupby(level=0).mean()

# Combine averaged datasets into a single dataframe
combined_averages_df = pd.concat(average_datasets).reset_index(level=0)
#combined_averages_df = combined_averages_df.drop(columns=['level_0'])
# # Print current column names to check
# print("Current column names:", combined_averages_df.columns)
# print("Number of columns:", len(combined_averages_df.columns))
combined_averages_df.columns = ['Extra','Dataset Name', 'Add_Ratio','Aeda_Acc', 'Num_Acc', 'Alpha_Acc', 'Hybrid_Acc']
combined_averages_df = combined_averages_df.drop(columns=['Dataset Name'])
combined_averages_df.columns = ['Dataset', 'Add_Ratio', 'Aeda_Acc', 'Num_Acc', 'Alpha_Acc', 'Hybrid_Acc']
combined_averages_df = combined_averages_df.dropna()
# Save the combined dataframe to a CSV file
combined_averages_df.to_csv("experiments/addratio_experiment/outputs/initial/aggregated_results.csv", index=False)

# Group by 'Increment' and calculate the mean for each group
grouped = combined_averages_df.drop(columns=['Dataset']).groupby('Add_Ratio').mean()
#print(grouped)

# Plotting each metric across increments
plt.figure(figsize=(10, 6))

# Assuming your metrics columns are named 'Orig_Acc', 'Eda_Acc', 'Aeda_Acc', 'Num_Acc', 'Alpha_Acc', 'Hybrid_Acc'
# Adjust the column names if they are different
metrics = ['Aeda_Acc', 'Num_Acc', 'Alpha_Acc', 'Hybrid_Acc']
for metric in metrics:
    plt.plot(grouped.index, grouped[metric], marker='o', label=metric)

plt.title('Average Metrics Across All Datasets Grouped by Add Ratio')
plt.xlabel('Add Ratio')
plt.ylabel('Average Metric Value')
plt.legend()
plt.grid(True)

#plt.yticks(np.arange(0.6, 0.9, 0.02))

# Save the plot to a placeholder path
placeholder_path = 'experiments/addratio_experiment/outputs/initial/plots/average_metrics_across_addratio.png'
plt.savefig(placeholder_path)




























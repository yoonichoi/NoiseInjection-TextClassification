import pandas as pd
import matplotlib.pyplot as plt
import glob

folder_path = 'reproduce_fig2/outputs'
file_paths = []
for sd in range(5):
    file_paths.extend(glob.glob(f'{folder_path}/result_{sd}.csv'))

dfs = [pd.read_csv(file_path, sep=' ', names=['Dataset', 'Percentage', 'Original', 'EDA', 'AEDA']) for file_path in file_paths]
aggregated_df = pd.concat(dfs, ignore_index=True)

average_df = aggregated_df.groupby(['Dataset', 'Percentage']).mean().reset_index()

average_df.to_csv(f'{folder_path}/aggregated_results.csv', index=False)

datasets = ['sst2', 'cr', 'subj', 'trec', 'pc']
# datasets = ['sst2', 'cr']

fig, axes = plt.subplots(1, len(datasets), figsize=(15, 5), sharey=True)

for i, dataset in enumerate(datasets):
    dataset_df = average_df[average_df['Dataset'] == dataset]

    axes[i].plot(dataset_df['Percentage'], dataset_df['Original'], label=f'Original')
    axes[i].plot(dataset_df['Percentage'], dataset_df['EDA'], label=f'EDA')
    axes[i].plot(dataset_df['Percentage'], dataset_df['AEDA'], label=f'AEDA')

    axes[i].set_xlabel('Percentage of Data')
    axes[i].set_ylabel('Accuracy')

    title = {'cr':'CR', 'pc':'PC', 'sst2':'SST-2', 'subj':'SUBJ','trec':'TREC'}
    axes[i].set_title(title[dataset])
    axes[i].legend()

# Save the figure with all subplots
plt.savefig(f'{folder_path}/accuracy_trend.png')
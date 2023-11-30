import pandas as pd
import matplotlib.pyplot as plt
import glob

from PIL import Image
import sys

basepath = 'experiments/addratio_experiment/outputs'
folder_path = f'{basepath}/{sys.argv[1]}'

file_paths = []
for sd in range(5):
    file_paths.extend(glob.glob(f'{folder_path}/result_{sd}.csv'))
dfs = [pd.read_csv(file_path, sep=',', names=['Dataset', 'Add_Ratio', 'Aeda_Acc', 'Num_Acc', 'Alpha_Acc', 'Hybrid_Acc']) for file_path in file_paths]
aggregated_df = pd.concat(dfs, ignore_index=True)

average_df = aggregated_df.groupby(['Dataset', 'Add_Ratio']).mean().reset_index()

average_df.to_csv(f'{folder_path}/aggregated_results.csv', index=False)

orig_acc_df = pd.read_csv(f'{folder_path}/orig_acc.csv', sep=',', names=['Dataset', 'Original_Acc'])
orig_acc_df = orig_acc_df.groupby(['Dataset']).mean().reset_index()

datasets = ['sst2', 'cr', 'subj', 'trec', 'pc']

for i, dataset in enumerate(datasets):
    plt.figure(figsize=(5, 5))  # Create a new figure for each plot

    dataset_df = average_df[average_df['Dataset'] == dataset]
    orig_acc = orig_acc_df[orig_acc_df['Dataset'] == dataset]['Original_Acc'].values[0]

    # Find the minimum and maximum values across the 'Original', 'EDA', and 'AEDA' columns for the current dataset

    y_min = min(dataset_df[['Aeda_Acc', 'Num_Acc', 'Alpha_Acc', 'Hybrid_Acc']].min().min(), orig_acc) - 0.005
    y_max = max(dataset_df[['Aeda_Acc', 'Num_Acc', 'Alpha_Acc', 'Hybrid_Acc']].max().max(), orig_acc) + 0.005

    plt.axhline(y=orig_acc, linestyle='--', color='grey', linewidth=2.5, label='Original Acc')

    plt.plot(dataset_df['Add_Ratio'], dataset_df['Aeda_Acc'], label=f'AEDA', marker='o')
    plt.plot(dataset_df['Add_Ratio'], dataset_df['Num_Acc'], label=f'Num_Aug', marker='o')
    plt.plot(dataset_df['Add_Ratio'], dataset_df['Alpha_Acc'], label=f'Alpha_Aug', marker='o')
    plt.plot(dataset_df['Add_Ratio'], dataset_df['Hybrid_Acc'], label=f'Hybrid', marker='o')

    plt.xlabel('Add Ratio')
    plt.ylabel('Accuracy')

    title = {'cr':'CR', 'pc':'PC', 'sst2':'SST-2', 'subj':'SUBJ','trec':'TREC'}
    plt.title(title[dataset])
    plt.ylim(ymin=y_min, ymax=y_max)
    plt.grid(True)  # Add grid lines
    plt.legend()

    # Save each figure separately
    plt.savefig(f'{folder_path}/plots/accuracy_trend_{dataset}.png')




folder_path = f'{folder_path}/plots'
datasets = ['sst2', 'cr', 'subj', 'trec', 'pc']

# Open the images and resize them to have the same height
images = [Image.open(f'{folder_path}/accuracy_trend_{dataset}.png') for dataset in datasets]
widths, heights = zip(*(i.size for i in images))
max_height = max(heights)
resized_images = [img.resize((int(img.width * max_height / img.height), max_height)) for img in images]

# Combine the images side by side
total_width = sum(img.width for img in resized_images)
combined_img = Image.new('RGB', (total_width, max_height))

x_offset = 0
for img in resized_images:
    combined_img.paste(img, (x_offset, 0))
    x_offset += img.width

# Save the combined image
combined_img.save(f'{folder_path}/combined_accuracy_trend.png')
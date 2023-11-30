import pandas as pd
import matplotlib.pyplot as plt
import glob

from PIL import Image
import sys

basepath = 'experiments/increments_experiment/outputs'
folder_path = f'{basepath}/{sys.argv[1]}'

file_paths = []
for sd in range(5):
    file_paths.extend(glob.glob(f'{folder_path}/result_{sd}.csv'))

dfs = [pd.read_csv(file_path, sep=',', names=['Dataset', 'Increment', 'Orig_Acc', 'Eda_Acc', 'Aeda_Acc', 'Num_Acc', 'Alpha_Acc', 'Hybrid_Acc']) for file_path in file_paths]
aggregated_df = pd.concat(dfs, ignore_index=True)

average_df = aggregated_df.groupby(['Dataset', 'Increment']).mean().reset_index()

average_df.to_csv(f'{folder_path}/aggregated_results.csv', index=False)

datasets = ['sst2', 'cr', 'subj', 'trec', 'pc']

for i, dataset in enumerate(datasets):
    plt.figure(figsize=(5, 5))  # Create a new figure for each plot

    dataset_df = average_df[average_df['Dataset'] == dataset]

    # Find the minimum and maximum values across the 'Original', 'EDA', and 'AEDA' columns for the current dataset
    y_min = dataset_df[['Orig_Acc', 'Eda_Acc', 'Aeda_Acc', 'Num_Acc', 'Alpha_Acc', 'Hybrid_Acc']].min().min() - 0.005
    y_max = dataset_df[['Orig_Acc', 'Eda_Acc', 'Aeda_Acc', 'Num_Acc', 'Alpha_Acc', 'Hybrid_Acc']].max().max() + 0.005

    plt.plot(dataset_df['Increment'], dataset_df['Orig_Acc'], label=f'Original', marker='o')
    plt.plot(dataset_df['Increment'], dataset_df['Eda_Acc'], label=f'EDA', marker='o')
    plt.plot(dataset_df['Increment'], dataset_df['Aeda_Acc'], label=f'AEDA', marker='o')
    plt.plot(dataset_df['Increment'], dataset_df['Num_Acc'], label=f'Num_Aug', marker='o')
    plt.plot(dataset_df['Increment'], dataset_df['Alpha_Acc'], label=f'Alpha_Aug', marker='o')
    plt.plot(dataset_df['Increment'], dataset_df['Hybrid_Acc'], label=f'Hybrid', marker='o')

    plt.xlabel('Percentage of Data')
    plt.ylabel('Accuracy')

    title = {'cr':'CR', 'pc':'PC', 'sst2':'SST-2', 'subj':'SUBJ','trec':'TREC'}
    plt.title(title[dataset])
    plt.ylim(ymin=y_min, ymax=y_max)
    plt.grid(True)  # Add grid lines
    plt.legend()

    # Save each figure separately
    if os.path.exists(f'{folder_path}/plots') == False:
        os.mkdir(f'{folder_path}/plots')
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
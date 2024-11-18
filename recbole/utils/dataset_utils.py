import os
import pandas as pd

def split_and_save_dataset(input_file_path, output_folder_path):
    # Read the dataset
    df = pd.read_csv(input_file_path, sep='\t', header=0, names=['user_id', 'item_id', 'rating', 'timestamp'])
    subset_size = len(df) // 10

    # Create subsets
    subsets = []
    for i in range(10):
        start_index = i * subset_size
        if i == 9:
            subsets.append(df.iloc[start_index:])
        else:
            subsets.append(df.iloc[start_index:start_index + subset_size])

    # Ensure output folder exists
    os.makedirs(output_folder_path, exist_ok=True)

    # Save each subset
    for i, subset in enumerate(subsets):
        subset_filename = f'{os.path.splitext(os.path.basename(input_file_path))[0]}_subset_{i+1}' #.csv 
        output_file_path = os.path.join(output_folder_path, subset_filename)
        subset.to_csv(output_file_path, index=False, sep='\t')

    print(f'Subsets saved in {output_folder_path}')

def process_datasets(main_folder):
    # Walk through each folder in the main directory
    for root, dirs, files in os.walk(main_folder):
        for file in files:
            if 'inter' in file:
                input_file_path = os.path.join(root, file)
                output_folder_path = os.path.join(root, 'inter_subsets')
                try:
                    split_and_save_dataset(input_file_path, output_folder_path)
                except:
                    continue

                
                


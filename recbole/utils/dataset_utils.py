import os
import pandas as pd

def split_save_dataset(input_file_path, output_folder_path):
    # Read the dataset
    df = pd.read_csv(input_file_path, sep='\t', header=0, names=['user_id:token', 'item_id:token', 'rating:float', 'timestamp:float'])
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
        subset_filename = f'subset_{i+1}.inter'
        output_file_path = os.path.join(output_folder_path, subset_filename)
        subset.to_csv(output_file_path, index=False, sep='\t')

    print(f'Subsets saved in {output_folder_path}')


def split_filter_save_dataset(input_file_path, output_folder_path):
    # Read the dataset
    df = pd.read_csv(input_file_path, sep='\t', header=0, names=['user_id:token', 'item_id:token', 'rating:float', 'timestamp:float'])
    subset_size = len(df) // 10

    # Create subsets and apply filter
    for i in range(10):
        start_index = i * subset_size
        if i == 9:
            subset = df.iloc[start_index:]
        else:
            subset = df.iloc[start_index:start_index + subset_size]
        
        # Filter items occurring less than 10 times
        item_counts = subset['item_id:token'].value_counts()
        items_to_keep = item_counts[item_counts >= 10].index
        filtered_subset = subset[subset['item_id:token'].isin(items_to_keep)]
        
        # Ensure the output folder exists
        os.makedirs(output_folder_path, exist_ok=True)

        # Save the filtered subset
        subset_filename = f'subset_{i+1}.inter'
        output_file_path = os.path.join(output_folder_path, subset_filename)
        filtered_subset.to_csv(output_file_path, index=False, sep='\t')

    print(f'Filtered subsets saved in {output_folder_path}')
                
                


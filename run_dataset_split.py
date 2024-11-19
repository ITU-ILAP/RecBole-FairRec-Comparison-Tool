from recbole.utils.dataset_utils import split_save_dataset, split_filter_save_dataset

input_file_path = 'dataset_v2\ml-1M\ml-1M.inter'
output_folder_path = 'dataset_v2\ml-1M\inter_subsets'
split_save_dataset(input_file_path, output_folder_path)

filtered_output_folder_path = 'dataset_v2\ml-1M\inter_subsets_filtered'
split_filter_save_dataset(input_file_path, filtered_output_folder_path)
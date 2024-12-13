import argparse
import pickle
import sys, os
import time
from recbole.config import Config
from recbole.quick_start import run_recbole
from recbole.data import data_preparation, create_dataset

if __name__ == '__main__':
    subset_list = [f"sample_{i}" for i in range(55, 601)]
    subset_folder_name = "URM_subsets_filtered"
    #subset_list = ["subset_1", "subset_2", "subset_3", "subset_4", "subset_5",
    #"subset_6", "subset_7", "subset_8", "subset_9", "subset_10"]
    #subset_folder_name = "inter_subsets_filtered"
    start_time = time.time()
    for subset_name in subset_list:
        # Argument parsing
        parser = argparse.ArgumentParser()
        parser.add_argument('--dataset', '-d', type=str, default='BX')
        parser.add_argument('--config_files', '-c', type=str, default='test.yaml')
        args = parser.parse_args()

        config_file_list = args.config_files.strip().split(' ') if args.config_files else None

        model_list_2 = ["PFCN_MLP"]
        model_list = ["NFCF", "FOCF", "PFCN_MLP", "FairGo_PMF"]

        # Step 1: Split the dataset once using a sample model configuration
        sample_config = Config(model=model_list_2[0], dataset=args.dataset, config_file_list=config_file_list)
        sample_config["data_path"] ='dataset_v2/BX'
        sample_config["data_path_inter"] = f'dataset_v2/BX/{subset_folder_name}/{subset_name}.inter'
        dataset = create_dataset(sample_config)
        train_data, valid_data, test_data = data_preparation(sample_config, dataset)

        # Step 2: Run each model with its own configuration and the pre-split data
        for smodel in model_list:
            if(subset_name == 'sample_55' and (smodel == "FOCF" or smodel == "NFCF")):
                continue
            """
            if (smodel + ".txt") in files:
                continue
            """
            # Create a new config for each model to ensure model-specific parameters are loaded
            config = Config(model=smodel, dataset=args.dataset, config_file_list=config_file_list)

            # Run the model using the pre-split data
            result = run_recbole(
                model=smodel, dataset=args.dataset, config_file_list=config_file_list,
                train_data=train_data, valid_data=valid_data, test_data=test_data
            )

            # Save the result
            #results_ml1m_URM_filtered_gender
            path = f"results/results_BX_URM_filtered_age/result_{subset_name}_{smodel}.txt"
            with open(path, 'wb') as handle:
                pickle.dump(result, handle)
    print("Total Time: ", time.time()-start_time)

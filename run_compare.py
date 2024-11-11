import argparse
import pickle
import sys, os

from recbole.config import Config
from recbole.quick_start import run_recbole
from recbole.data import data_preparation, create_dataset

if __name__ == '__main__':

    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', type=str, default='ml-1M')
    parser.add_argument('--config_files', '-c', type=str, default='test.yaml')
    args = parser.parse_args()

    config_file_list = args.config_files.strip().split(' ') if args.config_files else None
    
    model_list_2 = ["PFCN_MLP"]
    model_list = ["FOCF","PFCN_MLP", "PFCN_BiasedMF", "PFCN_DMF",  "PFCN_PMF", "FairGo_PMF"]
    #"PFCN_BiasedMF", "PFCN_DMF", "PFCN_PMF"
    files = os.listdir("results/comparison/")

    # Step 1: Split the dataset once using a sample model configuration
    sample_config = Config(model=model_list_2[0], dataset=args.dataset, config_file_list=config_file_list)
    dataset = create_dataset(sample_config)
    train_data, valid_data, test_data = data_preparation(sample_config, dataset)

    # Step 2: Run each model with its own configuration and the pre-split data
    for smodel in model_list:
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
        path = f"results/comparison/{smodel}.txt"
        with open(path, 'wb') as handle:
            pickle.dump(result, handle)

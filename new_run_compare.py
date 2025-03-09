import argparse
import pickle
import os
import time
from recbole.config import Config
from recbole.quick_start import run_recbole
from recbole.data import data_preparation, create_dataset

def update_sensitive(config, sensitive_feature):
    """
    Override the sensitive attribute settings in the config:
      - Set sst_attr_list to [sensitive_feature]
      - In load_col["user"], replace any occurrence of known sensitive attributes (e.g., "age" or "gender")
        with the current sensitive_feature.
    """
    config["sst_attr_list"] = [sensitive_feature]
    if "load_col" in config and "user" in config["load_col"]:
        # List of possible sensitive attribute names that might appear in the user columns.
        possible_sensitive = ["age", "gender"]
        new_user_list = []
        for col in config["load_col"]["user"]:
            if col.lower() in possible_sensitive:
                new_user_list.append(sensitive_feature)
            else:
                new_user_list.append(col)
        config["load_col"]["user"] = new_user_list

"""
This driver code is fully self-contained. When executed, it:
  - Iterates over 60 dataset subsets.
  - Runs experiments on two datasets: ml-1M and BX.
      * For ml-1M, it runs experiments twice (once with "age" and once with "gender" as the sensitive feature);
      * For BX, it uses "age" only.
  - Runs all non-FOCF models:
        FairGo_GCN, FairGo_PMF, NFCF,
        PFCN_BiasedMF, PFCN_DMF, PFCN_MLP, PFCN_PMF
  - Runs all FOCF submodels (each defined by a different fair objective: none, value, absolute, under, over, nonparity).
  - Overrides evaluation parameters: topk is set to [10] and valid_metric to "NDCG@10".
  - Overrides the sensitive attribute list (sst_attr_list) and also updates the load_col["user"] 
    so that the sensitive column in the user part matches the current sensitive feature.
  - Saves result metrics and the trained model (if available) in dataset-specific directories.
  - Prints checkpoints so the user can monitor progress.
  
Importantly, the dataset split is created only once per subset (and sensitive feature) and reused for all model runs,
ensuring consistency. If a result file already exists for a given run, that run is skipped.
All necessary folders are created automatically.
"""

if __name__ == '__main__':
    # Define datasets to run experiments on.
    datasets = ["BX"]

    # Define dataset subsets and folder name for intermediate files.
    subset_list = [f"sample_{i}" for i in range(1, 61)]
    total_subsets = len(subset_list)
    subset_folder_name = "URM_subsets_filtered"
    start_time = time.time()

    # Parse command-line arguments for config files.
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_files', '-c', type=str, default='test.yaml')
    args = parser.parse_args()
    config_file_list = args.config_files.strip().split(' ') if args.config_files else None

    # Define the list of non-FOCF models.
    models_to_run = [
        "FairGo_GCN", "FairGo_PMF", "NFCF",
        "PFCN_BiasedMF", "PFCN_DMF", "PFCN_MLP", "PFCN_PMF"
    ]
    # Define FOCF fair objectives (submodels).
    focf_fair_objectives = ["none", "value", "absolute", "under", "over", "nonparity"]

    # Compute total iterations for progress reporting.
    overall_count = 0
    total_iterations = 0
    for dataset_name in datasets:
        sens = ["age"]
        total_iterations += len(sens) * total_subsets * (len(models_to_run) + len(focf_fair_objectives))
    print(f"Starting experiments for datasets {datasets}. Total iterations: {total_iterations}.")

    # Loop over each dataset.
    for dataset_name in datasets:
        # Set data path based on dataset name.
        data_path = f"dataset_v2/{dataset_name}"
        # Set sensitive features for this dataset.
        sensitive_features = ["age"]
        print(f"\n==== Processing dataset: {dataset_name} ====")
        for s_index, sensitive_feature in enumerate(sensitive_features, 1):
            print(f"\nProcessing sensitive feature [{s_index}/{len(sensitive_features)}]: '{sensitive_feature}'.")
            for i, subset_name in enumerate(subset_list, 1):
                print(f"\n  Processing subset [{i}/{total_subsets}]: {subset_name}.")

                # STEP 1: Create the dataset split once per subset.
                sample_config = Config(model="PFCN_MLP", dataset=dataset_name, config_file_list=config_file_list)
                sample_config["data_path"] = data_path
                sample_config["data_path_inter"] = f"{data_path}/{subset_folder_name}/{subset_name}.inter"
                sample_config["topk"] = [10]
                sample_config["valid_metric"] = "NDCG@10"
                # Force override of sst_attr_list and update load_col["user"]
                update_sensitive(sample_config, sensitive_feature)
                dataset = create_dataset(sample_config)
                train_data, valid_data, test_data = data_preparation(sample_config, dataset)
                print(f"    Dataset split for subset '{subset_name}' created.")

                # STEP 2: Run each non-FOCF model.
                for smodel in models_to_run:
                    overall_count += 1
                    percent_complete = (overall_count / total_iterations) * 100
                    result_path = f"results/new_results_{dataset_name}_URM_filtered/result_{subset_name}_{smodel}_{sensitive_feature}.txt"
                    if os.path.exists(result_path):
                        print(f"    [{overall_count}/{total_iterations} | {percent_complete:.1f}%] Result for model '{smodel}' with sensitive feature '{sensitive_feature}' on subset '{subset_name}' already exists. Skipping training.")
                        continue
                    print(f"    [{overall_count}/{total_iterations} | {percent_complete:.1f}%] Running model '{smodel}' with sensitive feature '{sensitive_feature}' on subset '{subset_name}'.")
                    config = Config(model=smodel, dataset=dataset_name, config_file_list=config_file_list)
                    config["data_path"] = data_path
                    config["data_path_inter"] = f"{data_path}/{subset_folder_name}/{subset_name}.inter"
                    config["topk"] = [10]
                    config["valid_metric"] = "NDCG@10"
                    update_sensitive(config, sensitive_feature)
                    result = run_recbole(
                        model=smodel,
                        dataset=dataset_name,
                        config_file_list=config_file_list,
                        train_data=train_data,
                        valid_data=valid_data,
                        test_data=test_data
                    )
                    os.makedirs(os.path.dirname(result_path), exist_ok=True)
                    with open(result_path, 'wb') as handle:
                        pickle.dump(result, handle)
                    print(f"      Result saved to {result_path}.")
                    if 'best_model' in result:
                        model_path = f"models/model_{subset_name}_{smodel}_{sensitive_feature}.pkl"
                        os.makedirs(os.path.dirname(model_path), exist_ok=True)
                        with open(model_path, 'wb') as model_file:
                            pickle.dump(result['best_model'], model_file)
                        print(f"      Trained model saved to {model_path}.")

                # STEP 3: Run FOCF for each fair objective submodel.
                for fair_obj in focf_fair_objectives:
                    overall_count += 1
                    percent_complete = (overall_count / total_iterations) * 100
                    result_path = f"results/new_results_{dataset_name}_URM_filtered/result_{subset_name}_FOCF_{fair_obj}_{sensitive_feature}.txt"
                    if os.path.exists(result_path):
                        print(f"    [{overall_count}/{total_iterations} | {percent_complete:.1f}%] Result for FOCF with fair objective '{fair_obj}' and sensitive feature '{sensitive_feature}' on subset '{subset_name}' already exists. Skipping training.")
                        continue
                    print(f"    [{overall_count}/{total_iterations} | {percent_complete:.1f}%] Running FOCF with fair objective '{fair_obj}' and sensitive feature '{sensitive_feature}' on subset '{subset_name}'.")
                    config = Config(model="FOCF", dataset=dataset_name, config_file_list=config_file_list)
                    config["data_path"] = data_path
                    config["data_path_inter"] = f"{data_path}/{subset_folder_name}/{subset_name}.inter"
                    config["fair_objective"] = fair_obj
                    config["topk"] = [10]
                    config["valid_metric"] = "NDCG@10"
                    update_sensitive(config, sensitive_feature)
                    result = run_recbole(
                        model="FOCF",
                        dataset=dataset_name,
                        config_file_list=config_file_list,
                        train_data=train_data,
                        valid_data=valid_data,
                        test_data=test_data
                    )
                    os.makedirs(os.path.dirname(result_path), exist_ok=True)
                    with open(result_path, 'wb') as handle:
                        pickle.dump(result, handle)
                    print(f"      Result saved to {result_path}.")
                    if 'best_model' in result:
                        model_path = f"models/model_{subset_name}_FOCF_{fair_obj}_{sensitive_feature}.pkl"
                        os.makedirs(os.path.dirname(model_path), exist_ok=True)
                        with open(model_path, 'wb') as model_file:
                            pickle.dump(result['best_model'], model_file)
                        print(f"      Trained model saved to {model_path}.")

    print("\nAll experiments completed.")
    print("Total Time: ", time.time() - start_time)

import argparse
import pickle
import os
import time
from recbole.config import Config
from recbole.quick_start import run_recbole
from recbole.data import data_preparation, create_dataset

"""
This driver code is fully self-contained. When executed, it:
  - Iterates over 60 dataset subsets.
  - For ml-1M, it runs experiments twice (once with "age" and once with "gender" as the sensitive feature);
    for any other dataset (e.g., BookCrossing), it uses "age" only.
  - Runs all non-FOCF models:
        FairGo_GCN, FairGo_PMF, NFCF,
        PFCN_BiasedMF, PFCN_DMF, PFCN_MLP, PFCN_PMF
  - Runs all FOCF submodels (each defined by a different fair objective: none, value, absolute, under, over, nonparity).
  - Overrides evaluation parameters: topk is set to [10] and valid_metric to "NDCG@10".
  - Overrides the sensitive attribute list (sst_attr_list) so that only one sensitive feature is used per run.
  - Saves result metrics and the trained model (if available) in the specified directories.
  - Prints checkpoints so the user can monitor progress.
  
Importantly, the dataset split is created only once per subset (and sensitive feature) and then reused for every model run, ensuring the same train/validation/test split is used throughout.
Also, if a result file already exists for a given run, the training is skipped so that future runs can avoid retraining.
All necessary folders are created automatically to prevent file saving errors.
No further manual intervention is required after launching this script.
"""

if __name__ == '__main__':
    # Define dataset subsets and folder
    subset_list = [f"sample_{i}" for i in range(1, 61)]
    total_subsets = len(subset_list)
    subset_folder_name = "URM_subsets_filtered"
    start_time = time.time()

    # Parse command-line arguments for dataset and YAML config files.
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', type=str, default='ml-1M')
    parser.add_argument('--config_files', '-c', type=str, default='test.yaml')
    args = parser.parse_args()
    config_file_list = args.config_files.strip().split(' ') if args.config_files else None

    # Set sensitive features:
    # For ml-1M, run experiments for both "age" and "gender";
    # for any other dataset (e.g., BookCrossing), use "age" only.
    if args.dataset.lower() == "ml-1m":
        sensitive_features = ["age", "gender"]
    else:
        sensitive_features = ["age"]

    # Define the list of non-FOCF models.
    models_to_run = [
        "FairGo_GCN", "FairGo_PMF", "NFCF",
        "PFCN_BiasedMF", "PFCN_DMF", "PFCN_MLP", "PFCN_PMF"
    ]
    # Define FOCF fair objectives (submodels).
    focf_fair_objectives = ["none", "value", "absolute", "under", "over", "nonparity"]

    total_sensitive = len(sensitive_features)
    overall_count = 0
    total_iterations = total_sensitive * total_subsets * (len(models_to_run) + len(focf_fair_objectives))

    print(f"Starting experiments for dataset {args.dataset}. Total iterations: {total_iterations}.")

    # Loop over each sensitive feature.
    for s_index, sensitive_feature in enumerate(sensitive_features, 1):
        print(f"\nProcessing sensitive feature [{s_index}/{total_sensitive}]: '{sensitive_feature}'.")
        # Loop over each dataset subset.
        for i, subset_name in enumerate(subset_list, 1):
            print(f"\n  Processing subset [{i}/{total_subsets}]: {subset_name}.")

            # STEP 1: Create the dataset split once per subset.
            # This ensures that the same train/validation/test split is used for all model runs on this subset.
            sample_config = Config(model="PFCN_MLP", dataset=args.dataset, config_file_list=config_file_list)
            sample_config["data_path"] = 'dataset_v2/ml-1M'
            sample_config["data_path_inter"] = f'dataset_v2/ml-1M/{subset_folder_name}/{subset_name}.inter'
            sample_config["topk"] = [10]
            sample_config["valid_metric"] = "NDCG@10"
            sample_config["sst_attr_list"] = [sensitive_feature]

            dataset = create_dataset(sample_config)
            train_data, valid_data, test_data = data_preparation(sample_config, dataset)
            print(f"    Dataset split for subset '{subset_name}' created.")

            # STEP 2: Run each non-FOCF model.
            for smodel in models_to_run:
                overall_count += 1
                percent_complete = (overall_count / total_iterations) * 100

                # Define result file path.
                result_path = f"results/new_results_ml1m_URM_filtered_gender/result_{subset_name}_{smodel}_{sensitive_feature}.txt"
                if os.path.exists(result_path):
                    print(f"    [{overall_count}/{total_iterations} | {percent_complete:.1f}%] Result for model '{smodel}' with sensitive feature '{sensitive_feature}' on subset '{subset_name}' already exists. Skipping training.")
                    continue

                print(f"    [{overall_count}/{total_iterations} | {percent_complete:.1f}%] Running model '{smodel}' with sensitive feature '{sensitive_feature}' on subset '{subset_name}'.")

                config = Config(model=smodel, dataset=args.dataset, config_file_list=config_file_list)
                config["data_path"] = 'dataset_v2/ml-1M'
                config["data_path_inter"] = f'dataset_v2/ml-1M/{subset_folder_name}/{subset_name}.inter'
                config["topk"] = [10]
                config["valid_metric"] = "NDCG@10"
                config["sst_attr_list"] = [sensitive_feature]

                result = run_recbole(
                    model=smodel,
                    dataset=args.dataset,
                    config_file_list=config_file_list,
                    train_data=train_data,
                    valid_data=valid_data,
                    test_data=test_data
                )

                # Save result metrics.
                os.makedirs(os.path.dirname(result_path), exist_ok=True)
                with open(result_path, 'wb') as handle:
                    pickle.dump(result, handle)
                print(f"      Result saved to {result_path}.")

                # Save the trained model if available.
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

                # Define result file path.
                result_path = f"results/new_results_ml1m_URM_filtered_gender/result_{subset_name}_FOCF_{fair_obj}_{sensitive_feature}.txt"
                if os.path.exists(result_path):
                    print(f"    [{overall_count}/{total_iterations} | {percent_complete:.1f}%] Result for FOCF with fair objective '{fair_obj}' and sensitive feature '{sensitive_feature}' on subset '{subset_name}' already exists. Skipping training.")
                    continue

                print(f"    [{overall_count}/{total_iterations} | {percent_complete:.1f}%] Running FOCF with fair objective '{fair_obj}' and sensitive feature '{sensitive_feature}' on subset '{subset_name}'.")

                config = Config(model="FOCF", dataset=args.dataset, config_file_list=config_file_list)
                config["data_path"] = 'dataset_v2/ml-1M'
                config["data_path_inter"] = f'dataset_v2/ml-1M/{subset_folder_name}/{subset_name}.inter'
                config["fair_objective"] = fair_obj
                config["topk"] = [10]
                config["valid_metric"] = "NDCG@10"
                config["sst_attr_list"] = [sensitive_feature]

                result = run_recbole(
                    model="FOCF",
                    dataset=args.dataset,
                    config_file_list=config_file_list,
                    train_data=train_data,
                    valid_data=valid_data,
                    test_data=test_data
                )

                # Save result metrics.
                os.makedirs(os.path.dirname(result_path), exist_ok=True)
                with open(result_path, 'wb') as handle:
                    pickle.dump(result, handle)
                print(f"      Result saved to {result_path}.")

                # Save the trained model if available.
                if 'best_model' in result:
                    model_path = f"models/model_{subset_name}_FOCF_{fair_obj}_{sensitive_feature}.pkl"
                    os.makedirs(os.path.dirname(model_path), exist_ok=True)
                    with open(model_path, 'wb') as model_file:
                        pickle.dump(result['best_model'], model_file)
                    print(f"      Trained model saved to {model_path}.")

    print("\nAll experiments completed.")
    print("Total Time: ", time.time() - start_time)

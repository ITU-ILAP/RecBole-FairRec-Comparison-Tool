import pickle
from recbole.test_results.visualization import plot_table_v2

def adjust_metrics(metrics, model_name):
    # Filter and rename metrics for FairGo model
    if 'FairGo' in model_name:
        metrics = {key.replace(" gender", "").replace("pretrain-", ""): value 
                   for key, value in metrics.items() if "pretrain" in key}
    else:
        # Standardize other models' metrics by removing "gender" from names
        metrics = {key.replace(" gender", ""): value for key, value in metrics.items()}
    return metrics


def read_txt(file_path, model_name):
    with open(file_path, 'rb') as file:
        try:
            data = pickle.load(file)
            test_result = data['test_result'].get("sm-['gender']", data['test_result'].get("none", data['test_result']))
            # Adjust metrics based on model specifics
            adjusted_result = adjust_metrics(test_result, model_name)
            return adjusted_result
        except pickle.UnpicklingError as e:
            print("Error unpickling file:", e)
            return {}


def plot_all_models(model_list, base_path_1, base_path_2):
    dicts = []
    for model_name in model_list:
        for i in range(1, 11):
            file_path_1 = f"{base_path_1}/result_subset_{i}_{model_name}.txt"
            result_dict = read_txt(file_path_1, model_name)
            result_dict.update({"Model Name": model_name, "Subset ID": i, "Sensitive Feature": "Gender", "Is Filtered": "No"})
            dicts.append(result_dict)
            print([key for key in result_dict.keys()])

            file_path_2 = f"{base_path_2}/result_subset_{i}_{model_name}.txt"
            result_dict = read_txt(file_path_2, model_name)
            result_dict.update({"Model Name": model_name, "Subset ID": i, "Sensitive Feature": "Gender", "Is Filtered": "Yes"})
            dicts.append(result_dict)
    
    print(f"Metrics plotted")
    plot_table_v2(dicts)

model_list = ["FOCF","PFCN_MLP", "PFCN_BiasedMF", "PFCN_DMF",  "PFCN_PMF", "FairGo_PMF"]
base_path_1 = "./results/results_ml1m_gender"
base_path_2 = "./results/results_ml1m_filtered_gender"

plot_all_models(model_list, base_path_1, base_path_2)
import pickle
from recbole.test_results.visualization import normalize_metrics, plot_radar_chart

def read_txt(file_path):
    # Open the file in binary mode and load it using pickle
    with open(file_path, 'rb') as file:
        try:
            data = pickle.load(file)
            test_result = data['test_result']
            # test_result['none'] -> length is 12, includes 12 different metric results.
            return next(iter(test_result.values()))
        except pickle.UnpicklingError as e:
            print("Error unpickling file:", e)
            

# It works only for one model result for now.
def plot_all_models(model_list, base_path):
    for model_name in model_list:
        file_path = f"{base_path}/{model_name}.txt"
        result_dict = read_txt(file_path)
        if result_dict is not None:
            normalized_result_dic, top_k, sensitive_attribute = normalize_metrics(result_dict)
            print(f"Metrics for {model_name}: {normalized_result_dic}")
            plot_radar_chart(model_name, normalized_result_dic)
        else:
            print(f"No data for {model_name}")

# List of models
model_list = ["PFCN_MLP", "PFCN_BiasedMF", "PFCN_DMF", "PFCN_PMF"]
base_path = "./results/comparison"

# Plot radar charts for all models
plot_all_models(model_list, base_path)
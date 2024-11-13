import pickle
from recbole.test_results.visualization import plot_table

def read_txt(file_path):
    with open(file_path, 'rb') as file:
        try:
            data = pickle.load(file)
            test_result = data['test_result']
            return next(iter(test_result.values()))
        except pickle.UnpicklingError as e:
            print("Error unpickling file:", e)

def plot_all_models(model_list, base_path):
    for model_name in model_list:
        dicts = []
        for i in range(1,6):
            file_path = f"{base_path}_{i}/{model_name}.txt"
            result_dict = read_txt(file_path)
            dicts.append(result_dict)
        
        print(f"Metrics for {model_name}")
        plot_table(dicts, model_name)

# Only works for PFCN models for now
model_list = ["PFCN_MLP", "PFCN_BiasedMF", "PFCN_DMF",  "PFCN_PMF"]
base_path = "./results/comparison"

# Plot radar charts for all models
plot_all_models(model_list, base_path)
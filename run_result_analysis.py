import pickle
from recbole.test_results.visualization import plot_table

def read_txt(file_path):
    # Open the file in binary mode and load it using pickle
    with open(file_path, 'rb') as file:
        try:
            data = pickle.load(file)
            print(data)
            if "sm-['gender']" in data['test_result']:
                test_result = data['test_result']["sm-['gender']"]
            elif "none" in data['test_result']:
                test_result = data['test_result']["none"]
            else:
                 test_result = data['test_result']
            #test_result = data['best_valid_result']
            return test_result
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

model_list = ["FOCF","PFCN_MLP", "PFCN_BiasedMF", "PFCN_DMF",  "PFCN_PMF", "FairGo_PMF"]
base_path = "./results/comparison"

plot_all_models(model_list, base_path)
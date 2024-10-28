import pickle
from recbole.test_results.visualization import normalize_metrics, plot_radar_chart

def read_txt(file_path):
    # Open the file in binary mode and load it using pickle
    with open(file_path, 'rb') as file:
        try:
            data = pickle.load(file)
            test_result = data['test_result']
            # test_result['none'] -> length is 12, includes 12 different metric results.
            return test_result['none']
        except pickle.UnpicklingError as e:
            print("Error unpickling file:", e)

# It works only for one model result for now.
file_path = "./results/comparison/PFCN_PMF.txt"
result_dict = read_txt(file_path)
normalized_result_dic, top_k, sensitive_attribute = normalize_metrics(result_dict)
print(normalized_result_dic)
plot_radar_chart('PFCN_PMF', normalized_result_dic)




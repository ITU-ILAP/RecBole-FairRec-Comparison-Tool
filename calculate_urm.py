import pandas as pd

def calculate_max_min_avg(csv_file, output_file):
    stats = []

    for file in csv_file:
        data = pd.read_csv(file)

        user_col = "Number of Users"
        item_col = "Number of Items"
        rating_col = "Number of Ratings"
        
        sensitive_0 = data.columns[-3]
        sensitive_1 = data.columns[-2]

        user_max = data[user_col].max()
        user_min = data[user_col].min()
        user_avg = data[user_col].mean()

        item_max = data[item_col].max()
        item_min = data[item_col].min()
        item_avg = data[item_col].mean()

        rating_max = data[rating_col].max()
        rating_min = data[rating_col].min()
        rating_avg = data[rating_col].mean()

        sensitive_0_max = data[sensitive_0].max()
        sensitive_0_min = data[sensitive_0].min()
        sensitive_0_avg = data[sensitive_0].mean()

        sensitive_1_max = data[sensitive_1].max()
        sensitive_1_min = data[sensitive_1].min()
        sensitive_1_avg = data[sensitive_1].mean()

        stats.append({"User Max": user_max, 
                      "User Min": user_min, 
                      "User Avg": round(user_avg,3), 
                      "Item Max": item_max, 
                      "Item Min": item_min, 
                      "Item Avg": round(item_avg,3), 
                      "Rating Max": rating_max, 
                      "Rating Min": rating_min, 
                      "Rating Avg": round(rating_avg,3), 
                      "Sensitive 0 Max": sensitive_0_max, 
                      "Sensitive 0 Min": sensitive_0_min, 
                      "Sensitive 0 Avg": round(sensitive_0_avg,3), 
                      "Sensitive 1 Max": sensitive_1_max, 
                      "Sensitive 1 Min": sensitive_1_min, 
                      "Sensitive 1 Avg": round(sensitive_1_avg,3)})
        
    stats_df = pd.DataFrame(stats)
    stats_df.to_csv(output_file, index=False)

csv_file = ["stats/bx_urm_age.csv"]
output_file = "new_stats/bx_age_max_min_avg.csv"

calculate_max_min_avg(csv_file, output_file)
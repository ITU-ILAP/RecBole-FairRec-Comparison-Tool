import os
import pandas as pd

def calculate_statistics(path, dataset_name, user_file,sensitive_col,output):
    subsets = ["inter_subsets", "inter_subsets_filtered"]

    user_df = pd.read_csv(user_file, sep="\t")

    user_df = user_df[["user_id:token", sensitive_col]]


    stats = []

    for subset in subsets:
        subset_path = os.path.join(path, subset)
        filtered = "Yes" if "filtered" in subset else "No"


        for file_name in os.listdir(subset_path):
            if file_name.endswith(".inter"):
                file_path = os.path.join(subset_path, file_name)
                subset_name = file_name.split(".")[0]

                df = pd.read_csv(file_path, sep="\t")

                user_col = "user_id:token"
                item_col = "item_id:token"
                

                V = df[user_col].nunique()
                I = df[item_col].nunique()
                R = df.shape[0]

                space_size = V * I
                shape = round((V / I), 3)
                density = round((R / space_size),3)
                R_per_user = round((R / V),3)
                R_per_item = round((R / I),3)

                new_df = pd.merge(df, user_df, on="user_id:token")
                new_df = new_df.drop_duplicates(subset=["user_id:token", sensitive_col]).reset_index(drop=True)

                s1 = new_df[new_df[sensitive_col] == 0].value_counts().sum() / new_df[sensitive_col].shape[0] * 100
                s2 = new_df[new_df[sensitive_col] == 1].value_counts().sum() / new_df[sensitive_col].shape[0] * 100

                s1 = s1.round(2)
                s2 = s2.round(2)

                difference = round(abs(s1 - s2), 2)


                stats.append({"Dataset Name" : dataset_name, "Subset Name" : subset_name, "Is Filtered" : filtered, "Space Size" : space_size, "Shape" : shape, "Density" : density, "Rating per User" : R_per_user, "Rating per Item" : R_per_item, "Gender == 0 Percentage" : f"{s1}%", "Gender == 1 Percentage" :f"{s2}%", "Difference between Gender's Percentage" : f"{difference}%"})

                stats_df = pd.DataFrame(stats)
                stats_df = stats_df.sort_values(by=["Is Filtered", "Subset Name"], key=lambda col: col if col.name != "Subset Name" else col.str.extract(r'(\d+)$').iloc[:, 0].astype(int),ascending=[True, True])
                stats_df.to_csv(output, index=False)
                print(stats_df)

base_path = "dataset_v2/ml-1M"
dataset_name = "ml-1M"
user_file = "dataset_v2/ml-1M/ml-1M.user"
sensitive_col = "gender:float"
<<<<<<< Updated upstream
output_file = "stats.csv"
=======
output_file = "stats/stats.csv"
>>>>>>> Stashed changes
calculate_statistics(base_path, dataset_name, user_file, sensitive_col, output_file)
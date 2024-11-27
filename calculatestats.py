import os
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis

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

                user_ratings = df.groupby(user_col).size().values
                item_ratings = df.groupby(item_col).size().values

                user_ratings_sorted = np.sort(user_ratings)
                n_users = len(user_ratings_sorted)
                total_user_ratings = np.sum(user_ratings_sorted)

                Gini_u = 1 - 2 * np.sum(
                    [(n_users + 1 - (i + 1)) / (n_users + 1) * user_ratings_sorted[i] / total_user_ratings for i in range(n_users)]
                )

                item_ratings_sorted = np.sort(item_ratings)
                n_items = len(item_ratings_sorted)
                total_item_ratings = np.sum(item_ratings_sorted)

                Gini_i = 1 - 2 * np.sum(
                    [(n_items + 1 - (i + 1)) / (n_items + 1) * item_ratings_sorted[i] / total_item_ratings for i in range(n_items)]
                )
 

                Gini_i = round(Gini_i, 3)
                Gini_u = round(Gini_u, 3)

                item_popularity = df.groupby(item_col)[user_col].nunique() / df[user_col].nunique()
                user_profiles = df.groupby(user_col)[item_col].apply(list)
                user_popularity_profiles = user_profiles.apply(lambda items: np.mean(item_popularity.loc[items]))

                average_popularity = user_popularity_profiles.mean()
                std_popularity = user_popularity_profiles.std()
                skew_popularity = skew(user_popularity_profiles)
                kurtosis_popularity = kurtosis(user_popularity_profiles)

                average_popularity = round(average_popularity, 3)
                std_popularity = round(std_popularity, 3)
                skew_popularity = round(skew_popularity, 3)
                kurtosis_popularity = round(kurtosis_popularity, 3)


                item_counts = df.groupby(item_col).size()
                total_ratings = df.shape[0]
                
                threshold = 0.8 * total_ratings
                short_head_items = item_counts.cumsum() <= threshold
                long_tail_items = ~short_head_items
                
                long_tail_set = item_counts[long_tail_items].index
                
                long_tail_per_user = df[df[item_col].isin(long_tail_set)].groupby(user_col).size() / df.groupby(user_col).size()
                
                avg_long_tail = round(long_tail_per_user.mean(),3)
                std_long_tail = round(long_tail_per_user.std(),3)
                skew_long_tail = round(long_tail_per_user.skew(),3)
                kurtosis_long_tail = round(long_tail_per_user.kurtosis(),3)

                ratings = df["rating:float"]

                mean_rating = ratings.mean()
                std_rating = ratings.std()
                skew_rating = skew(ratings)
                kurtosis_rating = kurtosis(ratings)

                mean_rating = round(mean_rating, 3)
                std_rating = round(std_rating, 3)
                skew_rating = round(skew_rating, 3)
                kurtosis_rating = round(kurtosis_rating, 3)

                stats.append({"Dataset Name" : dataset_name, 
                              "Subset Name" : subset_name, 
                              "Is Filtered" : filtered, 
                              "Number of Users": V, 
                              "Number of Items": I, 
                              "Number of Ratings" : R, 
                              "Space Size" : space_size, 
                              "Shape" : shape, 
                              "Density" : density, 
                              "Rating per User" : R_per_user, 
                              "Rating per Item" : R_per_item,
                              "Gini Item" : Gini_i, 
                              "Gini User" : Gini_u, 
                              "Average Popularity" : average_popularity,
                              "Standart Deviation of Popularity Bias" : std_popularity,
                              "Skewness of Popularity Bias" : skew_popularity,
                              "Kurtosis of Popularity Bias" : kurtosis_popularity,
                              "Average Long Tail Items" : avg_long_tail,
                              "Standart Deviation of Long Tail Items" : std_long_tail,
                              "Skewness of Long Tail Items" : skew_long_tail,
                              "Kurtosis of Long Tail Items" : kurtosis_long_tail,
                              "Mean Rating" : mean_rating,
                              "Standart Deviation of Rating" : std_rating,
                              "Skewness of Rating" : skew_rating,
                              "Kurtosis of Rating" : kurtosis_rating,  
                              "Gender == 0 Percentage" : f"{s1}%", 
                              "Gender == 1 Percentage" :f"{s2}%", 
                              "Difference between Gender's Percentage" : f"{difference}%"})

                stats_df = pd.DataFrame(stats)
                stats_df = stats_df.sort_values(by=["Is Filtered", "Subset Name"], key=lambda col: col if col.name != "Subset Name" else col.str.extract(r'(\d+)$').iloc[:, 0].astype(int),ascending=[True, True])
                stats_df.to_csv(output, index=False)
                print(stats_df)

base_path = "dataset_v2/ml-1M"
dataset_name = "ml-1M"
user_file = "dataset_v2/ml-1M/ml-1M.user"
sensitive_col = "gender:float"
output_file = "stats/stats.csv"
calculate_statistics(base_path, dataset_name, user_file, sensitive_col, output_file)
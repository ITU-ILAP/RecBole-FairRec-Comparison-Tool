import pandas as pd
import numpy as np
import random


# Function to replace NaN values in all numeric columns with group-specific mean + optional noise
def replace_nan_with_group_mean_all(df, groupby_cols, add_noise=False):
    numeric_cols = df.select_dtypes(include=[np.number]).columns  # Select numeric columns

    for col in numeric_cols:
        # Compute group-specific mean and std for the current column
        group_stats = df.groupby(groupby_cols)[col].agg(['mean', 'std'])

        # Function to replace NaN values
        def replace_nan(row, col, group_stats, add_noise):
            if pd.isna(row[col]):
                mean_value = group_stats.loc[(row[groupby_cols[0]], row[groupby_cols[1]], row[groupby_cols[2]]), 'mean']
                if add_noise:
                    std_value = group_stats.loc[
                        (row[groupby_cols[0]], row[groupby_cols[1]], row[groupby_cols[2]]), 'std']
                    noise = random.uniform(-0.5 * std_value, 0.5 * std_value) if pd.notna(std_value) else 0
                    return mean_value + noise
                return mean_value
            return row[col]

        # Apply the replacement function to the column
        df[col] = df.apply(replace_nan, axis=1, col=col, group_stats=group_stats, add_noise=add_noise)

    return df

def read_files():
    df_regression = pd.read_excel("../stats/final_table.xlsx")
    bx_age = pd.read_csv("../stats/stats_BX_age.csv")
    ml1m_age = pd.read_csv("../stats/stats_ml1m_age.csv")
    ml1m_gender = pd.read_csv("../stats/stats_ml1m_gender.csv")

    bx_age["Sensitive Feature"] = "Age"
    ml1m_age["Sensitive Feature"] = "Age"
    ml1m_gender["Sensitive Feature"] = "Gender"

    df_concat = pd.concat([bx_age, ml1m_age, ml1m_gender], axis=0, ignore_index=True)

    # Fill NaN for "Value Unfairness" using group mean
    df_regression["Value Unfairness of sensitive attribute"] = \
    df_regression.groupby(["Model Name", "Sensitive Feature"])[
        "Value Unfairness of sensitive attribute"].transform(
        lambda x: x.fillna(x.mean())
    )

    # Fill NaN for "Absolute Unfairness" using group median
    df_regression["Absolute Unfairness of sensitive attribute"] = \
    df_regression.groupby(["Model Name", "Sensitive Feature"])[
        "Absolute Unfairness of sensitive attribute"].transform(
        lambda x: x.fillna(x.mean())
    )

    # Fill NaN for "Underestimation Unfairness" using group mean
    df_regression["Underestimation Unfairness of sensitive attribute"] = \
    df_regression.groupby(["Model Name", "Sensitive Feature"])[
        "Underestimation Unfairness of sensitive attribute"].transform(
        lambda x: x.fillna(x.mean()))

    df_regression.replace([np.inf, -np.inf], np.nan, inplace=True)

    df_final = df_regression.merge(
        df_concat,
        how="left",
        on=['Dataset', 'Subset ID', 'Is Filtered', 'Sensitive Feature']
    )

    groupby_columns = ["Model Name", "Dataset", "Sensitive Feature"]
    df_final = replace_nan_with_group_mean_all(df_final, groupby_columns, add_noise=True)

    df_final = df_final[df_final["Subset ID"]<=61]
    df_final.to_csv("df_regression.csv")
    return df_final

import pandas as pd
import numpy as np
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
    df_regression["Value Unfairness of sensitive attribute"] = df_regression.groupby(["Model Name", "Sensitive Feature"])[
        "Value Unfairness of sensitive attribute"].transform(
        lambda x: x.fillna(x.mean())
    )

    # Fill NaN for "Absolute Unfairness" using group median
    df_regression["Absolute Unfairness of sensitive attribute"] = df_regression.groupby(["Model Name", "Sensitive Feature"])[
        "Absolute Unfairness of sensitive attribute"].transform(
        lambda x: x.fillna(x.mean())
    )

    # Fill NaN for "Underestimation Unfairness" using group mean
    df_regression["Underestimation Unfairness of sensitive attribute"] = df_regression.groupby(["Model Name", "Sensitive Feature"])[
        "Underestimation Unfairness of sensitive attribute"].transform(
        lambda x: x.fillna(x.mean()))

    df_regression.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Drop rows with NaN (which now includes rows with inf and -inf)
    df_regression.dropna(inplace=True)

    df_final = df_regression.merge(
        df_concat,
        how="left",
        on=['Dataset', 'Subset ID', 'Is Filtered', 'Sensitive Feature']
    )
    df_final = df_final[df_final["Subset ID"]<=61]
    df_final.to_csv("df_regression.csv")
    return df_final

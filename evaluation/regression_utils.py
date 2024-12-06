import pandas as pd

def read_files():
    df_dc = pd.read_csv('../stats/stats_URM.csv')
    df_dc["Sensitive Feature"] = "Gender"
    final_table = pd.read_excel('../stats/final_table.xlsx')

    # Fill NaN for "Value Unfairness" using group mean
    final_table["Value Unfairness of sensitive attribute"] = final_table.groupby(["Model Name", "Sensitive Feature"])[
        "Value Unfairness of sensitive attribute"].transform(
        lambda x: x.fillna(x.mean())
    )

    # Fill NaN for "Absolute Unfairness" using group median
    final_table["Absolute Unfairness of sensitive attribute"] = final_table.groupby(["Model Name", "Sensitive Feature"])[
        "Absolute Unfairness of sensitive attribute"].transform(
        lambda x: x.fillna(x.mean())
    )

    # Fill NaN for "Underestimation Unfairness" using group mean
    final_table["Underestimation Unfairness of sensitive attribute"] = final_table.groupby(["Model Name", "Sensitive Feature"])[
        "Underestimation Unfairness of sensitive attribute"].transform(
        lambda x: x.fillna(x.mean()))

    columns = ["Dataset","Subset ID","Is Filtered", "Sensitive Feature"]
    df_result = final_table.merge(df_dc, how='outer', left_on=columns, right_on=columns)
    df_result.to_csv("df_regression.csv")

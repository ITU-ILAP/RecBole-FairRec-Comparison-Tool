import pandas as pd

def read_files():
    df_dc = pd.read_csv('../stats/stats.csv')
    df_dc["Sensitive Feature"] = "Gender"
    final_table = pd.read_excel('../stats/final_table.xlsx')

    columns = ["Dataset","Subset ID","Is Filtered", "Sensitive Feature"]
    df_result = final_table.merge(df_dc, how='outer', left_on=columns, right_on=columns)
    return df_result
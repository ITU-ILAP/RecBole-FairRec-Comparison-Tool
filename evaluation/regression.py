from evaluation import regression_utils
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
import statsmodels.api as sm
import category_encoders as ce


def run_regression_for_fairness_measures(data, model_based ,path):

    fairness_measures_selected = [
        'Value Unfairness of sensitive attribute',
        'Overestimation Unfairness of sensitive attribute',
        'Differential Fairness of sensitive attribute',
        'Generalized Cross Entropy',
        'KS Statistic of sensitive attribute'
    ]
    dropped_accuracy_metrics = ["recall@5", "ndcg@5", "mrr@5"]

    for measure in fairness_measures_selected:
        print(f"\n--- Running Regression for {measure} ---")

        data_temp = data[data["Is Filtered"] == "Yes"]

        if model_based:
            data_temp = data_temp.drop(
                columns=dropped_accuracy_metrics + fairness_measures_selected +
                        ["Is Filtered", 'Sensitive Attribute == 0 Percentage', 'Sensitive Attribute == 1 Percentage',
                         'Subset ID', 'Model Name'])
            categorical_features = ["Sensitive Feature", "Dataset"]
        else:
            data_temp = data_temp.drop(
                columns=dropped_accuracy_metrics + fairness_measures_selected +
                        ["Is Filtered", 'Sensitive Attribute == 0 Percentage', 'Sensitive Attribute == 1 Percentage',
                         'Subset ID'])
            categorical_features = ["Model Name", "Sensitive Feature", "Dataset"]

        numeric_features = [col for col in data_temp.columns if
                            col not in categorical_features + ["Is Filtered"]]

        # One-hot encode all categorical variables
        encoder = OneHotEncoder(sparse=False)
        X_encoded = encoder.fit_transform(data_temp[categorical_features])
        X_encoded_df = pd.DataFrame(X_encoded, columns=encoder.get_feature_names_out(categorical_features))

        #encoder = ce.BinaryEncoder(cols=categorical_features)
        #X_encoded_df = encoder.fit_transform(data_temp[categorical_features])

        # Combine encoded features with numeric features
        X_numeric = data_temp[numeric_features].astype(float)
        X_preprocessed = pd.concat([X_numeric.reset_index(drop=True), X_encoded_df.reset_index(drop=True)], axis=1)

        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X_preprocessed)

        # Add a constant for the OLS regression
        X_with_constant = sm.add_constant(X_scaled)
        features_with_const = ["const"] + list(X_preprocessed.columns)

        y = data[measure]

        ols_model = sm.OLS(y, X_with_constant).fit()

        coefficients = ols_model.params.values
        p_values = ols_model.pvalues.values

        # Adjust lengths to match
        min_length = min(len(features_with_const), len(coefficients), len(p_values))
        features_with_const = features_with_const[:min_length]
        coefficients = coefficients[:min_length]
        p_values = p_values[:min_length]

        # Add significance stars
        significance = []
        for p in p_values:
            if p <= 0.001:
                significance.append("***")
            elif p <= 0.01:
                significance.append("**")
            elif p <= 0.05:
                significance.append("*")
            else:
                significance.append("")

        # Create a DataFrame with the results
        importance_df = pd.DataFrame({
            "Feature": features_with_const,
            "Coefficient": coefficients,
            "P-Value": p_values,
            "Significance": significance
        }).sort_values(by="Coefficient", ascending=False)

        # Save the results to an Excel file for each measure
        output_file = f"./{path}/OLS_Regression_Feature_Analysis_{measure.replace(' ', '_')}.xlsx"
        importance_df.to_excel(output_file, index=False)

        # Print R² scores and summary
        ols_summary = ols_model.summary()
        print("R² Score:", ols_model.rsquared)
        print("Adjusted R² Score:", ols_model.rsquared_adj)
        print(ols_summary)

def concat_regression_results(model_based ,model_name):
    # List of uploaded files and their corresponding target measures
    files = {
        "Value Unfairness": "OLS_Regression_Feature_Analysis_Value_Unfairness_of_sensitive_attribute.xlsx",
        "Overrestimation Unfairness": "OLS_Regression_Feature_Analysis_Overestimation_Unfairness_of_sensitive_attribute.xlsx",
        "Differential Fairness": "OLS_Regression_Feature_Analysis_Differential_Fairness_of_sensitive_attribute.xlsx",
        "Generalized Cross Entropy": "OLS_Regression_Feature_Analysis_Generalized_Cross_Entropy.xlsx",
        'KS Statistic of sensitive attribute': "OLS_Regression_Feature_Analysis_KS_Statistic_of_sensitive_attribute.xlsx",
    }
    # Initialize an empty list to store dataframes
    dfs = []

    # Read each file and add a "Target Measure" column
    for target, file in files.items():
        df = pd.read_excel("./fairness_measure_based/"+file)
        df["Target Measure"] = target
        dfs.append(df)

    # Concatenate all dataframes
    concatenated_df = pd.concat(dfs, ignore_index=True)

    if model_based == True:
        # Save the concatenated dataframe to a single Excel file
        output_file = f"OLS_Regression_Feature_Analysis_fairness_measure_based_{model_name}.xlsx"
    else:
        # Save the concatenated dataframe to a single Excel file
        output_file = "OLS_Regression_Feature_Analysis_fairness_measure_based.xlsx"


    concatenated_df.to_excel(output_file, index=False)

def run_regression_for_accuracy_measures(data, model_based):

    accuracy_metrics = [
        'ndcg@5',
        'recall@5',
        'mrr@5',
        'hit@5',
    ]
    for accuracy_measure in accuracy_metrics:
        data_temp = data.drop(
            columns=accuracy_metrics)
        data_temp = data_temp[data_temp["Is Filtered"] == "Yes"]
        if model_based==True:
            data_temp = data_temp.drop(
                columns=["Is Filtered", 'Sensitive Attribute == 0 Percentage', 'Sensitive Attribute == 1 Percentage',
                         'Subset ID', "Model Name"])
            categorical_features = [ "Sensitive Feature", "Dataset"]
        else:
            data_temp = data_temp.drop(
                columns=["Is Filtered", 'Sensitive Attribute == 0 Percentage', 'Sensitive Attribute == 1 Percentage', 'Subset ID'])
            categorical_features = ["Model Name", "Sensitive Feature", "Dataset"]
        numeric_features = [col for col in data_temp.columns if
                            col not in categorical_features + ["Is Filtered"]]


        # One-hot encode all categorical variables
        encoder = OneHotEncoder(sparse=False)
        X_encoded = encoder.fit_transform(data_temp[categorical_features])
        X_encoded_df = pd.DataFrame(X_encoded, columns=encoder.get_feature_names_out(categorical_features))

        # Combine encoded features with numeric features
        X_numeric = data_temp[numeric_features].astype(float)
        X_preprocessed = pd.concat([X_numeric.reset_index(drop=True), X_encoded_df.reset_index(drop=True)], axis=1)

        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X_preprocessed)

        # Add a constant for the OLS regression
        X_with_constant = sm.add_constant(X_scaled)
        features_with_const = ["const"] + list(X_preprocessed.columns)

        y = data[accuracy_measure]

        ols_model = sm.OLS(y, X_with_constant).fit()

        coefficients = ols_model.params.values
        p_values = ols_model.pvalues.values

        # Adjust lengths to match
        min_length = min(len(features_with_const), len(coefficients), len(p_values))
        features_with_const = features_with_const[:min_length]
        coefficients = coefficients[:min_length]
        p_values = p_values[:min_length]

        # Add significance stars
        significance = []
        for p in p_values:
            if p <= 0.001:
                significance.append("***")
            elif p <= 0.01:
                significance.append("**")
            elif p <= 0.05:
                significance.append("*")
            else:
                significance.append("")

        # Create a DataFrame with the results
        importance_df = pd.DataFrame({
            "Feature": features_with_const,
            "Coefficient": coefficients,
            "P-Value": p_values,
            "Significance": significance
        }).sort_values(by="Coefficient", ascending=False)
        output_file = f"./accuracy_metric_based/OLS_Regression_Feature_Analysis_{accuracy_measure.replace(' ', '_')}.xlsx"
        importance_df.to_excel(output_file, index=False)

        # Print R² scores and summary
        ols_summary = ols_model.summary()
        print("R² Score:", ols_model.rsquared)
        print("Adjusted R² Score:", ols_model.rsquared_adj)
        print(ols_summary)

def concat_accuracy_based_regression_results(model_based ,model_name):
    # List of uploaded files and their corresponding target measures
    files = {
        #"ndcg@5": "OLS_Regression_Feature_Analysis_ndcg@5.xlsx",
        #"recall@5": "OLS_Regression_Feature_Analysis_recall@5.xlsx",
        #"mrr@5": "OLS_Regression_Feature_Analysis_mrr@5.xlsx",
        "hit@5": "OLS_Regression_Feature_Analysis_hit@5.xlsx",
        }

    # Initialize an empty list to store dataframes
    dfs = []

    # Read each file and add a "Target Measure" column
    for target, file in files.items():
        df = pd.read_excel("./accuracy_metric_based/"+file)
        df["Target Measure"] = target
        dfs.append(df)

    # Concatenate all dataframes
    concatenated_df = pd.concat(dfs, ignore_index=True)

    if model_based == True:
        # Save the concatenated dataframe to a single Excel file
        output_file = f"OLS_Regression_Feature_Analysis_accuracy_metric_based_{model_name}.xlsx"
    else:
        # Save the concatenated dataframe to a single Excel file
        output_file = f"OLS_Regression_Feature_Analysis_accuracy_metric_based.xlsx"

    concatenated_df.to_excel(output_file, index=False)

data = pd.read_csv("df_regression.csv", index_col=0)

dropped_fairness_measures = [
    'Absolute Unfairness of sensitive attribute',
    'Underestimation Unfairness of sensitive attribute',
    'NonParity Unfairness of sensitive attribute',
    'Absolute Difference',
    'giniindex@5', "popularitypercentage@5"
    ]
dropped_dc_08 = ['Space Size', 'Average Popularity', 'Standart Deviation of Popularity Bias', 'Kurtosis of Popularity Bias', 'Average Long Tail Items', 'Standart Deviation of Long Tail Items', 'Kurtosis of Long Tail Items', 'Skewness of Rating', 'Kurtosis of Rating']
dropped_dc_07 = ['Number of Ratings', 'Space Size', 'Rating per User', 'Rating per Item', 'Gini Item', 'Gini User', 'Average Popularity', 'Standart Deviation of Popularity Bias', 'Skewness of Popularity Bias', 'Kurtosis of Popularity Bias', 'Average Long Tail Items', 'Standart Deviation of Long Tail Items', 'Skewness of Long Tail Items', 'Kurtosis of Long Tail Items', 'Mean Rating', 'Standart Deviation of Rating', 'Skewness of Rating', 'Kurtosis of Rating']

# drop unnecessary fairness metrics + accuracy metrics + data characteristics
data = data.drop(columns=dropped_fairness_measures + dropped_dc_08)
model_list = ["NFCF", "FOCF", "PFCN_MLP"]
model_based = False


if model_based == True:
    for i in model_list:
        data = data[data["Model Name"]==model_list[0]]
        run_regression_for_fairness_measures(data,  model_based,"fairness_measure_based")
        concat_regression_results(model_based, i)
        print("RQ1 DONE")

        # Research Question 2
        run_regression_for_accuracy_measures(data, model_based)
        concat_accuracy_based_regression_results(model_based, i)
        print("RQ2 DONE")
# Research Question 1
#data = data[data["Dataset"]=="ml1m"]
else:

    run_regression_for_fairness_measures(data, model_based, "fairness_measure_based")
    concat_regression_results(model_based, "")
    print("RQ1 DONE")


    # Research Question 2
    run_regression_for_accuracy_measures(data, model_based)
    concat_accuracy_based_regression_results(model_based, "")
    print("RQ2 DONE")

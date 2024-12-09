from evaluation import regression_utils
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
import statsmodels.api as sm

def run_regression_for_accuracy_measures(data, accuracy_measure_list):
    for accuracy_measure in accuracy_measure_list:
        data_temp = data.drop(
            columns=accuracy_measure_list)
        data_temp = data_temp[data_temp["Is Filtered"] == "Yes"]
        data_temp = data_temp.drop(
            columns=["Is Filtered", 'Sensitive Attribute == 0 Percentage', 'Sensitive Attribute == 1 Percentage', 'Subset ID'])
        categorical_features = ["Model Name", "Sensitive Feature", "Dataset"]
        numeric_features = [col for col in data_temp.columns if
                            col not in categorical_features + fairness_measures + ["Is Filtered"]]

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

def run_regression_for_fairness_measures(data, fairness_measures, path):
    # Run regression for each fairness measure
    for measure in fairness_measures:
        data_temp = data.drop(
            columns=fairness_measures)
        print(f"\n--- Running Regression for {measure} ---")
        data_temp = data_temp[data_temp["Is Filtered"] == "Yes"]
        data_temp = data_temp.drop(
            columns=["Is Filtered", 'Sensitive Attribute == 0 Percentage', 'Sensitive Attribute == 1 Percentage', 'Subset ID'])
        categorical_features = ["Model Name", "Sensitive Feature", "Dataset"]
        numeric_features = [col for col in data_temp.columns if
                            col not in categorical_features + fairness_measures + ["Is Filtered"]]

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

def concat_regression_results():
    # List of uploaded files and their corresponding target measures
    files = {
        "Absolute Difference": "OLS_Regression_Feature_Analysis_Absolute_Difference.xlsx",
        "Absolute Unfairness": "OLS_Regression_Feature_Analysis_Absolute_Unfairness_of_sensitive_attribute.xlsx",
        "Generalized Cross Entropy": "OLS_Regression_Feature_Analysis_Generalized_Cross_Entropy.xlsx",
        "KS Statistic": "OLS_Regression_Feature_Analysis_KS_Statistic_of_sensitive_attribute.xlsx",
        "NonParity Unfairness": "OLS_Regression_Feature_Analysis_NonParity_Unfairness_of_sensitive_attribute.xlsx",
        "Overestimation Unfairness": "OLS_Regression_Feature_Analysis_Overestimation_Unfairness_of_sensitive_attribute.xlsx",
        "Underestimation Unfairness": "OLS_Regression_Feature_Analysis_Underestimation_Unfairness_of_sensitive_attribute.xlsx",
        "Value Unfairness": "OLS_Regression_Feature_Analysis_Value_Unfairness_of_sensitive_attribute.xlsx",
        'Differential Fairness of sensitive attribute': "OLS_Regression_Feature_Analysis_Differential_Fairness_of_sensitive_attribute.xlsx"
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

    # Save the concatenated dataframe to a single Excel file
    output_file = "OLS_Regression_Feature_Analysis_fairness_measure_based.xlsx"
    concatenated_df.to_excel(output_file, index=False)

def concat_accuracy_based_regression_results():
    # List of uploaded files and their corresponding target measures
    files = {
        "ndcg@5": "OLS_Regression_Feature_Analysis_ndcg@5.xlsx",
        "recall@5": "OLS_Regression_Feature_Analysis_recall@5.xlsx",
        "mrr@5": "OLS_Regression_Feature_Analysis_mrr@5.xlsx",
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

    # Save the concatenated dataframe to a single Excel file
    output_file = "OLS_Regression_Feature_Analysis_accuracy_metric_based.xlsx"
    concatenated_df.to_excel(output_file, index=False)

data = regression_utils.read_files()

fairness_measures = [
    'Value Unfairness of sensitive attribute',
    'Absolute Unfairness of sensitive attribute',
    'Underestimation Unfairness of sensitive attribute',
    'Overestimation Unfairness of sensitive attribute',
    'NonParity Unfairness of sensitive attribute',
    'Absolute Difference',
    'KS Statistic of sensitive attribute',
    'Generalized Cross Entropy',
    'Differential Fairness of sensitive attribute'
]

# Research Question 1
run_regression_for_fairness_measures(data, fairness_measures, "fairness_measure_based")
concat_regression_results()
print("RQ1 DONE")

accuracy_metrics = [
    'ndcg@5',
    'recall@5',
    'mrr@5',
    'hit@5',
]
# Research Question 2
run_regression_for_accuracy_measures(data, accuracy_metrics)
concat_accuracy_based_regression_results()
print("RQ2 DONE")

"""
# Research Question 3
model_names = ["NFCF", "FOCF", "PFCN_MLP", "FairGo_PMF"]
for model in model_names:
    data_model = data[data["Model Name"]==model]
    run_regression_for_fairness_measures(data, fairness_measures, "model_based")
print("RQ3 DONE")"""

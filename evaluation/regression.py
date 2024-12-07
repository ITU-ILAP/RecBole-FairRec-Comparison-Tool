from evaluation import regression_utils
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import statsmodels.api as sm

def group_based_regression(data):

    data = data[data["Is Filtered"]=="Yes"]

    model_results = {}
    fairness_measures = [
        "Value Unfairness of sensitive attribute",
        "Absolute Unfairness of sensitive attribute",
        "Underestimation Unfairness of sensitive attribute",
        "Overestimation Unfairness of sensitive attribute",
        "NonParity Unfairness of sensitive attribute",
        "KS Statistic of sensitive attribute",
    ]
    for model_name, group in data.groupby("Model Name"):
        X_group = group[fairness_measures]
        y_group = group["hit@5"]

        X_group_with_constant = sm.add_constant(X_group)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_group_with_constant)
        fairness_model = sm.OLS(y_group, X_scaled).fit()

        # Save the results
        model_results[model_name] = {
            #"summary": fairness_model.summary(),
            "coefficients": fairness_model.params,
            "p_values": fairness_model.pvalues
        }

    for i in model_results:
        print(i)
        print(model_results[i])

def run_regression(data):

    data = data[data["Is Filtered"]=="Yes"]
    X = data.drop(columns=["ndcg@5","recall@5","mrr@5", "hit@5","Is Filtered"])
    y = data["hit@5"]

    categorical_features = ["Model Name", "Sensitive Feature", "Dataset"]

    # One-hot encode all categorical variables
    encoder = OneHotEncoder(sparse=False, drop='first')
    X_encoded = encoder.fit_transform(X[categorical_features])
    X_encoded_df = pd.DataFrame(X_encoded, columns=encoder.get_feature_names_out(categorical_features))

    # Combine encoded features with the rest of X
    X_numeric = X.drop(columns=categorical_features).astype(float)
    X_preprocessed = pd.concat([X_numeric.reset_index(drop=True), X_encoded_df.reset_index(drop=True)], axis=1)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_preprocessed)

    # Add a constant for the OLS regression
    X_with_constant = sm.add_constant(X_scaled)

    ols_model = sm.OLS(y, X_with_constant).fit()

    coefficients = ols_model.params.values
    p_values = ols_model.pvalues.values

    features_with_const = ["const"] + list(X_preprocessed.columns)
    min_length = min(len(features_with_const), len(coefficients), len(p_values))
    features_with_const = features_with_const[:min_length]
    coefficients = coefficients[:min_length]
    p_values = p_values[:min_length]


    importance_df = pd.DataFrame({
        "Feature": features_with_const,
        "Coefficient": coefficients,
        "P-Value": p_values
    }).sort_values(by="Coefficient", ascending=False)
    output_file = "OLS_Regression_Feature_Analysis.xlsx"
    importance_df.to_excel(output_file, index=False)


    ols_summary = ols_model.summary()
    print("R² Score:", ols_model.rsquared)
    print("Adjusted R² Score:", ols_model.rsquared_adj)
    print(ols_summary)

data = regression_utils.read_files()
run_regression(data)
group_based_regression(data)

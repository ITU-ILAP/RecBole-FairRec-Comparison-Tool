<<<<<<< Updated upstream
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from evaluation import regression_utils

def run_regression():
    # This dataset is the dataset for the regression model
    data = regression_utils.read_files()
    #df_result.to_csv("test.csv")

    grouped = data.copy()


=======
from evaluation import regression_utils
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import statsmodels.api as sm

def group_based_regression():


    data = regression_utils.read_files()
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

def run_regression():
    data = regression_utils.read_files()
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

    """
>>>>>>> Stashed changes
    results = []

    for model_name in grouped['Model Name'].unique():
        for dataset_name in grouped['Dataset'].unique():
            subset = grouped[(grouped['Model Name'] == model_name) & (grouped['Dataset'] == dataset_name)]

            # Explanatory variables
            X = subset[['Differential Fairness of sensitive attribute',
              'Value Unfairness of sensitive attribute', 'Absolute Unfairness of sensitive attribute',
              'Underestimation Unfairness of sensitive attribute', 'Overestimation Unfairness of sensitive attribute',
              'NonParity Unfairness of sensitive attribute', 'KS Statistic of sensitive attribute',
              'Generalized Cross Entropy', 'Number of Users', 'Number of Items', 'Density',
              'Gender == 0 Percentage', 'Difference between Gender\'s Percentage']]

            # Dependent variable
            y = subset['hit@5']

            # Standardize variables
<<<<<<< Updated upstream
            scaler = StandardScaler()
=======
            scaler = MinMaxScaler()
>>>>>>> Stashed changes
            X_scaled = scaler.fit_transform(X)
            X_scaled = sm.add_constant(X_scaled)

            # Fit regression
            model = sm.OLS(y, X_scaled).fit()

            print(f"Regression results for {model_name} on {dataset_name}")
            print(model.summary())

            adjusted_r2 = model.rsquared_adj
            coefficients = model.params
            p_values = model.pvalues

            formatted_coefficients = []
            for i, coeff in enumerate(coefficients):
                # Add significance stars based on p-value
                if p_values[i] < 0.001:
                    significance = "***"
                elif p_values[i] < 0.01:
                    significance = "**"
                elif p_values[i] < 0.05:
                    significance = "*"
                else:
                    significance = ""
                formatted_coefficients.append(f"{coeff:.4f}{significance}")

            # Append results for the current model and dataset
            results.append({
                'Model Name': model_name,
                'Dataset': dataset_name,
                'Adjusted R²': f"{adjusted_r2:.3f}",
                'Coefficients': formatted_coefficients,
                'P-values': p_values.tolist()
            })

    formatted_results = pd.DataFrame(results)
    print(formatted_results)
<<<<<<< Updated upstream

run_regression()
=======
    """

run_regression()
group_based_regression()
>>>>>>> Stashed changes

import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from evaluation import regression_utils

def run_regression():
    # This dataset is the dataset for the regression model
    data = regression_utils.read_files()
    #df_result.to_csv("test.csv")

    grouped = data.copy()


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
            scaler = StandardScaler()
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

run_regression()
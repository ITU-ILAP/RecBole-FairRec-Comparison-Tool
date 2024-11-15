import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.plotting import table

def normalize_metrics(metrics_dict):
    """
    Normalizes and organizes metrics from the provided dictionary by extracting base metric names
    and identifying any associated sensitive attributes or top-k values.

    Args:
        metrics_dict (dict): A dictionary where keys are metric names (which may include specific attributes
                             or top-k values) and values are the corresponding metric values.

    Returns:
        tuple: A tuple containing:
            - dict: A dictionary where keys are the base metric names and values are the metric values.
            - int: The top-k value if present, otherwise None.
            - str: The sensitive attribute if present, otherwise None.
    """
    normalized_metrics = {}
    sensitive_attribute = None
    top_k = None

    for key, value in metrics_dict.items():
        # Check if the metric relates to a sensitive attribute
        if 'sensitive attribute' in key:
            parts = key.split(' of sensitive attribute ')
            metric_name = parts[0].strip()
            sensitive_attribute = parts[1].strip()  # Store the sensitive attribute
            normalized_metrics[metric_name] = value
        elif '@' in key:
            # Extract the base metric name and the top-k value (e.g., 'ndcg' and '5' from 'ndcg@5')
            parts = key.split('@')
            metric_name = parts[0].strip()
            top_k = int(parts[1].strip())  # Store the top-k value as an integer
            normalized_metrics[metric_name] = value
        else:
            # If no specific attribute or top-k value, store the metric name as is
            normalized_metrics[key] = value

    return normalized_metrics, top_k, sensitive_attribute

def normalize_values(metrics):
    min_value = min(metrics.values())
    max_value = max(metrics.values())
    return {k: (v - min_value) / (max_value - min_value) for k, v in metrics.items()}

def plot_radar_chart(model_name, metrics):
    # Normalize the metric values
    normalized_metrics = normalize_values(metrics)

    # Set up the radar chart
    categories = list(normalized_metrics.keys())
    values = list(normalized_metrics.values())

    # Complete the loop for the radar chart
    values += values[:1]  # Add the first value to the end to close the loop

    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]  # Add the first angle to the end to close the loop

    # Plotting
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.fill(angles, values, color='skyblue', alpha=0.4)
    ax.plot(angles, values, color='black', linewidth=1.5)
    ax.set_yticklabels([])

    # Set the labels for each category
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)

    # Title and display
    plt.title(f'Radar Chart for {model_name}', size=15, color='black', y=1.1)
    plt.show()

def plot_table(data_dicts, model_name="Model Results"):
    df = pd.DataFrame()
    
    # Process each dictionary and add it to the DataFrame
    for i, data_dict in enumerate(data_dicts, 1):
        # Keep the rounded values as floats for accurate average calculation
        formatted_dict = {key: round(value, 3) if isinstance(value, float) else value
                          for key, value in data_dict.items()}
        temp_df = pd.DataFrame(list(formatted_dict.items()), columns=[f'Metrics', f'Calculation {i}'])
        if i == 1:
            df = temp_df
        else:
            df = pd.concat([df, temp_df.drop('Metrics', axis=1)], axis=1)

    # Calculate the average across calculations and round it
    calculation_columns = [col for col in df.columns if 'Calculation' in col]
    df['Average'] = df[calculation_columns].mean(axis=1).round(3)

    # Convert all numeric values to formatted strings
    for col in calculation_columns + ['Average']:
        df[col] = df[col].apply(lambda x: f"{x:.3f}")

    # Set up the figure and axis for the plot
    fig, ax = plt.subplots(figsize=(24, 8))
    ax.axis('tight')
    ax.axis('off')

    # Adjust column widths to accommodate the average column
    col_widths = [0.3 if 'Metrics' in col else 0.1 for col in df.columns[:-1]] + [0.15]

    # Create the table with specified column widths
    the_table = table(ax, df, loc='center', cellLoc='center', colWidths=col_widths)
    the_table.scale(1.2, 1.2)

    # Wrap text in the metric column
    for key, cell in the_table.get_celld().items():
        if key[1] == 0:
            cell.set_text_props(wrap=True)

    # Add a title
    plt.title(model_name, fontsize=16, weight='bold')

    # Display the plot
    plt.show()

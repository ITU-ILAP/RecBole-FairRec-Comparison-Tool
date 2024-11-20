import matplotlib.pyplot as plt
import numpy as np

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

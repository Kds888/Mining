import pandas as pd
import numpy as np

# Load the input CSV file
def load_data(file_path):
    data = pd.read_csv(file_path)
    # Drop empty columns if present
    data = data.dropna(axis=1, how='all')
    return data

# Calculate the orientation factor based on planned vs actual azimuth
# Penalizes deviations from the planned azimuth
def orientation_factor(actual_azimuth, planned_azimuth):
    deviation = np.abs(actual_azimuth - planned_azimuth)
    # Assuming effectiveness decreases linearly with deviation up to 90 degrees
    factor = np.maximum(0, 1 - deviation / 90)
    return factor

# Calculate the interaction factor based on blasting pattern
# Adjusts for interaction effects between holes in the same pattern
def interaction_factor(pattern_name):
    # Example: Assigning weights to patterns (can be customized further)
    pattern_weights = {
        'C1_328_109': 1.0,  # Default pattern
        # Add other patterns with specific weights if applicable
    }
    return pattern_weights.get(pattern_name, 1.0)  # Default weight is 1.0

# Calculate the effectiveness of each blast hole
def calculate_effectiveness(data):
    # Compute the orientation factor using actual and planned azimuth
    data['Orientation_Factor'] = orientation_factor(data['Drillhole.Azimuth'], 0)  # Assuming 0 as planned azimuth

    # Compute the interaction factor based on the blasting pattern
    data['Interaction_Factor'] = data['Pattern.Name'].apply(interaction_factor)

    # Calculate the final effectiveness
    data['Effectiveness'] = (
        data['Drillhole.Length'] *
        np.cos(np.radians(data['Drillhole.Dip'])) *
        data['Orientation_Factor'] *
        data['Interaction_Factor']
    )
    return data

# Save the results to a new CSV file
def save_results(data, output_path):
    data.to_csv(output_path, index=False)

# Main function to process the data
def main():
    input_file = "D:\\mining\\s1\\C1_328_109\\C1_328_109.csv"
    output_file = "D:\\mining\\s1\\C1_328_109\\blast_effectiveness_results.csv"

    # Load data
    data = load_data(input_file)

    # Calculate blast effectiveness
    data_with_effectiveness = calculate_effectiveness(data)

    # Save results
    save_results(data_with_effectiveness, output_file)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()

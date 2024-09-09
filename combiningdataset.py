import os
import pandas as pd

def combine_csv_files(input_folder, output_file):
    # Check if the input folder exists
    if not os.path.exists(input_folder):
        print(f"Error: Input folder '{input_folder}' not found.")
        return

    # Get a list of all CSV files in the input folder
    csv_files = [file for file in os.listdir(input_folder) if file.endswith('.csv')]

    # Check if there are any CSV files
    if not csv_files:
        print("Error: No CSV files found in the input folder.")
        return

    # Initialize an empty DataFrame to store combined data
    combined_data = pd.DataFrame()

    # Read each CSV file and append its data to the combined DataFrame
    for csv_file in csv_files:
        file_path = os.path.join(input_folder, csv_file)
        data = pd.read_csv(file_path)

        # Check if the data has the same structure as the existing combined data
        if combined_data.empty or data.columns.equals(combined_data.columns):
            # Check for duplicate rows before appending
            new_rows = data[~data.duplicated()]
            combined_data = combined_data.append(new_rows, ignore_index=True)
        else:
            print(f"Skipping {csv_file} as it has a different structure.")

    # Write the combined data to a new CSV file
    combined_data.to_csv(output_file, index=False)
    print(f"Combined data saved to {output_file}")

# Example usage:
input_folder = 'D:\OpenCV Project\heart\combine csv'  # Change this to the folder containing your CSV files
output_file = 'combined_data.csv'  # Change this to the desired output file name

combine_csv_files(input_folder, output_file)

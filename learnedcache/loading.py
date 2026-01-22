from pathlib import Path
import pandas as pd
import csv
import glob


def parse_log_to_csv(input_filepath: Path, output_filepath: Path):
    """
    Parses a log file line-by-line and writes directly to a CSV 
    """
    with open(input_filepath, 'r') as infile:
        first_line = infile.readline()
        if not first_line:
            print("The file is empty.")
            return

        def parse_line(line):
            return dict(item.split('=') for item in line.strip().split() if '=' in item)

        initial_data = parse_line(first_line)
        fieldnames = list(initial_data.keys())

        with open(output_filepath, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            writer.writerow(initial_data)

            for line in infile:
                if line.strip():
                    row_data = parse_line(line)
                    writer.writerow(row_data)

def transform_logs_to_csvs(input_pattern: str):
    """
    Parses multiple log files matching the input pattern and writes each to a corresponding CSV file.
    """
    filepaths = glob.glob(input_pattern)
    for filepath in filepaths:
        parse_log_to_csv(filepath, Path(filepath).with_suffix('.csv'))

def read_csvs_to_dataframe(file_pattern: str) -> pd.DataFrame:
    """
    Reads multiple CSV files and concatenates them into a single dataframe.
    """
    filepaths = glob.glob(file_pattern)
    dataframes = [pd.read_csv(filepath) for filepath in filepaths]
    combined_df = pd.concat(dataframes, ignore_index=True)
    
    return combined_df
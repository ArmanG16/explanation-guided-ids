import os
import glob
import pandas as pd
from src.utils.Print_Helper import MyPrint

def CSV_to_DF(input_path, max_rows=None):
    csv_files = glob.glob(os.path.join(input_path, "*.csv"))
    MyPrint("CSV_Files_To_DataFrame", f"Number of CSV files found: {len(csv_files)}")

    df_list = []
    total_rows = 0

    for file in csv_files:
        temp_df = pd.read_csv(file)
        
        if max_rows is not None:
            remaining_rows = max_rows - total_rows
            if remaining_rows <= 0:
                break
            if len(temp_df) > remaining_rows:
                temp_df = temp_df.head(remaining_rows)
        
        df_list.append(temp_df)
        total_rows += len(temp_df)

    if total_rows == 0:
        MyPrint("CSV_Files_To_DataFrame", f"Error, no rows found in input path: {input_path}", error=True, line_num=24)
        return

    df = pd.concat(df_list, ignore_index=True)
    MyPrint("CSV_Files_To_DataFrame", f"Total rows loaded: {len(df)}")
    return df

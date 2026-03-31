import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.pyIDS_Functions.Preprocessing_Func import preprocess_data

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
INPUT_DIR = os.path.join(BASE_DIR, "data/unprocessed/UNR-IDD")
OUTPUT_DIR = os.path.join(BASE_DIR, "data/processed/UNR-IDD_preprocessed.csv")
OUTPUT_META = os.path.join(BASE_DIR, "data", "processed", "unridd_preprocess_metadata.json")

def Preprocess_UNR_IDD():   
    preprocess_data(INPUT_DIR, OUTPUT_DIR, "Label", safe_name="benign", malicious_name="malicious", safe_values=["Normal"], metadata_output_path=OUTPUT_META)

if __name__ == "__main__":
    Preprocess_UNR_IDD()

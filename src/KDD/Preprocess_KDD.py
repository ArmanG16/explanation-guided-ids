import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.pyIDS_Functions.Preprocessing_Func import preprocess_data

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
INPUT_DIR = os.path.join(BASE_DIR, "data/unprocessed/NSL-KDD")
OUTPUT_DIR = os.path.join(BASE_DIR, "data/processed/KDD_preprocessed.csv")

def Preprocess_UNR_IDD():   
    preprocess_data(INPUT_DIR, OUTPUT_DIR, "class")

if __name__ == "__main__":
    Preprocess_UNR_IDD()
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.utils.Preprocessing_Func import preprocess_data

def Preprocess_Beth():
    dataset_files = [
        "labelled_2021may-ip-10-100-1-4.csv",
        "labelled_2021may-ip-10-100-1-4-dns.csv",
        "labelled_2021may-ip-10-100-1-26.csv",
        "labelled_2021may-ip-10-100-1-26-dns.csv",
        "labelled_2021may-ip-10-100-1-95.csv",
        "labelled_2021may-ip-10-100-1-95-dns.csv",
        "labelled_2021may-ip-10-100-1-105.csv",
        "labelled_2021may-ip-10-100-1-105-dns.csv",
        "labelled_2021may-ip-10-100-1-186.csv",
        "labelled_2021may-ip-10-100-1-186-dns.csv",
        "labelled_2021may-ubuntu.csv",
        "labelled_2021may-ubuntu-dns.csv",
        "labelled_training_data.csv",
        "labelled_testing_data.csv",
        "labelled_validation_data.csv"
    ]

    input_dir = "data/unprocessed/bethdataset"
    output_dir = "data/processed/bethdataset"
    for file in dataset_files:
        in_path = os.path.join(input_dir, file)
        out_path = os.path.join(output_dir, "processed_" + file)
        preprocess_data(in_path, out_path)

Preprocess_Beth()
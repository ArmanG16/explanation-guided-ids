import os
import sys

# Add project root to sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from src.pyIDS_Functions.Preprocessing_Func import preprocess_data
from src.utils.Print_Helper import MyPrint


def Preprocess_USNW():
    """
    Preprocess USNW-NB15 dataset and generate:
      - processed CSV (for PyIDS training)
      - preprocessing metadata JSON (for XAI rule translation)
    """

    # ---- Paths ----
    INPUT_DIR = os.path.join(BASE_DIR, "data", "unprocessed", "USNW-NB15_clean")
    OUTPUT_CSV = os.path.join(BASE_DIR, "data", "processed", "usnw_preprocessed.csv")
    OUTPUT_META = os.path.join(BASE_DIR, "data", "processed", "usnw_preprocess_metadata.json")

    # ---- IMPORTANT: Change this if your label column name differs ----
    CLASS_COLUMN = "label"   # For USNW-NB15, it's "label"

    MyPrint("Preprocess_USNW.py", f"Input directory: {INPUT_DIR}")
    MyPrint("Preprocess_USNW.py", f"Output CSV: {OUTPUT_CSV}")
    MyPrint("Preprocess_USNW.py", f"Output metadata: {OUTPUT_META}")

    preprocess_data(
        input_path=INPUT_DIR,
        output_path=OUTPUT_CSV,
        class_column=CLASS_COLUMN,
        columns=None,
        variance_threshold=0.01,
        metadata_output_path=OUTPUT_META,
    )

    MyPrint("Preprocess_USNW.py", "USNW-NB15 preprocessing complete.")


if __name__ == "__main__":
    Preprocess_USNW()
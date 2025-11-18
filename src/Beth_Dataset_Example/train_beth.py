import os
import sys

# Add the repo root (/home/nsewell/explanation-guided-ids) to sys.path
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(THIS_DIR, "../.."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.Beth_Dataset_Example.Train_On_Beth_Dataset import Beth_Train

if __name__ == "__main__":
    print("Starting IDS training on 1,000,000 rows...")
    Beth_Train(250000)
    print("Training complete.")

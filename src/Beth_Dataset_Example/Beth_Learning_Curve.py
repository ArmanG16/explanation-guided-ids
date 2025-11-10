import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.utils.Learning_Curve_Func import make_learning_curve

make_learning_curve("data/processed/bethdataset/processed_labelled_2021may-ip-10-100-1-4-dns.csv")
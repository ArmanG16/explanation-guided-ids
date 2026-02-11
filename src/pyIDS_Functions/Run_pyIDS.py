import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

sys.path.insert(0, os.path.join(BASE_DIR, "pyIDS"))

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.pyIDS_Functions.Mining_Cars_Func import Mine_Cars
from src.pyIDS_Functions.Training_Func import Train
from src.pyIDS_Functions.Optimizing_Lambdas import Optimize_Lambdas
from src.utils.Print_Helper import MyPrint

def Run_pyIDS(algorithm, train_df, cars_path, output_path, val_df = None, lambdas_path = None ):
    MyPrint("Run_pyIDS", "Beginning to Train")

    cars = Mine_Cars(100, train_df, cars_path)

    lambda_array = [1, 1, 1, 1, 1, 1, 1]
    if (val_df is not None and lambdas_path is not None):
        lambda_array = Optimize_Lambdas(
            algorithm=algorithm,
            cars=cars,
            df=val_df,
            output_path=lambdas_path,
            precision=50,
            iterations=3
        )

    Train(algorithm, lambda_array, cars, train_df, output_path)

    MyPrint("Run_pyIDS", "Training complete!")
import sys
import os
import glob
import pandas as pd
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))

sys.path.insert(0, os.path.join(BASE_DIR, "pyIDS"))

data_dir = os.path.join(BASE_DIR, "pyIDS/data/")
cars_dir = os.path.join(BASE_DIR, "data/cars/titanic_cars.csv")
output_path = os.path.join(BASE_DIR, "data/rules/titanic_rules.csv")
lambdas_path = os.path.join(BASE_DIR, "data/lambdas/titanic_best_lambdas.csv")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from src.pyIDS_Functions.Mining_Cars_Func import Mine_Cars
from src.pyIDS_Functions.Training_Func import Train
from src.pyIDS_Functions.Optimizing_Lambdas import Optimize_Lambdas
from src.utils.Print_Helper import MyPrint

def Titanic_Lambdas_Train(max_rows, val_fraction = 0.2, random_state=42):
    MyPrint("Train_Titanic_With_Lambdas", "Beginning to Train Titanic")

    titanic_files = glob.glob(os.path.join(data_dir, "titanic*.csv"))
    titanic_files = [f for f in titanic_files if os.path.basename(f) != "titanic.csv"]

    full_df = pd.concat((pd.read_csv(f) for f in titanic_files), ignore_index=True)
    full_df = full_df.head(max_rows)
    full_df["surv"] = full_df["surv"].astype(str)

    train_df, val_df = train_test_split(
        full_df,
        test_size=val_fraction,
        stratify=full_df["surv"],
        random_state=random_state
    )

    cars = Mine_Cars(100, train_df, cars_dir)

    lambda_array = Optimize_Lambdas(
        algorithm="SLS",
        cars=cars,
        df=val_df,
        output_path=lambdas_path,
        individual_precision=50,
        individiual_iterations=3,
        precision=50,
        iterations=1,
        grid_step=200,
        search_type="coordinate"
    )

    Train("SLS", lambda_array, cars, train_df, output_path)

    MyPrint("Train_Titanic_With_Lambdas", "Training complete!")

if __name__ == "__main__":
    Titanic_Lambdas_Train(10000)
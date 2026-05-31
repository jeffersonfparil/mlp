import argparse
from pathlib import Path
import re
from sklearn.preprocessing import OneHotEncoder
import lightgbm as lgb
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.datasets import make_regression
from sklearn.metrics import root_mean_squared_error, r2_score
import numpy as np
import pandas as pd
import shap

parser = argparse.ArgumentParser(description="LightGBM for trial analysis and genomic prediction")
parser.add_argument("analysis_type", help="The type of analysis to perform (trials or gp)")
parser.add_argument("input_file", type=Path, help="Path to the input TSV file")
parser.add_argument("output_dir", type=Path, help="Directory to save the output")
parser.add_argument("randomisation_input_file", type=Path, default=Path("."), help="Path to the randomisation input TSV file (for gp analysis)")
parser.add_argument("n_replicates", type=int, default=3, help="Number of replicates for randomisation (for gp analysis)")
parser.add_argument("n_folds", type=int, default=10, help="Number of folds for cross-validation (for gp analysis)")

args = parser.parse_args()

# class Args:
#     def __init__(self, analysis_type, input_file, output_dir, randomisation_input_file, n_replicates, n_folds):
#         self.analysis_type = analysis_type
#         self.input_file = input_file
#         self.output_dir = output_dir
#         self.randomisation_input_file = randomisation_input_file
#         self.n_replicates = n_replicates
#         self.n_folds = n_folds

print(f"Analysis Type: {args.analysis_type}")
print(f"Input File: {args.input_file}")
print(f"Output Directory: {args.output_dir}")
print(f"Randomisation Input File: {args.randomisation_input_file}")
print(f"Number of Replicates: {args.n_replicates}")
print(f"Number of Folds: {args.n_folds}")

def get_params(args):
    if (args.analysis_type != "trials") and (args.analysis_type != "gp"):
        raise ValueError("Invalid analysis type. Must be 'trials' or 'gp'.")
    if not args.input_file.exists():
        raise FileNotFoundError(f"Input file {args.input_file} does not exist.")
    if not args.output_dir.exists():
        raise FileNotFoundError(f"Output directory {args.output_dir} does not exist.")
    if args.analysis_type == "trials":
        return {
            "analysis_type": args.analysis_type,
            "input_file": args.input_file,
            "output_dir": args.output_dir,
            "objective": 'regression',
            "n_estimators": 10,
            "learning_rate": 0.1,
            "max_depth": 5,
            "random_state": 42,
        }
    else:
        if not args.randomisation_input_file.exists():
            raise FileNotFoundError(f"Randomisation input file {args.randomisation_input_file} does not exist.")
        if (args.analysis_type == "gp") and (args.n_replicates <= 0):
            raise ValueError("Number of replicates must be a positive integer.")
        if (args.analysis_type == "gp") and (args.n_folds <= 1):
            raise ValueError("Number of folds must be greater than 1.")
        return {
            "analysis_type": args.analysis_type,
            "input_file": args.input_file,
            "output_dir": args.output_dir,
            "randomisation_input_file": args.randomisation_input_file,
            "n_replicates": args.n_replicates,
            "n_folds": args.n_folds,
            "objective": 'regression',
            # GP-Specific Hyperparameter Grid
            "n_estimators": [1_000, 5_000],          # High estimators, rely on early stopping
            "learning_rate": [0.01, 0.05],           # Lower is generally better for GP
            "num_leaves": [15, 31, 63],              # Constrain leaf growth
            "max_depth": [3, 5, 7],                  # Keep trees relatively shallow
            "colsample_bytree": [0.1, 0.3, 0.5],     # Force exploration of minor SNPs
            "subsample": [0.7, 0.85, 1.0],           # Row subsampling
            "reg_alpha": [0.0, 0.1, 1.0],            # L1 Regularization (Sparsity)
            "reg_lambda": [0.0, 1.0, 5.0],           # L2 Regularization (Shrinkage)
            "min_child_samples": [20, 30],           # Prevent fitting to tiny groups of samples
            "random_state": 42,
            "early_stopping_rounds": 50,             # Increased slightly due to lower learning rate
            "within_fold_cv_frac": 10,
        }

def extract_X_y(args):
    params = get_params(args) 
    df = pd.read_csv(params["input_file"], sep="\t")
    X_tmp = df.drop(df.columns[0], axis=1)
    at_least_one_str = False
    for col in X_tmp.columns:
        if X_tmp[col].dtype == "str":
            at_least_one_str = True
            break
    
    if not at_least_one_str:
        X = X_tmp
    else:
        X = None
        encoder = OneHotEncoder(sparse_output=False)
        for col in X_tmp.columns:
            if X_tmp[col].dtype == "str":
                X_1_hot = pd.DataFrame(encoder.fit_transform(X_tmp[col].values.reshape(-1, 1)))
                X_1_hot.columns = encoder.categories_[0]
            else:
                X_1_hot = X_tmp[col]
            if X is None:
                X = X_1_hot
            else:
                X = pd.concat([X, X_1_hot], axis=1)
    
    # # LightGBM prefers clean feature names (no special JSON characters)
    # X = X.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', str(x)))
    
    y = df[df.columns[0]]
    return X, y

def extract_randomisations(args):
    params = get_params(args) 
    df_randomisation = pd.read_csv(params["randomisation_input_file"], sep="\t", header=None)
    idx_training = []
    idx_validation = []
    i = 1
    for r in range(params["n_replicates"]):
        for f in range(params["n_folds"]):
            idx_training.append([int(x)-1 for x in df_randomisation.iloc[i-1,0].split(",")])
            idx_validation.append([int(x)-1 for x in df_randomisation.iloc[i,0].split(",")])
            i += 2
    return (idx_validation, idx_training)

def define_fname_output(args):
    return args.output_dir / f"output-{args.input_file.stem}-TREES.tsv"
    
def extract_entries_effects(args):
    params = get_params(args)
    X, y = extract_X_y(args)
    model = lgb.LGBMRegressor(
        objective=params["objective"],
        n_estimators=params["n_estimators"],
        learning_rate=params["learning_rate"],
        max_depth=params["max_depth"],
        random_state=params["random_state"],
        device="cuda",
        verbose=-1 # Suppress LightGBM warnings
    )
    model.fit(X, y)
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X)
    global_effects = pd.DataFrame({'ids': X.columns, 'effects': abs(shap_values.values).mean(axis=0)})
    
    fname_output = define_fname_output(args)
    global_effects.to_csv(fname_output, sep="\t", index=False)
    print(f"Global effects saved to {fname_output}")
    return None

def gp_repeated_kfold_cv(args):
    # args = Args(analysis_type="gp", input_file=Path.home() / Path("Documents/mlp/tests/tmp/gp/simulated-DATA_TYPE_BINARY-N_500-P_1000-HIDDEN_LAYERS_1.tsv"), output_dir=Path.home() / Path("Documents/mlp/tests/tmp/gp"), randomisation_input_file=Path.home() / Path("Documents/mlp/tests/tmp/gp/output-sorghum-YLD-RANDOMISATION.tsv"), n_replicates=3, n_folds=5)
    # args = Args(analysis_type="gp", input_file=Path.home() / Path("Documents/mlp/tests/tmp/gp/sorghum-YLD.tsv"), output_dir=Path.home() / Path("Documents/mlp/tests/tmp/gp"), randomisation_input_file=Path.home() / Path("Documents/mlp/tests/tmp/gp/output-sorghum-YLD-RANDOMISATION.tsv"), n_replicates=3, n_folds=5)
    params = get_params(args) 
    X, y = extract_X_y(args)
    idx_validation, idx_training = extract_randomisations(args)
    df_out = pd.DataFrame(columns=["replicate", "fold", "best_n_estimators", "best_learning_rate", "best_max_depth", "best_subsample", "rmse", "r2", "corr"])
    
    for r in range(params["n_replicates"]):
        for f in range(params["n_folds"]):
            # r = 0; f = 0;
            print(f"Replicate {r}, Fold {f}")
            idx = r*params["n_folds"] + f
            X_train = X.iloc[idx_training[idx]]
            X_test = X.iloc[idx_validation[idx]]
            y_train = y.iloc[idx_training[idx]]
            y_test = y.iloc[idx_validation[idx]]
            lgb_reg = lgb.LGBMRegressor(
                objective=params["objective"],
                random_state=params["random_state"],
                device="cuda",
                verbose=-1 # Suppress LightGBM warnings
            )
            lgb_params = {
                'n_estimators': params["n_estimators"],
                'learning_rate': params["learning_rate"],
                'num_leaves': params["num_leaves"],
                'max_depth': params["max_depth"],
                'colsample_bytree': params["colsample_bytree"],
                'subsample': params["subsample"],
                'reg_alpha': params["reg_alpha"],
                'reg_lambda': params["reg_lambda"],
                'min_child_samples': params["min_child_samples"]
            }
            # Note: Increased n_iter from 5 to 20 to better explore the larger GP space
            rs = RandomizedSearchCV(
                lgb_reg, 
                lgb_params, 
                n_iter=20, 
                cv=3, 
                scoring='neg_mean_squared_error', 
                n_jobs=-1,
                random_state=params["random_state"]
            )
            callbacks = [lgb.early_stopping(stopping_rounds=params["early_stopping_rounds"], verbose=True)]
            rs.fit(
                X_train, y_train, 
                eval_set=[(X_test, y_test)], 
                callbacks=callbacks
            )
            
            print(f"Best Params: {rs.best_params_}")
            best_model = rs.best_estimator_
            y_pred = best_model.predict(X_test)
            
            print(f"RMSE: {root_mean_squared_error(y_test, y_pred):.4f}")
            print(f"R^2: {r2_score(y_test, y_pred):.4f}")
            
            df_out = pd.concat([
                df_out, 
                pd.DataFrame({
                    "replicate": [r+1],
                    "fold": [f+1],
                    "best_n_estimators": [rs.best_params_["n_estimators"]],
                    "best_learning_rate": [rs.best_params_["learning_rate"]],
                    "best_max_depth": [rs.best_params_["max_depth"]],
                    "best_subsample": [rs.best_params_["subsample"]],
                    "rmse": [root_mean_squared_error(y_test, y_pred)],
                    "r2": [r2_score(y_test, y_pred)],
                    "corr": [np.corrcoef(y_test, y_pred)[0, 1]]
                })
            ], ignore_index=True)
    fname_output = define_fname_output(args)
    df_out.to_csv(fname_output, sep="\t", index=False)
    print(f"GP results saved to {fname_output}")
    return None

if __name__ == "__main__":
    params = get_params(args)
    if params["analysis_type"] == "trials":
        print("Extracting entries effects...")
        extract_entries_effects(args)
    else:
        print("Performing GP with repeated K-fold CV...")
        gp_repeated_kfold_cv(args)
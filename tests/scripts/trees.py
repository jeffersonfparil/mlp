import argparse
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.datasets import make_regression
from sklearn.metrics import root_mean_squared_error, r2_score
import numpy as np
import pandas as pd
import shap

parser = argparse.ArgumentParser(description="XGBoost for trial analysis and genomic prediction")
parser.add_argument("analysis_type", help="The type of analysis to perform (trials or gp or remotesensing)")
parser.add_argument("input_file", type=Path, help="Path to the input TSV file")
parser.add_argument("output_dir", type=Path, help="Directory to save the output")

parser.add_argument("--n-estimators", type=int, default=1_000, help="Number of estimators for XGBoost (for trials analysis)")
parser.add_argument("--max-depth", type=int, default=3, help="Maximum depth of the trees (for trials analysis)")
parser.add_argument("--learning-rate", type=float, default=0.1, help="Learning rate (for trials analysis)")

parser.add_argument("--seed", type=int, default=42, help="Randomisation seed")

parser.add_argument("--randomisation-input-file", type=Path, default=Path("."), help="Path to the randomisation input TSV file (for gp analysis)")
parser.add_argument("--n-replicates", type=int, default=3, help="Number of replicates for randomisation (for gp analysis)")
parser.add_argument("--n-folds", type=int, default=10, help="Number of folds for cross-validation (for gp analysis)")
parser.add_argument("--early-stopping-rounds", type=int, default=10, help="Number of early stopping rounds for hyperparameter tuning/optimisation")

def list_of_ints(arg):
    return [int(x) for x in arg.split(',')]

def list_of_floats(arg):
    return [float(x) for x in arg.split(',')]

parser.add_argument("--optim-n-estimators", type=list_of_ints, default=[1_000, 10_000], help="Number of estimators to test for hyperparameter tuning/optimisation")
parser.add_argument("--optim-max-depth", type=list_of_ints, default=[3, 5, 10], help="Maximum depths of the trees to test for hyperparameter tuning/optimisation")
parser.add_argument("--optim-learning-rate", type=list_of_floats, default=[0.01, 0.1], help="Learning rates to test for hyperparameter tuning/optimisation")
parser.add_argument("--optim-subsample", type=list_of_floats, default=[0.5, 0.75, 1.0], help="Subsampling rates to test for hyperparameter tuning/optimisation")


args = parser.parse_args()

def get_params(args):
    if (args.analysis_type != "trials") and (args.analysis_type != "gp") and (args.analysis_type != "remotesensing"):
        raise ValueError("Invalid analysis type. Must be 'trials' or 'gp' or 'remotesensing'.")
    if not args.input_file.exists():
        raise FileNotFoundError(f"Input file {args.input_file} does not exist.")
    if not args.output_dir.exists():
        raise FileNotFoundError(f"Output directory {args.output_dir} does not exist.")
    if args.analysis_type == "trials":
        return {
            "analysis_type": args.analysis_type,
            "input_file": args.input_file,
            "output_dir": args.output_dir,
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "objective": 'reg:squarederror',
            "learning_rate": args.learning_rate,
            "random_state": args.seed,
        }
    else:
        # 'gp' or 'remotesensing'
        if not args.randomisation_input_file.exists():
            raise FileNotFoundError(f"Randomisation input file {args.randomisation_input_file} does not exist.")
        if ((args.analysis_type == "gp") or (args.analysis_type == "remotesensing")) and (args.n_replicates <= 0):
            raise ValueError("Number of replicates must be a positive integer.")
        if ((args.analysis_type == "gp") or (args.analysis_type == "remotesensing")) and (args.n_folds <= 1):
            raise ValueError("Number of folds must be greater than 1.")
        return {
            "analysis_type": args.analysis_type,
            "input_file": args.input_file,
            "output_dir": args.output_dir,
            "randomisation_input_file": args.randomisation_input_file,
            "n_replicates": args.n_replicates,
            "n_folds": args.n_folds,
            "objective": 'reg:squarederror',
            "early_stopping_rounds": args.early_stopping_rounds,
            "n_estimators": args.optim_n_estimators,
            "max_depth": args.optim_max_depth,
            "learning_rate": args.optim_learning_rate,
            "subsample": args.optim_subsample,
            "random_state": args.seed,
        }

def extract_X_y(args):
    params = get_params(args) 
    df = pd.read_csv(params["input_file"], sep="\t")
    X_tmp = df.drop(df.columns[0], axis=1)
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
    y = df[df.columns[0]]
    return X, y

def extract_randomisations(args):
    params = get_params(args) 
    df_randomisation = pd.read_csv(params["randomisation_input_file"], sep="\t", header=None)
    print(df_randomisation)
    idx_training = []
    idx_validation = []
    i = 1
    for r in range(params["n_replicates"]):
        for f in range(params["n_folds"]):
            print(f"Replicate {r}, Fold {f}, i: {i}")
            idx_training.append([int(x)-1 for x in df_randomisation.iloc[i-1,0].split(",")])
            idx_validation.append([int(x)-1 for x in df_randomisation.iloc[i,0].split(",")])
            i += 2
    return (idx_validation, idx_training)

def define_fname_output(args):
    return args.output_dir / f"output-{args.input_file.stem}-TREES.tsv"
    
def extract_entries_effects(args):
    # args = Args(analysis_type="trials", input_file=Path("/home/jp3h/Documents/mlp/tests/tmp/trials/australia.soybean-yield.tsv"), output_dir=Path("/home/jp3h/Documents/mlp/tests/tmp/trials"), randomisation_input_file=Path("."), n_replicates=3, n_folds=5)
    params = get_params(args)
    X, y = extract_X_y(args)
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=params["n_estimators"],
        learning_rate=params["learning_rate"],
        max_depth=params["max_depth"],
        random_state=params["random_state"],
        device = "cuda",
        # early_stopping_rounds=params["early_stopping_rounds"],
    )
    model.fit(X, y)
    # y_pred = model.predict(X)
    # print(f"RMSE: {root_mean_squared_error(y, y_pred):.4f}")
    # print(f"R^2: {r2_score(y, y_pred):.4f}")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X)
    global_effects = pd.DataFrame({'ids': X.columns, 'effects': abs(shap_values.values).mean(axis=0)})
    # Save as TSV
    fname_output = define_fname_output(args)
    global_effects.to_csv(fname_output, sep="\t", index=False)
    print(f"Global effects saved to {fname_output}")
    return None

def gp_repeated_kfold_cv(args):
    # args = Args(analysis_type="gp", input_file=Path("/home/jp3h/Documents/mlp/tests/tmp/gp/simulated-DATA_TYPE_BINARY-N_500-P_1000-HIDDEN_LAYERS_1.tsv"), output_dir=Path("/home/jp3h/Documents/mlp/tests/tmp/gp"), randomisation_input_file=Path("/home/jp3h/Documents/mlp/tests/tmp/gp/output-simulated-DATA_TYPE_BINARY-N_500-P_1000-HIDDEN_LAYERS_1-RANDOMISATION.tsv"), n_replicates=3, n_folds=5)
    params = get_params(args) 
    X, y = extract_X_y(args)
    idx_validation, idx_training = extract_randomisations(args)
    df_out = pd.DataFrame(columns=["datasets", "reps", "folds", "nt", "nv", "models", "best_n_estimators", "best_learning_rate", "best_max_depth", "best_subsample", "rmse", "r2", "corr"])
    for r in range(params["n_replicates"]):
        for f in range(params["n_folds"]):
            # r = 0; f = 0;
            print(f"Replicate {r}, Fold {f}")
            idx = r*params["n_folds"] + f
            X_train = X.iloc[idx_training[idx]]
            X_test = X.iloc[idx_validation[idx]]
            y_train = y.iloc[idx_training[idx]]
            y_test = y.iloc[idx_validation[idx]]
            xgb_reg = xgb.XGBRegressor(
                objective="reg:squarederror",
                early_stopping_rounds=params["early_stopping_rounds"],
                random_state=params["random_state"],
                device="cuda",
            )
            xgb_params = {
                'n_estimators': params["n_estimators"],
                'learning_rate': params["learning_rate"],
                'max_depth': params["max_depth"],
                'subsample': params["subsample"]
            }
            rs = RandomizedSearchCV(xgb_reg, xgb_params, n_iter=5, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
            rs.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
            print(f"Best Params: {rs.best_params_}")
            best_model = rs.best_estimator_
            best_model.set_params(device="cuda")
            y_pred = best_model.predict(X_test)
            print(f"RMSE: {root_mean_squared_error(y_test, y_pred):.4f}")
            print(f"R^2: {r2_score(y_test, y_pred):.4f}")
            df_out = pd.concat([
                df_out, 
                pd.DataFrame({
                    "datasets": [args.input_file.stem],
                    "reps": [r+1],
                    "folds": [f+1],
                    "nt": [len(X_train)],
                    "nv": [len(X_test)],
                    "models": ["XGBoost"],
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
    print(args)
    params = get_params(args)
    print(params)
    if params["analysis_type"] == "trials":
        print("Extracting entries effects...")
        extract_entries_effects(args)
    else:
        # "gp" or "remotesensing"
        print(f"Performing {params['analysis_type']} with repeated K-fold CV...")
        gp_repeated_kfold_cv(args)
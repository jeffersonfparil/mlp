import argparse
import os
from pathlib import Path
import pandas as pd
import numpy as np
import rasterio
import xgboost as xgb
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error

parser = argparse.ArgumentParser(description="Remote-sensing modelling using TIFF images per plot, i.e. no input shapefiles")
parser.add_argument("fname_traits", help="Path to the CSV file containing trait data (e.g., constant_agronomic_traits_2021.csv)")
parser.add_argument("image_root_dir", help="Root directory where the plot image folders are stored (e.g., the folder containing date subfolders)")
parser.add_argument("date", help="Date of the flight to use for feature extraction (e.g., '06-14-2021'). You can run this script multiple times with different dates to build a multitemporal dataset.")
parser.add_argument("target", help="Target variable for modelling (e.g., 'Yield').")


def get_params(args):
    """Parse command-line arguments."""
    if not args.fname_traits.exists():
        raise FileNotFoundError(f"Input file {args.fname_traits} does not exist.")
    if not args.image_root_dir.exists():
        raise FileNotFoundError(f"Image root directory {args.image_root_dir} does not exist.")
    if not args.date:
        raise ValueError("Date argument is required.")
    if not (args.image_root_dir / args.date).exists():
        raise FileNotFoundError(f"Date folder {args.date} does not exist in the image root directory.")
    if not args.target:
        raise ValueError("Target argument is required.")
    return {
        'traits_csv': args.fname_traits,
        'image_root_dir': args.image_root_dir,
        'date': args.date,
        'target': args.target,
        'n_repeats': 5,
        'n_folds': 5,
        "objective": 'reg:squarederror',
        "n_estimators": [10, 20, 50, 100, 200],
        "learning_rate": [0.01, 0.1],
        "max_depth": [3, 5, 10],
        "subsample": [0.5, 0.75, 1.0],
        "random_state": 42,
        "early_stopping_rounds": 10,
        'random_state': 42
    }

def read_band(path):
    """Helper function to read a TIFF band and mask out zero/background values."""
    with rasterio.open(path) as src:
        array = src.read(1).astype(float)
        # Assuming 0 is background/nodata from the plot cropping process
        array[array == 0] = np.nan 
        return array

def extract_features_for_plot(plot_folder):
    # plot_folder = Path.home() / Path("Documents/mlp/tests/datasets/farag_2024/Images/Train/06-14-2021/33761")
    # Define paths (Adjust filenames if they differ in your downloaded folders)
    band_paths = {
        'red': Path(plot_folder) / 'red.tiff',
        'green': Path(plot_folder) / 'green.tiff',
        'blue': Path(plot_folder) / 'blue.tiff',
        'nir': Path(plot_folder) / 'nir.tiff',
        'rededge': Path(plot_folder) / 'red_edge.tiff'
    }
    
    # Read the masked arrays
    try:
        red = read_band(band_paths['red'])
        green = read_band(band_paths['green'])
        blue = read_band(band_paths['blue'])
        nir = read_band(band_paths['nir'])
        rededge = read_band(band_paths['rededge'])
    except FileNotFoundError as e:
        print(f"Missing band in {plot_folder}: {e}")
        return None

    # Calculate Vegetation Indices
    # Suppress warnings for expected invalid operations (like all-NaN slices)
    with np.errstate(divide='ignore', invalid='ignore'):
        ndvi = (nir - red) / (nir + red)
        gndvi = (nir - green) / (nir + green)
        ndre = (nir - rededge) / (nir + rededge)

    # Calculate statistics (ignoring NaNs)
    features = {
        'NDVI_mean': np.nanmean(ndvi),
        'NDVI_max': np.nanmax(ndvi),
        'GNDVI_mean': np.nanmean(gndvi),
        'NDRE_mean': np.nanmean(ndre),
        # You can also add raw band statistics if desired
        'NIR_mean': np.nanmean(nir)
    }
    
    return features

def extract_X_y(params):
    # params = {'traits_csv': Path.home() / Path("Documents/mlp/tests/datasets/farag_2024/constant_agronomic_traits_2021.csv"), 'image_root_dir': Path.home() / Path("Documents/mlp/tests/datasets/farag_2024"), 'date': "06-14-2021", 'target': 'Yield', 'n_repeats': 5, 'n_folds': 5, "objective": 'reg:squarederror', "n_estimators": [10, 20, 50, 100, 200], "learning_rate": [0.01, 0.1], "max_depth": [3, 5, 10], "subsample": [0.5, 0.75, 1.0], "random_state": 42, "early_stopping_rounds": 10, 'random_state': 42}
    df_traits = pd.read_csv(params['traits_csv'])
    df_traits = df_traits.dropna(subset=[params['target']]) # Drop rows with missing target
    all_extracted_data = []
    print("Extracting features from plot images...")
    for index, row in df_traits.iterrows():
        # index = 0; row = df_traits.iloc[index]
        plot_number = row['Plot_Number']
        plot_features = {'Plot_Number': plot_number}
        plot_folder = os.path.join(params['image_root_dir'], params['date'], f"{plot_number}")
        if os.path.exists(plot_folder):
            stats = extract_features_for_plot(plot_folder)
            if stats:
                # Append time-step prefix (e.g., T0_NDVI_mean)
                for key, val in stats.items():
                    plot_features[f"{key}"] = val
        all_extracted_data.append(plot_features)
    df_features = pd.DataFrame(all_extracted_data)
    # Merge spectral features with agronomic traits
    data = pd.merge(df_traits, df_features, on='Plot_Number', how='inner')
    drop_cols = ['Plot_Number', 'Rice_Cultivar', 'Experiment_Name', 'Plot_Center', 
                 'Yield', 'Emergence_Date_DOY', 'Heading_25', 'Heading_50', 'Heading_100', 'Final_Lodge',
                 'Seeding_Rate', 'Nitrogen_Rare', 'Replicate']
    X = data.drop(columns=[col for col in drop_cols if col in data.columns])
    if not data.columns.isin([params['target']]).any():
        raise ValueError(f"Target column '{params['target']}' not found in data")
    y = data[params['target']]
    return X, y

def train_model(params):
    # params = {'traits_csv': Path.home() / Path("Documents/mlp/tests/datasets/farag_2024/constant_agronomic_traits_2021.csv"), 'image_root_dir': Path.home() / Path("Documents/mlp/tests/datasets/farag_2024"), 'date': "06-14-2021", 'target': 'Yield', 'n_repeats': 5, 'n_folds': 5, "objective": 'reg:squarederror', "n_estimators": [10, 20, 50, 100, 200], "learning_rate": [0.01, 0.1], "max_depth": [3, 5, 10], "subsample": [0.5, 0.75, 1.0], "random_state": 42, "early_stopping_rounds": 10, 'random_state': 42}
    X, y = extract_X_y(params)
    df_out = pd.DataFrame(columns=["datasets", "reps", "folds", "nt", "nv", "models", "best_n_estimators", "best_learning_rate", "best_max_depth", "best_subsample", "rmse", "r2", "corr"])
    for r in range(params['n_repeats']):
        gkf = GroupKFold(n_splits=params['n_folds'])
        randomisations = list(gkf.split(X, y, groups=X.index))
        for f in range(params['n_folds']):
            # f = 0
            idx_training, idx_validation = randomisations[f]
            X_train = X.iloc[idx_training]
            X_test = X.iloc[idx_validation]
            y_train = y.iloc[idx_training]
            y_test = y.iloc[idx_validation]
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
            print(f"Correlation: {np.corrcoef(y_test, y_pred)[0, 1]:.4f}")
            df_out = pd.concat([
                df_out, 
                pd.DataFrame({
                    "datasets": "farag_2024",
                    "reps": [r+1],
                    "folds": [f+1],
                    "nt": [len(X_train)],
                    "nv": [len(X_test)],
                    "models": ["XGBoost"],
                    "best_n_estimators": [rs.best_params_['n_estimators']],
                    "best_learning_rate": [rs.best_params_['learning_rate']],
                    "best_max_depth": [rs.best_params_['max_depth']],
                    "best_subsample": [rs.best_params_['subsample']],
                    "rmse": [root_mean_squared_error(y_test, y_pred)],
                    "r2": [r2_score(y_test, y_pred)],
                    "corr": [np.corrcoef(y_test, y_pred)[0, 1]]
                })
            ], ignore_index=True)
    return df_out


# ==========================================
# EXECUTION
# ==========================================
if __name__ == "__main__":
    params = {
        'traits_csv': Path.home() / Path("Documents/mlp/tests/datasets/farag_2024/constant_agronomic_traits_2021.csv"), 
        'image_root_dir': Path.home() / Path("Documents/mlp/tests/datasets/farag_2024"), 
        'date': "06-14-2021", 
        'target': 'Yield', 
        'n_repeats': 5, 
        'n_folds': 5,
        "objective": 'reg:squarederror',
        "n_estimators": [10, 20, 50, 100, 200],
        "learning_rate": [0.01, 0.1],
        "max_depth": [3, 5, 10],
        "subsample": [0.5, 0.75, 1.0],
        "random_state": 42,
        "early_stopping_rounds": 10, 
        'random_state': 42
    }
    df = train_model(params)
    print(df)
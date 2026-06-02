import os
from pathlib import Path
import pandas as pd
import numpy as np
import rasterio
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error, r2_score

# ==========================================
# 1. IMAGE PROCESSING (Per Plot)
# ==========================================
def read_band(path):
    """Helper function to read a TIFF band and mask out zero/background values."""
    with rasterio.open(path) as src:
        array = src.read(1).astype(float)
        # Assuming 0 is background/nodata from the plot cropping process
        array[array == 0] = np.nan 
        return array

def extract_features_for_plot(plot_folder):
    """
    Reads the 5 spectral bands for a single plot, calculates indices, 
    and returns a dictionary of aggregate statistics.
    """
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

# ==========================================
# 2. DATASET BUILDING
# ==========================================
def build_dataset(traits_csv, image_root_dir, dates):
    """
    Matches the Plot_Number in the CSV to the corresponding plot image folders.
    """
    # traits_csv = Path.home() / Path("Documents/mlp/tests/datasets/farag_2024/constant_agronomic_traits_2021.csv"); image_root_dir = Path.home() / Path("Documents/mlp/tests/datasets/farag_2024"); dates = ["06-14-2021", "07-14-2021", "08-03-2021", "09-03-2021"]
    df_traits = pd.read_csv(traits_csv)
    df_traits = df_traits.dropna(subset=['Yield']) # Drop rows with missing target
    all_extracted_data = []
    print("Extracting features from plot images...")
    for index, row in df_traits.iterrows():
        # index = 0; row = df_traits.iloc[index]
        plot_number = row['Plot_Number']
        plot_features = {'Plot_Number': plot_number}
        # Loop through multitemporal dates/flights
        for t_idx, date in enumerate(dates):
            # t_idx = 0; date = dates[t_idx]
            # ASSUMPTION: Folder structure is like: image_root_dir/date/plot_33761/
            # Adjust this path logic to match exactly how the dataset unzipped on your machine
            plot_folder = os.path.join(image_root_dir, date, f"{plot_number}")
            if os.path.exists(plot_folder):
                stats = extract_features_for_plot(plot_folder)
                if stats:
                    # Append time-step prefix (e.g., T0_NDVI_mean)
                    for key, val in stats.items():
                        plot_features[f"T{t_idx}_{key}"] = val
        all_extracted_data.append(plot_features)
    df_features = pd.DataFrame(all_extracted_data)
    # Merge spectral features with agronomic traits
    final_data = pd.merge(df_traits, df_features, on='Plot_Number', how='inner')
    return final_data

# ==========================================
# 3. MODEL TRAINING & VALIDATION
# ==========================================
def train_model(data, target='Yield', group='Experiment_Name'):
    """
    Trains the LightGBM model using GroupKFold to ensure it generalizes 
    across different experiments/environments.
    """
    # Define columns to drop (metadata + other targets you aren't predicting right now)
    drop_cols = ['Plot_Number', 'Rice_Cultivar', 'Experiment_Name', 'Plot_Center', 
                 'Yield', 'Emergence_Date_DOY', 'Heading_25', 'Heading_50', 'Heading_100', 'Final_Lodge']
    
    X = data.drop(columns=[col for col in drop_cols if col in data.columns])
    y = data[target]
    groups = data[group] 
    
    gkf = GroupKFold(n_splits=3)
    oof_preds = np.zeros(len(y))
    
    print(f"\nTraining model to predict {target}...")
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model = lgb.LGBMRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            random_state=42
        )
        
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(30, verbose=False)])
        oof_preds[val_idx] = model.predict(X_val)
        
    rmse = np.sqrt(mean_squared_error(y, oof_preds))
    r2 = r2_score(y, oof_preds)
    
    print("\n--- Final Model Performance ---")
    print(f"R2 Score: {r2:.3f}")
    print(f"RMSE:     {rmse:.3f}")

# ==========================================
# EXECUTION
# ==========================================
if __name__ == "__main__":
    TRAITS_CSV = Path.home() / "Documents/mlp/tests/datasets/farag_2024/constant_agronomic_traits_2021.csv"
    
    # You will need to define where the downloaded images are stored
    IMAGE_ROOT_DIR = Path.home() / "Documents/mlp/tests/datasets/farag_2024/"
    
    # List the dates/flights you want to include (to build the multitemporal profile)
    FLIGHT_DATES = ["06-14-2021", "07-14-2021", "08-03-2021", "09-03-2021"]
    
    # 1. Build Dataset
    final_dataset = build_dataset(TRAITS_CSV, IMAGE_ROOT_DIR, FLIGHT_DATES)
    
    # 2. Train Model on Yield
    # train_model(final_dataset, target='Yield', group='Experiment_Name')
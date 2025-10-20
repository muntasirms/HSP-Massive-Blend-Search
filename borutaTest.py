import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from boruta import BorutaPy
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import csv
import os
import math
import time

# ============================================================
# Global references for worker processes
# ============================================================
x2_values = None
x2_columns = None
csv_path = None
csv_lock = None
y_index = None  # target column index for this worker

def init_globals(values, columns, shared_csv_path, shared_lock):
    """Initializer: assigns globals for each worker."""
    global x2_values, x2_columns, csv_path, csv_lock
    x2_values = values
    x2_columns = columns  # pandas.Index
    csv_path = shared_csv_path
    csv_lock = shared_lock

# ============================================================
def run_boruta_for_column(y_idx):
    """
    Run Boruta for one target column.
    Predictors are all other columns.
    """
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor
    from boruta import BorutaPy
    import csv

    x2 = pd.DataFrame(x2_values, columns=x2_columns)

    # Y column
    y_name = x2_columns[y_idx]
    y = x2.iloc[:, y_idx].values

    # X columns = all except y
    X = x2.drop(columns=y_name).values
    X_names = x2.drop(columns=y_name).columns.tolist()

    # Handle degenerate case: y constant
    if len(np.unique(y)) <= 1:
        print(f"Column {y_name} skipped (constant values).")
        return None

    # Setup Boruta
    rf = RandomForestRegressor(n_jobs=-1, max_depth=5, random_state=42)
    feat_selector = BorutaPy(rf, n_estimators='auto', verbose=0, random_state=1)

    # Fit Boruta
    feat_selector.fit(X, y)

    # Gather results
    ranks = feat_selector.ranking_
    # Each row: predictor_name, rank
    rows_to_write = list(zip(X_names, ranks))

    # Write to CSV safely
    csv_lock.acquire()
    try:
        with open(csv_path, 'a', newline='') as fh:
            writer = csv.writer(fh, delimiter=',')
            for name, rank in rows_to_write:
                writer.writerow([y_name, name, int(rank)])
    finally:
        csv_lock.release()

    print(f"Boruta iteration for target column '{y_name}' finished.")
    return None  # nothing needed returned

# ============================================================
# Main execution
# ============================================================
if __name__ == '__main__':
    start_time = time.time()

    # Load CSV
    X = pd.read_csv('C:\\Users\munta\PycharmProjects\PVDFDissolution\\finalparametersexperimental.csv', index_col=0, delimiter="|")
    X = X.drop(columns=['Name', 'CAS', 'SMILES', 'Formula'], errors='ignore')

    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')

    X_imputed = X
    for col in X.columns:
        X_imputed[col] = X[col].fillna(X[col].median())


    # ============================================================
    # Impute invalid / NaN values with column median
    # ============================================================
    X_imputed = X.fillna(0) #X.apply(lambda col: col.fillna(col.median()))

    x2_values = X_imputed.values
    x2_columns = X_imputed.columns  # pandas.Index
    print("Shape before imputation:", X.shape)
    print("Shape after imputation:", X_imputed.shape)
    print("Any NaNs remaining?", X_imputed.isna().sum().sum())
    print("Number of all-zero columns:", (X_imputed == 0).all().sum())
    # ============================================================
    # Output CSV
    # ============================================================
    output_csv = "BorutaResults_per_target.csv"
    if os.path.exists(output_csv):
        os.remove(output_csv)
    # Header: target_column, predictor_column, rank
    with open(output_csv, 'w', newline='') as fh:
        writer = csv.writer(fh, delimiter=',')
        writer.writerow(['target_column', 'predictor_column', 'rank'])

    # ============================================================
    # Multiprocessing lock
    # ============================================================
    manager = multiprocessing.Manager()
    lock = manager.Lock()

    # ============================================================
    # Parallel Boruta over each target column
    # ============================================================
    max_workers = max(1, os.cpu_count() - 2)
    with ProcessPoolExecutor(max_workers=max_workers, initializer=init_globals,
                             initargs=(x2_values, x2_columns, output_csv, lock)) as executor:
        futures = {executor.submit(run_boruta_for_column, idx): idx for idx in range(len(x2_columns))}
        for future in as_completed(futures):
            idx = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"Boruta failed for column {x2_columns[idx]}: {e}")

    print(f"All Boruta iterations finished. Total runtime: {time.time() - start_time:.2f} seconds")
    print("Incremental CSV results written to:", output_csv)

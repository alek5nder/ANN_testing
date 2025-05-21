import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from perceptron_nn import MultiLayerPerceptron, cross_validate_mlp
import traceback


def preprocess_clean_data(df: pd.DataFrame, y_col: str):
    df = df.rename(columns={"#Layovers": "Num_Layovers", "Price [PLN]": "Price"})

    if "Flight_date" in df.columns:
        df["Flight_date"] = pd.to_datetime(df["Flight_date"])
    if "Extraction_Time" in df.columns:
        df["Extraction_Time"] = pd.to_datetime(df["Extraction_Time"].astype(str).str.split(" ").apply(lambda x: x[0]), dayfirst=True)

    drop_cols = [
        "Extraction_Time", "Flight_date", "arr_city", "dep_city",
        "Departure_airport_name", "Destination_airport_name",
        "layover_airport", "ujemne", "low_cost1", "low_cost2"
    ]
    df = df.drop(columns=drop_cols, errors='ignore')

    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.dropna()

    # Poprawne rozpoznanie celu po rename
    target_actual_name = "Price" if y_col == "Price [PLN]" else y_col
    X = df.drop(columns=[target_actual_name], errors='ignore')
    y = df[target_actual_name].values.astype(np.float64)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y


# === Główna część ===
if __name__ == "__main__":
    df = pd.read_excel("loty_clean.xlsx")
    target_col = "Price [PLN]"

    X_full = df.drop(columns=[target_col])
    y = df[target_col]

    base_model_params = {
        'num_layers': [10, 5],
        'learning_rate': 0.01,
        'activation_function': 'relu'
    }

    results = []

    for col in X_full.columns:
        print(f"\n===> Usuwam kolumnę: {col}")
        X_subset = X_full.drop(columns=[col])
        temp_df = X_subset.copy()

        if target_col not in temp_df.columns:
            temp_df[target_col] = y

        try:
            X_processed, y_array = preprocess_clean_data(temp_df, y_col=target_col)
        except Exception as e:
            print(f"⚠️ Błąd przetwarzania przy kolumnie '{col}': {e}")
            traceback.print_exc()
            continue

        try:
            cv_results = cross_validate_mlp(
                X_processed, y_array,
                model_params=base_model_params,
                num_folds=5,
                num_epochs=1000,
                early_stopping=True,
                patience=25,
                verbose=False
            )

            results.append({
                'usunięta_kolumna': col,
                'val_mse_mean': cv_results['val_mse_mean'],
                'val_mae_mean': cv_results['val_mae_mean'],
                'val_mse_std': cv_results['val_mse_std'],
                'val_mae_std': cv_results['val_mae_std']
            })

        except Exception as e:
            print(f"❌ Błąd treningu przy kolumnie '{col}': {e}")
            traceback.print_exc()
            continue

    # Zapis wyników
    results_df = pd.DataFrame(results)
    if len(results_df) == 0:
        print("❌ Brak danych w wynikach. Sprawdź preprocessing i dane wejściowe.")
    else:
        results_df.to_csv("analiza_usuwania_kolumn.csv", index=False)
        print("\n✅ Zapisano wyniki do analiza_usuwania_kolumn.csv")
        print(results_df)

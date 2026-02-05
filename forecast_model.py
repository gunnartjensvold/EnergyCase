import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from weather_analysis import load_consumption_data, load_weather_data, merge_data


def split_data(df, target="total", test_size=0.2, shuffle=True) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Splitter data i trenings- og testsett."""
    features = ["temperatur", "nedbor", "vind"]
    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        shuffle=shuffle,
    )
    return pd.DataFrame(X_train), pd.DataFrame(X_test), pd.Series(y_train), pd.Series(y_test)


def train_model(X_train, y_train):
    """Tren lineær regresjonsmodell."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model, None, None


def train_svr_model(X_train, y_train):
    """Tren SVR-modell. Krever skalering av data."""
    # SVR krever skalerte features for å fungere godt
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train_scaled = scaler_X.fit_transform(X_train)
    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()

    model = SVR(kernel="rbf", C=1.0, epsilon=0.1)
    model.fit(X_train_scaled, y_train_scaled)

    return model, scaler_X, scaler_y


def predict_svr(model, X_test, scaler_X, scaler_y):
    """Prediksjon med SVR-modell."""
    X_test_scaled = scaler_X.transform(X_test)
    y_pred_scaled = model.predict(X_test_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    return y_pred


def predict(model, X_test):
    """prediksjon på test sett"""
    return model.predict(X_test)


def evaluate_model(y_true, y_pred):
    """Hent ut ulike scores som er vanlige for linear regression"""
    metrics = {
        "R2": r2_score(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAPE": np.mean(np.abs((y_true - y_pred) / y_true)) * 100,
    }
    return metrics


def get_feature_importance(model, feature_names):
    """Henter feature-vekter fra modellen."""
    importance = dict(zip(feature_names, model.coef_))
    importance["intercept"] = model.intercept_
    return importance


def print_model_summary(metrics, importance):
    """Skriver ut modellsammendrag."""
    print("MODELL: Lineær Regresjon")
    print("=" * 50)
    print("Predikerer daglig totalt strømforbruk fra værdata")
    print()

    print("EVALUERINGSMETRIKKER")
    print("-" * 50)
    print(f"R² (forklart varians):     {metrics['R2']:.3f}")
    print(f"MAE (gj.snitt avvik):      {metrics['MAE']:,.0f} kWh")
    print(f"RMSE (rot av kv.avvik):    {metrics['RMSE']:,.0f} kWh")
    print(f"MAPE (prosentavvik):       {metrics['MAPE']:.1f}%")
    print()

    print("FEATURE-VEKTER (koeffisienter)")
    print("-" * 50)
    for feature, weight in importance.items():
        if feature == "intercept":
            print(f"Konstantledd:              {weight:,.0f}")
        else:
            print(f"{feature:25} {weight:,.0f} kWh per enhet")


def print_strengths_weaknesses(metrics):
    """Skriver ut styrker og svakheter ved modellen."""
    print("\n\nSTYRKER")
    print("=" * 50)
    print("- Enkel og tolkbar modell")
    print("- Rask å trene og predikere")
    print("- Koeffisientene viser direkte effekt av hver værfaktor")
    print(f"- Forklarer {metrics['R2'] * 100:.1f}% av variasjonen med shuffle=True")

    print("\n\nSVAKHETER")
    print("=" * 50)
    print("- Antar lineær sammenheng (kan misse ikke-lineære effekter)")
    print("- Feiler på fremtidig data (negativ R² med shuffle=False)")
    print("- Takler ikke strukturelle endringer (f.eks. industrifallet i 2024)")
    print("- Ignorerer andre viktige faktorer (strømpris, økonomisk aktivitet)")

    print("\n\nKONKLUSJON")
    print("=" * 50)
    print("Værdata kan forklare noe av variasjonen i forbruk, men er ikke")
    print("tilstrekkelig for å prognosere fremtidig forbruk. Modellen feiler")
    print("når forbruksmønsteret endrer seg over tid.")

    print("\n\nFORBEDRINGSMULIGHETER")
    print("=" * 50)
    print("- Inkludere strømpris som feature")
    print("- Legge til økonomiske indikatorer")
    print("- Bruke tidsseriemodeller (ARIMA, Prophet)")
    print("- Trene separate modeller per forbrukergruppe")


def plot_predictions(y_test, y_pred):
    """Plotter faktisk vs predikert forbruk."""
    # Sorter etter dato
    sorted_idx = y_test.index.argsort()
    y_test_sorted = y_test.iloc[sorted_idx]
    y_pred_sorted = y_pred[sorted_idx]

    plt.figure(figsize=(12, 5))
    plt.plot(y_test_sorted.index, y_test_sorted.values, label="Faktisk", alpha=0.7)
    plt.plot(y_test_sorted.index, y_pred_sorted, label="Predikert", alpha=0.7)
    plt.xlabel("Dato")
    plt.ylabel("Forbruk (kWh)")
    plt.title("Tidsserie: Faktisk vs Predikert")
    plt.legend()
    plt.tight_layout()
    plt.show()


def load_data():
    """Laster og kombinerer forbruks- og værdata."""
    print("Laster data...")
    consumption = load_consumption_data()
    weather = load_weather_data()
    df = merge_data(consumption, weather)
    print(f"Datasett: {len(df)} dager")
    print(f"Kolonner: {list(df.columns)}\n")
    return df


def run_experiment(df, shuffle: bool) -> tuple[dict, dict]:
    """Kjører eksperiment med begge modeller og returnerer evalueringer."""
    X_train, X_test, y_train, y_test = split_data(df, test_size=0.2, shuffle=shuffle)

    lr_model, _, _ = train_model(X_train, y_train)
    lr_pred = predict(lr_model, X_test)
    lr_metrics = evaluate_model(y_test, lr_pred)

    svr_model, scaler_X, scaler_y = train_svr_model(X_train, y_train)
    svr_pred = predict_svr(svr_model, X_test, scaler_X, scaler_y)
    svr_metrics = evaluate_model(y_test, svr_pred)

    return lr_metrics, svr_metrics


def print_experiment(lr_metrics: dict, svr_metrics: dict, shuffle: bool):
    """Skriver ut resultater fra et eksperiment."""
    if shuffle:
        print("=" * 60)
        print("TEST 1: SHUFFLE=TRUE")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("TEST 2: SHUFFLE=FALSE (prognoseevne)")
        print("=" * 60)

    print("Lineær Regresjon:")
    print(f"  R²:   {lr_metrics['R2']:.3f}  MAPE: {lr_metrics['MAPE']:.1f}%")
    print("SVR:")
    print(f"  R²:   {svr_metrics['R2']:.3f}  MAPE: {svr_metrics['MAPE']:.1f}%")


def print_conclusion(svr_shuffle: dict, svr_noshuffle: dict):
    """Skriver ut konklusjon."""
    print("\n" + "=" * 60)
    print("KONKLUSJON")
    print("=" * 60)
    print(f"Med shuffle=True:  R² ≈ {svr_shuffle['R2']:.2f} - Vær forklarer {svr_shuffle['R2']*100:.0f}% av variasjonen")
    print(f"Med shuffle=False: R² ≈ {svr_noshuffle['R2']:.2f}")
    print()
    print("Vi ser at når modellen trenes på historisk data og prøver å predikere")
    print("framtida, så predikerer den høgare strømforbruk i forhold til det som er")
    print("reelt. Dette er nok på grunn av at vi har for få features. Det betyr at")
    print("temperatør, nedbør og vind ikkje er nok til å forklare variansen i")
    print("strømforbruket. Vi kunne vurdert å legge til Måned, Vekedag, Strømpris")
    print("osv som ein feature for å sjå om det hjalp prediksjonen.")


def plot_comparison(df, svr_shuffle: dict, svr_noshuffle: dict, save_path=None):
    """Plotter sammenligning av shuffle=True og shuffle=False."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Shuffle=True
    X_train, X_test, y_train, y_test = split_data(df, test_size=0.2, shuffle=True)
    svr_model, scaler_X, scaler_y = train_svr_model(X_train, y_train)
    svr_pred = predict_svr(svr_model, X_test, scaler_X, scaler_y)
    sorted_idx = y_test.index.argsort()
    y_test_sorted = y_test.iloc[sorted_idx]
    y_pred_sorted = svr_pred[sorted_idx]
    axes[0].plot(y_test_sorted.index, y_test_sorted.values, label="Faktisk", alpha=0.7)
    axes[0].plot(y_test_sorted.index, y_pred_sorted, label="Predikert", alpha=0.7)
    axes[0].set_xlabel("Dato")
    axes[0].set_ylabel("Forbruk (kWh)")
    axes[0].set_title(f"Shuffle=True (R² = {svr_shuffle['R2']:.2f})")
    axes[0].legend()

    # Shuffle=False
    X_train, X_test, y_train, y_test = split_data(df, test_size=0.2, shuffle=False)
    svr_model, scaler_X, scaler_y = train_svr_model(X_train, y_train)
    svr_pred = predict_svr(svr_model, X_test, scaler_X, scaler_y)
    axes[1].plot(y_test.index, y_test.values, label="Faktisk", alpha=0.7)
    axes[1].plot(y_test.index, svr_pred, label="Predikert", alpha=0.7)
    axes[1].set_xlabel("Dato")
    axes[1].set_ylabel("Forbruk (kWh)")
    axes[1].set_title(f"Shuffle=False (R² = {svr_noshuffle['R2']:.2f})")
    axes[1].legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


if __name__ == "__main__":
    df = load_data()

    # Test 1: Shuffle=True
    lr_shuffle, svr_shuffle = run_experiment(df, shuffle=True)
    print_experiment(lr_shuffle, svr_shuffle, shuffle=True)

    # Test 2: Shuffle=False
    lr_noshuffle, svr_noshuffle = run_experiment(df, shuffle=False)
    print_experiment(lr_noshuffle, svr_noshuffle, shuffle=False)

    # Konklusjon og plot
    print_conclusion(svr_shuffle, svr_noshuffle)
    plot_comparison(df, svr_shuffle, svr_noshuffle, save_path='forecast_shuffle_comparison.png')

import pandas as pd
import matplotlib.pyplot as plt
from consumption_trends import fetch_full_history


def load_consumption_data():
    """Laster forbruksdata og aggregerer til daglig nivå."""
    df = fetch_full_history()
    df_pivot = df.pivot(columns='consumptionGroup', values='quantityKwh')
    daily = df_pivot.resample('D').sum()
    daily.index = daily.index.tz_localize(None) #Ta vekk tidssone i datoene for å vere kompatibel med weather data
    return daily


def load_weather_data(filepath='vaer_data_2021_to_2026.csv'):
    """Laster og renser værdata fra CSV."""
    df = pd.read_csv(filepath, sep=';', decimal=',', encoding='latin-1')
    df.columns = ['navn', 'stasjon', 'dato', 'temperatur', 'nedbor', 'vind']
    df['dato'] = pd.to_datetime(df['dato'], dayfirst=True)

    for col in ['temperatur', 'nedbor', 'vind']:
        df[col] = df[col].astype(str).str.replace(',', '.').str.replace('"', '')
        df[col] = pd.to_numeric(df[col], errors='coerce') #Converts errors to NaN

    df.set_index('dato', inplace=True)
    return df


def merge_data(consumption, weather):
    """Kombinerer forbruks- og værdata på dato."""
    merged = consumption.join(weather[['temperatur', 'nedbor', 'vind']], how='inner')
    merged['total'] = merged[['cabin', 'household', 'primary', 'secondary', 'tertiary']].sum(axis=1)
    # Fjerner rader der værdata mangler (NaN)
    merged = merged.dropna()
    return merged


def calculate_correlation(df):
    """Beregner korrelasjon mellom værfaktorer og forbruk."""
    corr = df.corr()
    weather_corr = corr[['temperatur', 'nedbor', 'vind']].drop(['temperatur', 'nedbor', 'vind', 'total'])
    return weather_corr


def find_strongest_factors(weather_corr):
    """Finner sterkeste værfaktor for hver forbrukergruppe."""
    results = {}
    for group in weather_corr.index:
        abs_corr = weather_corr.loc[group].abs()
        strongest = abs_corr.idxmax()
        value = weather_corr.loc[group, strongest]
        results[group] = (strongest, value)
    return results


def print_analysis(weather_corr, strongest_factors):
    """Skriver ut korrelasjonsanalysen."""
    print("KORRELASJON MELLOM VÆRFAKTORER OG FORBRUK")
    print("=" * 50)
    print(weather_corr.round(3).to_string())

    print("\n\nSTERKESTE VÆRFAKTOR PER FORBRUKERGRUPPE")
    print("=" * 50)
    for group, (factor, value) in strongest_factors.items():
        print(f"{group}: {factor} (r = {value:.3f})")

    print("\n\nGJENNOMSNITTLIG KORRELASJON PER VÆRFAKTOR")
    print("=" * 50)
    avg_corr = weather_corr.abs().mean()
    print(avg_corr.round(3).to_string())
    print(f"\nKonklusjon: {avg_corr.idxmax()} har sterkest korrelasjon med strømforbruk")


def plot_weather_vs_total(df, save_path=None):
    """Plotter alle værfaktorer mot totalt forbruk."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Sammenheng mellom værfaktorer og totalt strømforbruk', fontsize=14)

    weather_factors = [('temperatur', 'Temperatur (C)'),
                       ('nedbor', 'Nedbør (mm)'),
                       ('vind', 'Vind (m/s)')]

    for i, (col, label) in enumerate(weather_factors):
        axes[i].scatter(df[col], df['total'], alpha=0.3, s=5)
        axes[i].set_xlabel(label)
        axes[i].set_ylabel('Totalt forbruk (kWh)')
        corr = df[col].corr(df['total'])
        axes[i].set_title(f"{label.split()[0]} (r={corr:.3f})")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_temperature_per_group(df, save_path=None):
    """Plotter temperatur mot forbruk for hver gruppe."""
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    fig.suptitle('Temperatur vs forbruk per gruppe', fontsize=14)
    groups = ['cabin', 'household', 'primary', 'secondary', 'tertiary']

    for i, group in enumerate(groups):
        ax = axes[i // 3, i % 3]
        ax.scatter(df['temperatur'], df[group], alpha=0.3, s=5)
        ax.set_xlabel('Temperatur (C)')
        ax.set_ylabel(f'{group} (kWh)')
        corr = df['temperatur'].corr(df[group])
        ax.set_title(f"{group} (r={corr:.3f})")

    axes[1, 2].axis('off')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


if __name__ == "__main__":
    # Last inn data
    daily_consumption = load_consumption_data()
    weather = load_weather_data()

    # Kombiner datasett
    df_merged = merge_data(daily_consumption, weather)
    print(f"Kombinert datasett: {len(df_merged)} dager\n")

    # Analyser korrelasjoner
    weather_corr = calculate_correlation(df_merged)
    strongest = find_strongest_factors(weather_corr)
    print_analysis(weather_corr, strongest)

    # Visualiser
    plot_weather_vs_total(df_merged, save_path='weather_vs_total_scatter.png')
    plot_temperature_per_group(df_merged, save_path='weather_temperature_per_group.png')

    print("Andre faktorer som kan påvirke energiforbruk i tillegg til temperatur:")
    print("Strømpris, tid på dagen, Ukedag vs Helg")
    print("Sesongvariasjon, man bruker strøm til andre ting om sommeren enn vinter")
    print("I nyere tid har også folk begynt å lage masse elbil, så antal elbiler på nettet er relevant")

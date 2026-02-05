import os
import pandas as pd
import requests
import time
from datetime import datetime
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt


def generate_monthly_ranges(start_year=2021, end_year=2026):
    """Genererer liste med månedsintervaller for API-kall."""
    start_date = datetime(start_year, 1, 1)
    max_date = datetime(end_year, 1, 1)
    ranges = []
    current_start = start_date

    while current_start < max_date:
        current_end = current_start + relativedelta(months=1)
        if current_end > max_date:
            current_end = max_date

        start_str = current_start.strftime('%Y-%m-%d')
        end_str = current_end.strftime('%Y-%m-%d')
        ranges.append((start_str, end_str))
        current_start = current_end

    return ranges


def get_price_area(from_date, to_date):
    """Henter rådata fra Elhub API for et gitt tidsintervall."""
    url = "https://api.elhub.no/energy-data/v0/price-areas"
    headers = {
        "Accept": "application/json",
        "User-Agent": "EnergyDataAnalysisProject/1.0",
    }
    params = {
        "dataset": "CONSUMPTION_PER_GROUP_MBA_HOUR",
        "id": "NO5",
        "startDate": from_date,
        "endDate": to_date
    }

    try:
        response = requests.get(url, params=params, headers=headers, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error: {e.response.status_code}")
    except Exception as err:
        print(f"Error: {err}")

    return None


def fetch_and_parse(start_time, end_time):
    """Henter og parser data for et tidsintervall til DataFrame."""
    data = get_price_area(start_time, end_time)

    if data is None:
        return None

    raw = data["data"]
    no5_list = [i for i in raw if i["id"] == "NO5"]

    if not no5_list:
        return None

    time_series = no5_list[0]["attributes"]["consumptionPerGroupMbaHour"]
    df = pd.DataFrame(time_series)
    df['startTime'] = pd.to_datetime(df['startTime'], utc=True).dt.tz_convert('Europe/Oslo')
    df['endTime'] = pd.to_datetime(df['endTime'], utc=True).dt.tz_convert('Europe/Oslo')
    df['lastUpdatedTime'] = pd.to_datetime(df['lastUpdatedTime'], utc=True).dt.tz_convert('Europe/Oslo')
    df.set_index('startTime', inplace=True)

    return df


def fetch_full_history(start_year=2021, end_year=2026, save_to_csv=True, load_from_csv=True):
    """Henter komplett historikk, enten fra CSV eller API."""
    filename = f"elhub_data_{start_year}_to_{end_year}.csv"

    if load_from_csv and os.path.exists(filename):
        print(f"Loading data from {filename}...")
        df = pd.read_csv(filename)
        df['startTime'] = pd.to_datetime(df['startTime'], utc=True).dt.tz_convert('Europe/Oslo')
        df.set_index('startTime', inplace=True)
        print(f"Loaded {len(df)} rows from {filename}")
        return df

    ranges = generate_monthly_ranges(start_year, end_year)
    all_dataframes = []

    print(f"Starting to fetch history from {start_year}...")

    for start_str, end_str in ranges:
        print(f"Fetching: {start_str} to {end_str}...", end="\r")

        try:
            df_month = fetch_and_parse(start_str, end_str)
            if df_month is not None and not df_month.empty:
                all_dataframes.append(df_month)
        except Exception as e:
            print(f"\nError fetching {start_str}: {e}")

        time.sleep(2)

    if all_dataframes:
        final_df = pd.concat(all_dataframes)
        final_df = final_df.sort_index()

        if save_to_csv:
            final_df.to_csv(filename)
            print(f"\nDone! Saved {len(final_df)} rows to {filename}")

        return final_df
    else:
        print("\nNo data found.")
        return pd.DataFrame()


def calculate_yearly_consumption(df):
    """Aggregerer forbruk til årlig nivå i GWh."""
    df_pivot = df.pivot(columns='consumptionGroup', values='quantityKwh')
    yearly = df_pivot.resample('YE').sum() / 1e6
    yearly.index = yearly.index.year
    yearly['total'] = yearly.sum(axis=1)
    return yearly


def calculate_yearly_change(yearly):
    """Beregner prosentvis endring fra år til år."""
    pct_change = yearly.pct_change() * 100
    return pct_change.dropna()


def find_group_trends(yearly):
    """Finner høyeste og laveste verdi for hver gruppe."""
    trends = {}
    for group in yearly.columns:
        max_year = yearly[group].idxmax()
        min_year = yearly[group].idxmin()
        max_val = yearly[group].max()
        min_val = yearly[group].min()
        trends[group] = {
            'max': (max_year, max_val),
            'min': (min_year, min_val)
        }
    return trends


def plot_yearly_consumption_per_group_bar(yearly, save_path=None):
    """Plotter årlig forbruk per gruppe som søylediagram i separate subplots."""
    groups = yearly.columns.tolist()
    n = len(groups)

    fig, axes = plt.subplots(1, n, figsize=(3 * n, 5), sharey=False)
    fig.suptitle('Årlig strømforbruk per gruppe i NO5', fontsize=14)

    for i, group in enumerate(groups):
        axes[i].bar(yearly.index.astype(str), yearly[group])
        axes[i].set_title(group)
        axes[i].set_ylabel('GWh')
        axes[i].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_yearly_change_per_group_bar(pct_change, save_path=None):
    """Plotter prosentvis endring fra år til år i separate subplots."""
    groups = pct_change.columns.tolist()
    n = len(groups)

    fig, axes = plt.subplots(1, n, figsize=(3 * n, 5), sharey=False)
    fig.suptitle('Årlig prosentvis endring i strømforbruk per gruppe i NO5', fontsize=14)

    for i, group in enumerate(groups):
        axes[i].bar(pct_change.index.astype(str), pct_change[group])
        axes[i].set_title(group)
        axes[i].set_ylabel('Endring (%)')
        axes[i].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        axes[i].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_daily_consumption_per_group_line(df, save_path=None):
    """Plotter daglig forbruk per gruppe i separate subplots."""
    df_pivot = df.pivot(columns='consumptionGroup', values='quantityKwh')
    daily = df_pivot.resample('D').sum() / 1e6  # Konverter til GWh

    groups = daily.columns.tolist()
    n = len(groups)

    fig, axes = plt.subplots(n, 1, figsize=(12, 1.5 * n), sharex=True)
    fig.suptitle('Daglig strømforbruk per gruppe i NO5', fontsize=14)

    for i, group in enumerate(groups):
        axes[i].plot(daily.index, daily[group])
        axes[i].set_ylabel('GWh')
        axes[i].set_title(group)

    axes[-1].set_xlabel('Dato')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_yearly_total_consumption_line(yearly, save_path=None):
    """Plotter totalt årlig forbruk som linjediagram."""
    plt.figure(figsize=(10, 5))
    plt.plot(yearly.index, yearly['total'], marker='o')
    plt.title('Totalt årlig strømforbruk i NO5')
    plt.xlabel('År')
    plt.ylabel('Forbruk (GWh)')
    plt.xticks(yearly.index)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def print_yearly_summary(yearly):
    """Skriver ut årlig forbruk per gruppe."""
    print("\nSTRØMFORBRUK PER ÅR PER GRUPPE (GWh)")
    print("=" * 50)
    print(yearly.round(1).to_string())


def print_change_summary(pct_change):
    """Skriver ut prosentvis endring."""
    print("\n\nPROSENTVIS ENDRING FRA ÅR TIL ÅR")
    print("=" * 50)
    print(pct_change.round(1).to_string())


def print_trends(trends):
    """Skriver ut trender per gruppe."""
    print("\n\nUTVIKLING PER FORBRUKERGRUPPE")
    print("=" * 50)
    for group, data in trends.items():
        max_year, max_val = data['max']
        min_year, min_val = data['min']
        print(f"{group}: Høyest i {max_year} ({max_val:.0f} GWh), lavest i {min_year} ({min_val:.0f} GWh)")


def print_timeline(yearly, pct_change):
    """Skriver ut tidslinje med verdier og prosentendring per gruppe."""
    print("\n\nTIDSLINJE PER FORBRUKERGRUPPE")
    print("=" * 50)

    years = yearly.index.tolist()

    for group in yearly.columns:
        parts = []
        for i, year in enumerate(years):
            val = yearly.loc[year, group]
            if i == 0:
                parts.append(f"{val:.0f}")
            else:
                change = pct_change.loc[year, group]
                sign = "+" if change >= 0 else ""
                parts.append(f"{val:.0f}({sign}{change:.1f}%)")

        print(f"{group}:")
        print(f"  {' -> '.join(parts)}")
        print()

if __name__ == "__main__":
    # Hent data
    df = fetch_full_history()

    # Analyser
    yearly = calculate_yearly_consumption(df)
    pct_change = calculate_yearly_change(yearly)
    trends = find_group_trends(yearly)

    # Skriv ut resultater
    print_yearly_summary(yearly)
    print_change_summary(pct_change)
    print_trends(trends)
    print_timeline(yearly, pct_change)

    # Visualiser
    plot_daily_consumption_per_group_line(df, save_path='consumption_daily_per_group_line.png')
    plot_yearly_consumption_per_group_bar(yearly, save_path='consumption_yearly_per_group_bar.png')
    plot_yearly_change_per_group_bar(pct_change, save_path='consumption_yearly_change_per_group_bar.png')
    plot_yearly_total_consumption_line(yearly, save_path='consumption_yearly_total_line.png')

    print("Vi ser at forbruksumønsteret har endret seg siden 2021.")
    print("Forbruket på hytter har gått ned fra 400 og stabilisert ser på 350Gwh")
    print("Primærnæring har vert relativ stabil gjennom hele perioden")
    print("Sekundernæring har sunket kraftig fra 9420 til 6426 og synker enda")
    print("Tertiernæring hadde en oppgang i 2022 og senere stabilisert seg rund 2500 fra 2021")
    print("Totalt så bruker vi mindre energi, fra 16000 Gwh til 13000 Gwh")

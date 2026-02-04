import pandas as pd
import requests
import time
from datetime import datetime
from dateutil.relativedelta import relativedelta

def generate_monthly_ranges(start_year=2021):
    start_date = datetime(start_year, 1, 1)
    end_of_all = datetime.now()
    
    ranges = []
    current_start = start_date
    
    while current_start < end_of_all:
        # Legg til nøyaktig én måned
        current_end = current_start + relativedelta(months=1)
        
        # Pass på at vi ikke spør om fremtiden
        if current_end > end_of_all:
            current_end = end_of_all
            
        # Formater til ISO-formatet Elhub krever (YYYY-MM-DDTHH:MM:SSZ)
        start_str = current_start.strftime('%Y-%m-%d')
        end_str = current_end.strftime('%Y-%m-%d')
        
        ranges.append((start_str, end_str))
        
        # Neste start er der forrige slutt slapp
        current_start = current_end
        
    return ranges

def get_consumption(
        from_date,
        to_date
        ):
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
    
    print(from_date, to_date)

    try:
        response = requests.get(url, params=params, headers=headers, timeout=10)
        
        # VIKTIG: Sjekk hva URL-en faktisk ble!
        print(f"DEBUG: Henter data fra: {response.url}")
        
        response.raise_for_status()
        return response.json()

    except requests.exceptions.HTTPError as e:
        if e.response.status_code >= 500:
            print(f"Server responded with statuscode: {e.response.status_code}")
        else:
            print(f"Bad request, statuscode: {e.response.status_code}")
        print(f"Error: {e}")

    except Exception as err:
        print(f"Something unexpectedly went wrong: {err}")

    return None

def fetch_and_parse(start_time, end_time):
    
    data = get_consumption(start_time, end_time)
    
    if data is not None:
        raw = data["data"]
        no5_list = [i for i in raw if i["id"] == "NO5"]
        
        if no5_list:
            time_series = no5_list[0]["attributes"]["consumptionPerGroupMbaHour"]
            df = pd.DataFrame(time_series)
            df['startTime'] = pd.to_datetime(df['startTime'], utc=True)
            df.set_index('startTime', inplace=True)
             
        return df

def fetch_full_history(start_year=2021, save_to_csv=True):
    """
    Henter strømdata måned for måned fra start_year frem til i dag.
    """
    current_start = datetime(start_year, 1, 1)
    end_of_all = datetime.now()
    all_dataframes = []

    print(f"🚀 Starter henting av historikk fra {start_year}...")

    while current_start < end_of_all:
        # 1. Beregn slutten på denne måneden
        current_end = current_start + relativedelta(months=1)
        if current_end > end_of_all:
            current_end = end_of_all

        # 2. Formater datoer for API-et
        start_str = current_start.strftime('%Y-%m-%d')
        end_str = current_end.strftime('%Y-%m-%d')

        print(f"📥 Henter: {start_str} til {end_str}...", end="\r")

        # 3. Kall din eksisterende funksjon (må endres til å ta start/end)
        try:
            df_month = fetch_and_parse(start_str, end_str)
            if not df_month.empty:
                all_dataframes.append(df_month)
        except Exception as e:
            print(f"\n❌ Feil ved henting av {start_str}: {e}")

        # 4. Gå til neste måned og vent litt (Rate Limiting)
        current_start = current_end
        time.sleep(0.5) 

    # 5. Slå sammen alt
    if all_dataframes:
        final_df = pd.concat(all_dataframes)
        # Sorter for sikkerhets skyld etter tid
        final_df = final_df.sort_index()
        
        if save_to_csv:
            filename = f"elhub_data_{start_year}_to_present.csv"
            final_df.to_csv(filename)
            print(f"\n✅ Ferdig! Lagret {len(final_df)} rader til {filename}")
        
        return final_df
    else:
        print("\n⚠️ Fant ingen data.")
        return pd.DataFrame()
    

if __name__ == "__main__":
    df = fetch_full_history()
    
        # Lag en tabell der hver gruppe får sin egen kolonne
    # Dette gjør at hver linje i grafen representerer én gruppe
    df_pivot = df.pivot(columns='consumptionGroup', values='quantityKwh')
    
    # Nå kan du se utviklingen time for time
    df_pivot.plot(figsize=(12, 6))

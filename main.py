import pandas as pd
import requests


def get_actual_consumption():
    url = "https://api.elhub.zno/energy-data/v0/price-areas"

    headers = {
        "Accept": "application/json",
        "User-Agent": "EnergyDataAnalysisProject/1.0",
    }

    params = {
        "dataset": "CONSUMPTION_PER_GROUP_MBA_HOUR",
        "id": "NO5",
        "fromDate": "2023-01-01T00:00:00Z",
        "toDate": "2023-01-01T00:04:00Z",
    }

    try:
        response = requests.get(url, params=params, headers=headers, timeout=10)
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


if __name__ == "__main__":
    data = get_actual_consumption()
    if data is not None:
        raw_list = data["data"][0]["attributes"]
        print(raw_list)

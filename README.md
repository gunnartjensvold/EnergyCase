# Energi Case - Strømforbruk NO5

Analyse av strømforbruk i prisområde NO5 (Vestland) basert på data fra Elhub API og værdata.

## Slik kjører du prosjektet

Opprett et virtuelt miljø: `python3 -m venv .venv`

Aktiver miljøet: `source .venv/bin/activate`

Installer nødvendige biblioteker: `pip install -r requirements.txt`

### Oppgave 1 - Forbrukstrender

Kjør analysen: `python consumption_trends.py`

Henter forbruksdata fra Elhub API, analyserer årlige trender og prosentvis endring per forbrukergruppe.

### Oppgave 2 - Værfaktorer

Kjør analysen: `python weather_analysis.py`

Kombinerer forbruks- og værdata, beregner korrelasjoner og finner hvilke værfaktorer som påvirker forbruket mest.

### Oppgave 3 - Prognosemodell

Kjør analysen: `python forecast_model.py`

Trener Lineær Regresjon og SVR-modeller for å predikere strømforbruk fra værdata. Sammenligner resultater med og uten shuffling av data.

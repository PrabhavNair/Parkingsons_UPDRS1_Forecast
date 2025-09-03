# Parkingsons_UPDRS1_Forecast
Forecast near-term UPDRS-I from routine clinic scores (XGBoost + Streamlit)

## Quickstart

```bash
1) Clone
git clone https://github.com/PrabhavNair/Parkinsons_UPDRS1_Forecast.git
cd Parkinsons_UPDRS1_Forecast

2) Create a clean environment (Windows PowerShell shown)
python -m venv .venv
. .venv/Scripts/Activate.ps1

3) Install deps
pip install --upgrade pip
pip install -r requirements.txt

4) Run the app
streamlit run app.py

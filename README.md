# heart_disease
# Heart Disease Risk Assessment

open the app via link:
https://heartdisease-i76uzpgvnqnsr3s2uimnqe.streamlit.app/

## Quick start

1. Create a virtual environment:
   - `python -m venv venv`
   - `source venv/bin/activate` (Linux/Mac) or `venv\Scripts\activate` (Windows)

2. Install dependencies:
   - `pip install -r requirements.txt`

3. Train the model (optional — app will train automatically if model is missing):
   - `python train_model.py --data heart_disease_sample.csv --model_out heart_model.joblib`

4. Run Streamlit app:
   - `streamlit run app.py`

## Files
- `heart_disease_sample.csv` — sample dataset
- `train_model.py` — training script
- `app.py` — Streamlit application
- `report.md` — project report template
- `requirements.txt` — dependencies

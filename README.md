# Electricity Consumption Forecasting

Interactive Streamlit mini-project with manual input, demo data, charts, statistics, and 24-hour forecasting. CSV upload is not required.

## Run on Windows
1. Extract the project ZIP.
2. Open the extracted folder in VS Code.
3. Open Terminal > New Terminal.
4. Install Streamlit if needed: `python -m pip install streamlit`
5. Run: `python -m streamlit run app.py`
6. Open `http://localhost:8501`.

NumPy, Pandas, Matplotlib, and Seaborn are used for analysis and charts. The forecasting model uses NumPy least squares; scikit-learn is not required.

## Using the app
- Choose **Enter readings manually** to add date, time, consumption (kWh), and temperature.
- Add readings for different times. For quick testing, select **Add 7 days of example readings**.
- Choose **Use demo data** to see charts immediately.
- View Overview, All chart analysis, 24-hour forecast, and Readings & export tabs.

Manual readings are held in Streamlit session memory and are not a permanent database. They may be lost when the session resets. A production application should save meter readings to a database and connect to a real smart meter/utility API. Demo readings are synthetic and should not be represented as real meter data.

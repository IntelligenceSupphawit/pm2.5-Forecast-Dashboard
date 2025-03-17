from dash import Dash, html, dcc
import dash_bootstrap_components as dbc
from pycaret.regression import RegressionExperiment
import pandas as pd
from datetime import datetime, timedelta
import plotly.express as px
from dash import dash_table

# ✅ Load PyCaret Models
exp_pm10 = RegressionExperiment()
exp_pm2_5 = RegressionExperiment()
pm10_model = exp_pm10.load_model(r'C:\Users\snpdp\pm2.5-Forecast-Dashboard\Final_model\pm10_nofeture')
pm2_5_model = exp_pm2_5.load_model(r'C:\Users\snpdp\pm2.5-Forecast-Dashboard\Final_model\final_pm2_5_model')

# ✅ ฟังก์ชันเตรียม Feature สำหรับ PM10git 
    df['day'] = df['date'].dt.day
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    return df

# ✅ ฟังก์ชันเตรียม Feature สำหรับ PM2.5
def prepare_features_pm2_5(df):
    
    df = df.copy()
    if 'pm2.5' not in df.columns:
        df['pm2.5'] = 0.0  # กำหนดค่าเริ่มต้น
    df['date_ordinal'] = df['date'].map(pd.Timestamp.toordinal)
    df['day'] = df['date'].dt.day
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['pm2_5_lag1'] = df['pm2.5'].shift(1)
    df['pm2_5_lag2'] = df['pm2.5'].shift(2)
    df['humidity_lag1'] = df['humidity'].shift(1)
    df['humidity_lag2'] = df['humidity'].shift(2)
    df['temperature_lag1'] = df['temperature'].shift(1)
    df['temperature_lag2'] = df['temperature'].shift(2)
    df['temperature humidity'] = df['temperature'] * df['humidity']
    df.fillna(method='bfill', inplace=True)
    return df

# ✅ เตรียมข้อมูล 7 วันข้างหน้า
future = pd.DataFrame()
start_date = pd.Timestamp.now()
future['date'] = [start_date + timedelta(days=i) for i in range(7)]
future['humidity'] = [62, 67, 65, 69, 70, 69, 72]
future['temperature'] = [27, 26, 27, 27, 27, 28, 29]

# ✅ สร้าง Feature ให้ตรงกับโมเดล
future_pm10 = prepare_features_pm10(future)
future_pm2_5 = prepare_features_pm2_5(future)

# ✅ พยากรณ์ PM10 และ PM2.5
future_pm10['pm10'] = exp_pm10.predict_model(pm10_model, data=future_pm10)['prediction_label']
future_pm2_5['pm2.5'] = exp_pm2_5.predict_model(pm2_5_model, data=future_pm2_5)['prediction_label']

# ✅ รวมผลลัพธ์ของ PM10 และ PM2.5
future['pm10'] = future_pm10['pm10']
future['pm2.5'] = future_pm2_5['pm2.5']

# ✅ ข้อมูลของวันนี้
today_temperature = future['temperature'].iloc[0]
today_humidity = future['humidity'].iloc[0]
today_pm10 = f"{future['pm10'].iloc[0]:.2f}"
today_pm2_5 = f"{future['pm2.5'].iloc[0]:.2f}"

# ✅ กราฟ PM10 และ PM2.5
fig_pm10 = px.bar(future, x='date', y='pm10', labels={'date': 'Date', 'pm10': 'PM10 Value'})
fig_pm2_5 = px.bar(future, x='date', y='pm2.5', labels={'date': 'Date', 'pm2.5': 'PM2.5 Value'})

# ✅ Format Table Data
future['pm10'] = future['pm10'].apply(lambda x: f'{x:.2f}')
future['pm2.5'] = future['pm2.5'].apply(lambda x: f'{x:.2f}')
future['date'] = future['date'].dt.strftime('%Y-%m-%d')

# ✅ Dash App
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

today_temperature = future['temperature'].iloc[0]
today_humidity = future['humidity'].iloc[0]

# Create the bar graph
fig = px.bar(future, x='date', y='pm10',
             labels={'date': 'Date', 'pm10': 'PM10 Value'})

# Format the pm10 values to two decimal places
future['pm10'] = future['pm10'].apply(lambda x: f'{x:.2f}')
# Format the date column to 'YYYY-MM-DD'
future['date'] = future['date'].dt.strftime('%Y-%m-%d')

# Dash app
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Calculate the height for both table and graph
table_height = (len(future) if len(future) <= 5 else 5) * 40 + 70  # Adjust 40 and 70 as needed for row height and header + padding
graph_height = table_height
app.layout = dbc.Container([
    dbc.Row(dbc.Col(html.H1("PM 2.5 Forecast Dashboard", className="text-center my-4"))),
    dbc.Row(dbc.Col(dbc.Alert("Real-time Air Quality Prediction", color="info", className="text-center"))),
    html.Hr(),

    # ✅ Cards สำหรับแสดงผลลัพธ์
    dbc.Row([
        dbc.Col(dbc.Card([dbc.CardBody([html.H4("Today's Temperature"), html.P(f"{today_temperature}°C")])], className="mb-3 bg-primary text-white text-center shadow"), width=3),
        dbc.Col(dbc.Card([dbc.CardBody([html.H4("Today's Humidity"), html.P(f"{today_humidity}%")])], className="mb-3 bg-success text-white text-center shadow"), width=3),
        dbc.Col(dbc.Card([dbc.CardBody([html.H4("PM2.5 Prediction"), html.P(f"{today_pm2_5} µg/m³")])], className="mb-3 bg-warning text-white text-center shadow"), width=3),
        dbc.Col(dbc.Card([dbc.CardBody([html.H4("PM10 Prediction"), html.P(f"{today_pm10} µg/m³")])], className="mb-3 bg-danger text-white text-center shadow"), width=3)
    ], justify="center"),

    
    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H4("PM2.5 Forecast Table"),
                dash_table.DataTable(
                    id='pm2.5-table',
                    columns=[{'name': 'Date', 'id': 'date'}, {'name': 'PM2.5 (µg/m³)', 'id': 'pm2.5'}],
                    data=future[['date', 'pm2.5']].to_dict('records'),
                    style_table={'width': '100%', 'margin': 'auto'}
                )
            ])
        ], className="mb-3 shadow"), width=6),

        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H4("PM10 Forecast Table"),
                dash_table.DataTable(
                    id='pm10-table',
                    columns=[{'name': 'Date', 'id': 'date'}, {'name': 'PM10 (µg/m³)', 'id': 'pm10'}],
                    data=future[['date', 'pm10']].to_dict('records'),
                    style_table={'width': '100%', 'margin': 'auto'}
                )
            ])
        ], className="mb-3 shadow"), width=6),
    ], align="start", justify="center"),

    
    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardBody([html.H4("PM2.5 Levels Over 7 Days"), dcc.Graph(figure=fig_pm2_5)])
        ], className="mb-3 shadow"), width=6),

        dbc.Col(dbc.Card([
            dbc.CardBody([html.H4("PM10 Levels Over 7 Days"), dcc.Graph(figure=fig_pm10)])
        ], className="mb-3 shadow"), width=6)
    ], align="start", justify="center")

], fluid=True)

# ✅ Run App
if __name__ == '__main__':
    app.run_server(debug=True)

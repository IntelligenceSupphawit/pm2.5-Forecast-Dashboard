from dash import Dash, html, dcc
import dash_bootstrap_components as dbc
from pycaret.regression import RegressionExperiment
import pandas as pd
from datetime import datetime, timedelta
import plotly.express as px

# โหลดโมเดล
exp = RegressionExperiment()
pm10_model = exp.load_model('pm10_nofeture')


# เตรียมข้อมูลสำหรับทำนาย
future = pd.DataFrame()
start_date = pd.Timestamp.now()
future['date'] = [start_date + timedelta(days=i) for i in range(7)]  # เพิ่มคอลัมน์วันที่และเดือนสำหรับอีก 7 วัน
future['day'] = future['date'].dt.day
future['month'] = future['date'].dt.month
future['year'] = future['date'].dt.year
future['humidity'] = [50, 55, 60, 65, 70, 75, 80]  # ตัวอย่างค่า humidity
future['temperature'] = [25, 26, 27, 28, 29, 30, 31]  # ตัวอย่างค่า temperature
future['pm10'] = exp.predict_model(pm10_model, data=future)['prediction_label']  # ทำนายค่า PM10

today_temperature = future['temperature'].iloc[0]
today_humidity = future['humidity'].iloc[0]

# สร้างกราฟแท่ง
fig = px.bar(future, x='day', y='pm10', title='PM10 Levels Over 7 Days', labels={'day': 'Day', 'pm10': 'PM10 Value'})

# Dash app
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.layout = dbc.Container([
    dbc.Row(
        dbc.Col(html.H1("PM 2.5 Forecast"), width=12)
    ),
    html.Hr(),
    # dbc.Row( # pm 2.5 เลือกที่ได้
    #     dbc.Col(html.Pre(future[['date', 'pm10']].to_string(index=False), className="border p-3"), width=12)
    # ),
    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H4("Today's Temperature", className="card-title text-white"),
                html.P(f"{today_temperature}°C", className="card-text text-white")
            ])
        ], className="mb-3 bg-dark text-center", style={"width": "150px", "height": "150px"}), width="auto"),
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H4("Today's Humidity", className="card-title text-white"),
                html.P(f"{today_humidity}%", className="card-text text-white")
            ])
        ], className="mb-3 bg-dark text-center", style={"width": "150px", "height": "150px"}), width="auto")
    ], justify="center"),
    
    dbc.Row( #ตารางทำนาย
        dbc.Col(html.Pre(future[['date', 'pm10']].to_string(index=False), className="border p-3"), width=12)
    ),
    dbc.Row(
        dbc.Col(dbc.Card([
            dbc.CardBody([
                dcc.Graph(figure=fig)
            ])
        ]), width=12)
    )
], fluid=True)
    

# Run the application
if __name__ == '__main__':
    app.run_server(debug=True)

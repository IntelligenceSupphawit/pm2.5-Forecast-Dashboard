import pickle
import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
from dash.dependencies import Input, Output

# โหลดโมเดล
model_path = 'pm10_forecast.pkl'
with open(model_path, 'rb') as f:
    model = pickle.load(f)

# พยากรณ์ PM10 สำหรับ 7 วันข้างหน้า
def forecast_pm10():
    future_dates = pd.date_range(start=pd.Timestamp.today(), periods=7, freq='D')
    predictions = model.predict([[i] for i in range(1, 8)])  # ตัวอย่างการพยากรณ์
    df = pd.DataFrame({'Date': future_dates, 'PM10': predictions})
    return df

data = forecast_pm10()

# สร้างแอป Dash
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Layout
app.layout = dbc.Container([
    html.H1("PM10 Forecast for the Next 7 Days", className="text-center mt-4"),
    dcc.Graph(id='pm10-forecast-graph'),
], fluid=True)

# Callback อัปเดตกราฟ
@app.callback(
    Output('pm10-forecast-graph', 'figure'),
    Input('pm10-forecast-graph', 'id')
)
def update_graph(_):
    fig = px.line(data, x='Date', y='PM10', markers=True, title="Predicted PM10 Levels")
    fig.update_layout(xaxis_title="Date", yaxis_title="PM10 Levels")
    return fig

# Run แอป
if __name__ == '__main__':
    app.run_server(debug=True)
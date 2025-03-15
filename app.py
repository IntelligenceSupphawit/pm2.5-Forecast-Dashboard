from dash import Dash, html, dcc
import dash_bootstrap_components as dbc
from pycaret.time_series import TSForecastingExperiment
import pandas as pd

# Load PyCaret Experiment
exp = TSForecastingExperiment()

# Load the trained model
pm10_model = exp.load_model('pm10_model')

# Forecast PM10 for the next 7 days
future_forecast = exp.predict_model(pm10_model, fh=7)

# Ensure the forecast has a proper index and column name
future_forecast = future_forecast.rename(columns={'y_pred': 'pm10'})
future_forecast.index = pd.date_range(start=pd.Timestamp.now(), periods=7, freq='D')

# Create the Dash app
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Layout of the app
app.layout = dbc.Container([
    dbc.Row(
        dbc.Col(html.H1("PM10 7-Day Forecast"), width=12)
    ),
    dbc.Row(
        dbc.Col(html.Pre(future_forecast.to_string(index=False), className="border p-3"), width=12)
    ),
    dbc.Row(
        dbc.Col(dcc.Graph(
            figure={
                'data': [
                    {'x': future_forecast.index, 'y': future_forecast['pm10'], 'type': 'line', 'name': 'PM10'},
                ],
                'layout': {
                    'title': 'PM10 7-Day Forecast',
                    'xaxis': {'title': 'Date'},
                    'yaxis': {'title': 'PM10 Level'}
                }
            }
        ), width=12)
    )
], fluid=True)

# Run the application
if __name__ == '__main__':
    app.run_server(debug=True)
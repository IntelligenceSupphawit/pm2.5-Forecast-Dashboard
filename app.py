from dash import Dash, html, dcc
import dash_bootstrap_components as dbc
from pycaret.regression import RegressionExperiment, load_model
import pandas as pd
from datetime import datetime, timedelta
import plotly.express as px
import os
from dash import dash_table  # Import dash_table

# Initialize the model variable outside the try block
pm10_model = None
exp = RegressionExperiment()

# Attempt to load the model with error handling
try:
    pm10_model = load_model('pm10_nofeture')
    # Check if the model was loaded successfully
    if pm10_model is None:
        raise FileNotFoundError("Model 'pm10_nofeture' loaded as None.")
    print("Model loaded successfully.")
except FileNotFoundError as e:
    print(f"Error: Model 'pm10_nofeture' not found. Please ensure it exists in the same directory. {e}")
except Exception as e:
    print(f"An unexpected error occurred while loading the model: {e}")

# Prepare data for prediction
future = pd.DataFrame()
start_date = pd.Timestamp.now()
future['date'] = [start_date + timedelta(days=i) for i in range(7)]
future['day'] = future['date'].dt.day
future['month'] = future['date'].dt.month
future['year'] = future['date'].dt.year
future['humidity'] = [50, 55, 60, 65, 70, 75, 80]
future['temperature'] = [25, 26, 27, 28, 29, 30, 31]

if pm10_model is not None:
    try:
        future['pm10'] = exp.predict_model(pm10_model, data=future)['prediction_label']
    except KeyError as e:
        print(f"Error: The model output is not in the expected format: {e}")
        future['pm10'] = [0] * 7 # Default values
    except Exception as e:
        print(f"An unexpected error occurred during prediction: {e}")
        future['pm10'] = [0] * 7  # Default values
else:
    # If the model did not load, set default values
    future['pm10'] = [0] * 7

today_temperature = future['temperature'].iloc[0]
today_humidity = future['humidity'].iloc[0]

# Create the bar graph
fig = px.bar(future, x='day', y='pm10', title='PM10 Levels Over 7 Days',
             labels={'day': 'Day', 'pm10': 'PM10 Value'})

# Format the pm10 values to two decimal places
future['pm10'] = future['pm10'].apply(lambda x: f'{x:.2f}')
# Format the date column to 'YYYY-MM-DD'
future['date'] = future['date'].dt.strftime('%Y-%m-%d')

# Dash app
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.layout = dbc.Container([
    dbc.Row(
        dbc.Col(html.H1("PM 2.5 Forecast", className="text-center my-4"), width=12)
    ),
    html.Hr(),
    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H4("Today's Temperature", className="card-title text-white"),
                html.P(f"{today_temperature}°C", className="card-text text-white")
            ])
        ], className="mb-3 bg-primary text-center shadow", style={"height": "150px"}), width=4),
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H4("Today's Humidity", className="card-title text-white"),
                html.P(f"{today_humidity}%", className="card-text text-white")
            ])
        ], className="mb-3 bg-success text-center shadow", style={"height": "150px"}), width=4),
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H4("PM10 Prediction", className="card-title text-white"),
                html.P(f"{future['pm10'].iloc[0]} µg/m³", className="card-text text-white")
            ])
        ], className="mb-3 bg-danger text-center shadow", style={"height": "150px"}), width=4)
    ], justify="center"),

    dbc.Row(
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H4("PM10 Forecast Table", className="card-title"),
                dash_table.DataTable(
                    id='pm10-table',
                    columns=[
                        {'name': 'Date', 'id': 'date'},
                        {'name': 'PM10 (µg/m³)', 'id': 'pm10'}
                    ],
                    data=future[['date', 'pm10']].to_dict('records'),
                    page_size=5,  # Number of rows per page
                    style_header={
                        'backgroundColor': 'rgb(210, 210, 210)',
                        'color': 'black',
                        'fontWeight': 'bold'
                    },
                    style_cell={
                        'textAlign': 'left',
                        'minWidth': '80px',
                        'maxWidth': '80px',
                        'whiteSpace': 'normal'
                    },
                    style_data_conditional=[
                       {
                            'if': {'row_index': 'odd'},
                            'backgroundColor': 'rgb(248, 248, 248)'
                        }
                    ],
                    style_table={'width': '70%', 'margin': 'auto'}, #reduce the width of the table
                )
            ])
        ], className="mb-3 shadow"), width=8, style={'margin': 'auto'}), #reduce the column to 8 and add margin
    ),
    dbc.Row(
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H4("PM10 Levels Over 7 Days", className="card-title"),
                dcc.Graph(figure=fig, style={'height': '400px','width':'95%'}) #reduce the height and width of the graph
            ])
        ], className="mb-3 shadow"), width=10, style={'margin':'auto'}) #reduce the column to 10 and add margin
    )
], fluid=True)

# Run the application
if __name__ == '__main__':
    app.run_server(debug=True)

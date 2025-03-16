from dash import Dash, html, dcc
import dash_bootstrap_components as dbc
from pycaret.regression import RegressionExperiment, load_model
import pandas as pd
from datetime import datetime, timedelta
import plotly.express as px
import os
from dash import dash_table  # Import dash_table

# Initialize the model variable outside the try block
exp = RegressionExperiment()
pm10_model = load_model('pm10_nofeture')

# Prepare data for prediction
future = pd.DataFrame()
start_date = pd.Timestamp.now()
future['date'] = [start_date + timedelta(days=i) for i in range(7)]
future['day'] = future['date'].dt.day
future['month'] = future['date'].dt.month
future['year'] = future['date'].dt.year
future['humidity'] = [50, 55, 60, 65, 70, 75, 80]
future['temperature'] = [25, 26, 27, 28, 29, 30, 31]
future['pm10'] = exp.predict_model(pm10_model, data=future)['prediction_label']

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

    dbc.Row([ #New row
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
                            'minWidth': '100px',
                            'maxWidth': '100px',
                            'whiteSpace': 'normal'
                        },
                        style_data_conditional=[
                           {
                                'if': {'row_index': 'odd'},
                                'backgroundColor': 'rgb(248, 248, 248)'
                            }
                        ],
                        style_table={'width': '100%', 'margin': 'auto', 'height': f'{table_height}px', 'overflowY': 'auto', 'minWidth': '100%'}, #Set table height
                    )
                ])
        ], className="mb-3 shadow", style={'width': '100%'}), width=6), # 6 column for table

        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H4("PM10 Levels Over 7 Days", className="card-title"),
                dcc.Graph(figure=fig, style={'height': f'{graph_height}px','width':'100%'}) #Set graph height
            ])
        ], className="mb-3 shadow", style={'width': '100%'}), width=6) # 6 column for graph
    ], align="start", justify="center"), #align to start and center

], fluid=True)

# Run the application
if __name__ == '__main__':
    app.run_server(debug=True)

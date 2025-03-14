from dash import Dash, html, dcc, callback, Output, Input
import plotly.express as px
import pandas as pd

from pycaret.regression import RegressionExperiment
import dash_bootstrap_components as dbc
exp = RegressionExperiment()

model = exp.load_model('pm10_model')
app = Dash()

app.layout = dbc.Container([
    dbc.Row(
        
        dbc.Col(
            html.H1(
                children='Title of Dash App', 
                className='text-center my-4'
            )
        ),
    

    ),
    dbc.Row([
        dbc.Col(html.Label('Age'), width=6),

    dbc.Col(
            dcc.Dropdown(
                options=[
                    {'label': 'Male', 'value': 'male'},
                    {'label': 'Female', 'value': 'female'}
                ], 
                value='male', 
                id='sex', 
                className='form-control mb-3'
            ), 
            width=6
        )


    ]),
    dbc.Row([
        dbc.Col(html.Label('Sex'), width=6),
        dbc.Col(
            dcc.Input(
                id='age', 
                type='number', 
                placeholder='Enter Age', 
                className='form-control mb-3'
            ), 
            width=6
        ),
            ]),
    dbc.Row([
        dbc.Col(html.Label('BMI'), width=6),


        dbc.Col(
            dcc.Input(
                id='bmi', 
                type='number', 
                placeholder='Enter BMI', 
                className='form-control mb-3'
            ), 
            width=6
        ),




    ]),
    dbc.Row([
        
        dbc.Col(html.Label('Number of Children'), width=6),
        dbc.Col(
            dcc.Input(
                id='children', 
                type='number', 
                placeholder='Enter Number of Children', 
                className='form-control mb-3'
            ), 
            width=6
        )
    ]),
    dbc.Row([
        dbc.Col(html.Label('Smoker'), width=6),


        dbc.Col(
            dcc.Dropdown(
                options=[
                    {'label': 'Yes', 'value': 'yes'},
                    {'label': 'No', 'value': 'no'}
                ], 
                value='no', 
                id='smoker', 
                className='form-control mb-3'
            ), 
            width=6
        ),


    ]),
    dbc.Row([

        
        dbc.Col(html.Label('Region'), width=6),
        dbc.Col(
            dcc.Dropdown(
                options=[
                    {'label': 'North', 'value': 'north'},
                    {'label': 'Northeast', 'value': 'northeast'},
                    {'label': 'East', 'value': 'east'},
                    {'label': 'Southeast', 'value': 'southeast'},
                    {'label': 'South', 'value': 'south'},
                    {'label': 'Southwest', 'value': 'southwest'},
                    {'label': 'West', 'value': 'west'},
                    {'label': 'Northwest', 'value': 'northwest'}
                ], 
                value='north', 
                id='region', 
                className='form-control mb-3'
            ), 
            width=6
        )
    ]),
    dbc.Row(
        dbc.Col(
            html.Div(
                id='prediction-content', 
                className='mt-4'
            )
        )
    )
], fluid=True)

@callback(
    Output('prediction-content', 'children'),
    Input('sex', 'value'),
    Input('age', 'value'),
    Input('bmi', 'value'),
    Input('children', 'value'), 
    Input('smoker', 'value'), 
    Input('region', 'value')
)
def predict(sex, age, bmi, children, smoker, region):
    if None in [sex, age, bmi, children, smoker, region]:
        return 'Please provide all input values.'
    
    input_data = pd.DataFrame({
        'sex': [sex],
        'age': [age],
        'bmi': [bmi],
        'children': [children],
        'smoker': [smoker],
        'region': [region]
    })
    prediction = exp.predict_model(model, data=input_data)

    return f'Prediction: {prediction["prediction_label"][0]}'

if __name__ == '__main__':
    app.run(debug=True)
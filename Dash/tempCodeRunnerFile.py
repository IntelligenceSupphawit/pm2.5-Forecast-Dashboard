app = dash.Dash(__name__)

app.layout = html.Div([
    html.H2("ไฟล์ CSV อื่นๆ"),
    dcc.Upload(
        id='upload-data',
        children=html.Button('เลือกไฟล์ CSV', style={'fontSize':18}),
        accept='.csv'
    ),
    html.Div(id='output-data-upload')
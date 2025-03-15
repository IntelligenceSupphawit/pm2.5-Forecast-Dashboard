import dash
from dash import dcc, html, Input, Output, State, dash_table
import pandas as pd
import io
import base64

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H2("ไฟล์ CSV อื่นๆ"),
    dcc.Upload(
        id='upload-data',
        children=html.Button('เลือกไฟล์ CSV', style={'fontSize': 18}),
        accept='.csv'
    ),
    html.Div(id='output-data-upload')
])

def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if filename.endswith('.csv'):
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        else:
            return html.Div(['ไฟล์ที่เลือกไม่ใช่ CSV'])
    except Exception as e:
        return html.Div(['เกิดข้อผิดพลาด: ' + str(e)])

    return html.Div([
        html.H5(filename),
        dash_table.DataTable(
            data=df.to_dict('records'),
            columns=[{'name': col, 'id': col} for col in df.columns],
            page_size=10,
            style_table={'overflowX': 'auto'}
        )
    ])

@app.callback(
    Output('output-data-upload', 'children'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def update_output(contents, filename):
    if contents is not None:
        return parse_contents(contents, filename)
    else:
        return html.Div('ยังไม่ได้เลือกไฟล์ CSV')

if __name__ == '__main__':
    app.run_server(debug=True)

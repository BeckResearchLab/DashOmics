from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html

from app import app
import homepage
from ModelEvaluation import SilhouetteAnalysis, ElbowMethod


app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])


@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    print(pathname)
    if pathname is None:
         return "Loading"
    elif pathname == '/ModelEvaluation/SilhouetteAnalysis':
         return SilhouetteAnalysis.layout
    elif pathname == '/ModelEvaluation/ElbowMethod':
         return ElbowMethod.layout
    else:
        return homepage.layout

if __name__ == '__main__':
    app.run_server(debug=True)
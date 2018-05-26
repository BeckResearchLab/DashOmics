from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
import dash_table_experiments as dt

from app import app
import homepage
from ModelEvaluation import SilhouetteAnalysis, ElbowMethod
from ClustersProfile import ClustersOverview, ChooseCluster, ChooseGene

import sqlite3

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content'),
    html.Div(dt.DataTable(rows=[{}]), style={'display': 'none'})
])


@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    print('run.py -- Call display_page function')
    if pathname is None:
         return "Loading"
    elif pathname == '/ModelEvaluation/SilhouetteAnalysis':
         return SilhouetteAnalysis.layout
    elif pathname == '/ModelEvaluation/ElbowMethod':
         return ElbowMethod.layout
    elif pathname == '/ClustersProfile/ClustersOverview':
         return ClustersOverview.layout
    elif pathname == '/ClustersProfile/ChooseCluster':
         return ChooseCluster.layout
    elif pathname == '/ClustersProfile/ChooseGene':
         return ChooseGene.layout
    else:
        return homepage.layout

print('run.py -- running successfully')

if __name__ == '__main__':
    app.run_server(debug=True)
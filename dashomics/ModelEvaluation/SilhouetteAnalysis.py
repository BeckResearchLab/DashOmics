import dash
from dash.dependencies import Input, Output
import dash_html_components as html
import dash_core_components as dcc

import plotly.graph_objs as go
import plotly
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
#import sys
#import os
#sys.path.append(os.path.join(os.path.dirname(sys.path[0])))
from app import app

import sqlite3
import re

app.config.supress_callback_exceptions = True

layout = html.Div([
    html.H3('Model Evaluation: Silhouette Analysis'),
    dcc.Input(id='k-range', value= 10, type='number'),
    dcc.Graph(id='graph-silhouette_analysis'),
    html.Div(id='app-1-display-value'),
    html.Div([
        dcc.Link('Go to Home Page', href='/'),
        html.P(''),
        dcc.Link('Go to Elbow Method', href='/ModelEvaluation/ElbowMethod'),
        html.P(''),
        dcc.Link('Go to Clusters Overview', href='/ClustersProfile/ClustersOverview'),
        html.P(''),
        dcc.Link('Go to Choose Gene', href='/ClustersProfile/ChooseGene')
        ])
    ])

@app.callback(
    Output('graph-silhouette_analysis', 'figure'),
    [Input(component_id='k-range',component_property='value')]
)

def silhouette_analysis(n):
    """
    n: the maximum of k value

    """
    con = sqlite3.connect('dashomics_test.db')
    c = con.cursor()
    # create a table in db to store what users choose at homepage
    c.execute('''SELECT filename FROM sql_master
                 WHERE Choose_or_Not = 'Yes'
              ''')
    filename = re.findall(r"'(.*?)'", str(c.fetchone()))[0]  # choose the first filename that match the pattern

    con.commit()
    df = pd.read_sql_query('SELECT * FROM %s' % str(filename), con).set_index('id')
    con.close()

    k_values = np.array([])
    silhouette_scores = np.array([])

    K = range(2, n + 1)

    for i in K:
        # K-means for range of clusters
        kmeans = KMeans(n_clusters=i, max_iter=300, random_state=3)
        kmeans.fit(df)

        # Silhouette score for every k computed
        silhouette_ave = silhouette_score(df.values, kmeans.labels_)

        # x and y axis to plot k value and silhouette score
        k_values = np.append(k_values, [int(i)])
        silhouette_scores = np.append(silhouette_scores, [silhouette_ave])

    return {
        'data': [go.Bar(
            x=k_values,
            y=silhouette_scores,

        )],
        'layout': go.Layout(
            xaxis={'title': 'K Value'},
            yaxis={'title': 'Silhouette coefficient (Sc) Value'},
            margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
            hovermode='closest'
        )
    }
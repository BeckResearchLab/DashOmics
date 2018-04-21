from dash.dependencies import Input, Output

import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import plotly
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist, pdist

#import sys
#import os
#sys.path.append(os.path.join(os.path.dirname(sys.path[0])))

print(__file__)
from app import app

df = pd.read_csv('df5_log2_ratio.csv', index_col = ['locus_tag'])

layout = html.Div([
    html.H3('Model Evaluation：Elbow Method'),
    dcc.Input(id='k-range', value= 10, type='number'),
    dcc.Graph(id='graph-elbow_method'),
    html.Div(id='app-2-display-value'),
    html.Div([
        dcc.Link('Go to Home Page', href='/'),
        html.P(''),
        dcc.Link('Go to Silhouette Analysis', href='/apps/app1')
    ])
])

@app.callback(
    Output('graph-elbow_method', 'figure'),
    [Input(component_id='k-range',component_property='value')]
)
def elbow_method_evaluation(n):
    """
    n: the maximum of k value

    """
    # Fit the kmeans model for k in a certain range
    K = range(1, n + 1)
    KM = [KMeans(n_clusters=k).fit(df) for k in K]
    # Pull out the cluster centroid for each model
    centroids = [k.cluster_centers_ for k in KM]

    # Calculate the distance between each data point and the centroid of its cluster
    k_euclid = [cdist(df.values, cent, 'euclidean') for cent in centroids]
    dist = [np.min(ke, axis=1) for ke in k_euclid]

    # Total within sum of square
    wss = [sum(d ** 2) / 1000 for d in dist]
    # The total sum of square
    tss = sum(pdist(df.values) ** 2) / df.values.shape[0]
    # The between-clusters sum of square
    bss = tss - wss

    # Difference of sum of within cluster distance to next smaller k
    dwss = [wss[i + 1] - wss[i] for i in range(len(wss) - 1)]
    dwss.insert(0, 0) # insert value of 0 at first position of dwss

    # Create the graph with subplots
    fig = plotly.tools.make_subplots(rows=2, cols=1, vertical_spacing=0.2, shared_xaxes=True,
                                     subplot_titles=('Sum of Within-cluster Distance/1000',
                                                     'Difference of Sum of Within-cluster Distance to Next Lower K/1000'))
    fig['layout']['margin'] = {
        'l': 40, 'r': 40, 'b': 40, 't': 40
    }
    #fig['layout']['legend'] = {'x': 0, 'y': 1, 'xanchor': 'left'}

    fig.append_trace({
        'x': list(K),
        'y': list(wss),
        #'name': 'Sum of Within-cluster Distance/1000',
        'mode': 'lines+markers',
        'type': 'scatter'
    }, 1, 1)
    fig.append_trace({
        'x': list(K),
        'y': list(dwss),
        #'text': data['time'],
        #'name': 'Difference of Sum of Within-cluster Distance to Next Lower K/1000',
        'mode': 'lines+markers',
        'type': 'scatter'
    }, 2, 1)

    fig['layout']['xaxis1'].update(title='K Value')
    fig['layout']['xaxis2'].update(title='K Value')
    fig['layout']['yaxis1'].update(title='Distance Value')
    fig['layout']['yaxis2'].update(title='Distance Value')

    fig['layout'].update(height=600, width=1000,
                         title='Model Evaluation: Elbow Method for Optimal K Value')

    return fig

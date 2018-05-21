import pandas as pd

import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go

from sklearn.cluster import KMeans

print(__file__)
from app import app

import sqlite3
import re

#df = pd.read_csv('df5_log2_ratio.csv', index_col = ['locus_tag'])

layout = html.Div([
    html.Div([
        html.H3('Choose Gene and Display its cluster'),
        dcc.Input(id='input_gene', value='gene name', type='text'),

        html.P(''),

        html.Label('Choose K Value'),
        dcc.Input(id='k-value', value= 15, type='number')
    ]),

    dcc.Graph(id='graph-gene-clusterprofile'),

    # Links
    html.Div([
        dcc.Link('Go to Home Page', href='/'),
        html.P(''),
        dcc.Link('Go to Silhouette Analysis', href='/ModelEvaluation/SilhouetteAnalysis'),
        html.P(''),
        dcc.Link('Go to Elbow Method', href='/ModelEvaluation/ElbowMethod'),
        html.P(''),
        dcc.Link('Go to Clusters Overview', href='/ClustersProfile/ClustersOverview')
    ])
])


#Display a single cluster that input gene belongs to
@app.callback(
    Output('graph-gene-clusterprofile', 'figure'),
    [Input(component_id='input_gene',component_property='value'),
     Input(component_id='k-value',component_property='value')]
)

def gene_clusterprofile(input_gene, k_value):

    # extract df from sqlite database
    con = sqlite3.connect('dashomics_test.db')
    c = con.cursor()
    # create a table in db to store what users choose at homepage
    c.execute('''SELECT filename FROM sql_master
                 WHERE Choose_or_Not = 'Yes'
              ''')
    con.commit()
    filename = re.findall(r"'(.*?)'", str(c.fetchone()))[0]  # choose the first filename that match the pattern

    df = pd.read_sql_query('SELECT * FROM %s' % str(filename), con).set_index('id')
    con.close()

    X = df

    kmeans = KMeans(n_clusters=k_value, max_iter=300, random_state=4)
    kmeans.fit(X)

    labels_kmeans = kmeans.labels_
    df_clusterid = pd.DataFrame(labels_kmeans, index=df.index)
    df_clusterid.rename(columns={0: "cluster"}, inplace=True)
    df_clusters = pd.concat([df, df_clusterid], axis=1)

    genes_clusterid = df_clusterid.loc[input_gene]

    count = df_clusters.groupby('cluster').count().iloc[:, 0]

    y_stdev = df_clusters.groupby("cluster").std()
    y_mean = df_clusters.groupby("cluster").mean()

    y_low = y_mean.subtract(y_stdev, fill_value=0)
    y_high = y_mean.add(y_stdev, fill_value=0)

    title_str = "Cluster #" + str(genes_clusterid) + \
                " Profile Overview (including " + str(count[genes_clusterid]) + " genes)"

    tracey = go.Scatter(
                x = list(range(len(df_clusters.columns) - 1)),
                y = y_mean.values[genes_clusterid])

    tracey_lo = go.Scatter(
                x = list(range(len(df_clusters.columns) - 1)),
                y = y_low.values[genes_clusterid])

    tracey_hi = go.Scatter(
            x=list(range(len(df_clusters.columns) - 1)),
            y=y_high.values[genes_clusterid])

    return {'data':[tracey, tracey_lo, tracey_hi],
        'layout': go.Layout(height=300, width=800,
            title=title_str,
            #xaxis = {'title':'cluster id'},
            #yaxis = {'title':'number of genes in each cluster'},
            margin={'l': 40, 'b': 40, 't': 40, 'r': 40},
            hovermode='closest'
        )
    }
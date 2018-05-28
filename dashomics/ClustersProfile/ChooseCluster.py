import pandas as pd

from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import dash_table_experiments as dt

from sklearn.cluster import KMeans

print(__file__)
from app import app

import sqlite3
import re


layout = html.Div([
    html.H3('Step 2 -- Cluster Profile: Choose Cluster'),

    #Links
    html.Div([
        dcc.Link('Go to Home Page -- Step 0: Define Input Data', href='/'),
        html.P(''),
        dcc.Link('Go to Step 1 -- Model Evaluation: Elbow Method', href='/ModelEvaluation/ElbowMethod'),
        html.P(''),
        dcc.Link('Go to Step 1 -- Model Evaluation: Silhouette Analysis', href='/ModelEvaluation/SilhouetteAnalysis'),
        html.P(''),
        dcc.Link('Go to Step 2 -- Cluster Profile: Clusters Overview', href='/ClustersProfile/ClustersOverview')
    ]),
    html.P(''),

    html.Div([
        html.Div([
            html.H4('Choose K Value'),
            dcc.Input(id='k-value', value= 15, type='number')],className="row1"),
        html.Div([
            html.H4('Choose Cluster id'),
            dcc.Input(id='cluster-id', value=0, type='number')],className="row1")
        ], className="row"),

    dcc.Graph(id='graph-single-cluster-profile'),
    dt.DataTable(
        rows=[{}],
        editable=True, id='gene-table'
        ),

    # Links
    html.Div([
        dcc.Link('Go to Step 2 -- Cluster Profile: Choose Gene', href='/ClustersProfile/ChooseGene')
    ])
])


#Display a single cluster
@app.callback(Output('graph-single-cluster-profile', 'figure'),
            [Input(component_id='k-value',component_property='value'),
             Input(component_id='cluster-id', component_property='value')])

def cluster_profile(kvalue,clusterid):

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

    kmeans = KMeans(n_clusters=kvalue, max_iter=300, random_state=4)
    kmeans.fit(X)

    labels_kmeans = kmeans.labels_
    df_clusterid = pd.DataFrame(labels_kmeans, index=df.index)
    df_clusterid.rename(columns={0: "cluster"}, inplace=True)
    df_clusters = pd.concat([df, df_clusterid], axis=1)

    count = df_clusters.groupby('cluster').count().iloc[:, 0]

    y_stdev = df_clusters.groupby("cluster").std()
    y_mean = df_clusters.groupby("cluster").mean()

    y_low = y_mean.subtract(y_stdev, fill_value=0)
    y_high = y_mean.add(y_stdev, fill_value=0)

    title_str = "Cluster #" + str(clusterid) + \
                " Profile Overview (including " + str(count[clusterid]) + " genes)"

    #set sample name as x-ticks
    samplename = list(df.columns)

    tracey = go.Scatter(
                x = samplename,
                y = y_mean.values[clusterid],
                name = 'the mean of gene expression level')

    tracey_lo = go.Scatter(
                x=samplename,
                y = y_low.values[clusterid],
                name='the minimum of gene expression level')

    tracey_hi = go.Scatter(
                x=samplename,
                y=y_high.values[clusterid],
                name='the maximum of gene expression level')

    return {'data':[tracey, tracey_lo, tracey_hi],
        'layout': go.Layout(height=500, width=1300,
            title=title_str,
            titlefont=dict(family='Arial, sans-serif',size=18),
            xaxis=dict(
                    title="Sample Name",
                    showgrid=True,
                    showline=True,
                    showticklabels=True,
                    tickangle=45),
            yaxis=dict(
                    title="Expression Level",
                    showgrid=True,
                    showline=True),
            legend=dict(x=0.1, y=1.1,
                        traceorder='normal',
                        orientation="h",
                        font=dict(
                            family='sans-serif',
                            size=12)),
            showlegend=True,
            margin={'l': 60, 'b': 150, 't': 60, 'r': 60},
            hovermode='closest'
        )
    }

@app.callback(Output('gene-table', 'rows'),
              [Input(component_id='k-value', component_property='value'),
               Input(component_id='cluster-id', component_property='value')])

def generate_gene_table(kvalue,clusterid):

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

    kmeans = KMeans(n_clusters=kvalue, max_iter=300, random_state=4)
    kmeans.fit(X)

    labels_kmeans = kmeans.labels_
    df_clusterid = pd.DataFrame(labels_kmeans, index=df.index)
    df_clusterid.rename(columns={0: "cluster"}, inplace=True)
    #count = df_clusterid.groupby('cluster').count().iloc[:, 0]
    gene_table = pd.DataFrame(list(df_clusterid.loc[df_clusterid.cluster == 5].index),
                              columns=['Genes_in_Cluster#%s' % clusterid])

    return gene_table.to_dict('records')
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
from dash.dependencies import Input, Output

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

print(__file__)
from app import app

import sqlite3
import re


layout = html.Div([
        html.H3('Clusters Profile: Overview'),

        html.Label('Choose K Value'),
        dcc.Input(id='k-value', value= 15, type='number'),

        html.P(''),

        html.Label('Choose Cluster id'),
        dcc.Input(id='cluster-id', value = 0, type = 'number'),

    dcc.Graph(id='graph-cluster-size'),
    dcc.Graph(id='graph-pca-2d'),
    dcc.Graph(id='graph-cluster-profile'),

    #Links
    html.Div([
        dcc.Link('Go to Home Page', href='/'),
        html.P(''),
        dcc.Link('Go to Silhouette Analysis', href='/ModelEvaluation/SilhouetteAnalysis'),
        html.P(''),
        dcc.Link('Go to Elbow Method', href='/ModelEvaluation/ElbowMethod'),
        html.P(''),
        dcc.Link('Go to Choose Gene', href='/ClustersProfile/ChooseGene')
    ])
])


#Cluster Sizes
@app.callback(
    Output('graph-cluster-size', 'figure'),
    [Input(component_id='k-value',component_property='value')]
)

def cluster_size_figure(kvalue):

    #extract df from sqlite database
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

    return {
        'data':[go.Bar(
            x = list(count.index),
            y = count.values
        )],
        'layout': go.Layout(height=300, width=800,
            xaxis = {'title':'cluster id'},
            yaxis = {'title':'number of genes in each cluster'},
            margin={'l': 40, 'b': 40, 't': 40, 'r': 40},
            hovermode='closest'
        )
    }

#Display the PCA Projection 2D figure
@app.callback(
    Output('graph-pca-2d','figure'),
    [Input(component_id='k-value',component_property='value')])
def pca_projection(kvalue):
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
    df_clusterid = pd.DataFrame(labels_kmeans, index=data.index)
    df_clusterid.rename(columns={0: "cluster"}, inplace=True)
    df = pd.concat([df, df_clusterid], axis=1)

    features = list(df.columns)[:-1]  # delete the last one column:clusterid
    # Separating out the features
    x = df.loc[:, features].values
    # Separating out the target
    y = df.loc[:, ['cluster']].values
    # Standardizing the features
    x = StandardScaler().fit_transform(x)

    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(x)
    principalDf = pd.DataFrame(data=principalComponents
                               , columns=['PC1', 'PC2'])
    principalDf.index = df.index
    finalDf = pd.concat([principalDf, df[['cluster']]], axis=1)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Principal Component 1', fontsize=15)
    ax.set_ylabel('Principal Component 2', fontsize=15)
    ax.set_title('2 component PCA', fontsize=20)

    cluster_id = list(finalDf['cluster'].unique())
    # colors = ['r', 'g', 'b']
    # for cluster_id, color in zip(targets,colors):
    for i in cluster_id:
        indicesToKeep = finalDf['cluster'] == i
        ax.scatter(finalDf.loc[indicesToKeep, 'PC1']
                   , finalDf.loc[indicesToKeep, 'PC2']
                   # , c = color
                   , s=50)
    ax.legend(cluster_id)
    ax.grid()

    return fig


#Display a single cluster
@app.callback(
    Output('graph-cluster-profile', 'figure'),
    [Input(component_id='k-value',component_property='value'),
     Input(component_id='cluster-id', component_property='value')]
)

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

    tracey = go.Scatter(
                x = list(range(len(df_clusters.columns) - 1)),
                y = y_mean.values[clusterid])

    tracey_lo = go.Scatter(
                x = list(range(len(df_clusters.columns) - 1)),
                y = y_low.values[clusterid])

    tracey_hi = go.Scatter(
            x=list(range(len(df_clusters.columns) - 1)),
            y=y_high.values[clusterid])

    return {'data':[tracey, tracey_lo, tracey_hi],
        'layout': go.Layout(height=300, width=800,
            title=title_str,
            #xaxis = {'title':'cluster id'},
            #yaxis = {'title':'number of genes in each cluster'},
            margin={'l': 40, 'b': 40, 't': 40, 'r': 40},
            hovermode='closest'
        )
    }
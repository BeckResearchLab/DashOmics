import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
from dash.dependencies import Input, Output

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly
from sklearn.decomposition import PCA

print(__file__)
from app import app

import sqlite3
import re


layout = html.Div([
        html.H3('Step 2 -- Cluster Profile: Clusters Overview'),

        html.P(''),

        html.Div([
            dcc.Link('Go to Home Page -- Step 0: Define Input Data', href='/'),
            html.P(''),
            dcc.Link('Go to Step 1 -- Model Evaluation: Elbow Method', href='/ModelEvaluation/ElbowMethod'),
            html.P(''),
            dcc.Link('Go to Step 1 -- Model Evaluation: Silhouette Analysis', href='/ModelEvaluation/SilhouetteAnalysis')
        ]),

        html.H4('Choose K Value'),
        dcc.Input(id='k-value', value= 15, type='number'),

        html.P(''),
    html.Div([
        dcc.Graph(id='graph-cluster-size'),
        html.P(''),
        dcc.Graph(id='graph-pca-2d'),
        html.P(''),
        dcc.Graph(id='graph-pca-variance')],
    style = {'padding': 10}),

    #Links
    html.P(''),
    html.Div([
        dcc.Link('Go to Step 2 -- Cluster Profile: Choose Cluster', href='/ClustersProfile/ChooseCluster'),
        html.P(''),
        dcc.Link('Go to Step 2 -- Cluster Profile: Choose Gene', href='/ClustersProfile/ChooseGene')
    ])
])


#Cluster Sizes Figure
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
        'layout': go.Layout(height=300, width=1000,
            xaxis = {'title':'Cluster Id'},
            yaxis = {'title':'Number of Genes in Each Cluster'},
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
    df_clusterid = pd.DataFrame(labels_kmeans, index=df.index)
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

    cluster_id = sorted(list(finalDf['cluster'].unique()))

    PC1_variance = pca.explained_variance_ratio_[0]
    PC2_variance = pca.explained_variance_ratio_[1]

    # Create the graph with subplots
    fig = plotly.tools.make_subplots(rows=1, cols=1, vertical_spacing=0.2)
    fig['layout']['margin'] = {
        'l': 40, 'r': 40, 'b': 40, 't': 40
    }

    for i in cluster_id:
        indicesToKeep = finalDf['cluster'] == i
        PCA_point = go.Scatter(x=finalDf.loc[indicesToKeep, 'PC1'],
                               y=finalDf.loc[indicesToKeep, 'PC2'],
                               name = 'cluster %s' % i,
                               showlegend=True,
                               mode='markers',
                               marker=dict(size=4)
                               )
        fig.append_trace(PCA_point, 1, 1)

    # add variance it explained
    fig['layout']['xaxis'].update(title='PC1, explained variance ratio: %s' % PC1_variance)
    fig['layout']['yaxis'].update(title='PC2, explained variance ratio: %s' % PC2_variance)

    fig['layout'].update(height=600, width=1000,
                         title='Principle Component Analysis 2-D Projection')

    return fig


#Display the PCA variance explained
@app.callback(
    Output('graph-pca-variance','figure'),
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

    pc_number = len(df.columns)
    kmeans = KMeans(n_clusters=kvalue, max_iter=300, random_state=4)
    kmeans.fit(X)

    labels_kmeans = kmeans.labels_
    df_clusterid = pd.DataFrame(labels_kmeans, index=df.index)
    df_clusterid.rename(columns={0: "cluster"}, inplace=True)
    df = pd.concat([df, df_clusterid], axis=1)

    features = list(df.columns)[:-1]  # delete the last one column:clusterid
    # Separating out the features
    x = df.loc[:, features].values
    # Separating out the target
    y = df.loc[:, ['cluster']].values
    # Standardizing the features
    x = StandardScaler().fit_transform(x)

    pca = PCA(n_components=pc_number)
    principalComponents = pca.fit_transform(x)
    #variance = pca.explained_variance_ratio_  # calculate variance ratios
    var = np.cumsum(np.round(pca.explained_variance_ratio_, decimals=3) * 100)

    single_variance = []
    for i in range(pc_number):
        variance = pca.explained_variance_ratio_[i] * 100
        single_variance.append(variance)

    trace1 = go.Scatter(
            x=list(range(pc_number)),
            y=var,
            mode='lines+markers',
            name='Total Variance Explained'
        )
    trace2 = go.Bar(
            x=list(range(pc_number)),
            y=single_variance,
            name='Variance Explained by Single Principle Component'
    )


    return {
        'data': [trace1,trace2],
        'layout': go.Layout(height=600, width=1000,
            xaxis={'title': 'Number of Principle Components'},
            yaxis={'title': '% Variance Explained'},
            title='Principle Component Analysis -- Total Variance Explained',
            margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
            legend=dict(orientation="h"),
            hovermode='closest'
        )
    }


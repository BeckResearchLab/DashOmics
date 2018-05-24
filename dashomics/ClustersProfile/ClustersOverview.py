import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
from dash.dependencies import Input, Output

import pandas as pd
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

        html.H4('Choose K Value'),
        dcc.Input(id='k-value', value= 15, type='number'),

        html.P(''),

    dcc.Graph(id='graph-cluster-size'),
    html.P(''),
    dcc.Graph(id='graph-pca-2d'),
    html.P(''),

    html.H4('Choose Cluster id to Display'),
    dcc.Input(id='cluster-id', value=0, type='number'),
    dcc.Graph(id='graph-cluster-profile'),

    #Links
    html.P(''),
    html.Div([
        dcc.Link('Go to Home Page -- Step 0: Define Input Data', href='/'),
        html.P(''),
        dcc.Link('Go to Step 1 -- Model Evaluation: Elbow Method', href='/ModelEvaluation/ElbowMethod'),
        html.P(''),
        dcc.Link('Go to Step 1 -- Model Evaluation: Silhouette Analysis', href='/ModelEvaluation/SilhouetteAnalysis'),
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

    #set sample name as x-ticks
    xlabels = list(df.columns)
    #print(xlabels)
    tracey = go.Scatter(
                x = list(range(len(df_clusters.columns) - 1)),
                y = y_mean.values[clusterid],
                name = 'the mean of gene expression level')

    tracey_lo = go.Scatter(
                x = list(range(len(df_clusters.columns) - 1)),
                y = y_low.values[clusterid],
                name='the minimum of gene expression level')

    tracey_hi = go.Scatter(
                x=list(range(len(df_clusters.columns) - 1)),
                y=y_high.values[clusterid],
                name='the maximum of gene expression level')
    """
    bandxaxis = dict(
        title="sample name",
        #range=[0, len(bands.kpoints)],
        showgrid=True,
        showline=True,
        #ticks="",
        showticklabels=True,
        tickangle=45,
        #mirror=True,
        #linewidth=2,
        ticktext=xlabels
        #tickvals=[i for i in range(len(xlabels))]
    ),
    bandyaxis = dict(
        title="expression level",
        #range=[emin, emax],
        showgrid=True,
        showline=True,
        #zeroline=True,
        #mirror="ticks",
        #ticks="inside",
        linewidth=2,
        tickwidth=2,
        zerolinewidth=2
    )
    """

    return {'data':[tracey, tracey_lo, tracey_hi],
        'layout': go.Layout(height=300, width=1000,
            title=title_str,
            titlefont=dict(family='Arial, sans-serif',size=18),
            xaxis=dict(
                    title="sample name",
                    showgrid=True,
                    showline=True,
                    showticklabels=True,
                    tickangle=45,
                    ticktext=xlabels),
            yaxis=dict(
                    title="expression level",
                    showgrid=True,
                    showline=True),
            showlegend=True,
            margin={'l': 40, 'b': 40, 't': 40, 'r': 40},
            hovermode='closest'
        )
    }

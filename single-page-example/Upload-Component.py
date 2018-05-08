from dash.dependencies import Input, Output
import datetime
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table_experiments as dt
import plotly.graph_objs as go
import plotly

import base64
import io

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist, pdist

### new
import sqlite3
###

#create sql_master table
con = sqlite3.connect("dashomics-data.db")
sql_master = pd.DataFrame({'name': ['example-1']})
sql_master.to_sql('sql_master', con, if_exists='append')
con.close()

#import sys
#import os
#sys.path.append(os.path.join(os.path.dirname(sys.path[0])))

print(__file__)


#df = pd.read_csv('./data/example-1.csv', index_col = ['locus_tag'])
app = dash.Dash()

app.layout = html.Div([
    html.Div([

        html.H5("Upload Files"),
        dcc.Upload(
            id='upload-data',
            children=html.Div([
                'Drag and Drop or ',
                html.A('Select Files')
            ]),
            style={
                'width': '100%',
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'margin': '10px'
            },
            multiple=False),

        html.Br(),
        html.H5("Updated Table"),
        html.Div(dt.DataTable(rows=[{}], id='table'))

    ]),

    html.H3('Model Evaluation：Elbow Method'),

    dcc.Input(id='k-range', value= 10, type='number'),
    dcc.Graph(id='graph-elbow_method'),
    html.Div(id='app-2-display-value'),


])

# Functions

# file upload function
def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))

    except Exception as e:
        print(e)
        return None

    return df,html.Div([
        html.H5(filename),
        html.Hr(),  # horizontal line
        # For debugging, display the raw contents provided by the web browser
        html.Div('Raw Content Upload Successfully')
    ])


# callback table creation
@app.callback(Output('table', 'rows'),
              [Input('upload-data', 'contents'),
               Input('upload-data', 'filename')])
def update_output(contents, filename):
    if contents is not None:
        ### add df to sqlite
        con = sqlite3.connect("dashomics-data.db")
        df = parse_contents(contents, filename)[0]
        if df is not None:
            df.to_sql(filename, con, if_exists="replace")
            dff = pd.read_sql_query('SELECT * FROM "example-1.csv"', con)
            print(dff)
            con.close()
            ###
            return dff.to_dict('records')
        ###
        else:
            return [{}]
    else:
        return [{}]


@app.callback(
    Output('graph-elbow_method', 'figure'),
    [Input(component_id='k-range',component_property='value'),
     Input('table', 'rows')]
)
def elbow_method_evaluation(n, tablerows):
    """
    n: the maximum of k value

    """
    # Fit the kmeans model for k in a certain range

    dff = pd.DataFrame(tablerows).set_index('locus_tag')
    #print(dff)

    K = range(1, n + 1)
    KM = [KMeans(n_clusters=k).fit(dff) for k in K]
    # Pull out the cluster centroid for each model
    centroids = [k.cluster_centers_ for k in KM]

    # Calculate the distance between each data point and the centroid of its cluster
    k_euclid = [cdist(dff.values, cent, 'euclidean') for cent in centroids]
    dist = [np.min(ke, axis=1) for ke in k_euclid]

    # Total within sum of square
    wss = [sum(d ** 2) / 1000 for d in dist]
    # The total sum of square
    tss = sum(pdist(dff.values) ** 2) / dff.values.shape[0]
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

if __name__ == '__main__':
    app.run_server(debug=True)


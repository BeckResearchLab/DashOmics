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

import sqlite3

#create sql_master table
con = sqlite3.connect("dbtest-1.db")
c = con.cursor()
master_data = [['example_1','No'],['example_2_cancer','No']]
sql_master = pd.DataFrame(master_data, columns = ['Filename','Choose_or_Not'])
sql_master.to_sql('sql_master', con, if_exists='replace')

#add example data into sqlite
example_1 = pd.read_csv('../data/example-1.csv', index_col = ['id'])
example_1.to_sql('example_1', con, if_exists="replace")

example_2 = pd.read_csv('../data/example-2-cancer.csv', index_col = ['id'])
example_2.to_sql('example_2_cancer', con, if_exists="replace")
con.close()

#import sys
#import os
#sys.path.append(os.path.join(os.path.dirname(sys.path[0])))

print(__file__)

app = dash.Dash()

app.layout = html.Div([
    html.Div([

        html.H3("Choosing Example Data to Explore DashOmics"),
        dcc.Dropdown(
            id='example-data',
            options = [
                {'label':'example_1', 'value':'example_1'},
                {'label':'example_2', 'value':'example_2_cancer'}
            ]),

        html.H3("Or Upload Your Own Files"),
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
        html.H4("Updated Table"),
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


# update sqlite database and display in layout DataTable
@app.callback(Output('table', 'rows'),
              [Input('upload-data', 'contents'),
               Input('upload-data', 'filename'),
               Input('example-data','value')])

def update_database(upload_contents, upload_filename, example_filename):
    if upload_contents is not None:
        # add uploaded df to sqlite
        con = sqlite3.connect("dbtest-1.db")
        c = con.cursor()

        #use parse_contents to get df
        df = parse_contents(upload_contents, upload_filename)[0]
        if df is not None:
            df.to_sql(upload_filename, con, if_exists="replace")
            # add upload data filename in sql_master table
            c.execute('''INSERT INTO sql_master(Filename, Choose_or_Not) 
                         VALUES ('%s', 'Yes')
                      ''' % upload_filename)
            con.commit()
            con.close()
            #display table in layout
            return df.to_dict('records')
        else:
            return [{}]

    if example_filename is not None:
        con = sqlite3.connect("dbtest-1.db")
        c = con.cursor()
        df = pd.read_sql_query('SELECT * FROM %s' % str(example_filename).split('.')[0], con)
        if df is not None:
            # update "Choose or Not" status to "Yes" in sql_master table
            c.execute('''UPDATE sql_master
                         SET Choose_or_Not = 'Yes'
                         WHERE Filename = '%s'
                      ''' % str(example_filename).split('.')[0])
            con.commit()
            con.close()
            return df.to_dict('records')
        else:
            return [{}]
    if (upload_contents is not None) & (example_filename is not 'Choose an example data'):
        raise ValueError('Upload data conflicts with Example data')
    else:
        return [{}]

# apply elbow method analysis
@app.callback(
    Output('graph-elbow_method', 'figure'),
    [Input(component_id='k-range',component_property='value'),
     Input('example-data', 'value')]
)

def elbow_method_evaluation(n, filename):
    """
    n: the maximum of k value
    """
    # Fit the kmeans model for k in a certain range

    # read dataframe from sqlite database
    con = sqlite3.connect("dbtest-1.db")
    if filename is None:
        print('No Input Yet!')
        return
    if filename is not None:
        # make sure file extension is not in sqlite
        dff = pd.read_sql_query('SELECT * FROM %s' % str(filename).split('.')[0], con).set_index(['id'])
        con.close()

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


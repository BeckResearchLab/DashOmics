import base64
import io

from dash.dependencies import Input, Output
import dash_html_components as html
import dash_core_components as dcc
import dash_table_experiments as dt

from app import app
import pandas as pd

import sqlite3

#create sql_master table
con = sqlite3.connect("dashomics_test.db")
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

print('homepage.py -- create sqlite database successfully')

#app = dash.Dash()
##Don't create a new dash app object, use the original one from app.py
app.config.supress_callback_exceptions = True
app.scripts.config.serve_locally = True

layout = html.Div(children=[

    html.H1(children='DashOmics'),

    html.Div(children='''
        DashOmics is a visualization tool to explore *omics data using clustering analysis. 
        It is created by Dash Plot.ly, a Python framework for building interactive analytical tools.
        Users can play with existing example data, or upload their own data in SQLite database. 
        K-Means clustering method would be applied on RNA-seq data, including two model evaluation methods — elbow method and silhouette analysis, 
        to help find the optimal k value (the number of clusters). 
        Users can explore cluster profiles of grouped genes based on specific k value and generate insights into gene functions and networks.
    '''),
    html.P(''),
    html.H3(children='Step 0: Define Input Data'),
    html.Div([
        html.H4("Choosing Example Data to Explore DashOmics"),
        dcc.Dropdown(
            id='example-data',
            options = [
                {'label':'example_1', 'value':'example_1'},
                {'label':'example_2', 'value':'example_2_cancer'}
            ]),

        html.H4("Or Upload Your Own Files"),
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
        html.Div(id='data-filename'),

        dt.DataTable(
            rows=[{}],
            editable=True, id='input-data-table'
        )
    ]),

    #Links
    html.Div([
        dcc.Link('Go to Step 1 -- Model Evaluation: Elbow Method', href='/ModelEvaluation/ElbowMethod'),
        html.P(''),
        dcc.Link('Go to Step 1 -- Model Evaluation: Silhouette Analysis', href='/ModelEvaluation/SilhouetteAnalysis'),
        html.P(''),
        dcc.Link('Go to Step 2 -- Cluster Profile: Clusters Overview', href='/ClustersProfile/ClustersOverview'),
        html.P(''),
        dcc.Link('Go to Step 2 -- Cluster Profile: Choose Cluster', href='/ClustersProfile/ChooseCluster'),
        html.P(''),
        dcc.Link('Go to Step 2 -- Cluster Profile: Choose Gene', href='/ClustersProfile/ChooseGene')
    ])
])

print('homepage.py -- create layout successfully')

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
        return html.Div([
            'There was an error processing this file.'
        ])

    return df, html.Div([
        html.H5(filename),
        html.Hr(),  # horizontal line
        # For debugging, display the raw contents provided by the web browser
        print('Raw Content Upload Successfully')

        # Use the DataTable prototype component:
        # github.com/plotly/dash-table-experiments

        # DataTable cannot be crawled from layout
        #dt.DataTable(rows=df.to_dict('records')),
    ])



# update sqlite database
# and display in layout DataTable
#@app.callback(Output('data-filename','children'),
@app.callback(Output('input-data-table', 'rows'),
              [Input('upload-data', 'contents'),
               Input('upload-data', 'filename'),
               Input('example-data','value')])

#create a table in db to store what users choose at homepage
def update_database(upload_contents, upload_filename, example_filename):
    print('homepage.py -- Call update_database function')
    if (upload_filename is None) & (example_filename is None):
        print('No input data')
        return
    #display upload data
    if (upload_contents is not None) & (example_filename is None):
        # add uploaded df to sqlite
        con = sqlite3.connect("dashomics_test.db")
        c = con.cursor()
        df = parse_contents(upload_contents, upload_filename)[0]
        if df is not None:
            #add df into sqlite database as table
            df.to_sql(upload_filename, con, if_exists="replace")
            #add upload data filename in sql_master table
            c.execute('''INSERT INTO sql_master(Filename, Choose_or_Not) 
                         VALUES ('%s', 'Yes')
                      ''' % upload_filename)
            con.commit()
            con.close()
            #display table in layout
            print('homepage -- add upload data successfully')
            return df.to_dict('records')
        else:
            con.close()
            return [{}]

    #display example data
    if (upload_contents is None) & (example_filename is not None):
        con = sqlite3.connect("dashomics_test.db")
        c = con.cursor()
        df = pd.read_sql_query('SELECT * FROM %s' % str(example_filename).split('.')[0], con)
        if df is not None:
            #update "Choose or Not" status to "Yes" in sql_master table
            c.execute('''UPDATE sql_master
                         SET Choose_or_Not = 'Yes'
                         WHERE Filename = '%s'
                      ''' % str(example_filename).split('.')[0])
            con.commit()
            con.close()
            print('homepage -- choose example file successfully')
            return df.to_dict('records')
        else:
            con.close()
            return [{}]

    if (upload_contents is not None) & (example_filename is not None):
        raise ValueError('Upload data conflicts with Example data')
    else:
        return [{}]

print('homepage.py -- running successfully')

app.css.append_css({
    'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'
})



#if __name__ == '__main__':
#    app.run_server(debug=True)
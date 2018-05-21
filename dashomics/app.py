import dash

app = dash.Dash()
server = app.server
app.config.supress_callback_exceptions = True

print('app.py -- running successfully')

app.css.append_css({
    "external_url": "https://codepen.io/chriddyp/pen/bWLwgP.css"
})
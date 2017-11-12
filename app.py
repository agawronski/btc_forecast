#-------------------------------------------------------------------------------
from flask import render_template
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
from flask import Flask
import pandas as pd
import pyflux as pf
import quandl
import base64
import os
import io

from contextlib import redirect_stdout

app = Flask(__name__)

@app.route('/')
def build_plot():


    data = quandl.get('BCHARTS/BITSTAMPUSD', authtoken = 'oTBhHqG_L_3PHuxpLEBW')
    data = data.fillna(method = 'ffill')
    model = pf.ARIMA(data = data, ar = 5, integ = 1, ma = 4, target = 'Close', family = pf.Normal())
    x = model.fit("MLE")

    # model_summary = 'string'
    f = io.StringIO()
    with redirect_stdout(f):
        x.summary()
    model_summary = f.getvalue()
    model_summary2 = model_summary.split('\n')

    img = io.BytesIO()
    plt.plot(data.index, data.Close)
    plt.savefig(img, format = 'png')
    plot_url = base64.b64encode(img.getvalue()).decode()

    img2 = io.BytesIO()
    model.plot_z(figsize = (8,5))
    plt.savefig(img2, format = 'png')
    plot_urls2 = base64.b64encode(img2.getvalue()).decode()

    img3 = io.BytesIO()
    model.plot_fit(figsize = (8,5))
    plt.savefig(img3, format = 'png')
    plot_urls3 = base64.b64encode(img3.getvalue()).decode()

    img4 = io.BytesIO()
    model.plot_predict_is(h = 50, figsize = (8,5))
    plt.savefig(img4, format = 'png')
    plot_urls4 = base64.b64encode(img4.getvalue()).decode()

    # img5 = io.BytesIO()
    # model.plot_predict(h = 20, past_values = 20, figsize = (8,5))
    # plt.savefig(img5, format = 'png')
    # plot_urls5 = base64.b64encode(img5.getvalue()).decode()

    pred = model.predict(h = 30).reset_index()
    pred['predictions'] = None
    pred['predictions'][0] = data.Close.tail(1) + pred.loc[0, 'Differenced Close']
    for row in range(1, pred.shape[0]):
        last_val = pred.loc[pred.index == (row - 1), 'predictions'].values
        current_diff = pred.loc[pred.index == row, 'Differenced Close'].values
        current_val = last_val + current_diff
        pred.loc[pred.index == row, 'predictions'] = last_val + current_diff

    pred.predictions = pred.predictions.astype(float)

    plt.cla()
    img5 = io.BytesIO()
    plt.plot(pred.Date.astype(str), pred.predictions)
    plt.xticks(rotation = 90)
    plt.tight_layout()
    plt.savefig(img5, format = 'png')
    plot_urls5 = base64.b64encode(img5.getvalue()).decode()

    return render_template('main_template.html',
                            plot_url = plot_url,
                            model_summary2 = model_summary2,
                            plot_urls2 = plot_urls2,
                            plot_urls3 = plot_urls3,
                            plot_urls4 = plot_urls4,
                            plot_urls5 = plot_urls5)



if __name__ == '__main__':
    app.debug = True
    app.run()

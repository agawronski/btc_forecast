#-------------------------------------------------------------------------------
from flask import render_template
import matplotlib
matplotlib.use("agg")
from contextlib import redirect_stdout
import matplotlib.pyplot as plt
from flask import Flask
import pandas as pd
import pyflux as pf
import numpy as np
import quandl
import base64
import os
import io

# import random
# from io import BytesIO
# # from StringIO import StringIO  # python 2.7x
#
# from flask import Flask, make_response
# from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
# from matplotlib.figure import Figure




app = Flask(__name__)

@app.route('/')
def build_plot():
    data = quandl.get('BCHARTS/BITSTAMPUSD', authtoken = 'oTBhHqG_L_3PHuxpLEBW')
    data = data.fillna(method = 'ffill')
    data['Close2'] = np.log(data.Close + 0.0001)
    model = pf.ARIMA(data = data, ar = 5, integ = 1, ma = 5, target = 'Close2', family = pf.Normal())
    x = model.fit("MLE")

    f = io.StringIO()
    with redirect_stdout(f):
        x.summary()
    model_summary = f.getvalue()
    model_summary2 = model_summary.split('\n')

    img = io.BytesIO()
    dates = matplotlib.dates.date2num(list(data.index))
    plt.plot_date(dates, data.Close, '-')
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
    pred['predictions'][0] = data.Close2.tail(1) + pred.loc[0, 'Differenced Close2']
    for row in range(1, pred.shape[0]):
        last_val = pred.loc[pred.index == (row - 1), 'predictions'].values
        current_diff = pred.loc[pred.index == row, 'Differenced Close2'].values
        current_val = last_val + current_diff
        pred.loc[pred.index == row, 'predictions'] = last_val + current_diff

    pred.predictions = pred.predictions.astype(float)
    pred['predictions_final'] = np.exp(pred['predictions']) - 0.0001

    plt.cla()
    img5 = io.BytesIO()
    dates2 = matplotlib.dates.date2num(list(pred.Date))
    plt.plot_date(dates2, pred.predictions_final, '-')
    # plt.xticks(rotation = 90)
    # plt.tight_layout()
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

#-------------------------------------------------------------------------------
from flask import render_template
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
from flask import Flask
import pandas as pd
# import pyflux as pf
# import quandl
import base64
import os
import io

app = Flask(__name__)

@app.route('/')
def build_plot():

    img = io.BytesIO()
    # data = quandl.get('LBMA/SILVER', authtoken = 'oTBhHqG_L_3PHuxpLEBW')
    # data = data.fillna(method = 'ffill')
    # model = pf.ARIMA(data = data, ar = 5, integ = 1, ma = 4, target = 'USD', family = pf.Normal())
    # x = model.fit("MLE")
    # model_summary = base64.b64encode( x.summary() ).decode()

    # plot1 = model.plot_z(figsize = (15,5))
    y = [1,2,3,4,5]
    x = [0,2,1,3,4]
    plt.plot(x,y)

    plt.savefig(img, format='png')
    # plot1.savefig(img, format='png')
    img.seek(0)

    plot_url = base64.b64encode(img.getvalue()).decode()
    return render_template('main_template.html',
                            plot_url = plot_url)





if __name__ == '__main__':
    app.debug = True
    app.run()

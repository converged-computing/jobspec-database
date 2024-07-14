import numpy as np
import pandas as pd
import seaborn as sn
import tensorflow as tf
from tensorflow import keras
import statistics
import scipy
from scipy import integrate
from scipy import signal
from scipy.stats import norm,lognorm
from scipy.fft import rfft, rfftfreq
#import obspy.signal
#from obspy.signal.tf_misfit import plot_tf_gofs, eg, pg
import itertools
import matplotlib
from matplotlib import pyplot as plt
from sklearn.utils import shuffle
from RepGAN_model import RepGAN
import RepGAN_losses
#from tensorflow.keras.optimizers import Adam, RMSprop

from matplotlib.pyplot import *
from matplotlib import cm
#from colorspacious import cspace_converter
from collections import OrderedDict
cmaps = OrderedDict()
cmaps['Qualitative'] = ['Pastel1', 'Pastel2', 'Paired', 'Accent','Dark2', 'Set1', 'Set2', 'Set3',
                        'tab10', 'tab20', 'tab20b', 'tab20c']
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import multilabel_confusion_matrix
from statistics import mean
# import plotly.graph_objects as go
# import plotly.express as px

from numpy.lib.type_check import imag
import sklearn
from PIL import Image
import io
import numpy as np

# from bokeh.layouts import layout
# from bokeh.plotting import figure
# from bokeh.models import CustomJS, Slider, ColumnDataSource
# from bokeh.resources import CDN
# from bokeh.embed import file_html
# from bokeh.io import curdoc,output_file, show
# import bokeh
# from bokeh.models import Text, Label
# import panel as pn
#pn.extension()


#from interferometry import *
import MDOFload as mdof
from RepGAN_utils import *
#from interferometry_utils import *
from sklearn.manifold import TSNE

from random import seed
from random import randint

import matplotlib.font_manager
rcParams['font.family'] = 'serif'
rcParams['font.sans-serif'] = ['Helvetica']
#families = ['serif', 'sans-serif', 'cursive', 'fantasy', 'monospace']
#rcParams['text.usetex'] = True

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

from mpl_toolkits import mplot3d

from sklearn import cluster, datasets, mixture
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice


def correlation_lags(in1_len, in2_len, mode='full'):
    r"""
    Calculates the lag / displacement indices array for 1D cross-correlation.
    Parameters
    ----------
    in1_size : int
        First input size.
    in2_size : int
        Second input size.
    mode : str {'full', 'valid', 'same'}, optional
        A string indicating the size of the output.
        See the documentation `correlate` for more information.
    See Also
    --------
    correlate : Compute the N-dimensional cross-correlation.
    Returns
    -------
    lags : array
        Returns an array containing cross-correlation lag/displacement indices.
        Indices can be indexed with the np.argmax of the correlation to return
        the lag/displacement.
    Notes
    -----
    Cross-correlation for continuous functions :math:`f` and :math:`g` is
    defined as:
    .. math::
        \left ( f\star g \right )\left ( \tau \right )
        \triangleq \int_{t_0}^{t_0 +T}
        \overline{f\left ( t \right )}g\left ( t+\tau \right )dt
    Where :math:`\tau` is defined as the displacement, also known as the lag.
    Cross correlation for discrete functions :math:`f` and :math:`g` is
    defined as:
    .. math::
        \left ( f\star g \right )\left [ n \right ]
        \triangleq \sum_{-\infty}^{\infty}
        \overline{f\left [ m \right ]}g\left [ m+n \right ]
    Where :math:`n` is the lag.
    Examples
    --------
    Cross-correlation of a signal with its time-delayed self.
    >>> from scipy import signal
    >>> from numpy.random import default_rng
    >>> rng = default_rng()
    >>> x = rng.standard_normal(1000)
    >>> y = np.concatenate([rng.standard_normal(100), x])
    >>> correlation = signal.correlate(x, y, mode="full")
    >>> lags = signal.correlation_lags(x.size, y.size, mode="full")
    >>> lag = lags[np.argmax(correlation)]
    """

    # calculate lag ranges in different modes of operation
    if mode == "full":
        # the output is the full discrete linear convolution
        # of the inputs. (Default)
        lags = np.arange(-in2_len + 1, in1_len)
    elif mode == "same":
        # the output is the same size as `in1`, centered
        # with respect to the 'full' output.
        # calculate the full output
        lags = np.arange(-in2_len + 1, in1_len)
        # determine the midpoint in the full output
        mid = lags.size // 2
        # determine lag_bound to be used with respect
        # to the midpoint
        lag_bound = in1_len // 2
        # calculate lag ranges for even and odd scenarios
        if in1_len % 2 == 0:
            lags = lags[(mid-lag_bound):(mid+lag_bound)]
        else:
            lags = lags[(mid-lag_bound):(mid+lag_bound)+1]
    elif mode == "valid":
        # the output consists only of those elements that do not
        # rely on the zero-padding. In 'valid' mode, either `in1` or `in2`
        # must be at least as large as the other in every dimension.

        # the lag_bound will be either negative or positive
        # this let's us infer how to present the lag range
        lag_bound = in1_len - in2_len
        if lag_bound >= 0:
            lags = np.arange(lag_bound + 1)
        else:
            lags = np.arange(lag_bound, 1)
    return lags

def arias_intensity(dtm,tha,pc=0.95,nf=9.81):
    aid = np.pi/2./nf*scipy.integrate.cumtrapz(tha**2, dx=dtm, axis=-1, initial = 0.)
    mai = np.max(aid,axis=-1)
    ait = np.empty_like(mai)
    idx = np.empty_like(mai)
    if mai.size>1:
        for i in range(mai.size):
            ths = np.where(aid[i,...]/mai[i]>=pc)[0][0]
            ait[i] = aid[i,ths]
            idx[i] = ths*dtm
    else:
        ths = np.where(aid/mai>=pc)[0][0]
        ait = aid[ths]
        idx = ths*dtm
    return aid,ait,idx


def PlotReconstructedTHs(model,realXC,results_dir):
    # Plot reconstructed time-histories
    realX = np.concatenate([x for x, c, m, d, y in realXC], axis=0)
    realC = np.concatenate([c for x, c, m, d, y in realXC], axis=0)
    mag = np.concatenate([m for x, c, m, d, y in realXC], axis=0)
    di = np.concatenate([d for x, c, m, d, y in realXC], axis=0)
    y = np.concatenate([y for x, c, m, d, y in realXC], axis=0)

    recX,fakeC,fakeS,fakeN,fakeX = model.plot(realX,realC)
    print(recX.shape)
    y_pred = model.pred(realX)
    print(y_pred.shape)
    t = np.zeros(realX.shape[1])
    for k in range(realX.shape[1]-1):
        t[k+1] = (k+1)*0.04

    recX_fft = tf.make_ndarray(tf.make_tensor_proto(recX))

    # Print real vs reconstructed signal
    for j in range(realX.shape[2]):
        for k in range(5):
            #i = randint(0, realX.shape[0]-1)
            i=k*30
            hfg = plt.figure(figsize=(12,6),tight_layout=True)
            hax = hfg.add_subplot(111)
            hax.plot(t,realX[i,:,j], color='black')
            hax.plot(t,recX[i,:,j], color='orange',linestyle="--")
            #compute mse
            mse = np.square(np.subtract(realX[i,:,j], recX[i,:,j])).mean()
            zero = np.zeros_like(realX[i,:,j])
            mse0 = np.square(np.subtract(realX[i,:,j], zero)).mean()
            hax.set_title(str(mse)+" "+str(mse0), fontsize=22,fontweight='bold')
            #hax.set_title(r'$X \hspace{0.5} reconstruction$', fontsize=22,fontweight='bold')
            hax.set_ylabel(r'$X(t) \hspace{0.5} [1]$', fontsize=26,fontweight='bold')
            hax.set_xlabel(r'$t \hspace{0.5} [s]$', fontsize=26,fontweight='bold')
            hax.legend([r'$X$', r"$G_z(F_x(x))$"], loc='best',frameon=False,fontsize=20)
            hax.set_ylim([-1.0, 1.0])
            hax.tick_params(axis='both', labelsize=18)
            plt.savefig(results_dir + '/reconstruction_{:>d}_{:>d}.png'.format(j,i),bbox_inches = 'tight')
            #plt.savefig(results_dir + '/reconstruction_{:>d}_{:>d}.eps'.format(j,i),bbox_inches = 'tight',dpi=200)
            plt.close()

            #plot predicted
            hfg = plt.figure(figsize=(12,6),tight_layout=True)
            hax = hfg.add_subplot(111)
            hax.plot(t,y[i,:,j], color='black')
            hax.plot(t,y_pred[i,:,j], color='red',linestyle="--")
            hax.plot(t[:-32],y_pred[i,:-32,j], color='orange',linestyle="--")
            #hax.set_title(r'$X \hspace{0.5} reconstruction$', fontsize=22,fontweight='bold')
            hax.set_ylabel(r'$X(t) \hspace{0.5} [1]$', fontsize=26,fontweight='bold')
            hax.set_xlabel(r'$t \hspace{0.5} [s]$', fontsize=26,fontweight='bold')
            hax.legend([r'$y$', r"$y_{pred}$"], loc='best',frameon=False,fontsize=20)
            hax.set_ylim([-1.0, 1.0])
            hax.tick_params(axis='both', labelsize=18)
            plt.savefig(results_dir + '/prediction_{:>d}_{:>d}.png'.format(j,i),bbox_inches = 'tight')
            #plt.savefig(results_dir + '/reconstruction_{:>d}_{:>d}.eps'.format(j,i),bbox_inches = 'tight',dpi=200)
            plt.close()
            

            hfg = plt.figure(figsize=(12,6),tight_layout=True)
            hax = hfg.add_subplot(111)
            N = realX.shape[1]
            SAMPLE_RATE = 25
            yf_real = rfft(realX[i,:,j])
            xf_real = rfftfreq(N, 1 / SAMPLE_RATE)
            hax.plot(xf_real, np.abs(yf_real), color='black')
            yf_rec = rfft(recX_fft[i,:,j])
            xf_rec = rfftfreq(N, 1 / SAMPLE_RATE)
            hax.plot(xf_rec, np.abs(yf_rec), color='orange',linestyle="--")
            hax.set_ylabel(r'$Amplitude \hspace{0.5} [1]$', fontsize=26,fontweight='bold')
            hax.set_xlabel(r'$Frequency \hspace{0.5} [Hz]$', fontsize=26,fontweight='bold')
            hax.legend([r'$X$', r"$G_z(F_x(x))$"], loc='best',frameon=False,fontsize=20)
            hax.tick_params(axis='both', labelsize=18)
            plt.savefig(results_dir + '/fft_{:>d}_{:>d}.png'.format(j,i),bbox_inches = 'tight')
            #plt.savefig(results_dir + '/fft_{:>d}_{:>d}.eps'.format(j,i),bbox_inches = 'tight',dpi=200)
            plt.close()



def cross_2d_dam(und,dam,i0,dt,nw,kspec,fmin,fmax,tmin,tmax):
    
    fnyq = 0.5/dt
    wn   = [fmin/fnyq,fmax/fnyq]
    b, a = butter(6, wn,'bandpass',output='ba')

    ntr = und.shape[1] 
    x   = und[:,i0]
    for i in range(ntr):
        y = dam[:,i]
        Pxy  = MTCross(y,x,nw,kspec,dt,iadapt=2,wl=0.0)
        xcorr, dcohe, dconv  = Pxy.mt_corr()
        dconv = filtfilt(b, a, dcohe[:,0])
        if (i==0):
            k    = np.linspace(-Pxy.npts,Pxy.npts,len(xcorr),dtype=int)
            t2   = k*dt
            tloc = np.where((t2>=tmin) & (t2<=tmax))[0]
            irf  = np.zeros((len(tloc),ntr))
        irf[:,i] = dconv[tloc]
        t        = t2[tloc]
    
    return [irf,t]


def PlotSwitchedTHs(model,real_u,real_d,d,results_dir):
    # Plot reconstructed time-histories
    
    realX_u = np.concatenate([x for x, c, m, d in real_u], axis=0)
    realC_u = np.concatenate([c for x, c, m, d in real_u], axis=0)
    mag_u = np.concatenate([x for x, c, m, d in real_u], axis=0)
    di_u = np.concatenate([c for x, c, m, d in real_u], axis=0)

    recX_u,_,_,_ = model.predict(realX_u)

    realX_d = np.concatenate([x for x, c, m, d in real_d], axis=0)
    realC_d = np.concatenate([c for x, c, m, d in real_d], axis=0)
    mag_d = np.concatenate([x for x, c, m, d in real_d], axis=0)
    di_d = np.concatenate([c for x, c, m, d in real_d], axis=0)

    recX_d,_,_,_ = model.predict(realX_d)

    t = np.zeros(realX_u.shape[1])
    for m in range(realX_u.shape[1]-1):
        t[m+1] = (m+1)*0.04

    if d==1:

        recX_fft = tf.make_ndarray(tf.make_tensor_proto(recX_u))

        # Print real vs reconstructed signal
        for j in range(realX_u.shape[2]):
            for k in range(10):
                i = randint(0, realX_u.shape[0]-1)
                hfg = plt.figure(figsize=(12,6),tight_layout=True)
                hax = hfg.add_subplot(111)
                hax.plot(t,realX_u[i,:,j], color='black')
                hax.plot(t,recX_u[i,:,j], color='orange',linestyle="--")
                #hax.set_title(r'$X \hspace{0.5} reconstruction$', fontsize=22,fontweight='bold')
                hax.set_ylabel(r'$X(t) \hspace{0.5} [1]$', fontsize=26,fontweight='bold')
                hax.set_xlabel(r'$t \hspace{0.5} [s]$', fontsize=26,fontweight='bold')
                hax.legend([r'$X$', r"$G_z(F_x(x))$"], loc='best',frameon=False,fontsize=20)
                hax.set_ylim([-1.0, 1.0])
                hax.tick_params(axis='both', labelsize=18)
                plt.savefig(results_dir + '/reconstruction0_{:>d}_{:>d}.png'.format(j,i),bbox_inches = 'tight')
                #plt.savefig(results_dir + '/reconstruction0_{:>d}_{:>d}.eps'.format(j,i),bbox_inches = 'tight',dpi=200)
                plt.close()


                hfg = plt.figure(figsize=(12,6),tight_layout=True)
                hax = hfg.add_subplot(111)
                N = realX_u.shape[1]
                SAMPLE_RATE = 25
                yf_real = rfft(realX_u[i,:,j])
                xf_real = rfftfreq(N, 1 / SAMPLE_RATE)
                hax.plot(xf_real, np.abs(yf_real), color='black')
                yf_rec = rfft(recX_fft[i,:,j])
                xf_rec = rfftfreq(N, 1 / SAMPLE_RATE)
                hax.plot(xf_rec, np.abs(yf_rec), color='orange',linestyle="--")
                hax.set_ylabel(r'$Amplitude \hspace{0.5} [1]$', fontsize=26,fontweight='bold')
                hax.set_xlabel(r'$Frequency \hspace{0.5} [Hz]$', fontsize=26,fontweight='bold')
                hax.legend([r'$X$', r"$G_z(F_x(x))$"], loc='best',frameon=False,fontsize=20)
                hax.tick_params(axis='both', labelsize=18)
                plt.savefig(results_dir + '/fft0_{:>d}_{:>d}.png'.format(j,i),bbox_inches = 'tight')
                #plt.savefig(results_dir + '/fft0_{:>d}_{:>d}.eps'.format(j,i),bbox_inches = 'tight',dpi=200)
                plt.close()

    recX_fft = tf.make_ndarray(tf.make_tensor_proto(recX_d))

    # Print real vs reconstructed signal
    for j in range(realX_d.shape[2]):
        for k in range(10):
            i = randint(0, realX_d.shape[0]-1)
            hfg = plt.figure(figsize=(12,6),tight_layout=True)
            hax = hfg.add_subplot(111)
            hax.plot(t,realX_d[i,:,j], color='black')
            hax.plot(t,recX_d[i,:,j], color='orange',linestyle="--")
            #hax.set_title(r'$X \hspace{0.5} reconstruction$', fontsize=22,fontweight='bold')
            hax.set_ylabel(r'$X(t) \hspace{0.5} [1]$', fontsize=26,fontweight='bold')
            hax.set_xlabel(r'$t \hspace{0.5} [s]$', fontsize=26,fontweight='bold')
            hax.legend([r'$X$', r"$G_z(F_x(x))$"], loc='best',frameon=False,fontsize=20)
            hax.set_ylim([-1.0, 1.0])
            hax.tick_params(axis='both', labelsize=18)
            plt.savefig(results_dir + '/reconstruction{:>d}_{:>d}_{:>d}.png'.format(d,j,i),bbox_inches = 'tight')
            #plt.savefig(results_dir + '/reconstruction{:>d}_{:>d}_{:>d}.eps'.format(d,j,i),bbox_inches = 'tight',dpi=200)
            plt.close()


            hfg = plt.figure(figsize=(12,6),tight_layout=True)
            hax = hfg.add_subplot(111)
            N = realX_d.shape[1]
            SAMPLE_RATE = 25
            yf_real = rfft(realX_d[i,:,j])
            xf_real = rfftfreq(N, 1 / SAMPLE_RATE)
            hax.plot(xf_real, np.abs(yf_real), color='black')
            yf_rec = rfft(recX_fft[i,:,j])
            xf_rec = rfftfreq(N, 1 / SAMPLE_RATE)
            hax.plot(xf_rec, np.abs(yf_rec), color='orange',linestyle="--")
            hax.set_ylabel(r'$Amplitude \hspace{0.5} [1]$', fontsize=26,fontweight='bold')
            hax.set_xlabel(r'$Frequency \hspace{0.5} [Hz]$', fontsize=26,fontweight='bold')
            hax.legend([r'$X$', r"$G_z(F_x(x))$"], loc='best',frameon=False,fontsize=20)
            hax.tick_params(axis='both', labelsize=18)
            plt.savefig(results_dir + '/fft{:>d}_{:>d}_{:>d}.png'.format(d,j,i),bbox_inches = 'tight')
            #plt.savefig(results_dir + '/fft{:>d}_{:>d}_{:>d}.eps'.format(d,j,i),bbox_inches = 'tight',dpi=200)
            plt.close()

        
    fakeC_new = np.zeros_like((realC_d))
    fakeC_new[:,d] = 1.0
    fakeX_new = model.generate(realX_u,fakeC_new)
    fakeX_new_fft = tf.make_ndarray(tf.make_tensor_proto(fakeX_new))

    corr_real = np.zeros((realX_u.shape[0],realX_u.shape[1]*2-1,realX_u.shape[2]))
    corr_switch = np.zeros((realX_u.shape[0],realX_u.shape[1]*2-1,realX_u.shape[2]))
    lags_real = np.zeros((realX_u.shape[0],realX_u.shape[1]*2-1,realX_u.shape[2]))
    lags_switch = np.zeros((realX_u.shape[0],realX_u.shape[1]*2-1,realX_u.shape[2]))

    for j in range(realX_u.shape[0]):
        for i in range(realX_u.shape[2]):
            corr_real[j,:,i] = signal.correlate(realX_d[j,:,i], realX_u[j,:,i])
            lags_real[j,:,i] = correlation_lags(len(realX_d[j,:,i]), len(realX_u[j,:,i]))
            corr_real[j,:,i] /= np.max(corr_real[j,:,i])

            corr_switch[j,:,i] = signal.correlate(fakeX_new[j,:,i], realX_u[j,:,i])
            lags_switch[j,:,i] = correlation_lags(len(fakeX_new[j,:,i]), len(realX_u[j,:,i]))
            corr_switch[j,:,i] /= np.max(corr_switch[j,:,i])

    
    t = np.zeros(realX_u.shape[1])
    for m in range(realX_u.shape[1]-1):
        t[m+1] = (m+1)*0.04

    for j in range(realX_u.shape[2]):
        for k in range(10):
            i = randint(0, realX_u.shape[0]-1)
            fig, (ax0, ax1, ax2) = plt.subplots(3, 1,figsize=(12,18))
            ax0.plot(t,realX_u[i,:,j], color='green')
            ax1.plot(t,realX_d[i,:,j], color='black')
            ax2.plot(t,fakeX_new[i,:,j], color='orange')
            #hfg = plt.subplots(3,1,figsize=(12,6),tight_layout=True)
            #hax = hfg.add_subplot(111)
            #hax.plot(t,realX_u[0,:,0],t,realX_d[0,:,0],t,fakeX_d[0,:,0])
            #hax.plot(t,fakeX_u[0,:,0], color='orange')
            ax0.set_ylim([-1.0, 1.0])
            ax1.set_ylim([-1.0, 1.0])
            ax2.set_ylim([-1.0, 1.0])
            #hax.set_title(r'$X \hspace{0.5} reconstruction$', fontsize=22,fontweight='bold')
            ax0.set_ylabel(r'$X(t) \hspace{0.5} [1]$', fontsize=26,fontweight='bold')
            ax0.set_xlabel(r'$t \hspace{0.5} [s]$', fontsize=26,fontweight='bold')
            ax0.legend([r'$X_u$'], loc='best',frameon=False,fontsize=20)
            ax0.tick_params(axis='both', labelsize=18)
            ax1.set_ylabel(r'$X(t) \hspace{0.5} [1]$', fontsize=26,fontweight='bold')
            ax1.set_xlabel(r'$t \hspace{0.5} [s]$', fontsize=26,fontweight='bold')
            ax1.legend([r'$X_d$'], loc='best',frameon=False,fontsize=20)
            ax1.tick_params(axis='both', labelsize=18)
            ax2.set_ylabel(r'$X(t) \hspace{0.5} [1]$', fontsize=26,fontweight='bold')
            ax2.set_xlabel(r'$t \hspace{0.5} [s]$', fontsize=26,fontweight='bold')
            ax2.legend([r"$G_z(F_x(x_u))$"], loc='best',frameon=False,fontsize=20)
            ax2.tick_params(axis='both', labelsize=18)
            plt.savefig(results_dir + '/reconstruction_switch{:>d}_{:>d}_{:>d}.png'.format(d,j,i),bbox_inches = 'tight')
            #plt.savefig(results_dir + '/reconstruction_switch{:>d}_{:>d}_{:>d}.eps'.format(d,j,i),bbox_inches = 'tight',dpi=200)
            plt.close()

            hfg = plt.figure(figsize=(12,6),tight_layout=True)
            hax = hfg.add_subplot(111)
            N = realX_u.shape[1]
            SAMPLE_RATE = 25
            yf_real_d = rfft(realX_d[i,:,j])
            xf_real_d = rfftfreq(N, 1 / SAMPLE_RATE)
            hax.plot(xf_real_d, np.abs(yf_real_d), color='black')
            yf_switch = rfft(fakeX_new_fft[i,:,j])
            xf_switch = rfftfreq(N, 1 / SAMPLE_RATE)
            hax.plot(xf_switch, np.abs(yf_switch), color='orange',linestyle="--")
            hax.set_ylabel(r'$Amplitude \hspace{0.5} [1]$', fontsize=26,fontweight='bold')
            hax.set_xlabel(r'$Frequency \hspace{0.5} [Hz]$', fontsize=26,fontweight='bold')
            hax.legend([r'$X_d$', r"$G_z(F_x(x_u))$"], loc='best',frameon=False,fontsize=20)
            hax.tick_params(axis='both', labelsize=18)
            plt.savefig(results_dir + '/fft_switch{:>d}_{:>d}_{:>d}.png'.format(d,j,i),bbox_inches = 'tight')
            #plt.savefig(results_dir + '/fft_switch{:>d}_{:>d}_{:>d}.eps'.format(d,j,i),bbox_inches = 'tight',dpi=200)
            plt.close()

            hfg = plt.figure(figsize=(12,6),tight_layout=True)
            hax = hfg.add_subplot(111)
            hax.plot(t,realX_d[i,:,j], color='black')
            hax.plot(t,fakeX_new[i,:,j], color='orange',linestyle="--")
            hax.set_ylabel(r'$X(t) \hspace{0.5} [1]$', fontsize=26,fontweight='bold')
            hax.set_xlabel(r'$t \hspace{0.5} [s]$', fontsize=26,fontweight='bold')
            hax.legend([r'$X_d$',r"$G_z(F_x(x_u))$"], loc='best',frameon=False,fontsize=20)
            hax.tick_params(axis='both', labelsize=18)
            hax.set_ylim([-1.0, 1.0])           

            plt.savefig(results_dir + '/switch{:>d}_{:>d}_{:>d}.png'.format(d,j,i),bbox_inches = 'tight')
            #plt.savefig(results_dir + '/switch{:>d}_{:>d}_{:>d}.eps'.format(d,j,i),bbox_inches = 'tight',dpi=200)
            plt.close()

    #         fig, axs = plt.subplots(2, 2, figsize=(24,12))
    #         axs[0,0].plot(t,realX_u[i,:,j], color='black')
    #         axs[0,0].plot(t,realX_d[i,:,j], color='orange',linestyle="--")
    #         axs[0,0].set_title('Signals', fontsize=30,fontweight='bold')
    #         axs[0,0].set_ylabel(r'$X(t) \hspace{0.5} [1]$', fontsize=26,fontweight='bold')
    #         axs[0,0].set_xlabel(r'$t \hspace{0.5} [s]$', fontsize=26,fontweight='bold')
    #         axs[0,0].legend([r'$X_u$',r"$X_d$"], loc='best',frameon=False,fontsize=20)

    #         axs[1,0].plot(lags_real[i,:,j],corr_real[i,:,j])
    #         axs[1,0].set_title('Cross-correlated signal', fontsize=30,fontweight='bold')
    #         axs[1,0].set_ylabel(r'$X(t) \hspace{0.5} [1]$', fontsize=26,fontweight='bold')
    #         axs[1,0].set_xlabel(r'$Lag$', fontsize=26,fontweight='bold')

    #         axs[0,1].plot(t,realX_u[i,:,j], color='black')
    #         axs[0,1].plot(t,fakeX_new[i,:,j], color='orange',linestyle="--")
    #         axs[0,1].set_title('Signals', fontsize=30,fontweight='bold')
    #         axs[0,1].set_ylabel(r'$X(t) \hspace{0.5} [1]$', fontsize=26,fontweight='bold')
    #         axs[0,1].set_xlabel(r'$t \hspace{0.5} [s]$', fontsize=26,fontweight='bold')
    #         axs[0,1].legend([r'$X_u$',r"$G_z(F_x(x_u))$"], loc='best',frameon=False,fontsize=20)

    #         axs[1,1].plot(lags_switch[i,:,j],corr_switch[i,:,j])
    #         axs[1,1].set_title('Cross-correlated signal', fontsize=30,fontweight='bold')
    #         axs[1,1].set_xlabel(r'$Lag$', fontsize=26,fontweight='bold')
    #         axs[1,1].set_ylabel(r'$X(t) \hspace{0.5} [1]$', fontsize=26,fontweight='bold')
            

    #         fig.tight_layout()

    #         plt.savefig(results_dir + '/cross-corr{:>d}_{:>d}_{:>d}.png'.format(d,j,i),bbox_inches = 'tight')
    #         #plt.savefig(results_dir + '/cross-corr{:>d}_{:>d}_{:>d}.eps'.format(d,j,i),bbox_inches = 'tight',dpi=200)
    #         plt.close()

    #         hfg = plt.figure(figsize=(12,6),tight_layout=True)
    #         hax = hfg.add_subplot(111)
    #         hax.plot(lags_real[i,:,j],corr_real[i,:,j], color='black')
    #         hax.plot(lags_switch[i,:,j],corr_switch[i,:,j], color='orange',linestyle="--")
    #         hax.set_title('Cross-correlated signals - Comparison', fontsize=30,fontweight='bold')
    #         hax.set_ylabel(r'$X(t) \hspace{0.5} [1]$', fontsize=26,fontweight='bold')
    #         hax.set_xlabel(r'$Lag$', fontsize=26,fontweight='bold')
    #         hax.legend([r'$Original$',r"$Switch$"], loc='best',frameon=False,fontsize=20)

    #         plt.savefig(results_dir + '/cross-corr-comparison{:>d}_{:>d}_{:>d}.png'.format(d,j,i),bbox_inches = 'tight')
    #         #plt.savefig(results_dir + '/cross-corr-comparison{:>d}_{:>d}_{:>d}.eps'.format(d,j,i),bbox_inches = 'tight',dpi=200)
    #         plt.close()

    # deconvolution(realX_u,realX_d,fakeX_new,d)



def PlotTHSGoFs(model,realXC,results_dir):
    # Plot reconstructed time-histories
    #realX, realC = realXC

    realX = np.concatenate([x for x, c, m, d in realXC], axis=0)
    realC = np.concatenate([c for x, c, m, d in realXC], axis=0)
    mag = np.concatenate([m for x, c, m, d in realXC], axis=0)
    di = np.concatenate([d for x, c, m, d in realXC], axis=0)

    recX,fakeC,fakeS,fakeN = model.predict(realX)

    ## Print signal GoF
    for j in range(realX.shape[2]):
        for k in range(10):
            i = randint(0, realX.shape[0]-1)
            plot_tf_gofs(realX[i,:,j],recX[i,:,j],dt=0.04,fmin=0.1,fmax=30.0,t0=0.0,nf=100,w0=6,norm='global',st2_isref=True,
                a=10.,k=1.,left=0.1,bottom=0.125, h_1=0.2,h_2=0.125,h_3=0.2,w_1=0.2,w_2=0.6,w_cb=0.01, d_cb=0.0,show=False,
                plot_args=['k', 'r', 'b'],ylim=0., clim=0.)
            plt.savefig(results_dir + '/gof_{:>d}_{:>d}.png'.format(j,i),bbox_inches = 'tight')
            #plt.savefig(results_dir + '/gof_{:>d}_{:>d}.eps'.format(j,i),bbox_inches = 'tight',dpi=200)
            plt.close()

def colored_scatter(*args, **kwargs):
    plt.scatter(*args, **kwargs)
    return

def PlotEGPGgrid(col_x,col_y,col_k,i,results_dir,df,k_is_color=False, scatter_alpha=.7):
    k=0
    for name, df_group in df.groupby(col_k):
        k+=1
    plt.figure(figsize=(10,6), dpi= 500)
    sn.color_palette("Paired", k)
    def colored_scatter(x, y, c=None, edgecolor='black', linewidth=0.8):
        def scatter(*args, **kwargs):
            args = (x, y)
            if c is not None:
                kwargs['c'] = c
            kwargs['alpha'] = scatter_alpha
            kwargs['edgecolor']=edgecolor
            kwargs['linewidth']=linewidth
            plt.scatter(*args, **kwargs)

        return scatter
    g = sn.JointGrid(x=col_x,y=col_y,data=df,space=0.0)
    color = None
    legends=[]
    for name, df_group in df.groupby(col_k):
        legends.append(name)
        if k_is_color:
            color=name
        g.plot_joint(colored_scatter(df_group[col_x],df_group[col_y],color),)
        hax=sn.distplot(df_group[col_x].values,ax=g.ax_marg_x,kde=False,color=color,norm_hist=True)
        hay=sn.distplot(df_group[col_y].values,ax=g.ax_marg_y,kde=False,color=color,norm_hist=True,vertical=True)
        hax.set_xticks(list(np.linspace(0,10,11)))
        hay.set_yticks(list(np.linspace(0,10,11)))
    ## Do also global Hist:
    g.ax_joint.set_xticks(list(np.linspace(0,10,11)))
    g.ax_joint.set_yticks(list(np.linspace(0,10,11)))
    g.ax_joint.spines['right'].set_visible(False)
    g.ax_joint.spines['left'].set_visible(True)
    g.ax_joint.spines['bottom'].set_visible(True)
    g.ax_joint.spines['top'].set_visible(False)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel('EG', fontsize=14)
    plt.ylabel('PG', fontsize=14)
    plt.legend(legends,frameon=False,fontsize=14)
    plt.savefig(results_dir + '/Gz(Fx(X))_gofs_{:>d}.png'.format(i),bbox_inches = 'tight')
    #plt.savefig(results_dir + '/Gz(Fx(X))_gofs_{:>d}.eps'.format(i),bbox_inches = 'tight',dpi=200)
    plt.close()


def PlotBatchGoFs(model,Xtrn,Xvld,i,results_dir):
    # Plot GoFs on a batch

    
    realX_trn = np.concatenate([x for x, c, m, d in Xtrn], axis=0)
    realC_trn = np.concatenate([c for x, c, m, d in Xtrn], axis=0)
    mag_trn = np.concatenate([m for x, c, m, d in Xtrn], axis=0)
    di_trn = np.concatenate([d for x, c, m, d in Xtrn], axis=0)

    fakeX_trn,_,_,_ = model.predict(realX_trn)

    realX_vld = np.concatenate([x for x, c, m, d in Xvld], axis=0)
    realC_vld = np.concatenate([c for x, c, m, d in Xvld], axis=0)
    mag_vld = np.concatenate([m for x, c, m, d in Xvld], axis=0)
    di_vld = np.concatenate([d for x, c, m, d in Xvld], axis=0)

    fakeX_vld,_,_,_ = model.predict(realX_vld)

    egpg_trn = {}
    for j in range(realX_trn.shape[2]):
        egpg_trn['egpg_trn_%d' % j] = np.zeros((realX_trn.shape[0],2),dtype=np.float32)
        st1 = np.zeros((realX_trn.shape[0],realX_trn.shape[1]))
        st2 = np.zeros((realX_trn.shape[0],realX_trn.shape[1]))
        for k in range(realX_trn.shape[0]):
            st1 = realX_trn[k,:,j]
            st2 = fakeX_trn[k,:,j]
            egpg_trn['egpg_trn_%d' % j][k,0] = eg(st1,st2,dt=0.04,fmin=0.1,fmax=30.0,nf=100,w0=6,norm='global',st2_isref=True,a=10.,k=1.)
            egpg_trn['egpg_trn_%d' % j][k,1] = pg(st1,st2,dt=0.04,fmin=0.1,fmax=30.0,nf=100,w0=6,norm='global',st2_isref=True,a=10.,k=1.)

    egpg_vld = {}
    for j in range(realX_vld.shape[2]):
        egpg_vld['egpg_vld_%d' % j] = np.zeros((realX_vld.shape[0],2),dtype=np.float32)
        st1 = np.zeros((realX_vld.shape[0],realX_vld.shape[1]))
        st2 = np.zeros((realX_vld.shape[0],realX_vld.shape[1]))
        for k in range(realX_vld.shape[0]):
            st1 = realX_vld[k,:,j]
            st2 = fakeX_vld[k,:,j]
            egpg_vld['egpg_vld_%d' % j][k,0] = eg(st1,st2,dt=0.04,fmin=0.1,fmax=30.0,nf=100,w0=6,norm='global',st2_isref=True,a=10.,k=1.)
            egpg_vld['egpg_vld_%d' % j][k,1] = pg(st1,st2,dt=0.04,fmin=0.1,fmax=30.0,nf=100,w0=6,norm='global',st2_isref=True,a=10.,k=1.)


    egpg_df_trn = {}
    for j in range(realX_trn.shape[2]):
        egpg_df_trn['egpg_df_trn_%d' % j] = pd.DataFrame(egpg_trn['egpg_trn_%d' % j],columns=['EG','PG'])
        egpg_df_trn['egpg_df_trn_%d' % j]['kind']=r"$G_z(F_x(x)) \hspace{0.5} train$"

    egpg_df_vld = {}
    for j in range(realX_vld.shape[2]):
        egpg_df_vld['egpg_df_vld_%d' % j] = pd.DataFrame(egpg_vld['egpg_vld_%d' % j],columns=['EG','PG'])
        egpg_df_vld['egpg_df_vld_%d' % j]['kind']=r"$G_z(F_x(x)) \hspace{0.5} validation$"

    egpg_data = []
    for j in range(realX_trn.shape[2]):
        egpg_data.append(egpg_df_trn['egpg_df_trn_%d' % j])
    for j in range(realX_vld.shape[2]):
        egpg_data.append(egpg_df_vld['egpg_df_vld_%d' % j])
    egpg_df = pd.concat(egpg_data)

    egpg_df.to_csv(results_dir + '/EG_PG_{:>d}.csv'.format(i), index= True)
    PlotEGPGgrid('EG','PG','kind',i,results_dir,df=egpg_df)

def PlotClassificationMetrics(model,realXC,results_dir):
    # Plot classification metrics
    realX = np.concatenate([x for x, c, m, d in realXC], axis=0)
    realC = np.concatenate([c for x, c, m, d in realXC], axis=0)
    mag = np.concatenate([m for x, c, m, d in realXC], axis=0)
    di = np.concatenate([d for x, c, m, d in realXC], axis=0)

    fakeC, recC = model.label_predictor(realX,realC)

    labels_fake = np.zeros((fakeC.shape[0]))
    for i in range(fakeC.shape[0]):
        labels_fake[i] = np.argmax(fakeC[i,:])

    labels_rec = np.zeros((recC.shape[0]))
    for i in range(recC.shape[0]):
        labels_rec[i] = np.argmax(recC[i,:])
    
    labels_real = np.zeros((realC.shape[0]))
    for i in range(realC.shape[0]):
        labels_real[i] = np.argmax(realC[i,:])

    labels_fake = labels_fake.astype(int)
    labels_rec = labels_rec.astype(int)
    labels_real = labels_real.astype(int)

    target_names = []
    for i in range(options['latentCdim']):
        target_names.append('damage class %d'% i) 

    fig, ax = plt.subplots()
    report = classification_report(y_true = labels_real, y_pred = labels_fake,
            target_names=target_names,output_dict=True,zero_division=1)
    df = pd.DataFrame(report).transpose()
    df.to_csv(results_dir + '/Classification Report C.csv', index= True)
    cr = sn.heatmap(pd.DataFrame(report).iloc[:-1, :].T, annot=True, vmin=0, vmax=1, annot_kws={"size": 12})
    cr.tick_params(axis='both', labelsize=12)
    cr.set_yticklabels(cr.get_yticklabels(), rotation=0)
    plt.savefig(results_dir + '/classification_report_fakeC.png',bbox_inches = 'tight')
    #plt.savefig(results_dir + '/classification_report.eps',bbox_inches = 'tight',dpi=200)
    plt.close()

    conf_mat = confusion_matrix(labels_real, labels_fake)
    fig, ax = plt.subplots(figsize=(10,10),tight_layout=True)
    sn.heatmap(conf_mat, annot=True, fmt='d', xticklabels=target_names,yticklabels=target_names, vmin=0, vmax=realC.shape[0],
        annot_kws={"size": 20})
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=18)
    plt.tick_params(axis='both', labelsize=20)
    plt.ylabel("True class",fontsize=22,labelpad=10)
    plt.xlabel("Predicted class",fontsize=22,labelpad=10)
    plt.savefig(results_dir + '/confusion_matrix_fakeC.png',bbox_inches = 'tight')
    #plt.savefig(results_dir + '/confusion_matrixC.eps',bbox_inches = 'tight',dpi=200)
    plt.close()


    fig, ax = plt.subplots()
    report = classification_report(y_true = labels_real, y_pred = labels_rec,
            target_names=target_names,output_dict=True,zero_division=1)
    df = pd.DataFrame(report).transpose()
    df.to_csv(results_dir + '/Classification Report recC.csv', index= True)
    cr = sn.heatmap(pd.DataFrame(report).iloc[:-1, :].T, annot=True, vmin=0, vmax=1, annot_kws={"size": 12})
    cr.tick_params(axis='both', labelsize=12)
    cr.set_yticklabels(cr.get_yticklabels(), rotation=0)
    plt.savefig(results_dir + '/classification_report_recC.png',bbox_inches = 'tight')
    #plt.savefig(results_dir + '/classification_reportrec.eps',bbox_inches = 'tight',dpi=200)
    plt.close()

    
    conf_mat = confusion_matrix(labels_real, labels_rec)
    fig, ax = plt.subplots(figsize=(10,10),tight_layout=True)
    sn.heatmap(conf_mat, annot=True, fmt='d', xticklabels=target_names,yticklabels=target_names, vmin=0, vmax=realC.shape[0],
        annot_kws={"size": 20})
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=18)
    plt.tick_params(axis='both', labelsize=20)
    plt.ylabel("True class",fontsize=22,labelpad=10)
    plt.xlabel("Predicted class",fontsize=22,labelpad=10)
    plt.savefig(results_dir + '/confusion_matrix_recC.png',bbox_inches = 'tight')
    #plt.savefig(results_dir + '/confusion_matrixrecC.eps',bbox_inches = 'tight',dpi=200)
    plt.close()

    return

def scatter_hist(x, y, ax, ax_histx, ax_histy):
    # no labels
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    # the scatter plot:
    ax.scatter(x, y)

    # now determine nice limits by hand:
    binwidth = 0.25
    xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))
    lim = (int(xymax/binwidth) + 1) * binwidth

    bins = np.arange(-lim, lim + binwidth, binwidth)
    ax_histx.hist(x, bins=bins)
    ax_histy.hist(y, bins=bins, orientation='horizontal')

def PlotLatentSpace(model,realXC,results_dir):
    s_list = []
    n_list = []
    cq_list = []
    c_list = []

    iterator = iter(realXC)  # restart data iter
    for b in range(len(realXC)):
        data = iterator.get_next()
        realX = data[0]
        realC = data[1]
        # import pdb
        # pdb.set_trace()
        [_,C_set,S_set,N_set] = model.predict(realX,batch_size=1)
        s_list.append(S_set)
        n_list.append(N_set)
        cq_list.append(C_set)
        c_list.append(realC)

    s_np = tf.concat(s_list, axis=0).numpy().squeeze()
    n_np = tf.concat(n_list, axis=0).numpy().squeeze()
    cq_tensor = tf.concat(cq_list, axis=0)
    c_tensor = tf.concat(cq_list, axis=0)
    cq_np = np.argmax(cq_tensor.numpy().squeeze(), axis = -1)
    c_np = np.argmax(c_tensor.numpy().squeeze(), axis = -1)

    fig = plt.figure()

    #hist plot
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    spacing = 0.005

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom + height + spacing, width, 0.2]
    rect_histy = [left + width + spacing, bottom, 0.2, height]

    ax = fig.add_axes(rect_scatter)

    ax = fig.add_axes(rect_scatter)
    ax_histx = fig.add_axes(rect_histx, sharex=ax)
    ax_histy = fig.add_axes(rect_histy, sharey=ax)

    # use the previously defined function
    scatter_hist(s_np[:,0],s_np[:,1], ax, ax_histx, ax_histy)
    fig.savefig(results_dir + '/s_all.png')
    plt.close()

    for n_i in range(1):
        #per example
        fig, ax = plt.subplots()
        for i,n in enumerate(n_np):
            if cq_np[i] == 0 and c_np[i] == cq_np[i] :
                plt.scatter(n[n_i],n[n_i+1],c = 'b',marker='o', alpha=0.8,s=12)
            elif cq_np[i] == 0 and c_np[i] != cq_np[i] :
                plt.scatter(n[n_i],n[n_i+1],c = 'b',marker='x', alpha=0.8,s=12)
            elif cq_np[i] == 1 and c_np[i] == cq_np[i] :
                plt.scatter(n[n_i],n[n_i+1],c = 'r',marker='o', alpha=0.8,s=12)
            elif cq_np[i] == 1 and c_np[i] != cq_np[i] :
                plt.scatter(n[n_i],n[n_i+1],c = 'r',marker='x', alpha=0.8,s=12)

        plt.ylabel("N1",fontsize=12,labelpad=10)
        plt.xlabel("N0",fontsize=12,labelpad=10)
        plt.title("N variables",fontsize=16)
        plt.legend(["0","1"],frameon=False,fontsize=14)
        fig.savefig(results_dir + '/n_{:>d}_{:>d}'.format(n_i,n_i+1),dpi=300,bbox_inches = 'tight')
        plt.close()

    
    for s_i in range(1):
        #per example
        fig, ax = plt.subplots()
        for i,s in enumerate(s_np):
            if cq_np[i] == 0 and c_np[i] == cq_np[i] :
                plt.scatter(s[s_i],s[s_i+1],c = 'b',marker='o', alpha=0.8,s=12)
            elif cq_np[i] == 0 and c_np[i] != cq_np[i] :
                plt.scatter(s[s_i],s[s_i+1],c = 'b',marker='x', alpha=0.8,s=12)
            elif cq_np[i] == 1 and c_np[i] == cq_np[i] :
                plt.scatter(s[s_i],s[s_i+1],c = 'r',marker='o', alpha=0.8,s=12)
            elif cq_np[i] == 1 and c_np[i] != cq_np[i] :
                plt.scatter(s[s_i],s[s_i+1],c = 'r',marker='x', alpha=0.8,s=12)
            
        plt.ylabel("S1",fontsize=12,labelpad=10)
        plt.xlabel("S0",fontsize=12,labelpad=10)
        plt.title("S variables",fontsize=16)
        plt.legend(["0","1"],frameon=False,fontsize=14)

        fig.savefig(results_dir + '/s_{:>d}_{:>d}'.format(s_i,s_i+1),dpi=300,bbox_inches = 'tight')
        plt.close()

    return

def PlotTSNE(model,realXC,results_dir):
    
    realX = np.concatenate([x for x, c, m, d in realXC], axis=0)
    realC = np.concatenate([c for x, c, m, d in realXC], axis=0)
    mag = np.concatenate([m for x, c, m, d in realXC], axis=0)
    di = np.concatenate([d for x, c, m, d in realXC], axis=0)

    _,fakeC,fakeS,fakeN = model.predict(realX)

    labels = np.zeros((realC.shape[0]))
    for i in range(realC.shape[0]):
        labels[i] = np.argmax(realC[i,:])

    
    transformerN = TSNE(n_components=3, verbose=1, random_state=123)
    n = transformerN.fit_transform(fakeN)

    dfN = pd.DataFrame()
    dfN["C"] = labels
    dfN["Dimension 1"] = n[:,0]
    dfN["Dimension 2"] = n[:,1]
    dfN["Dimension 3"] = n[:,2]

    i1 = 0
    i2 = 0
    mark_low = []
    mark_high = []
    lab = []
    for i in range(realX.shape[0]):
        if mag[i,0]<=4.0:
            i1 = i1+1
            mark_low.append('^')
            lab.append('0')
            if i1==1:
                fakeN_low = fakeN[i,:].reshape((1,fakeN.shape[1]))
                fakeS_low = fakeS[i,:].reshape((1,fakeS.shape[1]))
                fakeC_low = labels[i].reshape((1,1))
                mag_low = mag[i].reshape((1,1))
                d_low = di[i].reshape((1,1))
            else:
                fakeN_low = np.concatenate((fakeN_low, (fakeN[i,:]).reshape((1,fakeN.shape[1])))).reshape((i1,fakeN.shape[1]))
                fakeS_low = np.concatenate((fakeS_low, (fakeS[i,:]).reshape((1,fakeS.shape[1])))).reshape((i1,fakeS.shape[1]))
                fakeC_low = np.concatenate((fakeC_low, (labels[i]).reshape((1,1)))).reshape((i1,1))
                mag_low = np.concatenate((mag_low, (mag[i]).reshape((1,1)))).reshape((i1,1))
                d_low = np.concatenate((d_low, (di[i]).reshape((1,1)))).reshape((i1,1))
        elif 5.0<=mag[i,0]:
            mark_high.append('o')
            lab.append('1')
            i2 = i2+1
            if i2==1:
                fakeN_high = fakeN[i,:].reshape((1,fakeN.shape[1]))
                fakeS_high = fakeS[i,:].reshape((1,fakeS.shape[1]))
                fakeC_high = labels[i].reshape((1,1))
                mag_high = mag[i].reshape((1,1))
                d_high = di[i].reshape((1,1))
            else:
                fakeN_high = np.concatenate((fakeN_high, (fakeN[i,:]).reshape((1,fakeN.shape[1])))).reshape((i2,fakeN.shape[1]))
                fakeS_high = np.concatenate((fakeS_high, (fakeS[i,:]).reshape((1,fakeS.shape[1])))).reshape((i2,fakeS.shape[1]))
                fakeC_high = np.concatenate((fakeC_high, (labels[i]).reshape((1,1)))).reshape((i2,1))
                mag_high = np.concatenate((mag_high, (mag[i]).reshape((1,1)))).reshape((i2,1))
                d_high = np.concatenate((d_high, (di[i]).reshape((1,1)))).reshape((i2,1))

    s1=30
    s2=70
    s3=110

    size = []
    size_low = []
    size_high = []
    col = []

    for i in range(realX.shape[0]):
        if di[i] <= 0.4:
            size.append(s1)
        elif 0.4 < di[i] <= 1:
            size.append(s2)
        else:
            size.append(s3)
        if labels[i] == 0:
            col.append('red')
        else:
            col.append('blue')

    for i in range(realX.shape[0]):
        if mag[i,0]<=4.0:
            if di[i] <= 0.4:
                size_low.append(s1)
            elif 0.4 < di[i] <= 1:
                size_low.append(s2)
            else:
                size_low.append(s3)
        elif 5.0<=mag[i,0]:
            if di[i] <= 0.4:
                size_high.append(s1)
            elif 0.4 < di[i] <= 1:
                size_high.append(s2)
            else:
                size_high.append(s3)

    fig, ax = plt.subplots()
    ax.scatter(di, mag, c=col, s=size, marker="o",label=None, alpha=0.8,edgecolors='w')
    ax.set_xlabel(r'$Damage \hspace{0.5} index$',fontsize=14)
    ax.set_ylabel(r'$Magnitude$',fontsize=14)
    ax.xaxis.set_tick_params(labelsize=14)
    ax.yaxis.set_tick_params(labelsize=14)
    p1 = ax.scatter([], [], c='red', s=50, label=r'$0$', marker="o", alpha=0.8)
    p2 = ax.scatter([], [], c='blue', s=50, label=r'$1$', marker="o", alpha=0.8)

    first_legend = plt.legend(handles=[p1,p2], bbox_to_anchor=(1.04,1), loc="upper left",
                    borderaxespad=(0), title=r'$Class$',frameon=False,fontsize=14,title_fontsize=14)
    plt.gca().add_artist(first_legend)
    d1 = ax.scatter([], [], c='k', s=s1, label=r'$Undamaged$', marker="o", alpha=0.3)
    d2 = ax.scatter([], [], c='k', s=s2, label=r'$Damaged$', marker="o", alpha=0.3)
    d3 = ax.scatter([], [], c='k', s=s3, label=r'$Collapsed$', marker="o", alpha=0.3)
    plt.legend(handles=[d1,d2,d3], bbox_to_anchor=(1.04,0), loc="lower left",borderaxespad=(0),frameon=False,
            title=r'$Park \hspace{0.5} & \hspace{0.5} Ang \hspace{0.5} Index$',fontsize=14,title_fontsize=14)
    plt.savefig(results_dir + '/data.png',bbox_inches = 'tight')
    plt.close()
    
    size = np.array(size,dtype=np.float32)
            
    colors = np.array(["red","blue"])

    cat = np.array(['Undamaged', 'Damaged'])

    n_low = transformerN.fit_transform(fakeN_low)

    dfN_low = pd.DataFrame()
    dfN_low["C"] = fakeC_low[:,0]
    dfN_low["mag"] = mag_low[:,0]
    dfN_low["d"] = d_low[:,0]
    dfN_low["Dimension 1"] = n_low[:,0]
    dfN_low["Dimension 2"] = n_low[:,1]
    dfN_low["Dimension 3"] = n_low[:,2]

    n_high = transformerN.fit_transform(fakeN_high)

    dfN_high = pd.DataFrame()
    dfN_high["C"] = fakeC_high[:,0]
    dfN_high["mag"] = mag_high[:,0]
    dfN_high["d"] = d_high[:,0]
    dfN_high["Dimension 1"] = n_high[:,0]
    dfN_high["Dimension 2"] = n_high[:,1]
    dfN_high["Dimension 3"] = n_high[:,2]

    col_low=[]
    col_low0=[]
    lab_low=[]
    for i in range(n_low.shape[0]):
        if fakeC_low[i]==0:
            col_low.append('red')
            col_low0.append('green')
            lab_low.append('0')
        else:
            col_low.append('blue')
            col_low0.append('orange')
            lab_low.append('1')
    
    col_high=[]
    col_high0=[]
    lab_high=[]
    for i in range(n_high.shape[0]):
        if fakeC_high[i]==0:
            col_high.append('red')
            col_high0.append('green')
            lab_high.append('0')
        else:
            col_high.append('blue')
            col_high0.append('orange')
            lab_high.append('1')
    
    n_low_x = np.min((n_low[:,0]))
    n_high_x = np.min((n_high[:,0]))
    n_min_x = int(np.min((n_low_x,n_high_x))-5)

    n_low_x = np.max((n_low[:,0]))
    n_high_x = np.max((n_high[:,0]))
    n_max_x = int(np.max((n_low_x,n_high_x))+5)

    n_low_y = np.min((n_low[:,1]))
    n_high_y = np.min((n_high[:,1]))
    n_min_y = int(np.min((n_low_y,n_high_y))-5)

    n_low_y = np.max((n_low[:,1]))
    n_high_y = np.max((n_high[:,1]))
    n_max_y = int(np.max((n_low_y,n_high_y))+5)

    col=[]
    n0_x=[]
    n1_x=[]
    n0_y=[]
    n1_y=[]
    for i in range(realX.shape[0]):
        if labels[i]==0:
            col.append('red')
            n0_x.append(n[i,0])
            n0_y.append(n[i,1])
        else:
            col.append('blue')
            n1_x.append(n[i,0])
            n1_y.append(n[i,1])
    n0_x = np.array(n0_x,dtype=np.float32)
    n0_y = np.array(n0_y,dtype=np.float32)
    n1_x = np.array(n1_x,dtype=np.float32)
    n1_y = np.array(n1_y,dtype=np.float32)

    fig, ax = plt.subplots()
    ax.scatter(n_low[:,0], n_low[:,1], c=col_low, s=size_low, marker="o",label=None, alpha=0.8,edgecolors='w')
    ax.scatter(n_high[:,0], n_high[:,1], c=col_high, s=size_high, marker="o",label=None, alpha=0.8,edgecolors='w')
    ax.set_xlabel(r'$Dimension \hspace{0.5} 1$',fontsize=14)
    ax.set_ylabel(r'$Dimension \hspace{0.5} 2$',fontsize=14)
    ax.xaxis.set_tick_params(labelsize=14)
    ax.yaxis.set_tick_params(labelsize=14)
    
    p1 = ax.scatter([], [], c='red', s=50, label=r'$0$', marker="o", alpha=0.8)
    p2 = ax.scatter([], [], c='blue', s=50, label=r'$1$', marker="o", alpha=0.8)

    first_legend = plt.legend(handles=[p1,p2], bbox_to_anchor=(1.04,1), loc="upper left",
                    borderaxespad=(0), title=r'$Class$',frameon=False,fontsize=14,title_fontsize=14)
    plt.gca().add_artist(first_legend)
    d1 = ax.scatter([], [], c='k', s=s1, label=r'$Undamaged$', marker="o", alpha=0.3)
    d2 = ax.scatter([], [], c='k', s=s2, label=r'$Damaged$', marker="o", alpha=0.3)
    d3 = ax.scatter([], [], c='k', s=s3, label=r'$Collapsed$', marker="o", alpha=0.3)
    plt.legend(handles=[d1,d2,d3], bbox_to_anchor=(1.04,0), loc="lower left",borderaxespad=(0),frameon=False,
            title=r'$Park \hspace{0.5} & \hspace{0.5} Ang \hspace{0.5} Index$',fontsize=14,title_fontsize=14)
    plt.savefig(results_dir + '/tsne_N.png',bbox_inches = 'tight')
    plt.close()

    fig, ax = plt.subplots(1,2, figsize=(16,6))
    ax[0].set_title(r"$Low \hspace{0.5} magnitude \hspace{0.5} earthquakes$",fontsize=16)
    ax[0].scatter(n_low[:,0], n_low[:,1], c=col_low, marker="o",label=None,alpha=0.8,edgecolors='w')
    for cat,colors in zip([r'$0$',r'$1$'],colors):
        ax[0].scatter([], [], c=colors, marker="o",alpha=0.8, label=cat)
    ax[0].legend(scatterpoints=1, frameon=False, labelspacing=1,title=r'$Class$',fontsize=14,title_fontsize=14)
    ax[0].set_xlim(n_min_x, n_max_x)
    ax[0].set_ylim(n_min_y, n_max_y)
    ax[0].set_xlabel(r'$Dimension \hspace{0.5} 1$',fontsize=14)
    ax[0].set_ylabel(r'$Dimension \hspace{0.5} 2$',fontsize=14)
    ax[0].xaxis.set_tick_params(labelsize=14)
    ax[0].yaxis.set_tick_params(labelsize=14)
    ax[1].set_title(r"$High \hspace{0.5} magnitude \hspace{0.5} earthquakes$",fontsize=16)
    ax[1].scatter(n_high[:,0], n_high[:,1], c=col_high, marker="o",label=None,alpha=0.8,edgecolors='w')
    for cat,colors in zip([r'$1$'],['blue']):
        ax[1].scatter([], [], c=colors, marker="o",alpha=0.8, label=cat)
    ax[1].legend(scatterpoints=1, frameon=False, labelspacing=1,title=r'$Class$',fontsize=14,title_fontsize=14)
    ax[1].set_xlim(n_min_x, n_max_x)
    ax[1].set_ylim(n_min_y, n_max_y)
    ax[1].set_xlabel(r'$Dimension \hspace{0.5} 1$',fontsize=14)
    ax[1].set_ylabel(r'$Dimension \hspace{0.5} 2$',fontsize=14)
    ax[1].xaxis.set_tick_params(labelsize=14)
    ax[1].yaxis.set_tick_params(labelsize=14)
    plt.savefig(results_dir + '/tsne_N_low_high.png',bbox_inches = 'tight')
    plt.close()

    
    transformerS = TSNE(n_components=3, verbose=1, random_state=123)

    s = transformerS.fit_transform(fakeS)

    dfS = pd.DataFrame()
    dfS["C"] = labels
    dfS["Dimension 1"] = s[:,0]
    dfS["Dimension 2"] = s[:,1]
    dfS["Dimension 3"] = s[:,2]

    
    s0_x=[]
    s1_x=[]
    s0_y=[]
    s1_y=[]

    for i in range(realX.shape[0]):
        if labels[i]==0:
            s0_x.append(s[i,0])
            s0_y.append(s[i,1])
        else:
            s1_x.append(s[i,0])
            s1_y.append(s[i,1])

    s0_x = np.array(s0_x,dtype=np.float32)
    s0_y = np.array(s0_y,dtype=np.float32)
    s1_x = np.array(s1_x,dtype=np.float32)
    s1_y = np.array(s1_y,dtype=np.float32)
       

    s_low = transformerS.fit_transform(fakeS_low)

    dfS_low = pd.DataFrame()
    dfS_low["C"] = fakeC_low[:,0]
    dfS_low["mag"] = mag_low[:,0]
    dfS_low["d"] = d_low[:,0]
    dfS_low["Dimension 1"] = s_low[:,0]
    dfS_low["Dimension 2"] = s_low[:,1]
    dfS_low["Dimension 3"] = s_low[:,2]

    s_high = transformerS.fit_transform(fakeS_high)

    dfS_high = pd.DataFrame()
    dfS_high["C"] = fakeC_high[:,0]
    dfS_high["mag"] = mag_high[:,0]
    dfS_high["d"] = d_high[:,0]
    dfS_high["Dimension 1"] = s_high[:,0]
    dfS_high["Dimension 2"] = s_high[:,1]
    dfS_high["Dimension 3"] = s_high[:,2]

    s_low_x = np.min((s_low[:,0]))
    s_high_x = np.min((s_high[:,0]))
    s_min_x = int(np.min((s_low_x,s_high_x))-5)

    s_low_x = np.max((s_low[:,0]))
    s_high_x = np.max((s_high[:,0]))
    s_max_x = int(np.max((s_low_x,s_high_x))+5)

    s_low_y = np.min((s_low[:,1]))
    s_high_y = np.min((s_high[:,1]))
    s_min_y = int(np.min((s_low_y,s_high_y))-5)

    s_low_y = np.max((s_low[:,1]))
    s_high_y = np.max((s_high[:,1]))
    s_max_y = int(np.max((s_low_y,s_high_y))+5)


    fig, ax = plt.subplots()
    ax.scatter(s_low[:,0], s_low[:,1], c=col_low, s=size_low, marker="o",label=None, alpha=0.8,edgecolors='w')
    ax.scatter(s_high[:,0], s_high[:,1], c=col_high, s=size_high, marker="o",label=None, alpha=0.8,edgecolors='w')
    ax.set_xlabel(r'$Dimension \hspace{0.5} 1$',fontsize=14)
    ax.set_ylabel(r'$Dimension \hspace{0.5} 2$',fontsize=14)

    p1 = ax.scatter([], [], c='red', s=50, label=r'$0$', marker="o", alpha=0.8)
    p2 = ax.scatter([], [], c='blue', s=50, label=r'$1$', marker="o", alpha=0.8)
    first_legend = plt.legend(handles=[p1,p2], bbox_to_anchor=(1.04,1), loc="upper left", borderaxespad=(0),
                    title=r'$Class$',frameon=False,fontsize=14,title_fontsize=14)
    plt.gca().add_artist(first_legend)
    d1 = ax.scatter([], [], c='k', s=s1, label=r'$Undamaged$', marker="o", alpha=0.3)
    d2 = ax.scatter([], [], c='k', s=s2, label=r'$Damaged$', marker="o", alpha=0.3)
    d3 = ax.scatter([], [], c='k', s=s3, label=r'$Collapsed$', marker="o", alpha=0.3)
    plt.legend(handles=[d1,d2,d3], bbox_to_anchor=(1.04,0), loc="lower left", borderaxespad=(0),frameon=False,
            title=r'$Park \hspace{0.5} & \hspace{0.5} Ang \hspace{0.5} Index$',fontsize=14,title_fontsize=14)
    plt.savefig(results_dir + '/tsne_S.png',bbox_inches = 'tight')
    plt.close()

    
    fig, ax = plt.subplots(1,2, figsize=(16,6))
    ax[0].set_title(r"$Low \hspace{0.5} magnitude \hspace{0.5} earthquakes$")
    ax[0].scatter(s_low[:,0], s_low[:,1], c=col_low, marker="o",alpha=0.8,edgecolors='w')
    ax[0].set_xlim(s_min_x,s_max_x)
    ax[0].set_ylim(s_min_y,s_max_y)
    ax[0].set_xlabel(r'$Dimension \hspace{0.5} 1$',fontsize=14)
    ax[0].set_ylabel(r'$Dimension \hspace{0.5} 2$',fontsize=14)
    for cat,colors in zip([r'$0$',r'$1$'],['red','blue']):
        ax[0].scatter([], [], c=colors, marker="o",alpha=0.8, label=cat)
    ax[0].legend(scatterpoints=1, frameon=False, labelspacing=1, title=r'$Class$',fontsize=14,title_fontsize=14)
    ax[1].set_title(r"$High \hspace{0.5} magnitude \hspace{0.5} earthquakes$")
    ax[1].scatter(s_high[:,0], s_high[:,1], c=col_high, marker="o",alpha=0.8,edgecolors='w')
    for cat,colors in zip([r'$1$'],['blue']):
        ax[1].scatter([], [], c=colors, marker="o",alpha=0.8, label=cat)
    ax[1].legend(scatterpoints=1, frameon=False, labelspacing=1, title=r'$Class$',fontsize=14,title_fontsize=14)
    ax[1].set_xlim(s_min_x,s_max_x)
    ax[1].set_ylim(s_min_y,s_max_y)
    ax[1].set_xlabel(r'$Dimension \hspace{0.5} 1$',fontsize=14)
    ax[1].set_ylabel(r'$Dimension \hspace{0.5} 2$',fontsize=14)
    plt.savefig(results_dir + '/tsne_S_low_high.png',bbox_inches = 'tight')
    plt.close()

    fig, ax = plt.subplots()

    ax.scatter(s_low[:,0], s_low[:,1], c=col_low, s=size_low, marker="o",label=None, alpha=0.8,edgecolors='w')
    ax.scatter(s_high[:,0], s_high[:,1], c=col_high, s=size_high, marker="o",label=None, alpha=0.8,edgecolors='w')
    ax.scatter(n_low[:,0], n_low[:,1], c=col_low0, s=size_low, marker="^",label=None, alpha=0.8,edgecolors='w')
    ax.scatter(n_high[:,0], n_high[:,1], c=col_high0, s=size_high, marker="^",label=None, alpha=0.8,edgecolors='w')

    p1 = ax.scatter([], [], c='red', s=50, label=r'$S_0$', marker="o", alpha=0.8)
    p2 = ax.scatter([], [], c='blue', s=50, label=r'$S_1$', marker="o", alpha=0.8)
    p3 = ax.scatter([], [], c='green', s=50, label=r'$N_0$', marker="^", alpha=0.8)
    p4 = ax.scatter([], [], c='orange', s=50, label=r'$N_1$', marker="^", alpha=0.8)
    
    first_legend = plt.legend(handles=[p1,p2,p3,p4], bbox_to_anchor=(1.04,1), loc="upper left", borderaxespad=(0),
                    title=r'$Variable \hspace{0.5} and \hspace{0.5} Class$',frameon=False,fontsize=14,title_fontsize=14)
    plt.gca().add_artist(first_legend)
    d1 = ax.scatter([], [], c='k', s=s1, label=r'$Undamaged$', marker="o", alpha=0.3)
    d2 = ax.scatter([], [], c='k', s=s2, label=r'$Damaged$', marker="o", alpha=0.3)
    d3 = ax.scatter([], [], c='k', s=s3, label=r'$Collapsed$', marker="o", alpha=0.3)
    plt.legend(handles=[d1,d2,d3], bbox_to_anchor=(1.04,0), loc="lower left", borderaxespad=(0),frameon=False,
                title=r'$Park \hspace{0.5} & \hspace{0.5} Ang \hspace{0.5} Index$',fontsize=14,title_fontsize=14)
    plt.xlabel(r'$Dimension \hspace{0.5} 1$',fontsize=14)
    plt.ylabel(r'$Dimension \hspace{0.5} 2$',fontsize=14)
    plt.title(r"$Variables \hspace{0.5} S \hspace{0.5} and \hspace{0.5} N: \hspace{0.5} T-SNE \hspace{0.5} projection$")
    plt.savefig(results_dir + '/tsne_S_N.png',bbox_inches = 'tight')
    plt.close()

    fig = plt.figure(figsize = (10, 7))
    ax = plt.axes(projection ="3d")
    color_map = plt.get_cmap('cool')

    d_conc = [np.concatenate((d_low[:,0],d_high[:,0]))]
    mark_low.append(mark_high)

    v = int(np.max(d_conc))
    
    # p1 = ax.scatter3D(s_low[:,0], s_low[:,1], d_low[:,0], c=d_low[:,0], edgecolors='dimgray', alpha=1, marker='^', s=50, cmap=color_map, vmin=0,vmax=v)
    # p2 = ax.scatter3D(s_high[:,0], s_high[:,1], d_high[:,0], c=d_high[:,0], edgecolors='dimgray', alpha=1, marker='o', s=50, cmap=color_map, vmin=0,vmax=v)
    p1 = ax.scatter3D(s_low[:,0], s_low[:,1], s_low[:,2], c=d_low[:,0], edgecolors='dimgray', alpha=1, marker='^', s=50, cmap=color_map, vmin=0,vmax=v)
    p2 = ax.scatter3D(s_high[:,0], s_high[:,1], s_high[:,2], c=d_high[:,0], edgecolors='dimgray', alpha=1, marker='o', s=50, cmap=color_map, vmin=0,vmax=v)
    
    ax.set_xlabel(r"$Dimension \hspace{0.5} 1$", fontweight ='bold',fontsize=14)
    ax.set_ylabel(r"$Dimension \hspace{0.5} 2$", fontweight ='bold',fontsize=14)
    ax.set_zlabel(r"$Dimension \hspace{0.5} 3$", fontweight ='bold',fontsize=14)
    symbol = ['^','o']
    title = [r'$Low$', r'$High$']
    for t, m in zip(title,symbol):
        plt.scatter([], [], s=50, c='k', alpha=0.3, marker=m, label=t)
    plt.legend(scatterpoints=1, frameon=False, labelspacing=1, title=r'$Magnitude$', loc="upper left",fontsize=14,title_fontsize=14) 
    cbar = plt.colorbar(p1)
    cbar.set_label(r"$Damage \hspace{0.5} index$",fontsize=14)
    cbar.ax.yaxis.set_tick_params(labelsize=14)
    plt.savefig(results_dir + '/3d_S_index.png',bbox_inches = 'tight')
    plt.close()

    fig = plt.figure(figsize = (10, 7))
    ax = plt.axes(projection ="3d")
    p1 = ax.scatter3D(s_low[:,0], s_low[:,1], s_low[:,2], c=mag_low[:,0], edgecolors='dimgray', alpha=1, marker='^', s=50, cmap=color_map, vmin=0,vmax=v)
    p2 = ax.scatter3D(s_high[:,0], s_high[:,1], s_high[:,2], c=mag_high[:,0], edgecolors='dimgray', alpha=1, marker='o', s=50, cmap=color_map, vmin=0,vmax=v)
    ax.set_xlabel(r"$Dimension \hspace{0.5} 1$", fontweight ='bold',fontsize=14)
    ax.set_ylabel(r"$Dimension \hspace{0.5} 2$", fontweight ='bold',fontsize=14)
    ax.set_zlabel(r"$Dimension \hspace{0.5} 3$", fontweight ='bold',fontsize=14)
    symbol = ['^','o']
    title = [r'$Low$', r'$High$']
    for t, m in zip(title,symbol):
        plt.scatter([], [], s=50, c='k', alpha=0.3, marker=m, label=t)
    plt.legend(scatterpoints=1, frameon=False, labelspacing=1, title=r'$Magnitude$', loc="upper left",fontsize=14,title_fontsize=14) 
    cbar = plt.colorbar(p1)
    cbar.set_label(r"$Magnitude$",fontsize=14)
    cbar.ax.yaxis.set_tick_params(labelsize=14)
    plt.savefig(results_dir + '/3d_S_mag.png',bbox_inches = 'tight')
    plt.close()

    fig = plt.figure(figsize = (10, 7))
    ax = plt.axes(projection ="3d")
    p1 = ax.scatter3D(n_low[:,0], n_low[:,1], n_low[:,2], c=d_low[:,0], edgecolors='dimgray', alpha=1,  marker='^', s=50, cmap=color_map, vmin=0,vmax=v)
    p2 = ax.scatter3D(n_high[:,0], n_high[:,1], n_high[:,2], c=d_high[:,0], edgecolors='dimgray', alpha=1,  marker='o', s=50, cmap=color_map, vmin=0,vmax=v)
    ax.set_xlabel(r"$Dimension \hspace{0.5} 1$", fontweight ='bold',fontsize=14)
    ax.set_ylabel(r"$Dimension \hspace{0.5} 2$", fontweight ='bold',fontsize=14)
    ax.set_zlabel(r"$Dimension \hspace{0.5} 3$", fontweight ='bold',fontsize=14)
    symbol = ['^','o']
    title = [r'$Low$', r'$High$']
    for t, m in zip(title,symbol):
        plt.scatter([], [], s=50, c='k', alpha=0.3, marker=m, label=t)
    cbar = plt.colorbar(p1)
    cbar.ax.set_ylabel(r"$Damage \hspace{0.5} index$",fontsize=14)
    cbar.ax.yaxis.set_tick_params(labelsize=14)
    plt.legend(scatterpoints=1, frameon=False, labelspacing=1, title=r'$Magnitude$',loc="upper left",fontsize=14,title_fontsize=14) 
    plt.savefig(results_dir + '/3d_N_index.png',bbox_inches = 'tight')
    plt.close()

    fig = plt.figure(figsize = (10, 7))
    ax = plt.axes(projection ="3d")
    p1 = ax.scatter3D(n_low[:,0], n_low[:,1], n_low[:,2], c=mag_low[:,0], edgecolors='dimgray', alpha=1,  marker='^', s=50, cmap=color_map, vmin=0,vmax=v)
    p2 = ax.scatter3D(n_high[:,0], n_high[:,1], n_high[:,2], c=mag_high[:,0], edgecolors='dimgray', alpha=1,  marker='o', s=50, cmap=color_map, vmin=0,vmax=v)
    ax.set_xlabel(r"$Dimension \hspace{0.5} 1$", fontweight ='bold',fontsize=14)
    ax.set_ylabel(r"$Dimension \hspace{0.5} 2$", fontweight ='bold',fontsize=14)
    ax.set_zlabel(r"$Dimension \hspace{0.5} 3$", fontweight ='bold',fontsize=14)
    symbol = ['^','o']
    title = [r'$Low$', r'$High$']
    for t, m in zip(title,symbol):
        plt.scatter([], [], s=50, c='k', alpha=0.3, marker=m, label=t)
    cbar = plt.colorbar(p1)
    cbar.ax.set_ylabel(r"$Magnitude$",fontsize=14)
    cbar.ax.yaxis.set_tick_params(labelsize=14)
    plt.legend(scatterpoints=1, frameon=False, labelspacing=1, title=r'$Magnitude$',loc="upper left",fontsize=14,title_fontsize=14) 
    plt.savefig(results_dir + '/3d_N_mag.png',bbox_inches = 'tight')
    plt.close()

    fig = plt.figure(figsize = (10, 7))
    ax = plt.axes(projection ="3d")
    v_max = int(np.max(di[:]))
    
    p1 = ax.scatter3D(s[:,0], s[:,1], s[:,2], c=di[:,0], edgecolors='dimgray', alpha=1,  marker='o', s=50, cmap=color_map, vmin=0,vmax=v_max)
    p2 = ax.scatter3D(n[:,0], n[:,1], n[:,2], c=di[:,0], edgecolors='dimgray', alpha=1,  marker='^', s=50, cmap=color_map, vmin=0,vmax=v_max)
    ax.set_xlabel(r"$Dimension \hspace{0.5} 1$", fontweight ='bold',fontsize=14)
    ax.set_ylabel(r"$Dimension \hspace{0.5} 2$", fontweight ='bold',fontsize=14)
    ax.set_zlabel(r"$Dimension \hspace{0.5} 3$", fontweight ='bold',fontsize=14)
    symbol = ['o','^']
    title = [r'$S$', r'$N$']
    for t, m in zip(title,symbol):
        plt.scatter([], [], s=50, c='k', alpha=0.3, marker=m, label=t)
    cbar = plt.colorbar(p1)
    cbar.ax.set_ylabel(r"$Damage \hspace{0.5} index$",fontsize=14)
    cbar.ax.yaxis.set_tick_params(labelsize=14)
    plt.legend(scatterpoints=1, frameon=False, labelspacing=1, title=r'$Variable$',loc="upper left",fontsize=14,title_fontsize=14) 
    plt.savefig(results_dir + '/3d_S_N_index.png',bbox_inches = 'tight')
    plt.close()

    fig = plt.figure(figsize = (10, 7))
    ax = plt.axes(projection ="3d")
    v_max = int(np.max(mag[:]))
    
    p1 = ax.scatter3D(s[:,0], s[:,1], s[:,2], c=mag[:,0], edgecolors='dimgray', alpha=1,  marker='o', s=50, cmap=color_map, vmin=0,vmax=v_max)
    p2 = ax.scatter3D(n[:,0], n[:,1], n[:,2], c=mag[:,0], edgecolors='dimgray', alpha=1,  marker='^', s=50, cmap=color_map, vmin=0,vmax=v_max)
    ax.set_xlabel(r"$Dimension \hspace{0.5} 1$", fontweight ='bold',fontsize=14)
    ax.set_ylabel(r"$Dimension \hspace{0.5} 2$", fontweight ='bold',fontsize=14)
    ax.set_zlabel(r"$Damage \hspace{0.5}  Index$", fontweight ='bold',fontsize=14)
    symbol = ['o','^']
    title = [r'$S$', r'$N$']
    for t, m in zip(title,symbol):
        plt.scatter([], [], s=50, c='k', alpha=0.3, marker=m, label=t)
    cbar = plt.colorbar(p1)
    cbar.ax.set_ylabel(r"$Magnitude$",fontsize=14)
    cbar.ax.yaxis.set_tick_params(labelsize=14)
    plt.legend(scatterpoints=1, frameon=False, labelspacing=1, title=r'$Variable$',loc="upper left",fontsize=14,title_fontsize=14) 
    plt.savefig(results_dir + '/3d_S_N_mag.png',bbox_inches = 'tight')
    plt.close()

    return

def PlotChangeS(model,realXC,results_dir):

    realX = np.concatenate([x for x, c, m, d in realXC], axis=0)
    realC = np.concatenate([c for x, c, m, d in realXC], axis=0)
    mag = np.concatenate([m for x, c, m, d in realXC], axis=0)
    di = np.concatenate([d for x, c, m, d in realXC], axis=0)

    realS,realN,recS,recC,recN = model.cycling(realX,realC)

    labels_rec = np.zeros((recC.shape[0]))
    for i in range(recC.shape[0]):
        labels_rec[i] = np.argmax(recC[i,:])
    
    labels_real = np.zeros((realC.shape[0]))
    for i in range(realC.shape[0]):
        labels_real[i] = np.argmax(realC[i,:])

    labels_rec = labels_rec.astype(int)
    labels_real = labels_real.astype(int)

    target_names = []
    for i in range(options['latentCdim']):
        target_names.append('damage class %d'% i) 

    fig, ax = plt.subplots()
    report = classification_report(y_true = labels_real, y_pred = labels_rec,
            target_names=target_names,output_dict=True,zero_division=1)
    df = pd.DataFrame(report).transpose()
    df.to_csv(results_dir + '/ChangeS_ClassificationC.csv', index= True)
    cr = sn.heatmap(pd.DataFrame(report).iloc[:-1, :].T, annot=True, vmin=0, vmax=1, annot_kws={"size": 12})
    cr.tick_params(axis='both', labelsize=12)
    cr.set_yticklabels(cr.get_yticklabels(), rotation=0)
    plt.savefig(results_dir + '/ChangeS_ClassificationC.png',bbox_inches = 'tight')
    plt.close()

    conf_mat = confusion_matrix(labels_real, labels_rec)
    fig, ax = plt.subplots(figsize=(10,10),tight_layout=True)
    sn.heatmap(conf_mat, annot=True, fmt='d', xticklabels=target_names,yticklabels=target_names, vmin=0, vmax=realC.shape[0],
        annot_kws={"size": 20})
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=18)
    plt.tick_params(axis='both', labelsize=20)
    plt.ylabel("True class",fontsize=22,labelpad=10)
    plt.xlabel("Predicted class",fontsize=22,labelpad=10)
    plt.savefig(results_dir + '/ChangeS_ConfusionC.png',bbox_inches = 'tight')
    plt.close()

    transformerN1 = TSNE(n_components=2, verbose=1, random_state=123)
    n1 = transformerN1.fit_transform(realN)

    dfN1 = pd.DataFrame()
    dfN1["C"] = labels_real
    dfN1["Dimension 1"] = n1[:,0]
    dfN1["Dimension 2"] = n1[:,1]

    transformerN2 = TSNE(n_components=2, verbose=1, random_state=123)
    n2 = transformerN2.fit_transform(recN)

    dfN2 = pd.DataFrame()
    dfN2["C"] = labels_rec
    dfN2["Dimension 1"] = n2[:,0]
    dfN2["Dimension 2"] = n2[:,1]

    fig, ax = plt.subplots(1,2, figsize=(16,6))
    ax[0].set_title("N: T-SNE projection")
    sn.scatterplot(ax=ax[0], x="Dimension 1", y="Dimension 2", hue=dfN1.C.tolist(),palette=sn.color_palette("hls", 2),data=dfN1)

    ax[1].set_title(r"$F_x(G_z(c,s,n))$: T-SNE projection")
    sn.scatterplot(ax=ax[1], x="Dimension 1", y="Dimension 2", hue=dfN2.C.tolist(),palette=sn.color_palette("hls", 2),data=dfN2)
    plt.savefig(results_dir + '/ChangeS_tsne_N.png',bbox_inches = 'tight')
    plt.close()
    
    transformerS1 = TSNE(n_components=2, verbose=1, random_state=123)
    s1 = transformerS1.fit_transform(realS)

    dfS1 = pd.DataFrame()
    dfS1["C"] = labels_real
    dfS1["Dimension 1"] = s1[:,0]
    dfS1["Dimension 2"] = s1[:,1]

    transformerS2 = TSNE(n_components=2, verbose=1, random_state=123)
    s2 = transformerS2.fit_transform(recS)

    dfS2 = pd.DataFrame()
    dfS2["C"] = labels_rec
    dfS2["Dimension 1"] = s2[:,0]
    dfS2["Dimension 2"] = s2[:,1]

    fig, ax = plt.subplots(1,2, figsize=(16,6))
    ax[0].set_title("S: T-SNE projection")
    sn.scatterplot(ax=ax[0], x="Dimension 1", y="Dimension 2", hue=dfS1.C.tolist(),palette=sn.color_palette("hls", 2),data=dfS1)

    ax[1].set_title(r"$F_x(G_z(c,s,n))$: T-SNE projection")
    sn.scatterplot(ax=ax[1], x="Dimension 1", y="Dimension 2", hue=dfS2.C.tolist(),palette=sn.color_palette("hls", 2),data=dfS2)
    plt.savefig(results_dir + '/ChangeS_tsne_S.png',bbox_inches = 'tight')
    plt.close

    return

def PlotDistributions(model,realXC,results_dir):
    
    realX = np.concatenate([x for x, c, m, d in realXC], axis=0)
    realC = np.concatenate([c for x, c, m, d in realXC], axis=0)
    mag = np.concatenate([m for x, c, m, d in realXC], axis=0)
    di = np.concatenate([d for x, c, m, d in realXC], axis=0)

    realS, realN, fakeS, fakeN, recS, recN, Zmu, Zlogvar, Recmu, Reclogvar = model.distribution(realX,realC)

    realS_mean = tf.reduce_mean(realS,axis=0)
    fakeS_mean = tf.reduce_mean(fakeS,axis=0)
    recS_mean = tf.reduce_mean(recS,axis=0)

    Zσ2 = tf.exp(Zlogvar)

    fig, ax = plt.subplots()
    ax.scatter(Zmu, Zσ2,marker="o",label=None, alpha=0.8,edgecolors='w')
    ax.set_xlabel(r'$Z_{μ}$',fontsize=14)
    ax.set_ylabel(r'$Z_{σ2}$',fontsize=14)
    plt.savefig(results_dir + '/fakeS_sampling.png',bbox_inches = 'tight')
    plt.close()

    Recσ2 = tf.exp(Reclogvar)

    fig, ax = plt.subplots()
    ax.scatter(Recmu, Recσ2,marker="o",label=None, alpha=0.8,edgecolors='w')
    ax.set_xlabel(r'$Z_{μ}$',fontsize=14)
    ax.set_ylabel(r'$Z_{σ2}$',fontsize=14)
    plt.savefig(results_dir + '/recS_sampling.png',bbox_inches = 'tight')
    plt.close()


    fig, ax = plt.subplots(1,3, figsize=(24,6))
    ax[0].set_title(r"$Distribution \hspace{0.5} S$")
    ax[0].set_xlabel("Variable S")
    ax[0].set_ylabel("Density")
    sn.distplot(ax=ax[0], x = realS_mean, hist=True, kde=True, color = 'b', hist_kws={'edgecolor':'black'},kde_kws={'linewidth': 2})
    
    ax[1].set_title(r"$ Distribution \hspace{0.5} F_x(x)$")
    ax[1].set_xlabel("Variable S")
    ax[1].set_ylabel("Density")
    sn.distplot(ax=ax[1], x = fakeS_mean, hist=True, kde=True, color = 'g', hist_kws={'edgecolor':'black'},kde_kws={'linewidth': 2})

    ax[2].set_title(r"$ Distribution \hspace{0.5} F_x(G_z(c,s,n))$")
    ax[2].set_xlabel("Variable S")
    ax[2].set_ylabel("Density")
    sn.distplot(ax=ax[2], x = recS_mean, hist=True, kde=True, color = 'r', hist_kws={'edgecolor':'black'},kde_kws={'linewidth': 2})

    plt.savefig(results_dir + '/Distribution_Smean.png',bbox_inches = 'tight')
    plt.close()


    for i in range(10):
        j = randint(0, realX.shape[0]-1)

        fig, ax = plt.subplots(1,3, figsize=(24,6))
        ax[0].set_title(r"$Distribution \hspace{0.5} S$")
        ax[0].set_xlabel("Variable S")
        ax[0].set_ylabel("Density")
        sn.distplot(ax=ax[0], x = realS[j,:], hist=True, kde=True, color = 'b', hist_kws={'edgecolor':'black'},kde_kws={'linewidth': 2})
        
        ax[1].set_title(r"$ Distribution \hspace{0.5} F_x(x)$")
        ax[1].set_xlabel("Variable S")
        ax[1].set_ylabel("Density")
        sn.distplot(ax=ax[1], x = fakeS[j,:], hist=True, kde=True, color = 'g', hist_kws={'edgecolor':'black'},kde_kws={'linewidth': 2})

        ax[2].set_title(r"$ Distribution \hspace{0.5} F_x(G_z(c,s,n))$")
        ax[2].set_xlabel("Variable S")
        ax[2].set_ylabel("Density")
        sn.distplot(ax=ax[2], x = recS[j,:], hist=True, kde=True, color = 'r', hist_kws={'edgecolor':'black'},kde_kws={'linewidth': 2})

        plt.savefig(results_dir + '/Distribution_S_{:>d}.png'.format(j),bbox_inches = 'tight')
        plt.close()

    return


options = ParseOptions()

# MODEL LOADING
optimizers = RepGAN_losses.getOptimizers(**options)
losses = RepGAN_losses.getLosses(**options)

# Instantiate the RepGAN model.
GiorgiaGAN = RepGAN(options)

# Compile the RepGAN model.
GiorgiaGAN.compile(optimizers, losses)  # run_eagerly=True

#Xtrn, Xvld, _ = mdof.LoadData(**options)

# GiorgiaGAN.Fx = keras.models.load_model(options['checkpoint_dir'] + '/Fx',compile=False)
# GiorgiaGAN.Gz = keras.models.load_model(options['checkpoint_dir'] + '/Gz',compile=False)
# GiorgiaGAN.Dx = keras.models.load_model(options['checkpoint_dir'] + '/Dx',compile=False)
# GiorgiaGAN.Ds = keras.models.load_model(options['checkpoint_dir'] + '/Ds',compile=False)
# GiorgiaGAN.Dn = keras.models.load_model(options['checkpoint_dir'] + '/Dn',compile=False)
# GiorgiaGAN.Dc = keras.models.load_model(options['checkpoint_dir'] + '/Dc',compile=False)

for m in GiorgiaGAN.models:
    filepath= os.path.join(options["results_dir"], "{:>s}.h5".format(m.name))
    m.load_weights(filepath)


GiorgiaGAN.build(input_shape=(options['batchSize'], options['Xsize'], options['nXchannels']))


#load_status = GiorgiaGAN.load_weights("ckpt")

#latest = tf.train.latest_checkpoint(checkpoint_dir)
#print('restoring model from ' + latest)
#GiorgiaGAN.load_weights(latest)
#initial_epoch = int(latest[len(checkpoint_dir) + 7:])
GiorgiaGAN.summary()

if options['CreateData']:
    # Create the dataset
    Xtrn,  Xvld, _ = mdof.CreateData(**options)
else:
    # Load the dataset
    Xtrn, Xvld  = mdof.LoadData(**options)

PlotReconstructedTHs(GiorgiaGAN,Xvld,options['results_dir']) # Plot reconstructed time-histories

# PlotTHSGoFs(GiorgiaGAN,Xvld,options['results_dir']) # Plot reconstructed time-histories

# PlotClassificationMetrics(GiorgiaGAN,Xvld,options['results_dir']) # Plot classification metrics

# PlotLatentSpace(GiorgiaGAN,Xvld,options['results_dir'])

# PlotTSNE(GiorgiaGAN,Xvld,options['results_dir'])

# PlotDistributions(GiorgiaGAN,Xvld,options['results_dir'])

# Xtrn = {}
# Xvld = {}
# for i in range(options['latentCdim']):
#     Xtrn['Xtrn_%d' % i], Xvld['Xvld_%d' % i], _  = mdof.Load_Un_Damaged(i,**options)

# for i in range(options['latentCdim']):
#     PlotBatchGoFs(GiorgiaGAN,Xtrn['Xtrn_%d' % i],Xvld['Xvld_%d' % i],i,options['results_dir'])

# for i in range(1,options['latentCdim']):
#     PlotSwitchedTHs(GiorgiaGAN,Xvld['Xvld_%d' % 0],Xvld['Xvld_%d' % i],i,options['results_dir']) # Plot switched time-histories



import detect_peaks as dp
import nanonis
#from modules import functions
import distributions
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from matplotlib import rc
from scipy.signal import savgol_filter
import glob
import colorcet as cc
import csv
import numpy as np
import pickle
from lmfit.models import LorentzianModel, ConstantModel, GaussianModel, PolynomialModel
plt.rcParams.update({'font.size': 12})
plt.rcParams["figure.figsize"] = [10,5]
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('font',**{'family':'serif','serif':['Helvetica']})
# plt.switch_backend('Qt5Agg')
colour_code =cc.glasbey_category10


########################################
#README
#This package allows multi fit of several spectras.
#You can use it if you want to follow a peak shifting across different spectra.
# How to:
#1) gather filenames in a filenames list
#2) convert them with dataTodf(filenames) procedure
#3) use one of the fit_plot type of functions. Input all the parameters and the fit will be executed.
#4) the output is a list of fit output (lmfit style), you can extract than the parameters you need with one of the extract functions.
##########################################


spectra = nanonis.biasSpectroscopy()

def find_nearest_idx(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx
def cutSpec(x,y,center,interval):
    x = np.array(x)
    y = np.array(y)
    startidx = find_nearest_idx(x,center+interval)
    endidx = find_nearest_idx(x,center-interval)
    x = x[startidx:endidx]
    y = y[startidx:endidx]
    return x, y
def save_obj(obj, name ):
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL) 
def load_obj(name ):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)
#converts a series of spectra (from filenames) to a dataframe compatible with the fitting modules
def specTodf(filenames):
    data = pd.DataFrame()
    count = 0
    for filename in filenames:
        spectra.load(filename)
        if count == 0:
            data[0]=spectra.bias*1e3
        data[count+1]=spectra.conductance
        count+=1
    return data
###################################################################
################ FUNCTIONS DEFINITIONS #############################
####################################################################
def fitFunc1g_const(x,y,center,sigma): #1 gaussian with constant background
    model = GaussianModel() + ConstantModel(prefix='bkg_')
    params = model.make_params()
    params['center'].set(center)
    params['amplitude'].set(1,min=0)
    params['sigma'].set(sigma)
    params['sigma'].set(vary=True)
    params['bkg_c'].set(2)
    result=model.fit(y,params,x=x)
    return result

def fitFunc1g_para(x,y,center,sigma): #1 gaussian with parabolic background
    model = GaussianModel() + PolynomialModel(2,prefix='bkg_')
    params = model.make_params()
    params['center'].set(center)
    params['amplitude'].set(1,min=0)
    params['sigma'].set(sigma)
    params['sigma'].set(vary=True)
    params['bkg_c0'].set(2)
    params['bkg_c1'].set(2)
    params['bkg_c2'].set(2)
    result=model.fit(y,params,x=x)
    return result

def fitFunc1g_lin(x,y,center,sigma): #1 gaussian with parabolic background
    model = GaussianModel() + PolynomialModel(2,prefix='bkg_')
    params = model.make_params()
    params['center'].set(center)
    params['amplitude'].set(1,min=0)
    params['sigma'].set(sigma)
    params['bkg_c0'].set(2)
    params['bkg_c1'].set(0)
    params['bkg_c2'].set(0,vary=False)
    result=model.fit(y,params,x=x)
    return result

## double peak fit for shiba n1 particle and hole
def fitFunc2g_para(x,y,c_out,c_in,sigma): #2 gaussians with parabolic background
    model = GaussianModel(prefix='out_')+ GaussianModel(prefix='in_') + PolynomialModel(2)
    
    params = model.make_params()
    ## Initialize parameters peak1
    params['out_center'].set(vary=True)
    params['out_center'].set(c_out)
    params['out_amplitude'].set(3, min=0)
    params['out_sigma'].set(sigma)
    params['out_sigma'].set(vary=False)
    ## Initialize parameters peak2
    params['in_center'].set(c_in)
    params['in_amplitude'].set(3, min=0)
    params['in_sigma'].set(sigma)
    params['in_sigma'].set(vary=False)
    ## Initialize parameters constant baseline
    params['c0'].set(2)
    params['c1'].set(2)
    params['c2'].set(5,vary=False)
    result=model.fit(y,params,x=x)
    return result

def fitFunc2g_const(x,y,c_out,c_in,sigma): #2 gaussians with constant background
    model = GaussianModel(prefix='out_')+ GaussianModel(prefix='in_') + PolynomialModel(1)
    
    params = model.make_params()
    ## Initialize parameters peak1
    params['out_center'].set(c_out)
    params['out_amplitude'].set(3, min=0)
    params['out_sigma'].set(sigma)
    params['out_sigma'].set(vary=True)
    ## Initialize parameters peak2
    params['in_center'].set(c_in)
    params['in_amplitude'].set(3, min=0)
    params['in_sigma'].set(sigma)
    params['in_sigma'].set(vary=True)
    ## Initialize parameters constant baseline
    params['c0'].set(2)
    params['c1'].set(2)
    result=model.fit(y,params,x=x)
    return result

def fitFunc2l_const(x,y,c_out,c_in,sigma): #2 gaussians with constant background
    model = LorentzianModel(prefix='out_')+ LorentzianModel(prefix='in_') + PolynomialModel(1)
    
    params = model.make_params()
    ## Initialize parameters peak1
    params['out_center'].set(c_out)
    params['out_amplitude'].set(3, min=0)
    params['out_sigma'].set(sigma)
    params['out_sigma'].set(vary=True)
    ## Initialize parameters peak2
    params['in_center'].set(c_in)
    params['in_amplitude'].set(3, min=0)
    params['in_sigma'].set(sigma)
    params['in_sigma'].set(vary=True)
    ## Initialize parameters constant baseline
    params['c0'].set(2)
    params['c1'].set(2)
    result=model.fit(y,params,x=x)
    return result
####################################################################################################
######################### FUNCTIONS TO PERFORM AND PLOT THE FIT ####################################
####################################################################################################
def fit_plotfull(data,energy, interval,sigma,spac,bkg,plotype='fullPlot',plotGauss=True):
    fig, ax = plt.subplots(1,figsize=[8,3])
    i = 0
    multifit = []
    ######### FITTING ###########
    for column in data.columns:
        if column == 0:
            continue
        x , y = cutSpec(data[0],data[column],energy,interval)
        if bkg == 'para':
            fit_res = fitFunc1g_para(x,y,energy,sigma)
        elif bkg == 'lin':
            fit_res = fitFunc1g_lin(x,y,energy,sigma)
        else:
            fit_res = fitFunc1g_const(x,y,energy,sigma)
        ######PLOT RESULTS ###########
        if plotype == 'fullPlot':
            ax.plot(data[0],data[column]+i*spac)
        elif plotype == 'cutPlot':
            ax.plot(x,y+i*spac) #plot the spectras (ranges)
        xx=np.linspace(fit_res.params['center'].value-interval,fit_res.params['center'].value +interval,1000)
        ax.plot(xx,fit_res.eval(x=xx)+i*spac,color='r')
        comps = fit_res.eval_components() #for plotting the unconvoluted gaussians
        if plotGauss == True:
            for name, comp in comps.items():
                if 'bkg_c' in name:
                    continue
                ax.plot(x,comp+i*spac,color='b', label=name)
        i += 1
        
        multifit.append(fit_res.params)
    return multifit


## single peak fit with a set of initial energies
def fit_plot_Einput(data,energy, interval,sigma,spac,para,plotype):
    fig, ax = plt.subplots(1,figsize=[2,8])
    temp = []
    temp_S = []
    i = 0

    ######### FITTING ###########
    for column in data.columns:
        if column == 0:
            continue
        x , y = cutSpec(data[0],data[column],energy[i],interval)
        if para == True:
            fit_res = fitFunc1g_para(x,y,energy[i],sigma)
        else:
            fit_res = fitFunc1g_const(x,y,energy[i],sigma)
        ######PLOT RESULTS ###########
        if plotype == 'fullPlot':
            ax.plot(data[0],data[column]+i*spac)
        elif plotype == 'cutPlot':
            ax.plot(x,y+i*spac) #plot the spectras (ranges)
        xx=np.linspace(fit_res.params['center'].value-interval,fit_res.params['center'].value +interval,1000)
        ax.plot(xx,fit_res.eval(x=xx)+i*spac,color='r')
        i += 1
        ########### SAVE DATA ##########
        if fit_res.params['center'].stderr == None:
            fit_res.params['center'].stderr = 0
        temp.append(fit_res.params['center'].value)
        temp_S.append(fit_res.params['center'].stderr)
    return temp, temp_S



### centralE is the initial energy of the peaks (or the middle between two peaks)
### interval energy range you want to fit centered in centralE


#fit with 2 gaussian with linear or parabolic background returning all parameters
def fit_plot2gfull(data,centralE,c_out,c_in, interval,sigma,spac,plotype='fullPlot',plotGauss=True,plotfit=True,background='gauss_constant'):
    fig, ax = plt.subplots(1,figsize=[10,5])
    i=0
    multifit = []
    for column in data.columns:
        if column == 0: #jump if the column is bias
            continue
        x , y = cutSpec(data[0],data[column],centralE,interval) #cut the spectra

        if background == 'gauss_parabolic': #decide the background to use
            fit_res = fitFunc2g_para(x,y,c_out,c_in,sigma) #do the fit
        elif background == 'gauss_constant':
            fit_res = fitFunc2g_const(x,y,c_out,c_in,sigma) #do the fit
        elif background == 'lorentz_constant':
            fit_res = fitFunc2l_const(x,y,c_out,c_in,sigma)
        else:
            break

        comps = fit_res.eval_components() #for plotting the unconvoluted gaussians

        ####plot things
        if plotype == 'fullPlot':
            ax.plot(data[0],data[column]+i*spac)
        elif plotype == 'cutPlot':
            ax.plot(x,y+i*spac) #plot the spectras (ranges)
        xx=np.linspace(centralE-interval,centralE+interval,1000) #create an interpolated bias coordinate to plot the fit
        if plotfit==True:
            ax.plot(xx,fit_res.eval(x=xx)+i*spac,color='g')  #plot the fits
           ## plot the separated gaussians  
        if plotGauss == True:
            for name, comp in comps.items():
                if 'c' in name:
                    continue
                if 'in' in name:
                    color='b'
                if 'out' in name:
                    color='r'
                ax.plot(x,comp+i*spac,color=color, label=name)
        i += 1
        ######## save results checking for nones values
        if fit_res.params['out_center'].stderr == None:
            fit_res.params['out_center'].stderr = 0
        if fit_res.params['in_center'].stderr == None:
            fit_res.params['in_center'].stderr = 0
        multifit.append(fit_res.params)
    return multifit


####################################################################################################
#################### USEFUL FUNCTIONS CORRELATED WITH THIS PACKAGE #################################
####################################################################################################
def plotshibafit(data,sets,offset): #data contains the values of the fits, sets is a list of column names in string format
    fig, ax = plt.subplots(1,figsize=[5,5])
    plt.rcParams.update({'font.size': 12})
    i=0
    for item in sets:
        if i!=0:
            offset=0 
        ax.errorbar(data[item]+offset,data['d'],data[item + '_S'],marker='o', markersize=8, capsize=5)
        i+=1
    ax.set_xlabel('Bias (mV)')
    ax.set_ylabel('Distance (nm)')
    plt.tight_layout()


#to extract peak data from packed fit objects


#to extract parameters from list of fit results
def fitParamExtract2g(fit_result,param,in_or_out='in'):
    param_V = []
    param_S = []
    for i in fit_result:
        param_V.append(i[in_or_out+'_'+param].value)
        param_S.append(i[in_or_out+'_'+param].stderr)
    param_V = np.array(param_V)
    param_S = np.array(param_S)
    return param_V, param_S

#to extract parameters from list of fit results
def fitParamExtract1g(fit_result,param):
    param_V = []
    param_S = []
    for i in fit_result:
        param_V.append(i[param].value)
        param_S.append(i[param].stderr)
    param_V = np.array(param_V)
    param_S = np.array(param_S)
    return param_V, param_S

def E_extract(raw_data, peak):
    energy = []
    energy_S = []
    count = 0
    for i in raw_data[peak]:
        energy.append(i['center'].value)
        energy_S.append(i['center'].stderr)
        count += 1
    return energy, energy_S

def E_extract_2g(raw_data,peak,in_or_out):
    energy = []
    energy_S = []
    count = 0
    for i in raw_data[peak]:
        energy.append(i[in_or_out + 'center'].value)
        energy_S.append(i[in_or_out + 'center'].stderr)
    return energy, energy_S




    ############# OLD CODE ####################
''' def fit_plot2g(data,centralE,c_out,c_in, interval,sigma,spac,plotype,plotGauss,plotfit,background):
    fig, ax = plt.subplots(1,figsize=[2,16])
    i=0
    multifit = []
    for column in data.columns:
        if column == 0: #jump if the column is bias
            continue
        x , y = cutSpec(data[0],data[column],centralE,interval) #cut the spectra

        if background == 'para': #decide the background to use
            fit_res = fitFunc2g_para(x,y,c_out,c_in,sigma) #do the fit
        elif background == 'constant':
            fit_res = fitFunc2g_const(x,y,c_out,c_in,sigma) #do the fit
        else:
            break

        comps = fit_res.eval_components() #for plotting the unconvoluted gaussians

        ####plot things
        if plotype == 'fullPlot':
            ax.plot(data[0],data[column]+i*spac)
        elif plotype == 'cutPlot':
            ax.plot(x,y+i*spac) #plot the spectras (ranges)
        xx=np.linspace(centralE-interval,centralE+interval,1000) #create an interpolated bias coordinate to plot the fit
        if plotfit==True:
            ax.plot(xx,fit_res.eval(x=xx)+i*spac,color='r')  #plot the fits
           ## plot the separated gaussians  
        if plotGauss == True:
            for name, comp in comps.items():
                if 'c' in name:
                    continue
                if 'in' in name:
                    color='b'
                if 'out' in name:
                    color='r'
                ax.plot(x,comp+i*spac,color=color, label=name)
        i += 1
        ######## save results checking for nones values
        if fit_res.params['out_center'].stderr == None:
            fit_res.params['out_center'].stderr = 0
        if fit_res.params['in_center'].stderr == None:
            fit_res.params['in_center'].stderr = 0
        multifit.append(fit_res.params)
    return multifit



def fit_plot(data,energy=0, interval=0,sigma=0,spac=0,bkg='lin',plotype='fullPlot',plotGauss=True):
    fig, ax = plt.subplots(1,figsize=[8,3])
    cent = []
    cent_S = []
    amp = []
    amp_S = []
    i = 0

    ######### FITTING ###########
    for column in data.columns:
        if column == 0:
            continue
        x , y = cutSpec(data[0],data[column],energy,interval)
        if bkg == 'para':
            fit_res = fitFunc1g_para(x,y,energy,sigma)
        elif bkg == 'lin':
            fit_res = fitFunc1g_lin(x,y,energy,sigma)
        else:
            fit_res = fitFunc1g_const(x,y,energy,sigma)
        ######PLOT RESULTS ###########
        if plotype == 'fullPlot':
            ax.plot(data[0],data[column]+i*spac)
        elif plotype == 'cutPlot':
            ax.plot(x,y+i*spac) #plot the spectras (ranges)
        xx=np.linspace(fit_res.params['center'].value-interval,fit_res.params['center'].value +interval,1000)
        ax.plot(xx,fit_res.eval(x=xx)+i*spac,color='r')
        comps = fit_res.eval_components() #for plotting the unconvoluted gaussians
        if plotGauss == True:
            for name, comp in comps.items():
                if 'bkg_c' in name:
                    continue
                ax.plot(x,comp+i*spac,color='b', label=name)
        i += 1
        ########### SAVE DATA ##########
        if fit_res.params['center'].stderr == None:
            fit_res.params['center'].stderr = 0
        cent.append(fit_res.params['center'].value)
        cent_S.append(fit_res.params['center'].stderr)
        amp.append(fit_res.params['amplitude'].value)
        amp_S.append(fit_res.params['amplitude'].stderr)        
    return cent, cent_S, amp, amp_S
'''
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 18:25:54 2020

@author: mibsspc2
"""

import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.colors as mcolors
import pandas as pd 
import random
import math
import time
#from sklearn.linear_model import LinearRegression, BayesianRidge
#from sklearn.model_selection import RandomizedSearchCV, train_test_split
#from sklearn.svm import SVR
#from sklearn.metrics import mean_squared_error, mean_absolute_error
from datetime import timedelta, datetime
import operator
plt.style.use('seaborn-darkgrid')
import seaborn as sn
#%%
# Retrieve world-wide data
reload = True
if 'reload' not in locals() or reload:
    confirmed_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Confirmed.csv')
    deaths_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Deaths.csv')
    recoveries_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Recovered.csv')
    reload = False
cols = confirmed_df.keys()
confirmed = confirmed_df.loc[:, cols[4]:cols[-1]]
deaths = deaths_df.loc[:, cols[4]:cols[-1]]
recoveries = recoveries_df.loc[:, cols[4]:cols[-1]]
#%%
# Confirmed cases in Spain
ix = confirmed_df['Country/Region'] == 'Spain'
dates = confirmed.keys()

start = '1/22/20'
start_date = datetime.strptime(start, '%m/%d/%y')
period = []
days = []
for i in range(len(dates)):
    day = datetime.strptime(dates[i], '%m/%d/%y')
    days.append(day)
    period.append( (day - start_date).days )
days = np.array(days)
conf = np.array(confirmed.loc[ix,dates])[0]
dea = np.array(deaths.loc[ix,dates])[0]
#%%

plt.plot(period,conf,marker='o' )
plt.xlabel('Day')
plt.ylabel('Confirmed')
plt.show()
#%%
# Model predictions
dmol = [
['12/03/2020',	'Previsión mínima de reportados 3477 4688 6729 Previsión media de reportados 4763 7732 12495 Previsión máxima de reportados 6059 11454 20862'],
['13/03/2020',	'Previsión mínima de reportados 4886 6423 8860 Previsión media de reportados 6558 10221 15886 Previsión máxima de reportados 8230 14896 25609'],
['14/03/2020',	'Previsión mínima de reportados 6613 8553 11742 Previsión media de reportados 8806 13490 20681 Previsión máxima de reportados 10999 19612 32953'],
['15/03/2020',	'Previsión mínima de reportados 7944 9069 11190 Previsión media de reportados 10500 14258 19384 Previsión máxima de reportados 13082 20404 30602'],
['16/03/2020',	'Previsión mínima de reportados 9343 10624 12533 Previsión media de reportados 12149 16028 21146 Previsión máxima de reportados 14919 22635 32695'],
['17/03/2020',	'Previsión mínima de reportados 11184 12203 13990 Previsión media de reportados 13950 17390 21664 Previsión máxima de reportados 16714 23431 31539'],
['18/03/2020',	'Previsión mínima de reportados 13638 14759 16688 Previsión media de reportados 16843 20687 25415 Previsión máxima de reportados 20043 27574 36406'],
['19/03/2020',	'Previsión mínima de reportados 17333 19472 23124 Previsión media de reportados 22413 29348 38417 Previsión máxima de reportados 27482 41139 58613'],
['20/03/2020',	'Previsión mínima de reportados 19536 21168 23975 Previsión media de reportados 24150 29691 36514 Previsión máxima de reportados 28739 39638 52395'],
['21/03/2020', 'Previsión mínima de reportados 24488 26986 31445 Previsión media de reportados 30961 39327 49897 Previsión máxima de reportados 37435 53834 73963']
]
preds = pd.DataFrame(dmol, columns= ['Date','Data'])
#preds = pd.read_excel('preds.xlsx')
#%%
# Load prediction data
dpred = []
for i in np.arange(preds.shape[0]):
    data = preds.iloc[i,1].split('Prev')
    vals = []
    vals.append( [int(x) for x in data[1].split(' ')[-4:-1]] )
    vals.append( [int(x) for x in data[2].split(' ')[-4:-1]] )
    vals.append( [int(x) for x in data[3].split(' ')[-3:]] )
    dpred.append( vals )
dpred = np.array(dpred)
ddays = []
for i in np.arange(preds.shape[0]):
#    ds = "{}/{}/{}".format( preds.iloc[i,0].day, preds.iloc[i,0].month, preds.iloc[i,0].year)
#    day = datetime.datetime.strptime(ds, '%d/%m/%Y')
    day = datetime.strptime(preds.iloc[i,0],'%d/%m/%Y')
    ddays.append(day)
ddays = np.array(ddays)

#%%
# Calculate prediction errors
shift = np.argwhere( ddays[0]  == days )
delay = False
ptype = 1
error1 = []
if delay:
    error2 = [None]
    error3 = [None,None]
else:
    error2 = []
    error3 = []
logo = False
if logo:
    conf1 = np.log10( conf )
    dpred1 = np.log10( dpred )
else:
    conf1 = conf
    dpred1 = dpred
for i in np.arange(shift,len(days)-1):
    try:
        d = 1
        py = float( dpred1[i-shift,ptype,d-1] )
        y = conf1[i+d]
        val = y - py
        val = 100*val/y
        error1.append( val )
        d = 2
        py = float( dpred1[i-shift,ptype,d-1] )
        y = conf1[i+d]
        val = y - py
        val = 100*val/y
        error2.append( val )
        d = 3
        py = float( dpred1[i-shift,ptype,d-1] )
        y = conf1[i+d]
        val = y - py
        val = 100*val/y
        error3.append( val )
    except:
        continue
error1 = np.array(error1)
error2 = np.array(error2)
error3 = np.array(error3)
plt.plot(error1, marker='o')
plt.plot(error2, marker='o')
plt.plot(error3, marker='o')
plt.legend([1,2,3])
plt.grid()
plt.xlabel('Day')
plt.ylabel('% prediction error')

#%%
if delay:
    w = np.array([error1[2:],error2[2:], error3[2:]])
else:
    w = np.array([error1[2:],error2[1:], error3[0:]])
    
print( np.corrcoef( w.astype(float) ) )

#%%

# https://www.lewuathe.com/covid-19-dynamics-with-sir-model.html

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from datetime import timedelta, datetime

START_DATE = {
  'Japan': '1/22/20',
  'Italy': '2/15/20',
  'Republic of Korea': '1/22/20',
  'Iran (Islamic Republic of)': '2/19/20',
  'Spain': '3/1/20',
  'France':'3/15/20'
}

# Total population, N.
country = 'Spain'
ix = confirmed_df['Country/Region'] == country

POPULATION = {
        'Italy' : 60*1000000.0,
        'Spain' : 45*1000000.0,
        'France': 67*1e6
        }
N = POPULATION[country]
# Initial number of infected and recovered individuals, I0 and R0.

incubation_days = 1
incub = int( np.argwhere( confirmed_df.keys() == START_DATE[country] ) ) - incubation_days
I_0 = confirmed_df[ix].iloc[0].loc[START_DATE[country]]
R_0 = recoveries_df[ix].iloc[0].loc[START_DATE[country]]
R_0 = I_0-deaths_df[ix].iloc[0].loc[START_DATE[country]]

I_0 = confirmed_df[ix].iloc[0].loc[START_DATE[country]] - confirmed_df[ix].iloc[0].iloc[incub]
R_0 = I_0-deaths_df[ix].iloc[0].loc[START_DATE[country]] + deaths_df[ix].iloc[0].iloc[incub]

# Everyone else, S0, is susceptible to infection initially.
S_0 = N - I_0 - R_0
#S_0 = S_0/N
#I_0 = I_0/N
#R_0 = R_0/N
class Learner(object):
    def __init__(self, country, loss, confirmed=None, plot= False, verbose=False ):
        self.country = country
        self.loss = loss
        self.confirmed = confirmed
        self.plot = plot
        self.verbose = verbose

    def load_confirmed(self, country):
      """
      Load confirmed cases downloaded from HDX
      """
      df = pd.read_csv('data/time_series_19-covid-Confirmed.csv')
      country_df = df[df['Country/Region'] == country]
      return country_df.iloc[0].loc[START_DATE[country]:]
  
    def load_confirmed_pablo(self, country, confirmed_df, incubation=True):
        ix = confirmed_df['Country/Region'] == country
        conf = confirmed_df[ix].iloc[0].loc[START_DATE[country]:]
        if incubation:
            incub = int( np.argwhere( confirmed_df.keys() == START_DATE[country] ) ) - incubation_days
            conf1 = conf.copy()
            for i in np.arange(len(conf)):
                conf1.iloc[i] = conf.iloc[i] - confirmed_df[ix].iloc[0].iloc[incub+i] 
            conf = conf1
        return conf

    def extend_index(self, index, new_size):
        values = index.values
        current = datetime.strptime(index[-1], '%m/%d/%y')
        while len(values) < new_size:
            current = current + timedelta(days=1)
            values = np.append(values, datetime.strftime(current, '%m/%d/%y'))
        return values

    def predict(self, beta, gamma, data):
        """
        Predict how the number of people in each compartment can be changed through time toward the future.
        The model is formulated with the given beta and gamma.
        """
        predict_range = 60
        new_index = self.extend_index(data.index, predict_range)
        size = len(new_index)
        def SIR(t, y):
            S = y[0]
            I = y[1]
            R = y[2]
            return [-beta*S*I, beta*S*I-gamma*I, gamma*I]
        extended_actual = np.concatenate((data.values, [None] * (size - len(data.values))))
        return new_index, extended_actual, solve_ivp(SIR, [0, size], [S_0,I_0,R_0], t_eval=np.arange(0, size, 1))

    def predict_pablo(self, beta, gamma, data):
        """
        Predict how the number of people in each compartment can be changed through time toward the future.
        The model is formulated with the given beta and gamma.
        """
        predict_range = 60
        new_index = self.extend_index(data.index, predict_range)
        size = len(new_index)
        def dSIR(t, y):
            S = y[0]
            I = y[1]
            R = y[2]
            return [S-beta*S*I/N, I+beta*S*I/N - gamma*I, R+gamma*I]
        extended_actual = np.concatenate((data.values, [None] * (size - len(data.values))))
        return new_index, extended_actual, dsolve_ivp(dSIR, t_eval=[0, size], y0=[S_0,I_0,R_0])



    def train(self,pablo=True):
        """
        Run the optimization to estimate the beta and gamma fitting the given confirmed cases.
        """
        if pablo:      
            data = self.load_confirmed_pablo(self.country, self.confirmed)
        else:
            data = self.load_confirmed(self.country) 
        self.data = data
        optimal = minimize(
            self.loss,
#            [0.001, 0.001],
            [1/S_0, R_0/I_0],
            args=(data),
            method='Powell',
#            method='L-BFGS-B',
 #           bounds=[(0.00000001, 0.4), (0.00000001, 0.4)],
            bounds=[(1e-3, 10), (0.1, 10)],
            options={'maxiter':1000000,'disp':True,'ftol':1e-6}
        )
        beta, gamma = optimal.x
        new_index, extended_actual, prediction = self.predict_pablo(beta, gamma, data)
        df = pd.DataFrame({
            'Actual': extended_actual,
            'S': prediction[:,0],
            'I': prediction[:,1],
            'R': prediction[:,2]
        }, index=new_index)
#        df = df*N
#        df[['I','Actual']].plot(ax=ax)
        if self.plot:
            fig, ax = plt.subplots(figsize=(15, 10))
            ax.set_title(self.country)
            plt.plot(df[['I','Actual']])
            plt.xticks(rotation=90)
            plt.legend(['Infected (predicted)','Infected (actual)'])
#        plt.grid()
        self.df = df
        self.beta = beta
        self.gamma = gamma

def loss(point, data):
    """
    RMSE between actual confirmed cases and the estimated infectious people with given beta and gamma.
    """
    size = len(data)
    beta, gamma = point
    def SIR(t, y):
        S = y[0]
        I = y[1]
        R = y[2]
        return [-beta*S*I, beta*S*I-gamma*I, gamma*I]
    solution = solve_ivp(SIR, [0, size], [S_0,I_0,R_0], t_eval=np.arange(0, size, 1), vectorized=True)
    return np.sqrt(np.mean((solution.y[1] - data)**2))

def dsolve_ivp(fun, t_eval, y0):
        y = y0
        sol = []
        for t in np.arange(t_eval[0], t_eval[1]):
            y = fun(t,y)
            sol.append(y)
        sol = np.array(sol)
        return sol



def loss_pablo(point, data):
    """
    RMSE between actual confirmed cases and the estimated infectious people with given beta and gamma.
    Using a discrete model
    """
    size = len(data)
    beta, gamma = point
    def dSIR(t, y):
        S = y[0]
        I = y[1]
        R = y[2]
        return [S-beta*S*I/N, I+beta*S*I/N - gamma*I, R+gamma*I]
    sol = dsolve_ivp(dSIR, t_eval=[0, size], y0=[S_0,I_0,R_0])
    rmsd = np.sqrt(np.mean((sol[:,1] - data)**2))
    return rmsd

   


#%%
learn = Learner(country, loss_pablo, confirmed_df, plot=True)
learn.train()


#%%
""" A test about model evolution """

st = START_DATE[country]
vals = []
win = 20
for i in np.arange(win):
    j = -win+i
    learn = Learner(country, loss_pablo, confirmed_df.iloc[:,:j])
    learn.train()
    vals.append( (confirmed_df.columns[j], learn.beta, learn.gamma) )

#%%
    
vals_df = pd.DataFrame(vals, columns=['Date','beta', 'gamma'])
plt.plot(vals_df['Date'], vals_df[['beta','gamma']], marker='o')

plt.plot(vals_df['Date'], vals_df['beta']/vals_df['gamma'], marker='o')
plt.legend(['Average contacts', 'Recovered rate', 'Reproduction'])
plt.xticks(rotation=90)
#plt.yscale("log")





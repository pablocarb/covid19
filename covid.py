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
import sklearn.metrics as metrics
#from sklearn.linear_model import LinearRegression, BayesianRidge
#from sklearn.model_selection import RandomizedSearchCV, train_test_split
#from sklearn.svm import SVR
#from sklearn.metrics import mean_squared_error, mean_absolute_error
from datetime import timedelta, datetime
import operator
plt.style.use('seaborn-darkgrid')
import seaborn as sn
import os
#%%
# Retrieve world-wide data


reload = True
if 'reload' not in locals() or reload:
    confirmed_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
    deaths_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')
    recoveries_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv')
    reload = False
cols = confirmed_df.keys()
recoveries_df.columns = np.concatenate( [recoveries_df.columns[0:4], cols[4:]] )
confirmed = confirmed_df.loc[:, cols[4]:]
cols = deaths_df.keys()
deaths = deaths_df.loc[:, cols[4]:]
cols = recoveries_df.keys()
recoveries = recoveries_df.loc[:, cols[4]:]
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
  'Italy': '2/25/20',
  'Korea, South': '1/22/20',
  'Iran (Islamic Republic of)': '2/10/20',
  'Spain': '2/15/20',
  'France':'3/1/20',
  'India': '3/1/20',  
  'United Kingdom': '1/22/20',
  'US': '1/22/20'

}

def align(df,country):
    return np.logical_and( df['Country/Region'] == country, pd.isna(df['Province/State']) )

country = 'Korea, South'
    # Total population, N.
    
POPULATION = {
    'Italy' : 60*1000000.0,
    'Spain' : 45*1000000.0,
    'France': 67*1e6,
    'Korea, South': 52*1e6,
    'India': 1.35e3*1e6,
    'United Kingdom': 66.44*1e6,
    'US': 327.2*1e6
   }
   

def load_data(country, incubation=True, incubation_days=14):
    # Initial number of infected and recovered individuals, I0 and R0.
    
    start = int( np.argwhere( confirmed_df.keys() == START_DATE[country] ) )
    incub = start - incubation_days
    ix = align(confirmed_df,country)
    confirmed = confirmed_df.loc[ix,START_DATE[country]:].iloc[0]
    ix = align(deaths_df,country)
    deaths = deaths_df.loc[ix,START_DATE[country]:].iloc[0]
    ix = align(recoveries_df,country)
    recoveries = recoveries_df.loc[ix,START_DATE[country]:].iloc[0]
    confirmed_delta = confirmed.copy()
    for i in confirmed_delta.index:
        confirmed_delta[i] = 0
    if incubation:
        for i in np.arange(len(confirmed)):        
            d = i + start - incubation_days
            if d >= 4:
                ix = align(confirmed_df,country)
                confirmed[i] = confirmed[i] - recoveries[i] - deaths[i] - confirmed_df.loc[ix].iloc[0,d] 
                confirmed_delta[i] = confirmed_df.loc[ix].iloc[0,d]
                ix = align(deaths_df,country)
                deaths[i] = deaths[i] - deaths_df.loc[ix].iloc[0,d]
                ix = align(recoveries_df,country)
                recoveries[i] = recoveries[i] - recoveries_df.loc[ix].iloc[0,d]
    
    return confirmed, deaths, recoveries, confirmed_delta

confirmed, deaths, recoveries, confirmed_delta = load_data(country)

#%%
N = POPULATION[country]
I_0 = confirmed[0]
R_0 = recoveries[0]
# Everyone else, S0, is susceptible to infection initially.
S_0 = N - I_0 - R_0
#S_0 = S_0/N
#I_0 = I_0/N
#R_0 = R_0/N

class Learner(object):
    def __init__(self, country, population, confirmed, recoveries, deaths, loss, plot= False, verbose=False ):
        self.country = country
        self.loss = loss
        self.confirmed = confirmed
        self.recoveries = recoveries
        self.population = population
        I_0 = confirmed[0]
        R_0 = recoveries[0] + deaths[0]
        S_0 = population - I_0 - R_0
        self.y0 = [S_0, I_0, R_0]
        self.plot = plot
        self.verbose = verbose


    def extend_index(self, index, new_size):
        values = index.values
        current = datetime.strptime(index[-1], '%m/%d/%y')
        last = current + timedelta(days=new_size)
        while current != last:
            current = current + timedelta(days=1)
            values = np.append(values, datetime.strftime(current, '%m/%d/%y'))
        return values


    def predict(self, beta, gamma, data, predict_range=30):
        """
        Predict how the number of people in each compartment can be changed through time toward the future.
        The model is formulated with the given beta and gamma.
        """
        new_index = self.extend_index(data.index, predict_range)
        size = len(new_index)
        extended_actual = np.concatenate((data.values, [None] * (size - len(data.values))))
        return new_index, extended_actual, dsolve_ivp(dSIR, t_eval=[0, size], y0=self.y0, beta=beta, gamma=gamma)


    def train(self):
        """
        Run the optimization to estimate the beta and gamma fitting the given confirmed cases.
        """
        data = self.confirmed
        y0 = self.y0
        self.data = data
        optimal = minimize(
            self.loss,
#            [0.001, 0.001],
            [0.01, 0.01*10],
            args=[data, y0],
            method='Powell',
#            method='Powell',
#            method='L-BFGS-B',
 #           bounds=[(0.00000001, 0.4), (0.00000001, 0.4)],
#            bounds=[(1e-3, 10), (1e-3, 10)],
            options={'maxiter':100000000,'disp':self.verbose,'ftol':1e-12,'xtol':1e-12}
        )
        beta, gamma = optimal.x
        new_index, extended_actual, prediction = self.predict(beta, gamma, data)
        df = pd.DataFrame({
            'Actual': extended_actual,
            'S': prediction[:,0],
            'I': prediction[:,1],
            'R': prediction[:,2]
        }, index=new_index)
#        df = df*N
#        df[['I','Actual']].plot(ax=ax)
        self.df = df
        self.beta = beta
        self.gamma = gamma
        self.cost = self.loss([beta,gamma], [data,y0])
        if self.plot:
            self.graph()
#        plt.grid()

    def graph(self, delta=None, save=False):
            fig, ax = plt.subplots(figsize=(15, 10))
            ax.set_title("%s" % (self.country))
#            ax.set_title("%s beta=%.2f gamma =%.2f R0 = %.3f" % (self.country, self.beta, self.gamma, self.beta/self.gamma))
            if delta is not None:
                self.df['Total'] = self.df['Actual']
                for i in delta.index:
                    self.df.loc[i,'Total'] += delta[i]
                plt.plot(self.df[['I','Actual','Total']], linewidth=3)
                plt.legend(['Infected (predicted)','Infected (actual)','Infected (total)'])
            else:
                plt.plot(self.df[['I','Actual']])
                plt.legend(['Infected (predicted)','Infected (actual)'])                
            plt.xticks(rotation=90)
            plt.ylabel('Incubation window confirmed cases')
            if save:
                name = '-'.join([self.country,datetime.strftime(datetime.now(), '%Y-%m-%d-%H-%M-%S')])
                plt.savefig(os.path.join('figs', name+'.png'))
        

def dSIR(t, y, beta, gamma):
        dT = 1
        S = y[0]
        I = y[1]
        R = y[2]
        return [S-dT*beta*S*I/N, I+dT*(beta*S*I/N - gamma*I), R+dT*gamma*I]

def dsolve_ivp(fun, t_eval, y0, beta, gamma):
        y = y0
        sol = []
        for t in np.arange(t_eval[0], t_eval[1]):
            y = fun(t,y, beta=beta, gamma=gamma)
            sol.append(y)
        sol = np.array(sol)
        return sol



def loss(point,args):
    """
    RMSE between actual confirmed cases and the estimated infectious people with given beta and gamma.
    Using a discrete model
    """
    data, y0 = args
    size = len(data)
    beta, gamma = point
    if beta < 0 or gamma < 0:
        return 1e6
    sol = dsolve_ivp(dSIR, t_eval=[0, size], y0=y0, beta=beta, gamma=gamma)
#    import pdb
#    pdb.set_trace()
    yp = sol[:,1]
    y = data.tolist()
    if len( np.where( np.logical_not( np.isfinite( yp) )  )[0]) > 0:
        return 1e6
    myloss = metrics.mean_squared_error
    try:
        rmsd = np.sqrt( myloss(yp,y) )
    except:
        import pdb
        pdb.set_trace()
#    rmsd = np.sqrt( np.mean( np.abs( yp - y) )**2 )
#    rmsd += np.sqrt( np.mean( np.abs( np.diff(yp) - np.diff(y)) )**2 )
    return rmsd

   


#%%
    
def calculate_model(country,start_date, plot=False, verbose=False, incubation_days=14):
    START_DATE[country] = start_date
    N = POPULATION[country]
    confirmed, deaths, recoveries, confirmed_delta = load_data(country, incubation=True, incubation_days=incubation_days)
    I_0 = confirmed[0]
    R_0 = recoveries[0]
    S_0 = N - I_0 - R_0
    learn = Learner(country, N, confirmed, recoveries, deaths, loss, plot=plot, verbose=verbose)
    learn.train()
    learn.delta = confirmed_delta
    return learn    

#%%
incubation_days = 14
country = 'United Kingdom'
START_DATE[country] = '1/22/20'
confirmed, deaths, recoveries, confirmed_delta = load_data(country, incubation=True, incubation_days=incubation_days)

costs = []
dates = []
for i in confirmed.index[:-30]:
    learn = calculate_model(country, i, incubation_days=incubation_days)
    print(i,learn.cost)
    dates.append(i)
    costs.append( learn.cost )
#%%

best =  dates[np.argmin(costs)]
learn = calculate_model(country, best, plot=False, incubation_days=incubation_days)
learn.graph(learn.delta, save=True)

#%%
""" A test about model evolution """

st = START_DATE[country]
vals = []
win = 14
for i in np.arange(win):
    j = -win+i
    learn = Learner(confirmed[:j], loss)
    learn.train()
    vals.append( (confirmed_df.columns[j], learn.beta, learn.gamma) )

#%%
    
vals_df = pd.DataFrame(vals, columns=['Date','beta', 'gamma'])
#plt.plot(vals_df['Date'], vals_df[['beta','gamma']], marker='o')

plt.plot(vals_df['Date'], vals_df['beta']/vals_df['gamma'], marker='o')
#plt.legend(['Average contacts', 'Recovered rate', 'Reproduction'])
plt.legend(['Reproduction ratio'])

xt = plt.xticks(rotation=90)
#plt.yscale("log")





#%% -*- coding: utf-8 -*-
"""
Investment Strategy

Date: Feb 10, 2021
Authors: af, jk, jv
"""

import pandas as pd
import numpy as np
import time
import statsmodels.api as sm
from tqdm import tqdm

# set directory to your data files
path_data = '/Users/Forrest/Desktop/MBAN/BAFI 580C/Final Project/Code/data/'


#%%############################################################################
# Step 1: Preparing the CRSP file
###############################################################################
print("Prepare CRSP file")
t = time.time() # record the current time, so we can measure how long the code takes to run

# load data
crsp = pd.read_csv(path_data+'crsp.csv')

# Have a look at the data
print(crsp.head())
print(crsp.dtypes)


### formatting ###
# make all variable names lowercase
crsp.columns = map(str.lower,crsp.columns)

# You should see that one of the important variables 'RET' (return) is not a number but 'object'.
# It is preferable to have this variable as a number, which Python denotes as float64 (float64 is just a special way of saying that a variable is a number)
# If you are interested search for 'floating point number'on internet. But it is computer-science issue!

# Changes the returns to number format. Non-numeric data will be NAN
crsp['ret'] = pd.to_numeric(crsp['ret'],errors='coerce') 

# Change the dateformat
crsp['date'] = pd.to_datetime(crsp['date'], format='%Y%m%d')

# Create separate 'year' and 'month' variables (we will use them later to merge CRSP with Compustat)
crsp['year'] = crsp['date'].apply(lambda date: date.year)
crsp['month'] = crsp['date'].apply(lambda date: date.month)

# Calculate market cap
crsp['mktcap'] = crsp['shrout'] * crsp['prc'].abs()


### Some basic data cleaning ###
# keep only common shares
crsp = crsp[crsp['shrcd'].isin([10,11])]

# keep only stocks from NYSE, AMEX and NASDAQ
crsp = crsp[crsp['exchcd'].isin([1,2,3])]

# make sure that there are no duplicates
# usually, we would investigate why there are duplicates and then decide which observation we want to keep
#    For here, it is enough to simply drop the duplicates.
crsp = crsp.drop_duplicates(subset=['date','permno'])


print('Completed in %.1fs' % (time.time()-t)) # show how long it took to run this code block


#%%############################################################################
# Step 2: Preparing the Compustat (CCM) file
###############################################################################
print("Prepare Compustat file")
t = time.time() # reset our timer

ccm = pd.read_csv(path_data+'compustat2.csv')

# Have a look at the data
print(ccm.head())
print(ccm.dtypes)


### formatting ###
# make all variable names lowercase
ccm.columns = map(str.lower,ccm.columns)

# Change the dateformat 
ccm['datadate'] = pd.to_datetime(ccm['datadate'], format='%Y%m%d')

# Create separate 'year' and 'month' variables
ccm['year'] = ccm['datadate'].apply(lambda x: x.year)
ccm['month'] = ccm['datadate'].apply(lambda x: x.month)


### Some basic data cleaning ###
# make sure that there are no duplicates (same as above)
ccm = ccm.drop_duplicates(subset=['datadate','gvkey'])
ccm = ccm.drop_duplicates(subset=['year','gvkey'])
ccm = ccm.drop_duplicates(subset=['year','lpermno'])


### Calculate the variables we will use for sorting ###
# Create lagged retained earnings variable 
# Note 1) Pandas does not know the panel data structure, so we need to make sure that the previous
#    record belongs to the same gvkey, and that there are no gaps in the data
# Note 2) We can use the backslash "\" do break long lines
ccm = ccm.sort_values(['gvkey','datadate']) # sort data by gvkey and date
ccm['earn_lagged'] = ccm['epspx'].shift(1) # take the previous record
ccm.loc[(ccm['gvkey'].shift(1) != ccm['gvkey']) | \
        (ccm['year'].shift(1) != ccm['year']-1) | \
        (ccm['month'].shift(1) != ccm['month']),'earn_lagged'] = np.NAN # only use the previous record if it 1) belongs to the same gvkey and 2) is one year older

# lagged re over price * shares outstanding
ccm['earn_o_price'] = ccm['earn_lagged'] / ccm['prcc_c']

# change in re
ccm['change_in_earn'] = ccm['epspx'] / ccm['earn_lagged'] - 1

# It is useful to know how many observations are missing
print('Fraction of observations missing:')
print(1 - ccm.count() / len(ccm))


print('Completed in %.1fs' % (time.time()-t)) # show how long it took to run this code block


#%%############################################################################
# Step 3: Sort stocks into portfolios and calculate returns
###############################################################################
print("Create portfolios")
t = time.time() # reset our timer

# loop over all years in the data
# Note: the first loop loops over the years in range(1981,2017).
#    You can wrap any list by the tqdm command to display a progress bar while looping over the list
portfolios = [] # create an empty list to collect the portfolio returns
for year in tqdm(range(1981,2020),desc="years"):
    # take the companies that were alive at t-1
    permno_list=list(crsp[crsp['year']==year-1]['permno'].unique()) 
    
    # get the sorting variable for these companies at t-1
    sorting_data = ccm.loc[(ccm['year']==(year-1)) & \
                           (ccm['lpermno'].isin(permno_list)), \
                           ['gvkey','lpermno','earn_o_price']]

    # remove stocks w/ negative re_o_price ratio (as per Ken French portfolio)
    sorting_data = sorting_data[sorting_data['earn_o_price']>0]
    
    # sort into 5 baskets by re over price
    nportfolios = 5 # number of portfolios
    sorting_data['rank'] = pd.qcut(sorting_data['earn_o_price'],nportfolios, labels=False)
    
    # select the return data with some time lag to make sure that the accounting information is public (data from July at year t to June in year t+1)
    crsp_window = crsp[((crsp['year']==year) & (crsp['month']>=6)) | \
                       ((crsp['year']==year+1) & (crsp['month']<=6))]

    # create the portfolio returns for the current window and collect them in portfolios_window
    portfolios_window = [] 
    for p in range(nportfolios):
        # get list of permnos that are in this portfolio
        basket = sorting_data.loc[sorting_data['rank'] == p,'lpermno'].tolist()
        
        # get returns of these permnos
        crsp_p_firms = crsp_window[crsp_window['permno'].isin(basket)]
        
        # pivot returns
        returns = crsp_p_firms.pivot(index='date', columns='permno', values='ret')
        returns = returns.iloc[1:,:] # drop the first row
        
        # create equally weighted portfolio (monthly rebalancing)
        return_port = returns.mean(axis=1)
        return_port.name = str(p)
        
        # collect portfolio returns in dec_port
        portfolios_window += [return_port]
        
    # merge the portfolios
    portfolios_window = pd.concat(portfolios_window,axis=1)
        
    # collect results in portfolios
    portfolios += [portfolios_window]

# merge the returns from all windows
portfolios = pd.concat(portfolios,axis=0)


print('Step 3 completed in %.1fs' % (time.time()-t)) # show how long it took to run this code block


#%%############################################################################
# Step 4: Performance Evaluation
# Step 4a: Merge Portfolio returns with Fama French data
###############################################################################

### load and prepare fama french data ###
# load Fama French monthly factors
ff = pd.read_csv(path_data+'F-F_Research_Data_Factors_2018.csv')

# rename columns
ff.rename({'Mkt-RF':'ExMkt',
           'DATE':'date'},axis=1,inplace=True)

# date variables
ff['year'] = ff['date'] // 100
ff['month'] = ff['date'] % 100
ff.set_index('date',inplace=True)


### formatting ###
# FF data is in percent. Convert to simple returns
ff[['ExMkt', 'SMB', 'HML', 'RF']] /= 100


### merge portfolio returns with Fama French data ###
# date variables
portfolios_ff = portfolios.copy() # create a copy of the portfolios dataframe so we can use it again later
portfolios_ff[f'{nportfolios}'] = portfolios_ff[f'{nportfolios-1}']-portfolios_ff['0']
portfolios_ff['year'] = portfolios_ff.index.year
portfolios_ff['month'] = portfolios_ff.index.month

# merge
portfolios_ff = pd.merge(portfolios_ff,ff,on=['year','month'])

#%%############################################################################
# Step 4b-1: Summary Statistics
###############################################################################

from scipy.stats import ttest_1samp

returns = portfolios_ff.iloc[:,:nportfolios+1]
sum_table = pd.DataFrame({'mean':returns.mean(),
                          'mean(annualized)':returns.mean()*12,
                          't-stat':[ttest_1samp(returns[i], 0)[0] for i in returns.columns],
                          'p-value':[ttest_1samp(returns[i], 0)[1] for i in returns.columns],
                          'SD':returns.std(),
                          'Skewness':returns.skew(),
                          'Kurtosis':returns.kurtosis()}).T

sum_table.columns = ['Lo 20', 'Qnt 2', 'Qnt 3', 'Qnt 4', 'Hi 20', 'Hi 20 - Lo 20']


#%%############################################################################
# Step 4b-2: Regressions
###############################################################################

# show average returns (annualized and in percent)
print("Average returns (annualized percent)\n",((1+portfolios.mean(axis=0))**12-1)*100)

# Calculate the excess returns
for p in range(nportfolios):
    portfolios_ff['ExRet_'+str(p)] = portfolios_ff[str(p)]-portfolios_ff['RF']

# No need to minus risk-free rate for winners-losers portfolio
portfolios_ff['ExRet_'+str(nportfolios)] = portfolios_ff[f'{nportfolios}']


### Market model regressions ###
table_capm = []
for p in range(nportfolios+1):
    # regress portfolio excess return on market excess return
    results = sm.OLS(portfolios_ff['ExRet_'+str(p)],
                     sm.add_constant(portfolios_ff['ExMkt'])).fit()
    
    # collect results
    table_row = pd.DataFrame({'alpha':results.params['const'],
                              'alpha_t':results.tvalues['const'],
                              'alpha_p':results.pvalues['const'],
                              'beta_mkt':results.params['ExMkt'],
                              'beta_t':results.tvalues['ExMkt'],
                              'beta_p':results.pvalues['ExMkt'],
                              'rmse':np.sqrt(results.mse_resid),
                              'R2':results.rsquared},
                             index=[p])
    
    table_capm += [table_row]

# Combine the results for all portfolios
table_capm = pd.concat(table_capm,axis=0)
table_capm.index.name = 'quintile'

# show results
print("CAPM\n",table_capm)


### Three Factor model regressions ###
table_ff = []
for p in range(nportfolios+1):
    # regress portfolio excess return on market excess return
    results = sm.OLS(portfolios_ff['ExRet_'+str(p)],
                     sm.add_constant(portfolios_ff[['ExMkt','SMB','HML']])).fit()
    
    # collect results
    table_row = pd.DataFrame({'alpha':results.params['const'],
                              'alpha_t':results.tvalues['const'],
                              'alpha_p':results.pvalues['const'],
                              'beta_mkt':results.params['ExMkt'],
                              'beta_mkt_t':results.tvalues['ExMkt'],
                              'beta_mkt_p':results.pvalues['ExMkt'],
                              'beta_size':results.params['SMB'],
                              'beta_size_t':results.tvalues['SMB'],
                              'beta_size_p':results.pvalues['SMB'],
                              'beta_hml':results.params['HML'],
                              'beta_hml_t':results.tvalues['HML'],
                              'beta_hml_p':results.pvalues['HML'],
                              'rmse':np.sqrt(results.mse_resid),
                              'R2':results.rsquared},
                             index=[p])
    
    table_ff += [table_row]


# Combine the results for all portfolios
table_ff = pd.concat(table_ff,axis=0)
table_ff.index.name = 'quintile'

# show results
print("Fama-French 3\n",table_ff)

#%%############################################################################
# Step 4b-4: Save Results
###############################################################################

from pandas import ExcelWriter

def save_xls(list_dfs, xls_path):
    with ExcelWriter(xls_path) as writer:
        for n, df in enumerate(list_dfs):
            df.to_excel(writer,'sheet%s' % n)
        writer.save()


sum_table_capm = table_capm.T
sum_table_capm.columnes = ['Lo 20', 'Qnt 2', 'Qnt 3', 'Qnt 4', 'Hi 20', 'Hi 20 - Lo 20']
sum_table_ff = table_ff.T
sum_table_ff.columnes = ['Lo 20', 'Qnt 2', 'Qnt 3', 'Qnt 4', 'Hi 20', 'Hi 20 - Lo 20']

tables = [sum_table.round(4), sum_table_capm.round(4), sum_table_ff.round(4)]
path = path_data+'sum_tables.xlsx'

save_xls(tables, path)
#%%############################################################################
# Step 5: Sort stocks into portfolios and calculate returns (Extension)
###############################################################################
print("Create portfolios")
t = time.time() # reset our timer

# loop over all years in the data
# Note: the first loop loops over the years in range(1981,2017).
#    You can wrap any list by the tqdm command to display a progress bar while looping over the list
portfolios = [] # create an empty list to collect the portfolio returns
for year in tqdm(range(1981,2020),desc="years"):
    # take the companies that were alive at t-1
    permno_list=list(crsp[crsp['year']==year-1]['permno'].unique()) 
    
    # get the sorting variable for these companies at t-1
    sorting_data = ccm.loc[(ccm['year']==(year-1)) & \
                           (ccm['lpermno'].isin(permno_list)), \
                           ['gvkey','lpermno','earn_o_price']]

    # remove stocks w/ positive re_o_price ratio (as per Ken French portfolio)
    sorting_data = sorting_data[sorting_data['earn_o_price']<=0]
    
    # sort into 5 baskets by re over price
    nportfolios = 5 # number of portfolios
    sorting_data['rank'] = pd.qcut(sorting_data['earn_o_price'],nportfolios, labels=False)
    
    # select the return data with some time lag to make sure that the accounting information is public (data from July at year t to June in year t+1)
    crsp_window = crsp[((crsp['year']==year) & (crsp['month']>=6)) | \
                       ((crsp['year']==year+1) & (crsp['month']<=6))]
    
    # create the portfolio returns for the current window and collect them in portfolios_window
    portfolios_window = [] 
    for p in range(nportfolios):
        # get list of permnos that are in this portfolio
        basket = sorting_data.loc[sorting_data['rank'] == p,'lpermno'].tolist()

        # get returns of these permnos
        crsp_p_firms = crsp_window[crsp_window['permno'].isin(basket)]

        # pivot returns
        returns = crsp_p_firms.pivot(index='date', columns='permno', values='ret')
        returns = returns.iloc[1:,:] # drop the first row
        
        # create equally weighted portfolio (monthly rebalancing)
        return_port = returns.mean(axis=1)
        return_port.name = str(p)
        
        # collect portfolio returns in dec_port
        portfolios_window += [return_port]
        
    # merge the portfolios
    portfolios_window = pd.concat(portfolios_window,axis=1)
        
    # collect results in portfolios
    portfolios += [portfolios_window]

# merge the returns from all windows
portfolios = pd.concat(portfolios,axis=0)


print('Step 3 completed in %.1fs' % (time.time()-t)) # show how long it took to run this code block


#%%############################################################################
# Step 6: Performance Evaluation (Extension)
# Step 6a: Merge Portfolio returns with Fama French data (Extension)
###############################################################################

### load and prepare fama french data ###
# load Fama French monthly factors
ff = pd.read_csv(path_data+'F-F_Research_Data_Factors_2018.csv')

# rename columns
ff.rename({'Mkt-RF':'ExMkt',
           'DATE':'date'},axis=1,inplace=True)

# date variables
ff['year'] = ff['date'] // 100
ff['month'] = ff['date'] % 100
ff.set_index('date',inplace=True)


### formatting ###
# FF data is in percent. Convert to simple returns
ff[['ExMkt', 'SMB', 'HML', 'RF']] /= 100


### merge portfolio returns with Fama French data ###
# date variables
portfolios_ff = portfolios.copy() # create a copy of the portfolios dataframe so we can use it again later
portfolios_ff[f'{nportfolios}'] = portfolios_ff['0']-portfolios_ff[f'{nportfolios-1}']
portfolios_ff['year'] = portfolios_ff.index.year
portfolios_ff['month'] = portfolios_ff.index.month

# merge
portfolios_ff = pd.merge(portfolios_ff,ff,on=['year','month'])

#%%############################################################################
# Step 6b-1: Summary Statistics (Extension)
###############################################################################

from scipy.stats import ttest_1samp

returns = portfolios_ff.iloc[:,:nportfolios+1]
sum_table = pd.DataFrame({'mean':returns.mean(),
                          'mean(annualized)':returns.mean()*12,
                          't-stat':[ttest_1samp(returns[i], 0)[0] for i in returns.columns],
                          'p-value':[ttest_1samp(returns[i], 0)[1] for i in returns.columns],
                          'SD':returns.std(),
                          'Skewness':returns.skew(),
                          'Kurtosis':returns.kurtosis()}).T

sum_table.columns = ['Lo 20', 'Qnt 2', 'Qnt 3', 'Qnt 4', 'Hi 20', 'Lo 20 - Hi 20']


#%%############################################################################
# Step 6b-2: Regressions (Extension)
###############################################################################

# show average returns (annualized and in percent)
print("Average returns (annualized percent)\n",((1+portfolios.mean(axis=0))**12-1)*100)

# Calculate the excess returns
for p in range(nportfolios):
    portfolios_ff['ExRet_'+str(p)] = portfolios_ff[str(p)]-portfolios_ff['RF']

# No need to minus risk-free rate for winners-losers portfolio
portfolios_ff['ExRet_'+str(nportfolios)] = portfolios_ff[f'{nportfolios}']


### Market model regressions ###
table_capm = []
for p in range(nportfolios+1):
    # regress portfolio excess return on market excess return
    results = sm.OLS(portfolios_ff['ExRet_'+str(p)],
                     sm.add_constant(portfolios_ff['ExMkt'])).fit()
    
    # collect results
    table_row = pd.DataFrame({'alpha':results.params['const'],
                              'alpha_t':results.tvalues['const'],
                              'alpha_p':results.pvalues['const'],
                              'beta_mkt':results.params['ExMkt'],
                              'beta_t':results.tvalues['ExMkt'],
                              'beta_p':results.pvalues['ExMkt'],
                              'rmse':np.sqrt(results.mse_resid),
                              'R2':results.rsquared},
                             index=[p])
    
    table_capm += [table_row]

# Combine the results for all portfolios
table_capm = pd.concat(table_capm,axis=0)
table_capm.index.name = 'quintile'

# show results
print("CAPM\n",table_capm)


### Three Factor model regressions ###
table_ff = []
for p in range(nportfolios+1):
    # regress portfolio excess return on market excess return
    results = sm.OLS(portfolios_ff['ExRet_'+str(p)],
                     sm.add_constant(portfolios_ff[['ExMkt','SMB','HML']])).fit()
    
    # collect results
    table_row = pd.DataFrame({'alpha':results.params['const'],
                              'alpha_t':results.tvalues['const'],
                              'alpha_p':results.pvalues['const'],
                              'beta_mkt':results.params['ExMkt'],
                              'beta_mkt_t':results.tvalues['ExMkt'],
                              'beta_mkt_p':results.pvalues['ExMkt'],
                              'beta_size':results.params['SMB'],
                              'beta_size_t':results.tvalues['SMB'],
                              'beta_size_p':results.pvalues['SMB'],
                              'beta_hml':results.params['HML'],
                              'beta_hml_t':results.tvalues['HML'],
                              'beta_hml_p':results.pvalues['HML'],
                              'rmse':np.sqrt(results.mse_resid),
                              'R2':results.rsquared},
                             index=[p])
    
    table_ff += [table_row]


# Combine the results for all portfolios
table_ff = pd.concat(table_ff,axis=0)
table_ff.index.name = 'quintile'

# show results
print("Fama-French 3\n",table_ff)

#%%############################################################################
# Step 6b-4: Save Results (Extension)
###############################################################################

from pandas import ExcelWriter

def save_xls(list_dfs, xls_path):
    with ExcelWriter(xls_path) as writer:
        for n, df in enumerate(list_dfs):
            df.to_excel(writer,'sheet%s' % n)
        writer.save()


sum_table_capm = table_capm.T
sum_table_capm.columnes = ['Lo 20', 'Qnt 2', 'Qnt 3', 'Qnt 4', 'Hi 20', 'Lo 20 - Hi 20']
sum_table_ff = table_ff.T
sum_table_ff.columnes = ['Lo 20', 'Qnt 2', 'Qnt 3', 'Qnt 4', 'Hi 20', 'Lo 20 - Hi 20']

tables = [sum_table.round(4), sum_table_capm.round(4), sum_table_ff.round(4)]
path = path_data+'sum_tables_extension.xlsx'

save_xls(tables, path)

# %%

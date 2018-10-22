
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


df=pd.read_csv('C:/Users/dhrit/Desktop/azd.csv')


# In[4]:


df.head()


# In[5]:


df.describe()


# In[6]:


df.info()


# In[7]:


df[df['minPrice'].isnull()]


# In[8]:


df[df['modalPrice'].isnull()]


# In[9]:


df[df['maxPrice'].isnull()]


# In[10]:


np.average(df['maxPrice'].drop([1356]))


# In[11]:


df['maxPrice']=df['maxPrice'].fillna(np.average(df['maxPrice'].drop([1356])))


# In[12]:


df['minPrice']=df['minPrice'].fillna(np.average(df['minPrice'].drop([733])))


# In[13]:


df['modalPrice']=df['modalPrice'].fillna(np.average(df['modalPrice'].drop([142,193,363,364,365,368,371,394,395,1356])))


# In[14]:


plt.rcParams['patch.force_edgecolor']=True
sns.distplot(df['modalPrice'])


# In[15]:


df=df.drop('variety',axis=1)


# In[16]:


df=df.drop('itemName',axis=1)


# In[17]:


df=df.drop('state',axis=1)


# In[18]:


df=df.drop('mandiName',axis=1)


# In[19]:


df=df.drop('arrivals',axis=1)


# In[20]:


df=df.drop('unitArrivals',axis=1)


# In[21]:


df=df.drop('priceUnit',axis=1)


# In[22]:


df.head()


# In[23]:


from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

print(__version__)


# In[24]:


import cufflinks as cf


# In[25]:


# For Notebooks
init_notebook_mode(connected=True)


# In[26]:


# For offline use
cf.go_offline()


# In[27]:


df.iplot(x='priceDate')


# In[28]:


df.head()


# In[29]:


from datetime import datetime
con=df['priceDate']


# In[30]:


df['priceDate']=pd.to_datetime(df['priceDate'])
df.set_index('priceDate', inplace=True)


# In[31]:


df.index


# In[32]:


#convert to time series:
ts1 = df['minPrice']
ts1.head(10)


# In[33]:


ts2 = df['maxPrice']
ts2.head(10)


# In[34]:


ts3 = df['modalPrice']
ts3.head(10)


# In[35]:


ts1.iplot()


# In[36]:


from statsmodels.tsa.stattools import adfuller


# In[48]:


def test_stationarity(timeseries):
    
    #Determing rolling statistics
    rolmean = timeseries.rolling(365).mean()
    rolstd = timeseries.rolling(365).std()
#Plot rolling statistics:
    plt.plot(timeseries, color='blue',label='Original')
    plt.plot(rolmean, color='red', label='Rolling Mean')
    plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()
    #Perform Dickey-Fuller test:
    print( 'Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)


# In[49]:


plt.figure(figsize=(14,14))
test_stationarity(ts1)


# In[104]:


plt.figure(figsize=(14,14))
test_stationarity(ts2)


# In[50]:


ts1_log=np.log(ts1)


# In[105]:


ts2_log=np.log(ts2)


# In[107]:


ts3_log=np.log(ts3)


# In[51]:


plt.plot(ts1_log)


# In[52]:


rolling_mean=ts1_log.rolling(12).mean()
plt.plot(ts1_log)
plt.plot(rolling_mean,color='red')


# In[106]:


rolling_mean_ts2=ts2_log.rolling(12).mean()
plt.plot(ts2_log)
plt.plot(rolling_mean_ts2,color='red')


# In[108]:


rolling_mean_ts3=ts3_log.rolling(12).mean()
plt.plot(ts3_log)
plt.plot(rolling_mean_ts3,color='red')


# In[53]:


ts1_moving_avg_diff=ts1_log-rolling_mean


# In[109]:


ts2_moving_avg_diff=ts2_log-rolling_mean_ts2


# In[110]:


ts3_moving_avg_diff=ts1_log-rolling_mean_ts3


# In[54]:


ts1_moving_avg_diff.head()


# In[55]:


ts1_moving_avg_diff.dropna(inplace=True)
ts1_moving_avg_diff.head()


# In[112]:


ts2_moving_avg_diff.dropna(inplace=True)
ts2_moving_avg_diff.head()


# In[113]:


ts3_moving_avg_diff.dropna(inplace=True)
ts3_moving_avg_diff.head()


# In[56]:


plt.figure(figsize=(14,14))
test_stationarity(ts1_moving_avg_diff)


# In[114]:


plt.figure(figsize=(14,14))
test_stationarity(ts2_moving_avg_diff)


# In[115]:


plt.figure(figsize=(14,14))
test_stationarity(ts3_moving_avg_diff)


# In[57]:


ts1_log_diff=ts1_log-ts1_log.shift()
plt.plot(ts1_log_diff)


# In[116]:


ts2_log_diff=ts2_log-ts2_log.shift()
plt.plot(ts2_log_diff)


# In[117]:


ts3_log_diff=ts3_log-ts3_log.shift()
plt.plot(ts3_log_diff)


# In[58]:


plt.figure(figsize=(10,10))
ts1_log_diff.dropna(inplace=True)
test_stationarity(ts1_log_diff)


# In[118]:


plt.figure(figsize=(10,10))
ts2_log_diff.dropna(inplace=True)
test_stationarity(ts2_log_diff)


# In[119]:


plt.figure(figsize=(10,10))
ts3_log_diff.dropna(inplace=True)
test_stationarity(ts3_log_diff)


# In[77]:


from statsmodels.tsa.seasonal import seasonal_decompose
decomposition=seasonal_decompose(ts1_log,freq=365)

trend=decomposition.trend
seasonal=decomposition.seasonal
residual=decomposition.resid

plt.figure(figsize=(15,10))
plt.subplot(411)
plt.plot(ts1_log,label='Original')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend,label='Trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal,label='Seasonality')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual,label='Residuals')
plt.legend(loc='best')


# In[120]:


from statsmodels.tsa.seasonal import seasonal_decompose
decomposition_ts2=seasonal_decompose(ts2_log,freq=365)

trend_ts2=decomposition.trend
seasonal_ts2=decomposition.seasonal
residual_ts2=decomposition.resid

plt.subplot(411)
plt.plot(ts2_log,label='Original')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend,label='Trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal,label='Seasonality')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual,label='Residuals')
plt.legend(loc='best')


# In[121]:


from statsmodels.tsa.seasonal import seasonal_decompose
decomposition_ts3=seasonal_decompose(ts3_log,freq=365)

trend_ts3=decomposition.trend
seasonal_ts3=decomposition.seasonal
residual_ts3=decomposition.resid

plt.subplot(411)
plt.plot(ts3_log,label='Original')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend,label='Trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal,label='Seasonality')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual,label='Residuals')
plt.legend(loc='best')


# In[78]:


ts1_log_decompose=residual
ts1_log_decompose.dropna(inplace=True)
test_stationarity(ts1_log_decompose)


# In[122]:


ts2_log_decompose=residual
ts2_log_decompose.dropna(inplace=True)
test_stationarity(ts2_log_decompose)


# In[124]:


ts3_log_decompose=residual
ts3_log_decompose.dropna(inplace=True)
test_stationarity(ts3_log_decompose)


# In[83]:


from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import acf,pacf

lag_acf=acf(ts1_log_diff,nlags=20)
lag_pacf=pacf(ts1_log_diff,nlags=20,method='ols')

plt.figure(figsize=(10,10))
plt.subplot(121)
plt.plot(lag_acf)
plt.axhline(y=0,linestyle='--',color='grey')
plt.axhline(y=-1.96/np.sqrt(len(ts1_log_diff)),linestyle='--',color='grey')
plt.axhline(y=1.96/np.sqrt(len(ts1_log_diff)),linestyle='--',color='grey')
plt.title('ACF')

plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0,linestyle='--',color='grey')
plt.axhline(y=-1.96/np.sqrt(len(ts1_log_diff)),linestyle='--',color='grey')
plt.axhline(y=1.96/np.sqrt(len(ts1_log_diff)),linestyle='--',color='grey')
plt.title('PACF')


# In[125]:


from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import acf,pacf

lag_acf_ts2=acf(ts2_log_diff,nlags=20)
lag_pacf_ts2=pacf(ts2_log_diff,nlags=20,method='ols')

plt.subplot(121)
plt.plot(lag_acf_ts2)
plt.axhline(y=0,linestyle='--',color='grey')
plt.axhline(y=-1.96/np.sqrt(len(ts2_log_diff)),linestyle='--',color='grey')
plt.axhline(y=1.96/np.sqrt(len(ts2_log_diff)),linestyle='--',color='grey')
plt.title('ACF')

plt.subplot(122)
plt.plot(lag_pacf_ts2)
plt.axhline(y=0,linestyle='--',color='grey')
plt.axhline(y=-1.96/np.sqrt(len(ts2_log_diff)),linestyle='--',color='grey')
plt.axhline(y=1.96/np.sqrt(len(ts2_log_diff)),linestyle='--',color='grey')
plt.title('PACF')


# In[126]:


from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import acf,pacf

lag_acf_ts3=acf(ts3_log_diff,nlags=20)
lag_pacf_ts3=pacf(ts3_log_diff,nlags=20,method='ols')

plt.subplot(121)
plt.plot(lag_acf_ts3)
plt.axhline(y=0,linestyle='--',color='grey')
plt.axhline(y=-1.96/np.sqrt(len(ts3_log_diff)),linestyle='--',color='grey')
plt.axhline(y=1.96/np.sqrt(len(ts3_log_diff)),linestyle='--',color='grey')
plt.title('ACF')

plt.subplot(122)
plt.plot(lag_pacf_ts2)
plt.axhline(y=0,linestyle='--',color='grey')
plt.axhline(y=-1.96/np.sqrt(len(ts3_log_diff)),linestyle='--',color='grey')
plt.axhline(y=1.96/np.sqrt(len(ts3_log_diff)),linestyle='--',color='grey')
plt.title('PACF')


# In[84]:


#AR model
model=ARIMA(ts1_log,order=(2,1,0))
results_AR=model.fit(disp=-1)
plt.plot(ts1_log_diff)
plt.plot(results_AR.fittedvalues,color='red')
plt.title('RSS: %.4f'% sum((results_AR.fittedvalues-ts1_log_diff)**2))


# In[85]:


#MA model
model=ARIMA(ts1_log,order=(0,1,2))
results_MA=model.fit(disp=-1)
plt.plot(ts1_log_diff)
plt.plot(results_MA.fittedvalues,color='red')
plt.title('RSS: %.4f'% sum((results_MA.fittedvalues-ts1_log_diff)**2))


# In[94]:


#ARIMA model
model=ARIMA(ts1_log,order=(2,1,2))
results_ARIMA=model.fit(disp=5)
plt.plot(ts1_log_diff)
plt.plot(results_ARIMA.fittedvalues,color='red')
plt.title('RSS: %.4f'% sum((results_ARIMA.fittedvalues-ts1_log_diff)**2))


# In[128]:


#ARIMA model
model2=ARIMA(ts2_log,order=(2,1,2))
results_ARIMA_ts2=model2.fit(disp=5)
plt.plot(ts2_log_diff)
plt.plot(results_ARIMA_ts2.fittedvalues,color='red')
plt.title('RSS: %.4f'% sum((results_ARIMA_ts2.fittedvalues-ts2_log_diff)**2))


# In[129]:


#ARIMA model
model3=ARIMA(ts3_log,order=(2,1,2))
results_ARIMA_ts3=model3.fit(disp=5)
plt.plot(ts3_log_diff)
plt.plot(results_ARIMA_ts3.fittedvalues,color='red')
plt.title('RSS: %.4f'% sum((results_ARIMA_ts3.fittedvalues-ts3_log_diff)**2))


# In[95]:


predictions_ARIMA_diff=pd.Series(results_ARIMA.fittedvalues,copy=True)
print(predictions_ARIMA_diff.head(30))


# In[131]:


predictions_ARIMA_diff_ts2=pd.Series(results_ARIMA_ts2.fittedvalues,copy=True)
print(predictions_ARIMA_diff_ts2.head())


# In[132]:


predictions_ARIMA_diff_ts3=pd.Series(results_ARIMA_ts3.fittedvalues,copy=True)
print(predictions_ARIMA_diff_ts3.head())


# In[91]:


predictions_ARIMA_diff_cumsum=predictions_ARIMA_diff.cumsum()
print(predictions_ARIMA_diff.head())


# In[133]:


predictions_ARIMA_diff_cumsum_ts2=predictions_ARIMA_diff_ts2.cumsum()
print(predictions_ARIMA_diff_ts2.head())


# In[134]:


predictions_ARIMA_diff_cumsum_ts3=predictions_ARIMA_diff_ts3.cumsum()
print(predictions_ARIMA_diff_ts3.head())


# In[93]:


predictions_ARIMA_log=pd.Series(ts1_log.ix[0],index=ts1_log.index)
predictions_ARIMA_log=predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,fill_value=0)
predictions_ARIMA_log.head()


# In[135]:


predictions_ARIMA_log_ts2=pd.Series(ts2_log.ix[0],index=ts2_log.index)
predictions_ARIMA_log_ts2=predictions_ARIMA_log_ts2.add(predictions_ARIMA_diff_cumsum_ts2,fill_value=0)
predictions_ARIMA_log_ts2.head()


# In[136]:


predictions_ARIMA_log_ts3=pd.Series(ts3_log.ix[0],index=ts3_log.index)
predictions_ARIMA_log_ts3=predictions_ARIMA_log_ts3.add(predictions_ARIMA_diff_cumsum_ts3,fill_value=0)
predictions_ARIMA_log_ts3.head()


# In[96]:


df.head()


# In[141]:


predictions_ARIMA=np.exp(predictions_ARIMA_log)


# In[102]:


predictions_ARIMA.head(30)


# In[137]:


predictions_ARIMA_ts2=np.exp(predictions_ARIMA_log_ts2)
predictions_ARIMA_ts2.head(30)


# In[138]:


predictions_ARIMA_ts3=np.exp(predictions_ARIMA_log_ts3)
predictions_ARIMA_ts3.head(30)


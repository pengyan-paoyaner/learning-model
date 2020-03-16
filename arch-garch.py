# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 20:47:59 2019

@author: lenovo
"""

import pandas as pd
import numpy as np
import os
import statsmodels.tsa.api as smt

#tsa为Time Series analysis缩写
import statsmodels.api as sm
import scipy.stats as scs
from arch import arch_model
#画图
import matplotlib.pyplot as plt
import matplotlib as mpl

#正常显示画图时出现的中文和负号
from pylab import mpl
mpl.rcParams['font.sans-serif']=['SimHei']
mpl.rcParams['axes.unicode_minus']=False
#matplotlib官方提供了五种不同的图形风格，
#包括bmh、ggplot、dark_background、
#fivethirtyeight和grayscale
'''from pandas_datareader import data as web
data= web.get_data_yahoo('MSFT', start = '2016-01-01', end = '2019-01-01')'''
data0=pd.read_table('C:/Users/lenovo/Desktop/399300.txt',encoding='utf-8')
data0=data0[['日期','收盘价']]
data0['ret']=np.log(data0['收盘价']/data0['收盘价'].shift(1))
data=data0['ret']    #创建series
data.index=data0['日期']
data.index = pd.to_datetime(data.index)
data=data.sort_index()
#基本描述
'''earn_mean_daily = np.mean(data)
print("日平均收益：",earn_mean_daily)
plt.hist(data, bins=75)
plt.show()'''

# 模拟正态分布数据，其均值和标准差与文中的股票的日收益率相同。
'''mu=np.mean(data)
sigma=np.std(data)
norm=np.random.normal(mu,sigma,size=10000)
# 绘制正态分布的概率密度分布图
plt. hist(norm, bins=100, alpha=0.8, density=True, label='正太分布')
# 绘制收益的概率密度分布图
plt.hist(data, bins=75, alpha=0.7, density=True,label='收益率分布')
plt.legend()
#plt.show()'''
#取对数、差分，arma模型用差分后的模型预测时，预测结果需要还原
'''data1=data0['收盘价']
data1.index=data0['日期']
data1.index = pd.to_datetime(data1.index)
data1=data1.sort_index()
data_log = np.log(data1)
diff_11 = data_log.diff(1)
diff_11.dropna(inplace=True)
from statsmodels.tsa.arima_model import ARMA
model = ARMA(diff_11, order=(1, 1)) 
result_arma = model.fit( disp=-1, method='css')
predict_data = result_arma.predict()
# 一阶差分还原,diff(2)不是用来计算二阶差分的，而是周期为2的差分;diff(a=test,n=2)可用来计算test数列的二阶差分。 
diff_shift_data = data_log.shift(1)
diff_recover_1 = predict_data.add(diff_shift_data)
# 对数还原
log_recover = np.exp(diff_recover_1)
log_recover.dropna(inplace=True)
fig = plt.figure(figsize=(12, 6))
layout = (1, 2)
ax1 = plt.subplot2grid(layout, (0, 0))
ax1.plot(log_recover,'r',alpha=0.5)
ax1.plot(data1,'g',alpha=0.5)
'''

#单位根检验ADF,平稳性检验 原假设是有单位根 即非平稳
from statsmodels.tsa.stattools import adfuller
def testStationarity(ts):
    dftest = adfuller(ts)
    # 对上述函数求得的值进行语义描述
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():  #这里dftest[4]是字典  
        dfoutput['Critical Value (%s)'%key] = value
    return dfoutput
#dftest = adfuller(data.dropna())
print(testStationarity(data.dropna()))
#testStationarity(data)


# 从 scipy.stats 导入shapiro
'''from scipy.stats import shapiro
# 对股票收益进行Shapiro-Wilk检验 正太分布检验
shapiro_results = shapiro(data.values)
print("Shapiro-Wilk检验结果: ", shapiro_results)
# 提取P值
p_value = shapiro_results[1]
print("P值: ", p_value)
#这里使用 scipy.stats 提供的 shapiro() 函数，对股票收益分布进行 Shapiro-Wilk 检验。该函数有两个返回值，一个是检验的 t 统计量，另一个是 p 值。它越接近1就越表明数据和正态分布拟合得越好。
'''
def tsplot(y, lags=None, figsize=(10, 8), style='bmh'):
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
    with plt.style.context(style):    
        fig = plt.figure(figsize=figsize)
        #mpl.rcParams['font.family'] = 'Ubuntu Mono'
        layout = (3, 2)
        ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
        acf_ax = plt.subplot2grid(layout, (1, 0))
        pacf_ax = plt.subplot2grid(layout, (1, 1))
        qq_ax = plt.subplot2grid(layout, (2, 0))
        pp_ax = plt.subplot2grid(layout, (2, 1))
        
        y.plot(ax=ts_ax)
        ts_ax.set_title('沪深300收益率时序图')
        smt.graphics.plot_acf(y, lags=lags, ax=acf_ax, alpha=0.5)
        smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax, alpha=0.5)
        sm.qqplot(y, line='s', ax=qq_ax)
        qq_ax.set_title('QQ Plot')         
        scs.probplot(y, sparams=(y.mean(), y.std()), plot=pp_ax)

        plt.tight_layout()
    return 
#tsplot(data.dropna(),lags=30)

#不事先确定滞后阶数，而是通过信息准则选择最佳的滞后阶数
#先将初始值设置为无穷大
'''from statsmodels.tsa.arima_model import ARMA 
best_aic=np.inf
best_order=None
best_mdl=None
rng=range(5)
for i in rng:
    for j in rng:
        try:
            tmp_mdl = ARMA(data.dropna(), (i,j)).fit()
            tmp_aic = tmp_mdl.aic
            if tmp_aic < best_aic:
                best_aic = tmp_aic
                best_order = (i,j)
                best_mdl = tmp_mdl
        except: continue
print(f'最佳滞后阶数:{best_order}')
print(best_mdl.summary())
resid=pd.Series(best_mdl.resid,index=data.index)
tsplot(resid, lags=30,title='沪深300指数ARMA拟合残差')'''
ret=data.dropna()
Y=ret*100.0
#决定ARMA模型的阶数
from statsmodels.tsa.arima_model import ARMA
pmax = 2  
qmax = 2
bic_matrix = []
for p in range(pmax +1):
    temp= []
    for q in range(qmax+1):
        try:
            temp.append(ARMA(Y, (p, q)).fit().bic)
        except:
            temp.append(None)
    bic_matrix.append(temp)

bic_matrix = pd.DataFrame(bic_matrix)   #将其转换成Dataframe 数据结构
p,q = bic_matrix.astype('float64').stack().idxmin()   #先使用stack 展平， 然后使用 idxmin 找出最小值的位置
print(u'BIC 最小的p值 和 q 值：%s,%s' %(p,q))  #  BIC 最小的p值 和 q 值：0,0
#所以可以建立ARIMA 模型，ARMA(0,0)
AR=ARMA(Y, (0, 0)).fit()
print(AR.summary())

AR=ARMA(data.dropna(), (1, 1))


#sm.tsa.stattools.arma_order_select_ic(data,max_ma=3)
def ret_plot(ts, title=''):

    ts1=ts**2

    ts2=np.abs(ts)

    with plt.style.context('ggplot'):

        fig = plt.figure(figsize=(12, 6))

    layout = (2, 1)

    ts1_ax = plt.subplot2grid(layout, (0, 0), colspan=2)

    ts2_ax = plt.subplot2grid(layout, (1, 0))

    ts1.plot(ax=ts1_ax)

    ts1_ax.set_title('日收益率平方')

    ts2.plot(ax=ts2_ax)

    ts2_ax.set_title('日收益率绝对值')

    plt.tight_layout()

    return

#ret_plot(data.dropna(), title='沪深300')


def whitenoise_test(ts):
    from statsmodels.stats.diagnostic import acorr_ljungbox

    q,p=acorr_ljungbox(ts)

    with plt.style.context('ggplot'):

        fig = plt.figure(figsize=(10, 4))
        

        axes = fig.subplots(1,2)
        

        axes[0].plot(q, label='Q统计量')

        axes[0].set_ylabel('Q')
        axes[0].set_title('收益率残差平方自相关性检验')
        axes[1].plot(p, label='p值')

        axes[1].set_ylabel('P')
        axes[1].set_title('收益率残差平方自相关性检验')
        axes[0].legend()

        axes[1].legend()

        plt.tight_layout()

    return

ret=data.dropna()
#whitenoise_test(AR.resid)#?这里有问题 arma模型的残差怎么表示

#garch 
Y=ret*100.0

am= arch_model(Y,p=1, o=0, q=1, dist='skewt')
am3=arch_model(Y,p=1, o=1, q=1, dist='StudentsT')
#mean: 均值模型的名称，可选: ‘Constant’, ‘Zero’, ‘ARX’ 以及 ‘HARX’
#vol :波动率模型，可选: ‘GARCH’ （默认）, ‘ARCH’, ‘EGARCH’, ‘FIARCH’ 以及 ‘HARCH’
#p :– 对称随机数的滞后阶,即扣除均值后的部分。
#o ：非对称数据的滞后阶。
#q ：波动率或对应变量的滞后阶。
#power：使用GARCH或相关模型的精度。
#dist：误差分布，可选：正态分布: ‘normal’, ‘gaussian’ (default)；学生T分布: ‘t’, ‘studentst’；偏态学生T分布: ‘skewstudent’, ‘skewt’；通用误差分布: ‘ged’, ‘generalized error”。
#hold_back：对同一样本使用不同的滞后阶来比较模型时使用该参数。
am2 = arch_model(Y,p=1, o=0, q=1, dist='normal')
print(am3.fit().summary())
res = am.fit(update_freq=0)
#update_freq=0表示不输出中间结果，只输出最终结
#预测
'''forecasts = res.forecast(horizon=5, method='simulation')
sims = forecasts.simulations
lines = plt.plot(sims.residual_variances[-1,::10].T, color='#9cb2d6')
lines[0].set_label('Simulated path')
line = plt.plot(forecasts.variance.iloc[-1].values, color='#002868')
line[0].set_label('Expected variance')
legend = plt.legend()'''
'''import datetime as dt
split_date = dt.datetime(2002,11,14)
end = dt.datetime(2025,1,1)
forecasts = res.forecast(horizon=5, start=split_date, method='simulation')
forecasts.variance[split_date:end].plot(figsize=(12,5))
plt.title('预测条件方差',size=15)
print(forecasts.variance.tail())
#forecast 中有三个值，mean - 预测条件均值，variance - 预测条件方差；residual_variance - 预测残差的方差
#print(am2.fit().summary())
print(res.summary())'''

'''res.resid.plot(figsize=(12,5))
plt.title('沪深300收益率拟合GARCH(1,1)残差',size=15)
plt.show()
res.conditional_volatility.plot(figsize=(12,5),color='r')
plt.title('沪深300收益率条件方差',size=15)
plt.show()
#使用tushare获取沪深300交易数据'''

##计算VAR
forecasts = res.forecast(start='2018-1-1')
cond_mean = forecasts.mean['2018':]
cond_var = pd.DataFrame(forecasts.variance['2018':])
q = am2.distribution.ppf([0.01, 0.05])
value_at_risk = -cond_mean.values - np.sqrt(cond_var).values * q[None, :]
value_at_risk = pd.DataFrame(
    value_at_risk, columns=['1%', '5%'], index=cond_var.index)
ax = value_at_risk.plot(legend=False,figsize=(12,5))
xl = ax.set_xlim(value_at_risk.index[0], value_at_risk.index[-1])


rets_2018 = Y['2018':].dropna().copy()
rets_2018.name = 'S&P 500 Return'
c = []
for idx in value_at_risk.index:
    if rets_2018[idx] > -value_at_risk.loc[idx, '5%']:
        c.append('#000000')
    elif rets_2018[idx] < -value_at_risk.loc[idx, '1%']:
        c.append('#BB0000')
    else:
        c.append('#BB00BB')
c = np.array(c, dtype='object')
labels = {
    '#BB0000': '1% Exceedence',
    '#BB00BB': '5% Exceedence',
    '#000000': 'No Exceedence'
}
markers = {'#BB0000': 'x', '#BB00BB': 's', '#000000': 'o'}
for color in np.unique(c):
    sel = c == color  #判别语句
    ax.scatter(
        rets_2018.index[sel], #x
        -rets_2018.loc[sel], #y
        marker=markers[color],
        c=c[sel],
        label=labels[color])
ax.set_title('Parametric VaR')

leg = ax.legend(frameon=False, ncol=3)


#标准差的经验分布
std_rets = (Y[:'2017'] - res.params['mu']) / res.conditional_volatility
std_rets = std_rets.dropna()
q = std_rets.quantile([0.01, 0.05])
value_at_risk = -cond_mean.values - np.sqrt(cond_var).values * q.values[None, :]
value_at_risk = pd.DataFrame(
    value_at_risk, columns=['1%', '5%'], index=cond_var.index)
ax = value_at_risk.plot(legend=False,figsize=(12,5))
xl = ax.set_xlim(value_at_risk.index[0], value_at_risk.index[-1])
rets_2018 = Y['2018':].copy()
rets_2018.name = 'S&P 500 Return'
c = []
for idx in value_at_risk.index:
    if rets_2018[idx] > -value_at_risk.loc[idx, '5%']:
        c.append('#000000')
    elif rets_2018[idx] < -value_at_risk.loc[idx, '1%']:
        c.append('#BB0000')
    else:
        c.append('#BB00BB')
c = np.array(c)
for color in np.unique(c):
    sel = c == color
    ax.scatter(
        rets_2018.index[sel],
        -rets_2018.loc[sel],
        marker=markers[color],
        c=c[sel],
        label=labels[color])
ax.set_title('Filtered Historical Simulation VaR')
leg = ax.legend(frameon=False, ncol=3)


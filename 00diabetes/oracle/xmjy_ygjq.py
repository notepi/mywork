#-*- coding:gbk -*-

'''
    author:huliangjun
    date:2017/10/30
    project name:用工景气预测
'''

import warnings
warnings.filterwarnings("ignore")
import cx_Oracle
import pandas as pd
import sys
import time
import datetime
import numpy as np
reload(sys)
sys.setdefaultencoding('gbk')

def create_dataset(dataset,look_back = 2):
    dataX,dataY = [],[]
    for i in range(len(dataset) - 1):
        a = tuple(dataset[i:(i + look_back)])
        dataX.append(a)
        dataY.append(dataset[i + 1])
    return dataX,dataY

def All_employ_index(database):
    database = cx_Oracle.connect('xmjy', 'xmjy$2017', '192.168.46.77:1521/bigdata')
    sql_month = "select DISTINCT T_MONTH from AB01_WITH_CD01_COUNT_RESULT order by T_MONTH"
    sql_emp_ind = "with DATA201702 as \
                (select NVL(mon1,mon2) as m1,NVL(id1,id2) as id1,nvl(sum1,0) as sum1, \
                NVL(mon2,mon1) as m2,NVL(id2,id1)as id2,NVL(sum2,0) as sum2  \
                from \
                (select t1.t_month as mon1,t1.aab001 as id1,t1.cnt as sum1, \
                t2.t_month as mon2,t2.aab001 as id2,t2.cnt as sum2  \
                from (select * from AB01_WITH_CD01_COUNT_RESULT where t_month = {})t1 \
                FULL join  (select * from AB01_WITH_CD01_COUNT_RESULT where t_month = {})t2 \
                on t1.aab001=t2.aab001)) \
                select t1.sum_all,t2.sum_raise,t3.sum_desc,(sum_raise-sum_desc)/sum_all*100+100 as sindex from \
                (select count(*) as sum_all from DATA201702 t) t1, \
                (select count(*) as sum_raise from DATA201702 t where t.sum1<t.sum2) t2, \
                (select count(*) as sum_desc from DATA201702 t where t.sum1>t.sum2) t3 "
    company_month = pd.read_sql_query(sql_month,con=database)
    company_month = company_month.values.ravel().tolist()
    month_X,month_Y = create_dataset(company_month)
    # month = []
    # for i in month_Y:
    #     i = i + '01'
    #     month.append(i)
    tmp_index = []
    for (i,j) in month_X:
        tmp =  sql_emp_ind.format(i,j)
        company_ygjq = pd.read_sql_query(tmp, con=database)
        tmp_index.append(company_ygjq['SINDEX'].values.tolist()[0])
    index = pd.DataFrame(tmp_index,columns=['SINDEX'])
    t_month = pd.DataFrame(month_Y,columns=['T_MONTH'])
    all_index = pd.concat([t_month,index],axis=1)
    all_index = all_index.set_index('T_MONTH')
    all_index.index.name = None
    return all_index

if __name__ == '__main__':

    db = cx_Oracle.connect('xmjy', 'xmjy$2017', '192.168.46.77:1521/bigdata')
    all_index = All_employ_index(db)
    sql_zc = 'select * from AB01_WITH_CD01_COUNT_RESULT'

    tmp_sql = "with DATA201702 as \
             (select aab, NVL(mon1,mon2) as m1,NVL(id1,id2) as id1,nvl(sum1,0) as sum1,  \
             NVL(mon2,mon1) as m2,NVL(id2,id1)as id2,NVL(sum2,0) as sum2  \
             from  \
             (select t1.aab020 as aab,t1.t_month as mon1,t1.aab001 as id1,t1.cnt as sum1, \
             t2.t_month as mon2,t2.aab001 as id2,t2.cnt as sum2  \
             from (select * from AB01_WITH_CD01_COUNT_RESULT where aab020 is not null )t1  \
             FULL join  (select * from AB01_WITH_CD01_COUNT_RESULT where aab020 is not null)t2  \
             on t1.aab001=t2.aab001 and t1.aab020=t2.aab020 and (t1.t_month+1 = t2.t_month or t1.t_month+89 = t2.t_month)))  \
             select t1aab,t1m,(sum_raise-sum_desc)/sum_all*100+100 as sindex from \
             (select t.aab as t1aab,t.m2 as t1m, count(*) as sum_all from DATA201702 t group by t.aab,m2) t1, \
             (select t.aab as t2aab,t.m2 as t2m,count(*) as sum_raise from DATA201702 t where t.sum1<t.sum2 group by t.aab,m2) t2, \
             (select t.aab as t3aab,t.m2 as t3m,count(*) as sum_desc from DATA201702 t where t.sum1>t.sum2 group by t.aab,m2) t3 \
             where (t1.t1aab=t2.t2aab and t1.t1aab=t3.t3aab) and (t1.t1m=t2.t2m and t1.t1m=t3.t3m) order by t1aab,t1m "

    tmp_all = pd.read_sql_query(tmp_sql,con=db)

    import time
    tmp = time.strptime('201701', '%Y%m')
    tmp_year = tmp.tm_year
    tmp_month = tmp.tm_mon
    date_data = datetime.datetime(tmp_year, tmp_month, 1)
    #时间序列
    a = all_index.reset_index(inplace=True)
    a = all_index['SINDEX']
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    all_index.plot()
    plt.show()

    from statsmodels.graphics.tsaplots import plot_acf
    plot_acf(all_index).show()

    # 平稳性检测
    from statsmodels.tsa.stattools import adfuller as ADF
    print(u'原始序列的ADF检验结果为：', ADF(all_index['SINDEX']))
    # 返回值依次为adf、pvalue、usedlag、nobs、critical values、icbest、regresults、resstore

    # 差分后的结果
    D_data = all_index.diff().dropna()
    D_data.columns = ['SINDEX']
    D_data.plot()  # 时序图
    plt.show()
    plot_acf(D_data).show()  # 自相关图
    from statsmodels.graphics.tsaplots import plot_pacf
    plot_pacf(D_data).show()  # 偏自相关图
    print(u'差分序列的ADF检验结果为：', ADF(D_data['SINDEX']))  # 平稳性检测
    # 白噪声检验
    from statsmodels.stats.diagnostic import acorr_ljungbox
    print(u'差分序列的白噪声检验结果为：', acorr_ljungbox(D_data, lags=1))  # 返回统计量和p值

    from statsmodels.tsa.arima_model import ARIMA
    all_index['SINDEX'] = all_index['SINDEX'].astype(float)
    # 定阶
    pmax = int(len(D_data) / 10)  # 一般阶数不超过length/10
    qmax = int(len(D_data) / 10)  # 一般阶数不超过length/10
    bic_matrix = []  # bic矩阵
    for p in range(pmax + 1):
        tmp = []
        for q in range(qmax + 1):
            try:  # 存在部分报错，所以用try来跳过报错。
                tmp.append(ARIMA(all_index, (p, 1, q)).fit().bic)
            except:
                tmp.append(None)
        bic_matrix.append(tmp)
    bic_matrix = pd.DataFrame(bic_matrix)  # 从中可以找出最小值

    p, q = bic_matrix.stack().idxmin()  # 先用stack展平，然后用idxmin找出最小值位置。
    print(u'BIC最小的p值和q值为：%s、%s' % (p, q))
    model = ARIMA(all_index, (1, 1, 0)).fit()  # 建立ARIMA(0, 1, 1)模型
    model.summary2()  # 给出一份模型报告
    model.forecast(10)[0]  # 作为期5天的预测，返回预测结果、标准误差、置信区间。
    ax = all_index.plot()
    fig = model.predict('2017-10-01', '2018-05-01', dynamic=True)
    plt.show()
    # sql_ygjq = "select NVL(mon1,mon2) as m1,NVL(id1,id2) as id1,nvl(sum1,0) as sum1, \
    #            NVL(mon2,mon1) as m2,NVL(id2,id1)as id2,NVL(sum2,0) as sum2 from \
    #            (select t1.t_month as mon1,t1.aab001 as id1,t1.cnt as sum1, \
    #            t2.t_month as mon2,t2.aab001 as id2,t2.cnt as sum2 \
    #            from (select * from AB01_WITH_CD01_COUNT_RESULT where t_month = '201701')t1 \
    #            FULL join  (select * from AB01_WITH_CD01_COUNT_RESULT where t_month = '201702')t2 \
    #            on t1.aab001=t2.aab001 )"
    # company = pd.read_sql_query(sql_zc, con=db)

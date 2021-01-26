# -*- coding: utf-8 -*-
import pandas as pd
import math
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# plt.rcParams['font.sans-serif'] = ['SimHei']  # 画图时用来正常显示中文标签
# plt.rcParams['axes.unicode_minus'] = False  # 画图时用来正常显示负号


def division(numerator, denominator):  # 第一个参数是分子，第二个参数是分母
    if (float(denominator) == 0):
        return 0
    else:
        return (float(numerator) / float(denominator))


def IV(y, x, binnum, interval, xName, rulesetid, ruleid, con_mode, q_mode, sort):
    # y是标签值，x是特征值，dataframe
    # binnum是
    # interval是list类型，指间隔点向量
    # xName是画图时特征的名字
    # rulesetid特征集or规则
    # ruleid规则
    # con_mode判断是否人工去判断是否是连续型变量,con_mode == 0 的时候肯定是离散型变量，con_mode==1的时候不一定是连续型变量
    # q_mode判断是等人数分还是等间距分
    # sort == 1的时候是按 woe排序

    newpd = pd.concat([y, x], axis=1)  # 把两列拼接成一个新的dataframe
    # distinct_x_num=len(x.unique())#unique函数相当于distinct,求得x
    distinct_x_num = 50
    # 判断变量为连续or离散，并进行分箱
    if (distinct_x_num > binnum) and con_mode == 1:
        # print("连续性变量")
        if q_mode == 1:
            newpd['bin_cat'] = pd.qcut(x.astype(float), interval,
                                       duplicates='drop')  # bin的类型是<class 'pandas.core.series.Series'>
        else:
            newpd['bin_cat'] = pd.cut(x.astype(float), interval)  # bin的类型是<class 'pandas.core.series.Series'>
        #       bin_cat = newpd['bin_cat'].astype(object)# 不转换类型跑不了，可以转换成object，但是这样的话后面的a.index就会变成Interval类型，所以这里转成str类型，a.index就是object类型
        bin_cat = newpd['bin_cat'].astype(object)

        # 如果作为object跑，不会出现nan这个分类，但是result.index会成为interval类型，这样就可以通过 result.index[i].right获得右边界

        a = x[y == 1].groupby(bin_cat).count()
        # NullValuey1 = pd.Series(len(x[x.isnull()==True][y == 1]), index=['NullValue'])#x列值是空且y==1
        # a = a.append(NullValuey1)
        b = x[y == 0].groupby(bin_cat).count()
        # b=b.append(NullValuey0)
    else:
        # print("离散型变量")
        a = x[y == 1].groupby(x).count()
        b = x[y == 0].groupby(x).count()

    result = pd.concat([a, b], axis=1)
    # result=result.sort_values(result.index,ascending=True)#防止莫名其妙的最后一段不是最大的
    if 'nan' in result.index:
        result = result.drop('nan')  # 如果x列有空数据，则删除index为nan的那一行
    result.columns = ['1', '0']
    result['woe'] = ((result['1'] / result['1'].sum()) / (result['0'] / result['0'].sum())).map(
        lambda x: math.log(x))  # 以e位
    result['iv'] = result['woe'] * (result['1'] / result['1'].sum() - result['0'] / result['0'].sum())
    result['1'] = result['1'].replace(np.nan, 0.0)
    result['0'] = result['0'].replace(np.nan, 0.0)
    result['sum_people'] = result['1'] + result['0']
    result['labels'] = result['0'].astype(str) + '/' + result['1'].astype(str)  # 柱状图标注
    result = result.reset_index()
    result.columns = ['bin_cat', '1', '0', 'woe', 'iv', 'sum_people', 'labels']
    result.index = result['bin_cat']
    # result结果展示校正
    # result=result[(result['woe']!=0)&(pd.notnull(result['woe']))]
    # result_sort=result.sort_values('woe', ascending=False)
    # result=result.sort_values('bin_cat', ascending=False)
    # print(result)
    iv_value = result['iv'].sum()

    return result, iv_value


def fengedian(x, k, d, n):
    # 计算给定series的分隔点list,
    # k为分隔点个数，段数+1
    # d为分位数，d分位数后并为一个区间
    # n为空值处理，n=1空值放在最前面，n=2空值放在最后面
    # x.drop(pd.isnull(x))
    xmin = x.min()
    xmax = x.quantile(d)  # 取99%
    if n == 1:
        if ((xmax - xmin) >= k):
            fg = math.ceil((xmax - xmin) / k)
            fglist = [xmin + fg * i for i in range(1, k + 1)]
            fglist.insert(0, xmin - 0.01)
            fglist.insert(0, -101)
            fglist.insert(1, -100)
            if (xmax != x.max()) & (x.max() > fglist[-1]):
                fglist.append(x.max())
        else:
            fg = (xmax - xmin) / k
            fglist = [xmin + fg * i for i in range(1, k + 1)]
            fglist.insert(0, xmin - 0.01)
            fglist.insert(0, -101)
            fglist.insert(1, -100)
            if (xmax != x.max()) & (x.max() > fglist[-1]):
                fglist.append(x.max())
    if n == 2:
        if ((xmax - xmin) >= k):
            fg = math.ceil((xmax - xmin) / k)
            fglist = [xmin + fg * i for i in range(1, k + 1)]
            fglist.insert(0, xmin - 0.01)
            if (xmax != x.max()) & (x.max() > fglist[-1]):
                fglist.append(x.max())
            fglist.append(1000000 - 10)
            fglist.append(1000000 + 10)
        else:
            fg = (xmax - xmin) / k
            fglist = [xmin + fg * i for i in range(1, k + 1)]
            fglist.insert(0, xmin - 0.0001)
            if (xmax != x.max()) & (x.max() > fglist[-1]):
                fglist.append(x.max())
            fglist.append(1000000 - 10)
            fglist.append(1000000 + 10)
    return fglist


if __name__ == '__main__':

    data = pd.read_csv('/BioII/lulab_b/liuxiaofan/project/pico_0210/other_data/SNP/discovery_1/discovery_data_gene_all.csv',sep='\t')
    data= data.set_index('feature')
    sample_label_all = pd.read_csv('/BioII/lulab_b/liuxiaofan/project/pico_feature_select/pico_0315_count/data/ALL/sample_classes.txt',sep='\t')
    batch_info_all = pd.read_csv('/BioII/lulab_b/liuxiaofan/project/pico_feature_select/pico_0315_count/data/ALL/batch_info.txt',sep='\t')

    data_ = data.T.reset_index()
    data_ = data_.rename(columns={'index':'sample_id'})

    data_ = pd.merge(data_,sample_label_all,on='sample_id',how='left')
    data_NC = data_.loc[data_['label']=='NC',:]
    data_CRC = data_.loc[data_['label']=='CRC',:]
    data_HCC = data_.loc[data_['label']=='HCC',:]
    data_ESCA = data_.loc[data_['label']=='ESCA',:]
    data_LUAD = data_.loc[data_['label']=='LUAD',:]
    data_STAD = data_.loc[data_['label']=='STAD',:]

    newdata = pd.concat([data_NC,data_LUAD]).replace('NC',0).replace('LUAD',1)
    newdata = newdata.set_index('sample_id')
    feature = newdata.sum(axis=0)
    feature = list(set(feature.loc[feature>0].index)-set('label'))+['label']
    newdata =  newdata[feature]
    newdata = newdata.reset_index()


    resultList = []
    for i in range(1, len(newdata.T)-1):
        if i%1000==0:
            print(i)
        htdata = newdata
        x = htdata.iloc[:, i]
        try:
            if ((x.max() - x.min()) > 0):

                listi = [-0.1, 0, 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1]
                x1 = x.copy()
                x1[pd.isnull(x1)] = -1
                Name = list(htdata.columns)[i]
                result, iv_value = IV(y=htdata.iloc[:, -1], x=x1, binnum=15, interval=listi, rulesetid='1',
                                                   ruleid='1', xName=Name, con_mode=1, q_mode=0, sort=0)
                resultList.append([Name, iv_value])
            else:
                Name = list(htdata.columns)[i]
                iv_value = 0
                resultList.append([Name, iv_value])
        except ValueError:
            Name = list(htdata.columns)[i]
            iv_value = 0
            resultList.append([Name, iv_value])

    resultDf = pd.DataFrame(resultList)
    resultDf.columns = ['columnName', 'IV_value']
    resultDf.to_csv('/BioII/lulab_b/liuxiaofan/project/pico_0210/other_data/SNP/discovery_IV_gene_all_LUAD.csv', sep=',', header=True, index=True)

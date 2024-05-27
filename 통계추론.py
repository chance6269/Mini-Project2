# -*- coding: utf-8 -*-
"""
Created on Thu May 23 00:01:50 2024

@author: pjc62
"""

import pandas as pd
data = '매매_실거래가격(비교)_수정_v8.xlsx'
# df = pd.read_excel(data)
# df = pd.read_excel(data,parse_dates=['시점'])
df = pd.read_excel(data,index_col='연도_월')


# %%
df.index

# float형인 날짜열을 str형으로 변경 + 10월 처리
df.index = df.index.astype(str).map(lambda x: x + '0' if len(x) < 7 else x)
df.columns
data = df.iloc[:,1:]

target = df.loc[:,['실거래가격지수']]

# %%
from scipy import stats
stats.zscore(target)

stats.norm.cdf(0)

# %%
'''
상관관계분석
'''
# 상관관계 행렬
df_corr = df.corr()


# %%
def reg():
    import matplotlib.font_manager as fm
    import os

    user_name = os.getlogin()

    fontpath = [f'C:/Users/{user_name}/AppData/Local/Microsoft/Windows/Fonts']
    font_files = fm.findSystemFonts(fontpaths=fontpath)
    for fpath in font_files:
        
        fm.fontManager.addfont(fpath)

# %%
import matplotlib.pyplot as plt
import seaborn as sns
reg()
plt.rc('font', family='NanumBarunGothic')
plt.rcParams['figure.dpi'] = 140


# %%
# 변수 간 상관 관계 분석
 # 실거래가격지수와 상관 관계가 높은 순서대로 정리
 
corr_order = df.corr().loc['대통령':,'실거래가격지수'].abs().sort_values(ascending=False)
corr_order
# 1인가구            0.916259
# 무주택가구_수         0.909575
# 종합부동산세_세율_개인    0.902650
# 혼인건수            0.898497
# 4인가구_이상         0.895094
# 2인가구            0.892621
# 4인가구            0.887294
# 국회의석수_진보        0.878452
# 외국인_장기체류        0.866395
# 주택부담구입지수        0.859785
# 이혼건수            0.842368
# 국회의석수_보수        0.842306
# 소비자물가지수         0.819332
# 대통령             0.771748
# 1인당_주거면적        0.632280
# PIR지수_전국        0.630363
# 자가보유율           0.615471
# 자가점유율           0.564836
# LIR지수_전국        0.526436
# 3인가구            0.522893
# LIR지수_서울        0.467044
# 주택매매거래량         0.405078
# 지지율             0.342276
# 아파트매매거래량        0.298548
# 예적금담보대출         0.283495
# PIR지수_서울        0.277978
# 주택담보대출          0.209577
# 매매수급동향          0.157530
# 기준금리            0.131498
# 토지매매거래량         0.086382
# 순수토지매매거래량       0.017562
# Name: 실거래가격지수, dtype: float64
corr_order.index

# %%
# 실거래가격지수 분포
sns.displot(x='실거래가격지수',kind='hist', data=df)
plt.show()
    # 좌편향인걸 확인
    


# -*- coding: utf-8 -*-
"""
Created on Sat May 25 02:29:52 2024

@author: pjc62
"""


import pandas as pd

def pop_price_df(loc):
    
    '''
    전처리 - 아파트 매매 실거래가격지수
    '''
    # 파일 전처리
    file = './data/아파트_매매_실거래가격지수_1301-2403.xlsx'
    
    df = pd.read_excel(file, header=0)
    df2 = pd.read_excel(file, header=1)

    # float형인 '시점' 열을 str형으로 변경 + 10월 처리
    df2['시점'] = df2['시점'].astype(str).map(lambda x: x + '0' if len(x) < 7 else x)
    
    # 컬럼명 수정
    new_columns = list(df2.columns)
    new_columns[2:5] = df.columns[2:5]
    new_columns[17] = df.columns[17]
    
    df2.columns = new_columns
    
    apt_mean = df2.copy()
    
    apt_mean.set_index('시점', inplace=True)
    loc_mean = apt_mean[[loc]]
    loc_mean.columns = ['실거래지수']
    
    return loc_mean.iloc[:-1, :]

# %%
df = pop_price_df('세종')
# file = './data/아파트_매매_실거래가격지수_1301-2403.xlsx'

# df = pd.read_excel(file, header=0)
# df.columns[17]

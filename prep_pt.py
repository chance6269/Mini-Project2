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


def pop_mov_df(loc):
    '''
    전처리 - 인구이동 데이터
    '''
    pop_mov = pd.read_excel('./data/시군구별_이동데이터/시군구별_이동자수_1301-2403.xlsx')
    loc_names = {'경북':'경상북도','경남':'경상남도','전북':'전북특별자치도','전남':'전라남도','충북':'충청북도','충남':'충청남도'}
    if loc in loc_names.keys():
        loc = loc_names[loc]
    # 지역 데이터 선택
    selected_col = pop_mov.filter(like=loc)
    loc_mov = pd.merge(pop_mov['시점'], selected_col, left_index=True, right_index=True)
    
    # 열 이름 지정
    loc_mov.columns = loc_mov.iloc[0]
     
    loc_mov = loc_mov.iloc[1:]
    
    loc_mov.set_index('시점', inplace=True)
    
    return loc_mov
    

def make_data(loc):
    
    loc_mean = pop_price_df(loc)
    
    loc_mov = pop_mov_df(loc)
    
    # 병합
    df_prep = pd.merge(loc_mean, loc_mov, how='outer', left_index=True, right_index=True)
    
    # 엑셀파일로 저장
    res_folder = './data/result/'
    df_prep.to_excel(res_folder+'{}_월별_실거래지수_이동데이터.xlsx'.format(loc))

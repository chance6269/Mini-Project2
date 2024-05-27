# -*- coding: utf-8 -*-
"""
Created on Fri May 24 23:41:42 2024

@author: pjc62
"""

'''
실거래매매 데이터 전처리 모듈
'''

def prep(file):
    # 엑셀 파일 읽기
    import pandas as pd
    
    df = pd.read_excel(file)
    
    # 불필요한 행 제거
    df = df[11:]
    
    # 열 이름 지정
    df.columns = df.iloc[0]
    
    df = df.iloc[1:]
    
    # 인덱스 설정
    # df.index = df['NO']
    df.set_index('NO', inplace=True)
    

    # 중복된 행 제거
    df = df.drop_duplicates()
    
    '''
    월별 평균 거래금액 집계하기
    '''
    # 계약년월, 거래금액(만원) 이외 열 제거, 결측치 처리
    ndf = df[['계약년월','거래금액(만원)']].dropna()
    
    ndf.loc[:,'거래금액(만원)'] = ndf['거래금액(만원)'].str.replace(',', '').astype(int)
    group_df = ndf.groupby(by='계약년월',dropna=False)
    return group_df.mean()

# -*- coding: utf-8 -*-
"""
Created on Sat May 25 02:29:52 2024

@author: pjc62
"""


def prep(loc):
    '''
    전처리 - 아파트 월별 평균 거래가
    '''
    import pandas as pd
    import os
    import preprocess as pp
    
    # 특정 디렉토리 지정
    target_dir = "./data/아파트매매_실거래가/{}/".format(loc)
    
    # 파일 목록 읽기
    file_list = os.listdir(target_dir)
    
    # folder_path = target_dir+'/{}/'.format(loc)
    folder_path = target_dir
    # 파일 전처리
    price_mon = pd.DataFrame()
    for file in file_list:
        file = folder_path + file
        if len(price_mon) == 0:
            price_mon = pp.prep(file)
        else:
            price_mon=pd.concat([price_mon, pp.prep(file)])
            
    price_mon.rename(columns={'거래금액(만원)':'월_평균거래가(만원)'}, inplace=True)
    
    # 결과물 폴더경로
    res_folder = './data/result/'
    # 엑셀파일로 저장
    price_mon.to_excel(res_folder + '{}_월별_평균거래가.xlsx'.format(loc))
    
    '''
    전처리 - 인구이동 데이터
    '''
    pop_mov = pd.read_excel('./data/시군구별_이동데이터/{}_시군구별_이동자수_1301-2403.xlsx'.format(loc),header=1)
    
    # float형인 날짜열을 str형으로 변경 + 10월 처리
    pop_mov['시점'] = pop_mov['시점'].astype(str).map(lambda x: x + '0' if len(x) < 7 else x)
    pop_mov['시점'] = pop_mov['시점'].str.replace('.','')
    pop_mov.set_index('시점', inplace=True)
    
    
    # 데이터 병합
    df_prep = pd.merge(price_mon, pop_mov, how='outer', left_index=True, right_index=True)
    
    # 엑셀파일로 저장
    df_prep.to_excel(res_folder+'{}_월별_평균거래가_이동데이터.xlsx'.format(loc))

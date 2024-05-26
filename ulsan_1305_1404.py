# -*- coding: utf-8 -*-
"""
Created on Fri May 24 23:41:42 2024

@author: pjc62
"""

'''
울산광역시 실거래매매 데이터 전처리
'''

# test
# 엑셀 파일 읽기
import pandas as pd

folder_path = './data/아파트매매_실거래가/'
file='아파트(매매)_실거래가_1305-1404.xlsx'
df = pd.read_excel(folder_path + file)

# %%
# 불필요한 행 제거
df = df[11:]

# 열 이름 지정
df.columns = df.iloc[0]

df = df.iloc[1:]

# %%
# 인덱스 설정
# df.index = df['NO']
df.set_index('NO', inplace=True)

# %%
'''
결측치 확인
'''
df.info()

# 결측치 없음 확인
# %%
'''
중복데이터 제거
'''
# 중복값 확인
sum(df.duplicated()) # 93

dup_rows = df.duplicated(keep=False)

dup_data = df[dup_rows]

# 중복된 행 제거
ndf = df.drop_duplicates()
ndf.shape # (20059, 20)

# 중복값인 93개 행이 제거됨.

# %%
# 불필요열 제거
# 데이터 구분에 필요한 열 : 시군구, 번지, 단지명, 전용면적, 계약년월, 계약일, 거래금액, 동, 층, 건축년도
selected_columns = ['시군구','번지','단지명','전용면적(㎡)','계약년월','계약일','거래금액(만원)','동','층','건축년도']

ndf2 = ndf.loc[:,selected_columns]

# %%
# 불필요열 제거 상태로 다시 중복값 확인
sum(ndf2.duplicated()) # 8

# 왜 첫 확인에서 걸러지지 않은걸까? raw 데이터를 확인해보기로 함.
dup_raw = df.loc[ndf2.loc[ndf2.duplicated(keep=False),:].index,:]

# 계약일, 거래금액, 전용면적 등 다른 항목은 같지만 도로명만 매산로 65, 매산로 66으로 다른 행들이 존재.
# 오타로 보고 중복처리할 것인가, 각기 다른 데이터로 볼 것인가 선택해야함.

# [추가]
# 국토교통부에 중복데이터인지, 아닌지 확인이 필요!!

# %%
'''
월별 평균 거래금액 집계하기
'''
# 계약년월, 거래금액(만원) 이외 열 제거
ndf3 = df[['계약년월','거래금액(만원)']]
ndf3.head()
ndf3.loc[:,'거래금액(만원)'] = ndf3['거래금액(만원)'].str.replace(',', '').astype(int)
group_df = ndf3.groupby(by='계약년월',dropna=False)
price_mon = group_df.mean()

print(price_mon)
'''
11          거래금액(만원)
계약년월                
201305  21538.979021
201306   20967.66323
201307  19829.662289
201308  19929.444177
201309  20950.577025
201310  21096.352459
201311  20451.758118
201312  22095.554795
201401  20388.355795
201402  20199.864286
201403  20541.496436
201404  21301.457831
'''
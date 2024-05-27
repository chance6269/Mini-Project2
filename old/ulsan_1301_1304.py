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
file='아파트(매매)_실거래가_1301-1304.xlsx'
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
# 
df.info()
'''
Index: 5950 entries, 1 to 5950
Data columns (total 20 columns):
 #   Column    Non-Null Count  Dtype 
---  ------    --------------  ----- 
 0   시군구       5950 non-null   object
 1   번지        5950 non-null   object
 2   본번        5950 non-null   object
 3   부번        5950 non-null   object
 4   단지명       5950 non-null   object
 5   전용면적(㎡)   5950 non-null   object
 6   계약년월      5950 non-null   object
 7   계약일       5950 non-null   object
 8   거래금액(만원)  5950 non-null   object
 9   동         5950 non-null   object
 10  층         5950 non-null   object
 11  매수자       5950 non-null   object
 12  매도자       5950 non-null   object
 13  건축년도      5950 non-null   object
 14  도로명       5950 non-null   object
 15  해제사유발생일   5950 non-null   object
 16  거래유형      5950 non-null   object
 17  중개사소재지    5950 non-null   object
 18  등기일자      5950 non-null   object
 19  주택유형      5950 non-null   object
dtypes: object(20)
'''
# 총 데이터갯수는 5950이며, 누락값이 없음을 확인

# %%
'''
중복데이터 제거
'''
# 중복값 확인
sum(df.duplicated()) # 54

dup_rows = df.duplicated(keep=False)

dup_data = df[dup_rows]
dup_data.shape # (94, 20)
# 중복된 행 제거
ndf = df.drop_duplicates()
ndf.shape # 5896, 20
# 94개의 행 중 고유값(첫 값) 40개를 남기고 
# 중복값인 54개 행이 제거됨.

# %%
# 불필요열 제거
# 데이터 구분에 필요한 열 : 시군구, 번지, 단지명, 전용면적, 계약년월, 계약일, 거래금액, 동, 층, 건축년도
selected_columns = ['시군구','번지','단지명','전용면적(㎡)','계약년월','계약일','거래금액(만원)','동','층','건축년도']

ndf2 = ndf.loc[:,selected_columns]

# %%
# 불필요열 제거 상태로 다시 중복값 확인
sum(ndf2.duplicated()) # 4

# 왜 첫 확인에서 걸러지지 않은걸까? raw 데이터를 확인해보기로 함.
dup_raw = df.loc[ndf2.loc[ndf2.duplicated(keep=False),:].index,:]

# 계약일, 거래금액, 전용면적 등 다른 항목은 같지만 도로명만 매산로 65, 매산로 66으로 다른 행들이 존재.
# 오타로 보고 중복처리할 것인가, 각기 다른 데이터로 볼 것인가 선택해야함.

# 검색해본 결과 '월드메르디앙월드시티'는 단지에 따라 864번지, 865번지로 나뉘어있으며,
# 도로명 주소가 864(매산로 65), 865(매산로 66)임을 확인.

# 따라서 현재 데이터에 다른 865번지 데이터가 있다면 도로명 입력 오타로 중복일 것이고,
# 없다면 주소 개편 이전에 864 하나의 번지를 사용했거나 865 번지 입력 오타일 것이라 가정.

# %%
df.loc[df['단지명'] == '월드메르디앙월드시티','번지'].unique() # array(['864'], dtype=object)

# 고유값은 1개로 확인되고, 864인 것으로 확인
# 이에 따라 후자로 보고 각기 다른 데이터로 구분하여 남기기로 함.

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

price_mon.head()
'''
11          거래금액(만원)
계약년월                
201301  20541.237776
201302  21244.608491
201303  21588.926626
201304  21748.837691
'''
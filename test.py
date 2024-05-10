# -*- coding: utf-8 -*-
"""
Created on Fri May 10 15:42:40 2024

@author: jcp
"""

import pandas as pd


data = pd.read_excel('경기도_아파트(매매)_실거래가_20240508.xlsx')

# %%
data = data.iloc[11:,:]

# %%
# 열 이름 주기

data.columns = data.iloc[0]

data = data.iloc[1:]

# %%

# 인덱스 변경
data.set_index('NO', inplace=True)

# 불필요한 열 제거
data.columns
drop_col = ['본번', '부번', '매수자', '매도자']

data.drop(drop_col, axis=1, inplace=True)

df = data


# %%

df['시군구'] = df['시군구'].astype('str')


# %%

# Identify rows with '시' but not '시흥시'
si_no_sihung = df[~df['시군구'].str.contains('시흥시')]

# Handle rows with '시' and not '시흥시'
df.loc[si_no_sihung.index, '시'] = df.loc[si_no_sihung.index, '시군구'].str.split('시').str[0] + '도'
df.loc[si_no_sihung.index, '구'] = df.loc[si_no_sihung.index, '시군구'].str.split('시').str[1:]
df.loc[si_no_sihung.index, '구'] = df.loc[si_no_sihung.index, '구'].str.replace('동', '').str[0]
df.loc[si_no_sihung.index, '시군'] = df.loc[si_no_sihung.index, '시'] + df.loc[si_no_sihung.index, '구']
df.loc[si_no_sihung.index, '구'] = df.loc[si_no_sihung.index, '구'] + df.loc[si_no_sihung.index, '동'].fillna('')

# Handle rows with '시흥시'
df.loc[df['시군구'].str.contains('시흥시'), '시'] = '시흥시'
df.loc[df['시군구'].str.contains('시흥시'), '구'] = ''
df.loc[df['시군구'].str.contains('시흥시'), '시군'] = '시흥시'
df.loc[df['시군구'].str.contains('시흥시'), '구'] = df.loc[df['시군구'].str.contains('시흥시'), '구']

# Drop '시군구' column
df = df.drop('시군구', axis=1)

print(df)

# %%
# Define a function to handle '시흥시' rows
def handle_sihung(row):
    if row.startswith('시흥시'):
        return row.split('시')[1:]
    else:
        return row.split('시')[0] + '도', row.split('시')[1:]

# Apply the function to '시군구' column
df[['시', '구']] = df['시군구'].apply(lambda x: handle_sihung(x))

# Handle '구' column further
df['구'] = df['구'].str.replace('동', '').str[0]
df['시군'] = df['시'] + df['구']
df['구'] = df['구'] + df['동'].fillna('')

# Drop '시군구' column if needed
# df = df.drop('시군구', axis=1)

print(df)
# %%
df['시'].unique()

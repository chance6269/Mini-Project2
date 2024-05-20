# -*- coding: utf-8 -*-
"""
Created on Mon May 20 17:31:18 2024

@author: jcp
"""

import matplotlib.pyplot as plt
import pandas as pd

# 데이터 불러오기 및 DataFrame 변환
data_file = "소비자물가지수_2020100__20240520121328.csv"
df = pd.read_csv(data_file, parse_dates=['시점'], infer_datetime_format=True)
df.info()
# %%

df['시점'] = df['시점'].astype(str).str.replace('-','/')
df['시점'] = df['시점'].str[2:-3]

# %%
# 전체 데이터 그래프
plt.figure(figsize=(16, 6))

plt.plot(df['시점'],  df['전국'])
# 그래프 제목 및 라벨 설정
plt.xticks(range(0,133,3),df['시점'][::3],rotation=45)
plt.title('전국 소비자물가지수 추이')
plt.xlabel('시점')
plt.ylabel('소비자물가지수')

# 그래프 표시

plt.show()

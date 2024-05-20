# -*- coding: utf-8 -*-
"""
Created on Mon May 20 17:31:18 2024

@author: jcp
"""

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.dates import MonthLocator, DateFormatter

# 데이터 불러오기 및 DataFrame 변환
data_file = "소비자물가지수_2020100__20240520121328.csv"
df = pd.read_csv(data_file, parse_dates=['시점'], infer_datetime_format=True)

# 날짜 형식 변환 및 Year, Month 열 추가
df['시점'] = pd.to_datetime(df['시점'], format='%Y.%m')
df['Year'] = df['시점'].dt.year
df['Month'] = df['시점'].dt.month

# 전체 데이터 그래프
fig, ax = plt.subplots(figsize=(15, 6))
ax.plot(df['시점'], df['전국'])

# 3개월 간격 x 축 눈금 및 라벨 설정
locator = MonthLocator(bymonth=3)  # 3개월 간격으로 눈금 표시
formatter = DateFormatter("%b %Y")  # 월 이름과 연도 표시
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(formatter)

# 그래프 제목 및 라벨 설정
plt.title('전국 소비자물가지수 추이')
plt.xlabel('시점')
plt.ylabel('소비자물가지수')

# 그래프 표시
plt.grid(True)
plt.tight_layout()
plt.show()

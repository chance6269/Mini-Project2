# -*- coding: utf-8 -*-
"""
Created on Fri May 17 15:30:45 2024

@author: jcp
"""

import pandas as pd
import matplotlib.pyplot as plt


# Read the CSV data into a DataFrame
data = pd.read_csv('한국은행 기준금리 및 여수신금리_17153127.csv', header=0, parse_dates=True)
data.columns = ['날짜', '기준금리']
# data.set_index('날짜')
# %%
# Extract the base rate values
dates = data['날짜']
values = data['기준금리']

# %%
# 문자열 날짜를 datetime 객체로 변환
import matplotlib.dates as mdates
from datetime import datetime

dates = [datetime.strptime(date, "%Y/%m") for date in dates]

# %%

# 폰트 경로
import add_font as af
af.reg()

# 폰트 설정
plt.rc('font', family='NaNumBarunGothic')
plt.rcParams['figure.dpi'] = 140
# %%
# 선 그래프 그리기
plt.figure(figsize=(16, 6))

plt.plot(dates, values)


# x축 눈금을 45도 회전
plt.xticks(rotation=45)

plt.title('한국 기준금리 월별 변화 (2013년 4월 ~ 2024년 5월)')
plt.xlabel('날짜')
plt.ylabel('기준금리')

plt.show()



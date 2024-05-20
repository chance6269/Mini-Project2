# -*- coding: utf-8 -*-
"""
Created on Fri May 17 15:30:45 2024

@author: jcp
"""

import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV data into a DataFrame
data = pd.read_csv('소비자물가지수_2020100__20240520121328.csv', header=0, parse_dates=True)
data.columns = ['날짜', 'CPI', '서울CPI','경기도CPI']
# data.set_index('날짜')

# %%
# 문자열 날짜를 datetime 객체로 변환
# 'Date' 열의 형식을 'YYYY.MM'으로 통일
data['날짜'] = data['날짜'].apply(lambda x: f"{int(x)}.{int((x*100) % 100):02d}")
from datetime import datetime

data['날짜'] = pd.to_datetime(data['날짜'], format='%Y.%m').dt.to_period('M').astype(str)
data.info()
# %%
# 폰트 경로
def fontpath():
    
    import os

    user_name = os.getlogin()

    return [f'C:/Users/{user_name}/AppData/Local/Microsoft/Windows/Fonts']

# %%
def reg_path():
    import matplotlib.font_manager as fm

    path = fontpath()
    font_files = fm.findSystemFonts(fontpaths=path)
    for fpath in font_files:
        
        fm.fontManager.addfont(fpath)
# %%

reg_path()
# font_path.reg_path()

# 폰트 설정
plt.rc('font', family='NaNumBarunGothic')
plt.rcParams['figure.dpi'] = 140
# %%
import matplotlib.dates as mdates
# 선 그래프 그리기
plt.figure(figsize=(16, 6))

plt.plot(data['날짜'], data['CPI'])

# x축 레이블 설정
xticks = pd.date_range(start=data['날짜'].min(), end=data['날짜'].max(), freq='3MS')
xtick_labels = [x.strftime('%y/%m') for x in xticks]

plt.xticks(ticks=mdates.date2num(xticks), labels=xtick_labels, rotation=45, fontsize=10)


# x축 범위 설정 (예: 데이터의 최대 날짜까지만 표시)
plt.xlim(data['날짜'].min(), data['날짜'].max())


plt.title('한국 소비자물가지수 월별 변화 (2013년 4월 ~ 2024년 4월)')
plt.xlabel('날짜')
plt.ylabel('CPI')

plt.show()



# -*- coding: utf-8 -*-
"""
Created on Sat May 25 03:32:31 2024

@author: pjc62
"""

'''
시각화
'''
import pandas as pd
# 전처리 파일 로드
folder_path = "./data/아파트매매_실거래가/"
file = '월별_평균거래가_이동데이터.xlsx'
df = pd.read_excel(folder_path+file, index_col=0)

corr = df.corr()

corr_order = df.corr().loc['총전입 (명)':,'월_평균거래가(만원)'].abs().sort_values(ascending=False)
corr_order

'''
시도내이동-시군구내 (명)       0.554443
총전입 (명)              0.551309
총전출 (명)              0.499931
시도내이동-시군구간 전입 (명)    0.484647
시도내이동-시군구간 전출 (명)    0.484647
시도간전입 (명)            0.440244
순이동 (명)              0.303706
시도간전출 (명)            0.200308
Name: 월_평균거래가(만원), dtype: float64
'''
# %%
import matplotlib.pyplot as plt
import seaborn as sns

def reg():
    import matplotlib.font_manager as fm
    import os

    user_name = os.getlogin()

    fontpath = [f'C:/Users/{user_name}/AppData/Local/Microsoft/Windows/Fonts']
    font_files = fm.findSystemFonts(fontpaths=fontpath)
    for fpath in font_files:
        
        fm.fontManager.addfont(fpath)
# %% 
plt.rc('font', family='NanumBarunGothic')
plt.rcParams['figure.dpi'] = 140
reg()

# %%
# 상관계수 히트맵
plt.figure(figsize=(20, 20))  # 그래프 크기 설정

sns.heatmap(data=corr, annot=True, 
            fmt='.2f', linewidths=.5, cmap='Blues', 
            annot_kws={"size": 20})  # 상관 계수 글꼴 크기 확대

# x축, y축 레이블 크기 조정
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.show()
# %%
# 월평균가 분포
sns.displot(x='월_평균거래가(만원)',kind='hist', data=df)
plt.show()

# %%

'''
피처 스케일링
'''
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()

df_scaled = df.iloc[:,:]
scaler.fit(df_scaled)
df_scaled = scaler.transform(df_scaled)

df.iloc[:, :] = df_scaled[:,:]
df.head()
# %%
# 선 그래프 그리기
plt.figure(figsize=(16,10))

df.index = df.index.astype(str).str[2:4]+ "/" + df.index.astype(str).str[4:]

fig, ax = plt.subplots(figsize=(16,8))
ax.plot(df.index, df['월_평균거래가(만원)'], label='월평균가')
# ax.plot(df.index, df['순이동 (명)'],label='순이동')
# ax.plot(df.index, df['총전입 (명)'],label='전입',color='green')
ax.plot(df.index, df['총전출 (명)'], label='전출', color='red')
ax.legend()
plt.xticks(range(0,136,3),df.index[::3],rotation=45)
fig.show()


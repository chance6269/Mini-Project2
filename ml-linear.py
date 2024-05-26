# -*- coding: utf-8 -*-
"""
Created on Sat May 25 04:39:53 2024

@author: pjc62
"""

'''머신러닝'''

import pandas as pd
# 전처리 파일 로드
loc = '세종'
folder_path = "./data/result/"
file = '{}_월별_평균거래가_이동데이터.xlsx'.format(loc)
df = pd.read_excel(folder_path+file, index_col=0)


# %%
'''
피처 스케일링
'''
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()

df_scaled = df.iloc[:, 1:]
scaler.fit(df_scaled)
df_scaled = scaler.transform(df_scaled)

df.iloc[:, 1:] = df_scaled[:,:]
df.head()

# %%
from sklearn.model_selection import train_test_split
# cols_selected = ['순이동 (명)']
# X_data = df.loc[:, cols_selected].dropna()
X_data = df.drop('월_평균거래가(만원)', axis=1).dropna()
y_data = df.loc[:, '월_평균거래가(만원)']
y_data = y_data[:len(X_data)]
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, shuffle=True)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

# %%

'''
베이스라인 모델 - 선형회귀
'''
from sklearn.linear_model import LinearRegression
import numpy as np
lr = LinearRegression()
lr.fit(X_train, y_train)

print('결정계수:',lr.score(X_test,y_test))
print('회귀계수:',np.round(lr.coef_, 1))
print('상수항:',np.round(lr.intercept_, 1))

# %%
# 예측
y_test_pred = lr.predict(X_test)


# %%
def reg():
    import matplotlib.font_manager as fm
    import os

    user_name = os.getlogin()

    fontpath = [f'C:/Users/{user_name}/AppData/Local/Microsoft/Windows/Fonts']
    font_files = fm.findSystemFonts(fontpaths=fontpath)
    for fpath in font_files:
        
        fm.fontManager.addfont(fpath)

# %%
# plt객체 설정
import matplotlib.pyplot as plt
import seaborn as sns
reg()
plt.rc('font', family='NanumBarunGothic')
plt.rcParams['figure.dpi'] = 140

# 실제값/예측값 분포도 그리기
plt.figure(figsize=(10,5))
ax1 = sns.kdeplot(y_test, label='실제값')
ax2 = sns.kdeplot(y_test_pred, label='예측값', ax=ax1)
plt.legend()
plt.show()

# %%
'''
모델 성능 평가
'''
# 평가
from sklearn.metrics import mean_squared_error
y_train_pred = lr.predict(X_train)

train_mse = mean_squared_error(y_train, y_train_pred)
print("Train MSE:%.4f" % train_mse)

test_mse = mean_squared_error(y_test, y_test_pred)
print("Test MSE:%.4f" % test_mse)

# Train MSE:38824031.0868
# Test MSE:42014414.6340
# 작을수록 모델 성능이 좋은 것.
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 00:01:50 2024

@author: pjc62
"""

import pandas as pd
data = '매매_실거래가격(비교)_수정_v8.xlsx'
# df = pd.read_excel(data)
# df = pd.read_excel(data,parse_dates=['시점'])
df = pd.read_excel(data,index_col='연도_월')


# %%
df.index

# float형인 날짜열을 str형으로 변경 + 10월 처리
df.index = df.index.astype(str).map(lambda x: x + '0' if len(x) < 7 else x)
df.columns
data = df.iloc[:,1:]

target = df.loc[:,['실거래가격지수']]

# %%
'''
상관관계분석
'''
# 상관관계 행렬
df_corr = df.corr()


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
import matplotlib.pyplot as plt
import seaborn as sns
reg()
plt.rc('font', family='NanumBarunGothic')
plt.rcParams['figure.dpi'] = 140


# %%

plt.figure(figsize=(15,20))  # 그림의 크기를 크게 설정
sns.heatmap(data=df.corr(), annot=True, 
fmt='.2f', linewidths=.5, cmap='Blues', 
annot_kws={"size": 10})  # 상관 계수 글꼴 크기 확대

# x축, y축 레이블 크기 조정
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.show()
# %%
# 변수 간 상관 관계 분석
 # 실거래가격지수와 상관 관계가 높은 순서대로 정리
 
corr_order = df.corr().loc['대통령':,'실거래가격지수'].abs().sort_values(ascending=False)
corr_order
# 1인가구            0.916259
# 무주택가구_수         0.909575
# 종합부동산세_세율_개인    0.902650
# 혼인건수            0.898497
# 4인가구_이상         0.895094
# 2인가구            0.892621
# 4인가구            0.887294
# 국회의석수_진보        0.878452
# 외국인_장기체류        0.866395
# 주택부담구입지수        0.859785
# 이혼건수            0.842368
# 국회의석수_보수        0.842306
# 소비자물가지수         0.819332
# 대통령             0.771748
# 1인당_주거면적        0.632280
# PIR지수_전국        0.630363
# 자가보유율           0.615471
# 자가점유율           0.564836
# LIR지수_전국        0.526436
# 3인가구            0.522893
# LIR지수_서울        0.467044
# 주택매매거래량         0.405078
# 지지율             0.342276
# 아파트매매거래량        0.298548
# 예적금담보대출         0.283495
# PIR지수_서울        0.277978
# 주택담보대출          0.209577
# 매매수급동향          0.157530
# 기준금리            0.131498
# 토지매매거래량         0.086382
# 순수토지매매거래량       0.017562
# Name: 실거래가격지수, dtype: float64
corr_order.index
# %%
# plot_cols = ['실거래가격지수','1인가구','무주택가구_수','종합부동산세_세율_개인','혼인건수']
# plot_df = df.loc[:, plot_cols]
# plot_df.head()

# %%
# plt.figure(figsize=(10,10))
# for idx, col in enumerate(plot_cols[1:]):
#     ax1 = plt.subplot(2, 2, idx+1)
#     sns.regplot(x=col, y=plot_cols[0], data=plot_df, ax=ax1)
# plt.show()

# %%
# 실거래가격지수 분포
sns.displot(x='실거래가격지수',kind='hist', data=df)
plt.show()
    # 좌편향인걸 확인
    
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
from statsmodels.stats.outliers_influence import variance_inflation_factor

df.drop(columns='대통령', inplace=True)
# VIF 출력을 위한 데이터 프레임 형성
vif = pd.DataFrame()

# VIF 값과 각 Feature 이름에 대해 설정
vif["VIF Factor"] = [variance_inflation_factor(df.iloc[:,1:].values, i) for i in range(df.iloc[:,1:].shape[1])]
vif["features"] = df.columns[1:] 

# VIF 값이 높은 순으로 정렬
vif = vif.sort_values(by="VIF Factor", ascending=False)
vif = vif.reset_index().drop(columns='index')
vif
# 출처: https://signature95.tistory.com/18 [DataAnalyst:티스토리]
# %%

from sklearn.model_selection import train_test_split
X_data = df.loc[:,corr_order.index[:13]]
y_data = df.loc[:, '실거래가격지수']
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, shuffle=True)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
# (105, 13) (105,)
# (27, 13) (27,)
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

# 예측값, 실제값의 분포
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

# Train MSE:6.3285
# Test MsE:7.5072
# 작을수록 모델 성능이 좋은 것.

# %%
# K-Fold 교차 검증
from sklearn.model_selection import cross_val_score
lr = LinearRegression()
mse_scores = -1*cross_val_score(lr, X_train, y_train, cv=5,
                                scoring='neg_mean_squared_error')
print("개별 Fold MSE:", np.round(mse_scores, 4))
print("평균 MSE:%.4f" % np.mean(mse_scores))
# 개별 Fold MSE: [21.5966 14.7245  6.8402  6.744  10.3203]
# 평균 MSE:12.0451

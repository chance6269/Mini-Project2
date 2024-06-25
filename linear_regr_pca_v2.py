# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 12:26:06 2024

@author: jcp
"""

# -*- coding: utf-8 -*-
"""
Created on Thu May 23 00:01:50 2024

@author: pjc62
"""

import pandas as pd
file = './data/매매_실거래가격(비교)_수정_v11.xlsx'
# df = pd.read_excel(data)
# df = pd.read_excel(data,parse_dates=['시점'])
df = pd.read_excel(file,index_col='연도_월')


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

# plt.figure(figsize=(20,20))
# sns.set(font_scale=0.6)
# sns.heatmap(df_corr, annot=True, cbar=False)
# plt.show

# %%
# 변수 간 상관 관계 분석
 # 실거래가격지수와 상관 관계가 높은 순서대로 정리
 # 이상적인 상관 계수 : 0.3~0.7
corr_order = df.corr().loc['지지율':,'실거래가격지수'].abs().sort_values(ascending=False)
corr_order
# 종합부동산세_세율_개인    0.897827
# 소비자물가지수         0.820183
# PIR지수_전국        0.628749
# 지지율             0.342047
# 예적금담보대출         0.308376
# 아파트매매거래량        0.287109
# 주택담보대출          0.185357
# 매매수급동향          0.165482
# 기준금리            0.150050
# Name: 실거래가격지수, dtype: float64

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

scaler.fit(data)
df_scaled = scaler.transform(data)

df.iloc[:, 1:] = df_scaled[:,:]
df.head()

# %%
'''주성분 분석'''
from sklearn.decomposition import PCA

pca = PCA(random_state=1004)
pca.fit_transform(df.iloc[:, 1:])

print(pca.explained_variance_ratio_)
explained_variance = pca.explained_variance_ratio_

plt.rcParams['figure.figsize'] = (15, 7)
plt.plot(range(1, df.iloc[:,1:].shape[1]+1), pca.explained_variance_ratio_)
plt.xlabel("number of Principal Components", fontsize=12)
plt.ylabel("% of Variance Explained", fontsize=12)
plt.show()

# %%
pca = PCA(n_components=7, random_state=1004)
df_pca = pca.fit_transform(df.iloc[:, 1:])


# %%
from sklearn.model_selection import train_test_split
X_data = df_pca
y_data = df.loc[:, '실거래가격지수']
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, shuffle=True)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
# (105, 2) (105,)
# (27, 2) (27,)
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

# Train MSE:2.1746
# Test MsE:4.4011
# 작을수록 모델 성능이 좋은 것.

# %%
# K-Fold 교차 검증
from sklearn.model_selection import cross_val_score
lr = LinearRegression()
mse_scores = -1*cross_val_score(lr, X_train, y_train, cv=20,
                                scoring='neg_mean_squared_error')
print("개별 Fold MSE:", np.round(mse_scores, 4))
print("평균 MSE:%.4f" % np.mean(mse_scores))

for score in mse_scores:
    if score > 30:
        print('개선필요')
# 개별 Fold MSE: [7.7913 2.0703 6.6352 7.921  1.626 ]
# 평균 MSE:5.2088


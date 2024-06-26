# -*- coding: utf-8 -*-
"""
Created on Thu May 23 00:01:50 2024

@author: pjc62
"""

import pandas as pd
file = './data/매매_실거래가격(비교)_수정_v11.xlsx'

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
'''
피처 스케일링
'''
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
scaled = scaler.fit_transform(data)
scaled = pd.DataFrame(scaled, columns = data.columns)

data = scaled

# df.head()

# %%
'''주성분 분석'''
from sklearn.decomposition import PCA

# %%
pca = PCA(n_components=5, random_state=1004)
df_pca = pca.fit_transform(data)
# %%
# 데이터 분할
from sklearn.model_selection import train_test_split
X_data = df_pca
y_data = df.loc[:, '실거래가격지수']
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, shuffle=True)
# print(X_train.shape, y_train.shape)
# print(X_test.shape, y_test.shape)


# %%
''' 다항식 변환'''
from sklearn.preprocessing import PolynomialFeatures
pf = PolynomialFeatures(degree=2)
X_train_poly = pf.fit_transform(X_train)

X_test_poly = pf.fit_transform(X_test)

# %%

'''
베이스라인 모델 - 선형회귀
'''
from sklearn.linear_model import LinearRegression
import numpy as np
lr = LinearRegression()
lr.fit(X_train_poly, y_train)

print('결정계수:',lr.score(X_test_poly,y_test))
print('회귀계수:',np.round(lr.coef_, 1))
print('상수항:',np.round(lr.intercept_, 1))

# %%
# 예측
y_test_pred = lr.predict(X_test_poly)

# 예측값, 실제값의 분포
plt.figure(figsize=(10,5))
ax1 = sns.kdeplot(y_test, label='실제값')
ax2 = sns.kdeplot(y_test_pred, label='예측값', ax=ax1)
plt.legend()
plt.show()

# %%
def kfc_val(model, x,y):
    # K-Fold 교차 검증
    from sklearn.model_selection import cross_val_score
    
    mse_scores = -1*cross_val_score(model, x, y, cv=20,
                                    scoring='neg_mean_squared_error')
    print("개별 Fold MSE:", np.round(mse_scores, 4))
    print("평균 MSE:%.4f" % np.mean(mse_scores))

# %%
'''
모델 성능 평가
'''

    
# 평가
from sklearn.metrics import mean_squared_error
y_train_pred = lr.predict(X_train_poly)

train_mse = mean_squared_error(y_train, y_train_pred)
print("Train MSE:%.4f" % train_mse)

test_mse = mean_squared_error(y_test, y_test_pred)
print("Test MSE:%.4f" % test_mse)

kfc_val(lr, X_train_poly, y_train)

# %%

'''
과대적합해소
'''
# L2/L1 규제
# Ridge(L2 규제)
from sklearn.linear_model import Ridge
rdg = Ridge(alpha=2.5)
rdg.fit(X_train_poly, y_train)
print('Ridge 결정계수:',rdg.score(X_test_poly,y_test))

y_train_pred = rdg.predict(X_train_poly)
train_mse = mean_squared_error(y_train, y_train_pred)
print('Train MSE:%.4f' % train_mse)
y_test_pred = rdg.predict(X_test_poly)
test_mse = mean_squared_error(y_test, y_test_pred)
print("Test MSE:%.4f" % test_mse)
# Train MSE:35.9484
# Test MSE:42.0011
kfc_val(rdg, X_train_poly, y_train)

# %%
# Lasso(L1 규제)
from sklearn.linear_model import Lasso
las = Lasso(alpha=0.05)
las.fit(X_train_poly, y_train)
print('Lasso 결정계수:',las.score(X_test_poly,y_test))

y_train_pred = las.predict(X_train_poly)
train_mse = mean_squared_error(y_train, y_train_pred)
print("Train MSE:%.4f" % train_mse)
y_test_pred = las.predict(X_test_poly)
test_mse = mean_squared_error(y_test, y_test_pred)
print("Test MSE:%.4f" % test_mse)
# Train MSE:32.3204
# Test MSE:37.7103
kfc_val(las, X_train_poly, y_train)
# %%
# ElasticNet(L2/L1 규제)
from sklearn.linear_model import ElasticNet
ela = ElasticNet(alpha=0.01, l1_ratio=0.7)
ela.fit(X_train_poly, y_train)
print('ElasticNet 결정계수:',ela.score(X_test_poly,y_test))

y_train_pred = ela.predict(X_train_poly)
train_mse = mean_squared_error(y_train, y_train_pred)
print("Train MSE:%.4f" % train_mse)
y_test_pred = ela.predict(X_test_poly)
test_mse = mean_squared_error(y_test, y_test_pred)
print("Test MSE:%.4f" % test_mse)
# Train MSE:33.7551
# Test MSE:39.4968
kfc_val(ela, X_train_poly, y_train)

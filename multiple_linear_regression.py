#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import matplotlib.pyplot as plt
# sklearn library
from sklearn import linear_model

df = pd.read_csv("multilinearregression.csv",sep = ";")

# linear regression modeli tanımlıyoruz:

reg = linear_model.LinearRegression()
reg.fit(df[['alan', 'odasayisi', 'binayasi']], df['fiyat'])

# Tahmin yapıyoruz

reg.predict([[230,4,10]])



# Birden fazla tahmin yapabiliriz.
reg.predict([[230,4,10], [230,6,0], [355,3,20]])







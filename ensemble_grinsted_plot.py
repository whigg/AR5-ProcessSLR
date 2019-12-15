import numpy as np
import os
import pandas as pd
from matplotlib import pyplot as plt

df = pd.read_excel('tas.models.rcp26.xlsx')

print(np.shape(df)[1])

for i in range(np.shape(df)[1]):
    print(i)
    if i ==0:
        continue
    print(df.ix[:,i].values, 'hi')
    mask = np.isfinite(df.iloc[:, i].values)
    plt.plot(df['Time'][mask], df.ix[:, i].values[mask]-np.mean(df.ix[:,i].values[0:51]))

plt.show()

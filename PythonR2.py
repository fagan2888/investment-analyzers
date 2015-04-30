import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

DOMEquity = pd.read_csv("Index_Differences", sep=",")
Index = DOMEquity['Index'][:, np.newaxis]
Change = DOMEquity['Change']

lr = LinearRegression()
lr.fit('Index', 'Change')

b_0 = lr.intercept_
coeff = lr.coef_

score = lr.score(Index[:], Change[:])
print score

plt.scatter('Index', 'Change')
plt.show()

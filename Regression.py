print(__doc__)

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model

# Load dataset
Indexes = datasets.load_Indexes()

Indexes_X = Indexes.data[:, np.newaxis]
Indexes_X_temp = Indexes_X[:, :, 2]

Indexes_X_train = Indexes_X_temp[:-10000]
Indexes_X_test = Indexes_X_temp[-10000:]

Indexes_y_train = Indexes.target[:-3000]
Indexes_y_test = diabetes.target[-3000:]

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(Indexes_X_train, Indexes_y_train)

# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean square error
print("Residual sum of squares: %.2f"
      % np.mean((regr.predict(Indexes_X_test) - Indexes_y_test) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(Indexes_X_test, Indexes_y_test))

# Plot outputs
plt.scatter(Indexes_X_test, Indexes_y_test,  color='red')
plt.plot(Indexes_X_test, regr.predict(Indexes_X_test), color='blue',
         linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()

__author__ = 'LIMU_North'
print(__doc__)

import numpy as np
import pylab as pl
from sklearn import svm

xx, yy = np.meshgrid(np.linspace(-3, 3, 500),
                     np.linspace(-3, 3, 500))
np.random.seed(0)

## X: data points
X = np.random.randn(300, 2)
## Y: label of classification of X
Y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0)

# fit the model
## Nu-Support Vector Classification
clf = svm.NuSVC(nu = 0.4, kernel = 'rbf', shrinking = False)
clf.fit(X, Y)

print clf.dual_coef_

# plot the decision function for each datapoint on the grid
## stands for the distance to the hypersurface?
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)


pl.imshow(Z, interpolation='nearest',
          extent=(xx.min(), xx.max(), yy.min(), yy.max()), aspect='auto',
          origin='lower', cmap=pl.cm.PuOr_r)
contours = pl.contour(xx, yy, Z, levels=[0], linewidths=2,
                      linetypes='--')
pl.scatter(X[:, 0], X[:, 1], s=30, c=Y, cmap=pl.cm.Paired)
pl.xticks(())
pl.yticks(())
pl.axis([-3, 3, -3, 3])
pl.show()
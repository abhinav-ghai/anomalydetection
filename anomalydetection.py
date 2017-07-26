# Credits: Deep Ganguli https://github.com/dganguli/robust-pca
# Credits: Robust PCA http://statweb.stanford.edu/~candes/papers/RobustPCA.pdf

import csv
import numpy
import matplotlib.pyplot as plt
from scipy import stats
from rpca import R_pca

with open('EtwHighPriority.csv', 'r') as etw:
	reader = csv.reader(etw)
	etwlist = list(reader)

etwfiltered = numpy.array(etwlist)[1:,1:].astype(numpy.int64)
print type(etwfiltered[0,0])
rpca = R_pca(etwfiltered)
L, S = rpca.fit(max_iter=20000, iter_print=1000)
S_scaled = stats.zscore(S)
numpy.savetxt("sparsematrix.txt", S_scaled, "%5.10f")

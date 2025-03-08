import os
import matplotlib.pyplot as plt

def get_plot(path, A, B, filename, labelA, labelB, xlabel, ylabel):
	plt.figure()
	if A is not None:
		plt.plot(A, '-bo', label=labelA)
	if B is not None:
		plt.plot(B, '-ro', label=labelB)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.legend()
	plt.savefig(os.path.join(path, filename+'.png'))
	plt.close()

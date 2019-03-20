import os 
import glob
import numpy as np
import matplotlib as mpl
mpl.use('Agg') # for server side
import matplotlib.pyplot as plt


z1 = np.load("/scratch/aziere/Project_PGM/Result/test_z1_zuniform/run_regular_ml.npz")
z2 = np.load("/scratch/aziere/Project_PGM/Result/test_z2_zuniform/run_regular_ml.npz")
z3 = np.load("/scratch/aziere/Project_PGM/Result/test_z3_zuniform/run_regular_ml.npz")

plt.figure(1)
plt.plot(z1['it_loss'], z1['divergences'])
plt.plot(z2['it_loss'], z2['divergences'])
plt.plot(z3['it_loss'], z3['divergences'])
plt.title("Priors on Generator")
plt.xlabel("Iteration")
plt.ylabel("JS Divergence")
plt.legend(['Uniform', 'Normal', 'Multivariate'])

"""
 
z1 = np.load("/scratch/aziere/Project_PGM/Result/test_z1_zuniform/run_regular_ml.npz")
z2 = np.load("/scratch/aziere/Project_PGM/Result/test_z1_znormal/run_regular_ml.npz")
z3 = np.load("/scratch/aziere/Project_PGM/Result/test_z1_zmultivariate/run_regular_ml.npz")

plt.figure(1)
plt.plot(z1['it_loss'], z1['divergences'])
plt.plot(z2['it_loss'], z2['divergences'])
plt.plot(z3['it_loss'], z3['divergences'])
plt.title("Priors on z")
plt.xlabel("Iteration")
plt.ylabel("JS Divergence")
plt.legend(['Uniform', 'Normal', 'Multivariate'])

plt.savefig("/scratch/aziere/Project_PGM/Result/priors.png")"""
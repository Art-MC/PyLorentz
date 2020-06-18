"""Worker functions for multiprocessing calls. 

Implementing multiprocessing on Windows requires the worker functions to be 
imported from a separate file than wherever it is being called. 

AUTHOR:
Arthur McCray, ANL, June 2020.
"""

import numpy as np

def exp_sum(inds, KY, KX, j_n, i_n, Dshp, my_n, mx_n, Sy, Sx):
    """Called linsupPhi when running with multiprocessing"""
    mphi_k = np.zeros(KX.shape,dtype=complex)
    ephi_k = np.zeros(KX.shape,dtype=complex)
    
    for ind in inds:
        z = ind[0]
        y = ind[1]
        x = ind[2]

        sum_term = np.exp(-1j * (KY*j_n[z,y,x] + KX*i_n[z,y,x]))
        ephi_k += sum_term * Dshp[z,y,x]
        mphi_k += sum_term * (my_n[z,y,x]*Sx - mx_n[z,y,x]*Sy)

    return (ephi_k, mphi_k)
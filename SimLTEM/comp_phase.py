"""This module consists of functions for simulating the phase shift of a given
object. 

It contained two functions:
1) linsupPhi - using the linear supeposition principle for application in MBIR type
               3D reconstruction of magnetization (both magnetic and electrostatic)
2) mansPhi - using the Mansuripur Algorithm to compute the phase shift (only magnetic)

Written, CD Phatak, ANL, 08.May.2015.
Modified, CD Phatak, ANL, 22.May.2016.
"""

import numpy as np
# from TIE_helper import *
import time
import multiprocessing as mp
# from mp_helper import exp_sum

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

def linsupPhi(mx=1.0, my=1.0, mz=1.0, Dshp=None, theta_x=0.0, theta_y=0.0, pre_B=1.0, pre_E=None, v=1, multiproc=True):
    """Applies linear supeposition principle for 3D reconstruction of magnetic and electrostatic phase shifts.

    This function will take the 3D arrays with Mx, My and Mz components of the magnetization
    and the Dshp array consisting of the shape function for the object (1 inside, 0 outside)
    and then the tilt angles about x and y axes to compute the magnetic phase shift and
    the electrostatic phase shift. Initial computation is done in Fourier space and then
    real space values are returned.

    Args: 
        mx: 3D array. x component of magnetization at each voxel (z,y,x) (gauss)
        my: 3D array. y component of magnetization at each voxel (z,y,x) (gauss)
        mz: 3D array. z component of magnetization at each voxel (z,y,x) (gauss)
        Dshp: 3D array. Weighted shape function of the object. Where value is 0
            phase is not computed, otherwise it is multiplied by the voxel value
            of Dshp. 
        theta_x: Float. Rotation around x-axis (degrees) 
        theta_y: Float. Rotation around y-axis (degrees) 
        pre_B: Float. Prefactor for unit conversion in calculating the magnetic 
            phase shift. Units 1/pixels^2. Generally (2*pi*b0*(nm/pix)^2)/phi0 
            where b0 is the Saturation induction and phi0 the magnetic flux
            quantum. 
        pre_E: Float. Prefactor for unit conversion in calculating the 
            electrostatic phase shift. Equal to sigma*V0, where sigma is the 
            interaction constant of the given TEM accelerating voltage (an 
            attribute of the microscope class), and V0 the mean inner potential.
        v: Int. Verbosity. v >= 1 will print progress, v=0 to suppress all prints.
        mp: Bool. Whether or not to implement multiprocessing. 
    Returns: [ephi, mphi]
        ephi: Electrostatic phase shift, 2D array
        mphi: magnetic phase shift, 2D array
    """
    vprint = print if v>=1 else lambda *a, **k: None
    if pre_E is None: # pre_E and pre_B should be set for material params
        print('adjusting')
        pre_E = 4.80233*pre_B # a generic value in case its not.  

    [dimz,dimy,dimx] = mx.shape
    dx2 = dimx//2
    dy2 = dimy//2
    dz2 = dimz//2

    ly = (np.arange(dimy)-dy2)/dimy
    lx = (np.arange(dimx)-dx2)/dimx
    [Y,X] = np.meshgrid(ly,lx, indexing='ij')
    dk = 2.0*np.pi # Kspace vector spacing
    KX = X*dk
    KY = Y*dk
    KK = np.sqrt(KX**2 + KY**2) # same as dist(ny, nx, shift=True)*2*np.pi
    zeros = np.where(KK == 0)   # but we need KX and KY later anyways. 
    KK[zeros] = 1.0 # remove points where KK is zero as will divide by

    #now compute constant factors (apply actual constants at very end)
    inv_KK =  1/KK**2
    Sx = 1j * KX * inv_KK
    Sy = 1j * KY * inv_KK
    Sx[zeros] = 0.0
    Sy[zeros] = 0.0

    #Now we loop through all coordinates and compute the summation terms
    mphi_k = np.zeros(KK.shape,dtype=complex)
    ephi_k = np.zeros(KK.shape,dtype=complex)
    
    #Trying to use nonzero elements in Dshape to limit the iterations.
    if Dshp is None: 
        Dshp = np.ones(mx.shape)
    # exclude indices where thickness is 0, compile into list of ((z1,y1,x1), (z2,y2...
    zz, yy, xx = np.where(Dshp != 0)
    inds = np.dstack((zz,yy,xx)).squeeze()
    inds = tuple(map(tuple,inds))

    # Compute the rotation angles
    st = np.sin(np.deg2rad(theta_x))
    ct = np.cos(np.deg2rad(theta_x))
    sg = np.sin(np.deg2rad(theta_y))
    cg = np.cos(np.deg2rad(theta_y))

    x = np.arange(dimx) - dx2
    y = np.arange(dimy) - dy2
    z = np.arange(dimz) - dz2
    Z,Y,X = np.meshgrid(z,y,x, indexing='ij') # grid of actual positions (centered on 0)

    # compute the rotated values; 
    # here we apply rotation about X first, then about Y
    i_n = Z*sg*ct + Y*sg*st + X*cg
    j_n = Y*ct - Z*st

    mx_n = mx*cg + my*sg*st + mz*sg*ct
    my_n = my*ct - mz*st

    cc = -1
    nelems = np.shape(inds)[0]
    stime = time.time()
    otime = time.time()
    vprint(f'Beginning phase calculation for {nelems:g} voxels.')
    if multiproc:
        batches = mp.cpu_count()
        vprint(f"Splitting and running on {batches} cpus")
        pool = mp.Pool(processes=batches)
        batch_inds = np.array_split(inds, batches)
        batches_phi = []
        for batch in batch_inds:
            batches_phi.append(pool.apply_async(exp_sum, args=(batch, KY, KX, j_n, i_n,
                                                Dshp, my_n, mx_n, Sy, Sx,)))
        # pool.close()
        # pool.join() 
        em_stack = [p.get() for p in batches_phi]
        # em_stack = list(map(lambda x: x.get(), batches_phi))
        em = np.sum(em_stack, axis=0)
        ephi_k = em[0]
        mphi_k = em[1]
        vprint(f"total time multiprocess with {batches} jobs: {time.time()-stime:.5g} sec, {(time.time()-stime)/nelems:.5g} sec/voxel.")

    else:
        vprint("Running on 1 cpu.")
        vprint('0.00%', end=' .. ')
        for ind in inds:
            cc += 1
            if time.time() - stime >= 15:
                vprint(f'{cc/nelems*100:.2f}%', end=' .. ')
                stime = time.time()
            # compute the expontential summation
            sum_term = np.exp(-1j * (KY*j_n[ind] + KX*i_n[ind]))
            ephi_k += sum_term * Dshp[ind]
            mphi_k += sum_term * (my_n[ind]*Sx - mx_n[ind]*Sy)

        vprint('100.0%')
        print(f"total time loop method: {time.time()-otime:.5g} sec, {(time.time()-otime)/nelems:.5g} sec/voxel.")
    #Now we have the phases in K-space. We convert to real space and return
    ephi_k[zeros] = 0.0
    mphi_k[zeros] = 0.0
    ephi = (np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(ephi_k)))).real*pre_E
    mphi = (np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(mphi_k)))).real*pre_B

    return [ephi,mphi]

def mansPhi(bx = 1.0,by = 1.0,bz = 1.0,beam = [0.0,0.0,1.0],thick = 1.0,embed = 0.0): 
    """Calculate magnetic phase shift using Mansuripur algorithm [1]. 

    Unlike the linear superposition method, this algorithm only accepts 2D 
    samples. The input given is expected to be 2D arrays for Bx, By, Bz. 

    Args: 
        bx: 2D array. x component of magnetization at each pixel.
        by: 2D array. y component of magnetization at each pixel.
        bz: 2D array. z component of magnetization at each pixel.  
        beam: List [x,y,z]. Vector direction of beam. Default [001]. 
        thick: Float. Thickness of the sample in pixels, need not be an int. 
        embed:  Int. Whether or not to embed the bx, by, bz into a larger array
            for fourier-space computation. This improves edge effects at the 
            cost of reduced speed. 
            embed = 0: Do not embed (default)
            embed = 1: Embed in (1024, 1024) array
            embed = x: Embed in (x, x) array. 

    Returns: 
        Magnetic phase shift, 2D array
    
    [1] Mansuripur, M. Computation of electron diffraction patterns in Lorentz 
        electron microscopy of thin magnetic films. J. Appl. Phys. 69, 5890 (1991).
    """
    #Normalize the beam direction
    beam = np.array(beam)
    beam /= np.sqrt(np.sum(beam**2))

    #Get dimensions
    [xsz,ysz] = bx.shape

    #Embed
    if (embed == 1.0):
        bdim = 1024.0
        bdimx,bdimy = bdim,bdim
    elif (embed == 0.0):
        bdimx,bdimy = xsz,ysz
    else:
        bdim = np.float(embed)
        bdimx,bdimy = bdim,bdim

    bigbx = np.zeros([bdimx,bdimy])
    bigby = np.zeros([bdimx,bdimy])
    bigbx[int(bdimx/2-xsz/2):int(bdimx/2+xsz/2),int(bdimy/2-ysz/2):int(bdimy/2+ysz/2)] = bx
    bigby[int(bdimx/2-xsz/2):int(bdimx/2+xsz/2),int(bdimy/2-ysz/2):int(bdimy/2+ysz/2)] = by
    if (bz != 1.0):
        bigbz = np.zeros([bdimx,bdimy])
        bigbz[bdimx/2-xsz/2:bdimx/2+xsz/2,bdimy/2-ysz/2:bdimy/2+ysz/2] = bz

    #Compute the auxiliary arrays requried for computation
    dsx = 2.0*np.pi/bdimx 
    linex = (np.arange(bdimx)-np.float(bdimx/2))*dsx
    dsy = 2.0*np.pi/bdimy
    liney = (np.arange(bdimy)-np.float(bdimy/2))*dsy
    [Sx,Sy] = np.meshgrid(linex,liney)
    S = np.sqrt(Sx**2 + Sy**2)
    zinds = np.where(S == 0)
    S[zinds] = 1.0
    sigx = Sx/S
    sigy = Sy/S
    sigx[zinds] = 0.0
    sigy[zinds] = 0.0

    #compute FFTs of the B arrays.
    fbx = np.fft.fftshift(np.fft.fftn(bigbx))
    fby = np.fft.fftshift(np.fft.fftn(bigby))
    if (bz != 1.0):
        fbz = np.fft.fftshift(np.fft.fftn(bigbz))

    #Compute vector products and Gpts
    if (bz == 1.0): # eq 13a in Mansuripur 
        prod = sigx*fby - sigy*fbx
        Gpts = 1+1j*0
    else:
        prod = sigx*(fby*beam[0]**2 - fbx*beam[0]*beam[1] - fbz*beam[1]*beam[2]+ fby*beam[2]**2
                ) + sigy*(fby*beam[0]*beam[1] - fbx*beam[1]**2 + fbz*beam[0]*beam[2] - fbx*beam[2]**2)
        arg = np.float(np.pi*thick*(sigx*beam[0]+sigy*beam[1])/beam[2])
        qq = np.where(arg == 0)
        denom = 1.0/((sigx*beam[0]+sigy*beam[1])**2 + beam[2]**2)
        Gpts = complex(denom*np.sin(arg)/arg)
        Gpts[qq] = denom[qq]

    #prefactor
    prefac = 1j*thick/S
    
    #F-space phase
    fphi = prefac * Gpts * prod
    fphi[zinds] = 0.0
    phi = np.fft.ifftn(np.fft.ifftshift(fphi)).real

    #return only the actual phase part from the embed file
    ret_phi = phi[bdimx//2-xsz//2:bdimx//2+xsz//2,bdimy//2-ysz//2:bdimy//2+ysz//2]

    return ret_phi
#done.
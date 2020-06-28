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
import time
import numba
from numba import jit


@jit(nopython=True, parallel=True)
def exp_sum(mphi_k, ephi_k, inds, KY, KX, j_n, i_n, my_n, mx_n, Sy, Sx):
    """Called by linsupPhi when running with multiprocessing and numba (default). 

    Numba incorporates just-in-time (jit) compiling and multiprocessing to numpy
    array calculations, greatly speeding up the phase-shift computation beyond 
    that of pure vectorization and without the memory cost. Running this 
    for the firs time each session will take an additional 5-10 seconds as it is
    compiled. 

    This function could be further improved by sending it to the GPU, or likely
    by other methods we haven't considered. If you have suggestions (or better
    yet, written and tested code) please email amccray@anl.gov. 
    """
    for i in numba.prange(np.shape(inds)[0]):
        z = int(inds[i,0])
        y = int(inds[i,1])
        x = int(inds[i,2])
        sum_term = np.exp(-1j * (KY*j_n[z,y,x] + KX*i_n[z,y,x]))
        ephi_k += sum_term 
        mphi_k += sum_term * (my_n[z,y,x]*Sx - mx_n[z,y,x]*Sy) 
    return ephi_k, mphi_k

def linsupPhi(mx=1.0, my=1.0, mz=1.0, Dshp=None, theta_x=0.0, theta_y=0.0, pre_B=1.0, pre_E=None, v=1, multiproc=True):
    """Applies linear supeposition principle for 3D reconstruction of magnetic and electrostatic phase shifts.

    This function will take the 3D arrays with Mx, My and Mz components of the magnetization
    and the Dshp array consisting of the shape function for the object (1 inside, 0 outside)
    and then the tilt angles about x and y axes to compute the magnetic phase shift and
    the electrostatic phase shift. Initial computation is done in Fourier space and then
    real space values are returned.

    Args: 
        mx: 3D array. x component of magnetization at each voxel (z,y,x)
        my: 3D array. y component of magnetization at each voxel (z,y,x)
        mz: 3D array. z component of magnetization at each voxel (z,y,x)
        Dshp: 3D array. Binary shape function of the object. Where value is 0,
            phase is not computed.  
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
    zeros = np.where(KK == 0)   # but we need KX and KY later. 
    KK[zeros] = 1.0 # remove points where KK is zero as will divide by it

    # compute S arrays (will apply constants at very end)
    inv_KK =  1/KK**2
    Sx = 1j * KX * inv_KK
    Sy = 1j * KY * inv_KK
    Sx[zeros] = 0.0
    Sy[zeros] = 0.0
    
    # Get indices for which to calculate phase shift. Skip all pixels where
    # thickness == 0 
    if Dshp is None: 
        Dshp = np.ones(mx.shape)
    # exclude indices where thickness is 0, compile into list of ((z1,y1,x1), (z2,y2...
    zz, yy, xx = np.where(Dshp != 0)
    inds = np.dstack((zz,yy,xx)).squeeze()

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

    # setup 
    mphi_k = np.zeros(KK.shape,dtype=complex)
    ephi_k = np.zeros(KK.shape,dtype=complex)

    nelems = np.shape(inds)[0]
    stime = time.time()
    vprint(f'Beginning phase calculation for {nelems:g} voxels.')
    if multiproc:
        vprint("Running in parallel with numba.")
        ephi_k, mphi_k = exp_sum(mphi_k, ephi_k, inds, KY, KX, j_n, i_n, my_n, mx_n, Sy, Sx)        

    else:
        vprint("Running on 1 cpu.")
        otime = time.time()
        vprint('0.00%', end=' .. ')
        cc = -1
        for ind in inds:
            cc += 1
            if time.time() - otime >= 15:
                vprint(f'{cc/nelems*100:.2f}%', end=' .. ')
                otime = time.time()
            # compute the expontential summation
            sum_term = np.exp(-1j * (KY*j_n[ind] + KX*i_n[ind]))
            ephi_k += sum_term 
            mphi_k += sum_term * (my_n[ind]*Sx - mx_n[ind]*Sy)
        vprint('100.0%')

    vprint(f"total time: {time.time()-stime:.5g} sec, {(time.time()-stime)/nelems:.5g} sec/voxel.")
    #Now we have the phases in K-space. We convert to real space and return
    ephi_k[zeros] = 0.0
    mphi_k[zeros] = 0.0
    ephi = (np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(ephi_k)))).real*pre_E
    mphi = (np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(mphi_k)))).real*pre_B

    return [ephi,mphi]

def mansPhi(bx = 1.0,by = 1.0,bz = None,beam = [0.0,0.0,1.0],thick = 1.0,embed = 0.0): 
    """Calculate magnetic phase shift using Mansuripur algorithm [1]. 

    Unlike the linear superposition method, this algorithm only accepts 2D 
    samples. The input given is expected to be 2D arrays for Bx, By, Bz. It can 
    compute beam angles close to (0,0,1), but for tilts 

    Args: 
        bx: 2D array. x component of magnetization at each pixel.
        by: 2D array. y component of magnetization at each pixel.
        bz: 2D array. z component of magnetization at each pixel.  
        beam: List [x,y,z]. Vector direction of beam. Default [001]. 
        thick: Float. Thickness of the sample in pixels. 
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
    beam = beam / np.sqrt(np.sum(beam**2))

    #Get dimensions
    [xsz,ysz] = bx.shape

    #Embed
    if (embed == 1.0):
        bdim = 1024
        bdimx,bdimy = bdim,bdim
    elif (embed == 0.0):
        bdimx,bdimy = xsz,ysz
    else:
        bdim = int(embed)
        bdimx,bdimy = bdim,bdim

    bigbx = np.zeros([bdimx,bdimy])
    bigby = np.zeros([bdimx,bdimy])
    bigbx[int(bdimx/2-xsz/2):int(bdimx/2+xsz/2),int(bdimy/2-ysz/2):int(bdimy/2+ysz/2)] = bx
    bigby[int(bdimx/2-xsz/2):int(bdimx/2+xsz/2),int(bdimy/2-ysz/2):int(bdimy/2+ysz/2)] = by
    if bz is not None:
        bigbz = np.zeros([bdimx,bdimy])
        bigbz[int(bdimx/2-xsz/2):int(bdimx/2+xsz/2),int(bdimy/2-ysz/2):int(bdimy/2+ysz/2)] = bz
    
    #Compute the auxiliary arrays requried for computation
    dsx = 2.0*np.pi/bdimx 
    linex = (np.arange(bdimx)-bdimx/2)*dsx
    dsy = 2.0*np.pi/bdimy
    liney = (np.arange(bdimy)-bdimy/2)*dsy
    [Sx,Sy] = np.meshgrid(linex,liney)
    S = np.sqrt(Sx**2 + Sy**2)
    zinds = np.where(S == 0)
    S[zinds] = 1.0
    sigx = Sx/S
    sigy = Sy/S
    sigx[zinds] = 0.0
    sigy[zinds] = 0.0

    #compute FFTs of the B arrays.
    fbx = np.fft.fftshift(np.fft.fft2(bigbx))
    fby = np.fft.fftshift(np.fft.fft2(bigby))

    if bz is not None:
        fbz = np.fft.fftshift(np.fft.fft2(bigbz))

    #Compute vector products and Gpts
    if bz is None: # eq 13a in Mansuripur 
        prod = sigx*fby - sigy*fbx
        Gpts = 1+1j*0

    else:
        e_x, e_y, e_z = beam
        prod = sigx*(fby*e_x**2 - fbx*e_x*e_y - fbz*e_y*e_z+ fby*e_z**2
                ) + sigy*(fby*e_x*e_y - fbx*e_y**2 + fbz*e_x*e_z - fbx*e_z**2)
        arg = np.pi*thick*(sigx*e_x+sigy*e_y)/e_z
        denom = 1.0/((sigx*e_x+sigy*e_y)**2 + e_z**2)
        qq = np.where(arg == 0)
        arg[qq] = 1
        Gpts = (denom*np.sin(arg)/arg).astype(complex)
        Gpts[qq] = denom[qq]

    #prefactor
    prefac = 1j*thick/S    
    #F-space phase
    fphi = prefac * Gpts * prod
    fphi[zinds] = 0.0
    phi = np.fft.ifft2(np.fft.ifftshift(fphi)).real

    #return only the actual phase part from the embed file
    if embed != 0:
        ret_phi = phi[bdimx//2-xsz//2:bdimx//2+xsz//2,bdimy//2-ysz//2:bdimy//2+ysz//2]
    else:
        ret_phi = phi

    return ret_phi
#done.
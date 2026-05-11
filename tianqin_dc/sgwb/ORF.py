from io import UnsupportedOperation
import numpy as np
#from geometry import geometry
import healpy as hp
from healpy import Alm
import numpy.linalg as LA
from scipy.special import sph_harm
from tqdm import tqdm

# -*- coding: utf-8 -*-
def tianqin_orbits(tsegmid):
    # 天琴轨道分为两部分，一部分是描述地球在公转过程中的坐标变化,第二部分对应天琴三颗卫星的绕地轨道坐标变化
    #第一部分：地球公转过程中的坐标变化
    times = tsegmid

    ## 轨道半长轴：单位/m
    R = 1.496e11
    ## 设置初始相位
    alphaphase = 0

    ## Orbital angle alpha(t)
    # at = 2*pi*fe*t-alpha   alpha = 102.9度
    at = (2 * np.pi / 31557600) * times - alphaphase

    ## 设置偏心率
    #e = self.armlength/(2*at*np.sqrt(3))
    e = 0.0167

    X = R * np.cos(at) + 0.5 * e * R * (np.cos(2 * at) - 3)
    Y = R * np.sin(at) + 0.5 * e * R * np.sin(2 * at)
    Z = 0
    # 第二部分对应天琴三颗卫星的绕地轨道坐标变化
    # 设置卫星轨道高度
    R1 = 1e8  # 单位：m
    thetas = np.radians(-4.7)
    phis = np.radians(120.5)
    kpa_n = (2 / 3) * np.pi * np.array([0, 1, 2])+np.pi/5
    # 周期3.65天
    at_n = (2 * np.pi / (3.65*3600*24)) * times
    #at_n = (2 * np.pi / 315360) * (times % 315360)
    kpa_n, at_n = np.meshgrid(kpa_n, at_n)
    x_n = R1 * (np.cos(phis) * np.sin(thetas) * np.sin(at_n + kpa_n) + np.cos(at_n + kpa_n) * np.sin(phis))
    y_n = R1 * (np.sin(phis) * np.sin(thetas) * np.sin(at_n + kpa_n) - np.cos(at_n + kpa_n) * np.cos(phis))
    z_n = -R1 * np.sin(at_n + kpa_n) * np.cos(thetas)
    # 设置整体日心黄道坐标

    rs1 = np.array([x_n[:, 0] + X, y_n[:, 0] + Y, z_n[:, 0] + Z])
    rs2 = np.array([x_n[:, 1] + X, y_n[:, 1] + Y, z_n[:, 1] + Z])
    rs3 = np.array([x_n[:, 2] + X, y_n[:, 2] + Y, z_n[:, 2] + Z])
    return rs1, rs2, rs3

def lisa_orbits(tsegmid):

    times = tsegmid
    ## Semimajor axis in m
    a = 1.496e11


    ## Alpha and beta phases allow for changing of initial satellite orbital phases; default initial conditions are alphaphase=betaphase=0.
    betaphase = 0
    #alphaphase = 0
    alphaphase = np.radians(-20)
    ## Orbital angle alpha(t)
    at = (2*np.pi/31557600)*times + alphaphase

    ## Eccentricity. L-dependent, so needs to be altered for time-varied arm length case.
    armlength = 2.5e9
    e = armlength/(2*a*np.sqrt(3))

    ## Initialize arrays
    beta_n = (2/3)*np.pi*np.array([0,1,2])+betaphase

    ## meshgrid arrays
    Beta_n, Alpha_t = np.meshgrid(beta_n, at)

    ## Calculate inclination and positions for each satellite.
    x_n = a*np.cos(Alpha_t) + a*e*(np.sin(Alpha_t)*np.cos(Alpha_t)*np.sin(Beta_n) - (1+(np.sin(Alpha_t))**2)*np.cos(Beta_n))
    y_n = a*np.sin(Alpha_t) + a*e*(np.sin(Alpha_t)*np.cos(Alpha_t)*np.cos(Beta_n) - (1+(np.cos(Alpha_t))**2)*np.sin(Beta_n))
    z_n = -np.sqrt(3)*a*e*np.cos(Alpha_t - Beta_n)


    ## Construct position vectors r_n
    rs1 = np.array([x_n[:, 0],y_n[:, 0],z_n[:, 0]])
    rs2 = np.array([x_n[:, 1],y_n[:, 1],z_n[:, 1]])
    rs3 = np.array([x_n[:, 2],y_n[:, 2],z_n[:, 2]])

    return rs1, rs2, rs3
def taiji_orbits(tsegmid):

    times = tsegmid
    ## Semimajor axis in m
    a = 1.496e11


    ## Alpha and beta phases allow for changing of initial satellite orbital phases; default initial conditions are alphaphase=betaphase=0.
    betaphase = 0
    alphaphase = np.radians(20)

    ## Orbital angle alpha(t)
    at = (2*np.pi/31557600)*times + alphaphase

    ## Eccentricity. L-dependent, so needs to be altered for time-varied arm length case.
    armlength = 3e9
    e = armlength/(2*a*np.sqrt(3))

    ## Initialize arrays
    beta_n = (2/3)*np.pi*np.array([0,1,2])+betaphase

    ## meshgrid arrays
    Beta_n, Alpha_t = np.meshgrid(beta_n, at)

    ## Calculate inclination and positions for each satellite.
    x_n = a*np.cos(Alpha_t) + a*e*(np.sin(Alpha_t)*np.cos(Alpha_t)*np.sin(Beta_n) - (1+(np.sin(Alpha_t))**2)*np.cos(Beta_n))
    y_n = a*np.sin(Alpha_t) + a*e*(np.sin(Alpha_t)*np.cos(Alpha_t)*np.cos(Beta_n) - (1+(np.cos(Alpha_t))**2)*np.sin(Beta_n))
    z_n = -np.sqrt(3)*a*e*np.cos(Alpha_t - Beta_n)


    ## Construct position vectors r_n
    rs1 = np.array([x_n[:, 0],y_n[:, 0],z_n[:, 0]])
    rs2 = np.array([x_n[:, 1],y_n[:, 1],z_n[:, 1]])
    rs3 = np.array([x_n[:, 2],y_n[:, 2],z_n[:, 2]])

    return rs1, rs2, rs3
# 球谐函数的指标转换

def idxtoalm(lmax, ii):

    '''
    index --> (l, m) function which works for negetive indices too
    '''

    alm_size = Alm.getsize(lmax)

    if ii >= (2*alm_size - lmax - 1):
        raise ValueError('Index larger than acceptable')
    elif ii < alm_size:
        l, m = Alm.getlm(lmax, ii)
    else:
        l, m = Alm.getlm(lmax, ii - alm_size + lmax + 1)

        if m ==0:
            raise ValueError('Something wrong with ind -> (l, m) conversion')
        else:
            m = -m

    return l, m


def asgwb_LSmich_response(frange, tsegmid, params):

        '''
        Calculate the Antenna pattern/ detector transfer function functions to acSGWB using michelson channels,
        and using a spherical harmonic decomposition. Note that the response function to power is integrated over
        sky direction with the appropriate spherical harmonics, and averaged over polarozation. The angular
        integral is numerically done by divvying up the sky into a healpix grid.

        Note that f0 is (pi*L*f)/c and is input as an array

        Parameters
        -----------

        f0   : float
            A numpy array of scaled frequencies (see above for def)

        Returns
        ---------

        R1, R2 and R3   :   float
            Antenna Patterns for the given sky direction for the three channels, integrated over sky direction and averaged
            over polarization. The arrays are 2-d, one direction corresponds to frequency and the other to the l coeffcient.
        '''

        ## array size of almax
        #alm_size = (self.almax + 1) ** 2

        npix = hp.nside2npix(params['nside'])

        # Array of pixel indices
        pix_idx = np.arange(npix)

        # Angular coordinates of pixel indcides
        theta, phi = hp.pix2ang(params['nside'], pix_idx)

        # Take cosine.
        ctheta = np.cos(theta)

        # Area of each pixel in sq.radians
        dOmega = hp.pixelfunc.nside2pixarea(params['nside'])

        # Create 2D array of (x,y,z) unit vectors for every sky direction.
        omegahat = -np.array([np.sqrt(1 - ctheta ** 2) * np.cos(phi), np.sqrt(1 - ctheta ** 2) * np.sin(phi), ctheta])

        # 获得天琴的轨道
        LSrs1, LSrs2, LSrs3 = lisa_orbits(tsegmid)
        # 获得臂长向量和传播方向omegahat的乘积
        LSudir = np.einsum('ij,ik', (LSrs2 - LSrs1) / LA.norm(LSrs2 - LSrs1, axis=0)[None, :], omegahat)
        LSvdir = np.einsum('ij,ik', (LSrs3 - LSrs1) / LA.norm(LSrs3 - LSrs1, axis=0)[None, :], omegahat)
        LSwdir = np.einsum('ij,ik', (LSrs3 - LSrs2) / LA.norm(LSrs3 - LSrs2, axis=0)[None, :], omegahat)
        ## NB --    An attempt to directly adapt e.g. (u o u):e+ as implicit tensor calculations
        ##             as opposed to the explicit forms we've previously used. '''
        mhat = np.array([-np.sin(phi), np.cos(phi), np.zeros(len(phi))])
        nhat = np.array([np.cos(phi) * ctheta, np.sin(phi) * ctheta, -np.sqrt(1 - ctheta ** 2)])

        # 1/2 u x u : eplus. These depend only on geometry so they only have a time and directionality dependence and not of frequency
        # lisa

        LSFplus_u = 0.5 * np.einsum("ijk,ijl", \
                                  np.einsum("ik,jk -> ijk", (LSrs2 - LSrs1) / LA.norm(LSrs2 - LSrs1, axis=0)[None, :],
                                            (LSrs2 - LSrs1) / LA.norm(LSrs2 - LSrs1, axis=0)[None, :]), \
                                  np.einsum("ik,jk -> ijk", nhat, nhat) - np.einsum("ik,jk -> ijk", mhat, mhat))

        LSFplus_v = 0.5 * np.einsum("ijk,ijl", \
                                  np.einsum("ik,jk -> ijk", (LSrs3 - LSrs1) / LA.norm(LSrs3 - LSrs1, axis=0)[None, :],
                                            (LSrs3 - LSrs1) / LA.norm(LSrs3 - LSrs1, axis=0)[None, :]), \
                                  np.einsum("ik,jk -> ijk", nhat, nhat) - np.einsum("ik,jk -> ijk", mhat, mhat))

        LSFplus_w = 0.5 * np.einsum("ijk,ijl", \
                                  np.einsum("ik,jk -> ijk", (LSrs3 - LSrs2) / LA.norm(LSrs3 - LSrs2, axis=0)[None, :],
                                            (LSrs3 - LSrs2) / LA.norm(LSrs3 - LSrs2, axis=0)[None, :]), \
                                  np.einsum("ik,jk -> ijk", nhat, nhat) - np.einsum("ik,jk -> ijk", mhat, mhat))

        # 1/2 u x u : ecross
        LSFcross_u = 0.5 * np.einsum("ijk,ijl", \
                                   np.einsum("ik,jk -> ijk", (LSrs2 - LSrs1) / LA.norm(LSrs2 - LSrs1, axis=0)[None, :],
                                             (LSrs2 - LSrs1) / LA.norm(LSrs2 - LSrs1, axis=0)[None, :]), \
                                   np.einsum("ik,jk -> ijk", mhat, nhat) + np.einsum("ik,jk -> ijk", nhat, mhat))

        LSFcross_v = 0.5 * np.einsum("ijk,ijl", \
                                   np.einsum("ik,jk -> ijk", (LSrs3 - LSrs1) / LA.norm(LSrs3 - LSrs1, axis=0)[None, :],
                                             (LSrs3 - LSrs1) / LA.norm(LSrs3 - LSrs1, axis=0)[None, :]), \
                                   np.einsum("ik,jk -> ijk", mhat, nhat) + np.einsum("ik,jk -> ijk", nhat, mhat))

        LSFcross_w = 0.5 * np.einsum("ijk,ijl", \
                                   np.einsum("ik,jk -> ijk", (LSrs3 - LSrs2) / LA.norm(LSrs3 - LSrs2, axis=0)[None, :],
                                             (LSrs3 - LSrs2) / LA.norm(LSrs3 - LSrs2, axis=0)[None, :]), \
                                   np.einsum("ik,jk -> ijk", mhat, nhat) + np.einsum("ik,jk -> ijk", nhat, mhat))
        LSfstar = params['cspeed'] / (2 * np.pi * params['LSarmlength']) #特征频率
        LSf0 = frange / (2 * LSfstar)
        LSFplus1 = np.zeros((LSf0.size, tsegmid.size, pix_idx.size), dtype='complex')
        LSFplus2 = np.zeros((LSf0.size, tsegmid.size, pix_idx.size), dtype='complex')
        LSFplus3 = np.zeros((LSf0.size, tsegmid.size, pix_idx.size), dtype='complex')
        LSFcross1 = np.zeros((LSf0.size, tsegmid.size, pix_idx.size), dtype='complex')
        LSFcross2 = np.zeros((LSf0.size, tsegmid.size, pix_idx.size), dtype='complex')
        LSFcross3 = np.zeros((LSf0.size, tsegmid.size, pix_idx.size), dtype='complex')

        # lisa
        for ii in tqdm(range(0, LSf0.size)):
            LSgammaU_plus = 1 / 2 * (np.sinc((LSf0[ii]) * (1 - LSudir) / np.pi) * np.exp(-1j * LSf0[ii] * (3 + LSudir)) + \
                                   np.sinc((LSf0[ii]) * (1 + LSudir) / np.pi) * np.exp(-1j * LSf0[ii] * (1 + LSudir)))

            LSgammaV_plus = 1 / 2 * (np.sinc((LSf0[ii]) * (1 - LSvdir) / np.pi) * np.exp(-1j * LSf0[ii] * (3 + LSvdir)) + \
                                   np.sinc((LSf0[ii]) * (1 + LSvdir) / np.pi) * np.exp(-1j * LSf0[ii] * (1 + LSvdir)))

            LSgammaW_plus = 1 / 2 * (np.sinc((LSf0[ii]) * (1 - LSwdir) / np.pi) * np.exp(-1j * LSf0[ii] * (3 + LSwdir)) + \
                                   np.sinc((LSf0[ii]) * (1 + LSwdir) / np.pi) * np.exp(-1j * LSf0[ii] * (1 + LSwdir)))

            # Calculate GW transfer function for the michelson channels
            LSgammaU_minus = 1 / 2 * (np.sinc((LSf0[ii]) * (1 + LSudir) / np.pi) * np.exp(-1j * LSf0[ii] * (3 - LSudir)) + \
                                    np.sinc((LSf0[ii]) * (1 - LSudir) / np.pi) * np.exp(-1j * LSf0[ii] * (1 - LSudir)))

            LSgammaV_minus = 1 / 2 * (np.sinc((LSf0[ii]) * (1 + LSvdir) / np.pi) * np.exp(-1j * LSf0[ii] * (3 - LSvdir)) + \
                                    np.sinc((LSf0[ii]) * (1 - LSvdir) / np.pi) * np.exp(-1j * LSf0[ii] * (1 - LSvdir)))

            LSgammaW_minus = 1 / 2 * (np.sinc((LSf0[ii]) * (1 + LSwdir) / np.pi) * np.exp(-1j * LSf0[ii] * (3 - LSwdir)) + \
                                    np.sinc((LSf0[ii]) * (1 - LSwdir) / np.pi) * np.exp(-1j * LSf0[ii] * (1 - LSwdir)))
            ## Calculate Fplus
            LSFplus1[ii,:,:] = LSFplus_u * LSgammaU_plus - LSFplus_v * LSgammaV_plus
            LSFplus2[ii,:,:] = LSFplus_w * LSgammaW_plus - LSFplus_u * LSgammaU_minus
            LSFplus3[ii,:,:] = LSFplus_v * LSgammaV_minus - LSFplus_w * LSgammaW_minus

            ## Calculate Fcross
            LSFcross1[ii,:,:] = LSFcross_u * LSgammaU_plus - LSFcross_v * LSgammaV_plus
            LSFcross2[ii,:,:] = LSFcross_w * LSgammaW_plus - LSFcross_u * LSgammaU_minus
            LSFcross3[ii,:,:] = LSFcross_v * LSgammaV_minus - LSFcross_w * LSgammaW_minus
        LSF = np.array([[LSFplus1,LSFplus2,LSFplus3],[LSFcross1,LSFcross2,LSFcross3]])
        return LSF
def LS_xyz_response0(frange, tsegmid, params):
    LSF = asgwb_LSmich_response(frange, tsegmid, params)
    LSfstar = params['cspeed'] / (2 * np.pi * params['LSarmlength'])
    LSF_xyz = (1-np.exp(-2j*frange[None,None,:,None,None]/LSfstar))*LSF
    return LSF_xyz
def LS_xyz_response(frange, tsegmid, params):
    LSF = LS_xyz_response0(frange, tsegmid, params)
    npix = hp.nside2npix(params['nside'])

    # Array of pixel indices
    pix_idx = np.arange(npix)

    # Angular coordinates of pixel indcides
    theta, phi = hp.pix2ang(params['nside'], pix_idx)

    # Take cosine.
    ctheta = np.cos(theta)

    # Area of each pixel in sq.radians
    dOmega = hp.pixelfunc.nside2pixarea(params['nside'])

    # Create 2D array of (x,y,z) unit vectors for every sky direction.
    omegahat = np.array([np.sqrt(1 - ctheta ** 2) * np.cos(phi), np.sqrt(1 - ctheta ** 2) * np.sin(phi), ctheta])
    LSrs1, LSrs2, LSrs3 = lisa_orbits(tsegmid)
    LSudir = np.einsum('ij,ik', (LSrs2 - LSrs1), omegahat)# AB
    LSvdir = np.einsum('ij,ik', (LSrs3 - LSrs1), omegahat)# AC
    LSwdir = np.einsum('ij,ik', (LSrs3 - LSrs2), omegahat)# BC
    c = params['cspeed']
    LSfstar = c / (2 * np.pi * params['LSarmlength']) #特征频率
    f0 = frange / (2 * LSfstar)
    f_LSudir = np.zeros((f0.size, tsegmid.size, pix_idx.size), dtype='complex')
    f_LSvdir = np.zeros((f0.size, tsegmid.size, pix_idx.size), dtype='complex')
    f_LSwdir = np.zeros((f0.size, tsegmid.size, pix_idx.size), dtype='complex')
    for i in range(f0.size):
        f_LSudir[i,:,:] = frange[i]*LSudir
        f_LSvdir[i,:,:] = frange[i]*LSvdir
        f_LSwdir[i,:,:] = frange[i]*LSwdir
    LS_x_plus = LSF[0,0,:,:,:]
    LS_x_cross = LSF[1,0,:,:,:]
    LS_y_plus = LSF[0,1,:,:,:]*np.exp(2j*np.pi*f_LSudir/c)
    LS_y_cross = LSF[1,1,:,:,:]*np.exp(2j*np.pi*f_LSudir/c)
    LS_z_plus = LSF[0,2,:,:,:]*np.exp(2j*np.pi*f_LSvdir/c)
    LS_z_cross = LSF[1,2,:,:,:]*np.exp(2j*np.pi*f_LSvdir/c)
    LSF_xyz =  np.array([[LS_x_plus,LS_y_plus,LS_z_plus],[LS_x_cross,LS_y_cross,LS_z_cross]])
    return LSF_xyz

def LS_aet_response(frange,tsegmid, params):
    npix = hp.nside2npix(params['nside'])

    # Array of pixel indices
    pix_idx = np.arange(npix)

    # Angular coordinates of pixel indcides
    theta, phi = hp.pix2ang(params['nside'], pix_idx)

    # Take cosine.
    ctheta = np.cos(theta)

    # Area of each pixel in sq.radians
    dOmega = hp.pixelfunc.nside2pixarea(params['nside'])

    # Create 2D array of (x,y,z) unit vectors for every sky direction.
    omegahat = np.array([np.sqrt(1 - ctheta ** 2) * np.cos(phi), np.sqrt(1 - ctheta ** 2) * np.sin(phi), ctheta])

    # Call lisa_orbits to compute satellite positions at the midpoint of each time segment
    LSrs1, LSrs2, LSrs3 = lisa_orbits(tsegmid)
    LSudir = np.einsum('ij,ik', (LSrs2 - LSrs1), omegahat)# AB
    LSvdir = np.einsum('ij,ik', (LSrs3 - LSrs1), omegahat)# AC
    LSwdir = np.einsum('ij,ik', (LSrs3 - LSrs2), omegahat)# BC
    c = params['cspeed']
    LSfstar = c / (2 * np.pi * params['LSarmlength']) #特征频率
    f0 = frange / (2 * LSfstar)
    LSF_xyz = LS_xyz_response0(frange, tsegmid, params)
    f_LSudir = np.zeros((f0.size, tsegmid.size, pix_idx.size), dtype='complex')
    f_LSvdir = np.zeros((f0.size, tsegmid.size, pix_idx.size), dtype='complex')
    f_LSwdir = np.zeros((f0.size, tsegmid.size, pix_idx.size), dtype='complex')
    for i in range(f0.size):
        f_LSudir[i,:,:] = frange[i]*LSudir
        f_LSvdir[i,:,:] = frange[i]*LSvdir
        f_LSwdir[i,:,:] = frange[i]*LSwdir
    LSF_a_cross = (LSF_xyz[0,2,:,:,:]*np.exp(2j*np.pi*f_LSvdir/c)-LSF_xyz[0,0,:,:,:])/np.sqrt(2)
    LSF_a_plus = (LSF_xyz[1,2,:,:,:]*np.exp(2j*np.pi*f_LSvdir/c)-LSF_xyz[1,0,:,:,:])/np.sqrt(2)
    LSF_e_cross = (LSF_xyz[0,0,:,:,:]-2*LSF_xyz[0,1,:,:,:]*np.exp(2j*np.pi*f_LSudir/c)+LSF_xyz[0,2,:,:,:]*np.exp(2j*np.pi*f_LSvdir/c))/np.sqrt(6)
    LSF_e_plus = (LSF_xyz[1,0,:,:,:]-2*LSF_xyz[1,1,:,:,:]*np.exp(2j*np.pi*f_LSudir/c)+LSF_xyz[1,2,:,:,:]*np.exp(2j*np.pi*f_LSvdir/c))/np.sqrt(6)
    LSF_t_cross = (LSF_xyz[0,0,:,:,:]+LSF_xyz[0,1,:,:,:]*np.exp(2j*np.pi*f_LSudir/c)+LSF_xyz[0,2,:,:,:]*np.exp(2j*np.pi*f_LSvdir/c))/np.sqrt(3)
    LSF_t_plus = (LSF_xyz[1,0,:,:,:]+LSF_xyz[1,1,:,:,:]*np.exp(2j*np.pi*f_LSudir/c)+LSF_xyz[1,2,:,:,:]*np.exp(2j*np.pi*f_LSvdir/c))/np.sqrt(3)
    LSF_aet = np.array([[LSF_a_cross,LSF_e_cross,LSF_t_cross],[LSF_a_plus,LSF_e_plus,LSF_t_plus]])
    return LSF_aet

def TQ_LS_response_pix(frange, tsegmid, params):
    #为了检验球谐函数变换的正确性，现将pix形式的响应函数形式写下
    npix = hp.nside2npix(params['nside'])

    # Array of pixel indices
    pix_idx = np.arange(npix)

    # Angular coordinates of pixel indcides
    theta, phi = hp.pix2ang(params['nside'], pix_idx)

    # Take cosine.
    ctheta = np.cos(theta)

    # Area of each pixel in sq.radians
    omegahat = np.array([np.sqrt(1 - ctheta ** 2) * np.cos(phi), np.sqrt(1 - ctheta ** 2) * np.sin(phi), ctheta])
    if params['response type'] == 'mich':
        LSF = asgwb_LSmich_response(frange, tsegmid, params)
        TQF = asgwb_TQmich_response_pix(frange, tsegmid, params)
    elif params['response type'] == 'xyz':
        LSF = LS_xyz_response(frange, tsegmid, params)
        TQF = TQ_xyz_response_pix(frange, tsegmid, params)
    elif params['response type'] == 'aet':
        LSF = LS_aet_response(frange, tsegmid, params)
        TQF = TQ_aet_response_pix(frange, tsegmid, params)
    else:
        print('The response type is wrong')
    RTQA_LSA = np.zeros((frange.size, tsegmid.size, pix_idx.size), dtype='complex')
    RTQE_LSE = np.zeros((frange.size, tsegmid.size, pix_idx.size), dtype='complex')
    RTQT_LST = np.zeros((frange.size, tsegmid.size, pix_idx.size), dtype='complex')
    
    RTQA_LSE = np.zeros((frange.size, tsegmid.size, pix_idx.size), dtype='complex')
    RTQA_LST = np.zeros((frange.size, tsegmid.size, pix_idx.size), dtype='complex')
    RTQE_LST = np.zeros((frange.size, tsegmid.size, pix_idx.size), dtype='complex')
    
    RTQE_LSA = np.zeros((frange.size, tsegmid.size, pix_idx.size), dtype='complex')
    RTQT_LSA = np.zeros((frange.size, tsegmid.size, pix_idx.size), dtype='complex')
    RTQT_LSE = np.zeros((frange.size, tsegmid.size, pix_idx.size), dtype='complex')
    
    LSrs1, LSrs2, LSrs3 = lisa_orbits(tsegmid)
    TQrs1, TQrs2, TQrs3 = tianqin_orbits(tsegmid)
    c = params['cspeed']
    # A B C干涉点的天琴-lisa方向向量和信号传播方向的乘积
    TQ_LSudir = np.einsum('ij,ik', (TQrs1 - LSrs1) , omegahat)# AA'
    for ii in range(0,frange.size):
        RTQA_LSA[ii,:,:] = 1 / (8 * np.pi)*((np.conj(LSF[0,0,ii,:,:])*TQF[0,0,ii,:,:]+np.conj(LSF[1,0,ii,:,:])*TQF[1,0,ii,:,:]))*np.exp(-2j*np.pi*frange[ii]*TQ_LSudir/c)
        RTQE_LSE[ii,:,:] = 1 / (8 * np.pi)*((np.conj(LSF[0,1,ii,:,:])*TQF[0,1,ii,:,:]+np.conj(LSF[1,1,ii,:,:])*TQF[1,1,ii,:,:]))*np.exp(-2j*np.pi*frange[ii]*TQ_LSudir/c)   
        RTQT_LST[ii,:,:] = 1 / (8 * np.pi)*((np.conj(LSF[0,2,ii,:,:])*TQF[0,2,ii,:,:]+np.conj(LSF[1,2,ii,:,:])*TQF[1,2,ii,:,:]))*np.exp(-2j*np.pi*frange[ii]*TQ_LSudir/c)
        RTQA_LSE[ii,:,:] = 1 / (8 * np.pi)*((np.conj(LSF[0,1,ii,:,:])*TQF[0,0,ii,:,:]+np.conj(LSF[1,1,ii,:,:])*TQF[1,0,ii,:,:]))*np.exp(-2j*np.pi*frange[ii]*TQ_LSudir/c)
        RTQA_LST[ii,:,:] = 1 / (8 * np.pi)*((np.conj(LSF[0,2,ii,:,:])*TQF[0,0,ii,:,:]+np.conj(LSF[1,2,ii,:,:])*TQF[1,0,ii,:,:]))*np.exp(-2j*np.pi*frange[ii]*TQ_LSudir/c)
        RTQE_LST[ii,:,:] = 1 / (8 * np.pi)*((np.conj(LSF[0,2,ii,:,:])*TQF[0,1,ii,:,:]+np.conj(LSF[1,2,ii,:,:])*TQF[1,1,ii,:,:]))*np.exp(-2j*np.pi*frange[ii]*TQ_LSudir/c)
        RTQE_LSA[ii,:,:] = 1 / (8 * np.pi)*((np.conj(LSF[0,0,ii,:,:])*TQF[0,1,ii,:,:]+np.conj(LSF[1,0,ii,:,:])*TQF[1,1,ii,:,:]))*np.exp(-2j*np.pi*frange[ii]*TQ_LSudir/c)
        RTQT_LSA[ii,:,:] = 1 / (8 * np.pi)*((np.conj(LSF[0,0,ii,:,:])*TQF[0,2,ii,:,:]+np.conj(LSF[1,0,ii,:,:])*TQF[1,2,ii,:,:]))*np.exp(-2j*np.pi*frange[ii]*TQ_LSudir/c)          
        RTQT_LSE[ii,:,:] = 1 / (8 * np.pi)*((np.conj(LSF[0,1,ii,:,:])*TQF[0,2,ii,:,:]+np.conj(LSF[1,1,ii,:,:])*TQF[1,2,ii,:,:]))*np.exp(-2j*np.pi*frange[ii]*TQ_LSudir/c)
        
    TQ_LS_response = np.array([[RTQA_LSA,RTQA_LSE,RTQA_LST],[RTQE_LSA,RTQE_LSE,RTQE_LST],[RTQT_LSA,RTQT_LSE,RTQT_LST]])
    return TQ_LS_response
def asgwb_TQmich_response_pix(frange, tsegmid, params):

        '''
        Calculate the Antenna pattern/ detector transfer function functions to acSGWB using michelson channels,
        and using a spherical harmonic decomposition. Note that the response function to power is integrated over
        sky direction with the appropriate spherical harmonics, and averaged over polarozation. The angular
        integral is numerically done by divvying up the sky into a healpix grid.

        Note that f0 is (pi*L*f)/c and is input as an array

        Parameters
        -----------

        f0   : float
            A numpy array of scaled frequencies (see above for def)

        Returns
        ---------

        R1, R2 and R3   :   float
            Antenna Patterns for the given sky direction for the three channels, integrated over sky direction and averaged
            over polarization. The arrays are 2-d, one direction corresponds to frequency and the other to the l coeffcient.
        '''

        ## array size of almax
        #alm_size = (params['lmax'] + 1) ** 2

        npix = hp.nside2npix(params['nside'])

        # Array of pixel indices
        pix_idx = np.arange(npix)

        # Angular coordinates of pixel indcides
        theta, phi = hp.pix2ang(params['nside'], pix_idx)

        # Take cosine.
        ctheta = np.cos(theta)

        # Area of each pixel in sq.radians
        dOmega = hp.pixelfunc.nside2pixarea(params['nside'])

        # Create 2D array of (x,y,z) unit vectors for every sky direction.
        omegahat = -np.array([np.sqrt(1 - ctheta ** 2) * np.cos(phi), np.sqrt(1 - ctheta ** 2) * np.sin(phi), ctheta])

        # 获得天琴的轨道
        TQrs1, TQrs2, TQrs3 = tianqin_orbits(tsegmid)
        # 获得臂长向量和传播方向omegahat的乘积
        TQudir = np.einsum('ij,ik', (TQrs2 - TQrs1) / LA.norm(TQrs2 - TQrs1, axis=0)[None, :], omegahat)
        TQvdir = np.einsum('ij,ik', (TQrs3 - TQrs1) / LA.norm(TQrs3 - TQrs1, axis=0)[None, :], omegahat)
        TQwdir = np.einsum('ij,ik', (TQrs3 - TQrs2) / LA.norm(TQrs3 - TQrs2, axis=0)[None, :], omegahat)
        ## NB --    An attempt to directly adapt e.g. (u o u):e+ as implicit tensor calculations
        ##             as opposed to the explicit forms we've previously used. '''
        #mhat = np.array([np.sin(phi), -np.cos(phi), np.zeros(len(phi))])
        mhat = np.array([-np.sin(phi), np.cos(phi), np.zeros(len(phi))])
        nhat = np.array([np.cos(phi) * ctheta, np.sin(phi) * ctheta, -np.sqrt(1 - ctheta ** 2)])

        # 1/2 u x u : eplus. These depend only on geometry so they only have a time and directionality dependence and not of frequency
        # 天琴
        TQFplus_u = 0.5 * np.einsum("ijk,ijl", \
                                  np.einsum("ik,jk -> ijk", (TQrs2 - TQrs1) / LA.norm(TQrs2 - TQrs1, axis=0)[None, :],
                                            (TQrs2 - TQrs1) / LA.norm(TQrs2 - TQrs1, axis=0)[None, :]), \
                                  np.einsum("ik,jk -> ijk", nhat, nhat) - np.einsum("ik,jk -> ijk", mhat, mhat))

        TQFplus_v = 0.5 * np.einsum("ijk,ijl", \
                                  np.einsum("ik,jk -> ijk", (TQrs3 - TQrs1) / LA.norm(TQrs3 - TQrs1, axis=0)[None, :],
                                            (TQrs3 - TQrs1) / LA.norm(TQrs3 - TQrs1, axis=0)[None, :]), \
                                  np.einsum("ik,jk -> ijk", nhat, nhat) - np.einsum("ik,jk -> ijk", mhat, mhat))

        TQFplus_w = 0.5 * np.einsum("ijk,ijl", \
                                  np.einsum("ik,jk -> ijk", (TQrs3 - TQrs2) / LA.norm(TQrs3 - TQrs2, axis=0)[None, :],
                                            (TQrs3 - TQrs2) / LA.norm(TQrs3 - TQrs2, axis=0)[None, :]), \
                                  np.einsum("ik,jk -> ijk", nhat, nhat) - np.einsum("ik,jk -> ijk", mhat, mhat))

        # 1/2 u x u : ecross
        TQFcross_u = 0.5 * np.einsum("ijk,ijl", \
                                   np.einsum("ik,jk -> ijk", (TQrs2 - TQrs1) / LA.norm(TQrs2 - TQrs1, axis=0)[None, :],
                                             (TQrs2 - TQrs1) / LA.norm(TQrs2 - TQrs1, axis=0)[None, :]), \
                                   np.einsum("ik,jk -> ijk", mhat, nhat) + np.einsum("ik,jk -> ijk", nhat, mhat))

        TQFcross_v = 0.5 * np.einsum("ijk,ijl", \
                                   np.einsum("ik,jk -> ijk", (TQrs3 - TQrs1) / LA.norm(TQrs3 - TQrs1, axis=0)[None, :],
                                             (TQrs3 - TQrs1) / LA.norm(TQrs3 - TQrs1, axis=0)[None, :]), \
                                   np.einsum("ik,jk -> ijk", mhat, nhat) + np.einsum("ik,jk -> ijk", nhat, mhat))

        TQFcross_w = 0.5 * np.einsum("ijk,ijl", \
                                   np.einsum("ik,jk -> ijk", (TQrs3 - TQrs2) / LA.norm(TQrs3 - TQrs2, axis=0)[None, :],
                                             (TQrs3 - TQrs2) / LA.norm(TQrs3 - TQrs2, axis=0)[None, :]), \
                                   np.einsum("ik,jk -> ijk", mhat, nhat) + np.einsum("ik,jk -> ijk", nhat, mhat))
        TQfstar = params['cspeed'] / (2 * np.pi * params['TQarmlength']) #特征频率
        f0 = frange / (2 * TQfstar)
        # 设置球谐函数
        #Ylms = np.zeros((npix, alm_size), dtype='complex')
        ## Get the spherical harmonics
        #for ii in range(alm_size):
        #    lval, mval = idxtoalm(params['lmax'], ii)
        #    Ylms[:, ii] = sph_harm(mval, lval, phi, theta)

        TQFplus1 = np.zeros((f0.size, tsegmid.size, pix_idx.size), dtype='complex')
        TQFplus2 = np.zeros((f0.size, tsegmid.size, pix_idx.size), dtype='complex')
        TQFplus3 = np.zeros((f0.size, tsegmid.size, pix_idx.size), dtype='complex')
        TQFcross1 = np.zeros((f0.size, tsegmid.size, pix_idx.size), dtype='complex')
        TQFcross2 = np.zeros((f0.size, tsegmid.size, pix_idx.size), dtype='complex')
        TQFcross3 = np.zeros((f0.size, tsegmid.size, pix_idx.size), dtype='complex')

        # 天琴
        for ii in tqdm(range(0, f0.size)):
            # Calculate GW transfer function for the michelson channels
            TQgammaU_plus = 1 / 2 * (np.sinc((f0[ii]) * (1 - TQudir) / np.pi) * np.exp(-1j * f0[ii] * (3 + TQudir)) + \
                                   np.sinc((f0[ii]) * (1 + TQudir) / np.pi) * np.exp(-1j * f0[ii] * (1 + TQudir)))

            TQgammaV_plus = 1 / 2 * (np.sinc((f0[ii]) * (1 - TQvdir) / np.pi) * np.exp(-1j * f0[ii] * (3 + TQvdir)) + \
                                   np.sinc((f0[ii]) * (1 + TQvdir) / np.pi) * np.exp(-1j * f0[ii] * (1 + TQvdir)))

            TQgammaW_plus = 1 / 2 * (np.sinc((f0[ii]) * (1 - TQwdir) / np.pi) * np.exp(-1j * f0[ii] * (3 + TQwdir)) + \
                                   np.sinc((f0[ii]) * (1 + TQwdir) / np.pi) * np.exp(-1j * f0[ii] * (1 + TQwdir)))

            # Calculate GW transfer function for the michelson channels
            TQgammaU_minus = 1 / 2 * (np.sinc((f0[ii]) * (1 + TQudir) / np.pi) * np.exp(-1j * f0[ii] * (3 - TQudir)) + \
                                    np.sinc((f0[ii]) * (1 - TQudir) / np.pi) * np.exp(-1j * f0[ii] * (1 - TQudir)))

            TQgammaV_minus = 1 / 2 * (np.sinc((f0[ii]) * (1 + TQvdir) / np.pi) * np.exp(-1j * f0[ii] * (3 - TQvdir)) + \
                                    np.sinc((f0[ii]) * (1 - TQvdir) / np.pi) * np.exp(-1j * f0[ii] * (1 - TQvdir)))

            TQgammaW_minus = 1 / 2 * (np.sinc((f0[ii]) * (1 + TQwdir) / np.pi) * np.exp(-1j * f0[ii] * (3 - TQwdir)) + \
                                    np.sinc((f0[ii]) * (1 - TQwdir) / np.pi) * np.exp(-1j * f0[ii] * (1 - TQwdir)))

            ## Calculate Fplus
            TQFplus1[ii,:,:] = TQFplus_u * TQgammaU_plus - TQFplus_v * TQgammaV_plus
            TQFplus2[ii,:,:] = TQFplus_w * TQgammaW_plus - TQFplus_u * TQgammaU_minus
            TQFplus3[ii,:,:] = TQFplus_v * TQgammaV_minus - TQFplus_w * TQgammaW_minus
            ## Calculate Fcross
            TQFcross1[ii,:,:] = TQFcross_u * TQgammaU_plus - TQFcross_v * TQgammaV_plus
            TQFcross2[ii,:,:] = TQFcross_w * TQgammaW_plus - TQFcross_u * TQgammaU_minus
            TQFcross3[ii,:,:] = TQFcross_v * TQgammaV_minus - TQFcross_w * TQgammaW_minus
        TQF = np.array([[TQFplus1,TQFplus2,TQFplus3],[TQFcross1,TQFcross2,TQFcross3]])
        return TQF


# 定义天琴的xyz响应
def TQ_xyz_response_pix0(frange, tsegmid, params):
    TQF = asgwb_TQmich_response_pix(frange, tsegmid, params)
    TQfstar = params['cspeed'] / (2 * np.pi * params['TQarmlength'])
    TQF_xyz = (1-np.exp(-2j*frange[None,None,:,None,None]/TQfstar))*TQF
    return TQF_xyz
def TQ_xyz_response_pix(frange, tsegmid, params):
    TQF = TQ_xyz_response_pix0(frange, tsegmid, params)
    npix = hp.nside2npix(params['nside'])

    # Array of pixel indices
    pix_idx = np.arange(npix)

    # Angular coordinates of pixel indcides
    theta, phi = hp.pix2ang(params['nside'], pix_idx)

    # Take cosine.
    ctheta = np.cos(theta)

    # Area of each pixel in sq.radians
    dOmega = hp.pixelfunc.nside2pixarea(params['nside'])

    # Create 2D array of (x,y,z) unit vectors for every sky direction.
    omegahat = np.array([np.sqrt(1 - ctheta ** 2) * np.cos(phi), np.sqrt(1 - ctheta ** 2) * np.sin(phi), ctheta])
    TQrs1, TQrs2, TQrs3 = tianqin_orbits(tsegmid)
    TQudir = np.einsum('ij,ik', (TQrs2 - TQrs1), omegahat)# AB
    TQvdir = np.einsum('ij,ik', (TQrs3 - TQrs1), omegahat)# AC
    TQwdir = np.einsum('ij,ik', (TQrs3 - TQrs2), omegahat)# BC
    c = params['cspeed']
    TQfstar = c / (2 * np.pi * params['TQarmlength']) #特征频率
    f0 = frange / (2 * TQfstar)
    f_TQudir = np.zeros((f0.size, tsegmid.size, pix_idx.size), dtype='complex')
    f_TQvdir = np.zeros((f0.size, tsegmid.size, pix_idx.size), dtype='complex')
    f_TQwdir = np.zeros((f0.size, tsegmid.size, pix_idx.size), dtype='complex')
    for i in range(f0.size):
        f_TQudir[i,:,:] = frange[i]*TQudir
        f_TQvdir[i,:,:] = frange[i]*TQvdir
        f_TQwdir[i,:,:] = frange[i]*TQwdir
    TQ_x_plus = TQF[0,0,:,:,:]
    TQ_x_cross = TQF[1,0,:,:,:]
    TQ_y_plus = TQF[0,1,:,:,:]*np.exp(2j*np.pi*f_TQudir/c)
    TQ_y_cross = TQF[1,1,:,:,:]*np.exp(2j*np.pi*f_TQudir/c)
    TQ_z_plus = TQF[0,2,:,:,:]*np.exp(2j*np.pi*f_TQvdir/c)
    TQ_z_cross = TQF[1,2,:,:,:]*np.exp(2j*np.pi*f_TQvdir/c)
    TQF_xyz = np.array([[TQ_x_plus,TQ_y_plus,TQ_z_plus],[TQ_x_cross,TQ_y_cross,TQ_z_cross]])
    return TQF_xyz



def TQ_aet_response_pix(frange,tsegmid, params):
    npix = hp.nside2npix(params['nside'])

    # Array of pixel indices
    pix_idx = np.arange(npix)

    # Angular coordinates of pixel indcides
    theta, phi = hp.pix2ang(params['nside'], pix_idx)

    # Take cosine.
    ctheta = np.cos(theta)

    # Area of each pixel in sq.radians
    dOmega = hp.pixelfunc.nside2pixarea(params['nside'])

    # Create 2D array of (x,y,z) unit vectors for every sky direction.
    omegahat = np.array([np.sqrt(1 - ctheta ** 2) * np.cos(phi), np.sqrt(1 - ctheta ** 2) * np.sin(phi), ctheta])

    # Call lisa_orbits to compute satellite positions at the midpoint of each time segment
    TQrs1, TQrs2, TQrs3 = tianqin_orbits(tsegmid)
    TQudir = np.einsum('ij,ik', (TQrs2 - TQrs1), omegahat)# AB
    TQvdir = np.einsum('ij,ik', (TQrs3 - TQrs1), omegahat)# AC
    TQwdir = np.einsum('ij,ik', (TQrs3 - TQrs2), omegahat)# BC
    c = params['cspeed']
    TQfstar = c / (2 * np.pi * params['TQarmlength']) #特征频率
    f0 = frange / (2 * TQfstar)
    TQF_xyz = TQ_xyz_response_pix0(frange, tsegmid, params)
    f_TQudir = np.zeros((f0.size, tsegmid.size, pix_idx.size), dtype='complex')
    f_TQvdir = np.zeros((f0.size, tsegmid.size, pix_idx.size), dtype='complex')
    f_TQwdir = np.zeros((f0.size, tsegmid.size, pix_idx.size), dtype='complex')
    for i in range(f0.size):
        f_TQudir[i,:,:] = frange[i]*TQudir
        f_TQvdir[i,:,:] = frange[i]*TQvdir
        f_TQwdir[i,:,:] = frange[i]*TQwdir
    TQF_a_cross = (TQF_xyz[0,2,:,:,:]*np.exp(2j*np.pi*f_TQvdir/c)-TQF_xyz[0,0,:,:,:])/np.sqrt(2)
    TQF_a_plus = (TQF_xyz[1,2,:,:,:]*np.exp(2j*np.pi*f_TQvdir/c)-TQF_xyz[1,0,:,:,:])/np.sqrt(2)
    TQF_e_cross = (TQF_xyz[0,0,:,:,:]-2*TQF_xyz[0,1,:,:,:]*np.exp(2j*np.pi*f_TQudir/c)+TQF_xyz[0,2,:,:,:]*np.exp(2j*np.pi*f_TQvdir/c))/np.sqrt(6)
    TQF_e_plus = (TQF_xyz[1,0,:,:,:]-2*TQF_xyz[1,1,:,:,:]*np.exp(2j*np.pi*f_TQudir/c)+TQF_xyz[1,2,:,:,:]*np.exp(2j*np.pi*f_TQvdir/c))/np.sqrt(6)
    TQF_t_cross = (TQF_xyz[0,0,:,:,:]+TQF_xyz[0,1,:,:,:]*np.exp(2j*np.pi*f_TQudir/c)+TQF_xyz[0,2,:,:,:]*np.exp(2j*np.pi*f_TQvdir/c))/np.sqrt(3)
    TQF_t_plus = (TQF_xyz[1,0,:,:,:]+TQF_xyz[1,1,:,:,:]*np.exp(2j*np.pi*f_TQudir/c)+TQF_xyz[1,2,:,:,:]*np.exp(2j*np.pi*f_TQvdir/c))/np.sqrt(3)
    TQF_aet = np.array([[TQF_a_cross,TQF_e_cross,TQF_t_cross],[TQF_a_plus,TQF_e_plus,TQF_t_plus]])
    return TQF_aet


def TQ_auto_response_pix(frange, tsegmid, params):
    npix = hp.nside2npix(params['nside'])

    # Array of pixel indices
    pix_idx = np.arange(npix)

    # Angular coordinates of pixel indcides
    theta, phi = hp.pix2ang(params['nside'], pix_idx)

    # Take cosine.
    ctheta = np.cos(theta)

    # Area of each pixel in sq.radians
    dOmega = hp.pixelfunc.nside2pixarea(params['nside'])

    TQfstar = params['cspeed'] / (2 * np.pi * params['TQarmlength']) #特征频率
    f0 = frange / (2 * TQfstar)
    if params['response type'] == 'mich':
        TQF = asgwb_TQmich_response_pix(frange, tsegmid, params)
    elif params['response type'] == 'xyz':
        TQF = TQ_xyz_response_pix(frange, tsegmid, params)
    elif params['response type'] == 'aet':
        TQF = TQ_aet_response_pix(frange,tsegmid, params)
    else:
        print('The response type is wrong')
            
    RTQ1 = np.zeros((f0.size, tsegmid.size, pix_idx.size), dtype='complex')
    RTQ2 = np.zeros((f0.size, tsegmid.size, pix_idx.size), dtype='complex')
    RTQ3 = np.zeros((f0.size, tsegmid.size, pix_idx.size), dtype='complex')
    RTQ12 = np.zeros((f0.size, tsegmid.size, pix_idx.size), dtype='complex')
    RTQ13 = np.zeros((f0.size, tsegmid.size, pix_idx.size), dtype='complex')
    RTQ23 = np.zeros((f0.size, tsegmid.size, pix_idx.size), dtype='complex')
    for ii in tqdm(range(0,f0.size)):
        RTQ1[ii,:,:] = 1 / (8 * np.pi)*((TQF[0,0,ii,:,:]*np.conj(TQF[0,0,ii,:,:])+np.conj(TQF[1,0,ii,:,:])*TQF[1,0,ii,:,:]))
        RTQ2[ii,:,:] = 1 / (8 * np.pi)*((np.conj(TQF[0,1,ii,:,:])*TQF[0,1,ii,:,:]+np.conj(TQF[1,1,ii,:,:])*TQF[1,1,ii,:,:]))
        RTQ3[ii,:,:] = 1 / (8 * np.pi)*((np.conj(TQF[0,2,ii,:,:])*TQF[0,2,ii,:,:]+np.conj(TQF[1,2,ii,:,:])*TQF[1,2,ii,:,:]))
        RTQ12[ii,:,:] = 1 / (8 * np.pi)*(TQF[0,0,ii,:,:]*np.conj(TQF[0,1,ii,:,:])+TQF[1,0,ii,:,:]*np.conj(TQF[1,1,ii,:,:]))
        RTQ13[ii,:,:] = 1 / (8 * np.pi)*(TQF[0,0,ii,:,:]*np.conj(TQF[0,2,ii,:,:])+TQF[1,0,ii,:,:]*np.conj(TQF[1,2,ii,:,:]))
        RTQ23[ii,:,:] = 1 / (8 * np.pi)*(TQF[0,1,ii,:,:]*np.conj(TQF[0,2,ii,:,:])+TQF[1,1,ii,:,:]*np.conj(TQF[1,2,ii,:,:]))
    RTQ21 = np.conj(RTQ12)
    RTQ31 = np.conj(RTQ13)
    RTQ32 = np.conj(RTQ23)
    TQ_response = np.array([[RTQ1,RTQ12,RTQ13],[RTQ21,RTQ2,RTQ23],[RTQ31,RTQ32,RTQ3]])
    return TQ_response

def LS_auto_response_pix(frange, tsegmid, params):
    npix = hp.nside2npix(params['nside'])

    # Array of pixel indices
    pix_idx = np.arange(npix)

    # Angular coordinates of pixel indcides
    theta, phi = hp.pix2ang(params['nside'], pix_idx)

    # Take cosine.
    ctheta = np.cos(theta)

    # Area of each pixel in sq.radians
    dOmega = hp.pixelfunc.nside2pixarea(params['nside'])

    LSfstar = params['cspeed'] / (2 * np.pi * params['LSarmlength']) #特征频率
    f0 = frange / (2 * LSfstar)
    if params['response type'] == 'mich':
        LSF = asgwb_LSmich_response(frange, tsegmid, params)
    elif params['response type'] == 'xyz':
        LSF = LS_xyz_response(frange, tsegmid, params)
    elif params['response type'] == 'aet':
        LSF = LS_aet_response(frange,tsegmid, params)
    else:
        print('The response type is wrong')
            
    RLS1 = np.zeros((f0.size, tsegmid.size, pix_idx.size), dtype='complex')
    RLS2 = np.zeros((f0.size, tsegmid.size, pix_idx.size), dtype='complex')
    RLS3 = np.zeros((f0.size, tsegmid.size, pix_idx.size), dtype='complex')
    RLS12 = np.zeros((f0.size, tsegmid.size, pix_idx.size), dtype='complex')
    RLS13 = np.zeros((f0.size, tsegmid.size, pix_idx.size), dtype='complex')
    RLS23 = np.zeros((f0.size, tsegmid.size, pix_idx.size), dtype='complex')
    for ii in tqdm(range(0,f0.size)):
        RLS1[ii,:,:] = 1 / (8 * np.pi)*((np.conj(LSF[0,0,ii,:,:])*LSF[0,0,ii,:,:]+np.conj(LSF[1,0,ii,:,:])*LSF[1,0,ii,:,:]))
        RLS2[ii,:,:] = 1 / (8 * np.pi)*((np.conj(LSF[0,1,ii,:,:])*LSF[0,1,ii,:,:]+np.conj(LSF[1,1,ii,:,:])*LSF[1,1,ii,:,:]))
        RLS3[ii,:,:] = 1 / (8 * np.pi)*((np.conj(LSF[0,2,ii,:,:])*LSF[0,2,ii,:,:]+np.conj(LSF[1,2,ii,:,:])*LSF[1,2,ii,:,:]))
        RLS12[ii,:,:] = 1 / (8 * np.pi)*((np.conj(LSF[0,0,ii,:,:])*LSF[0,1,ii,:,:]+np.conj(LSF[1,0,ii,:,:])*LSF[1,1,ii,:,:]))
        RLS13[ii,:,:] = 1 / (8 * np.pi)*((np.conj(LSF[0,0,ii,:,:])*LSF[0,2,ii,:,:]+np.conj(LSF[1,0,ii,:,:])*LSF[1,2,ii,:,:]))
        RLS23[ii,:,:] = 1 / (8 * np.pi)*((np.conj(LSF[0,1,ii,:,:])*LSF[0,2,ii,:,:]+np.conj(LSF[1,1,ii,:,:])*LSF[1,2,ii,:,:]))
    
    RLS21 = np.conj(RLS12)
    RLS31 = np.conj(RLS13)
    RLS32 = np.conj(RLS23)
    
    LS_response = np.array([[RLS1,RLS12,RLS13],[RLS21,RLS2,RLS23],[RLS31,RLS32,RLS3]])
    return LS_response

def asgwb_Tjmich_response(frange, tsegmid, params):

        '''
        Calculate the Antenna pattern/ detector transfer function functions to acSGWB using michelson channels,
        and using a spherical harmonic decomposition. Note that the response function to power is integrated over
        sky direction with the appropriate spherical harmonics, and averaged over polarozation. The angular
        integral is numerically done by divvying up the sky into a healpix grid.

        Note that f0 is (pi*L*f)/c and is input as an array

        Parameters
        -----------

        f0   : float
            A numpy array of scaled frequencies (see above for def)

        Returns
        ---------

        R1, R2 and R3   :   float
            Antenna Patterns for the given sky direction for the three channels, integrated over sky direction and averaged
            over polarization. The arrays are 2-d, one direction corresponds to frequency and the other to the l coeffcient.
        '''

        ## array size of almax
        #alm_size = (self.almax + 1) ** 2

        npix = hp.nside2npix(params['nside'])

        # Array of pixel indices
        pix_idx = np.arange(npix)

        # Angular coordinates of pixel indcides
        theta, phi = hp.pix2ang(params['nside'], pix_idx)

        # Take cosine.
        ctheta = np.cos(theta)

        # Area of each pixel in sq.radians

        # Create 2D array of (x,y,z) unit vectors for every sky direction.
        omegahat = -np.array([np.sqrt(1 - ctheta ** 2) * np.cos(phi), np.sqrt(1 - ctheta ** 2) * np.sin(phi), ctheta])

        # 获得天琴的轨道
        LSrs1, LSrs2, LSrs3 = taiji_orbits(tsegmid)
        # 获得臂长向量和传播方向omegahat的乘积
        LSudir = np.einsum('ij,ik', (LSrs2 - LSrs1) / LA.norm(LSrs2 - LSrs1, axis=0)[None, :], omegahat)
        LSvdir = np.einsum('ij,ik', (LSrs3 - LSrs1) / LA.norm(LSrs3 - LSrs1, axis=0)[None, :], omegahat)
        LSwdir = np.einsum('ij,ik', (LSrs3 - LSrs2) / LA.norm(LSrs3 - LSrs2, axis=0)[None, :], omegahat)
        ## NB --    An attempt to directly adapt e.g. (u o u):e+ as implicit tensor calculations
        ##             as opposed to the explicit forms we've previously used. '''
        mhat = np.array([-np.sin(phi), np.cos(phi), np.zeros(len(phi))])
        nhat = np.array([np.cos(phi) * ctheta, np.sin(phi) * ctheta, -np.sqrt(1 - ctheta ** 2)])

        # 1/2 u x u : eplus. These depend only on geometry so they only have a time and directionality dependence and not of frequency
        # lisa

        LSFplus_u = 0.5 * np.einsum("ijk,ijl", \
                                  np.einsum("ik,jk -> ijk", (LSrs2 - LSrs1) / LA.norm(LSrs2 - LSrs1, axis=0)[None, :],
                                            (LSrs2 - LSrs1) / LA.norm(LSrs2 - LSrs1, axis=0)[None, :]), \
                                  np.einsum("ik,jk -> ijk", nhat, nhat) - np.einsum("ik,jk -> ijk", mhat, mhat))

        LSFplus_v = 0.5 * np.einsum("ijk,ijl", \
                                  np.einsum("ik,jk -> ijk", (LSrs3 - LSrs1) / LA.norm(LSrs3 - LSrs1, axis=0)[None, :],
                                            (LSrs3 - LSrs1) / LA.norm(LSrs3 - LSrs1, axis=0)[None, :]), \
                                  np.einsum("ik,jk -> ijk", nhat, nhat) - np.einsum("ik,jk -> ijk", mhat, mhat))

        LSFplus_w = 0.5 * np.einsum("ijk,ijl", \
                                  np.einsum("ik,jk -> ijk", (LSrs3 - LSrs2) / LA.norm(LSrs3 - LSrs2, axis=0)[None, :],
                                            (LSrs3 - LSrs2) / LA.norm(LSrs3 - LSrs2, axis=0)[None, :]), \
                                  np.einsum("ik,jk -> ijk", nhat, nhat) - np.einsum("ik,jk -> ijk", mhat, mhat))

        # 1/2 u x u : ecross
        LSFcross_u = 0.5 * np.einsum("ijk,ijl", \
                                   np.einsum("ik,jk -> ijk", (LSrs2 - LSrs1) / LA.norm(LSrs2 - LSrs1, axis=0)[None, :],
                                             (LSrs2 - LSrs1) / LA.norm(LSrs2 - LSrs1, axis=0)[None, :]), \
                                   np.einsum("ik,jk -> ijk", mhat, nhat) + np.einsum("ik,jk -> ijk", nhat, mhat))

        LSFcross_v = 0.5 * np.einsum("ijk,ijl", \
                                   np.einsum("ik,jk -> ijk", (LSrs3 - LSrs1) / LA.norm(LSrs3 - LSrs1, axis=0)[None, :],
                                             (LSrs3 - LSrs1) / LA.norm(LSrs3 - LSrs1, axis=0)[None, :]), \
                                   np.einsum("ik,jk -> ijk", mhat, nhat) + np.einsum("ik,jk -> ijk", nhat, mhat))

        LSFcross_w = 0.5 * np.einsum("ijk,ijl", \
                                   np.einsum("ik,jk -> ijk", (LSrs3 - LSrs2) / LA.norm(LSrs3 - LSrs2, axis=0)[None, :],
                                             (LSrs3 - LSrs2) / LA.norm(LSrs3 - LSrs2, axis=0)[None, :]), \
                                   np.einsum("ik,jk -> ijk", mhat, nhat) + np.einsum("ik,jk -> ijk", nhat, mhat))
        LSfstar = params['cspeed'] / (2 * np.pi * params['Tjarmlength']) #特征频率
        LSf0 = frange / (2 * LSfstar)
        LSFplus1 = np.zeros((LSf0.size, tsegmid.size, pix_idx.size), dtype='complex')
        LSFplus2 = np.zeros((LSf0.size, tsegmid.size, pix_idx.size), dtype='complex')
        LSFplus3 = np.zeros((LSf0.size, tsegmid.size, pix_idx.size), dtype='complex')
        LSFcross1 = np.zeros((LSf0.size, tsegmid.size, pix_idx.size), dtype='complex')
        LSFcross2 = np.zeros((LSf0.size, tsegmid.size, pix_idx.size), dtype='complex')
        LSFcross3 = np.zeros((LSf0.size, tsegmid.size, pix_idx.size), dtype='complex')

        # lisa
        for ii in tqdm(range(0, LSf0.size)):
            LSgammaU_plus = 1 / 2 * (np.sinc((LSf0[ii]) * (1 - LSudir) / np.pi) * np.exp(-1j * LSf0[ii] * (3 + LSudir)) + \
                                   np.sinc((LSf0[ii]) * (1 + LSudir) / np.pi) * np.exp(-1j * LSf0[ii] * (1 + LSudir)))

            LSgammaV_plus = 1 / 2 * (np.sinc((LSf0[ii]) * (1 - LSvdir) / np.pi) * np.exp(-1j * LSf0[ii] * (3 + LSvdir)) + \
                                   np.sinc((LSf0[ii]) * (1 + LSvdir) / np.pi) * np.exp(-1j * LSf0[ii] * (1 + LSvdir)))

            LSgammaW_plus = 1 / 2 * (np.sinc((LSf0[ii]) * (1 - LSwdir) / np.pi) * np.exp(-1j * LSf0[ii] * (3 + LSwdir)) + \
                                   np.sinc((LSf0[ii]) * (1 + LSwdir) / np.pi) * np.exp(-1j * LSf0[ii] * (1 + LSwdir)))

            # Calculate GW transfer function for the michelson channels
            LSgammaU_minus = 1 / 2 * (np.sinc((LSf0[ii]) * (1 + LSudir) / np.pi) * np.exp(-1j * LSf0[ii] * (3 - LSudir)) + \
                                    np.sinc((LSf0[ii]) * (1 - LSudir) / np.pi) * np.exp(-1j * LSf0[ii] * (1 - LSudir)))

            LSgammaV_minus = 1 / 2 * (np.sinc((LSf0[ii]) * (1 + LSvdir) / np.pi) * np.exp(-1j * LSf0[ii] * (3 - LSvdir)) + \
                                    np.sinc((LSf0[ii]) * (1 - LSvdir) / np.pi) * np.exp(-1j * LSf0[ii] * (1 - LSvdir)))

            LSgammaW_minus = 1 / 2 * (np.sinc((LSf0[ii]) * (1 + LSwdir) / np.pi) * np.exp(-1j * LSf0[ii] * (3 - LSwdir)) + \
                                    np.sinc((LSf0[ii]) * (1 - LSwdir) / np.pi) * np.exp(-1j * LSf0[ii] * (1 - LSwdir)))
            ## Calculate Fplus
            LSFplus1[ii,:,:] = LSFplus_u * LSgammaU_plus - LSFplus_v * LSgammaV_plus
            LSFplus2[ii,:,:] = LSFplus_w * LSgammaW_plus - LSFplus_u * LSgammaU_minus
            LSFplus3[ii,:,:] = LSFplus_v * LSgammaV_minus - LSFplus_w * LSgammaW_minus

            ## Calculate Fcross
            LSFcross1[ii,:,:] = LSFcross_u * LSgammaU_plus - LSFcross_v * LSgammaV_plus
            LSFcross2[ii,:,:] = LSFcross_w * LSgammaW_plus - LSFcross_u * LSgammaU_minus
            LSFcross3[ii,:,:] = LSFcross_v * LSgammaV_minus - LSFcross_w * LSgammaW_minus
        LSF = np.array([[LSFplus1,LSFplus2,LSFplus3],[LSFcross1,LSFcross2,LSFcross3]])
        return LSF
def Tj_xyz_response0(frange, tsegmid, params):
    LSF = asgwb_Tjmich_response(frange, tsegmid, params)
    LSfstar = params['cspeed'] / (2 * np.pi * params['Tjarmlength'])
    LSF_xyz = (1-np.exp(-2j*frange[None,None,:,None,None]/LSfstar))*LSF
    return LSF_xyz
def Tj_xyz_response(frange, tsegmid, params):
    LSF = Tj_xyz_response0(frange, tsegmid, params)
    npix = hp.nside2npix(params['nside'])

    # Array of pixel indices
    pix_idx = np.arange(npix)

    # Angular coordinates of pixel indcides
    theta, phi = hp.pix2ang(params['nside'], pix_idx)

    # Take cosine.
    ctheta = np.cos(theta)
    # Create 2D array of (x,y,z) unit vectors for every sky direction.
    omegahat = np.array([np.sqrt(1 - ctheta ** 2) * np.cos(phi), np.sqrt(1 - ctheta ** 2) * np.sin(phi), ctheta])
    LSrs1, LSrs2, LSrs3 = taiji_orbits(tsegmid)
    LSudir = np.einsum('ij,ik', (LSrs2 - LSrs1), omegahat)# AB
    LSvdir = np.einsum('ij,ik', (LSrs3 - LSrs1), omegahat)# AC
    LSwdir = np.einsum('ij,ik', (LSrs3 - LSrs2), omegahat)# BC
    c = params['cspeed']
    LSfstar = c / (2 * np.pi * params['Tjarmlength']) #特征频率
    f0 = frange / (2 * LSfstar)
    f_LSudir = np.zeros((f0.size, tsegmid.size, pix_idx.size), dtype='complex')
    f_LSvdir = np.zeros((f0.size, tsegmid.size, pix_idx.size), dtype='complex')
    f_LSwdir = np.zeros((f0.size, tsegmid.size, pix_idx.size), dtype='complex')
    for i in range(f0.size):
        f_LSudir[i,:,:] = frange[i]*LSudir
        f_LSvdir[i,:,:] = frange[i]*LSvdir
        f_LSwdir[i,:,:] = frange[i]*LSwdir
    LS_x_plus = LSF[0,0,:,:,:]
    LS_x_cross = LSF[1,0,:,:,:]
    LS_y_plus = LSF[0,1,:,:,:]*np.exp(2j*np.pi*f_LSudir/c)
    LS_y_cross = LSF[1,1,:,:,:]*np.exp(2j*np.pi*f_LSudir/c)
    LS_z_plus = LSF[0,2,:,:,:]*np.exp(2j*np.pi*f_LSvdir/c)
    LS_z_cross = LSF[1,2,:,:,:]*np.exp(2j*np.pi*f_LSvdir/c)
    LSF_xyz =  np.array([[LS_x_plus,LS_y_plus,LS_z_plus],[LS_x_cross,LS_y_cross,LS_z_cross]])
    return LSF_xyz

def Tj_aet_response(frange,tsegmid, params):
    npix = hp.nside2npix(params['nside'])

    # Array of pixel indices
    pix_idx = np.arange(npix)

    # Angular coordinates of pixel indcides
    theta, phi = hp.pix2ang(params['nside'], pix_idx)

    # Take cosine.
    ctheta = np.cos(theta)

    # Create 2D array of (x,y,z) unit vectors for every sky direction.
    omegahat = np.array([np.sqrt(1 - ctheta ** 2) * np.cos(phi), np.sqrt(1 - ctheta ** 2) * np.sin(phi), ctheta])

    # Call lisa_orbits to compute satellite positions at the midpoint of each time segment
    LSrs1, LSrs2, LSrs3 = taiji_orbits(tsegmid)
    LSudir = np.einsum('ij,ik', (LSrs2 - LSrs1), omegahat)# AB
    LSvdir = np.einsum('ij,ik', (LSrs3 - LSrs1), omegahat)# AC
    LSwdir = np.einsum('ij,ik', (LSrs3 - LSrs2), omegahat)# BC
    c = params['cspeed']
    LSfstar = c / (2 * np.pi * params['Tjarmlength']) #特征频率
    f0 = frange / (2 * LSfstar)
    LSF_xyz = Tj_xyz_response0(frange, tsegmid, params)
    f_LSudir = np.zeros((f0.size, tsegmid.size, pix_idx.size), dtype='complex')
    f_LSvdir = np.zeros((f0.size, tsegmid.size, pix_idx.size), dtype='complex')
    f_LSwdir = np.zeros((f0.size, tsegmid.size, pix_idx.size), dtype='complex')
    for i in range(f0.size):
        f_LSudir[i,:,:] = frange[i]*LSudir
        f_LSvdir[i,:,:] = frange[i]*LSvdir
        f_LSwdir[i,:,:] = frange[i]*LSwdir
    LSF_a_cross = (LSF_xyz[0,2,:,:,:]*np.exp(2j*np.pi*f_LSvdir/c)-LSF_xyz[0,0,:,:,:])/np.sqrt(2)
    LSF_a_plus = (LSF_xyz[1,2,:,:,:]*np.exp(2j*np.pi*f_LSvdir/c)-LSF_xyz[1,0,:,:,:])/np.sqrt(2)
    LSF_e_cross = (LSF_xyz[0,0,:,:,:]-2*LSF_xyz[0,1,:,:,:]*np.exp(2j*np.pi*f_LSudir/c)+LSF_xyz[0,2,:,:,:]*np.exp(2j*np.pi*f_LSvdir/c))/np.sqrt(6)
    LSF_e_plus = (LSF_xyz[1,0,:,:,:]-2*LSF_xyz[1,1,:,:,:]*np.exp(2j*np.pi*f_LSudir/c)+LSF_xyz[1,2,:,:,:]*np.exp(2j*np.pi*f_LSvdir/c))/np.sqrt(6)
    LSF_t_cross = (LSF_xyz[0,0,:,:,:]+LSF_xyz[0,1,:,:,:]*np.exp(2j*np.pi*f_LSudir/c)+LSF_xyz[0,2,:,:,:]*np.exp(2j*np.pi*f_LSvdir/c))/np.sqrt(3)
    LSF_t_plus = (LSF_xyz[1,0,:,:,:]+LSF_xyz[1,1,:,:,:]*np.exp(2j*np.pi*f_LSudir/c)+LSF_xyz[1,2,:,:,:]*np.exp(2j*np.pi*f_LSvdir/c))/np.sqrt(3)
    LSF_aet = np.array([[LSF_a_cross,LSF_e_cross,LSF_t_cross],[LSF_a_plus,LSF_e_plus,LSF_t_plus]])
    return LSF_aet

def TQ_Tj_response_pix(frange, tsegmid, params):
    #为了检验球谐函数变换的正确性，现将pix形式的响应函数形式写下
    npix = hp.nside2npix(params['nside'])

    # Array of pixel indices
    pix_idx = np.arange(npix)

    # Angular coordinates of pixel indcides
    theta, phi = hp.pix2ang(params['nside'], pix_idx)

    # Take cosine.
    ctheta = np.cos(theta)

    # Area of each pixel in sq.radians
    omegahat = np.array([np.sqrt(1 - ctheta ** 2) * np.cos(phi), np.sqrt(1 - ctheta ** 2) * np.sin(phi), ctheta])
    if params['response type'] == 'mich':
        LSF = asgwb_Tjmich_response(frange, tsegmid, params)
        TQF = asgwb_TQmich_response_pix(frange, tsegmid, params)
    elif params['response type'] == 'xyz':
        LSF = Tj_xyz_response(frange, tsegmid, params)
        TQF = TQ_xyz_response_pix(frange, tsegmid, params)
    elif params['response type'] == 'aet':
        LSF = Tj_aet_response(frange, tsegmid, params)
        TQF = TQ_aet_response_pix(frange, tsegmid, params)
    else:
        print('The response type is wrong')
    RTQA_LSA = np.zeros((frange.size, tsegmid.size, pix_idx.size), dtype='complex')
    RTQE_LSE = np.zeros((frange.size, tsegmid.size, pix_idx.size), dtype='complex')
    RTQT_LST = np.zeros((frange.size, tsegmid.size, pix_idx.size), dtype='complex')
    
    RTQA_LSE = np.zeros((frange.size, tsegmid.size, pix_idx.size), dtype='complex')
    RTQA_LST = np.zeros((frange.size, tsegmid.size, pix_idx.size), dtype='complex')
    RTQE_LST = np.zeros((frange.size, tsegmid.size, pix_idx.size), dtype='complex')
    
    RTQE_LSA = np.zeros((frange.size, tsegmid.size, pix_idx.size), dtype='complex')
    RTQT_LSA = np.zeros((frange.size, tsegmid.size, pix_idx.size), dtype='complex')
    RTQT_LSE = np.zeros((frange.size, tsegmid.size, pix_idx.size), dtype='complex')
    
    LSrs1, LSrs2, LSrs3 = taiji_orbits(tsegmid)
    TQrs1, TQrs2, TQrs3 = tianqin_orbits(tsegmid)
    c = params['cspeed']
    # A B C干涉点的天琴-lisa方向向量和信号传播方向的乘积
    TQ_LSudir = np.einsum('ij,ik', (TQrs1 - LSrs1) , omegahat)# AA'
    for ii in range(0,frange.size):
        RTQA_LSA[ii,:,:] = 1 / (8 * np.pi)*((np.conj(LSF[0,0,ii,:,:])*TQF[0,0,ii,:,:]+np.conj(LSF[1,0,ii,:,:])*TQF[1,0,ii,:,:]))*np.exp(-2j*np.pi*frange[ii]*TQ_LSudir/c)
        RTQE_LSE[ii,:,:] = 1 / (8 * np.pi)*((np.conj(LSF[0,1,ii,:,:])*TQF[0,1,ii,:,:]+np.conj(LSF[1,1,ii,:,:])*TQF[1,1,ii,:,:]))*np.exp(-2j*np.pi*frange[ii]*TQ_LSudir/c)   
        RTQT_LST[ii,:,:] = 1 / (8 * np.pi)*((np.conj(LSF[0,2,ii,:,:])*TQF[0,2,ii,:,:]+np.conj(LSF[1,2,ii,:,:])*TQF[1,2,ii,:,:]))*np.exp(-2j*np.pi*frange[ii]*TQ_LSudir/c)
        RTQA_LSE[ii,:,:] = 1 / (8 * np.pi)*((np.conj(LSF[0,1,ii,:,:])*TQF[0,0,ii,:,:]+np.conj(LSF[1,1,ii,:,:])*TQF[1,0,ii,:,:]))*np.exp(-2j*np.pi*frange[ii]*TQ_LSudir/c)
        RTQA_LST[ii,:,:] = 1 / (8 * np.pi)*((np.conj(LSF[0,2,ii,:,:])*TQF[0,0,ii,:,:]+np.conj(LSF[1,2,ii,:,:])*TQF[1,0,ii,:,:]))*np.exp(-2j*np.pi*frange[ii]*TQ_LSudir/c)
        RTQE_LST[ii,:,:] = 1 / (8 * np.pi)*((np.conj(LSF[0,2,ii,:,:])*TQF[0,1,ii,:,:]+np.conj(LSF[1,2,ii,:,:])*TQF[1,1,ii,:,:]))*np.exp(-2j*np.pi*frange[ii]*TQ_LSudir/c)
        RTQE_LSA[ii,:,:] = 1 / (8 * np.pi)*((np.conj(LSF[0,0,ii,:,:])*TQF[0,1,ii,:,:]+np.conj(LSF[1,0,ii,:,:])*TQF[1,1,ii,:,:]))*np.exp(-2j*np.pi*frange[ii]*TQ_LSudir/c)
        RTQT_LSA[ii,:,:] = 1 / (8 * np.pi)*((np.conj(LSF[0,0,ii,:,:])*TQF[0,2,ii,:,:]+np.conj(LSF[1,0,ii,:,:])*TQF[1,2,ii,:,:]))*np.exp(-2j*np.pi*frange[ii]*TQ_LSudir/c)          
        RTQT_LSE[ii,:,:] = 1 / (8 * np.pi)*((np.conj(LSF[0,1,ii,:,:])*TQF[0,2,ii,:,:]+np.conj(LSF[1,1,ii,:,:])*TQF[1,2,ii,:,:]))*np.exp(-2j*np.pi*frange[ii]*TQ_LSudir/c)
        
    TQ_Tj_response = np.array([[RTQA_LSA,RTQA_LSE,RTQA_LST],[RTQE_LSA,RTQE_LSE,RTQE_LST],[RTQT_LSA,RTQT_LSE,RTQT_LST]])
    return TQ_Tj_response

def LS_Tj_response_pix(frange, tsegmid, params):
    #为了检验球谐函数变换的正确性，现将pix形式的响应函数形式写下
    npix = hp.nside2npix(params['nside'])

    # Array of pixel indices
    pix_idx = np.arange(npix)

    # Angular coordinates of pixel indcides
    theta, phi = hp.pix2ang(params['nside'], pix_idx)

    # Take cosine.
    ctheta = np.cos(theta)

    # Area of each pixel in sq.radians
    omegahat = np.array([np.sqrt(1 - ctheta ** 2) * np.cos(phi), np.sqrt(1 - ctheta ** 2) * np.sin(phi), ctheta])
    if params['response type'] == 'mich':
        LSF = asgwb_LSmich_response(frange, tsegmid, params)
        TQF = asgwb_Tjmich_response(frange, tsegmid, params)
    elif params['response type'] == 'xyz':
        LSF = LS_xyz_response(frange, tsegmid, params)
        TQF = Tj_xyz_response(frange, tsegmid, params)
    elif params['response type'] == 'aet':
        LSF = LS_aet_response(frange, tsegmid, params)
        TQF = Tj_aet_response(frange, tsegmid, params)
    else:
        print('The response type is wrong')
    RTQA_LSA = np.zeros((frange.size, tsegmid.size, pix_idx.size), dtype='complex')
    RTQE_LSE = np.zeros((frange.size, tsegmid.size, pix_idx.size), dtype='complex')
    RTQT_LST = np.zeros((frange.size, tsegmid.size, pix_idx.size), dtype='complex')
    
    RTQA_LSE = np.zeros((frange.size, tsegmid.size, pix_idx.size), dtype='complex')
    RTQA_LST = np.zeros((frange.size, tsegmid.size, pix_idx.size), dtype='complex')
    RTQE_LST = np.zeros((frange.size, tsegmid.size, pix_idx.size), dtype='complex')
    
    RTQE_LSA = np.zeros((frange.size, tsegmid.size, pix_idx.size), dtype='complex')
    RTQT_LSA = np.zeros((frange.size, tsegmid.size, pix_idx.size), dtype='complex')
    RTQT_LSE = np.zeros((frange.size, tsegmid.size, pix_idx.size), dtype='complex')
    
    LSrs1, LSrs2, LSrs3 = lisa_orbits(tsegmid)
    TQrs1, TQrs2, TQrs3 = taiji_orbits(tsegmid)
    c = params['cspeed']
    # A B C干涉点的天琴-lisa方向向量和信号传播方向的乘积
    TQ_LSudir = np.einsum('ij,ik', (TQrs1 - LSrs1) , omegahat)# AA'
    for ii in range(0,frange.size):
        RTQA_LSA[ii,:,:] = 1 / (8 * np.pi)*((np.conj(LSF[0,0,ii,:,:])*TQF[0,0,ii,:,:]+np.conj(LSF[1,0,ii,:,:])*TQF[1,0,ii,:,:]))*np.exp(-2j*np.pi*frange[ii]*TQ_LSudir/c)
        RTQE_LSE[ii,:,:] = 1 / (8 * np.pi)*((np.conj(LSF[0,1,ii,:,:])*TQF[0,1,ii,:,:]+np.conj(LSF[1,1,ii,:,:])*TQF[1,1,ii,:,:]))*np.exp(-2j*np.pi*frange[ii]*TQ_LSudir/c)   
        RTQT_LST[ii,:,:] = 1 / (8 * np.pi)*((np.conj(LSF[0,2,ii,:,:])*TQF[0,2,ii,:,:]+np.conj(LSF[1,2,ii,:,:])*TQF[1,2,ii,:,:]))*np.exp(-2j*np.pi*frange[ii]*TQ_LSudir/c)
        RTQA_LSE[ii,:,:] = 1 / (8 * np.pi)*((np.conj(LSF[0,1,ii,:,:])*TQF[0,0,ii,:,:]+np.conj(LSF[1,1,ii,:,:])*TQF[1,0,ii,:,:]))*np.exp(-2j*np.pi*frange[ii]*TQ_LSudir/c)
        RTQA_LST[ii,:,:] = 1 / (8 * np.pi)*((np.conj(LSF[0,2,ii,:,:])*TQF[0,0,ii,:,:]+np.conj(LSF[1,2,ii,:,:])*TQF[1,0,ii,:,:]))*np.exp(-2j*np.pi*frange[ii]*TQ_LSudir/c)
        RTQE_LST[ii,:,:] = 1 / (8 * np.pi)*((np.conj(LSF[0,2,ii,:,:])*TQF[0,1,ii,:,:]+np.conj(LSF[1,2,ii,:,:])*TQF[1,1,ii,:,:]))*np.exp(-2j*np.pi*frange[ii]*TQ_LSudir/c)
        RTQE_LSA[ii,:,:] = 1 / (8 * np.pi)*((np.conj(LSF[0,0,ii,:,:])*TQF[0,1,ii,:,:]+np.conj(LSF[1,0,ii,:,:])*TQF[1,1,ii,:,:]))*np.exp(-2j*np.pi*frange[ii]*TQ_LSudir/c)
        RTQT_LSA[ii,:,:] = 1 / (8 * np.pi)*((np.conj(LSF[0,0,ii,:,:])*TQF[0,2,ii,:,:]+np.conj(LSF[1,0,ii,:,:])*TQF[1,2,ii,:,:]))*np.exp(-2j*np.pi*frange[ii]*TQ_LSudir/c)          
        RTQT_LSE[ii,:,:] = 1 / (8 * np.pi)*((np.conj(LSF[0,1,ii,:,:])*TQF[0,2,ii,:,:]+np.conj(LSF[1,1,ii,:,:])*TQF[1,2,ii,:,:]))*np.exp(-2j*np.pi*frange[ii]*TQ_LSudir/c)
        
    LS_Tj_response = np.array([[RTQA_LSA,RTQA_LSE,RTQA_LST],[RTQE_LSA,RTQE_LSE,RTQE_LST],[RTQT_LSA,RTQT_LSE,RTQT_LST]])
    return LS_Tj_response

def Tj_auto_response_pix(frange, tsegmid, params):
    npix = hp.nside2npix(params['nside'])

    # Array of pixel indices
    pix_idx = np.arange(npix)

    # Angular coordinates of pixel indcides
    LSfstar = params['cspeed'] / (2 * np.pi * params['Tjarmlength']) #特征频率
    f0 = frange / (2 * LSfstar)
    if params['response type'] == 'mich':
        LSF = asgwb_Tjmich_response(frange, tsegmid, params)
    elif params['response type'] == 'xyz':
        LSF = Tj_xyz_response(frange, tsegmid, params)
    elif params['response type'] == 'aet':
        LSF = Tj_aet_response(frange,tsegmid, params)
    else:
        print('The response type is wrong')
            
    RLS1 = np.zeros((f0.size, tsegmid.size, pix_idx.size), dtype='complex')
    RLS2 = np.zeros((f0.size, tsegmid.size, pix_idx.size), dtype='complex')
    RLS3 = np.zeros((f0.size, tsegmid.size, pix_idx.size), dtype='complex')
    RLS12 = np.zeros((f0.size, tsegmid.size, pix_idx.size), dtype='complex')
    RLS13 = np.zeros((f0.size, tsegmid.size, pix_idx.size), dtype='complex')
    RLS23 = np.zeros((f0.size, tsegmid.size, pix_idx.size), dtype='complex')
    for ii in tqdm(range(0,f0.size)):
        RLS1[ii,:,:] = 1 / (8 * np.pi)*((np.conj(LSF[0,0,ii,:,:])*LSF[0,0,ii,:,:]+np.conj(LSF[1,0,ii,:,:])*LSF[1,0,ii,:,:]))
        RLS2[ii,:,:] = 1 / (8 * np.pi)*((np.conj(LSF[0,1,ii,:,:])*LSF[0,1,ii,:,:]+np.conj(LSF[1,1,ii,:,:])*LSF[1,1,ii,:,:]))
        RLS3[ii,:,:] = 1 / (8 * np.pi)*((np.conj(LSF[0,2,ii,:,:])*LSF[0,2,ii,:,:]+np.conj(LSF[1,2,ii,:,:])*LSF[1,2,ii,:,:]))
        RLS12[ii,:,:] = 1 / (8 * np.pi)*((np.conj(LSF[0,0,ii,:,:])*LSF[0,1,ii,:,:]+np.conj(LSF[1,0,ii,:,:])*LSF[1,1,ii,:,:]))
        RLS13[ii,:,:] = 1 / (8 * np.pi)*((np.conj(LSF[0,0,ii,:,:])*LSF[0,2,ii,:,:]+np.conj(LSF[1,0,ii,:,:])*LSF[1,2,ii,:,:]))
        RLS23[ii,:,:] = 1 / (8 * np.pi)*((np.conj(LSF[0,1,ii,:,:])*LSF[0,2,ii,:,:]+np.conj(LSF[1,1,ii,:,:])*LSF[1,2,ii,:,:]))
    
    RLS21 = np.conj(RLS12)
    RLS31 = np.conj(RLS13)
    RLS32 = np.conj(RLS23)
    
    Tj_response = np.array([[RLS1,RLS12,RLS13],[RLS21,RLS2,RLS23],[RLS31,RLS32,RLS3]])
    return Tj_response

import numpy as np
#from geometry import geometry
def TQfundamental_noise_spectrum(freqs, Np=1e-24, Na=1e-30):
    # 定义天琴的位置噪声和加速度噪声
    L = np.sqrt(3)*1e8
    Sp = freqs*Np/(4*L**2*freqs)
    Sa = (1+1e-4/freqs)*Na/((2*np.pi*freqs)**4*4*L**2)
    return Sp ,Sa

def fundamental_noise_spectrum(freqs, Np=2.25e-22, Na=9e-30):
    L = 2.5e9
    Sp = (Np/(4*L**2))*(1 + (2e-3/freqs)**4)
    Sa = (Na/(4*L**2))*(1 + (4e-4/freqs)**2)*(1 + (freqs/8e-3)**4)*(1.0/(2*np.pi*freqs)**4)

    return Sp,Sa


def LSaet_noise_spectrum(freqs, L, Np=2.25e-22, Na=9e-30):
    Sp = Np*(1 + (2e-3/freqs)**4)
    Sa = Na*(1 + (4e-4/freqs)**2)*(1 + (freqs/8e-3)**4)
    c = 3e8
    u = 2*np.pi*freqs*L/c
    CAA = 2*np.sin(u)**2/(L**2)*((np.cos(u)+2)*Sp+2*(np.cos(2*u)+2*np.cos(u)+3)*Sa/(2*np.pi*freqs)**4)
    CEE = CAA
    CTT = 8*np.sin(u)**2*np.sin(u/2)**2/(L**2)*(Sp + 4*np.sin(u/2)**2*Sa/(2*np.pi*freqs)**4)
    C_aet = np.array([CAA, CEE, CTT])
    return C_aet

def LSxyz_noise_spectrum(freqs,f0, Np=2.25e-22, Na=9e-30):
    C_mich = LSmich_noise_spectrum(freqs, f0, Np, Na)

    ## Noise spectra of the X, Y and Z channels
    #SX = 4*SM1* np.sin(2*f0)**2

    C_xyz =  4 * np.sin(2*f0)**2 * C_mich

    return C_xyz

def LSmich_noise_spectrum(freqs,f0, Np=2.25e-22, Na=9e-30):
    # Get Sp and Sa
    Sp, Sa = fundamental_noise_spectrum(freqs, Np, Na)
    ## Noise spectra of the michelson channels
    S_auto  = 4.0 * (2.0 * Sa * (1.0 + (np.cos(2*f0))**2)  + Sp)
    S_cross =  (-2 * Sp - 8 * Sa) * np.cos(2*f0)

    C_mich = np.array([[S_auto, S_cross, S_cross], [S_cross, S_auto, S_cross], [S_cross, S_cross, S_auto]])

    return C_mich
def TQmich_noise_spectrum(freqs, f0, Np=1e-24, Na=1e-30):

    # freqs:一个频率数组
    # f0: f/fstar
    # Np 位置噪声
    # Na 加速度噪声

    # 获取位置噪声和加速度噪声Sp Sa
    Sp, Sa = TQfundamental_noise_spectrum(freqs,Np,Na)
    # 天琴臂长：sqrt(3)*1e8  单位：m
    #L = np.sqrt(3)*1e8
    # 自相关
    S_auto = 4 * (2.0 * Sa * (1.0 + (np.cos(2 * f0)) ** 2) + Sp)
    # 互相关
    S_cross = (-2 * Sp - 8 * Sa) * np.cos(2 * f0)
    C_mich = np.array([[S_auto, S_cross, S_cross], [S_cross, S_auto, S_cross], [S_cross, S_cross, S_auto]])

    return C_mich
def TQxyz_noise_spectrum(freqs,f0, Np=1e-24, Na=1e-30):
    C_mich = TQmich_noise_spectrum(freqs, f0, Np, Na)

    ## Noise spectra of the X, Y and Z channels
    #SX = 4*SM1* np.sin(2*f0)**2

    C_xyz =  4 * np.sin(2*f0)**2 * C_mich

    return C_xyz


def TQaet_noise_spectrum(freqs, L, Np=2.25e-22, Na=9e-30):
    Sp = freqs*Np/(freqs)
    Sa = (1+1e-4/freqs)*Na
    c = 3e8
    u = 2*np.pi*freqs*L/c
    CAA = 2*np.sin(u)**2/(L**2)*((np.cos(u)+2)*Sp+2*(np.cos(2*u)+2*np.cos(u)+3)*Sa/(2*np.pi*freqs)**4)
    CEE = CAA
    CTT = 8*np.sin(u)**2*np.sin(u/2)**2/(L**2)*(Sp + 4*np.sin(u/2)**2*Sa/(2*np.pi*freqs)**4)
    C_aet = np.array([CAA, CEE, CTT])
    return C_aet

def Tjfundamental_noise_spectrum(freqs, Np=6.4e-23, Na=9e-30):
    L = 3e9
    Sp = (Np/(4*L**2))*(1 + (2e-3/freqs)**4)
    Sa = (Na/(4*L**2))*(1 + (4e-4/freqs)**2)*(1 + (freqs/8e-3)**4)*(1.0/(2*np.pi*freqs)**4)

    return Sp,Sa

def Tjaet_noise_spectrum(freqs, L, Np=2.25e-22, Na=9e-30):
    Sp = Np*(1 + (2e-3/freqs)**4)
    Sa = Na*(1 + (4e-4/freqs)**2)*(1 + (freqs/8e-3)**4)
    c = 3e8
    u = 2*np.pi*freqs*L/c
    CAA = 2*np.sin(u)**2/(L**2)*((np.cos(u)+2)*Sp+2*(np.cos(2*u)+2*np.cos(u)+3)*Sa/(2*np.pi*freqs)**4)
    CEE = CAA
    CTT = 8*np.sin(u)**2*np.sin(u/2)**2/(L**2)*(Sp + 4*np.sin(u/2)**2*Sa/(2*np.pi*freqs)**4)
    C_aet = np.array([CAA, CEE, CTT])
    return C_aet

def Tjxyz_noise_spectrum(freqs,f0, Np=6.4e-23, Na=9e-30):
    C_mich = LSmich_noise_spectrum(freqs, f0, Np, Na)

    ## Noise spectra of the X, Y and Z channels
    #SX = 4*SM1* np.sin(2*f0)**2

    C_xyz =  4 * np.sin(2*f0)**2 * C_mich

    return C_xyz

def Tjmich_noise_spectrum(freqs,f0, Np=6.4e-23, Na=9e-30):
    # Get Sp and Sa
    Sp, Sa = fundamental_noise_spectrum(freqs, Np, Na)
    ## Noise spectra of the michelson channels
    S_auto  = 4.0 * (2.0 * Sa * (1.0 + (np.cos(2*f0))**2)  + Sp)
    S_cross =  (-2 * Sp - 8 * Sa) * np.cos(2*f0)

    C_mich = np.array([[S_auto, S_cross, S_cross], [S_cross, S_auto, S_cross], [S_cross, S_cross, S_auto]])

    return C_mich
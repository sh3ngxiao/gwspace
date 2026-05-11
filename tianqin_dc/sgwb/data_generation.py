from multiprocessing import Pool
from signal import signal
import numpy as np
import ORF
import noise
from healpy import Alm
import healpy as hp
from Clebsch_Gardan import frequency_noise_from_psd
from Clebsch_Gardan import Clebsch_Gardan
from scipy.special import lpmn, sph_harm
from functools import partial
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
c = 3e8
#fmax = 1 #Hz
fmin = 0.001 #HZ
H0 = 2.2*10**(-18)
frange1 = np.linspace(1/3600,0.1,360)# 频率点
delta_f = frange1[1]-frange1[0]
##t = 4380
#tsegmid_TQ = np.zeros(87) # 时间：秒 一个周期以3600s切一段，一共87段
#for i in range(87):
#    tsegmid_TQ[i] = 3600*i+2190*3600*(i//2190)
params = {}
params['nside_in'] = 6 # 像素
params['nside'] = 6
params['TQarmlength'] = np.sqrt(3)*1e8 #单位：m
params['LSarmlength'] = 3e9 #单位：m
params['Tjarmlength'] = 2.5e9
params['cspeed'] = 3e8 # 光速
params['H0'] = 2.2*10**(-18)
params['response type'] = 'aet'
params['lmax'] = 2 #注入信号的球谐函数l阶数
#point alpha = 2/3
#params['omega0'] = 1e-9
#point alpha = 3
#params['omega0'] = 6e-14
params['omega0'] = 6e-14/20
#Gau alpha = 3
#params['omega0'] = 7e-15
params['alpha'] = float(2/3)# pow-law指数
params['Np'] = 1e-24
params['Na'] = 1e-30
params['gen_Plm'] = 'point'
params['det'] = 'TQ_LS_Tj'
# 扣除0点对应频率点
TQ_aet_noise = noise.TQaet_noise_spectrum(frange1,params['TQarmlength'], Np=1e-24, Na=1e-30)
LS_aet_noise = noise.LSaet_noise_spectrum(frange1,params['LSarmlength'], Np=6.4e-23, Na=9e-30)
frange2 = frange1[LS_aet_noise[0][0] >= 1e-48]
Tj_aet_noise = noise.Tjaet_noise_spectrum(frange2,params['Tjarmlength'], Np=2.25e-22, Na=9e-30)
frange = frange2[Tj_aet_noise[0][0] >= 1e-48]
TQ_aet_noise = noise.TQaet_noise_spectrum(frange,params['TQarmlength'], Np=1e-24, Na=1e-30)
LS_aet_noise = noise.LSaet_noise_spectrum(frange,params['LSarmlength'], Np=6.4e-23, Na=9e-30)
Tj_aet_noise = noise.Tjaet_noise_spectrum(frange,params['Tjarmlength'], Np=2.25e-22, Na=9e-30)
tsegmid_TQ = 4380
tsegmid = np.zeros(1)
print(frange.shape)
def inject_point(params):
    clebsch_Gardan = Clebsch_Gardan(params)
    alm_size1 = Alm.getsize(params['lmax'])
#print(alm_size1)
# 注入一个phi = pi/3 theta = pi/2 的信号
    Ylms = np.zeros(alm_size1, dtype='complex')
    phi = np.pi/6
    theta = np.pi/3
#每个像素点的面积
    dOmega = hp.pixelfunc.nside2pixarea(params['nside_in'])
    for ii in range(alm_size1):
        lval, mval = ORF.idxtoalm(params['lmax'], ii)
        Ylms[ii] = sph_harm(mval, lval, phi, theta)

#展示注入的信号，发现有些地方的数值为负数
    skymap_inj = hp.alm2map(Ylms, params['nside_in'])
#进行非负化处理
    blm_vals = Ylms # 注入信号的方位信息
#print(Ylms.size)
    num_blms = int(Ylms.size)
    blms = np.zeros(num_blms, dtype='complex')
    for ii in range(num_blms):
        blms[ii] = complex(blm_vals[ii])
    params['blms'] = blms
#print(params['blms'].size)
    alms_inj = clebsch_Gardan.blm_2_alm(blms)
    alms_inj = alms_inj/(alms_inj[0] * np.sqrt(4*np.pi))
## extrct only the non-negative components
    alms_non_neg = alms_inj[0:hp.Alm.getsize(clebsch_Gardan.almax)]
#print(alms_non_neg.size)
#非负化结果 
    skymap_inj = hp.alm2map(alms_non_neg, params['nside_in'])
    np.save(file = "Inject_Plm",arr = skymap_inj*dOmega)
    return
#norm = Normalize(vmin=0, vmax=0.8)
#hp.graticule(color = "gray")
def inject_point_random():
    #lmax = 24
    dOmega = hp.pixelfunc.nside2pixarea(params['nside_in'])
    lmax = 2*params['lmax']    # 最大多极子（需 >= 20）
    # 创建角功率谱数组（ℓ=0到ℓ=20有非零值）
    ell = np.arange(lmax + 1)
    cl = np.zeros_like(ell, dtype=float)
    # 设置ℓ≤20的功率（示例：幂律衰减）
    cl[:lmax+1] = ell[:lmax+1] + 1 # C_ℓ = 1/(ℓ+1) for ℓ ≤ 20
    cl[0] = 0  # ℓ=0通常设为0（单极子去除）
    np.random.seed(1234)
    alm = hp.synalm(cl, lmax=lmax)  # 生成符合C_ℓ的随机高斯场
    skymap_inj = abs(hp.alm2map(alm, nside=params['nside_in']))
    np.save(file = "Inject_Plm",arr = skymap_inj/np.sum(skymap_inj*dOmega))
    return 
def single_pix_inj(params):
    npix = hp.nside2npix(params['nside'])
    map = np.zeros(npix)
    map[150] = 0.25
    map[250] = 0.25
    map[300] = 0.25
    map[350] = 0.25
    np.save(file = "Inject_Plm",arr = map)
    return
if params['gen_Plm'] == 'point':
    inject_point(params)
elif params['gen_Plm'] == 'Gau':
    inject_point_random()
elif params['gen_Plm'] == 'single_pix':
    single_pix_inj(params)


#加载方位参数
dOmega = hp.pixelfunc.nside2pixarea(params['nside_in'])
skymap_inj = np.load(file='Inject_Plm.npy')

#生成强度参数
#生成数据 每个i是一个tsegmid
T = 3.64*24*360000
npix = hp.nside2npix(params['nside_in'])
Omegaf = params['omega0'] * (frange/1e-4)**(params['alpha'])
Sgw = Omegaf*(3/(4*frange**3))*(params['H0']/np.pi)**2
#t = 4380
# data_generation
def TQ_LS_for_i(i, frange, params, skymap_inj, Sgw, T, npix):
    #tsegmid = np.zero(1)
    tsegmid[0] = 3600 * i + 2190 * 3600 * (i // 2190)
    TQ_LS_response = ORF.TQ_LS_response_pix(frange, tsegmid, params)
    #print(TQ_LS_response.shape)
    Sgw_Gu = frequency_noise_from_psd(Sgw, 1 / T, seed=2382 + i)
    Sgw_Gaussian = Sgw_Gu * Sgw_Gu.conj() / T
    direction_in_isot = (4 * np.pi) * np.sum(skymap_inj[None, None, None, None, :] * TQ_LS_response, axis=(3, 4)) / npix
    signal_f = direction_in_isot[:, :, :] * Sgw_Gaussian[None, None, :]
    signal_f_PSD = direction_in_isot[:, :, :] * Sgw[None, None, :]
    noise_A_TQ = frequency_noise_from_psd(TQ_aet_noise[0][0], 1/T,seed = 233+100*i)
    noise_E_TQ = frequency_noise_from_psd(TQ_aet_noise[1][1], 1/T,seed = 853+1000*i)
    noise_T_TQ = frequency_noise_from_psd(TQ_aet_noise[2][2], 1/T,seed = 281+230001*i)
    noise_A_LS = frequency_noise_from_psd(LS_aet_noise[0][0], 1/T,seed = 213+1001*i)
    noise_E_LS = frequency_noise_from_psd(LS_aet_noise[1][1], 1/T,seed = 5333+10000*i)
    noise_T_LS = frequency_noise_from_psd(LS_aet_noise[2][2], 1/T,seed = 14+200001*i)
    noise_f1 = (1/(T))*np.array([[noise_A_TQ*noise_A_LS.conj(),noise_A_TQ*noise_E_LS.conj(),noise_A_TQ*noise_T_LS.conj()],[noise_E_TQ*noise_A_LS.conj(),noise_E_TQ*noise_E_LS.conj(), noise_E_TQ*noise_T_LS.conj()],[noise_T_TQ*noise_A_LS.conj(),noise_T_TQ*noise_E_LS.conj(),noise_T_TQ*noise_T_LS.conj()]])
    data_f1 = noise_f1+signal_f
    return tsegmid, signal_f, signal_f_PSD , noise_f1 , data_f1

def TQ_Tj_for_i(i, frange, params, skymap_inj, Sgw, T, npix):
    #tsegmid = np.zero(1)
    tsegmid[0] = 3600 * i + 2190 * 3600 * (i // 2190)
    TQ_Tj_response = ORF.TQ_Tj_response_pix(frange, tsegmid, params)
    #print(TQ_LS_response.shape)
    Sgw_Gu = frequency_noise_from_psd(Sgw, 1 / T, seed=2527 + i)
    Sgw_Gaussian = Sgw_Gu * Sgw_Gu.conj() / T
    direction_in_isot = (4 * np.pi) * np.sum(skymap_inj[None, None, None, None, :] * TQ_Tj_response, axis=(3, 4)) / npix
    signal_f = direction_in_isot[:, :, :] * Sgw_Gaussian[None, None, :]
    signal_f_PSD = direction_in_isot[:, :, :] * Sgw[None, None, :] 
    noise_A_TQ = frequency_noise_from_psd(TQ_aet_noise[0][0], 1/T,seed = 312+100*i)
    noise_E_TQ = frequency_noise_from_psd(TQ_aet_noise[1][1], 1/T,seed = 2793+1000*i)
    noise_T_TQ = frequency_noise_from_psd(TQ_aet_noise[2][2], 1/T,seed = 731+230001*i)
    noise_A_LS = frequency_noise_from_psd(Tj_aet_noise[0][0], 1/T,seed = 35213+1001*i)
    noise_E_LS = frequency_noise_from_psd(Tj_aet_noise[1][1], 1/T,seed = 2373+10000*i)
    noise_T_LS = frequency_noise_from_psd(Tj_aet_noise[2][2], 1/T,seed = 1974+200001*i)
    noise_f1 = (1/(T))*np.array([[noise_A_TQ*noise_A_LS.conj(),noise_A_TQ*noise_E_LS.conj(),noise_A_TQ*noise_T_LS.conj()],[noise_E_TQ*noise_A_LS.conj(),noise_E_TQ*noise_E_LS.conj(), noise_E_TQ*noise_T_LS.conj()],[noise_T_TQ*noise_A_LS.conj(),noise_T_TQ*noise_E_LS.conj(),noise_T_TQ*noise_T_LS.conj()]])
    data_f1 = noise_f1+signal_f
    return tsegmid, signal_f, signal_f_PSD , noise_f1 , data_f1

def Tj_LS_for_i(i, frange, params, skymap_inj, Sgw, T, npix):
    #tsegmid = np.zero(1)
    tsegmid[0] = 3600 * i
    LS_Tj_response = ORF.LS_Tj_response_pix(frange, tsegmid, params)
    #print(TQ_LS_response.shape)
    Sgw_Gu = frequency_noise_from_psd(Sgw, 1 / T, seed=955 + i)
    Sgw_Gaussian = Sgw_Gu * Sgw_Gu.conj() / T
    direction_in_isot = (4 * np.pi) * np.sum(skymap_inj[None, None, None, None, :] * LS_Tj_response, axis=(3, 4)) / npix
    signal_f = direction_in_isot[:, :, :] * Sgw_Gaussian[None, None, :]
    signal_f_PSD = direction_in_isot[:, :, :] * Sgw[None, None, :] 
    noise_A_TQ = frequency_noise_from_psd(Tj_aet_noise[0][0], 1/T,seed = 8163+100*i)
    noise_E_TQ = frequency_noise_from_psd(Tj_aet_noise[1][1], 1/T,seed = 5824+1000*i)
    noise_T_TQ = frequency_noise_from_psd(Tj_aet_noise[2][2], 1/T,seed = 3583+230001*i)
    noise_A_LS = frequency_noise_from_psd(LS_aet_noise[0][0], 1/T,seed = 5532413+1001*i)
    noise_E_LS = frequency_noise_from_psd(LS_aet_noise[1][1], 1/T,seed = 4213+10000*i)
    noise_T_LS = frequency_noise_from_psd(LS_aet_noise[2][2], 1/T,seed = 521+200001*i)
    noise_f1 = (1/(T))*np.array([[noise_A_TQ*noise_A_LS.conj(),noise_A_TQ*noise_E_LS.conj(),noise_A_TQ*noise_T_LS.conj()],[noise_E_TQ*noise_A_LS.conj(),noise_E_TQ*noise_E_LS.conj(), noise_E_TQ*noise_T_LS.conj()],[noise_T_TQ*noise_A_LS.conj(),noise_T_TQ*noise_E_LS.conj(),noise_T_TQ*noise_T_LS.conj()]])
    data_f1 = noise_f1+signal_f
    return tsegmid, signal_f, signal_f_PSD , noise_f1 , data_f1

if __name__ == '__main__':
    # 保证以下变量在主线程里都已定义：
    # frange, params, skymap_inj, Sgw, T, npix
    current_dir = Path(__file__).parent
    output_dir = current_dir / "save_data"
    output_dir.mkdir(exist_ok=True)

    with Pool(20) as pool:
        if params['det'] == 'TQ_LS':
            signal_gen_TQ_LS = partial(TQ_LS_for_i, frange=frange, params=params, skymap_inj=skymap_inj, Sgw=Sgw, T=T, npix=npix)
        #noise_gen = partial(noise_generation, frange = frange, T = T)
            signal_gen_TQ_LS_results = pool.map(signal_gen_TQ_LS, range(tsegmid_TQ))
        elif params['det'] == 'TQ_Tj':
            signal_gen_TQ_Tj = partial(TQ_Tj_for_i, frange=frange, params=params, skymap_inj=skymap_inj, Sgw=Sgw, T=T, npix=npix)
        #noise_gen = partial(noise_generation, frange = frange, T = T)
            signal_gen_TQ_Tj_results = pool.map(signal_gen_TQ_Tj, range(tsegmid_TQ))
        elif params['det'] == 'Tj_LS':
            signal_gen_Tj_LS = partial(Tj_LS_for_i, frange=frange, params=params, skymap_inj=skymap_inj, Sgw=Sgw, T=T, npix=npix)
        #noise_gen = partial(noise_generation, frange = frange, T = T)
            signal_gen_Tj_LS_results = pool.map(signal_gen_Tj_LS, range(2*tsegmid_TQ))
        elif params['det'] == 'TQ_LS_Tj':
            signal_gen_TQ_LS = partial(TQ_LS_for_i, frange=frange, params=params, skymap_inj=skymap_inj, Sgw=Sgw, T=T, npix=npix)
            signal_gen_TQ_LS_results = pool.map(signal_gen_TQ_LS, range(tsegmid_TQ))
            signal_gen_TQ_Tj = partial(TQ_Tj_for_i, frange=frange, params=params, skymap_inj=skymap_inj, Sgw=Sgw, T=T, npix=npix)
            signal_gen_TQ_Tj_results = pool.map(signal_gen_TQ_Tj, range(tsegmid_TQ))
            signal_gen_Tj_LS = partial(Tj_LS_for_i, frange=frange, params=params, skymap_inj=skymap_inj, Sgw=Sgw, T=T, npix=npix)
            signal_gen_Tj_LS_results = pool.map(signal_gen_Tj_LS, range(2*tsegmid_TQ))
        pool.close()
        pool.join()


# SNR calculation
    def SNR_TQ_LS(signal_gen_TQ_LS_results,output_dir):
        signal_f_PSD = [r[2] for r in signal_gen_TQ_LS_results]
        SNR = 0
        for i in range(len(signal_f_PSD[:])):
            SNR = SNR+ 2*3600*np.sum((abs(signal_f_PSD[i][0,0,:])**2)/(TQ_aet_noise[0,0,:]*LS_aet_noise[0,0,:])*(1/3600))
            #SNR = SNR+ np.sqrt(2*3600*3.64*24*np.sum((abs(signal_f_PSD[i][0,0,:])**2)/(TQ_aet_noise[0,0,:]*LS_aet_noise[0,0,:]+abs(signal_f_PSD[i][0,0,:])**2))*(1/3600))
    #SNR = np.sqrt(2*3600*3.64*24*np.sum((abs(signal_f_PSD[0:86][0,0,:])**2)/(TQ_aet_noise[None][0,0,:]*LS_aet_noise[None][0,0,:]+abs(signal_f_PSD[0:86][0,0,:])**2))*(1/87)*(1/3600),axis = (0,1))
        print("TQ+LS SNR:", np.sqrt(SNR))
        np.save(output_dir / "TQ+LS_SNR", np.sqrt(SNR))
        return

    def SNR_TQ_Tj(signal_gen_TQ_Tj_results,output_dir):
        signal_f_PSD = [r[2] for r in signal_gen_TQ_Tj_results]
        SNR = 0
        for i in range(len(signal_f_PSD[:])):
            SNR = SNR + 2*3600*np.sum((abs(signal_f_PSD[i][0,0,:])**2)/(TQ_aet_noise[0,0,:]*Tj_aet_noise[0,0,:])*(1/3600))
            #SNR = SNR+ np.sqrt(2*3600*3.64*24*np.sum((abs(signal_f_PSD[i][0,0,:])**2)/(TQ_aet_noise[0,0,:]*Tj_aet_noise[0,0,:]+abs(signal_f_PSD[i][0,0,:])**2))*(1/3600))
    #SNR = np.sqrt(2*3600*3.64*24*np.sum((abs(signal_f_PSD[0:86][0,0,:])**2)/(TQ_aet_noise[None][0,0,:]*LS_aet_noise[None][0,0,:]+abs(signal_f_PSD[0:86][0,0,:])**2))*(1/87)*(1/3600),axis = (0,1))
        print("TQ+Tj SNR:", np.sqrt(SNR) )
        np.save(output_dir / "TQ+Tj_SNR", np.sqrt(SNR))
        return

    def SNR_Tj_LS(signal_gen_Tj_LS_results,output_dir):
        signal_f_PSD = [r[2] for r in signal_gen_Tj_LS_results]
        SNR = 0
        for i in range(len(signal_f_PSD[:])):
            SNR = SNR+ 2*3600*np.sum((abs(signal_f_PSD[i][0,0,:])**2)/(Tj_aet_noise[0,0,:]*LS_aet_noise[0,0,:])*(1/3600))
            #SNR = SNR+ np.sqrt(2*3600*3.64*24*np.sum((abs(signal_f_PSD[i][0,0,:])**2)/(Tj_aet_noise[0,0,:]*LS_aet_noise[0,0,:]+abs(signal_f_PSD[i][0,0,:])**2))*(1/3600))
            #SNR = np.sqrt(2*3600*3.64*24*np.sum((abs(signal_f_PSD[0:86][0,0,:])**2)/(TQ_aet_noise[None][0,0,:]*LS_aet_noise[None][0,0,:]+abs(signal_f_PSD[0:86][0,0,:])**2))*(1/87)*(1/3600),axis = (0,1))
        print("Tj+LS SNR:", np.sqrt(SNR))
        np.save(output_dir / "Tj+LS_SNR", np.sqrt(SNR))
        return
    #print(signal_f_PSD[86].shape)
#Save the data 
    if params['det'] == 'TQ_LS':
        SNR_TQ_LS(signal_gen_TQ_LS_results,output_dir)
        with open("signal_gen_TQ_LS_results.pkl", "wb") as f:
            pickle.dump(signal_gen_TQ_LS_results, f)
    elif params['det'] == 'TQ_Tj':
        SNR_TQ_Tj(signal_gen_TQ_Tj_results,output_dir)
        with open("signal_gen_TQ_Tj_results.pkl", "wb") as f:
            pickle.dump(signal_gen_TQ_Tj_results, f)
    elif params['det'] == 'Tj_LS':
        SNR_Tj_LS(signal_gen_Tj_LS_results,output_dir)
        with open("signal_gen_Tj_LS_results.pkl", "wb") as f:
            pickle.dump(signal_gen_Tj_LS_results, f)
    elif params['det'] == 'TQ_LS_Tj':
        SNR_TQ_LS(signal_gen_TQ_LS_results,output_dir)
        with open("signal_gen_TQ_LS_results.pkl", "wb") as f:
            pickle.dump(signal_gen_TQ_LS_results, f)
        SNR_TQ_Tj(signal_gen_TQ_Tj_results,output_dir)
        with open("signal_gen_TQ_Tj_results.pkl", "wb") as f:
            pickle.dump(signal_gen_TQ_Tj_results, f)
        SNR_Tj_LS(signal_gen_Tj_LS_results,output_dir)
        with open("signal_gen_Tj_LS_results.pkl", "wb") as f:
            pickle.dump(signal_gen_Tj_LS_results, f)

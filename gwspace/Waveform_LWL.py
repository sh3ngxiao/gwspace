from gwspace.Orbit import detectors

def XYZ_to_AET(X, Y, Z):
    A = (2*X - Y - Z) / 3.0
    E = (Z - Y) / np.sqrt(3.0)
    T = (X + Y + Z) / 3.0
    return A, E, T

def _michelson_from_arms(hij, u, v):
    return 0.5 * (
        np.einsum("it,jt,ijt->t", u, u, hij) -
        np.einsum("it,jt,ijt->t", v, v, hij)
    )

def waveform(para_value, Tobs, frequency=None, eps=1e-5, modes=None):
    # 写回 EMRIWaveform 参数
    for name, val in zip(PARAMS, para_value):
        setattr(wf, name, val)

    # 时域采样
    t = np.arange(0, Tobs, dt)

    # 1) 源两偏振（不做TDI）
    hp, hc = wf.get_hphc(t, eps=eps, modes=modes)

    # 2) 极化张量
    p_plus, p_cross = wf.polarization()

    # 3) 组装 h_ij(t)
    hij = (p_plus[:, :, None] * hp[None, None, :] +
           p_cross[:, :, None] * hc[None, None, :])

    # 4) 天琴臂方向（只用 uni_vec_ij）
    det = detectors["TQ"](t)

    # 下面假设 uni_vec_ij 返回 (Nt,3)，所以要 .T 变成 (3,Nt)
    u12 = det.uni_vec_ij(1, 2).T
    u13 = det.uni_vec_ij(1, 3).T
    u21 = det.uni_vec_ij(2, 1).T
    u23 = det.uni_vec_ij(2, 3).T
    u31 = det.uni_vec_ij(3, 1).T
    u32 = det.uni_vec_ij(3, 2).T

    # 5) 三个 Michelson：X(在1)、Y(在2)、Z(在3)
    X = _michelson_from_arms(hij, u12, u13)
    Y = _michelson_from_arms(hij, u23, u21)
    Z = _michelson_from_arms(hij, u31, u32)

    # 6) A/E/T
    A, E, T = XYZ_to_AET(X, Y, Z)

    # 7) FFT
    N = len(t)
    f_fft = np.fft.rfftfreq(N, dt)
    A_f = np.fft.rfft(A) * dt
    E_f = np.fft.rfft(E) * dt
    T_f = np.fft.rfft(T) * dt

    if frequency is None:
        f = f_fft
        Aout, Eout, Tout = A_f, E_f, T_f
    else:
        f = np.asarray(frequency)
        if f[0] < f_fft[0] or f[-1] > f_fft[-1]:
            raise ValueError("frequency 超出 FFT 频率范围，请缩小频率范围或减小 dt / 增大 Tobs。")

        def interp_complex(y):
            return np.interp(f, f_fft, y.real) + 1j*np.interp(f, f_fft, y.imag)

        Aout, Eout, Tout = interp_complex(A_f), interp_complex(E_f), interp_complex(T_f)

    idx = np.arange(len(f))
    return f, Aout, Eout, Tout, idx




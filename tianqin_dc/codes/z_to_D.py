from scipy import integrate
from scipy.constants import c

omega_lambda = 0.689
omega_m = 0.311
omega_gama_and_mu = 2.47e-5 * pow(0.677, -2) + 0.1 * pow(93.12, -1) * pow(0.677, -2)
H_0 = 67.7  # (km/s)/Mpc

def z_to_Dl(p_z):
    y = lambda x: pow(pow(omega_lambda + omega_m * pow(1 + x, 3) + omega_gama_and_mu * pow(1 + x, 4), 0.5), -1)
    g = integrate.quad(y, 0, p_z)
    # return g[0] * (1 + p_z) / H_0  # c = 1
    # return g[0] * (1 + p_z) * c / H_0 / 1e6  # c/H_0--->m/s / ((km/s)/Mpc)= Mpc/1000=kpc-----> /1e6=>Gpc
    return g[0] * (1 + p_z) * c / H_0  # c/H_0--->m/s / ((km/s)/Mpc)= Mpc/1000=kpc

def z_to_Dz(p_z):
    y = lambda x: pow(pow(omega_lambda + omega_m * pow(1 + x, 3) + omega_gama_and_mu * pow(1 + x, 4), 0.5), -1)
    g = integrate.quad(y, 0, p_z)
    # return g[0] / H_0  # c = 1
    # return g[0] * c / H_0 / 1e6
    return g[0] * c / H_0

if __name__ == "__main__":
    z=0.01
    print("z =",z,"--> Dz =",z_to_Dz(z)/1e6,"Gpc")
    print("z =",z,"--> Dl =",z_to_Dl(z)/1e6,"Gpc")
    print("z =",z,"--> Dz =",z_to_Dz(z)/1e3,"Mpc")
    print("z =",z,"--> Dl =",z_to_Dl(z)/1e3,"Mpc")
    print("z =",z,"--> Dz =",z_to_Dz(z),"kpc")
    print("z =",z,"--> Dl =",z_to_Dl(z),"kpc")

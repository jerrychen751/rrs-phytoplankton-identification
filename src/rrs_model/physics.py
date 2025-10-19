import numpy as np

def betasw_ZHH2009(lambda_val, Tc, S, theta, delta=0.039):
    """
    This function is a Python translation of the MATLAB script by Xiaodong Zhang,
    Lianbo Hu, and Ming-Xia He (2009) from their paper "Scattering by pure seawater:
    Effect of salinity," published in Optics Express, Vol. 17, No. 7, pp. 5698-5710.

    It computes the volume scattering function, scattering coefficient, and
    backscattering coefficient of pure seawater.

    Args:
        lambda_val (np.ndarray): Wavelength in nanometers (nm).
        Tc (float): Temperature in degrees Celsius.
        S (float): Salinity in practical salinity units (psu).
        theta (np.ndarray): Scattering angles in degrees.
        delta (float, optional): Depolarization ratio. Defaults to 0.039.

    Returns:
        tuple: A tuple containing:
            - betasw (np.ndarray): Volume scattering at the specified angles.
            - beta90sw (np.ndarray): Volume scattering at 90 degrees.
            - bsw (np.ndarray): Total scattering coefficient.
    """
    if not isinstance(Tc, (int, float)) or not isinstance(S, (int, float)):
        raise ValueError("Temperature (Tc) and Salinity (S) must be scalar values.")

    lambda_val = np.array(lambda_val, dtype=float).flatten()
    rad = np.array(theta, dtype=float).flatten() * np.pi / 180

    Na = 6.0221417930e23
    Kbz = 1.3806503e-23
    Tk = Tc + 273.15
    M0 = 18e-3

    nsw, dnds = _RInw(lambda_val, Tc, S)
    IsoComp = _BetaT(Tc, S)
    density_sw = _rhou_sw(Tc, S)
    dlnawds = _dlnasw_ds(Tc, S)
    DFRI = _PMH(nsw)

    beta_df = (np.pi**2 / 2) * ((lambda_val * 1e-9)**(-4)) * Kbz * Tk * IsoComp * (DFRI**2) * (6 + 6 * delta) / (6 - 7 * delta)
    flu_con = S * M0 * dnds**2 / density_sw / (-dlnawds) / Na
    beta_cf = 2 * np.pi**2 * ((lambda_val * 1e-9)**(-4)) * nsw**2 * flu_con * (6 + 6 * delta) / (6 - 7 * delta)

    beta90sw = beta_df + beta_cf
    bsw = 8 * np.pi / 3 * beta90sw * (2 + delta) / (1 + delta)

    betasw = np.outer(1 + (np.cos(rad)**2) * (1 - delta) / (1 + delta), beta90sw)

    return betasw, beta90sw, bsw

def _RInw(lambda_val, Tc, S):
    """Calculates the refractive index of seawater."""
    n_air = 1.0 + (5792105.0 / (238.0185 - 1. / (lambda_val / 1e3)**2) + 167917.0 / (57.362 - 1. / (lambda_val / 1e3)**2)) / 1e8

    n0 = 1.31405
    n1 = 1.779e-4
    n2 = -1.05e-6
    n3 = 1.6e-8
    n4 = -2.02e-6
    n5 = 15.868
    n6 = 0.01155
    n7 = -0.00423
    n8 = -4382
    n9 = 1.1455e6

    nsw = n0 + (n1 + n2 * Tc + n3 * Tc**2) * S + n4 * Tc**2 + (n5 + n6 * S + n7 * Tc) / lambda_val + n8 / lambda_val**2 + n9 / lambda_val**3
    nsw *= n_air
    dnswds = (n1 + n2 * Tc + n3 * Tc**2 + n6 / lambda_val) * n_air
    return nsw, dnswds

def _BetaT(Tc, S):
    """Calculates isothermal compressibility."""
    kw = 19652.21 + 148.4206 * Tc - 2.327105 * Tc**2 + 1.360477e-2 * Tc**3 - 5.155288e-5 * Tc**4
    a0 = 54.6746 - 0.603459 * Tc + 1.09987e-2 * Tc**2 - 6.167e-5 * Tc**3
    b0 = 7.944e-2 + 1.6483e-2 * Tc - 5.3009e-4 * Tc**2
    Ks = kw + a0 * S + b0 * S**1.5
    return 1. / Ks * 1e-5

def _rhou_sw(Tc, S):
    """Calculates the density of seawater."""
    a0 = 8.24493e-1
    a1 = -4.0899e-3
    a2 = 7.6438e-5
    a3 = -8.2467e-7
    a4 = 5.3875e-9
    a5 = -5.72466e-3
    a6 = 1.0227e-4
    a7 = -1.6546e-6
    a8 = 4.8314e-4
    b0 = 999.842594
    b1 = 6.793952e-2
    b2 = -9.09529e-3
    b3 = 1.001685e-4
    b4 = -1.120083e-6
    b5 = 6.536332e-9

    density_w = b0 + b1 * Tc + b2 * Tc**2 + b3 * Tc**3 + b4 * Tc**4 + b5 * Tc**5
    density_sw = density_w + ((a0 + a1 * Tc + a2 * Tc**2 + a3 * Tc**3 + a4 * Tc**4) * S +
                              (a5 + a6 * Tc + a7 * Tc**2) * S**1.5 + a8 * S**2)
    return density_sw

def _dlnasw_ds(Tc, S):
    """
    Calculates the partial derivative of the natural logarithm of water activity
    with respect to salinity.
    """
    dlnawds = ((-5.58651e-4 + 2.40452e-7 * Tc - 3.12165e-9 * Tc**2 + 2.40808e-11 * Tc**3) +
               1.5 * (1.79613e-5 - 9.9422e-8 * Tc + 2.08919e-9 * Tc**2 - 1.39872e-11 * Tc**3) * S**0.5 +
               2 * (-2.31065e-6 - 1.37674e-9 * Tc - 1.93316e-11 * Tc**2) * S)
    return dlnawds

def _PMH(n_wat):
    """Calculates the density derivative of the refractive index."""
    n_wat2 = n_wat**2
    return (n_wat2 - 1) * (1 + 2/3 * (n_wat2 + 2) * (n_wat / 3 - 1 / (3 * n_wat))**2)
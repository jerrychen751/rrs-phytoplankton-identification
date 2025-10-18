import numpy as np
from scipy.io import loadmat
from kramer_functions import betasw_ZHH2009, gsm_invert

def main():
    """
    This script is a Python translation of the Kramer_hyperRrs.m MATLAB script.
    It models hyperspectral remote sensing reflectance (Rrs) and calculates the
    reflectance residual between measured and modeled values.
    """
    # Load data
    print("Loading data...")
    test_data = loadmat('data/Kramer_rrs_testdata.mat')
    Rrs = test_data['Rrs']
    chl = test_data['chl']
    T = test_data['T'].flatten()
    S = test_data['S'].flatten()

    # Convert above-surface reflectance (Rrs) to below-surface (rrs)
    rrs = Rrs / (0.52 + 1.7 * Rrs)

    # Transpose for correct orientation
    rrs = rrs.T
    Rrs = Rrs.T

    wave = np.arange(400, 701)

    # Load absorption and backscattering data
    asw_data = loadmat('data/asw_all.mat')
    asw = asw_data['asw_all'][50:, 1]  # Wavelengths from 400nm onwards

    ab_coeffs = loadmat('data/AB_coeffs.mat')
    A = ab_coeffs['A'][50:, 0]
    B = ab_coeffs['B'][50:, 0]

    # Calculate acdm (absorption due to CDOM and detritus)
    Rrs490 = Rrs[90, :]
    Rrs555 = Rrs[155, :]
    acdm_s = -(0.01447 + 0.00033 * Rrs490 / Rrs555)
    acdm = np.exp(np.outer(wave - 443, acdm_s)).T

    # Calculate seawater backscattering (bbsw)
    bbsw = np.zeros((len(wave), len(T)))
    for i in range(len(T)):
        _, _, bsw_i = betasw_ZHH2009(wave, T[i], S[i], np.arange(0, 181))
        bbsw[:, i] = 0.5 * bsw_i

    # Calculate particle backscattering (bbp) slope
    rrs440 = rrs[40, :]
    rrs555 = rrs[155, :]
    bbp_s = 2.0 * (1.0 - 1.2 * np.exp(-0.9 * rrs440 / rrs555))
    bbp = (443.0 / wave[:, np.newaxis])**bbp_s

    # Invert the GSM model to get IOPs
    print("Inverting GSM model to retrieve IOPs...")
    IOPs = np.zeros((rrs.shape[1], 3))
    for i in range(rrs.shape[1]):
        IOPs[i, :] = gsm_invert(
            rrs[:, i][np.newaxis, :], asw, bbsw[:, i],
            bbp[:, i], A, B, acdm[i, :]
        )
    print("IOPs retrieved:", IOPs)

    # Reconstruct Rrs from the retrieved IOPs
    print("Reconstructing Rrs...")
    a = np.zeros_like(rrs)
    bb = np.zeros_like(rrs)
    for i in range(rrs.shape[1]):
        a[:, i] = asw + (A * IOPs[i, 0]**B) + (IOPs[i, 1] * acdm[i, :])
        bb[:, i] = bbsw[:, i] + (IOPs[i, 2] * bbp[:, i])

    rrsP = bb / (a + bb)

    g = np.array([0.0949, 0.0794])
    modrrs = (g[0] + g[1] * rrsP) * rrsP

    # Convert modeled rrs back to Rrs
    modRrs = (0.52 * modrrs) / (1 - (1.7 * modrrs))

    # Calculate the residual
    RrsD = Rrs - modRrs

    print("Rrs residual (RrsD) calculated successfully.")
    print("Shape of RrsD:", RrsD.shape)

    # To-do: Save or use the RrsD result
    # For now, just printing a sample
    print("Sample of RrsD:", RrsD[:5, :5])

if __name__ == "__main__":
    main()
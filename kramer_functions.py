import numpy as np
from scipy.io import savemat
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

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

from scipy.optimize import minimize

def _PMH(n_wat):
    """Calculates the density derivative of the refractive index."""
    n_wat2 = n_wat**2
    return (n_wat2 - 1) * (1 + 2/3 * (n_wat2 + 2) * (n_wat / 3 - 1 / (3 * n_wat))**2)

def gsm_cost(IOPs, rrs, aw, bbw, bbpstar, A, B, admstar):
    """
    This function is a Python translation of the gsm_cost.m script. It is designed
    to be called by an optimization routine (e.g., scipy.optimize.minimize) to
    calculate the cost function for the GSM (Garver, Siegel, Maritorena) semi-analytical model.

    Args:
        IOPs (list or np.ndarray): A list or array containing the Inherent Optical
                                   Properties (IOPs) to be optimized. Expected to be
                                   in the order: [chl, acdm443, bbp443].
        rrs (np.ndarray): Measured remote-sensing reflectance.
        aw (np.ndarray): Absorption coefficient of pure water.
        bbw (np.ndarray): Backscattering coefficient of pure water.
        bbpstar (np.ndarray): Normalized backscattering coefficient of particles.
        A (np.ndarray): Coefficient for chlorophyll-specific absorption.
        B (np.ndarray): Exponent for chlorophyll-specific absorption.
        admstar (np.ndarray): Normalized absorption coefficient of detritus and CDOM.

    Returns:
        float: The calculated cost, which is the sum of squared differences
               between the measured and predicted reflectance.
    """
    g = np.array([0.0949, 0.0794])

    aph = A * (IOPs[0]**B)
    a = aw + aph + (IOPs[1] * admstar)
    bb = bbw + IOPs[2] * bbpstar

    x = bb / (a + bb)
    rrspred = (g[0] + g[1] * x) * x

    cost = np.sum((rrs - rrspred)**2)
    return cost

def gsm_invert(rrs, aw, bbw, bbpstar, A, B, admstar):
    """
    This function is a Python translation of the gsm_invert.m script. It inverts a
    semi-analytical model to retrieve inherent optical properties (IOPs) from
    remote-sensing reflectance (rrs).

    Args:
        rrs (np.ndarray): A 2D array of remote-sensing reflectance, where each
                          row is a spectrum.
        aw (np.ndarray): Absorption coefficient of pure water.
        bbw (np.ndarray): Backscattering coefficient of pure water.
        bbpstar (np.ndarray): Normalized backscattering coefficient of particles.
        A (np.ndarray): Coefficient for chlorophyll-specific absorption.
        B (np.ndarray): Exponent for chlorophyll-specific absorption.
        admstar (np.ndarray): Normalized absorption of detritus and CDOM.

    Returns:
        np.ndarray: A 2D array of the retrieved IOPs for each input spectrum.
                    Each row contains [chl, acdm443, bbp443].
    """
    IOPs_out = np.full((rrs.shape[0], 3), np.nan)
    initial_guess = [0.15, 0.01, 0.0029]

    for i in range(rrs.shape[0]):
        rrs_obs = rrs[i, :]

        bounds = [(0, None), (0, None), (0, None)] # Add bounds to ensure IOPs are non-negative

        result = minimize(
            gsm_cost,
            initial_guess,
            args=(rrs_obs, aw, bbw, bbpstar, A, B, admstar),
            method='L-BFGS-B',
            bounds=bounds
        )

        if result.success:
            IOPs_out[i, :] = result.x

    return IOPs_out

def rrs_model_train(daph, pft, pft_index, n_permutations, max_pcs, k, mdl_pick_metric, output_file_name):
    """
    This function is a Python translation of the rrsModelTrain.m script. It trains a
    principal component regression model to predict a phytoplankton functional type (PFT)
    index from spectral data, using cross-validation to optimize the model.

    Args:
        daph (np.ndarray): A 2D array of spectra, where each row is a spectrum.
        pft (np.ndarray): A 1D array of the PFT index values corresponding to the spectra.
        pft_index (str): Specifies constraints on the model output.
                         Options: 'pigment', 'EOFs', 'compositions'.
        n_permutations (int): The number of cross-validation permutations to perform.
        max_pcs (int): The maximum number of principal components to use in the model.
        k (int): The number of folds for k-fold cross-validation.
        mdl_pick_metric (str): The goodness-of-fit metric for model optimization.
                               Options: 'R2', 'RMSE', 'MAE', 'bias', 'avg', 'med'.
        output_file_name (str): The name of the output .mat file to save the results.
    """
    if np.isnan(pft).any():
        raise ValueError("Input 'pft' data contains NaNs. Please remove them and retry.")
    if np.isnan(daph).any():
        raise ValueError("Input 'daph' (spectral data) contains NaNs. Please remove them and retry.")
    if pft.shape[0] != daph.shape[0]:
        raise ValueError("The number of observations in 'pft' and 'daph' must be the same.")
    if not output_file_name.endswith('.mat'):
        raise ValueError("Output file name must end with .mat")

    valid_pft_indices = ['pigment', 'EOFs', 'compositions']
    if pft_index not in valid_pft_indices:
        raise ValueError(f"Invalid 'pft_index'. Must be one of {valid_pft_indices}")

    valid_metrics = ['R2', 'RMSE', 'avg', 'med', 'ens', 'bias', 'MAE']
    if mdl_pick_metric not in valid_metrics:
        raise ValueError(f"Invalid 'mdl_pick_metric'. Must be one of {valid_metrics}")

    np.random.seed(1)

    # Initialize storage for results across permutations
    all_mean_betas_nonstd = np.zeros((daph.shape[1], n_permutations))
    all_mean_alphas_nonstd = np.zeros(n_permutations)
    final_r2s, final_rmses, final_maes, final_biases = [], [], [], []
    final_avg_pct_errors, final_med_pct_errors = [], []

    for i in range(n_permutations):
        # Split data into training (75%) and validation (25%) sets
        daph_train, daph_val, pft_train, pft_val = train_test_split(
            daph, pft, test_size=0.25, random_state=i
        )

        kf = KFold(n_splits=k, shuffle=True, random_state=i)

        # Store results for each fold
        fold_betas = np.zeros((daph.shape[1], k))
        fold_alphas = np.zeros(k)

        for j, (train_idx, val_idx) in enumerate(kf.split(daph_train)):
            cv_train_spec, cv_val_spec = daph_train[train_idx], daph_train[val_idx]
            cv_train_pft, cv_val_pft = pft_train[train_idx], pft_train[val_idx]

            scaler = StandardScaler()
            cv_train_spec_std = scaler.fit_transform(cv_train_spec)
            cv_val_spec_std = scaler.transform(cv_val_spec)

            n_samples_fold = cv_train_spec_std.shape[0]
            current_max_pcs = min(max_pcs, n_samples_fold -1)

            pca = PCA(n_components=current_max_pcs)
            train_pcs = pca.fit_transform(cv_train_spec_std)

            # Find the best number of components for this fold
            gof_metrics = {
                'R2': [], 'RMSE': [], 'MAE': [], 'bias': [],
                'avg': [], 'med': [], 'ens': []
            }

            for l in range(1, max_pcs + 1):
                model = LinearRegression()
                model.fit(train_pcs[:, :l], cv_train_pft)

                val_pcs = pca.transform(cv_val_spec_std)
                pft_pred = model.predict(val_pcs[:, :l])

                if pft_index == 'pigment':
                    pft_pred[pft_pred < 0] = 0
                elif pft_index == 'compositions':
                    pft_pred = np.clip(pft_pred, 0, 1)

                gof_metrics['R2'].append(r2_score(cv_val_pft, pft_pred))
                gof_metrics['RMSE'].append(np.sqrt(mean_squared_error(cv_val_pft, pft_pred)))
                gof_metrics['MAE'].append(np.mean(np.abs(pft_pred - cv_val_pft)))
                gof_metrics['bias'].append(np.mean(pft_pred - cv_val_pft))

                # Percentage errors (avoid division by zero)
                safe_pft_val = np.where(cv_val_pft == 0, 1e-6, cv_val_pft)
                percent_errors = np.abs((pft_pred - safe_pft_val) / safe_pft_val) * 100
                gof_metrics['avg'].append(np.mean(percent_errors))
                gof_metrics['med'].append(np.median(percent_errors))
                gof_metrics['ens'].append((1 - gof_metrics['R2'][-1] + gof_metrics['RMSE'][-1]) / 100)

            # Select the best number of components based on the chosen metric
            if mdl_pick_metric in ['RMSE', 'MAE', 'bias', 'avg', 'med', 'ens']:
                best_l = np.argmin(gof_metrics[mdl_pick_metric]) + 1
            else: # 'R2'
                best_l = np.argmax(gof_metrics[mdl_pick_metric]) + 1

            # Retrain model with the best number of components
            final_pca = PCA(n_components=best_l)
            final_train_pcs = final_pca.fit_transform(cv_train_spec_std)

            final_model = LinearRegression()
            final_model.fit(final_train_pcs, cv_train_pft)

            # Store coefficients and intercept for this fold
            fold_betas[:, j] = final_pca.inverse_transform(final_model.coef_)
            fold_alphas[j] = final_model.intercept_

        # Average coefficients and intercepts from all folds
        mean_betas = np.mean(fold_betas, axis=1)
        mean_alphas = np.mean(fold_alphas)

        # De-standardize to apply to original data
        scaler_final = StandardScaler().fit(daph_train)
        mean_betas_nonstd = mean_betas / scaler_final.scale_
        mean_alphas_nonstd = mean_alphas - np.sum(mean_betas * scaler_final.mean_ / scaler_final.scale_)

        all_mean_betas_nonstd[:, i] = mean_betas_nonstd
        all_mean_alphas_nonstd[i] = mean_alphas_nonstd

        # Validate on the 25% hold-out set
        pft_final_pred = daph_val @ mean_betas_nonstd + mean_alphas_nonstd

        if pft_index == 'pigment':
            pft_final_pred[pft_final_pred < 0] = 0
        elif pft_index == 'compositions':
            pft_final_pred = np.clip(pft_final_pred, 0, 1)

        final_r2s.append(r2_score(pft_val, pft_final_pred))
        final_rmses.append(np.sqrt(mean_squared_error(pft_val, pft_final_pred)))
        final_maes.append(np.mean(np.abs(pft_final_pred - pft_val)))
        final_biases.append(np.mean(pft_final_pred - pft_val))

        safe_pft_val = np.where(pft_val == 0, 1e-6, pft_val)
        final_percent_errors = np.abs((pft_final_pred - safe_pft_val) / safe_pft_val) * 100
        final_avg_pct_errors.append(np.mean(final_percent_errors))
        final_med_pct_errors.append(np.median(final_percent_errors))

        print(f"Permutation {i+1}/{n_permutations} complete.")

    # Compile final results
    coefficients = all_mean_betas_nonstd
    intercepts = all_mean_alphas_nonstd

    summary_gofs = {
        'Mean_R2': np.mean(final_r2s), 'SD_R2': np.std(final_r2s),
        'Mean_RMSE': np.mean(final_rmses), 'SD_RMSE': np.std(final_rmses),
        'Mean_MAE': np.mean(final_maes), 'SD_MAE': np.std(final_maes),
        'Mean_pct_bias': np.mean(final_biases), 'SD_pct_bias': np.std(final_biases),
        'Mean_mean_pct_error': np.mean(final_avg_pct_errors), 'SD_mean_pct_error': np.std(final_avg_pct_errors),
        'Mean_median_pct_error': np.mean(final_med_pct_errors), 'SD_median_pct_error': np.std(final_med_pct_errors)
    }

    all_gofs = {
        'R2s': np.array(final_r2s), 'RMSEs': np.array(final_rmses), 'MAEs': np.array(final_maes),
        'pct_bias': np.array(final_biases), 'mean_pct_error': np.array(final_avg_pct_errors),
        'median_pct_error': np.array(final_med_pct_errors)
    }

    # Save results to a .mat file
    savemat(output_file_name, {
        'coefficients': coefficients,
        'intercepts': intercepts,
        'summary_gofs': summary_gofs,
        'all_gofs': all_gofs
    })

    return coefficients, intercepts, summary_gofs, all_gofs
import numpy as np
from scipy.optimize import minimize
from scipy.io import savemat
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

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

        bounds = [(0, None), (0, None), (0, None)]

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

    all_mean_betas_nonstd = np.zeros((daph.shape[1], n_permutations))
    all_mean_alphas_nonstd = np.zeros(n_permutations)
    final_r2s, final_rmses, final_maes, final_biases = [], [], [], []
    final_avg_pct_errors, final_med_pct_errors = [], []

    for i in range(n_permutations):
        daph_train, daph_val, pft_train, pft_val = train_test_split(
            daph, pft, test_size=0.25, random_state=i
        )

        kf = KFold(n_splits=k, shuffle=True, random_state=i)

        fold_betas = np.zeros((daph.shape[1], k))
        fold_alphas = np.zeros(k)

        for j, (train_idx, val_idx) in enumerate(kf.split(daph_train)):
            cv_train_spec, cv_val_spec = daph_train[train_idx], daph_train[val_idx]
            cv_train_pft, cv_val_pft = pft_train[train_idx], pft_train[val_idx]

            scaler = StandardScaler()
            cv_train_spec_std = scaler.fit_transform(cv_train_spec)
            cv_val_spec_std = scaler.transform(cv_val_spec)

            n_samples_fold = cv_train_spec_std.shape[0]
            current_max_pcs = min(max_pcs, n_samples_fold - 1)

            pca = PCA(n_components=current_max_pcs)
            train_pcs = pca.fit_transform(cv_train_spec_std)

            gof_metrics = {
                'R2': [], 'RMSE': [], 'MAE': [], 'bias': [],
                'avg': [], 'med': [], 'ens': []
            }

            for l in range(1, current_max_pcs + 1):
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

                safe_pft_val = np.where(cv_val_pft == 0, 1e-6, cv_val_pft)
                percent_errors = np.abs((pft_pred - safe_pft_val) / safe_pft_val) * 100
                gof_metrics['avg'].append(np.mean(percent_errors))
                gof_metrics['med'].append(np.median(percent_errors))
                gof_metrics['ens'].append((1 - gof_metrics['R2'][-1] + gof_metrics['RMSE'][-1]) / 100)

            if mdl_pick_metric in ['RMSE', 'MAE', 'bias', 'avg', 'med', 'ens']:
                best_l = np.argmin(gof_metrics[mdl_pick_metric]) + 1
            else:
                best_l = np.argmax(gof_metrics[mdl_pick_metric]) + 1

            final_pca = PCA(n_components=best_l)
            final_train_pcs = final_pca.fit_transform(cv_train_spec_std)

            final_model = LinearRegression()
            final_model.fit(final_train_pcs, cv_train_pft)

            fold_betas[:, j] = final_pca.inverse_transform(final_model.coef_)
            fold_alphas[j] = final_model.intercept_

        mean_betas = np.mean(fold_betas, axis=1)
        mean_alphas = np.mean(fold_alphas)

        scaler_final = StandardScaler().fit(daph_train)
        mean_betas_nonstd = mean_betas / scaler_final.scale_
        mean_alphas_nonstd = mean_alphas - np.sum(mean_betas * scaler_final.mean_ / scaler_final.scale_)

        all_mean_betas_nonstd[:, i] = mean_betas_nonstd
        all_mean_alphas_nonstd[i] = mean_alphas_nonstd

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

    savemat(output_file_name, {
        'coefficients': coefficients,
        'intercepts': intercepts,
        'summary_gofs': summary_gofs,
        'all_gofs': all_gofs
    })

    return coefficients, intercepts, summary_gofs, all_gofs
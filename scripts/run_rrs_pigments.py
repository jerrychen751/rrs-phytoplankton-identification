import numpy as np
from scipy.io import loadmat
import os
import sys

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from rrs_model.processing import rrs_model_train

def main():
    """
    This script is a Python translation of the Kramer_Rrs_pigments.m MATLAB script.
    It models phytoplankton pigment concentrations from hyperspectral reflectance residuals.
    """
    # Load data
    print("Loading data...")
    # Note: The original script loads 'HGSM145_20200918.mat', which is not provided.
    # We will use the output of kramer_hyper_rrs.py as a stand-in for the Rrs residual.
    # This part of the code will need to be adapted once the actual input data is available.
    try:
        data = loadmat('data/Kramer_rrs_testdata.mat')
        # As a placeholder, we'll use the original Rrs and create a dummy residual.
        # In a real scenario, RrsD from the previous script would be loaded.
        Rrs = data['Rrs'].T
        # Create a dummy RrsD for demonstration purposes
        RrsD = Rrs * 0.1  # Placeholder for the actual residual
        Global_RHPLC = data['chl'] # Using 'chl' as a placeholder for pigment data
    except FileNotFoundError:
        print("Error: Kramer_rrs_testdata.mat not found. Please ensure the test data is available.")
        return

    # Set up output directory
    output_dir = "model_output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Calculate 1st and 2nd derivatives of the Rrs residual
    diffD = np.diff(RrsD, n=1, axis=0)
    diffD2 = np.diff(RrsD, n=2, axis=0)

    # Set parameters for the model
    n_permutations = 100  # As in the original script
    max_pcs = 10         # Reduced for speed in this example, original was 30
    mdl_pick_metric = 'MAE'
    k = 5
    pft_index = 'pigment'
    ofn_suffix = f'_rrsD2_1nm_{mdl_pick_metric}_py.mat'

    # The original script models a list of pigments. We will use the available 'chl' data.
    pigs2mdl = ['Tchla'] # Total chlorophyll-a, as an example

    # Use the 2nd derivative of the residual as the input spectra
    daph = diffD2.T # Transpose to have spectra as rows

    print("Starting model training...")
    for pig_name in pigs2mdl:
        # In the original script, 'Global_RHPLC' is a matrix with multiple pigments.
        # Here, we only have 'chl', so we'll use it for this example.
        pft = Global_RHPLC.flatten()

        output_file_name = os.path.join(output_dir, f"{pig_name}{ofn_suffix}")

        print(f"Training model for: {pig_name}")
        rrs_model_train(
            daph, pft, pft_index, n_permutations,
            max_pcs, k, mdl_pick_metric, output_file_name
        )
        print(f"Model for {pig_name} trained and results saved to {output_file_name}")

    print("\nAll pigment models trained successfully.")

if __name__ == "__main__":
    main()
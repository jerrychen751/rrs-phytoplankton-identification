# Rrs_pigments

This code was created to: 

1) model hyperspectral remote sensing reflectance (Rrs) and calculate a reflectance residual between measured and modeled values (Kramer_hyperRrs) and
2) then run a principal components regression model for phytoplankton pigment concentrations (Kramer_Rrs_pigments). 

Kramer_hyperRrs requires: hyperspectral measurements of Rrs, chlorophyll, temperature, salinity, asw (in aw_mcf16_350_700_1nm.txt), bsw (calculated from betasw_ZHH2009.m), aph coefficients (in aph_A_B_Coeffs_Sasha_RSE_paper.txt), gsm_cost and gsm_invert.

Kramer_Rrs_pigments requires: spectral data (here, Rrs residuals), pigment concentrations for all pigments you want to model, and rrsModelTrain.m to run the model.

All code and data in this repository are freely available for use by anyone for any and all applications. If you use this code, we ask that you please cite the paper where these methods were used:

Kramer, S.J., D.A. Siegel, S. Maritorena, D. Catlett (2022). Modeling surface ocean phytoplankton pigments from hyperspectral remote sensing reflectance on global scales. Remote Sensing of Environment, 270, 1-14, https://doi.org/10.1016/j.rse.2021.112879. 

Please notify me if you find errors and/or inaccuracies or if you have any questions. Dylan Catlett's repositories may also be helpful: https://github.com/dcat4/bioOptix_and_PFTs and https://github.com/dcat4/SDP_programs. This project is part of the #pace-sat: https://pace.gsfc.nasa.gov.

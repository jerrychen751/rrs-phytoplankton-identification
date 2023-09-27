%Sasha Kramer
%skramer@mbari.org
%UCSB IGPMS

%%%Script to run hyperspectral reflectance model from Kramer et al. (2022)
%Runs with: betasw_ZHH2009.m, gsm_invert.m, gsm_cost.m
%Inputs: measured Rrs, chlorophyll, temp, sal, asw, A&B coeffs for aph

%Rrs is a function of IOPs, specifically absorption and backscattering. 
%Here, you will build the absorption and backscattering terms from their 
%component parts, and then match the modeled spectra to the measured 
%spectra to optimize the model fit.

%Make sure all functions are in your path / directory!

%Map the directory where you will load your data:
cd /Users/skramer/Documents/UCSB/Research/Data/HPLC_Aph_Rrs

%Load your data
load Smoothed_Rrs.mat %Rrs
load chlorophyll.mat %chl
load temp_sal.mat %T, S

%The model uses reflectance = f(IOPs) from Gordon et al. (1988) which uses
%below the surface reflectance (rrs = Lu(0-)/Ed(0-)). If you are using 
%above surface reflectance (Rrs = Lu(0+)/Ed(0+)), you will need to convert
%your reflectances before running this model, using the equation below from
%Lee et al. (2002):
rrs = Rrs/(0.52 + 1.7*Rrs);

%Define wavelengths
wave = 400:1:700;

%%First, define total absorption as a sum of seawater absorption (asw),
%phytoplankton absorption (aph) and CDOM plus other detrital matter (acdm)
%a_tot = asw + aphstar + acdm
%asw = import aw_mcf16_350_700_1nm.txt - subset for your wavelengths
asw = asw(50:end);

%aphstar = aph/chl --> aph = import A & B coefficients from
%aph_A_B_Coeffs_Sasha_RSE_paper.txt
aph = A.*chl.^B;
aphstar = aph./chl;

%acdm slope is a function of Rrs (just above surface):
acdm_s = -(0.01447 + 0.00033*Rrs490/Rrs555);
acdm = exp(-acdm_s*(wave-443));

%%Then, define backscattering as a sum of seawater backscattering (bbsw) and
%backscattering by particles (bbp)
%bb_tot = bbsw + bbp

%bsw comes from Zhang et al. (2009): 
[~,~,bsw] = betasw_ZHH2009(wave,T,[],S);
bbsw = 0.5*bsw;

%bbp slope is a function of rrs (just below surface):
bbp_s = 2.0*(1.-1.2*exp(-0.9*rrs440rrs555)); %You will need to define rrs440 and rrs555 based on your rrs data
bbp = (443./wave).^bbp_s;

%Put IOPs together
IOPs = gsm_invert(rrs,asw,bbsw,bbp,aphstar,acdm);
%outputs = chl, acdm443, bbp443

%Reconstruct Rrs
a = asw + IOPs(:,1)*aphstar + IOPs(:,2)*acdm;
bb = bbsw + IOPs(:,3)*bbp;
rrsP = bb./(a+bb);

g = [0.0949 0.0794]; %coefficients from Gordon et al. (1988)
modrrs = (g(1) + g(2)*rrsP).*rrsP;

%Residual between measured and modeled (to use for Kramer_hyperRrs.m)
rrsD = rrs-modrrs;

base_dir = "spot_upper_fixed_R0"

# Planet properties:
R0 = 1.275  # Reference radius, in Jupiter radii
Rs = 1.152  # Stellar radii, in Solar radii
g = 2925.3  # Planetary gravity, in cm/s^2

# Star properties:
host_star = "HATP23"  # Loads properties from host_star_parameters.dat

data_fname = "tspec_hp23_c.csv"

# Number of live points to explore the likelihood
nlive = 10000

# Define molecules and atoms whose vmr will be fit:
molecules = Na

# Define minimum and maximum temperatures for the atmosphere
# (prior will be uniform between those temperatures):
Tmin_atm = 100
Tmax_atm = 3000

# Define if E1 will be added (this is slower, but more accurate):
Include_E1 = True

# Define if an additional grey opacity will be fit. If they are, define
# minimum and maximum cross-section (in log-space) for that absorber. Note that
# a *cloud deck* is already fitted to the data via the R0 parameter:
clouds = False
clouds_min = -50.0
clouds_max = 0.0

# Fit reference pressure? If yes, define minimum and maximum reference pressures for the
# atmosphere (in bars) (prior will be log-uniform between those pressure). If not, define
# P0_fix to be the reference pressure:
fit_P0 = True
P0_min = 1e-6
P0_max = 1e6
# P0_fix = 10.0

# Fit R0 normalization? If yes, a factor with prior U (1-diff_factor,1+diff_factor)
# will be multiplied to R0
fit_R0 = False
diff_factor = 0.2

# If flat_line is true, it will be assumed that the underlying
# transmission model is a flat line (i.e., emulating a thick cloud deck):
flat_line = False

# Define if stellar heterogeneity will be fit:
# Define if hazes will be fit:
hazes = False
heterogeneity = False
# Method for modeling heterogeneity signal. There are two options:
#  'simple' = two component photosphere, in which the planet occults a
#             region with a spectrum S_occ and there is also an unocculted
#             heterogeneity present (S_het) with a covering fraction F_het
#  'full' =   The photosphere is composed of three components--immaculate
#             photosphere (S_phot), spots (S_spot), and faculae (S_fac).
#             The disk-integrated covering fractions for these are F_phot,
#             F_spot, and F_fac, respectively, and within the transit chord,
#             their covering fractions are f_phot, f_spot, and f_fac.
het_mode = "simple"

# Fixed spot parameters
spot_temp = 3800.0
spot_frac = 0.026

# Temperature limits of stellar spectral models to explore
Tmin_star = 2300
Tmax_star = 6500

# Temperature limits of occulted spectrum (Uniform prior)
T_occ_d = 5420.0
T_occ_u = 6420.0

# Define re-binning for the cross sections, nbin (counted in wavenumber; no binning
# implies nbin = 0). The larger the bin, the faster the code is:
nbin = 0

# Method to interpolate cross sections. Two options:
#  'exact'    preforms 1d interpolation under the given
#             cross-sections at a given temperature (slow).
#  'nearest'  uses the cross section with the closest temperature
#             to the target tempreature during the posterior sampling,
#             using a pre-defined binning Tbin. This latter method is
#             orders of magnitude faster than the exact, and approaches
#             the exact method as Tbin -> 0.
cs_method = "nearest"
Tbin = 10.0

# print(f"{base_dir=}", type(base_dir))
# print(f"{spot_temp=}", type(spot_temp))
# print(f"{spot_frac=}", type(spot_frac))
# print(f"{fit_R0=}", type(fit_R0))
# print(f"{molecules=}", type(molecules))
# print(f"{clouds=}", type(clouds))
# print(f"{hazes=}", type(hazes))
# print(f"{heterogeneity=}", type(heterogeneity))

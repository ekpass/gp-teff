""" This script calculates the effective temperature of an exoplanet from a collection of brightness temperatures
using the Gaussian Process method presented in Pass et al. 2018.  Input parameters can be set on lines 22-24.

Results using the Error-Weighted Mean (Schwartz & Cowan 2015) or the Linear Interpolation (Cowan & Agol 2011) methods
can also be determined by enabling the 'printer' toggle (line 27) although we strongly recommend the GP method for its
robust uncertainty estimation.  Here the EWM uses the 1/sigma weighting suggested in Pass et al 2018. """

#  ADJUST FONT SIZE
import matplotlib
matplotlib.rcParams.update({'font.size': 14})

#  IMPORTS
import george
import numpy as np
import matplotlib.pyplot as plt
from george import kernels
from scipy.interpolate import interp1d
import scipy.constants as const
import scipy.integrate as integral

#  INPUTS
bands = [1.4e-6, 3.55e-6, 4.493e-6]  # central wavelengths of observed bands (m)
t_brights = [1984.77651162, 1835.67521425, 1634.78162354]  # brightness temperatures in these bands (K)
uncs = [35.9555724, 71.52144923, 62.59183888]  # uncertainties on brightness temperatures (K)

#  TOGGLES
printer = False  # print GP, EWM, and LI fits
plotter = False  # show plot of GP, EWM, and LI fits

#  SCIENTIFIC CONSTANTS
c = const.c
pi = const.pi
h = const.value('Planck constant')
k = const.value('Boltzmann constant')
sig = const.value('Stefan-Boltzmann constant')
const_1 = 2.0 * h * c ** 2
const_2 = h*c/k


# CONVERT BRIGHTNESS TEMPERATURE SPECTRUM TO EFFECTIVE TEMPERATURE
def Convert_to_T_eff(x, y):
    Temps = const_1 / (x ** 5 * (np.exp(const_2 / (x * y)) - 1))
    t_eff = abs(integral.trapz(Temps, x))
    t_eff = ((t_eff * pi) / sig) ** .25
    return t_eff


#  FIT BRIGHTNESS TEMPERATURE SPECTRUM WITH LINEAR INTERPOLATION METHOD
def linearInterpolation(domain, bds, y):
    LI = interp1d(bds, y)
    min_band = np.argmin(bds)
    max_band = np.argmax(bds)
    interp_y = [y[min_band] if wav < bds[min_band] else y[max_band] if wav > bds[max_band] else LI(wav) for wav in domain]
    return interp_y


#  ESTIMATE EFFECTIVE TEMPERATURE USING GP, LI, and EWM METHODS
def main(bds, y, yerr, clr):
    domain = np.logspace(-7, -4, 100)
    divisor = 1.e15
    lin_int_y = linearInterpolation(domain, bds, y)  # calculate brightness temperature spectrum using LI method

    domain_freq = np.array([c/lam/divisor for lam in domain])  # convert domain to frequency in petahertz
    new_bds = np.array([c / bd / divisor for bd in bds])  # convert bands to frequency in petahertz

    kernal_value = 0.1
    kern = kernels.ExpSquaredKernel(kernal_value)  # initialize covariance kernel

    functions_GP = []
    functions_LI = []

    randomized = np.array(list(map(np.random.normal, y, yerr, [100] * len(yerr)))).T
    ewms = []

    for y_rand in randomized:   # Monte Carlo

        # Linear interpolation method
        functions_LI.append(linearInterpolation(domain, bds, y_rand))

        # Error-weighted mean method
        av = np.average(y_rand, weights=1./yerr)
        ewms.append(av)

        # Setting up the GP
        y_rand /= av  # normalize the data
        model = george.GP(np.var(y_rand) * kern, mean=1.)
        model.compute(new_bds, yerr=yerr/av)

        # Setting the hyperparameters
        model["kernel:k1:log_constant"] = -4  # signal variance trained on exoplanets.org secondary eclipse sample
        model['kernel:k2:metric:log_M_0_0'] = -8.55  # length scale trained on HITEMP spectrum of water

        functions_GP.extend(model.sample_conditional(y_rand, domain_freq, 100) * av)  # 100 samples from posterior

    # calculate uncertainty & z-score for EMW method
    mean_mean = np.mean(ewms)
    std_mean = np.std(ewms)

    # convert brightness temperature functions to effective temperatures
    temps_GP = [Convert_to_T_eff(domain, pred) for pred in functions_GP]
    temps_LI = [Convert_to_T_eff(domain, pred) for pred in functions_LI]

    # calculate uncertainty & z-score for GP-fixed method
    mean_gp = np.mean(temps_GP)
    std_gp = np.std(temps_GP)

    # calculate uncertainty & z-score for LI method
    mean_LI = np.mean(temps_LI)
    std_LI = np.std(temps_LI)

    if printer:  # print effective temperature estimates for the three model-independent methods
        print "GP:", mean_gp, std_gp
        print "EWM:", mean_mean, std_mean
        print "LI:", mean_LI, std_LI

    if plotter:  # plot the brightness temperature spectral fits for the three model-independent methods
        # initialize figure
        f = plt.figure()
        ax = f.add_subplot(1, 1, 1)

        # calculate GP confidence intervals
        all_results = np.array(functions_GP)
        all_results_mean = np.mean(all_results, axis=0)
        all_results_std = np.std(all_results, axis=0)
        domain_plt = [d * 1e6 for d in domain]  # convert domain to microns for plotting

        # plot GP confidence intervals
        plt.fill_between(domain_plt, all_results_mean-all_results_std, all_results_mean+all_results_std,
                         color=clr, alpha=0.2)
        plt.fill_between(domain_plt, all_results_mean-2*all_results_std, all_results_mean+2*all_results_std,
                     color=clr, alpha=0.2)
        plt.fill_between(domain_plt, all_results_mean-3*all_results_std, all_results_mean+3*all_results_std, color=clr, alpha=0.2)
        plt.semilogx(domain_plt, all_results_mean, color=clr, zorder=10)

        # plot observations and model spectrum
        bds_plt = [bd * 1e6 for bd in bds]
        plt.xlim(0.1, 100)
        plt.errorbar(bds_plt, y, yerr=yerr, fmt=".k", capsize=0, zorder=100, color='black', label="Observations")

        # plot GP, EWM, LI functions
        plt.semilogx(domain_plt, lin_int_y, linestyle='dashed', color='yellow', label="LI")
        plt.semilogx(domain_plt, [mean_mean]*len(domain_plt), linestyle='dotted', color='red', label="EWM")
        plt.semilogx(domain_plt, all_results_mean, color=clr, zorder=10, label="GP")

        # pretty up the figure
        ticks = ax.get_xticks()
        ax.set_xticklabels(ticks)
        plt.xlabel(r"$\lambda (\mu m)$")
        plt.ylabel("Brightness Temperature (K)")
        plt.legend(loc="upper right")
        plt.tight_layout()

        plt.show()

    return mean_gp, std_gp


#  convert inputs to arrays
bands = np.array(bands)
t_brights = np.array(t_brights)
uncs = np.array(uncs)

t_eff, unc_t_eff = main(bands, t_brights, uncs, "b")

if not printer:  # print GP results (only if detailed print statements are toggled off)
    print "Effective temperature:", t_eff, "+/-", unc_t_eff

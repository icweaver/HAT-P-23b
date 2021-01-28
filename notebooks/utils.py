import glob as glob
import matplotlib as mpl
import matplotlib.patheffects as PathEffects
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
import pandas as pd
import seaborn as sns

import bz2
import corner
import json
import pathlib
import pickle
import warnings

from astropy import constants as const
from astropy import units as uni
from astropy.io import ascii, fits
from astropy.time import Time
from mpl_toolkits.axes_grid1 import ImageGrid

warnings.filterwarnings("ignore", r"All-NaN (slice|axis) encountered")
warnings.filterwarnings("ignore", r"Degrees of freedom <= 0 for slice")


def _bad_idxs(s):
    if s == "[]":
        return []
    else:
        # Merges indices/idxs specified in `s` into a single numpy array of
        # indices to omit
        s = s.strip("[]").split(",")
        bad_idxs = list(map(_to_arr, s))
        bad_idxs = np.concatenate(bad_idxs, axis=0)
        return bad_idxs


def _to_arr(idx_or_slc):
    # Converts str to 1d numpy array
    # or slice to numpy array of ints.
    # This format makes it easier for flattening multiple arrays in `_bad_idxs`
    if ":" in idx_or_slc:
        lower, upper = map(int, idx_or_slc.split(":"))
        return np.arange(lower, upper + 1)
    else:
        return np.array([int(idx_or_slc)])


def compress_pickle(fname_out, fpath_pickle):
    data = load_pickle(fpath_pickle)
    with bz2.BZ2File(f"{fname_out}.pbz2", "wb") as f:
        pickle.dump(data, f)


def decompress_pickle(fname):
    data = bz2.BZ2File(fname, "rb")
    return pickle.load(data)


def get_lnZ(data):
    # load in exoretreival pkls
    #with open(fpath, 'rb') as f:
    #    post = pickle.load(f)
    #with open(fpath_flat, 'rb') as f:
    #    post_flat = pickle.load(f)

    #delta_ln_Z = post['lnZ'] - post_flat['lnZ']
    #model_unc = post['lnZerr']
    #model_flat_unc = post_flat['lnZerr']
    #delta_ln_Z_unc = np.sqrt(sM**2 + sF**2)
    #return delta_ln_Z, delta_ln_Z_unc, post['lnZ'], sM
    return data['lnZ'], data["lnZerr"]

def get_phases(t, P, t0):
    """
    Given input times, a period (or posterior dist of periods)
    and time of transit center (or posterior), returns the
    phase at each time t. From juliet =]
    """
    if type(t) is not float:
        phase = ((t - np.median(t0)) / np.median(P)) % 1
        ii = np.where(phase >= 0.5)[0]
        phase[ii] = phase[ii] - 1.0
    else:
        phase = ((t - np.median(t0)) / np.median(P)) % 1
        if phase >= 0.5:
            phase = phase - 1.0
    return phase


def get_result(fpath, key="t0", unc=True):
    data = np.genfromtxt(fpath, encoding=None, dtype=None)
    for line in data:
        if key in line:
            if unc:
                return line
            else:
                return line[1]

    print(f"{key} not found. Check results.dat file.")


def load_pickle(fpath):
    with open(fpath, "rb") as f:
        data = pickle.load(f, encoding="latin")  # Python 2 -> 3
    return data


def myparser(s):
    dt, day_frac = s.split(".")
    dt = datetime.strptime(dt, "%Y-%m-%d")
    ms = 86_400_000.0 * float(f".{day_frac}")
    ms = timedelta(milliseconds=int(ms))
    return dt + ms


def plot_binned(
    ax,
    idxs_used,
    fluxes,
    bins,
    offset,
    colors,
    annotate=False,
    utc=False,
    species=None,
    bold_species=True,
    plot_kwargs=None,
    annotate_kwargs=None,
    annotate_rms_kwargs=None,
    models=None,
):
    """
    Plots binned light curves.

    Parameters
    ----------
    ax : matplotib.axes object
        Current axis to plot on
    idxs_used: index, time, phase, etc.
    fluxes : ndarray
        `time[idxs_used]` x `wbin` array of fluxes. Each column corresponds to a wavelength
        binned LC, where `wbin` is the number of wavelength bins
    bins : ndarray
        `wbin` x 2 array of wavelength bins. The first column holds the lower
        bound of each bin, and the second column holds the upper bound for each.
    offset : int, float
        How much space to put between each binned LC on `ax`
    colors : ndarray
        `wbin` x 3 array of RGB values to set color palette
    annotate : bool, optional
        Whether to annotate wavelength bins on plot. Default is True.
    utc : bool, optional
        Whether to convert `time` to UTC or not. Default is False.
    bold_species : bool, optional
        Whether to make annotated bins bold if they are in
    plot_kwargs : dict, optional
        Optional keyword arguments to pass to plot function
    annotate_kwargs : dict, optional
        Optional keyword arguments to pass to annotate function

    Returns
    -------
    ax : matplotib.axes object
        Current axis that was plotted on.
    """
    if plot_kwargs is None:
        plot_kwargs = {}
    if annotate_kwargs is None:
        annotate_kwargs = {}
    if annotate_rms_kwargs is None:
        annotate_rms_kwargs = {}

    offs = 0
    if idxs_used is None:
        idx_used = range
        slc = slice(0, len(fluxes.shape[0]) + 1)
    else:
        slc = idxs_used

    # fluxes = fluxes[slc, :]
    N = bins.shape[0]  # number of wavelength bins

    for i in range(N):
        wav_bin = [round(bins[i][j], 3) for j in range(2)]

        if utc:
            t_date = Time(time, format="jd")
            ax.plot_date(
                t_date.plot_date,
                fluxes[:, i] + offs,
                c=colors[i],
                label=wav_bin,
                **plot_kwargs,
            )
        else:
            ax.plot(
                idxs_used,
                fluxes[:, i] + offs,
                c=0.9 * colors[i],
                label=wav_bin,
                # mec=0.9*colors[i],
                **plot_kwargs,
            )
            if models is not None:
                ax.plot(
                    idxs_used, models[:, i] + offs, c=0.6 * colors[i], lw=2
                )

        if annotate:
            # trans = transforms.blended_transform_factory(
            #            ax.transAxes, ax.transData
            #        )
            trans = transforms.blended_transform_factory(
                ax.transData, ax.transData
            )
            # Annotate wavelength bins
            ann = ax.annotate(
                wav_bin,
                # xy=(0, 1.004*(1 + offs)),
                xy=(idxs_used[-1], 1.002 * (1 + offs)),
                xycoords=trans,
                **annotate_kwargs,
            )
            rms = np.std(fluxes[:, i]) * 1e6
            ann_rms = ax.annotate(
                f"{int(rms)}",
                xy=(idxs_used[0], 1.002 * (1 + offs)),
                xycoords=trans,
                **annotate_rms_kwargs,
            )

            # Make annotations bold if bin is a species bin
            if bold_species:
                if species is None:
                    species = dict()
                for spec, spec_wav in species.items():
                    if wav_bin[0] <= spec_wav <= wav_bin[1]:
                        ann.set_text(f"{spec}\n{ann.get_text()}")
                        ann.set_weight("bold")

        offs += offset

    return ax


def plot_chips(
    dirpath,
    fpathame,
    target="",
    vmin=0,
    vmax=2_000,
    spec_ap=0,
    sky_ap=0,
):
    # This plots the chips by numbers:
    #
    # 1 2 3 4
    # 6 5 8 7
    #
    class CoordinateData:
        """
        A simple class to hold coordinate data.
        """

        def __init__(self, filename, skiplines=0):
            self.fpathame = filename
            self.obj = np.array([])
            self.chip = np.array([])
            self.x = np.array([])
            self.y = np.array([])

            with open(filename) as f:
                for _ in range(skiplines):
                    next(f)
                for line in f:
                    splitted = line.split()
                    if len(splitted) == 0:
                        break
                    self.obj = np.append(self.obj, splitted[0])
                    self.chip = np.append(self.chip, splitted[1][-1])
                    self.x = np.append(self.x, splitted[2])
                    self.y = np.append(self.y, splitted[3])
            self.chip = self.chip.astype(np.int)
            self.x = self.x.astype(np.float)
            self.y = self.y.astype(np.float)

    coords_file = f"{dirpath}/{target}.coords"
    coords = CoordinateData(coords_file)

    def biassec(x):
        x = x.split(",")
        fpathum = ((x[0])[1:]).split(":")
        snum = ((x[1])[: len(x[1]) - 1]).split(":")
        fpathum[0] = int(fpathum[0])
        fpathum[1] = int(fpathum[1])
        snum[0] = int(snum[0])
        snum[1] = int(snum[1])
        return fpathum, snum

    def zero_oscan(d):
        zrows = len(np.where(d[:, 0] == 0)[0])
        zcols = len(np.where(d[0, :] == 0)[0])
        if zrows > zcols:
            mode = "r"
            length = d.shape[1]
        else:
            mode = "c"
            length = d.shape[0]
        cmed = []
        for i in range(length):
            if mode == "r":
                data = d[:, i]
            else:
                data = d[i, :]
            I = np.where(data != 0)[0]
            cmed.append(np.median(data[I]))
        return np.median(np.array(cmed))

    def BiasTrim(d, c, h, otype, datasec=None):
        """
        Overscan/Bias correct and Trim an IMACS chip
        """
        # bias has no significant structure, so a single median suffices, I think
        # overscan = [0:49] [2097:2145]
        oxsec, oysec = biassec(h["biassec"])
        if datasec == None:
            dxsec, dysec = biassec(h["datasec"])
        else:
            dxsec, dysec = biassec(datasec)
        if otype == "ift":
            oscan_data = d[
                (oysec[0] - 1) : oysec[1], (oxsec[0] - 1) : oxsec[1]
            ]
            overscan = np.median(oscan_data)
            if overscan == 0:
                overscan = zero_oscan(oscan_data)
            newdata = (
                d[(dysec[0] - 1) : dysec[1], (dxsec[0] - 1) : dxsec[1]]
                - overscan
            )
        else:
            d = d.transpose()
            oscan_data = d[oxsec[0] - 1 : oxsec[1], oysec[0] - 1 : oysec[1]]
            overscan = np.median(oscan_data)
            if overscan == 0:
                overscan = zero_oscan(oscan_data)
            newdata = (
                d[dxsec[0] - 1 : dxsec[1], dysec[0] - 1 : dysec[1]] - overscan
            )
        # overscan = np.median(d[:,2048:2112])
        # newdata = d[:4095,0:2048] - overscan
        if (c == "c5") or (c == "c6") or (c == "c7") or (c == "c8"):
            if otype == "iff":
                newdata = newdata[::-1, :]
            else:
                newdata = newdata[::-1, ::-1]

        return newdata

    ############
    # Plot chips
    ############
    image = fpathame
    otype = image[0:3]
    chips = ["c1", "c2", "c3", "c4", "c6", "c5", "c8", "c7"]
    order = [1, 2, 3, 4, 6, 5, 8, 7]
    ranges = [vmin, vmax]

    fig = plt.figure(figsize=(11, 8))
    grid = ImageGrid(
        fig,
        111,
        nrows_ncols=(2, 4),
        axes_pad=0.05,
        cbar_location="right",
        cbar_mode="single",
        cbar_size="5%",
        cbar_pad=0.05,
    )

    for i, ax in enumerate(grid):
        # print('Chip number '+(chips[i])[1:]+' overscan:')
        # ax.plot(int('24'+str(i+1)))
        # print(image, chips)
        d, h = fits.getdata(f"{dirpath}/{image}{chips[i]}.fits", header=True)
        d = BiasTrim(d, chips[i], h, "ift")
        im = ax.imshow(d, vmin=ranges[0], vmax=ranges[1], cmap="magma")
        ax.xaxis.set_ticks(np.arange(250, 1_000, 250))
        ax.yaxis.set_ticks(np.arange(250, 2_000, 250))
        ax.tick_params(axis="both", which="major", labelsize=10)
        va = "bottom"
        if chips[i] in ["c6", "c5", "c8", "c7"]:
            va = "top"
        for c in range(len(coords.chip)):
            if order[i] == coords.chip[c]:
                col = "w"
                if target in coords.obj[c]:
                    col = "y"
                ax.axvline(coords.x[c] - sky_ap, color="g", lw=1)
                ax.axvline(coords.x[c] + sky_ap, color="g", lw=1)
                txt = ax.text(
                    coords.x[c],
                    coords.y[c],
                    coords.obj[c],
                    color=col,
                    ha="right",
                    va=va,
                    fontsize=14,
                    rotation=90,
                )
                txt.set_path_effects(
                    [PathEffects.withStroke(linewidth=2, foreground="k")]
                )
        # print('Shape:',d.shape)
        y = 1.05
        if i > 3:
            y = -0.2
        ax.set_title("Chip " + (chips[i])[1:], fontsize=12, y=y)
        ax.grid(False)
        date_time = f"{h['DATE-OBS']} -- {h['TIME-OBS']}"

    plt.axis("off")
    fig.colorbar(im, cax=grid.cbar_axes[0])
    return fig, im


def plot_corner(
    samples,
    fpath_truths=None,
    title=None,
    fig=None,
    c="C0",
    plot_truth=True,
    weights=None,
    ranges=None,
    params=None,
    corner_kwargs=None,
    hist_kwargs=None,
):
    if corner_kwargs is None:
        corner_kwargs = {}
    if hist_kwargs is None:
        hist_kwargs = {}
    # Load data
    if fpath_truths is not None:
        with open(fpath_truths) as f:
            params_dict = json.load(f)
        labels = [p["symbol"] for p in params_dict.values()]
    else:
        labels = list(params.values())

    #############
    # Plot corner
    #############
    fig = corner.corner(
        samples,
        plot_datapoints=False,
        color=c,
        labels=labels,
        # show_titles=False,
        label_kwargs={"rotation": 45},
        fill_contours=True,
        # hist_kwargs={'histtype':'step', 'lw':2, 'density':True},
        hist_kwargs=hist_kwargs,
        # title_fmt='.4e',
        # title_fmt='.5f',
        title_kwargs={"ha": "left", "x": -0.03},
        # quantiles=[0.16, 0.5, 0.84],
        use_math_text=True,
        labelpad=0.4,
        fig=fig,
        weights=weights,
        range=ranges,
        # truths=truths,
        # truth_color="grey",
        **corner_kwargs,
    )

    # Loop over 1D histograms and overplot truth values
    ndim = samples.shape[1]
    axes = np.array(fig.axes).reshape((ndim, ndim))

    # Turn grid lines off
    for ax in fig.axes:
        ax.grid(False)

    fig.suptitle(title, x=0.99, y=0.99, style="italic", ha="right")

    return fig, axes


def plot_divided_wlcs(
    ax,
    data,
    t0=0,
    ferr=0,
    c="b",
    comps_to_use=None,
    div_kwargs=None,
    bad_div_kwargs=None,
    bad_idxs_user=None,
):
    # Create plotting config dictionaries if needed
    if div_kwargs is None:
        div_kwargs = {}

    # Unpack data
    flux_target = np.c_[data["oLC"]]
    flux_comps = data["cLC"]
    cNames = data["cNames"]  # Original unsorted order from tepspec
    time = data["t"]
    airmass = data["Z"]
    flux_comps = flux_comps[:, np.argsort(cNames)]  # Sort columns by cName
    cNames = sorted(cNames)
    comps_to_use = sorted(comps_to_use)
    comps_to_use_idxs = [cNames.index(cName) for cName in comps_to_use]
    flux_comps_used = flux_comps[:, comps_to_use_idxs]

    ##############################################
    # Plot division by individual comparison stars
    ##############################################
    flux_divs = flux_target / flux_comps_used
    flux_divs /= np.median(flux_divs, axis=0)

    for flux_div, cName in zip(flux_divs.T, comps_to_use):
        bad_idxs = []
        for i in range(1, len(flux_div) - 1):
            diff_left = np.abs(flux_div[i] - flux_div[i - 1])
            diff_right = np.abs(flux_div[i] - flux_div[i + 1])
            if (diff_left > 2 * ferr) and (diff_right > 2 * ferr):
                bad_idxs.append(i)

        if bad_idxs_user is not None:
            bad_idxs_user = _bad_idxs(bad_idxs_user)
            ax.errorbar(
                (time[bad_idxs_user] - t0) * 24.0,
                flux_div[bad_idxs_user],
                yerr=ferr,
                zorder=10,
                **bad_div_kwargs,
            )
            bad_idxs = set(bad_idxs_user).union(set(bad_idxs))
            bad_idxs = list(bad_idxs)

        idxs = np.arange(len(flux_div))
        flux_div_used = flux_div[idxs != bad_idxs_user]
        idxs_used = idxs[idxs != bad_idxs_user]
        ax.errorbar(
            (time[idxs_used] - t0) * 24.0,
            flux_div_used,
            yerr=ferr,
            label=cName,
            **div_kwargs,
        )
        high_am = airmass > 2

    return ax


def plot_inset(
    ax,
    wav,
    tspec,
    tspec_unc,
    species,
    species_slc=slice(5, 10),
    box_lims=[0.38, 0.65, 0.2, 0.2],
    lims=(7657 - 10, 7697 + 10),
    mean_wlc_depth=0.0,
):
    axins = ax.inset_axes(box_lims)
    axins.axhline(mean_wlc_depth, color="darkgrey", zorder=0, ls="--")
    [
        axins.axvline(wav, ls="--", lw=0.5, color="grey", zorder=0)
        for name, wav in species.items()
    ]
    p_in = axins.errorbar(
        wav[species_slc],
        tspec[species_slc],
        yerr=tspec_unc[species_slc],
        c="w",
        mec="k",
        fmt="o",
        zorder=10,
        ecolor="k",
        lw=4,
    )
    axins.set_xlim(lims)
    # axins.set_ylim(ax.get_ylim())
    ax.indicate_inset_zoom(axins, alpha=1.0, edgecolor="k")


def plot_model(
    ax,
    model_dict,
    fill_kwargs=None,
    model_kwargs=None,
    sample=True,
    sample_kwargs=None,
):
    """Plots the retrieval model and 1-sigma region.

    Parameters
    ----------
    ax : matplotlib.axes
        Axis to plot onto.

    model_dict : Dict with the following keys:
        data : astropy.table.table.Table .
            The table is read from retr_model.txt and contains five columns:
            1) wav: wavelengths (Å), 2) flux: Transit depth (ppm),
            3/4) wav_d, wav_u: +/- on wav, 5) flux_err: +/- on flux.

        sampled_data : astropy.table.table.Table .
            sampled data at given wavelengths in transmission spectrum

    fill_kwargs : dict
        keyword arguments to pass to ax.fill_between().

    model_kwargs : dict
        keyword arguments to pass to ax.semilogx().
    """
    if fill_kwargs is None:
        fill_kwargs = {}
    if model_kwargs is None:
        model_kwargs = {}
    if sample_kwargs is None:
        sample_kwargs = {}

    # Plot fill
    model = model_dict["data"]
    wav, flux_d, flux_u = model["wav"], model["flux_d"], model["flux_u"]
    p = ax.fill_between(wav, flux_d, flux_u, **fill_kwargs)

    # Plot model
    flux_model = model["flux"]
    c = p.get_facecolor()[0][0:3]
    p = ax.plot(wav, flux_model, c=c, **model_kwargs)

    # Plot model sampled
    if sample:
        model_sampled = model_dict["sampled_data"]
        wav_sampled, flux_sampled = model_sampled["wav"], model_sampled["flux"]
        ax.plot(wav_sampled, flux_sampled, "s", c=c)

    return ax


def plot_instrument(
    ax,
    instrument_data=None,
    instr_kwargs=None,
):
    """Plot retrieved transmission spectrum.

    Parameters
    ----------
    ax : matplotlib.axes
        Axis to plot onto.

    instrument_data : Dict
        Dict with the following keys:
        data : Table
            The file is read from retr_<instrument>.txt and contains five
            columns: 1) wav: wavelengths (Å), 2) flux: Transit depth (ppm),
            3/4) wav_d, wav_u: +/- on wav, 5) flux_err: +/- on flux.

        plot_kwargs : Dict
            keyword arguments to pass to `ax.errorbar` for the whole plot.
    """
    # Plot instrument
    instr = instrument_data["data"]
    wav, flux = instr["wav"], instr["flux"]
    x_d, x_u, y_err = instr["wav_d"], instr["wav_u"], instr["flux_err"]
    ax.errorbar(
        wav,
        flux,
        # xerr=[x_d, x_u],
        yerr=y_err,
        **instrument_data["plot_kwargs"],
    )

    return ax


def plot_spec_file(
    ax, fpath=None, data=None, wavs=None, i=1, label=None, median_kwargs=None
):
    """
    plots items in <object>_spec.fits files.
    has shape time x spec_item x ypix [it's ~ 2048] (|| to wav direction)
    `i`:
    0: Wavelength
    1: Simple extracted object spectrum
    2: Simple extracted flat spectrum
    3: Pixel sensitivity (obtained by the flat)
    4: Simple extracted object spectrum/pixel sensitivity
    5: Sky flag (0 = note uneven sky, 1 = probably uneven profile,
       2 = Sky_Base failed)
    6: Optimally extracted object spectrum
    7: Optimally extracted object spectrum/pixel sensitivity
    """
    if median_kwargs is None:
        median_kwargs = {}

    if fpath is not None:
        data = fits_data(fpath)
        specs = data[:, i, :]
        wavs = range(specs.shape[1])
    else:
        specs = data
        wavs = wavs

    specs[specs <= 0] = np.nan  # Sanitize data
    specs_med = np.nanmedian(specs, axis=0)
    specs_std = np.nanstd(specs, axis=0)
    factor = np.nanmedian(specs_med)
    specs_med /= factor
    specs_std /= factor

    # Empty plot for custom legend
    p = ax.plot([], "-o", label=label, **median_kwargs)

    # Plot normalized median and 1-sigma region
    specs_med /= np.nanmax(specs_med)
    specs_std /= np.nanmax(specs_med)
    c = p[0].get_color()
    ax.fill_between(
        wavs,
        specs_med - specs_std,
        specs_med + specs_std,
        color=c,
        alpha=0.25,
        lw=0,
    )
    ax.plot(wavs, specs_med, lw=2, color=c)
    return ax, wavs, data


def savefig(fpath):
    pathlib.Path(fpath).parents[0].mkdir(parents=True, exist_ok=True)
    plt.savefig(fpath, bbox_inches="tight")


def wbin_num(fpath):
    # Extracts <num> from fpath = .../wbin<num>/...
    tokens = fpath.split("/")
    for token in tokens:
        if "wbin" in token:  # Do the extraction
            bin_str = token.split("wbin")[-1]
            bin_num = int(bin_str)
            return bin_num


def weighted_err(errs):
    weights = 1 / errs ** 2
    partials = weights / np.sum(weights, axis=0)
    deltas = np.sqrt(np.sum((partials * errs) ** 2, axis=0))
    return deltas


def weighted_mean_uneven_errors(k, k_up, k_low, model=1):
    """
    A function to calculate the weighted mean of multiple, concatenated,
    transmission spectra that have un-even (non-symmetric) uncertainties. This
    uses the models of Barlow 2003. Inputs: k - the concatenated Rp/Rs values
    k_up - the concatenated positive uncertainties in Rp/Rs k_low - the
    concatenated negative uncertainties in Rp/Rs model - the number of the
    model as given in Barlow 2003 (either 1 or 2) Returns: weighted mean Rp/Rs
    the uncertainties in the weighted mean Rp/Rs values
    """
    nvalues = len(k)
    sigma = {}
    alpha = {}
    V = {}
    b = {}
    w = {}
    x_numerator = 0
    x_denominator = 0
    e_numerator = 0
    e_denominator = 0
    for i in range(nvalues):
        sigma[i + 1] = (k_up[i] + k_low[i]) / 2.0  # eqn 1
        alpha[i + 1] = (k_up[i] - k_low[i]) / 2.0  # eqn 1
        if model == 1:
            V[i + 1] = (
                sigma[i + 1] ** 2 + (1 - 2 / np.pi) * alpha[i + 1] ** 2
            )  # eqn 18
            b[i + 1] = (k_up[i] - k_low[i]) / np.sqrt(2 * np.pi)  # eqn 17
        if model == 2:
            V[i + 1] = sigma[i + 1] ** 2 + 2 * alpha[i + 1] ** 2  # eqn 18
            b[i + 1] = alpha[i + 1]  # eqn 17
        w[i + 1] = 1 / V[i + 1]
        x_numerator += w[i + 1] * (k[i] - b[i + 1])  # eqn 16
        x_denominator += w[i + 1]
        e_numerator += (w[i + 1] ** 2) * V[i + 1]  # below eqn 17
        e_denominator += w[i + 1]
    return x_numerator / x_denominator, np.sqrt(
        e_numerator / (e_denominator ** 2)
    )


def write_latex_row(row):
    v, vu, vd = row
    return f"{v:.3f}^{{+{vu:.3f}}}_{{-{vd:.3f}}}"

def write_latex2(val, unc):
    return f'{val:.2f} \pm {unc:.2f}'

def write_latex_single(row):
    v, v_unc = row
    return f"{v:.3f} \pm {v_unc:.3f}"


def write_latex_wav(row):
    wav_d, wav_u = row
    return f"{wav_d:.1f} - {wav_u:.1f}"

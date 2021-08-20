import numpy as np
from scipy.integrate import trapz
from scipy.interpolate import Rbf, interp1d
from scipy.stats import skew
from sklearn.metrics import mean_squared_error


def _getLSPmetrics2(phen, xnew, nGS, num, phentype):
    """
    Obtain land surfurface phenology metrics

    Parameters
    ----------
    - phen: 1D array
        PhenoShape data
    - xnew: 1D array
        DOY values for PhenoShape data
    - n_phen: Integer
        Window size where to estimate SOS and EOS
    - num: Integer
        Number of output variables

    Outputs
    -------
    - 2D array with the following variables:

        SOS = DOY of start of season
        POS = DOY of peak of season
        EOS = DOY of end of season
        vSOS = Value at start of season
        vPOS = Value at peak of season
        vEOS = Value at end of season
        LOS = Length of season (DOY)
        MSP = Mean spring (DOY)
        MAU = Mean autum (DOY)
        vMSP = Mean spring value
        vMAU = Mean autum value
        AOS = Amplitude of season (in value units)
        IOS = Integral of season (SOS-EOS)
        ROG = Rate of greening [slope SOS-POS]
        ROS = Rate of senescence [slope POS-EOS]
        SW = Skewness of growing season
    """
    inds = np.isnan(phen)  # check if array has NaN values
    if inds.any():  # check is all values are NaN
        return np.repeat(np.nan, num)
    else:
        # basic variables
        vpos = np.max(phen)
        trough = np.min(phen)
        ampl = vpos - trough

        # get position of seasonal peak and trough
        ipos = np.where(phen == vpos)[0]
        pos = xnew[ipos]

        # scale annual time series to 0-1
        ratio = (phen - trough) / ampl

        # separate greening from senesence values
        dev = np.gradient(ratio)  # first derivative
        greenup = np.zeros([ratio.shape[0]],  dtype=bool)
        greenup[dev > 0] = True

        # select time where SOS and EOS are located (around trs value)
        # KneeLocator looks for the inflection index in the curve
        if phentype == 1:  # estimate SOS and EOS as median of the season
            i = np.median(xnew[:ipos[0]][greenup[:ipos[0]]])
            ii = np.median(xnew[ipos[0]:][~greenup[ipos[0]:]])
            sos = xnew[(np.abs(xnew - i)).argmin()]
            eos = xnew[(np.abs(xnew - ii)).argmin()]
            isos = np.where(xnew == int(sos))[0]
            ieos = np.where(xnew == eos)[0]
        elif phentype == 2:  # estimate SOS and EOS by inflection curves
            warnings.simplefilter("ignore")
            # consider only observation before POS for SOS
            knee1 = KneeLocator(xnew[0:ipos[0]], ratio[0:ipos[0]], S=2,
                                curve='convex', direction='increasing')
            sos = knee1.knee
            isos = np.where(xnew == knee1.knee)[0]

            # consider only observation after POS for EOS
            x = xnew[-(nGS - ipos[0] - 1):]
            y = ratio[-(nGS - ipos[0] - 1):]
            knee2 = KneeLocator(range(len(x)), np.flip(y), S=2,
                                curve='convex', direction='increasing')
            eos = x[np.where(
                np.flip(range(len(x))) == knee2.knee)[0]][0]
            ieos = np.where(xnew == eos)[0]
        else:
            print('phentype must be either 1 or 2')
        if sos is None:
            isos = 0
            sos = xnew[isos]
        if eos is None:
            ieos = len(xnew) - 1
            eos = xnew[ieos]

        # los: length of season
        los = eos - sos
        if los < 0:
            los = np.nan

        # get MSP, MAU (independent from SOS and EOS)
        
        # mean spring
        idx = np.mean(xnew[(xnew > sos) & (xnew < pos[0])])
        idx = (np.abs(xnew - idx)).argmin()  # indexing value
        msp = xnew[idx]  # DOY of MGS
        vmsp = phen[idx]  # mgs value

        # mean autum
        idx = np.mean(xnew[(xnew < eos) & (xnew > pos[0])])
        idx = (np.abs(xnew - idx)).argmin()  # indexing value
        mau = xnew[idx]  # DOY of MGS
        vmau = phen[idx]  # mgs value

        # doy of growing season
        green = xnew[(xnew > sos) & (xnew < eos)]
        id_ = []
        for i in range(len(green)):
            id_.append((xnew == green[i]).nonzero()[0])   
        # TODO: move id_ generation to a list comprehension = id_ = [(xnew == green[i]).nonzero()[0] for i in range(len(green))]
        
        # index of growing season
        id = np.array([item for sublist in id_ for item in sublist])

        # get intergral of green season
        ios = trapz(phen[id], xnew[id]) if len(id) > 0 else np.nan
        
        # skewness of growing season
        sw = skew(phen[id]) if len(id) > 0 else np.nan

        # rate of greening [slope SOS-POS]
        rog = (vpos - phen[isos]) / (pos - sos)

        # rate of senescence [slope POS-EOS]
        ros = (phen[ieos] - vpos) / (eos - pos)

        metrics = np.array((sos, pos[0], eos, phen[isos][0], vpos,
                            phen[ieos][0], los, msp, mau, vmsp, vmau, ampl, ios, rog[0],
                            ros[0], sw))

        return metrics
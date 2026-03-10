import numpy as np

def EstimRisk(data_):
    """Estimate the risk of mosquito presence based on environmental data based on the paper of Zarate et al. 2019. The World Lcim input variables are:
    T_mean: Mean temperature (Celsius)
    P_annual: Annual precipitation (mm)
    T_WarmestMonth: Temperature of the warmest month (Celsius)
    P_WarmestQuarter: Precipitation of the warmest quarter (mm)
    Returns a numpy array with the estimated risk."""
    
    
    T_mean = data_[:, 0]
    P_annual = data_[:, 1]
    T_WarmestMonth = data_[:, 2]
    P_WarmestQuarter = data_[:, 3]
    intercept = -17.45971
    coeff_T_mean = 0.2383233
    coeff_T_WarmestMonth = 0.5374626
    coeff_P_annual = 0.0008731735
    coeff_P_WarmestQuarter = 0.004820618
    x = intercept + coeff_T_mean * T_mean + coeff_T_WarmestMonth * T_WarmestMonth + coeff_P_annual * P_annual + coeff_P_WarmestQuarter * P_WarmestQuarter
    return x[:, np.newaxis]

def SigmaEstimRisk(data_):
    x = EstimRisk(data_)
    return 1 / (1 + np.exp(-x))

import math
import numpy as np


def aedes_precipitation_suitability(annual_precip_mm, min_annual_precip_mm=200.0):
    """Return True where annual precipitation (mm) exceeds the minimum."""
    return np.asarray(annual_precip_mm, dtype=float) > float(min_annual_precip_mm)


def aedes_temperature_suitability(monthly_temperatures, species="albopictus"):
    """
    Determine habitat suitability for Aedes mosquitoes based on temperature criteria.
    The limits have been adapted from 
    https://www.ecdc.europa.eu/en/disease-vectors/facts/mosquito-factsheets/aedes-albopictus
    section: Establishment thresholds (accessed 2025-09-04). 
    This function evaluates whether temperature conditions are suitable for Aedes
    mosquito survival and development by checking multiple temperature thresholds:
    - Minimum winter survival temperature (coldest month)
    - Average annual temperature requirement
    - Number of months with adequate development temperatures
    
    Parameters
    ----------
    monthly_temperatures : array-like, shape (n_locations, 12)
        Monthly mean temperatures in Celsius for n locations.
        Columns represent months from January to December.
    species : str, default "albopictus"
        Target mosquito species. Options: "albopictus" or "aegypti"
        
    Returns
    -------
    is_suitable : np.ndarray of bool, shape (n_locations,)
        Boolean array indicating temperature suitability for each location.
        True = suitable habitat, False = unsuitable habitat
        
    Raises
    ------
    ValueError
        If input shape is not (n_locations, 12) or species is not recognized
        
    Notes
    -----
    For Aedes albopictus:
    - Coldest month must be ≥ -3°C (egg overwintering survival)
    - Annual mean temperature must be ≥ 10°C
    - No minimum requirement for warm development months
    
    For Aedes aegypti:
    - Same temperature thresholds as albopictus (may need adjustment)
    - Requires at least 5 months ≥ 10°C for development
    """
    temperature_array = np.asarray(monthly_temperatures, dtype=float)
    
    # Validate input dimensions
    if temperature_array.ndim != 2 or temperature_array.shape[1] != 12:
        raise ValueError("Input must have shape (n_locations, 12) for 12 monthly values")

    # Validate species parameter
    if species not in ["albopictus", "aegypti"]:
        raise ValueError(f"Unsupported species '{species}'. Use 'albopictus' or 'aegypti'")
    
    # Define temperature thresholds based on species
    if species == "albopictus":
        min_winter_survival_temp = -3.0  # Coldest month threshold (°C)
        min_annual_mean_temp = 10.0      # Annual mean threshold (°C)
        min_warm_months_required = 0     # Minimum months ≥ development temp
    else:  # species == "aegypti"
        print("Warning: Using albopictus temperature criteria for aegypti. "
              "Consider species-specific parameter adjustment.")
        min_winter_survival_temp = -3.0
        min_annual_mean_temp = 10.0
        min_warm_months_required = 0
    
    # Calculate temperature criteria
    coldest_monthly_temp = temperature_array.min(axis=1)
    annual_mean_temp = temperature_array.mean(axis=1)
    warm_months_count = (temperature_array >= min_annual_mean_temp).sum(axis=1)
    
    # Evaluate suitability conditions
    survives_winter = coldest_monthly_temp >= min_winter_survival_temp
    adequate_annual_warmth = annual_mean_temp >= min_annual_mean_temp
    sufficient_development_months = warm_months_count >= min_warm_months_required
    
    # Location is suitable only if all criteria are met
    is_suitable = survives_winter & adequate_annual_warmth & sufficient_development_months
    
    return is_suitable

import numpy as np

def aedes_temperature_suitability_estimRisk(monthly_temperatures, species="albopictus"):
    """
    Estimate habitat suitability for Aedes based on monthly temperature-driven
    'risk' (mortality vs gonotrophic-cycle length). A location is suitable if
    at least one month satisfies: 1 / mortality_rate > length_gon_cycle.

    Parameters
    ----------
    monthly_temperatures : array-like, shape (n_locations, 12)
        Monthly mean temperatures in Celsius for n locations.
    species : {'albopictus', 'aegypti'}, default 'albopictus'
        Species model controlling temperature-response parameters.

    Returns
    -------
    is_suitable : np.ndarray of bool, shape (n_locations,)
        True if any month is suitable under the mortality/gonotrophic criterion.

    Raises
    ------
    ValueError
        If input shape is not (n_locations, 12) or species is not recognized.
    """
    T = np.asarray(monthly_temperatures, dtype=float)

    # Validate shape
    if T.ndim != 2 or T.shape[1] != 12:
        raise ValueError("Input must have shape (n_locations, 12) for 12 monthly values")

    if species not in {"albopictus", "aegypti"}:
        raise ValueError("Unsupported species. Use 'albopictus' or 'aegypti'.")

    # Prepare arrays
    mortality = np.zeros_like(T, dtype=float)

    with np.errstate(over='ignore', under='ignore', divide='ignore', invalid='ignore'):
        if species == "albopictus":
            # Piecewise mortality
            c1 = T < 15.0
            c2 = (T >= 15.0) & (T < 26.3)
            c3 = T >= 26.3

            mortality[c1] = 1.0 / (1.1 + np.exp(-4.04 + 0.576 * T[c1])) + 0.12
            mortality[c2] = 0.000339 * T[c2]**2 - 0.0189 * T[c2] + 0.336
            mortality[c3] = 1.0 / (1.065 + np.exp(32.2 - 0.92 * T[c3])) + 0.0747

            # Gonotrophic cycle (days)
            gon_len = 0.046 * T**2 - 2.77 * T + 45.3

        else:  # 'aegypti'
            Tk = T + 273.15  # Kelvin

            c1 = T < 22.0
            c2 = ~c1

            mortality[c1] = 1.0 / (1.22 + np.exp(-3.05 + 0.72 * T[c1])) + 0.196
            mortality[c2] = 1.0 / (1.14 + np.exp(51.4 - 1.3 * T[c2])) + 0.192

            NumD = np.exp(15725.0 / 1.987 * (1.0 / 298.0 - 1.0 / Tk))
            DenD = 1.0 + np.exp(1756481.0 / 1.987 * (1.0 / 447.2 - 1.0 / Tk))
            d = ((0.216 + 0.372) / 2.0) * Tk / 298.0 * NumD / DenD
            gon_len = 1.0 / d

        # Safety: invalid/zero/negative values -> not suitable for that month
        invalid = ~np.isfinite(mortality) | ~np.isfinite(gon_len) | (mortality <= 0) | (gon_len <= 0)
        month_ok = (1.0 / mortality) > gon_len
        month_ok[invalid] = False

    # A location is suitable if ANY month is suitable
    is_suitable = month_ok.any(axis=1)

    return is_suitable



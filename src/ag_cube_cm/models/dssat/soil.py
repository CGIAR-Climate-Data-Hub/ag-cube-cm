"""
ag_cube_cm.models.dssat.soil
=============================

Soil physics for the DSSAT profile (``.SOL``) and initial conditions.

Every function here replaces a value that used to be hardcoded. The guiding
rule is that a parameter should be *derived from the pixel's own soil* whenever
a published relation allows it, and configurable — never invented — when it
does not.

Sources
-------
Saturated conductivity
    Saxton, K.E. & Rawls, W.J. (2006). Soil water characteristic estimates by
    texture and organic matter for hydrologic solutions. *Soil Science Society
    of America Journal* 70(5), 1569–1578. Equation 16.
Stage-1 evaporation limit
    Ritchie, J.T. (1998), as implemented in the DSSAT soil generator.
Root growth factor
    ``exp(-0.02 × layer centre)`` below 15 cm — the DSSATTools convention,
    already implemented in :func:`ag_cube_cm.transform.soil_cube.calculate_rgf`.
Texture class
    Benham, E. & Ahrens, R.J. (2009). Clarification of Soil Texture Class
    Boundaries. USDA-NRCS.

Why not Rosetta
---------------
The reference implementation estimates conductivity with the Rosetta
pedotransfer model, which predicts the *whole* water-retention curve from
texture. aggeodata already supplies measured-equivalent retention points
(``wv1500``/``wv0033``/``wv0010`` → SLLL/SDUL/SSAT), so re-deriving them from
texture would contradict the numbers written into the same file. Saxton–Rawls
computes conductivity *from those retention points*, which keeps the profile
internally consistent and adds no dependency.
"""

from __future__ import annotations

import logging
import math

from ag_cube_cm.transform.soil_cube import (
    TEXTURE_CLASSES,
    calculate_rgf,
)

logger = logging.getLogger(__name__)

__all__ = [
    "DSSAT_TEXTURE_CODES",
    "texture_class",
    "dssat_texture_code",
    "saturated_conductivity",
    "stage1_evaporation_limit",
    "drainage_rate",
    "runoff_curve_number",
    "albedo",
    "root_growth_factors",
    "initial_mineral_nitrogen",
]

#: USDA class name → DSSAT's abbreviated texture code used in the .SOL header.
DSSAT_TEXTURE_CODES: dict[str, str] = {
    "sand": "S",
    "loamy sand": "LS",
    "sandy loam": "SL",
    "loam": "L",
    "silt loam": "SIL",
    "silt": "SI",
    "sandy clay loam": "SCL",
    "clay loam": "CL",
    "silty clay loam": "SICL",
    "sandy clay": "SC",
    "silty clay": "SIC",
    "clay": "C",
    "unknown": "-99",
}

#: Drainage rate (SLDR, fraction per day) by texture.
#: DSSAT soil-generator convention: free-draining sands lose most of the water
#: above field capacity each day; heavy clays retain it.
_DRAINAGE_BY_TEXTURE: dict[str, float] = {
    "sand": 0.85, "loamy sand": 0.75, "sandy loam": 0.65,
    "loam": 0.55, "silt loam": 0.50, "silt": 0.50,
    "sandy clay loam": 0.45, "clay loam": 0.40, "silty clay loam": 0.35,
    "sandy clay": 0.30, "silty clay": 0.25, "clay": 0.20,
}

#: SCS runoff curve number (SLRO) by texture, for row crops in fair condition.
#: Tracks the hydrologic soil group: A (sands) → D (clays).
_CURVE_NUMBER_BY_TEXTURE: dict[str, float] = {
    "sand": 68.0, "loamy sand": 68.0, "sandy loam": 73.0,
    "loam": 73.0, "silt loam": 73.0, "silt": 76.0,
    "sandy clay loam": 79.0, "clay loam": 80.0, "silty clay loam": 82.0,
    "sandy clay": 84.0, "silty clay": 85.0, "clay": 85.0,
}

_DEFAULT_DRAINAGE = 0.50
_DEFAULT_CURVE_NUMBER = 76.0


# ---------------------------------------------------------------------------
# Texture
# ---------------------------------------------------------------------------

def texture_class(sand_pct: float, clay_pct: float) -> str:
    """Return the USDA texture class for a single layer.

    Scalar wrapper around the package's vectorised classifier, so the DSSAT
    writer and the soil-cube builder cannot disagree about what a soil is.

    Parameters
    ----------
    sand_pct, clay_pct : float
        Sand and clay content as percentages (0–100).

    Returns
    -------
    str
        One of the twelve USDA classes, or ``"unknown"`` if the inputs are not
        a valid composition.
    """
    import numpy as np

    if not (math.isfinite(sand_pct) and math.isfinite(clay_pct)):
        return "unknown"
    if sand_pct < 0 or clay_pct < 0 or (sand_pct + clay_pct) > 100.0:
        logger.debug("Invalid composition: sand=%.1f clay=%.1f", sand_pct, clay_pct)
        return "unknown"

    code = find_textural_class_scalar(np.array([[float(sand_pct)]]),
                                      np.array([[float(clay_pct)]]))
    return TEXTURE_CLASSES.get(int(code), "unknown")


def find_textural_class_scalar(sand_arr, clay_arr):  # noqa: ANN001
    """Classify a 1×1 array and return the scalar code (internal helper)."""
    from ag_cube_cm.transform.soil_cube import find_soil_textural_class_in_nparray

    return find_soil_textural_class_in_nparray(sand_arr, clay_arr)[0][0]


def dssat_texture_code(sand_pct: float, clay_pct: float) -> str:
    """Return DSSAT's abbreviated texture code (``S``, ``L``, ``SCL``, …)."""
    return DSSAT_TEXTURE_CODES.get(texture_class(sand_pct, clay_pct), "-99")


# ---------------------------------------------------------------------------
# Hydraulic properties
# ---------------------------------------------------------------------------

def saturated_conductivity(
    theta_sat: float, theta_33: float, theta_1500: float
) -> float:
    """Saturated hydraulic conductivity (SSKS) in cm h-1.

    Saxton & Rawls (2006) Eq. 16, computed from the retention points already
    written as SSAT / SDUL / SLLL so the profile stays self-consistent::

        B      = (ln 1500 - ln 33) / (ln θ33 - ln θ1500)
        λ      = 1 / B
        Ks     = 1930 (θs - θ33) ^ (3 - λ)      [mm h-1]

    Parameters
    ----------
    theta_sat : float
        Saturated water content, cm3 cm-3 (SSAT).
    theta_33 : float
        Drained upper limit at 33 kPa, cm3 cm-3 (SDUL).
    theta_1500 : float
        Lower limit at 1500 kPa, cm3 cm-3 (SLLL).

    Returns
    -------
    float
        Conductivity in cm h-1, or ``-99.0`` when the inputs are not a usable
        retention curve (DSSAT's missing-value sentinel).

    Examples
    --------
    >>> round(saturated_conductivity(0.42, 0.28, 0.14), 3)   # loam
    0.757
    >>> round(saturated_conductivity(0.38, 0.12, 0.05), 3)   # sand
    4.62
    """
    values = (theta_sat, theta_33, theta_1500)
    if not all(math.isfinite(v) for v in values):
        return -99.0
    # The curve must be strictly ordered and positive, or the logs below are
    # undefined and the exponent meaningless.
    if not (0.0 < theta_1500 < theta_33 < theta_sat < 1.0):
        logger.debug(
            "Unusable retention curve: SLLL=%.3f SDUL=%.3f SSAT=%.3f",
            theta_1500, theta_33, theta_sat,
        )
        return -99.0

    b = (math.log(1500.0) - math.log(33.0)) / (
        math.log(theta_33) - math.log(theta_1500)
    )
    lambda_ = 1.0 / b
    ks_mm_h = 1930.0 * (theta_sat - theta_33) ** (3.0 - lambda_)
    return round(ks_mm_h / 10.0, 3)  # mm h-1 -> cm h-1


def stage1_evaporation_limit(clay_pct: float, sand_pct: float) -> float:
    """Stage-1 soil evaporation limit (SLU1) in mm.

    The depth of water evaporable before the soil switches to the slower,
    diffusion-limited stage. Ritchie's texture relation as used by the DSSAT
    soil generator.

    Parameters
    ----------
    clay_pct, sand_pct : float
        Clay and sand content as percentages (0–100).

    Returns
    -------
    float
        Stage-1 limit in mm; falls back to 6.0 for an invalid composition.
    """
    if not (math.isfinite(clay_pct) and math.isfinite(sand_pct)):
        return 6.0
    if not (0.0 <= clay_pct <= 100.0 and 0.0 <= sand_pct <= 100.0):
        logger.debug("SLU1: composition out of range, using default")
        return 6.0

    if sand_pct >= 80.0:
        limit = 20.0 - 0.15 * sand_pct
    elif clay_pct >= 50.0:
        limit = 11.0 - 0.06 * clay_pct
    else:
        limit = 8.0 - 0.08 * clay_pct
    return round(max(limit, 1.0), 2)


def drainage_rate(sand_pct: float, clay_pct: float) -> float:
    """Whole-profile drainage rate (SLDR), fraction per day.

    Fraction of the water above field capacity that drains each day. Taken
    from the texture class rather than a continuous function — DSSAT's own
    soil generator treats it as a categorical property.
    """
    return _DRAINAGE_BY_TEXTURE.get(texture_class(sand_pct, clay_pct), _DEFAULT_DRAINAGE)


def runoff_curve_number(sand_pct: float, clay_pct: float) -> float:
    """SCS runoff curve number (SLRO) for row crops in fair condition."""
    return _CURVE_NUMBER_BY_TEXTURE.get(
        texture_class(sand_pct, clay_pct), _DEFAULT_CURVE_NUMBER
    )


def albedo(soc_pct: float) -> float:
    """Bare-soil albedo (SALB), fraction.

    Organic matter darkens soil, so albedo falls as topsoil carbon rises. The
    range spans light mineral soils (0.18) to dark organic topsoils (0.09),
    the span DSSAT's standard profiles use.

    Parameters
    ----------
    soc_pct : float
        Topsoil organic carbon, %.

    Returns
    -------
    float
        Albedo as a fraction.
    """
    if not math.isfinite(soc_pct) or soc_pct < 0:
        return 0.13
    # Linear between 0 % and 4 % carbon, then flat — beyond ~4 % further
    # carbon darkens the soil very little.
    fraction = min(soc_pct, 4.0) / 4.0
    return round(0.18 - 0.09 * fraction, 3)


def root_growth_factors(depths: list[int]) -> list[float]:
    """Root growth factor (SRGF) per layer.

    Thin alias over :func:`ag_cube_cm.transform.soil_cube.calculate_rgf` so the
    DSSAT writer depends on a named soil concept rather than a cube helper.
    1.0 above 15 cm, then ``exp(-0.02 × layer centre)``.
    """
    return calculate_rgf(depths)


# ---------------------------------------------------------------------------
# Initial conditions
# ---------------------------------------------------------------------------

def initial_mineral_nitrogen(
    depths: list[int],
    total_n_pct: list[float] | None = None,
    topsoil_no3_ppm: float = 8.0,
    nh4_fraction: float = 0.2,
) -> list[tuple[float, float]]:
    """Initial ammonium and nitrate per layer, in ppm (ICNH4 / ICNO3).

    .. warning::
       Absolute pre-season mineral N cannot be derived from SoilGrids alone —
       it depends on residue, prior crop and recent rainfall, none of which are
       in the datacube. What *is* well founded is the **shape**: mineral N
       concentrates in the topsoil and declines with depth, and soils richer in
       total N mineralise more.

       So the depth profile and the between-soil contrast are derived, while the
       overall level is anchored on ``topsoil_no3_ppm`` — a configurable
       assumption, not a measurement. For a fertiliser-response study, measure
       it and set it explicitly.

    Parameters
    ----------
    depths : list[int]
        Layer bottom depths in cm.
    total_n_pct : list[float] | None
        Total soil nitrogen per layer, %. When given, layers are scaled by
        their N content relative to the topsoil. When ``None``, only the depth
        decline is applied.
    topsoil_no3_ppm : float
        Nitrate concentration assumed for the surface layer. Default 8.0 ppm,
        typical of an unfertilised tropical cropping soil at season start.
    nh4_fraction : float
        Ammonium as a fraction of nitrate. Default 0.2 — in aerobic soils
        nitrification keeps ammonium the minor pool.

    Returns
    -------
    list[tuple[float, float]]
        ``(nh4_ppm, no3_ppm)`` per layer, aligned with *depths*.
    """
    if not depths:
        return []

    centres = _layer_centres(depths)
    # Mineral N declines faster than roots do; e-folding over ~30 cm matches
    # the topsoil concentration typically observed in profile sampling.
    decline = [math.exp(-c / 30.0) for c in centres]

    scale = [1.0] * len(depths)
    if total_n_pct and len(total_n_pct) == len(depths):
        top = total_n_pct[0]
        if math.isfinite(top) and top > 0:
            scale = [
                (n / top if math.isfinite(n) and n > 0 else 0.0)
                for n in total_n_pct
            ]

    out: list[tuple[float, float]] = []
    for dec, sc in zip(decline, scale):
        no3 = max(topsoil_no3_ppm * dec * sc, 0.1)
        out.append((round(no3 * nh4_fraction, 2), round(no3, 2)))
    return out


def _layer_centres(depths: list[int]) -> list[float]:
    """Mid-depth of each layer, given its bottom depths."""
    centres: list[float] = []
    previous = 0.0
    for bottom in depths:
        centres.append((previous + float(bottom)) / 2.0)
        previous = float(bottom)
    return centres

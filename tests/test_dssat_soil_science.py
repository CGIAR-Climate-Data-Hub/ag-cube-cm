"""Agronomic correctness tests for the DSSAT soil profile.

Unlike the golden tests, these assert what the *model* requires, not what the
code currently emits.  Each one states a soil-physics expectation and cites the
reason, so a future change that re-breaks it fails loudly.

Written before the Phase 3 fix, so several start red by design — that is the
point: they define the target.
"""

from __future__ import annotations

import numpy as np
import pytest

from ag_cube_cm.transform.soil_cube import (
    TEXTURE_CLASSES,
    calculate_rgf,
    find_soil_textural_class_in_nparray,
)


# ---------------------------------------------------------------------------
# Root growth factor — already implemented, not yet wired into the writer
# ---------------------------------------------------------------------------

def test_rgf_decays_with_depth():
    """Roots exploit deep layers less than shallow ones (DSSAT SRGF)."""
    rgf = calculate_rgf([5, 15, 30, 60, 100])
    assert rgf == sorted(rgf, reverse=True), f"RGF must be non-increasing: {rgf}"
    assert rgf[0] == pytest.approx(1.0), "topsoil should be unrestricted"
    assert rgf[-1] < 0.3, (
        f"RGF at 100 cm was {rgf[-1]:.3f}; DSSATTools gives exp(-0.02*80)≈0.20"
    )


def test_rgf_matches_dssattools_formula():
    """Must equal 1.0 above 15 cm and exp(-0.02 * layer centre) below."""
    depths = [5, 15, 30, 60, 100]
    centres = [2.5, 10.0, 22.5, 45.0, 80.0]
    expected = [1.0 if c <= 15 else float(np.exp(-0.02 * c)) for c in centres]
    assert calculate_rgf(depths) == pytest.approx(expected)


def test_rgf_is_bounded():
    """SRGF is a fraction; DSSAT rejects values outside 0–1."""
    for rgf in calculate_rgf([5, 15, 30, 60, 100, 200]):
        assert 0.0 <= rgf <= 1.0


# ---------------------------------------------------------------------------
# Texture classification
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "sand,clay,expected",
    [
        (92.0, 3.0, "sand"),
        (40.0, 22.0, "loam"),
        (55.0, 25.0, "sandy clay loam"),
        (15.0, 55.0, "clay"),
        (10.0, 10.0, "silt"),      # silt = 80 % -> 'silt', not 'silt loam'
        (20.0, 10.0, "silt loam"),  # silt = 70 % -> 'silt loam'
    ],
)
def test_texture_classification_usda(sand, clay, expected):
    """USDA texture triangle, per Benham & Ahrens (2009)."""
    code = find_soil_textural_class_in_nparray(
        np.array([[sand]]), np.array([[clay]])
    )[0][0]
    assert TEXTURE_CLASSES[int(code)] == expected


# ---------------------------------------------------------------------------
# Profile written for DSSAT — the Phase 3 target
# ---------------------------------------------------------------------------

def _layers(model):
    """Return the parsed layer rows of the written TR.SOL."""
    lines = (model.working_dir / "TR.SOL").read_text().splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.startswith("@  SLB"))
    return [ln.split() for ln in lines[start + 1:] if ln.strip()]


def _profile_header(model):
    """Return the ``*TRAN0001 ... <texture> <depth>`` profile line."""
    lines = (model.working_dir / "TR.SOL").read_text().splitlines()
    return next(ln for ln in lines if ln.startswith("*TRAN"))


def test_sol_layer_fields_align_with_header(prepared_model):
    """Each layer row must have exactly as many fields as the header declares.

    Writing one field too many silently shifts every column after it — the
    original writer emitted 18 values against a 17-column header, which landed
    pH in SLHB (buffer pH) instead of SLHW (water pH).
    """
    lines = (prepared_model.working_dir / "TR.SOL").read_text().splitlines()
    header = next(ln for ln in lines if ln.startswith("@  SLB"))
    n_cols = len(header.replace("@", "").split())

    for row in _layers(prepared_model):
        assert len(row) == n_cols, (
            f"layer row has {len(row)} fields but the header declares {n_cols}; "
            "every column after the extra value is shifted"
        )


def test_sol_ph_written_to_slhw(prepared_model):
    """pH from SoilGrids is pH in water (SLHW), not pH in buffer (SLHB)."""
    lines = (prepared_model.working_dir / "TR.SOL").read_text().splitlines()
    header = next(ln for ln in lines if ln.startswith("@  SLB")).replace("@", "").split()
    idx = header.index("SLHW")

    for row in _layers(prepared_model):
        ph = float(row[idx])
        assert 3.0 < ph < 10.0, (
            f"SLHW = {ph}; expected a plausible soil pH — is the column shifted?"
        )


def test_srgf_written_decaying(prepared_model):
    """SRGF in the .SOL must decay with depth, not sit at 1.000 throughout.

    A constant 1.0 tells DSSAT roots exploit 100 cm as readily as the topsoil,
    inflating water and N uptake and suppressing drought response.
    """
    srgf = [float(row[5]) for row in _layers(prepared_model)]
    assert srgf != [1.0] * len(srgf), (
        "SRGF is constant 1.000 at every depth — roots are unrestricted to the "
        "bottom of the profile"
    )
    assert srgf == sorted(srgf, reverse=True), f"SRGF must be non-increasing: {srgf}"


def test_ssks_is_supplied(prepared_model):
    """SSKS drives drainage; -99 erases the sand/clay contrast."""
    ssks = [float(row[6]) for row in _layers(prepared_model)]
    assert all(v > 0 for v in ssks), f"SSKS missing (-99) in the profile: {ssks}"


def test_profile_texture_matches_layer_composition(prepared_model):
    """The profile's declared texture class must match its own sand/clay."""
    from ag_cube_cm.models.dssat.soil import DSSAT_TEXTURE_CODES

    # '*TRAN0001  <source:25><code:5><depth:>8> <NAME>'
    header = _profile_header(prepared_model)
    declared_code = header.split()[2]

    # Layer columns: SLB SLMH SLLL SDUL SSAT SRGF SSKS SBDM SLOC SLCL SLSI ...
    rows = _layers(prepared_model)
    clay_pct = float(rows[0][9])
    silt_pct = float(rows[0][10])
    sand_pct = 100.0 - clay_pct - silt_pct
    code = find_soil_textural_class_in_nparray(
        np.array([[sand_pct]]), np.array([[clay_pct]])
    )[0][0]
    expected_name = TEXTURE_CLASSES[int(code)]

    assert declared_code == DSSAT_TEXTURE_CODES[expected_name], (
        f"profile declares texture code {declared_code!r} but its topsoil "
        f"(sand {sand_pct:.0f}%, clay {clay_pct:.0f}%) is {expected_name!r} "
        f"({DSSAT_TEXTURE_CODES[expected_name]})"
    )


def test_profile_depth_matches_deepest_layer(prepared_model):
    """The declared profile depth must be the deepest layer, not a constant 100."""
    from tests.conftest import SOIL_DEPTHS

    header = _profile_header(prepared_model)
    assert int(header.split()[3]) == max(SOIL_DEPTHS)


def test_slu1_varies_with_texture(sim_config, weather_slice,
                                  sandy_soil_slice, clay_soil_slice):
    """Stage-1 evaporation limit depends on texture (Ritchie); a constant is wrong."""
    from ag_cube_cm.models.dssat.base import DSSATModel

    def slu1_for(soil):
        model = DSSATModel(sim_config)
        model.setup_working_directory(f"slu1_{id(soil)}")
        try:
            model.prepare_inputs(weather_slice, soil, elevation=600.0)
            lines = (model.working_dir / "TR.SOL").read_text().splitlines()
            idx = next(i for i, ln in enumerate(lines) if ln.startswith("@ SCOM"))
            return float(lines[idx + 1].split()[2])
        finally:
            model.cleanup_working_directory()

    assert slu1_for(sandy_soil_slice) != slu1_for(clay_soil_slice), (
        "SLU1 is identical for a sand and a clay — it is hardcoded"
    )


def test_initial_nitrogen_varies_with_soil(sim_config, weather_slice,
                                           sandy_soil_slice, clay_soil_slice):
    """Initial mineral N should reflect the soil, not be a domain-wide constant.

    For a rainfed, unfertilised run the starting N pool is a first-order control
    on yield, so a single value flattens real spatial variation.
    """
    from ag_cube_cm.models.dssat.base import DSSATModel

    def initial_n(soil):
        model = DSSATModel(sim_config)
        model.setup_working_directory(f"icn_{id(soil)}")
        try:
            model.prepare_inputs(weather_slice, soil, elevation=600.0)
            content = next(model.working_dir.glob("*.MZX")).read_text()
            block = content.split("@C  ICBL  SH2O  SNH4  SNO3")[1]
            rows = [ln.split() for ln in block.splitlines() if ln.strip()][:5]
            return [(float(r[3]), float(r[4])) for r in rows]
        finally:
            model.cleanup_working_directory()

    assert initial_n(sandy_soil_slice) != initial_n(clay_soil_slice), (
        "initial SNH4/SNO3 are identical for a sand and a clay — they are hardcoded"
    )

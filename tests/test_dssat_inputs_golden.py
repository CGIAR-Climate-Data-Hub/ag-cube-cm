"""Golden-file tests for the DSSAT input writers.

These lock the *current* bytes of the ``.WTH``, ``.SOL`` and ``.X`` files so any
change to the soil science shows up as an explicit, reviewable diff rather than
a silent shift in simulated yield.

The fixtures record current behaviour, **not** verified-correct behaviour — some
of what they capture is known to be wrong (see ``test_dssat_soil_science.py``).
Regenerate deliberately, and only alongside a reasoned explanation:

    pytest tests -k golden --update-golden
"""

from __future__ import annotations

import difflib
from pathlib import Path

import pytest

GOLDEN_DIR = Path(__file__).parent / "golden"


def _normalise(text: str) -> str:
    """Strip line-ending differences so fixtures are portable across platforms."""
    return text.replace("\r\n", "\n").replace("\r", "\n")


def assert_matches_golden(name: str, actual: str, request) -> None:
    """Compare *actual* against the stored fixture, or rewrite it on request."""
    GOLDEN_DIR.mkdir(exist_ok=True)
    path = GOLDEN_DIR / name
    actual = _normalise(actual)

    if request.config.getoption("--update-golden"):
        path.write_text(actual, encoding="utf-8")
        pytest.skip(f"rewrote golden fixture {name}")

    if not path.exists():
        path.write_text(actual, encoding="utf-8")
        pytest.skip(f"created missing golden fixture {name} — re-run to assert")

    expected = _normalise(path.read_text(encoding="utf-8"))
    if actual != expected:
        diff = "\n".join(
            difflib.unified_diff(
                expected.splitlines(), actual.splitlines(),
                fromfile=f"{name} (golden)", tofile=f"{name} (actual)",
                lineterm="", n=2,
            )
        )
        pytest.fail(
            f"{name} changed.\n\n{diff}\n\n"
            "If this change is intended, confirm the effect on simulated yield, "
            "then re-run with --update-golden.",
            pytrace=False,
        )


# ---------------------------------------------------------------------------
# Weather
# ---------------------------------------------------------------------------

def test_wth_files_written_per_year(prepared_model):
    """Two years of weather must produce one .WTH per year."""
    written = sorted(p.name for p in prepared_model.working_dir.glob("WTH*.WTH"))
    assert written == ["WTHE2001.WTH", "WTHE2101.WTH"]


def test_wth_golden(prepared_model, request):
    """The 2020 weather file is byte-stable."""
    content = (prepared_model.working_dir / "WTHE2001.WTH").read_text()
    assert_matches_golden("WTHE2001.WTH", content, request)


def test_wth_units_converted(prepared_model):
    """Radiation arrives as J m-2 d-1 and must be written as MJ m-2 d-1."""
    lines = (prepared_model.working_dir / "WTHE2001.WTH").read_text().splitlines()
    data = [ln for ln in lines if ln.strip() and ln[:4].isdigit()]
    assert data, "no data rows found in the .WTH file"

    srad_values = [float(ln.split()[-1]) for ln in data]
    assert all(5.0 < v < 45.0 for v in srad_values), (
        f"solar radiation outside a physical MJ m-2 d-1 range: "
        f"min={min(srad_values)} max={max(srad_values)}"
    )


def test_wth_rows_are_chronological_and_complete(prepared_model):
    """DSSAT force-harvests on a gap, so every day of the year must be present."""
    lines = (prepared_model.working_dir / "WTHE2001.WTH").read_text().splitlines()
    doys = [int(ln.split()[0][4:]) for ln in lines if ln.strip() and ln[:4].isdigit()]
    assert doys == sorted(doys), "weather rows are not in chronological order"
    assert doys == list(range(1, 367)), "2020 is a leap year; expected 366 rows"


# ---------------------------------------------------------------------------
# Soil
# ---------------------------------------------------------------------------

def test_sol_golden(prepared_model, request):
    """The soil profile is byte-stable."""
    content = (prepared_model.working_dir / "TR.SOL").read_text()
    assert_matches_golden("TR.SOL", content, request)


def test_sol_has_one_row_per_depth(prepared_model):
    """Every SoilGrids depth must reach the profile."""
    from tests.conftest import SOIL_DEPTHS

    lines = (prepared_model.working_dir / "TR.SOL").read_text().splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.startswith("@  SLB"))
    layers = [ln for ln in lines[start + 1:] if ln.strip()]
    assert len(layers) == len(SOIL_DEPTHS)
    assert [int(ln.split()[0]) for ln in layers] == SOIL_DEPTHS


def test_sol_water_limits_are_ordered(prepared_model):
    """SLLL < SDUL < SSAT must hold in every layer, or DSSAT's water balance breaks."""
    lines = (prepared_model.working_dir / "TR.SOL").read_text().splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.startswith("@  SLB"))
    for ln in (x for x in lines[start + 1:] if x.strip()):
        f = ln.split()
        slll, sdul, ssat = float(f[2]), float(f[3]), float(f[4])
        assert slll < sdul < ssat, f"water limits out of order in layer: {ln!r}"


# ---------------------------------------------------------------------------
# Experiment file
# ---------------------------------------------------------------------------

def test_xfile_golden(prepared_model, request):
    """The experiment file is byte-stable."""
    xfiles = list(prepared_model.working_dir.glob("*.MZX"))
    assert len(xfiles) == 1, f"expected exactly one .MZX, found {xfiles}"
    assert_matches_golden("EXPS0001.MZX", xfiles[0].read_text(), request)


def test_xfile_smodel_column(prepared_model):
    """SMODEL must start at the column its own header announces.

    DSSAT reads this field by fixed column, not by whitespace. One character of
    drift makes it read MZCER as "ZCER" or CRGRO as "RGRO", and every run dies
    with "Crop code incompatible with model specified" (rc=99) — an error that
    names the crop, never the column, and so reads like a cultivar problem.

    A byte-equal golden alone did not catch this: the golden was generated from
    the buggy writer and froze the defect in place. This asserts the invariant
    instead of the bytes, so regenerating the golden cannot reintroduce it.
    """
    content = next(prepared_model.working_dir.glob("*.MZX")).read_text()
    lines = content.splitlines()
    i = next(i for i, l in enumerate(lines) if l.startswith("@N GENERAL"))
    header, data = lines[i], lines[i + 1]
    col = header.index("SMODEL")
    assert data[col:].split(), f"SMODEL column {col} is blank in {data!r}"
    model = data[col:].split()[0]
    assert model in {"MZCER", "MLCER", "RICER", "SGCER", "CSCER", "CRGRO",
                     "BSCER", "SCCAN", "PTSUB", "CSYCA", "WHAPS"}, (
        f"SMODEL reads {model!r} at column {col} — truncated or misaligned. "
        f"line: {data!r}")


def test_xfile_sections_present(prepared_model):
    """All sections DSSAT requires must be emitted."""
    content = next(prepared_model.working_dir.glob("*.MZX")).read_text()
    for section in (
        "*TREATMENTS", "*CULTIVARS", "*FIELDS", "*INITIAL CONDITIONS",
        "*PLANTING DETAILS", "*FERTILIZERS", "*HARVEST DETAILS",
        "*SIMULATION CONTROLS",
    ):
        assert section in content, f"missing section {section}"


def test_xfile_cultivar_matches_config(prepared_model, sim_config):
    """The configured cultivar must reach the CULTIVARS section."""
    content = next(prepared_model.working_dir.glob("*.MZX")).read_text()
    line = content.split("@C CR INGENO CNAME")[1].splitlines()[1]
    assert sim_config.CROP.cultivar in line


def test_genotype_files_copied(prepared_model):
    """DSSAT resolves genotype files from the run directory.

    These are staged by ``run_simulation`` rather than ``prepare_inputs``, so
    the copy step is invoked directly — the test must not need the binary.
    """
    prepared_model._copy_genotype_files()
    names = {p.suffix for p in prepared_model.working_dir.iterdir()}
    for ext in (".CUL", ".ECO", ".SPE"):
        assert ext in names, f"missing genotype file {ext}"

import tempfile
from pathlib import Path

import numpy as np

from binah import OrcaTDDFTApp
from experimental_parser import ExperimentalParser, ExperimentalScan


class _Status:
    def set(self, _text):
        pass


def _make_app():
    app = OrcaTDDFTApp.__new__(OrcaTDDFTApp)
    app._exp_parser = ExperimentalParser()
    app._status = _Status()
    return app


def _edge_scan(label, shift):
    energy = np.linspace(8300.0 + shift, 8360.0 + shift, 121)
    ref = 1.0 / (1.0 + np.exp(-(energy - (8333.0 + shift)) / 1.2))
    sample = 1.0 / (1.0 + np.exp(-(energy - (8340.0 + shift)) / 1.5))
    return ExperimentalScan(
        label=label,
        source_file=f"{label}.dat",
        energy_ev=energy.copy(),
        mu=sample.copy(),
        e0=0.0,
        is_normalized=False,
        ref_energy_ev=energy.copy(),
        ref_mu=ref.copy(),
        ref_label="I2Detector",
    )


def test_auto_align_i2_batch():
    app = _make_app()
    a = _edge_scan("a", 0.0)
    b = _edge_scan("b", 4.0)
    moved = app._auto_align_i2_references([a, b])
    assert moved == 2
    assert abs(a.energy_ev[0] - b.energy_ev[0]) < 0.2
    assert b.metadata["reference_calibration"]["reference_label"] == "I2Detector"


def test_bioxas_dat_i2_reference_detection():
    text = """# Column.1: energy
# Column.2: I0Detector
# Column.3: I2Detector
# Column.4: NiKa1_InB
# Column.5: NiKa1_OutB
8300\t100\t95\t2\t3
8320\t100\t90\t5\t6
8340\t100\t60\t20\t24
8360\t100\t40\t30\t35
8380\t100\t35\t34\t37
"""
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "toy.dat"
        path.write_text(text, encoding="utf-8")
        scan = ExperimentalParser().parse_dat(str(path), mode="fluorescence", normalize=False)
    assert scan.has_reference()
    assert scan.ref_label == "I2Detector"
    assert len(scan.ref_mu) == len(scan.energy_ev)


if __name__ == "__main__":
    test_auto_align_i2_batch()
    test_bioxas_dat_i2_reference_detection()
    print("self_test_binah_reference: ok")

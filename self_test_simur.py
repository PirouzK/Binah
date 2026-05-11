import math
import tempfile
from pathlib import Path

import numpy as np

import simur


def _close(a, b, tol=1e-8):
    return abs(a - b) <= tol


def test_reorient_test_atoms():
    atoms = [
        ("Ni", 1.0, 2.0, 3.0),
        ("P", 3.0, 2.0, 3.0),
        ("Cl", 1.0, 5.0, 3.0),
        ("C", 1.0, 2.0, 5.0),
    ]
    origin = np.array([1.0, 2.0, 3.0])
    R = np.eye(3)
    moved = simur._apply_reorient(atoms, R, origin)
    assert moved[0] == ("Ni", 0.0, 0.0, 0.0)
    assert moved[1] == ("P", 2.0, 0.0, 0.0)
    assert moved[2] == ("Cl", 0.0, 3.0, 0.0)
    assert moved[3] == ("C", 0.0, 0.0, 2.0)


def test_comparison_ao_vector_with_filter():
    entry = {
        "atoms": {
            "Ni1": {"ao": {"dz2": 0.7714, "s": 0.0470}},
            "P2": {"ao": {"px": 0.0200, "s": 0.0100}},
        }
    }
    all_values = simur.SimurApp._comparison_ao_vector(entry)
    ni_values = simur.SimurApp._comparison_ao_vector(entry, {"Ni1"})
    assert _close(all_values["dz2"], 77.14)
    assert _close(all_values["s"], 5.70)
    assert "px" not in ni_values
    assert _close(ni_values["dz2"], 77.14)


def test_project_payload_shape_without_tk():
    payload = {
        "format": "simur-project",
        "version": 1,
        "files": [{"name": "test", "path": "test.out", "group": "fixtures"}],
        "reorient": {"test": {"R": np.eye(3).tolist(), "T": [1.0, 2.0, 3.0]}},
    }
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "project.simur.json"
        path.write_text(simur.json.dumps(payload), encoding="utf-8")
        loaded = simur.json.loads(path.read_text(encoding="utf-8"))
    assert loaded["format"] == "simur-project"
    assert loaded["reorient"]["test"]["R"][0] == [1.0, 0.0, 0.0]


if __name__ == "__main__":
    test_reorient_test_atoms()
    test_comparison_ao_vector_with_filter()
    test_project_payload_shape_without_tk()
    print("self_test_simur: ok")

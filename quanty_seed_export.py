"""
Utilities for turning empirical L-edge fit results into Quanty-oriented
seed files and interpretation reports.

The goal is not to claim a unique Hamiltonian from an empirical fit.
Instead, this module translates phenomenological fit features
(L3/L2/MLCT/LMCT peaks, edge-step widths, broadening) into:

1. A structured summary of experimentally constrained observables.
2. A physically motivated recommendation for crystal-field vs ligand-field
   modeling in Quanty.
3. A commented Lua scaffold that can be turned into a full Quanty script
   once oxidation state, symmetry, and ligand basis are fixed.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from quanty_true_hamiltonian import build_true_quanty_l_edge_xas_script


ELEMENT_Z = {
    "V": 23,
    "Mn": 25,
    "Fe": 26,
    "Co": 27,
    "Ni": 28,
    "Cu": 29,
}

SUPPORTED_SITE_SYMMETRIES = [
    "Oh",
    "Td",
    "D4h",
    "C4v",
    "D3d",
    "C3v",
    "D2h",
    "C2v",
    "Cs",
    "C2",
    "C1",
    "Custom",
]


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return float(default)


def _mean(values: Iterable[float], default=0.0) -> float:
    vals = [float(v) for v in values]
    return sum(vals) / len(vals) if vals else float(default)


def _weighted_mean(pairs: Iterable[Tuple[float, float]], default=0.0) -> float:
    vals = [(float(v), abs(float(w))) for v, w in pairs]
    wsum = sum(w for _, w in vals)
    if wsum <= 1e-12:
        return float(default)
    return sum(v * w for v, w in vals) / wsum


def _is_l3_manifold_kind(kind: Optional[str]) -> bool:
    return kind in ("L3", "L3_MLCT", "L3_LMCT")


def _is_l2_manifold_kind(kind: Optional[str]) -> bool:
    return kind in ("L2", "L2_MLCT", "L2_LMCT")


def _l2_scale(peaks: Sequence[dict], br: Optional[float]) -> float:
    if br is None:
        return 1.0
    il3 = sum(_safe_float(pk.get("I", 0.0))
              for pk in peaks
              if pk.get("enabled", True) and _is_l3_manifold_kind(pk.get("kind")))
    il2 = sum(_safe_float(pk.get("I", 0.0))
              for pk in peaks
              if pk.get("enabled", True) and _is_l2_manifold_kind(pk.get("kind")))
    return float(br) * il3 / il2 if il2 > 1e-12 else 1.0


def _group_features(features: Sequence[dict]) -> Dict[str, List[dict]]:
    grouped = {
        "MLCT": [],
        "LMCT": [],
        "L3": [],
        "L2": [],
        "L3_MLCT": [],
        "L3_LMCT": [],
        "L2_MLCT": [],
        "L2_LMCT": [],
    }
    for feat in features:
        kind = feat["kind"]
        if kind in grouped:
            grouped[kind].append(feat)
        else:
            grouped.setdefault(kind, []).append(feat)

        if kind in ("L3_MLCT", "L2_MLCT"):
            grouped["MLCT"].append(feat)
        elif kind in ("L3_LMCT", "L2_LMCT"):
            grouped["LMCT"].append(feat)
    for feats in grouped.values():
        feats.sort(key=lambda item: item["center_eV"])
    return grouped


def _strongest(features: Sequence[dict]) -> Optional[dict]:
    if not features:
        return None
    return max(features, key=lambda item: item["effective_intensity"])


def _covalency_bucket(ct_fraction: float) -> Tuple[str, str]:
    if ct_fraction >= 0.25:
        return (
            "strong",
            "Strong empirical charge-transfer weight. A ligand-field / "
            "charge-transfer model should be the default starting point.",
        )
    if ct_fraction >= 0.10:
        return (
            "moderate",
            "Moderate empirical charge-transfer weight. Start with a ligand-"
            "field model if chemistry suggests covalency; otherwise compare "
            "crystal-field and ligand-field fits side by side.",
        )
    return (
        "weak",
        "Charge-transfer peaks are weak relative to the white-line manifold. "
        "An ionic crystal-field model is a reasonable first pass.",
    )


def _normalize_symmetry(symmetry: Optional[str]) -> str:
    sym = (symmetry or "Oh").strip()
    if not sym:
        sym = "Oh"
    if sym.lower() == "custom":
        return "Custom"
    return sym


def _symmetry_class(symmetry: str) -> str:
    s = symmetry.strip().lower()
    if s in {"oh", "td"}:
        return "cubic"
    if s in {"d4h", "c4v", "d2d", "s4"}:
        return "tetragonal"
    if s in {"d3d", "c3v", "d3", "s6"}:
        return "trigonal"
    if s in {"d2h", "c2v", "cs", "c2", "c1", "ci", "custom"}:
        return "low_symmetry"
    return "low_symmetry"


def _symmetry_seed(symmetry: str, covalency_level: str) -> dict:
    sym_class = _symmetry_class(symmetry)

    if sym_class == "cubic":
        return {
            "site_symmetry": symmetry,
            "symmetry_class": sym_class,
            "crystal_field_parameterization": "TenDq",
            "crystal_field_seed": {
                "TenDq_eV": 1.00,
            },
            "crystal_field_ranges": {
                "TenDq_eV": [0.30, 3.50],
            },
            "ligand_field_parameterization": "eg_t2g_hoppings",
            "ligand_field_seed": {
                "Veg_eV": 2.20,
                "Vt2g_eV": 1.10,
            },
            "ligand_field_ranges": {
                "Veg_eV": [1.00, 3.50],
                "Vt2g_eV": [0.50, 2.50],
            },
            "notes": [
                "Cubic symmetry can be parameterized compactly with TenDq.",
                "Keep Veg and Vt2g as the natural ligand-field hopping coordinates.",
            ],
        }

    if sym_class == "tetragonal":
        return {
            "site_symmetry": symmetry,
            "symmetry_class": sym_class,
            "crystal_field_parameterization": "TenDq_Ds_Dt",
            "crystal_field_seed": {
                "TenDq_eV": 1.00,
                "Ds_eV": 0.00,
                "Dt_eV": 0.00,
            },
            "crystal_field_ranges": {
                "TenDq_eV": [0.30, 3.50],
                "Ds_eV": [-0.60, 0.60],
                "Dt_eV": [-0.30, 0.30],
            },
            "ligand_field_parameterization": "orbital_grouped_hoppings",
            "ligand_field_seed": {
                "Veg_eV": 2.20,
                "Vt2g_eV": 1.10,
                "Delta_eV": 3.00 if covalency_level == "weak" else 2.00,
            },
            "ligand_field_ranges": {
                "Delta_eV": [-1.00, 6.00],
                "Veg_eV": [1.00, 3.50],
                "Vt2g_eV": [0.50, 2.50],
            },
            "notes": [
                "Tetragonal symmetry is usually well captured by TenDq, Ds, and Dt.",
                "If the fit still needs extra structure, move to explicit orbital energies.",
            ],
        }

    if sym_class == "trigonal":
        return {
            "site_symmetry": symmetry,
            "symmetry_class": sym_class,
            "crystal_field_parameterization": "trigonal_split",
            "crystal_field_seed": {
                "TenDq_eV": 1.00,
                "Dtrig_eV": 0.00,
            },
            "crystal_field_ranges": {
                "TenDq_eV": [0.30, 3.50],
                "Dtrig_eV": [-0.80, 0.80],
            },
            "ligand_field_parameterization": "trigonal_hoppings",
            "ligand_field_seed": {
                "Delta_eV": 3.00 if covalency_level == "weak" else 2.00,
                "Va1g_eV": 2.20,
                "Veg_pi_eV": 1.10,
            },
            "ligand_field_ranges": {
                "Delta_eV": [-1.00, 6.00],
                "Va1g_eV": [1.00, 3.50],
                "Veg_pi_eV": [0.50, 2.50],
            },
            "notes": [
                "Trigonal symmetry is better handled with a dedicated trigonal splitting than with Ds/Dt alone.",
                "In Quanty this is often most transparent via Akm terms or explicit orbital energies in the rotated frame.",
            ],
        }

    return {
        "site_symmetry": symmetry,
        "symmetry_class": sym_class,
        "crystal_field_parameterization": "orbital_energies_traceless",
        "crystal_field_seed": {
            "E_dz2_eV": 0.00,
            "E_dx2y2_eV": 0.00,
            "E_dxy_eV": 0.00,
            "E_dxz_eV": 0.00,
            "E_dyz_eV": 0.00,
        },
        "crystal_field_constraint": "sum(E_m) = 0",
        "crystal_field_ranges": {
            "E_dz2_eV": [-1.50, 1.50],
            "E_dx2y2_eV": [-1.50, 1.50],
            "E_dxy_eV": [-1.50, 1.50],
            "E_dxz_eV": [-1.50, 1.50],
            "E_dyz_eV": [-1.50, 1.50],
        },
        "ligand_field_parameterization": "orbital_dependent_hoppings",
        "ligand_field_seed": {
            "Delta_eV": 3.00 if covalency_level == "weak" else 2.00,
            "V_dz2_eV": 2.20,
            "V_dx2y2_eV": 2.00,
            "V_dxy_eV": 1.40,
            "V_dxz_eV": 1.10,
            "V_dyz_eV": 1.10,
        },
        "ligand_field_ranges": {
            "Delta_eV": [-1.00, 6.00],
            "V_dz2_eV": [0.50, 3.50],
            "V_dx2y2_eV": [0.50, 3.50],
            "V_dxy_eV": [0.30, 2.50],
            "V_dxz_eV": [0.30, 2.50],
            "V_dyz_eV": [0.30, 2.50],
        },
        "notes": [
            "Low symmetry such as C2v should not be compressed into TenDq/Ds/Dt alone.",
            "Use five real-orbital energies, with the trace constrained to zero, or an equivalent full Akm expansion.",
            "In the ligand-field model, allow orbital-dependent hybridization because covalency need not respect higher-symmetry eg/t2g grouping.",
        ],
    }


def _render_feature_table(features: Sequence[dict]) -> str:
    lines = [
        "| Label | Center (eV) | Offset from E0 (eV) | FWHM (eV) | Eff. intensity | I / IL3_edge |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for feat in features:
        lines.append(
            "| {label} | {center:.4f} | {offset:.4f} | {width:.4f} | {eff:.5g} | {ratio:.5g} |".format(
                label=feat["label"],
                center=feat["center_eV"],
                offset=feat["offset_from_E0_eV"],
                width=feat["fwhm_eV"],
                eff=feat["effective_intensity"],
                ratio=feat["intensity_vs_IL3_edge"],
            )
        )
    return "\n".join(lines)


def _build_interpretation_notes(
    grouped: Dict[str, List[dict]],
    ct_fraction: float,
    covalency_note: str,
) -> List[str]:
    notes = [covalency_note]

    if len(grouped["L3"]) > 1 or len(grouped["L2"]) > 1:
        notes.append(
            "Multiple resolved white-line peaks are present. Treat them as "
            "evidence for multiplet and/or lower-symmetry splitting, not as "
            "standalone transitions with one-to-one Hamiltonian labels."
        )

    if grouped["MLCT"] or grouped["LMCT"]:
        notes.append(
            "Empirical MLCT/LMCT pseudo-Voigt peaks should be interpreted as "
            "markers of ligand-hole character and covalency. They motivate "
            "including charge-transfer configurations, but they do not map "
            "directly onto a single Delta value."
        )

    if ct_fraction < 0.10 and not grouped["LMCT"]:
        notes.append(
            "A crystal-field-only fit is a sensible first pass. Keep a "
            "ligand-field model in reserve if the residuals remain structured "
            "after refining the crystal-field splitting pattern and "
            "Slater-integral reduction."
        )

    notes.append(
        "Use the empirical edge-step widths as initial lifetime and "
        "instrumental broadening seeds in Quanty. They are not Hamiltonian "
        "terms and should remain separate from crystal-field or charge-"
        "transfer parameters."
    )
    return notes


def build_quanty_seed_bundle(
    *,
    element: str,
    zeta_2p: float,
    params: Dict[str, float],
    peaks: Sequence[dict],
    symmetry: str = "Oh",
    br: Optional[float] = None,
    mc_std_d: Optional[Dict[str, float]] = None,
    mc_std_pk: Optional[Sequence[dict]] = None,
    fit_metrics: Optional[Dict[str, float]] = None,
) -> Tuple[dict, str, str]:
    params = dict(params)
    peaks = [dict(pk) for pk in peaks]
    fit_metrics = dict(fit_metrics or {})

    e0 = _safe_float(params.get("E0"))
    tk = e0 + 1.5 * _safe_float(zeta_2p)
    il3_edge = max(_safe_float(params.get("IL3_Edge"), 1.0), 1e-12)
    l2_scale = _l2_scale(peaks, br)

    features = []
    for idx, pk in enumerate(peaks, start=1):
        kind = str(pk.get("kind", "Peak"))
        if not pk.get("enabled", True):
            continue
        eff_i = _safe_float(pk.get("I")) * (l2_scale if _is_l2_manifold_kind(kind) else 1.0)
        sd_pk = mc_std_pk[idx - 1] if mc_std_pk and idx - 1 < len(mc_std_pk) else {}
        features.append({
            "index": idx,
            "label": f"{kind} #{idx}",
            "kind": kind,
            "center_eV": _safe_float(pk.get("o")),
            "offset_from_E0_eV": _safe_float(pk.get("o")) - e0,
            "offset_from_Tk_eV": _safe_float(pk.get("o")) - tk,
            "fwhm_eV": _safe_float(pk.get("W")),
            "effective_intensity": eff_i,
            "raw_intensity": _safe_float(pk.get("I")),
            "intensity_vs_IL3_edge": eff_i / il3_edge,
            "gaussian_fraction": _safe_float(pk.get("G")),
            "center_sd_eV": _safe_float(sd_pk.get("o"), 0.0),
            "width_sd_eV": _safe_float(sd_pk.get("W"), 0.0),
            "intensity_sd": _safe_float(sd_pk.get("I"), 0.0),
        })

    features.sort(key=lambda item: item["center_eV"])
    grouped = _group_features(features)
    main_l3 = _strongest(grouped["L3"])
    main_l2 = _strongest(grouped["L2"])

    ct_intensity = sum(
        feat["effective_intensity"]
        for feat in grouped["MLCT"] + grouped["LMCT"]
    )
    wl_intensity = sum(
        feat["effective_intensity"]
        for feat in grouped["L3"] + grouped["L2"]
    )
    total_peak_intensity = max(ct_intensity + wl_intensity, 1e-12)
    ct_fraction = ct_intensity / total_peak_intensity
    covalency_level, covalency_note = _covalency_bucket(ct_fraction)
    symmetry = _normalize_symmetry(symmetry)
    symmetry_seed = _symmetry_seed(symmetry, covalency_level)

    recommended_family = (
        "ligand_field_charge_transfer"
        if covalency_level in {"moderate", "strong"}
        else "crystal_field"
    )

    weighted_peak_g = _weighted_mean(
        [(feat["gaussian_fraction"], feat["effective_intensity"]) for feat in features],
        default=_mean(
            [_safe_float(params.get("GL3_Edge")), _safe_float(params.get("GL2_Edge"))],
            default=0.0,
        ),
    )

    spectral_constraints = {
        "L3_onset_eV": e0,
        "L2_onset_eV": tk,
        "L3_edge_fwhm_eV": _safe_float(params.get("WL3_Edge")),
        "L2_edge_fwhm_eV": _safe_float(params.get("WL2_Edge")),
        "Gamma_L3_hwhm_seed_eV": 0.5 * _safe_float(params.get("WL3_Edge")),
        "Gamma_L2_hwhm_seed_eV": 0.5 * _safe_float(params.get("WL2_Edge")),
        "Gaussian_fraction_seed": weighted_peak_g,
        "Split_energy_seed_eV": 0.5 * (e0 + tk),
        "main_L3_peak_eV": main_l3["center_eV"] if main_l3 else None,
        "main_L2_peak_eV": main_l2["center_eV"] if main_l2 else None,
    }

    crystal_field_seed = {
        "required_user_inputs": [
            "oxidation_state",
            "n_3d_electrons",
            "site_symmetry",
        ],
        "recommended_start": {
            "Fdd_scale": 0.80,
            "Fpd_scale": 0.80,
            "zeta_3d_scale": 1.00,
            "TenDq_eV": 1.00,
            "Ds_eV": 0.00,
            "Dt_eV": 0.00,
        },
        "suggested_search_ranges": {
            "Fdd_scale": [0.70, 0.90],
            "Fpd_scale": [0.70, 0.90],
            "zeta_3d_scale": [0.70, 1.00],
            "TenDq_eV": [0.30, 3.50],
            "Ds_eV": [-0.60, 0.60],
            "Dt_eV": [-0.30, 0.30],
        },
        "hamiltonian_terms": [
            "H_i = H_atomic(3d^n) + H_cf(10Dq, Ds, Dt) + H_so_3d",
            "H_f = H_atomic(2p^5 3d^(n+1)) + H_cf(10Dq, Ds, Dt) + H_so_2p + H_so_3d",
            "Scale Fdd by Fdd_scale and Fpd/Gpd by Fpd_scale",
        ],
        "symmetry_specific": symmetry_seed,
    }

    ligand_field_seed = {
        "required_user_inputs": [
            "oxidation_state",
            "n_3d_electrons",
            "site_symmetry",
            "ligand_basis_choice",
        ],
        "recommended_start": {
            "Fdd_scale": 0.80,
            "Fpd_scale": 0.80,
            "zeta_3d_scale": 1.00,
            "TenDq_eV": 1.00,
            "Ds_eV": 0.00,
            "Dt_eV": 0.00,
            "Delta_eV": 3.00 if covalency_level == "weak" else 2.00,
            "Udd_eV": 6.50,
            "Upd_eV": 8.00,
            "Veg_eV": 2.20,
            "Vt2g_eV": 1.10,
        },
        "suggested_search_ranges": {
            "Delta_eV": [-1.00, 6.00],
            "Udd_eV": [5.00, 9.00],
            "Upd_eV": [6.50, 10.50],
            "Veg_eV": [1.00, 3.50],
            "Vt2g_eV": [0.50, 2.50],
        },
        "hamiltonian_terms": [
            "Initial basis: 3d^n + 3d^(n+1)L + 3d^(n+2)L^2",
            "Final basis: 2p^5 3d^(n+1) + 2p^5 3d^(n+2)L + 2p^5 3d^(n+3)L^2",
            "H = H_atomic + H_cf + H_so + H_ct(Delta, Veg, Vt2g, Udd, Upd)",
        ],
        "symmetry_specific": symmetry_seed,
    }

    workflow = [
        "Normalize and decompose the raw spectrum in SGM Normalizer.",
        "Use the empirical L3/L2 positions to seed energy alignment and broadening only.",
        "Fix oxidation state, n_3d, and site symmetry from chemistry before building Quanty Hamiltonians.",
        "Start with broadening and energy shift fixed, then fit Fdd/Fpd scaling and the dominant crystal-field coordinates for the selected symmetry.",
        "Only then unlock lower-symmetry crystal-field coordinates appropriate to the selected site symmetry.",
        "If MLCT/LMCT intensity is non-negligible, switch to the ligand-field / charge-transfer basis and fit Delta together with the symmetry-appropriate hopping coordinates.",
        "Treat empirical MLCT/LMCT peaks as covalency indicators, not as direct one-parameter surrogates.",
    ]

    mapping_table = [
        {
            "empirical_feature": "L3 / L2 peak manifold",
            "quanty_role": "Multiplet and crystal-field sensitive final-state structure",
            "fit_terms": ["10Dq", "Ds", "Dt", "orbital energies", "Fdd_scale", "Fpd_scale", "zeta_3d_scale"],
        },
        {
            "empirical_feature": "MLCT / LMCT peaks",
            "quanty_role": "Evidence for ligand-hole weight and covalency",
            "fit_terms": ["Delta", "Veg/Vt2g", "orbital-dependent hoppings", "Udd", "Upd"],
        },
        {
            "empirical_feature": "WL3_Edge / WL2_Edge",
            "quanty_role": "Seed core-hole lifetime broadening",
            "fit_terms": ["Gamma_L3", "Gamma_L2"],
        },
        {
            "empirical_feature": "Peak / edge Gaussian fraction",
            "quanty_role": "Seed instrumental convolution",
            "fit_terms": ["Gaussian_fraction"],
        },
    ]

    bundle = {
        "metadata": {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "source": "ledge_normalizer.py",
            "element": element,
            "atomic_number": ELEMENT_Z.get(element),
            "zeta_2p_eV": _safe_float(zeta_2p),
            "site_symmetry": symmetry,
            "symmetry_class": symmetry_seed["symmetry_class"],
        },
        "fit_parameters": {
            "static": params,
            "static_std_dev": dict(mc_std_d or {}),
            "peaks": features,
            "L2_to_L3_branching_ratio_constraint": br,
        },
        "fit_metrics": fit_metrics,
        "empirical_observables": spectral_constraints,
        "physical_interpretation": {
            "charge_transfer_fraction": ct_fraction,
            "covalency_level": covalency_level,
            "recommended_model_family": recommended_family,
            "notes": _build_interpretation_notes(grouped, ct_fraction, covalency_note),
        },
        "empirical_to_hamiltonian_mapping": mapping_table,
        "model_seeds": {
            "crystal_field": crystal_field_seed,
            "ligand_field_charge_transfer": ligand_field_seed,
        },
        "workflow": workflow,
    }

    report = _render_markdown_report(bundle)
    lua = _render_quanty_lua(bundle)
    return bundle, report, lua


def _render_markdown_report(bundle: dict) -> str:
    meta = bundle["metadata"]
    fit = bundle["fit_parameters"]
    obs = bundle["empirical_observables"]
    phys = bundle["physical_interpretation"]
    features = fit["peaks"]
    symmetry_seed = bundle["model_seeds"]["crystal_field"]["symmetry_specific"]

    lines = [
        "# Quanty Seed Report",
        "",
        f"Generated from the empirical L-edge fit for **{meta['element']}**.",
        "",
        "## Symmetry",
        "",
        f"- Selected site symmetry: `{meta['site_symmetry']}`",
        f"- Symmetry class: `{meta['symmetry_class']}`",
        f"- Recommended crystal-field parameterization: `{symmetry_seed['crystal_field_parameterization']}`",
        "",
        "## Empirical Constraints",
        "",
        f"- L3 onset E0: `{obs['L3_onset_eV']:.4f}` eV",
        f"- L2 onset Tk: `{obs['L2_onset_eV']:.4f}` eV",
        f"- L3 broadening seed (hwhm): `{obs['Gamma_L3_hwhm_seed_eV']:.4f}` eV",
        f"- L2 broadening seed (hwhm): `{obs['Gamma_L2_hwhm_seed_eV']:.4f}` eV",
        f"- Gaussian fraction seed: `{obs['Gaussian_fraction_seed']:.4f}`",
        "",
        "## Peak Features",
        "",
        _render_feature_table(features),
        "",
        "## Physical Reading",
        "",
        f"- Recommended model family: `{phys['recommended_model_family']}`",
        f"- Empirical charge-transfer fraction: `{phys['charge_transfer_fraction']:.4f}`",
        f"- Covalency level: `{phys['covalency_level']}`",
    ]

    for note in phys["notes"]:
        lines.append(f"- {note}")

    lines.extend([
        "",
        "## Symmetry-Specific Seed",
        "",
        f"- Crystal-field parameterization: `{symmetry_seed['crystal_field_parameterization']}`",
    ])
    if "crystal_field_constraint" in symmetry_seed:
        lines.append(f"- Constraint: `{symmetry_seed['crystal_field_constraint']}`")
    lines.append("- Crystal-field start values:")
    for key, value in symmetry_seed["crystal_field_seed"].items():
        lines.append(f"  {key}: `{value:.4f}` eV")
    lines.append("- Ligand-field start values:")
    for key, value in symmetry_seed["ligand_field_seed"].items():
        lines.append(f"  {key}: `{value:.4f}` eV")
    for note in symmetry_seed.get("notes", []):
        lines.append(f"- {note}")

    lines.extend([
        "",
        "## Mapping Into Quanty",
        "",
        "| Empirical feature | Quanty meaning | Main terms |",
        "|---|---|---|",
    ])
    for row in bundle["empirical_to_hamiltonian_mapping"]:
        lines.append(
            "| {emp} | {meaning} | `{terms}` |".format(
                emp=row["empirical_feature"],
                meaning=row["quanty_role"],
                terms=", ".join(row["fit_terms"]),
            )
        )

    lines.extend([
        "",
        "## Suggested Workflow",
        "",
    ])
    for step in bundle["workflow"]:
        lines.append(f"- {step}")

    lines.extend([
        "",
        "## Seed Hamiltonians",
        "",
        "Crystal-field starting point:",
        "",
        "- `H_i = H_atomic(3d^n) + H_cf + H_so_3d`",
        "- `H_f = H_atomic(2p^5 3d^(n+1)) + H_cf + H_so_2p + H_so_3d`",
        "- Scale Slater integrals with `Fdd_scale` and `Fpd_scale`.",
        "",
        "Ligand-field / charge-transfer starting point:",
        "",
        "- Initial basis: `3d^n + 3d^(n+1)L + 3d^(n+2)L^2`",
        "- Final basis: `2p^5 3d^(n+1) + 2p^5 3d^(n+2)L + 2p^5 3d^(n+3)L^2`",
        "- `H = H_atomic + H_cf + H_so + H_ct(Delta, Veg, Vt2g, Udd, Upd)`",
        "",
        "Generated Lua files:",
        "",
        "- `_quanty.lua`: symbolic scaffold and parameter handoff.",
        "- `_quanty_xas.lua`: runnable Quanty L-edge XAS script with explicit `H_i` and `H_f`.",
        "",
        "Both Lua files still need chemistry-specific choices for oxidation state,",
        "`n_3d`, symmetry axes, and ligand basis.",
    ])
    return "\n".join(lines) + "\n"


def _render_quanty_lua(bundle: dict) -> str:
    meta = bundle["metadata"]
    obs = bundle["empirical_observables"]
    phys = bundle["physical_interpretation"]
    cf = bundle["model_seeds"]["crystal_field"]
    lf = bundle["model_seeds"]["ligand_field_charge_transfer"]
    symmetry_seed = cf["symmetry_specific"]

    cf_seed_lines = []
    for key, value in symmetry_seed["crystal_field_seed"].items():
        cf_seed_lines.append(f"local {key} = {value:.6f}")
    lf_seed_lines = []
    for key, value in symmetry_seed["ligand_field_seed"].items():
        lf_seed_lines.append(f"local {key} = {value:.6f}")
    cf_seed_block = "\n".join(cf_seed_lines)
    lf_seed_block = "\n".join(lf_seed_lines)
    cf_constraint = symmetry_seed.get("crystal_field_constraint")
    cf_constraint_comment = (
        f"-- Constraint: {cf_constraint}\n" if cf_constraint else ""
    )

    return f"""-- Quanty seed scaffold generated from ledge_normalizer.py
-- This file is a starting point only. It captures empirical constraints
-- from the SGM Normalizer fit and maps them onto a physically meaningful
-- Hamiltonian structure for L-edge spectroscopy.

local Empirical = {{
    Element = "{meta['element']}",
    AtomicNumber = {meta['atomic_number'] if meta['atomic_number'] is not None else 'nil'},
    Zeta2p = {meta['zeta_2p_eV']:.6f},
    E0 = {obs['L3_onset_eV']:.6f},
    Tk = {obs['L2_onset_eV']:.6f},
    GammaL3 = {obs['Gamma_L3_hwhm_seed_eV']:.6f},
    GammaL2 = {obs['Gamma_L2_hwhm_seed_eV']:.6f},
    GaussianFraction = {obs['Gaussian_fraction_seed']:.6f},
    SplitEnergy = {obs['Split_energy_seed_eV']:.6f},
}}

local SuggestedModel = {{
    Family = "{phys['recommended_model_family']}",
    CovalencyLevel = "{phys['covalency_level']}",
    ChargeTransferFraction = {phys['charge_transfer_fraction']:.6f},
}}

-- User chemistry still required before the script becomes a full Quanty model.
local N3d = 8         -- TODO set from oxidation state / chemistry
local Symmetry = "{meta['site_symmetry']}"
local IncludeChargeTransfer = {"true" if phys['recommended_model_family'] != 'crystal_field' else "false"}
local SymmetryClass = "{meta['symmetry_class']}"

-- Crystal-field seed values.
local FddScale = {cf['recommended_start']['Fdd_scale']:.6f}
local FpdScale = {cf['recommended_start']['Fpd_scale']:.6f}
local Zeta3dScale = {cf['recommended_start']['zeta_3d_scale']:.6f}
local TenDq = {cf['recommended_start']['TenDq_eV']:.6f}
local Ds = {cf['recommended_start']['Ds_eV']:.6f}
local Dt = {cf['recommended_start']['Dt_eV']:.6f}
-- Symmetry-specific crystal-field coordinates.
-- Parameterization: {symmetry_seed['crystal_field_parameterization']}
{cf_constraint_comment}{cf_seed_block}

-- Ligand-field / charge-transfer seed values.
local Delta = {lf['recommended_start']['Delta_eV']:.6f}
local Udd = {lf['recommended_start']['Udd_eV']:.6f}
local Upd = {lf['recommended_start']['Upd_eV']:.6f}
local Veg = {lf['recommended_start']['Veg_eV']:.6f}
local Vt2g = {lf['recommended_start']['Vt2g_eV']:.6f}
-- Symmetry-specific ligand-field coordinates.
-- Parameterization: {symmetry_seed['ligand_field_parameterization']}
{lf_seed_block}

-- Symbolic Hamiltonians to implement with Quanty operators:
-- Crystal-field model:
--   H_i = H_atomic(3d^n) + H_cf + H_so_3d
--   H_f = H_atomic(2p^5 3d^(n+1)) + H_cf + H_so_2p + H_so_3d
--   For cubic/tetragonal cases, H_cf can be expressed with TenDq / Ds / Dt.
--   For C2v and lower, prefer a full orbital-energy or Akm expansion.
--
-- Ligand-field / charge-transfer model:
--   Initial basis = 3d^n + 3d^(n+1)L + 3d^(n+2)L^2
--   Final basis   = 2p^5 3d^(n+1) + 2p^5 3d^(n+2)L + 2p^5 3d^(n+3)L^2
--   H = H_atomic + H_cf + H_so + H_ct
--   For low symmetry, let H_ct carry orbital-dependent hybridization terms.
--
-- Practical fitting order:
--   1. Fix N3d and Symmetry from chemistry.
--   2. Use Empirical.GammaL3 / GammaL2 / GaussianFraction for the initial broadening.
--   3. Fit energy shift and amplitude.
--   4. Fit FddScale, FpdScale, and the dominant crystal-field coordinates.
--   5. Only then unlock the lower-symmetry coordinates required by Symmetry.
--   6. If IncludeChargeTransfer, add Delta and the symmetry-specific hopping terms.
--
-- The empirical MLCT / LMCT peaks should be treated as covalency markers,
-- not as one-to-one Hamiltonian terms.

print("Quanty seed scaffold loaded for " .. Empirical.Element)
"""


def write_quanty_seed_bundle(
    base_path: str,
    *,
    element: str,
    zeta_2p: float,
    params: Dict[str, float],
    peaks: Sequence[dict],
    symmetry: str = "Oh",
    br: Optional[float] = None,
    mc_std_d: Optional[Dict[str, float]] = None,
    mc_std_pk: Optional[Sequence[dict]] = None,
    fit_metrics: Optional[Dict[str, float]] = None,
) -> List[str]:
    stem, ext = os.path.splitext(base_path)
    if not stem:
        stem = base_path
    json_path = stem + ".json"
    report_path = stem + "_report.md"
    lua_path = stem + "_quanty.lua"
    xas_lua_path = stem + "_quanty_xas.lua"

    bundle, report, lua = build_quanty_seed_bundle(
        element=element,
        zeta_2p=zeta_2p,
        params=params,
        peaks=peaks,
        symmetry=symmetry,
        br=br,
        mc_std_d=mc_std_d,
        mc_std_pk=mc_std_pk,
        fit_metrics=fit_metrics,
    )
    xas_lua = build_true_quanty_l_edge_xas_script(bundle)

    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(bundle, fh, indent=2, sort_keys=False)
        fh.write("\n")

    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write(report)

    with open(lua_path, "w", encoding="utf-8") as fh:
        fh.write(lua)

    with open(xas_lua_path, "w", encoding="utf-8") as fh:
        fh.write(xas_lua)

    return [json_path, report_path, lua_path, xas_lua_path]

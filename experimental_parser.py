"""
Experimental XAS Scan Parser
Supports:
  .prj  — Athena/Demeter project files (gzip-compressed Perl serialization)
           @y arrays are already normalized mu(E) — used directly.
  .dat  — BioXAS XDI format (136+ column tab-separated raw counts)
           mu(E) computed from fluorescence (InB+OutB)/I0 or transmission ln(I0/I1)
           then normalized via pre/post edge fitting.
  .csv / .txt / generic two-column — plain energy, mu(E) columns.
"""

import gzip
import re
import os
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

# ── Optional Larch integration ────────────────────────────────────────────────
# xraylarch 2026.1.2 is installed; use its pre_edge() for the same normalization
# algorithm as Athena.  Falls back to our own polynomial implementation if larch
# is unavailable or throws on a particular scan.
_LARCH_SESSION = None
_LARCH_OK: Optional[bool] = None   # None = not yet tested

def _get_larch_session():
    global _LARCH_SESSION, _LARCH_OK
    if _LARCH_OK is None:
        try:
            from larch import Interpreter
            _LARCH_SESSION = Interpreter()
            _LARCH_OK = True
        except Exception:
            _LARCH_OK = False
    return _LARCH_SESSION if _LARCH_OK else None


# ═══════════════════════════════════════════════════════════════════════════════
@dataclass
class ExperimentalScan:
    label: str
    source_file: str
    energy_ev: np.ndarray
    mu: np.ndarray           # normalized μ(E), dimensionless (0→~1.5 for XANES)
    e0: float = 0.0          # edge energy in eV
    is_normalized: bool = True
    scan_type: str = ""      # "fluorescence", "transmission", "pre-normalized", "raw"
    metadata: Dict = field(default_factory=dict)
    # Simultaneously-measured reference channel (e.g. I₂ gas cell, diode).
    # None when the data file carries no reference signal.
    ref_energy_ev: np.ndarray = field(default=None)
    ref_mu: np.ndarray = field(default=None)
    ref_label: str = ""      # e.g. "I2", "PD", "DiodeDetector"

    def display_name(self) -> str:
        return self.label or os.path.basename(self.source_file)

    def has_reference(self) -> bool:
        return self.ref_mu is not None and len(self.ref_mu) > 0


# ═══════════════════════════════════════════════════════════════════════════════
class ExperimentalParser:

    # ── .prj (Athena Demeter) ─────────────────────────────────────────────────
    def parse_prj(self, filepath: str) -> List[ExperimentalScan]:
        """
        Parse a gzip-compressed Athena .prj file.
        Each $old_group block becomes one ExperimentalScan.
        @y is already normalized mu(E); @x is energy in eV.
        """
        with open(filepath, "rb") as f:
            raw = gzip.decompress(f.read())
        text = raw.decode("latin-1")

        scans: List[ExperimentalScan] = []
        blocks = re.split(r"\$old_group\s*=\s*'(\w+)'\s*;", text)

        # blocks alternates: [pre-text, group_name, block_text, group_name, ...]
        i = 1
        while i + 1 < len(blocks):
            group_name = blocks[i]
            block_text = blocks[i + 1]
            i += 2

            # Parse @args flat key-value list
            args_match = re.search(r"@args\s*=\s*\((.+?)\)\s*;", block_text, re.DOTALL)
            meta: Dict = {}
            if args_match:
                meta = self._parse_perl_kvlist(args_match.group(1))

            # Skip merge/reference/fit groups that aren't real scans
            if meta.get("is_fit") == "1" or meta.get("unreadable") == "1":
                continue

            # Parse @x and @y arrays
            x_arr = self._parse_perl_array(block_text, "x")
            y_arr = self._parse_perl_array(block_text, "y")

            if x_arr is None or y_arr is None or len(x_arr) == 0:
                continue

            energy = np.array(x_arr, dtype=float)
            mu     = np.array(y_arr, dtype=float)

            # Trim to same length (occasionally off by 1)
            n = min(len(energy), len(mu))
            energy, mu = energy[:n], mu[:n]

            label   = meta.get("label", group_name).strip()
            e0_str  = meta.get("bkg_e0", "0")
            try:
                e0 = float(e0_str)
            except ValueError:
                e0 = float(energy[np.argmax(np.gradient(mu))]) if len(mu) > 2 else 0.0

            # Apply Athena-style edge-step normalization using the stored @args
            # parameters so all scans land on the same 0→1 scale regardless of
            # their original intensity (concentration, beamline mode, etc.)
            try:
                pre1  = float(meta.get("bkg_pre1",  "-150"))
                pre2  = float(meta.get("bkg_pre2",  "-30"))
                nor1  = float(meta.get("bkg_nor1",  "150"))
                nor2  = float(meta.get("bkg_nor2",  "400"))
                nnorm = int(float(meta.get("bkg_nnorm", "2")))
                nnorm = max(1, min(3, nnorm))
            except (ValueError, TypeError):
                pre1, pre2, nor1, nor2, nnorm = -150.0, -30.0, 150.0, 400.0, 2

            if e0 > 0 and len(energy) > 5:
                mu, _ = self._normalize(
                    energy, mu, e0,
                    pre_range=(pre1, pre2),
                    post_range=(nor1, nor2),
                    nnorm=nnorm,
                )

            scan = ExperimentalScan(
                label=label,
                source_file=filepath,
                energy_ev=energy,
                mu=mu,
                e0=e0,
                is_normalized=True,
                scan_type="normalized",
                metadata=meta,
            )
            scans.append(scan)

        return scans

    # ── .dat (BioXAS XDI format) ──────────────────────────────────────────────
    def parse_dat(
        self,
        filepath: str,
        mode: str = "fluorescence",   # "fluorescence" | "transmission"
        normalize: bool = True,
    ) -> ExperimentalScan:
        """
        Parse a BioXAS XDI .dat file.
        Reads column headers from # Column.N: name lines.
        Computes mu(E) from fluorescence (NiKa1_InB + NiKa1_OutB) / I0
        or transmission ln(I0/I1), then optionally normalizes.
        """
        with open(filepath, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()

        # ── Parse column map from header ──────────────────────────────────────
        col_map: Dict[int, str] = {}   # 1-based index → name
        scan_label = os.path.basename(filepath)
        element = ""
        edge    = ""
        for line in lines:
            s = line.strip()
            if not s.startswith("#"):
                break
            m = re.match(r"#\s*Column\.(\d+):\s*(.+)", s)
            if m:
                col_map[int(m.group(1))] = m.group(2).strip()
            m2 = re.match(r"#\s*Scan:\s*(.+)", s)
            if m2:
                scan_label = m2.group(1).strip()
            m3 = re.match(r"#\s*Element\.symbol:\s*(.+)", s)
            if m3:
                element = m3.group(1).strip()
            m4 = re.match(r"#\s*Element\.edge:\s*(.+)", s)
            if m4:
                edge = m4.group(1).strip()

        if element and edge:
            scan_label = f"{scan_label} ({element} {edge})"

        # ── Find signal columns ───────────────────────────────────────────────
        energy_col = self._find_col(col_map, ["energy", "eV"], required=True)
        i0_col     = self._find_col(col_map, ["I0", "I0Detector"])
        i1_col     = self._find_col(col_map, ["I1", "I1Detector"])
        inb_col    = self._find_col(col_map, ["InB_DarkCorrect", "NiKa1_InB"])
        outb_col   = self._find_col(col_map, ["OutB_DarkCorrect", "NiKa1_OutB"])

        # Identify ALL inboard spectra columns for summed fluorescence
        inb_spectra_cols  = [k for k, v in col_map.items() if "spectra" in v.lower() and "InB" in v and "ICR" not in v]
        outb_spectra_cols = [k for k, v in col_map.items() if "spectra" in v.lower() and "OutB" in v and "ICR" not in v]

        # ── Read data rows ────────────────────────────────────────────────────
        data_rows = []
        for line in lines:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split("\t")
            try:
                data_rows.append([float(p) if p.strip() else 0.0 for p in parts])
            except ValueError:
                continue

        if not data_rows:
            raise ValueError(f"No numeric data found in {filepath}")

        n_cols_data = len(data_rows[0])
        arr = np.array(data_rows, dtype=float)

        def col(idx):
            """Get column by 1-based index, returns zeros if out of range."""
            c = idx - 1
            if 0 <= c < arr.shape[1]:
                return arr[:, c]
            return np.zeros(len(arr))

        energy = col(energy_col)

        # ── Compute mu(E) ─────────────────────────────────────────────────────
        if mode == "fluorescence":
            if inb_spectra_cols and outb_spectra_cols:
                # Sum all individual MCA spectra channels
                fluor  = sum(col(c) for c in inb_spectra_cols)
                fluor += sum(col(c) for c in outb_spectra_cols)
            elif inb_col and outb_col:
                fluor = col(inb_col) + col(outb_col)
            elif inb_col:
                fluor = col(inb_col)
            else:
                raise ValueError("No fluorescence columns found for mode='fluorescence'.")
            i0 = col(i0_col) if i0_col else np.ones(len(energy))
            with np.errstate(divide="ignore", invalid="ignore"):
                raw_mu = np.where(i0 > 0, fluor / i0, 0.0)
        else:
            # Transmission: ln(I0/I1)
            if not i0_col or not i1_col:
                raise ValueError("I0 or I1 column not found for mode='transmission'.")
            i0 = col(i0_col)
            i1 = col(i1_col)
            with np.errstate(divide="ignore", invalid="ignore"):
                raw_mu = np.where((i0 > 0) & (i1 > 0), np.log(i0 / i1), 0.0)

        # ── Sort by energy ────────────────────────────────────────────────────
        sort_idx = np.argsort(energy)
        energy   = energy[sort_idx]
        raw_mu   = raw_mu[sort_idx]

        # ── Find E0 (max of first derivative) ─────────────────────────────────
        e0 = self._find_e0(energy, raw_mu)

        # ── Normalize ─────────────────────────────────────────────────────────
        if normalize:
            mu, e0 = self._normalize(energy, raw_mu, e0)
            is_norm = True
            scan_type = f"{mode} (normalized)"
        else:
            mu = raw_mu
            is_norm = False
            scan_type = f"{mode} (raw)"

        return ExperimentalScan(
            label=scan_label,
            source_file=filepath,
            energy_ev=energy,
            mu=mu,
            e0=e0,
            is_normalized=is_norm,
            scan_type=scan_type,
            metadata={"mode": mode, "element": element, "edge": edge,
                      "col_map": col_map},
        )

    # ── Generic CSV / two-column text ─────────────────────────────────────────
    def parse_csv(
        self,
        filepath: str,
        energy_col: int = 0,
        mu_col: int = 1,
        skip_rows: int = 0,
        delimiter: Optional[str] = None,
        normalize: bool = False,
    ) -> ExperimentalScan:
        """
        Parse any plain-text two-column file.
        energy_col, mu_col: 0-based column indices.
        """
        rows = []
        with open(filepath, encoding="utf-8", errors="replace") as f:
            for _ in range(skip_rows):
                next(f)
            for line in f:
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                try:
                    parts = s.split(delimiter) if delimiter else s.split()
                    e  = float(parts[energy_col])
                    mu = float(parts[mu_col])
                    rows.append((e, mu))
                except (ValueError, IndexError):
                    continue

        if not rows:
            raise ValueError(f"No numeric data found in {filepath}")

        rows.sort(key=lambda r: r[0])
        energy = np.array([r[0] for r in rows])
        mu_raw = np.array([r[1] for r in rows])

        e0 = self._find_e0(energy, mu_raw)

        if normalize:
            mu, e0 = self._normalize(energy, mu_raw, e0)
            is_norm = True
        else:
            mu = mu_raw
            is_norm = False

        return ExperimentalScan(
            label=os.path.basename(filepath),
            source_file=filepath,
            energy_ev=energy,
            mu=mu,
            e0=e0,
            is_normalized=is_norm,
            scan_type="generic",
        )

    # ── Dispatch by extension ─────────────────────────────────────────────────
    # ── .nor (Athena XDI normalized export) ──────────────────────────────────
    def parse_nor(self, filepath: str) -> List[ExperimentalScan]:
        """
        Parse an Athena XDI .nor file (normalized XAS export).

        Header lines start with '#' and carry XDI metadata.
        Data columns (space-separated, Fortran scientific notation supported):
            1: energy (eV)
            2: norm       — normalized mu(E)
            3: nbkg       — background on norm grid
            4: flat       — flat-normalized mu(E)   ← preferred
            5: fbkg       — background on flat grid
            6: nder       — derivative
            7: nsec       — 2nd derivative

        Returns a single-element list containing one ExperimentalScan.
        """
        col_names: Dict[int, str] = {}   # 0-indexed col → name
        meta: Dict[str, str] = {}

        with open(filepath, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if not line.startswith("#"):
                    break
                body = line.lstrip("#").strip()
                if ":" in body:
                    key, _, val = body.partition(":")
                    key = key.strip().lower()
                    val = val.strip()
                    # Column.N: name
                    if key.startswith("column."):
                        try:
                            idx = int(key.split(".")[1]) - 1   # 0-indexed
                            col_names[idx] = val.split()[0].lower()
                        except (ValueError, IndexError):
                            pass
                    else:
                        meta[key] = val

        # numpy handles Fortran scientific notation (0.12345E-01) natively
        data = np.loadtxt(filepath, comments="#")
        if data.ndim == 1:
            data = data.reshape(1, -1)
        if data.shape[0] == 0:
            raise ValueError(f"No data rows found in {os.path.basename(filepath)}")

        energy = data[:, 0].astype(float)

        # Prefer 'flat' (Athena flat-normalized), fall back to 'norm', then col 1
        mu            = None
        col_used      = "norm"
        for preferred in ("flat", "norm"):
            for idx, name in col_names.items():
                if name == preferred and idx < data.shape[1]:
                    mu       = data[:, idx].astype(float)
                    col_used = preferred
                    break
            if mu is not None:
                break
        if mu is None:
            mu       = data[:, 1].astype(float) if data.shape[1] > 1 else data[:, 0].astype(float)
            col_used = "col2"

        # Extract metadata from header
        e0 = 0.0
        for e0_key in ("athena.e0", "element.e0", "xdi.e0"):
            if e0_key in meta:
                try:
                    e0 = float(meta[e0_key])
                    break
                except ValueError:
                    pass

        element = meta.get("element.symbol", "")
        edge    = meta.get("element.edge", "")
        basename = os.path.splitext(os.path.basename(filepath))[0]
        if element:
            label = f"{basename}  ({element} {edge}-edge)" if edge else f"{basename}  ({element})"
        else:
            label = basename

        return [ExperimentalScan(
            label        = label,
            source_file  = filepath,
            energy_ev    = energy,
            mu           = mu,
            e0           = e0,
            is_normalized= True,
            scan_type    = f"normalized ({col_used})",
        )]

    # ── SXRMB beamline .dat ───────────────────────────────────────────────────
    def parse_sxrmb(self, filepath: str, signal: str = "auto") -> List[ExperimentalScan]:
        """
        Parse a CLS SXRMB beamline .dat file.

        Header lines start with '#'.  The column-header line (last # line before data)
        is tab-separated and names each column.

        Recognised columns:
          energy  → EnergyFeedback.X  (or EnergyFeedback)
          I0      → BeamlineI0Detector
          TEY     → TEYDetector  (raw) or norm_TEYDetector (pre-normalised)
          fluor   → norm_<Element>Ka1  (e.g. norm_ClKa1, norm_SKa1 …)

        signal = "auto"   → try fluorescence first, fall back to TEY
        signal = "tey"    → TEY / I0
        signal = "fluor"  → fluorescence / I0 (first norm_*Ka1 column found)
        signal = "both"   → return two ExperimentalScan objects (TEY + fluor)
        """
        meta: Dict[str, str] = {}
        col_header_line: str = ""

        with open(filepath, "r", encoding="utf-8", errors="replace") as fh:
            lines = fh.readlines()

        # Parse header
        data_start = 0
        for i, line in enumerate(lines):
            s = line.strip()
            if not s.startswith("#"):
                data_start = i
                break
            body = s.lstrip("#").strip()
            # Last non-empty # line before data = column headers
            if body and not body.startswith("-"):
                col_header_line = body

        # Parse column names
        col_names = [c.strip() for c in col_header_line.split("\t") if c.strip()]

        # Parse data rows
        data_rows = []
        for line in lines[data_start:]:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            try:
                vals = [float(v) for v in s.split()]
                if vals:
                    data_rows.append(vals)
            except ValueError:
                continue

        if not data_rows:
            raise ValueError(f"No data rows found in SXRMB file: {os.path.basename(filepath)}")

        data = np.array(data_rows)

        def _col(names):
            """Return index of first matching column name (case-insensitive partial)."""
            for name in names:
                for j, c in enumerate(col_names):
                    if name.lower() in c.lower():
                        return j
            return None

        # Identify columns
        i_energy = _col(["EnergyFeedback.X", "EnergyFeedback"])
        i_i0     = _col(["BeamlineI0Detector", "I0"])
        i_tey    = _col(["norm_TEYDetector", "TEYDetector"])
        # Fluorescence: prefer pre-normalised norm_*Ka1 columns
        i_fluor  = _col(["norm_ClKa1", "norm_SKa1", "norm_PKa1",
                          "norm_NiKa1", "norm_TiKa1", "norm_FeKa1",
                          "norm_CuKa1", "norm_ZnKa1", "norm_MnKa1"])
        # Fallback: raw *Ka1 columns
        if i_fluor is None:
            i_fluor = _col(["ClKa1", "SKa1", "PKa1", "NiKa1", "TiKa1",
                             "FeKa1", "CuKa1", "ZnKa1", "MnKa1"])

        # Reference channel: I₂ gas cell or photodiode placed after sample.
        # Priority: explicit I2 column → diode/PD → transmission channel.
        # The reference name is stored alongside the signal for energy alignment.
        _ref_candidates = [
            # Explicit I2 names (case-insensitive match via _col)
            ["I2Detector", "I_2", "I2"],
            # Photodiode / transmission diode
            ["PDDetector", "DiodePHDetector", "DiodeDetector", "PhotoDiode", "PD"],
            # Generic "reference" keyword
            ["reference", "ReferenceDetector", "RefDetector"],
            # Transmission detector (In / It)
            ["It", "Itrans", "I_t"],
        ]
        i_ref      = None
        ref_label  = ""
        for candidates in _ref_candidates:
            idx = _col(candidates)
            if idx is not None and idx != i_energy and idx != i_i0:
                i_ref     = idx
                ref_label = col_names[idx]
                break

        if i_energy is None:
            raise ValueError("Could not find energy column in SXRMB file.")

        energy = data[:, i_energy]
        basename = os.path.splitext(os.path.basename(filepath))[0]

        # Extract edge/element from header metadata
        edge_str = ""
        for line in lines:
            if "Scanned Edge" in line or "Edge:" in line:
                edge_str = line.split(":")[-1].strip().rstrip()
                break

        def _norm_signal(raw_col, i0_col):
            """Divide raw signal by I0 if I0 available."""
            sig = data[:, raw_col]
            if i0_col is not None:
                i0 = data[:, i0_col]
                i0 = np.where(np.abs(i0) < 1e-6, 1.0, i0)
                sig = sig / i0
            # Shift so minimum is 0
            sig = sig - sig.min()
            return sig

        # Build reference arrays once (shared by all scans from this file)
        if i_ref is not None:
            ref_sig = _norm_signal(i_ref, i_i0)
        else:
            ref_sig = None

        def _make_scan(label, mu, scan_type):
            return ExperimentalScan(
                label=label, source_file=filepath,
                energy_ev=energy.copy(), mu=mu,
                e0=0.0, is_normalized=False,
                scan_type=scan_type,
                ref_energy_ev=energy.copy() if ref_sig is not None else None,
                ref_mu=ref_sig.copy() if ref_sig is not None else None,
                ref_label=ref_label,
            )

        results = []

        # TEY scan
        if signal in ("auto", "tey", "both") and i_tey is not None:
            tey_col = col_names[i_tey]
            if "norm_" in tey_col.lower():
                mu_tey = data[:, i_tey].copy()
            else:
                mu_tey = _norm_signal(i_tey, i_i0)
            mu_tey = mu_tey - mu_tey.min()
            lbl = f"{basename}  TEY"
            if edge_str:
                lbl += f"  ({edge_str})"
            results.append(_make_scan(lbl, mu_tey, "SXRMB TEY"))

        # Fluorescence scan
        if signal in ("auto", "fluor", "both") and i_fluor is not None:
            fluor_col = col_names[i_fluor]
            if "norm_" in fluor_col.lower():
                mu_fl = data[:, i_fluor].copy()
            else:
                mu_fl = _norm_signal(i_fluor, i_i0)
            mu_fl = mu_fl - mu_fl.min()
            lbl = f"{basename}  Fluor ({fluor_col})"
            if edge_str:
                lbl += f"  ({edge_str})"
            results.append(_make_scan(lbl, mu_fl, "SXRMB Fluorescence"))

        # Fallback: auto picked neither
        if not results and signal == "auto" and i_tey is not None:
            mu_tey = _norm_signal(i_tey, i_i0)
            results.append(_make_scan(f"{basename}  TEY", mu_tey, "SXRMB TEY"))

        if not results:
            raise ValueError(
                "Could not identify TEY or fluorescence columns in SXRMB file.\n"
                f"Columns found: {col_names}")

        return results

    @staticmethod
    def is_sxrmb(filepath: str) -> bool:
        """Quick check: does this .dat file look like SXRMB output?"""
        try:
            with open(filepath, "r", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    if "SXRMB" in line or "CLS SXRMB" in line:
                        return True
                    if not line.startswith("#"):
                        break
        except Exception:
            pass
        return False

    # ── ln(I₀/I₂) reference helpers ──────────────────────────────────────────

    # Column-name candidates for I0 and I2 channels.
    _I0_COLS = ["BeamlineI0Detector", "I0Detector", "I0", "I_0"]
    _I2_COLS = ["I2Detector", "I_2", "I2"]

    def _read_dat_raw_columns(self, filepath: str):
        """
        Read a # header + whitespace-data .dat file and locate energy, I0 and
        I2 column indices.  Returns (col_names, i_energy, i_i0, i_i2, data_np)
        or None if the file cannot be parsed.
        """
        with open(filepath, "r", encoding="utf-8", errors="replace") as fh:
            lines = fh.readlines()

        col_header_line = ""
        data_start = 0
        for i, line in enumerate(lines):
            s = line.strip()
            if not s.startswith("#"):
                data_start = i
                break
            body = s.lstrip("#").strip()
            if body and not body.startswith("-"):
                col_header_line = body

        col_names = [c.strip() for c in col_header_line.split("\t") if c.strip()]
        if not col_names:
            col_names = col_header_line.split()

        data_rows = []
        for line in lines[data_start:]:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            try:
                vals = [float(v) for v in s.split()]
                if vals:
                    data_rows.append(vals)
            except ValueError:
                continue

        if not data_rows:
            return None

        data = np.array(data_rows)

        def _find(candidates):
            for name in candidates:
                for j, c in enumerate(col_names):
                    if name.lower() in c.lower():
                        return j
            return None

        i_energy = _find(["EnergyFeedback.X", "EnergyFeedback",
                           "energy", "Energy", "eV"])
        i_i0     = _find(self._I0_COLS)
        i_i2     = _find(self._I2_COLS)
        return col_names, i_energy, i_i0, i_i2, data

    def peek_i0_i2(self, filepath: str) -> Tuple[bool, bool, str, str]:
        """
        Quick peek at column headers.
        Returns (has_i0, has_i2, i0_colname, i2_colname).
        """
        try:
            result = self._read_dat_raw_columns(filepath)
            if result is None:
                return False, False, "", ""
            col_names, _, i_i0, i_i2, _ = result
            i0_name = col_names[i_i0] if i_i0 is not None else ""
            i2_name = col_names[i_i2] if i_i2 is not None else ""
            return (i_i0 is not None), (i_i2 is not None), i0_name, i2_name
        except Exception:
            return False, False, "", ""

    def extract_reference_scan(self, filepath: str) -> Optional[ExperimentalScan]:
        """
        Compute ln(I₀/I₂) from a .dat file and return it as an ExperimentalScan.
        Returns None when I0 or I2 columns are absent or the file cannot be read.
        """
        try:
            result = self._read_dat_raw_columns(filepath)
            if result is None:
                return None
            col_names, i_energy, i_i0, i_i2, data = result
            if i_energy is None or i_i0 is None or i_i2 is None:
                return None

            energy = data[:, i_energy]
            i0     = data[:, i_i0]
            i2     = data[:, i_i2]

            with np.errstate(divide="ignore", invalid="ignore"):
                ratio  = np.where((i0 > 0) & (i2 > 0), i0 / i2, np.nan)
                ref_mu = np.where(np.isfinite(ratio) & (ratio > 0),
                                  np.log(ratio), np.nan)

            mask = np.isfinite(ref_mu)
            energy_clean = energy[mask]
            mu_clean     = ref_mu[mask]

            if len(energy_clean) < 3:
                return None

            basename = os.path.splitext(os.path.basename(filepath))[0]
            i0_name  = col_names[i_i0]
            i2_name  = col_names[i_i2]

            return ExperimentalScan(
                label=f"{basename}  ln(I\u2080/I\u2082)",
                source_file=filepath,
                energy_ev=energy_clean,
                mu=mu_clean,
                e0=0.0,
                is_normalized=False,
                scan_type=f"reference  ln({i0_name}/{i2_name})",
            )
        except Exception:
            return None

    # ── Channel preview (raw data for display before import) ─────────────────

    def preview_channels(self, filepath: str) -> list:
        """
        Detect every meaningful signal channel in a .dat file and return them
        for visual preview before the user commits to importing.

        Returns a list of (display_name, kind, energy_array, signal_array)
        where kind ∈ {'tey', 'fluorescence', 'transmission', 'reference'}.
        Signals are min-shifted (not normalised) so the spectral shape is clear.
        """
        try:
            if self.is_sxrmb(filepath):
                return self._preview_sxrmb_channels(filepath)
            else:
                return self._preview_bioxas_channels(filepath)
        except Exception:
            return []

    def _preview_sxrmb_channels(self, filepath: str) -> list:
        """Extract preview channels from a CLS SXRMB .dat file."""
        with open(filepath, "r", encoding="utf-8", errors="replace") as fh:
            lines = fh.readlines()

        col_header_line = ""
        data_start = 0
        for i, line in enumerate(lines):
            s = line.strip()
            if not s.startswith("#"):
                data_start = i
                break
            body = s.lstrip("#").strip()
            if body and not body.startswith("-"):
                col_header_line = body

        col_names = [c.strip() for c in col_header_line.split("\t") if c.strip()]

        data_rows = []
        for line in lines[data_start:]:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            try:
                vals = [float(v) for v in s.split()]
                if vals:
                    data_rows.append(vals)
            except ValueError:
                continue

        if not data_rows:
            return []

        data = np.array(data_rows)

        def _col(names):
            for name in names:
                for j, c in enumerate(col_names):
                    if name.lower() in c.lower():
                        return j
            return None

        def _get(idx):
            return data[:, idx].copy() if (idx is not None and idx < data.shape[1]) else None

        i_energy = _col(["EnergyFeedback.X", "EnergyFeedback"])
        i_i0     = _col(["BeamlineI0Detector", "I0"])
        i_tey    = _col(["norm_TEYDetector", "TEYDetector"])
        i_fluor  = _col(["norm_ClKa1", "norm_SKa1", "norm_PKa1",
                          "norm_NiKa1", "norm_TiKa1", "norm_FeKa1",
                          "norm_CuKa1", "norm_ZnKa1", "norm_MnKa1"])
        if i_fluor is None:
            i_fluor = _col(["ClKa1", "SKa1", "PKa1", "NiKa1", "TiKa1",
                             "FeKa1", "CuKa1", "ZnKa1", "MnKa1"])
        i_i2 = _col(self._I2_COLS)

        energy = _get(i_energy)
        if energy is None:
            return []

        channels = []

        def _norm_sig(raw_col):
            """Divide by I0 if available, then min-shift."""
            sig = _get(raw_col)
            if sig is None:
                return None
            if i_i0 is not None and "norm_" not in col_names[raw_col].lower():
                i0v = _get(i_i0)
                if i0v is not None:
                    i0v = np.where(np.abs(i0v) < 1e-9, 1.0, i0v)
                    sig = sig / i0v
            sig = sig - sig.min()
            return sig

        if i_tey is not None:
            sig = _norm_sig(i_tey)
            if sig is not None:
                channels.append(("TEY", "tey", energy.copy(), sig))

        if i_fluor is not None:
            fluor_col_name = col_names[i_fluor]
            sig = _norm_sig(i_fluor)
            if sig is not None:
                channels.append((f"Fluorescence  ({fluor_col_name})",
                                  "fluorescence", energy.copy(), sig))

        if i_i0 is not None and i_i2 is not None:
            i0v = _get(i_i0)
            i2v = _get(i_i2)
            if i0v is not None and i2v is not None:
                with np.errstate(divide="ignore", invalid="ignore"):
                    ratio  = np.where((i0v > 0) & (i2v > 0), i0v / i2v, np.nan)
                    ref_mu = np.where(
                        np.isfinite(ratio) & (ratio > 0), np.log(ratio), np.nan)
                mask = np.isfinite(ref_mu)
                if mask.sum() > 3:
                    channels.append(("Reference  ln(I\u2080/I\u2082)",
                                      "reference",
                                      energy[mask].copy(), ref_mu[mask]))

        return channels

    def _preview_bioxas_channels(self, filepath: str) -> list:
        """Extract preview channels from a BioXAS XDI .dat file."""
        with open(filepath, encoding="utf-8", errors="replace") as fh:
            lines = fh.readlines()

        col_map: Dict[int, str] = {}
        for line in lines:
            s = line.strip()
            if not s.startswith("#"):
                break
            m = re.match(r"#\s*Column\.(\d+):\s*(.+)", s)
            if m:
                col_map[int(m.group(1))] = m.group(2).strip()

        energy_col = self._find_col(col_map, ["energy", "eV"], required=True)
        i0_col     = self._find_col(col_map, ["I0", "I0Detector"])
        i1_col     = self._find_col(col_map, ["I1", "I1Detector"])
        inb_col    = self._find_col(col_map, ["InB_DarkCorrect", "NiKa1_InB"])
        outb_col   = self._find_col(col_map, ["OutB_DarkCorrect", "NiKa1_OutB"])
        # BioXAS rarely has a dedicated I2 but check anyway
        i2_col_name = None
        for v in col_map.values():
            if any(k.lower() in v.lower() for k in self._I2_COLS):
                i2_col_name = v
                break

        data_rows = []
        for line in lines:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split("\t")
            try:
                data_rows.append([float(p) if p.strip() else 0.0 for p in parts])
            except ValueError:
                continue

        if not data_rows:
            return []

        arr = np.array(data_rows, dtype=float)

        def col(idx):
            c = idx - 1
            return arr[:, c].copy() if (idx and 0 <= c < arr.shape[1]) else None

        energy = col(energy_col)
        if energy is None:
            return []

        channels = []

        # Fluorescence: (InB + OutB) / I0
        if inb_col and outb_col:
            fluor = col(inb_col) + col(outb_col)
        elif inb_col:
            fluor = col(inb_col)
        else:
            fluor = None

        if fluor is not None:
            i0v = col(i0_col) if i0_col else None
            if i0v is not None:
                i0v = np.where(np.abs(i0v) < 1e-9, 1.0, i0v)
                fluor = fluor / i0v
            fluor = fluor - fluor.min()
            channels.append(("Fluorescence  (InB + OutB) / I\u2080",
                              "fluorescence", energy.copy(), fluor))

        # Transmission: ln(I0/I1)
        if i0_col and i1_col:
            i0v = col(i0_col)
            i1v = col(i1_col)
            with np.errstate(divide="ignore", invalid="ignore"):
                trans = np.where((i0v > 0) & (i1v > 0),
                                 np.log(i0v / i1v), np.nan)
            mask = np.isfinite(trans)
            if mask.sum() > 3:
                channels.append(("Transmission  ln(I\u2080/I\u2081)",
                                  "transmission",
                                  energy[mask].copy(), trans[mask]))

        # Reference: ln(I0/I2) if I2 present
        if i0_col and i2_col_name:
            i2_idx = next((k for k, v in col_map.items()
                           if v == i2_col_name), None)
            if i2_idx:
                i0v = col(i0_col)
                i2v = col(i2_idx)
                if i0v is not None and i2v is not None:
                    with np.errstate(divide="ignore", invalid="ignore"):
                        ratio  = np.where((i0v > 0) & (i2v > 0), i0v / i2v, np.nan)
                        ref_mu = np.where(
                            np.isfinite(ratio) & (ratio > 0), np.log(ratio), np.nan)
                    mask = np.isfinite(ref_mu)
                    if mask.sum() > 3:
                        channels.append(("Reference  ln(I\u2080/I\u2082)",
                                          "reference",
                                          energy[mask].copy(), ref_mu[mask]))

        return channels

    def parse_any(self, filepath: str, **kwargs) -> List[ExperimentalScan]:
        """Auto-detect format and return a list of scans."""
        ext = os.path.splitext(filepath)[1].lower()
        if ext == ".prj":
            return self.parse_prj(filepath)
        elif ext == ".dat":
            return [self.parse_dat(filepath, **kwargs)]
        elif ext == ".nor":
            return self.parse_nor(filepath)
        else:
            return [self.parse_csv(filepath, **kwargs)]

    # ─────────────────────────────────────────────────────────────────────────
    #  Normalization
    # ─────────────────────────────────────────────────────────────────────────
    def normalize_scan(
        self,
        scan: ExperimentalScan,
        pre_range: Tuple[float, float] = (-150, -20),
        post_range: Tuple[float, float] = (50, 300),
        e0: Optional[float] = None,
        nnorm: int = 2,
    ) -> ExperimentalScan:
        """Return a new ExperimentalScan with normalized mu."""
        e0 = e0 or scan.e0
        mu_norm, e0_new = self._normalize(scan.energy_ev, scan.mu, e0,
                                          pre_range, post_range, nnorm)
        import copy
        out = copy.copy(scan)
        out.mu = mu_norm
        out.e0 = e0_new
        out.is_normalized = True
        return out

    def _normalize(
        self,
        energy: np.ndarray,
        mu: np.ndarray,
        e0: float,
        pre_range: Tuple[float, float] = (-150, -20),
        post_range: Tuple[float, float] = (50, 300),
        nnorm: int = 2,
    ) -> Tuple[np.ndarray, float]:
        """
        Athena/Larch-style normalization.
        Tries xraylarch pre_edge() first (exact Athena algorithm).
        Falls back to our own polynomial implementation if larch is unavailable.

        nnorm: degree of the post-edge polynomial (1=linear, 2=quadratic, 3=cubic).
               Athena stores this as bkg_nnorm; default 2 matches Athena's default.
        """
        session = _get_larch_session()
        if session is not None:
            try:
                return self._normalize_larch(
                    session, energy, mu, e0, pre_range, post_range, nnorm)
            except Exception:
                pass   # larch failed for this scan → use polynomial fallback

        return self._normalize_poly(energy, mu, e0, pre_range, post_range, nnorm)

    @staticmethod
    def _norm_is_valid(energy: np.ndarray, mu_norm: np.ndarray,
                       e0: float, pre_range, post_range) -> bool:
        """Quick sanity check: pre-edge ≈ 0, post-edge ≈ 1."""
        pre_m  = (energy >= e0 + pre_range[0])  & (energy <= e0 + pre_range[1])
        post_m = (energy >= e0 + post_range[0]) & (energy <= e0 + post_range[1])
        if pre_m.sum() < 2 or post_m.sum() < 2:
            return False
        pre_mean  = float(np.mean(np.abs(mu_norm[pre_m])))
        post_mean = float(np.mean(mu_norm[post_m]))
        return pre_mean < 0.15 and 0.4 < post_mean < 1.6

    @staticmethod
    def _safe_ranges(energy: np.ndarray, e0: float,
                     pre_range: Tuple[float, float],
                     post_range: Tuple[float, float]):
        """Clamp pre/post ranges to what the scan actually covers."""
        lo = energy.min() - e0
        hi = energy.max() - e0
        p1 = max(pre_range[0], lo + 5.0)
        p2 = min(pre_range[1], -5.0)
        n1 = max(post_range[0], 30.0)
        n2 = min(post_range[1], hi - 30.0)
        # Make sure ranges are valid
        if p2 <= p1:  p2 = p1 + 20.0
        if n2 <= n1:  n2 = n1 + 50.0
        return (p1, p2), (n1, n2)

    @staticmethod
    def _normalize_larch(
        session,
        energy: np.ndarray,
        mu: np.ndarray,
        e0: float,
        pre_range: Tuple[float, float],
        post_range: Tuple[float, float],
        nnorm: int,
    ) -> Tuple[np.ndarray, float]:
        """Delegate to xraylarch pre_edge().  Auto-retries with clamped ranges
        if the first attempt produces physically unreasonable values."""
        from larch import Group
        from larch.xafs import pre_edge

        def _try(pr, nr):
            g = Group(energy=energy.copy(), mu=mu.copy())
            pre_edge(g, e0=e0,
                     pre1=pr[0], pre2=pr[1],
                     norm1=nr[0], norm2=nr[1],
                     nnorm=nnorm, _larch=session)
            # Use g.flat (post-edge polynomial evaluated at each E) rather than
            # g.norm (constant edge-step at e0).  g.flat is what Athena displays
            # as "normalized mu(E)" — it removes the smooth background curvature
            # so the post-edge region is perfectly flat at 1.0.
            flat = getattr(g, "flat", None)
            if flat is None or not np.isfinite(flat).all():
                flat = g.norm   # fallback if flat not computed
            return flat, float(g.e0)

        norm, new_e0 = _try(pre_range, post_range)

        # If the result looks wrong, try again with ranges clamped to data bounds
        if not ExperimentalParser._norm_is_valid(energy, norm, new_e0, pre_range, post_range):
            safe_pre, safe_post = ExperimentalParser._safe_ranges(
                energy, new_e0, pre_range, post_range)
            norm2, new_e0_2 = _try(safe_pre, safe_post)
            if ExperimentalParser._norm_is_valid(energy, norm2, new_e0_2,
                                                  safe_pre, safe_post):
                return norm2, new_e0_2
            # Last resort: wide conservative defaults
            norm3, new_e0_3 = _try((-200, -20), (50, min(800, energy.max()-new_e0-30)))
            return norm3, new_e0_3

        return norm, new_e0

    @staticmethod
    def _normalize_poly(
        energy: np.ndarray,
        mu: np.ndarray,
        e0: float,
        pre_range: Tuple[float, float],
        post_range: Tuple[float, float],
        nnorm: int,
    ) -> Tuple[np.ndarray, float]:
        """
        Polynomial Athena-style normalization (scipy-only fallback).

        The critical fix vs. the previous linear version: post-edge is fit with
        a polynomial of degree `nnorm` (usually 2).  A degree-1 line cannot
        follow the smooth E^-3 decay of the atomic background, causing spectra
        to diverge at high energy after normalization.
        """
        pre_mask  = (energy >= e0 + pre_range[0]) & (energy <= e0 + pre_range[1])
        post_mask = (energy >= e0 + post_range[0]) & (energy <= e0 + post_range[1])

        # Pre-edge: linear fit → subtract
        if pre_mask.sum() >= 2:
            pre_fit = np.polyfit(energy[pre_mask], mu[pre_mask], 1)
        else:
            pre_fit = np.polyfit(energy, mu, 1)
        pre_line = np.polyval(pre_fit, energy)
        mu_sub   = mu - pre_line

        # Post-edge: polynomial of degree nnorm fit to the post-edge region.
        # Divide mu_sub by the polynomial *evaluated at every energy point*
        # (Athena "flat" normalisation) rather than by the single constant
        # edge_step = poly(e0).  This removes the smooth E^-3 background curve
        # so the post-edge stays flat at 1.0 across the full energy range.
        deg = min(nnorm, max(1, int(post_mask.sum()) - 1))   # can't exceed data pts
        if post_mask.sum() >= deg + 1:
            post_fit   = np.polyfit(energy[post_mask], mu_sub[post_mask], deg)
            post_poly  = np.polyval(post_fit, energy)
        elif post_mask.sum() >= 1:
            # Not enough points for a polynomial; use a flat constant
            const      = float(np.mean(mu_sub[post_mask]))
            post_poly  = np.full_like(energy, const)
        else:
            const      = float(mu_sub.max()) if mu_sub.max() != 0 else 1.0
            post_poly  = np.full_like(energy, const)

        # Guard against near-zero denominator (avoid div-by-zero in pre-edge)
        edge_step_at_e0 = float(np.polyval(post_fit, e0)) if post_mask.sum() >= deg + 1 \
                          else float(post_poly[0])
        if abs(edge_step_at_e0) < 1e-10:
            edge_step_at_e0 = 1.0
            post_poly = np.full_like(energy, 1.0)

        # Clamp denominator so it never flips sign or goes tiny far from the edge
        sign     = 1.0 if edge_step_at_e0 > 0 else -1.0
        post_poly = np.where(np.abs(post_poly) < 0.05 * abs(edge_step_at_e0),
                             sign * 0.05 * abs(edge_step_at_e0),
                             post_poly)

        return mu_sub / post_poly, e0

    def _find_e0(self, energy: np.ndarray, mu: np.ndarray) -> float:
        """Estimate E0 as the energy of maximum first derivative."""
        if len(energy) < 4:
            return float(energy[len(energy) // 2])
        # Only look in the rising-edge region (exclude far pre-edge noise)
        mid = len(energy) // 4
        grad = np.gradient(mu[mid:], energy[mid:])
        idx  = int(np.argmax(grad)) + mid
        return float(energy[idx])

    # ─────────────────────────────────────────────────────────────────────────
    #  Perl format helpers
    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _parse_perl_array(block: str, name: str) -> Optional[List[float]]:
        """
        Extract @name = ('v1','v2',...); from a Perl-serialized block.
        Handles single-line and multi-line arrays.
        """
        pattern = re.compile(
            r"@" + re.escape(name) + r"\s*=\s*\((.+?)\)\s*;",
            re.DOTALL
        )
        m = pattern.search(block)
        if not m:
            return None
        content = m.group(1)
        values  = re.findall(r"'([^']*)'", content)
        result  = []
        for v in values:
            try:
                result.append(float(v))
            except ValueError:
                pass
        return result if result else None

    @staticmethod
    def _parse_perl_kvlist(content: str) -> Dict:
        """
        Parse a flat Perl key-value list: 'key','value','key2','value2',...
        Returns dict of str→str.
        """
        tokens = re.findall(r"'([^']*)'", content)
        d = {}
        it = iter(tokens)
        for k in it:
            try:
                d[k] = next(it)
            except StopIteration:
                break
        return d

    @staticmethod
    def _find_col(col_map: Dict[int, str], keywords: List[str],
                  required: bool = False) -> Optional[int]:
        """Find the first column index whose name contains any of the keywords."""
        for idx, name in sorted(col_map.items()):
            if any(kw.lower() in name.lower() for kw in keywords):
                return idx
        if required:
            raise ValueError(f"Required column not found. Keywords tried: {keywords}")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
#  Scan averaging with optional reference-based energy alignment
# ═══════════════════════════════════════════════════════════════════════════════

def _find_e0_simple(energy: np.ndarray, mu: np.ndarray) -> float:
    """Return the energy of the max first derivative (simple E0 finder)."""
    if len(energy) < 4:
        return float(energy[len(energy) // 2])
    grad = np.gradient(mu, energy)
    lo, hi = len(energy) // 10, len(energy) * 9 // 10
    return float(energy[lo + int(np.argmax(np.abs(grad[lo:hi])))])


def align_and_average_scans(
    scans: List[ExperimentalScan],
    use_reference: bool = True,
    label: str = "",
) -> ExperimentalScan:
    """
    Align multiple scans (optionally via their reference channel) and average.

    Alignment algorithm (reference-based)
    ──────────────────────────────────────
    For each scan i, find the inflection point of its reference spectrum:
        E0_ref_i  = argmax |dμ_ref/dE|
    Compute shift relative to scan 0:
        ΔE_i = E0_ref_0 − E0_ref_i
    Apply shift to the main energy axis:
        energy_i_aligned = energy_i + ΔE_i

    Without reference (or if any scan is missing one), no shift is applied —
    scans are assumed to be on the same energy axis already.

    Averaging
    ─────────
    1. Find the intersection energy range across all (aligned) scans.
    2. Use the finest energy step among all scans for the common grid.
    3. Linearly interpolate each scan onto the common grid.
    4. Average: μ_avg(E) = mean_i [ interp(μ_i, E) ]

    Returns
    ───────
    A new ExperimentalScan whose .ref_mu is the averaged reference (if present).
    The .metadata dict contains 'n_averaged', 'shifts_ev', 'ref_e0s'.
    """
    if not scans:
        raise ValueError("No scans provided to average.")
    if len(scans) == 1:
        sc = scans[0]
        out = ExperimentalScan(
            label=label or f"{sc.label} (avg n=1)",
            source_file=sc.source_file,
            energy_ev=sc.energy_ev.copy(),
            mu=sc.mu.copy(),
            e0=sc.e0,
            is_normalized=sc.is_normalized,
            scan_type=sc.scan_type,
            ref_energy_ev=sc.ref_energy_ev.copy() if sc.ref_energy_ev is not None else None,
            ref_mu=sc.ref_mu.copy() if sc.ref_mu is not None else None,
            ref_label=sc.ref_label,
            metadata={"n_averaged": 1, "shifts_ev": [0.0], "ref_e0s": []},
        )
        return out

    # ── Step 1: compute energy shifts from reference ──────────────────────────
    shifts = [0.0] * len(scans)
    ref_e0s: List[float] = []
    all_have_ref = use_reference and all(s.has_reference() for s in scans)

    if all_have_ref:
        # Find E0 of each scan's reference spectrum
        for sc in scans:
            ref_e0s.append(_find_e0_simple(sc.ref_energy_ev, sc.ref_mu))
        # Shift each scan so its reference E0 matches scan-0's reference E0
        anchor = ref_e0s[0]
        shifts = [anchor - e for e in ref_e0s]   # ΔE_i = E0_ref_0 − E0_ref_i

    # ── Step 2: build shifted energy arrays ───────────────────────────────────
    shifted_energies = [sc.energy_ev + sh for sc, sh in zip(scans, shifts)]

    # ── Step 3: common energy grid (intersection range, finest step) ──────────
    e_min = max(e[0]  for e in shifted_energies)
    e_max = min(e[-1] for e in shifted_energies)
    if e_max <= e_min:
        # Fallback to union if intersection is empty
        e_min = min(e[0]  for e in shifted_energies)
        e_max = max(e[-1] for e in shifted_energies)

    steps = [float(np.min(np.diff(e))) for e in shifted_energies if len(e) > 1]
    step  = min(steps) if steps else 0.1
    grid  = np.arange(e_min, e_max + step * 0.5, step)

    # ── Step 4: interpolate and average μ ────────────────────────────────────
    interp_mu = [np.interp(grid, e, sc.mu) for e, sc in zip(shifted_energies, scans)]
    avg_mu    = np.mean(interp_mu, axis=0)

    # ── Step 5: average reference spectra (if all present) ───────────────────
    avg_ref_mu = None
    avg_ref_en = None
    if all_have_ref:
        shifted_ref_en = [sc.ref_energy_ev + sh
                          for sc, sh in zip(scans, shifts)]
        ref_e_min = max(e[0]  for e in shifted_ref_en)
        ref_e_max = min(e[-1] for e in shifted_ref_en)
        if ref_e_max > ref_e_min:
            ref_steps = [float(np.min(np.diff(e)))
                         for e in shifted_ref_en if len(e) > 1]
            ref_step  = min(ref_steps) if ref_steps else step
            ref_grid  = np.arange(ref_e_min, ref_e_max + ref_step * 0.5, ref_step)
            interp_ref = [np.interp(ref_grid, e, sc.ref_mu)
                          for e, sc in zip(shifted_ref_en, scans)]
            avg_ref_mu = np.mean(interp_ref, axis=0)
            avg_ref_en = ref_grid

    lbl = label or f"{scans[0].label} (avg n={len(scans)})"
    return ExperimentalScan(
        label=lbl,
        source_file=scans[0].source_file,
        energy_ev=grid,
        mu=avg_mu,
        e0=scans[0].e0,
        is_normalized=scans[0].is_normalized,
        scan_type=scans[0].scan_type,
        ref_energy_ev=avg_ref_en,
        ref_mu=avg_ref_mu,
        ref_label=scans[0].ref_label if all_have_ref else "",
        metadata={
            "n_averaged":  len(scans),
            "source_labels": [s.label for s in scans],
            "shifts_ev":   [round(sh, 4) for sh in shifts],
            "ref_e0s":     [round(e, 2) for e in ref_e0s],
            "aligned_by_reference": all_have_ref,
        },
    )

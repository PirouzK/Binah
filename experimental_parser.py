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

    def display_name(self) -> str:
        return self.label or os.path.basename(self.source_file)


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
    def parse_any(self, filepath: str, **kwargs) -> List[ExperimentalScan]:
        """Auto-detect format and return a list of scans."""
        ext = os.path.splitext(filepath)[1].lower()
        if ext == ".prj":
            return self.parse_prj(filepath)
        elif ext == ".dat":
            return [self.parse_dat(filepath, **kwargs)]
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

"""
ORCA Output File Parser — TDDFT / XAS Section Extractor
Supports ORCA 4, 5, and 6 output formats.

ORCA 4/5 row layout (UV-style):
    State  Energy(cm-1)  Wavelength(nm)  fosc  ...
    State  cm-1  nm  D2  m2(*1e6)  Q2(*1e6)  total  ... (combined)

ORCA 6 row layout (XAS-style, used for ALL section types):
    0-1A -> N-1A   Energy(eV)  Energy(cm-1)  Wavelength(nm)  fosc  ...
    0-1A -> N-1A   eV  cm-1  nm  D2  m2(*1e6)  Q2(*1e6)  total  ... (combined)

Section types present in ORCA 6 XAS output (11 total):
  Absorption:
    1. Electric Dipole                               (length gauge)
    2. Velocity Dipole                               (velocity gauge)
    3. Electric Dipole + Magnetic Dipole + Electric Quadrupole  (combined, length)
    4. Combined D2+m2+Q2 (Velocity)                  (combined, velocity)
    5. D2 + m2 + Q2 (Origin Adjusted)               (combined, origin adjusted)
    6. Combined D2+m2+Q2 (Origin Indep., Length)     (origin independent, length)
    7. Combined D2+m2+Q2 (Origin Indep., Velocity)   (origin independent, velocity)
    8. Absorption (Semi-Classical)                   (FFMIO formulation)
  CD:
    9. CD Spectrum                                   (via electric dipole moments)
   10. CD Spectrum (Velocity)                        (via velocity dipole moments)
   11. CD Spectrum (Semi-Classical)

Section types present in ORCA 4/5 UV/Vis output (up to 6):
    1. Electric Dipole
    2. Velocity Dipole
    3. CD Spectrum                                   (bare section header)
    4. CD Spectrum (Velocity)
    5. Electric Dipole + Magnetic Dipole + Electric Quadrupole
    6. D2 + m2 + Q2 (Origin Adjusted)
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple


# ═══════════════════════════════════════════════════════════════════════════════
#  Data classes
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ExcitedState:
    index: int
    energy_au: float
    energy_ev: float
    energy_cm: float
    transitions: List[Tuple[int, int, float]] = field(default_factory=list)


@dataclass
class TDDFTSpectrum:
    """One parsed spectrum block — UV/Vis or XAS, any section type."""
    label: str
    section_index: int
    is_xas: bool = False

    states: List[int] = field(default_factory=list)
    transition_labels: List[str] = field(default_factory=list)
    energies_ev: List[float] = field(default_factory=list)
    energies_cm: List[float] = field(default_factory=list)
    wavelengths_nm: List[float] = field(default_factory=list)
    fosc: List[float] = field(default_factory=list)
    rotatory_strength: List[float] = field(default_factory=list)

    # Combined-spectrum component breakdown (D2 + m2 + Q2)
    fosc_d2: List[float] = field(default_factory=list)   # electric dipole
    fosc_m2: List[float] = field(default_factory=list)   # magnetic dipole  (*1e-6 de-scaled)
    fosc_q2: List[float] = field(default_factory=list)   # electric quadrupole (*1e-6 de-scaled)

    excited_states: List["ExcitedState"] = field(default_factory=list)

    def is_cd(self) -> bool:
        return "CD" in self.label.upper()

    def is_combined(self) -> bool:
        return bool(self.fosc_d2)

    def display_name(self) -> str:
        prefix = f"[Run {self.section_index + 1}] " if self.section_index > 0 else ""
        return prefix + self.label


@dataclass
class PartialDavidsonState:
    index: int
    energy_ev: float
    energy_cm: float
    energy_au: float
    iteration: int


@dataclass
class ParseDiagnosis:
    is_complete: bool = True
    termination_reason: str = ""
    termination_line: str = ""
    tddft_started: bool = False
    tddft_converged: bool = False
    davidson_iterations: int = 0
    n_roots_requested: int = 0
    partial_states: List[PartialDavidsonState] = field(default_factory=list)
    xas_mode: bool = False

    def summary(self) -> str:
        lines = []
        if self.is_complete:
            lines.append("Status: Calculation completed normally.")
        else:
            lines.append("Status: Calculation DID NOT complete.")
            if self.termination_reason:
                lines.append(f"Reason: {self.termination_reason}")
        if self.tddft_started:
            lines.append(f"TD-DFT: Initialised (nroots={self.n_roots_requested})")
            if self.xas_mode:
                lines.append("Mode:   XAS / core-excitation")
            if self.tddft_converged:
                lines.append("Davidson: Converged")
            else:
                lines.append(
                    f"Davidson: Stopped at iteration {self.davidson_iterations} "
                    f"(not converged -- no spectrum printed)"
                )
        if self.partial_states:
            lines.append(
                f"Partial eigenvalues recovered: {len(self.partial_states)}"
            )
        return "\n".join(lines)


@dataclass
class ParseResult:
    spectra: List[TDDFTSpectrum] = field(default_factory=list)
    diagnosis: ParseDiagnosis = field(default_factory=ParseDiagnosis)


# ═══════════════════════════════════════════════════════════════════════════════
#  Parser
# ═══════════════════════════════════════════════════════════════════════════════

class OrcaParser:

    # ── Section header patterns ───────────────────────────────────────────────
    # Tuple: (regex, label, is_cd, is_combined)
    # Order: most-specific patterns FIRST to avoid false matches.
    _SECTIONS = [

        # ── Combined multipole: Origin Independent variants (ORCA 6 only) ────
        (re.compile(r"COMBINED.*QUADRUPOLE SPECTRUM\s*\(Origin Independent.*Velocity\)", re.I),
         "Combined D2+m2+Q2 (Origin Indep., Velocity)", False, True),
        (re.compile(r"COMBINED.*QUADRUPOLE SPECTRUM\s*\(Origin Independent.*Length\)", re.I),
         "Combined D2+m2+Q2 (Origin Indep., Length)", False, True),

        # ── Combined multipole: origin adjusted (ORCA 4/5 and 6) ─────────────
        (re.compile(r"COMBINED.*QUADRUPOLE SPECTRUM\s*\(origin adjusted\)", re.I),
         "D2 + m2 + Q2 (Origin Adjusted)", False, True),

        # ── Combined multipole: velocity gauge (ORCA 6) ───────────────────────
        (re.compile(r"COMBINED.*QUADRUPOLE SPECTRUM\s*\(Velocity\)", re.I),
         "Combined D2+m2+Q2 (Velocity)", False, True),

        # ── Combined multipole: standard (no parenthetical suffix) ────────────
        # Negative lookahead prevents matching the variants above.
        (re.compile(r"COMBINED.*QUADRUPOLE SPECTRUM(?!\s*\()", re.I),
         "Electric Dipole + Magnetic Dipole + Electric Quadrupole", False, True),

        # ── Absorption: semi-classical (ORCA 6) ───────────────────────────────
        (re.compile(r"ABSORPTION SPECTRUM VIA FULL SEMI-CLASSICAL FORMULATION", re.I),
         "Absorption (Semi-Classical)", False, False),

        # ── Absorption: electric / velocity dipole ────────────────────────────
        (re.compile(r"ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS", re.I),
         "Electric Dipole", False, False),
        (re.compile(r"ABSORPTION SPECTRUM VIA TRANSITION VELOCITY DIPOLE MOMENTS", re.I),
         "Velocity Dipole", False, False),

        # ── CD: semi-classical (ORCA 6) ───────────────────────────────────────
        (re.compile(r"CD SPECTRUM VIA FULL SEMI-CLASSICAL FORMULATION", re.I),
         "CD Spectrum (Semi-Classical)", True, False),

        # ── CD: electric / velocity dipole moments ────────────────────────────
        # "VIA TRANSITION ELECTRIC DIPOLE" before velocity so it's checked first
        (re.compile(r"CD SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS", re.I),
         "CD Spectrum", True, False),
        (re.compile(r"CD SPECTRUM VIA TRANSITION VELOCITY DIPOLE MOMENTS", re.I),
         "CD Spectrum (Velocity)", True, False),

        # ── CD: bare header (ORCA 4/5 only) ──────────────────────────────────
        # Must come last; $ anchor prevents matching longer CD SPECTRUM lines.
        (re.compile(r"^\s*CD SPECTRUM\s*$", re.I),
         "CD Spectrum", True, False),
    ]

    # ── Row patterns ──────────────────────────────────────────────────────────

    # ORCA 6 / XAS-style: 0-1A -> N-1A   eV   cm-1   nm   value ...
    _XAS6_ROW = re.compile(
        r"^\s*(\d+)-\w+\s*->\s*(\d+)-\w+\s+"   # from -> to transition label
        r"([\d.]+)\s+"                           # energy (eV)
        r"([\d.]+)\s+"                           # energy (cm-1)
        r"([\d.]+)\s+"                           # wavelength (nm)
        r"([-\d.eE+]+)"                          # fosc / R (first value column)
    )

    # ORCA 6 combined: same prefix but has D2  m2  Q2  total
    _XAS6_COMB_ROW = re.compile(
        r"^\s*(\d+)-\w+\s*->\s*(\d+)-\w+\s+"
        r"([\d.]+)\s+"                           # eV
        r"([\d.]+)\s+"                           # cm-1
        r"([\d.]+)\s+"                           # nm
        r"([-\d.eE+]+)\s+"                       # D2
        r"([-\d.eE+]+)\s+"                       # m2 (*1e6)
        r"([-\d.eE+]+)\s+"                       # Q2 (*1e6)
        r"([-\d.eE+]+)"                          # total (D2+m2+Q2 or larger)
    )

    # ORCA 4/5 UV-style:  State   cm-1   nm   value  ...
    _UV_ROW = re.compile(
        r"^\s*(\d+)\s+([\d.]+)\s+([\d.]+)\s+([-\d.eE+]+)"
    )

    # ORCA 4/5 combined UV:  State  cm-1  nm  D2  m2(*1e6)  Q2(*1e6)  total ...
    _UV_COMB_ROW = re.compile(
        r"^\s*(\d+)\s+"
        r"([\d.]+)\s+"            # cm-1
        r"([\d.]+)\s+"            # nm
        r"([-\d.eE+]+)\s+"        # D2
        r"([-\d.eE+]+)\s+"        # m2 (*1e6)
        r"([-\d.eE+]+)\s+"        # Q2 (*1e6)
        r"([-\d.eE+]+)"           # total
    )

    # ── Diagnostic patterns ───────────────────────────────────────────────────
    _TDDFT_INIT    = re.compile(r"TD-DFT CALCULATION INITIALIZED", re.I)
    _DAVIDSON_DONE = re.compile(r"(TDDFT DONE|DAVIDSON DONE)", re.I)   # ORCA 4/5 vs 6
    _DAVIDSON_ITER = re.compile(r"\*\*\*\*Iteration\s+(\d+)\*\*\*\*")  # TDDFT Davidson only
    _DAVIDSON_EMIN = re.compile(r"Lowest Energy\s+:\s+([\d.]+)")
    _NROOTS        = re.compile(r"Number of roots to be determined\s+\.\.\.\s+(\d+)")
    _XAS_ARRAY     = re.compile(r"XAS localization array", re.I)
    _NORMAL_END    = re.compile(r"ORCA TERMINATED NORMALLY", re.I)
    _SLURM_CANCEL  = re.compile(
        r"(CANCELLED|TIMEOUT|TIME.LIMIT|DUE TO TIME|slurmstepd.*error|job.*cancel)", re.I
    )
    _KILLED        = re.compile(r"(Killed|Segmentation fault|Bus error|Out of memory)", re.I)
    _ORCA_ERROR    = re.compile(r"ORCA finished with error", re.I)

    # ── Separator ─────────────────────────────────────────────────────────────
    _SEP = re.compile(r"^\s*-{10,}")

    # ─────────────────────────────────────────────────────────────────────────
    def parse(self, filepath: str) -> ParseResult:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()

        result          = ParseResult()
        result.diagnosis = self._build_diagnosis(lines)

        spectra: List[TDDFTSpectrum] = []
        excited_blocks  = self._parse_excited_states(lines)
        type_counts: Dict[str, int] = {}

        i = 0
        while i < len(lines):
            line    = lines[i]
            matched = False
            for pattern, label, is_cd, is_combined in self._SECTIONS:  # noqa
                if pattern.search(line):
                    count = type_counts.get(label, 0)
                    type_counts[label] = count + 1
                    sp = TDDFTSpectrum(label=label, section_index=count)
                    i  = self._parse_block(lines, i + 1, sp, is_cd, is_combined)
                    if sp.states:
                        spectra.append(sp)
                    matched = True
                    break
            if not matched:
                i += 1

        # Attach excited-state MO data (match by root count)
        for sp in spectra:
            n = len(sp.states)
            for block in excited_blocks:
                if len(block) == n:
                    sp.excited_states = block
                    break

        result.spectra = spectra
        return result

    # ─────────────────────────────────────────────────────────────────────────
    #  Block parser
    # ─────────────────────────────────────────────────────────────────────────
    def _parse_block(
        self, lines: List[str], start: int,
        sp: TDDFTSpectrum, is_cd: bool, is_combined: bool
    ) -> int:
        """
        Walk the lines of one spectrum section.  The structure is always:
            (header line already consumed)
            ---sep 1---
            column labels  (1-2 lines, skipped)
            ---sep 2---      ← in_data = True from here
            data rows ...
            ---sep 3---      ← end of block
        or (for the last section in the file) end-of-file.

        Row format is auto-detected from the first data line:
          XAS6 format:  0-1A -> N-1A  eV  cm-1  nm  value ...
          UV format:    integer  cm-1  nm  value ...
        """
        i         = start
        sep_count = 0
        in_data   = False
        xas6      = False     # True once first data line confirms ORCA 6 XAS format

        while i < len(lines):
            line = lines[i]

            if self._SEP.match(line):
                sep_count += 1
                if sep_count == 2:
                    in_data = True
                elif sep_count == 3:
                    i += 1
                    break
                i += 1
                continue

            if not in_data or not line.strip():
                i += 1
                continue

            # Auto-detect row format on the first data line
            if not sp.states:
                if self._XAS6_ROW.match(line):
                    xas6       = True
                    sp.is_xas  = True

            if xas6:
                if is_combined:
                    self._parse_xas6_comb_row(line, sp)
                else:
                    self._parse_xas6_row(line, sp, is_cd)
            else:
                if is_combined:
                    self._parse_uv_comb_row(line, sp)
                else:
                    self._parse_uv_row(line, sp, is_cd)

            i += 1

        # Tag UV/Vis spectra with very high energies as XAS too
        if not xas6 and sp.energies_cm and sp.energies_cm[0] > 50_000:
            sp.is_xas = True

        return i

    # ── ORCA 6 / XAS6 simple row ──────────────────────────────────────────────
    # 0-1A -> N-1A   eV   cm-1   nm   fosc/R   [extra ignored]
    def _parse_xas6_row(self, line: str, sp: TDDFTSpectrum, is_cd: bool):
        m = self._XAS6_ROW.match(line)
        if not m:
            return
        try:
            to_state  = int(m.group(2))
            ev        = float(m.group(3))
            cm        = float(m.group(4))
            nm        = float(m.group(5))
            val       = float(m.group(6))
            frm_state = int(m.group(1))
            lbl       = f"{frm_state}-1A->{to_state}-1A"

            sp.states.append(to_state)
            sp.transition_labels.append(lbl)
            sp.energies_ev.append(ev)
            sp.energies_cm.append(cm)
            sp.wavelengths_nm.append(nm)
            if is_cd:
                sp.rotatory_strength.append(val)
            else:
                sp.fosc.append(val)
        except ValueError:
            pass

    # ── ORCA 6 / XAS6 combined row ────────────────────────────────────────────
    # 0-1A -> N-1A   eV   cm-1   nm   D2   m2(*1e6)   Q2(*1e6)   total   ...
    def _parse_xas6_comb_row(self, line: str, sp: TDDFTSpectrum):
        m = self._XAS6_COMB_ROW.match(line)
        if not m:
            return
        try:
            frm_state = int(m.group(1))
            to_state  = int(m.group(2))
            ev        = float(m.group(3))
            cm        = float(m.group(4))
            nm        = float(m.group(5))
            d2        = float(m.group(6))
            m2_scaled = float(m.group(7))
            q2_scaled = float(m.group(8))
            f_total   = float(m.group(9))
            lbl       = f"{frm_state}-1A->{to_state}-1A"

            sp.states.append(to_state)
            sp.transition_labels.append(lbl)
            sp.energies_ev.append(ev)
            sp.energies_cm.append(cm)
            sp.wavelengths_nm.append(nm)
            sp.fosc.append(f_total)
            sp.fosc_d2.append(d2)
            sp.fosc_m2.append(m2_scaled * 1e-6)
            sp.fosc_q2.append(q2_scaled * 1e-6)
        except ValueError:
            pass

    # ── ORCA 4/5 UV simple row ────────────────────────────────────────────────
    # State   cm-1   nm   fosc/R   [extra ignored]
    def _parse_uv_row(self, line: str, sp: TDDFTSpectrum, is_cd: bool):
        m = self._UV_ROW.match(line)
        if not m:
            return
        try:
            state     = int(m.group(1))
            cm        = float(m.group(2))
            nm        = float(m.group(3))
            val       = float(m.group(4))
            ev        = cm * 1.23984e-4

            sp.states.append(state)
            sp.transition_labels.append(str(state))
            sp.energies_cm.append(cm)
            sp.energies_ev.append(ev)
            sp.wavelengths_nm.append(nm)
            if is_cd:
                sp.rotatory_strength.append(val)
            else:
                sp.fosc.append(val)
        except ValueError:
            pass

    # ── ORCA 4/5 UV combined row ──────────────────────────────────────────────
    # State   cm-1   nm   D2   m2(*1e6)   Q2(*1e6)   total   ...
    def _parse_uv_comb_row(self, line: str, sp: TDDFTSpectrum):
        m = self._UV_COMB_ROW.match(line)
        if not m:
            return
        try:
            state     = int(m.group(1))
            cm        = float(m.group(2))
            nm        = float(m.group(3))
            d2        = float(m.group(4))
            m2_scaled = float(m.group(5))
            q2_scaled = float(m.group(6))
            f_total   = float(m.group(7))
            ev        = cm * 1.23984e-4

            sp.states.append(state)
            sp.transition_labels.append(str(state))
            sp.energies_cm.append(cm)
            sp.energies_ev.append(ev)
            sp.wavelengths_nm.append(nm)
            sp.fosc.append(f_total)
            sp.fosc_d2.append(d2)
            sp.fosc_m2.append(m2_scaled * 1e-6)
            sp.fosc_q2.append(q2_scaled * 1e-6)
        except ValueError:
            pass

    # ─────────────────────────────────────────────────────────────────────────
    #  Excited state block parser
    # ─────────────────────────────────────────────────────────────────────────
    _STATE_HEADER   = re.compile(
        r"STATE\s+(\d+):\s+E=\s*([\d.]+)\s+au\s+([\d.]+)\s+eV\s+([\d.]+)\s+cm\*\*-1"
    )
    _TRANSITION_LINE = re.compile(
        # Handles both plain numbers ("45 -> 46") and ORCA 6 spin-labelled
        # orbitals ("0a -> 62a", "0b -> 62b").  The \w* after each \d+ group
        # absorbs the optional 'a'/'b' spin suffix without capturing it.
        r"\s+(\d+)\w*\s*->\s*(\d+)\w*\s*:\s*([\d.]+)\s+\(c="
    )

    def _parse_excited_states(self, lines: List[str]) -> List[List[ExcitedState]]:
        groups: List[List[ExcitedState]] = []
        current_group: List[ExcitedState] = []
        current_state: Optional[ExcitedState] = None
        last_idx = -1

        for line in lines:
            m = self._STATE_HEADER.search(line)
            if m:
                idx = int(m.group(1))
                if idx == 1 and last_idx > 1:
                    if current_group:
                        groups.append(current_group)
                    current_group = []
                last_idx      = idx
                current_state = ExcitedState(
                    index=idx,
                    energy_au=float(m.group(2)),
                    energy_ev=float(m.group(3)),
                    energy_cm=float(m.group(4)),
                )
                current_group.append(current_state)
                continue
            if current_state:
                t = self._TRANSITION_LINE.match(line)
                if t:
                    current_state.transitions.append(
                        (int(t.group(1)), int(t.group(2)), float(t.group(3)))
                    )

        if current_group:
            groups.append(current_group)
        return groups

    # ─────────────────────────────────────────────────────────────────────────
    #  Diagnosis builder
    # ─────────────────────────────────────────────────────────────────────────
    def _build_diagnosis(self, lines: List[str]) -> ParseDiagnosis:
        d         = ParseDiagnosis()
        last_iter = -1
        last_energies: List[float] = []

        for line in lines:
            if self._TDDFT_INIT.search(line):
                d.tddft_started = True
            if self._XAS_ARRAY.search(line):
                d.xas_mode = True
            m = self._NROOTS.search(line)
            if m:
                d.n_roots_requested = int(m.group(1))
            m = self._DAVIDSON_ITER.search(line)
            if m:
                it = int(m.group(1))
                if it > last_iter:
                    last_iter     = it
                    last_energies = []
            m = self._DAVIDSON_EMIN.search(line)
            if m and last_iter >= 0:
                last_energies.append(float(m.group(1)))
            if self._DAVIDSON_DONE.search(line):
                d.tddft_converged = True
            if self._NORMAL_END.search(line):
                d.is_complete        = True
                d.termination_reason = "NORMAL"
                d.termination_line   = line.strip()
            if self._SLURM_CANCEL.search(line):
                d.is_complete        = False
                d.termination_reason = "TIME LIMIT / SLURM CANCELLATION"
                d.termination_line   = line.strip()
            if self._KILLED.search(line):
                d.is_complete        = False
                d.termination_reason = "PROCESS KILLED"
                d.termination_line   = line.strip()
            if self._ORCA_ERROR.search(line):
                d.is_complete        = False
                d.termination_reason = "ORCA ERROR"
                d.termination_line   = line.strip()

        d.davidson_iterations = last_iter + 1

        EV_PER_AU = 27.2114
        CM_PER_EV = 8065.54
        for idx, e_au in enumerate(last_energies, start=1):
            e_ev = e_au * EV_PER_AU
            e_cm = e_ev * CM_PER_EV
            d.partial_states.append(PartialDavidsonState(
                index=idx, energy_ev=round(e_ev, 4),
                energy_cm=round(e_cm, 1), energy_au=e_au,
                iteration=last_iter
            ))

        return d

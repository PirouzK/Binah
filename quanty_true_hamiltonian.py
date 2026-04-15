"""
Build actual Quanty L-edge XAS scripts from the empirical seed bundle.

These scripts implement:
- a crystal-field Hamiltonian on the 3d shell,
- optional ligand-field / charge-transfer physics via a ligand d shell,
- a distinct final-state Hamiltonian with a 2p core hole,
- diagonalization of the initial-state multiplets,
- dipole transitions for L-edge XAS,
- isotropic XAS output.

The generated Lua is intended to be runnable in Quanty after the user
reviews and, if needed, adjusts chemistry-specific inputs such as nd,
Slater integrals, and low-symmetry crystal-field parameters.
"""

from __future__ import annotations

from typing import Dict


NI_HF_DEFAULTS = {
    "F2dd": 11.14,
    "F4dd": 6.87,
    "F2pd": 6.67,
    "G1pd": 4.92,
    "G3pd": 2.80,
    "zeta_3d": 0.081,
    "zeta_2p": 11.51,
    "Udd": 7.3,
    "Upd": 8.5,
    "Delta": 4.7,
}


GENERIC_DEFAULTS = {
    "F2dd": 11.14,
    "F4dd": 6.87,
    "F2pd": 6.67,
    "G1pd": 4.92,
    "G3pd": 2.80,
    "zeta_3d": 0.08,
    "zeta_2p": 10.0,
    "Udd": 6.5,
    "Upd": 8.0,
    "Delta": 3.0,
}


def _fmt(value, default=0.0):
    try:
        return f"{float(value):.6f}"
    except Exception:
        return f"{float(default):.6f}"


def _defaults_for_element(element: str, zeta_2p: float) -> Dict[str, float]:
    if element == "Ni":
        out = dict(NI_HF_DEFAULTS)
        out["zeta_2p"] = float(zeta_2p)
        return out
    out = dict(GENERIC_DEFAULTS)
    out["zeta_2p"] = float(zeta_2p)
    return out


def _symmetry_block(bundle: dict) -> str:
    meta = bundle["metadata"]
    sym = meta["site_symmetry"]
    sym_class = meta["symmetry_class"]
    cf = bundle["model_seeds"]["crystal_field"]
    sym_seed = cf["symmetry_specific"]
    rec = cf["recommended_start"]

    if sym_class == "cubic":
        return f"""-- Crystal-field inputs.
local TenDq = {rec['TenDq_eV']:.6f}
-- For cubic symmetry we construct the orbital energies from TenDq.
function Get3dOrbitalEnergies()
    return {{
        dx2y2 =  0.6 * TenDq,
        dz2   =  0.6 * TenDq,
        dyz   = -0.4 * TenDq,
        dxz   = -0.4 * TenDq,
        dxy   = -0.4 * TenDq,
    }}
end

local TenDqL = 0.000000
function GetLigandOrbitalEnergies()
    return {{
        dx2y2 =  0.6 * TenDqL,
        dz2   =  0.6 * TenDqL,
        dyz   = -0.4 * TenDqL,
        dxz   = -0.4 * TenDqL,
        dxy   = -0.4 * TenDqL,
    }}
end

-- Grouped hybridization parameters for cubic symmetry.
local Veg = 2.200000
local Vt2g = 1.100000
function GetHybridizationEnergies()
    return {{
        dx2y2 = Veg,
        dz2   = Veg,
        dyz   = Vt2g,
        dxz   = Vt2g,
        dxy   = Vt2g,
    }}
end
"""

    if sym_class == "tetragonal":
        return f"""-- Crystal-field inputs.
local TenDq = {rec['TenDq_eV']:.6f}
local Ds = {rec['Ds_eV']:.6f}
local Dt = {rec['Dt_eV']:.6f}
function Get3dOrbitalEnergies()
    local Dq = TenDq / 10.0
    return {{
        dx2y2 =  6.0 * Dq + 2.0 * Ds - 1.0 * Dt,
        dz2   =  6.0 * Dq - 2.0 * Ds - 6.0 * Dt,
        dxy   = -4.0 * Dq + 2.0 * Ds - 1.0 * Dt,
        dxz   = -4.0 * Dq - 1.0 * Ds + 4.0 * Dt,
        dyz   = -4.0 * Dq - 1.0 * Ds + 4.0 * Dt,
    }}
end

local TenDqL = 0.000000
local DsL = 0.000000
local DtL = 0.000000
function GetLigandOrbitalEnergies()
    local Dq = TenDqL / 10.0
    return {{
        dx2y2 =  6.0 * Dq + 2.0 * DsL - 1.0 * DtL,
        dz2   =  6.0 * Dq - 2.0 * DsL - 6.0 * DtL,
        dxy   = -4.0 * Dq + 2.0 * DsL - 1.0 * DtL,
        dxz   = -4.0 * Dq - 1.0 * DsL + 4.0 * DtL,
        dyz   = -4.0 * Dq - 1.0 * DsL + 4.0 * DtL,
    }}
end

local Veg = 2.200000
local Vt2g = 1.100000
function GetHybridizationEnergies()
    return {{
        dx2y2 = Veg,
        dz2   = Veg,
        dyz   = Vt2g,
        dxz   = Vt2g,
        dxy   = Vt2g,
    }}
end
"""

    if sym_class == "trigonal":
        return f"""-- Crystal-field inputs.
local TenDq = {rec['TenDq_eV']:.6f}
local Dtrig = 0.000000
-- For trigonal cases it is usually clearer to work directly with orbital energies.
local E_dx2y2 = 0.000000
local E_dz2   = 0.000000
local E_dyz   = 0.000000
local E_dxz   = 0.000000
local E_dxy   = 0.000000
function Get3dOrbitalEnergies()
    local avg = (E_dx2y2 + E_dz2 + E_dyz + E_dxz + E_dxy) / 5.0
    return {{
        dx2y2 = E_dx2y2 - avg,
        dz2   = E_dz2   - avg,
        dyz   = E_dyz   - avg,
        dxz   = E_dxz   - avg,
        dxy   = E_dxy   - avg,
    }}
end

local E_L_dx2y2 = 0.000000
local E_L_dz2   = 0.000000
local E_L_dyz   = 0.000000
local E_L_dxz   = 0.000000
local E_L_dxy   = 0.000000
function GetLigandOrbitalEnergies()
    local avg = (E_L_dx2y2 + E_L_dz2 + E_L_dyz + E_L_dxz + E_L_dxy) / 5.0
    return {{
        dx2y2 = E_L_dx2y2 - avg,
        dz2   = E_L_dz2   - avg,
        dyz   = E_L_dyz   - avg,
        dxz   = E_L_dxz   - avg,
        dxy   = E_L_dxy   - avg,
    }}
end

local V_dx2y2 = 2.200000
local V_dz2   = 2.200000
local V_dyz   = 1.100000
local V_dxz   = 1.100000
local V_dxy   = 1.100000
function GetHybridizationEnergies()
    return {{
        dx2y2 = V_dx2y2,
        dz2   = V_dz2,
        dyz   = V_dyz,
        dxz   = V_dxz,
        dxy   = V_dxy,
    }}
end
"""

    seed = sym_seed["crystal_field_seed"]
    lf_seed = sym_seed["ligand_field_seed"]
    return f"""-- Crystal-field inputs.
-- Low symmetry ({sym}) is implemented as a true one-particle crystal field
-- in the cubic-harmonic basis and then rotated into Quanty's spherical basis.
local E_dx2y2 = {_fmt(seed.get('E_dx2y2_eV'))}
local E_dz2   = {_fmt(seed.get('E_dz2_eV'))}
local E_dyz   = {_fmt(seed.get('E_dyz_eV'))}
local E_dxz   = {_fmt(seed.get('E_dxz_eV'))}
local E_dxy   = {_fmt(seed.get('E_dxy_eV'))}
function Get3dOrbitalEnergies()
    local avg = (E_dx2y2 + E_dz2 + E_dyz + E_dxz + E_dxy) / 5.0
    return {{
        dx2y2 = E_dx2y2 - avg,
        dz2   = E_dz2   - avg,
        dyz   = E_dyz   - avg,
        dxz   = E_dxz   - avg,
        dxy   = E_dxy   - avg,
    }}
end

local E_L_dx2y2 = 0.000000
local E_L_dz2   = 0.000000
local E_L_dyz   = 0.000000
local E_L_dxz   = 0.000000
local E_L_dxy   = 0.000000
function GetLigandOrbitalEnergies()
    local avg = (E_L_dx2y2 + E_L_dz2 + E_L_dyz + E_L_dxz + E_L_dxy) / 5.0
    return {{
        dx2y2 = E_L_dx2y2 - avg,
        dz2   = E_L_dz2   - avg,
        dyz   = E_L_dyz   - avg,
        dxz   = E_L_dxz   - avg,
        dxy   = E_L_dxy   - avg,
    }}
end

local V_dx2y2 = {_fmt(lf_seed.get('V_dx2y2_eV', 2.0))}
local V_dz2   = {_fmt(lf_seed.get('V_dz2_eV', 2.2))}
local V_dyz   = {_fmt(lf_seed.get('V_dyz_eV', 1.1))}
local V_dxz   = {_fmt(lf_seed.get('V_dxz_eV', 1.1))}
local V_dxy   = {_fmt(lf_seed.get('V_dxy_eV', 1.4))}
function GetHybridizationEnergies()
    return {{
        dx2y2 = V_dx2y2,
        dz2   = V_dz2,
        dyz   = V_dyz,
        dxz   = V_dxz,
        dxy   = V_dxy,
    }}
end
"""


def build_true_quanty_l_edge_xas_script(bundle: dict) -> str:
    meta = bundle["metadata"]
    obs = bundle["empirical_observables"]
    phys = bundle["physical_interpretation"]
    cf = bundle["model_seeds"]["crystal_field"]
    lf = bundle["model_seeds"]["ligand_field_charge_transfer"]
    defaults = _defaults_for_element(meta["element"], meta["zeta_2p_eV"])
    model_type = "LigandField" if phys["recommended_model_family"] != "crystal_field" else "CrystalField"
    symmetry_block = _symmetry_block(bundle)

    return f"""Verbosity(0)
-- Generated from the empirical L-edge fit in ledge_normalizer.py.
-- This is an actual Quanty L-edge XAS script with separate initial and final
-- state Hamiltonians following the standard crystal-field / ligand-field
-- multiplet workflow used in the official Quanty NiO tutorials.
--
-- Official references used for the structure of this script:
--   Quanty NiO crystal-field XAS L2,3 tutorial
--   Quanty NiO ligand-field XAS L2,3 tutorial
--
-- The script is intentionally written so that the same operator definitions
-- can later be extended to RIXS with CreateResonantSpectra.

local Element = "{meta['element']}"
local SiteSymmetry = "{meta['site_symmetry']}"
local ModelType = "{model_type}" -- "CrystalField" or "LigandField"
local UseChargeTransfer = (ModelType == "LigandField")

-- Chemistry-specific inputs.
local nd = 8       -- TODO set the formal 3d count for your system
local Npsi = 3     -- number of initial-state eigenstates retained
local Temperature = 0.0
local kB = 8.617333262145e-5

-- Spectral window and broadening.
local Emin = -15.0
local Emax = 25.0
local NE = 2000
local InternalGamma = 0.10
local PostGaussianFWHM = 0.35
local PostLorentzianFWHM = {(float(obs['L3_edge_fwhm_eV']) + float(obs['L2_edge_fwhm_eV']))/2.0:.6f}
local EmpiricalEdgeShift = {float(obs['L3_onset_eV']):.6f}

-- Atomic / Slater defaults.
-- For Ni the defaults below follow the official Quanty NiO ligand-field tutorial.
-- For other elements replace these with radial-integral values appropriate to the ion.
local F2dd_HF = {defaults['F2dd']:.6f}
local F4dd_HF = {defaults['F4dd']:.6f}
local F2pd_HF = {defaults['F2pd']:.6f}
local G1pd_HF = {defaults['G1pd']:.6f}
local G3pd_HF = {defaults['G3pd']:.6f}
local zeta_3d_HF = {defaults['zeta_3d']:.6f}
local zeta_2p_HF = {defaults['zeta_2p']:.6f}
local Udd = {defaults['Udd']:.6f}
local Upd = {defaults['Upd']:.6f}
local Delta = {defaults['Delta']:.6f}

local FddScale = {cf['recommended_start']['Fdd_scale']:.6f}
local FpdScale = {cf['recommended_start']['Fpd_scale']:.6f}
local Zeta3dScale = {cf['recommended_start']['zeta_3d_scale']:.6f}

local F2dd = FddScale * F2dd_HF
local F4dd = FddScale * F4dd_HF
local F2pd = FpdScale * F2pd_HF
local G1pd = FpdScale * G1pd_HF
local G3pd = FpdScale * G3pd_HF
local zeta_3d = Zeta3dScale * zeta_3d_HF
local zeta_2p = zeta_2p_HF

-- Core-valence monopoles derived in the usual way from Upd / Slater integrals.
local F0dd = Udd + (F2dd + F4dd) * 2.0 / 63.0
local F0pd = Upd + G1pd / 15.0 + 3.0 * G3pd / 70.0

-- Small fields only used to lift exact degeneracies when needed.
local Bz = 0.000001
local Hex = 0.000000

{symmetry_block}

-- Helper functions ----------------------------------------------------------
function BoltzmannWeights(psiList, H, temperature)
    local weights = {{}}
    if temperature <= 0 then
        for i = 1, #psiList do
            weights[i] = (i == 1) and 1.0 or 0.0
        end
        return weights
    end
    local energies = {{}}
    local emin = nil
    for i = 1, #psiList do
        local e = Chop(psiList[i] * H * psiList[i])
        energies[i] = e
        if emin == nil or e < emin then
            emin = e
        end
    end
    local z = 0.0
    for i = 1, #energies do
        local w = math.exp(-(energies[i] - emin) / (kB * temperature))
        weights[i] = w
        z = z + w
    end
    for i = 1, #weights do
        weights[i] = weights[i] / z
    end
    return weights
end

function SpectraWeightsForPolarizations(stateWeights, npol)
    local weights = {{}}
    local idx = 1
    for p = 1, npol do
        for i = 1, #stateWeights do
            weights[idx] = stateWeights[i]
            idx = idx + 1
        end
    end
    return weights
end

function BuildRotationMatrix(modelType)
    if modelType == "LigandField" then
        return YtoKMatrix({{"p","d","d"}})
    else
        return YtoKMatrix({{"p","d"}})
    end
end

function BuildDiagonalShellOperator(indicesUp, indicesDn, orbitalWeights, rotationInverse)
    -- Orbital order follows the cubic-harmonic ordering implied by YtoKMatrix("d"):
    -- dx2y2, dz2, dyz, dxz, dxy.
    local oppK = NewOperator("Number", NF, indicesUp, indicesUp,
        {{orbitalWeights.dx2y2, orbitalWeights.dz2, orbitalWeights.dyz, orbitalWeights.dxz, orbitalWeights.dxy}})
        + NewOperator("Number", NF, indicesDn, indicesDn,
        {{orbitalWeights.dx2y2, orbitalWeights.dz2, orbitalWeights.dyz, orbitalWeights.dxz, orbitalWeights.dxy}})
    return Rotate(oppK, rotationInverse)
end

function BuildHybridizationOperator(indicesUpA, indicesDnA, indicesUpB, indicesDnB, hoppings, rotationInverse)
    -- Hybridization is defined in the same cubic-harmonic basis and then
    -- rotated back into Quanty's standard spherical-harmonic shell basis.
    local creationTable = {{}}
    local upA = indicesUpA
    local dnA = indicesDnA
    local upB = indicesUpB
    local dnB = indicesDnB
    local weights = {{hoppings.dx2y2, hoppings.dz2, hoppings.dyz, hoppings.dxz, hoppings.dxy}}
    for i = 1, #weights do
        local v = weights[i]
        if math.abs(v) > 1e-12 then
            table.insert(creationTable, {{upA[i], -upB[i], v}})
            table.insert(creationTable, {{upB[i], -upA[i], v}})
            table.insert(creationTable, {{dnA[i], -dnB[i], v}})
            table.insert(creationTable, {{dnB[i], -dnA[i], v}})
        end
    end
    local oppK = NewOperator(NF, NB, creationTable)
    return Rotate(oppK, rotationInverse)
end

-- Basis definition ----------------------------------------------------------
if ModelType == "LigandField" then
    NF = 26
    NB = 0
    IndexDn_2p = {{0, 2, 4}}
    IndexUp_2p = {{1, 3, 5}}
    IndexDn_3d = {{6, 8,10,12,14}}
    IndexUp_3d = {{7, 9,11,13,15}}
    IndexDn_Ld = {{16,18,20,22,24}}
    IndexUp_Ld = {{17,19,21,23,25}}
else
    NF = 16
    NB = 0
    IndexDn_2p = {{0, 2, 4}}
    IndexUp_2p = {{1, 3, 5}}
    IndexDn_3d = {{6, 8,10,12,14}}
    IndexUp_3d = {{7, 9,11,13,15}}
end

-- Standard operators --------------------------------------------------------
OppSx_3d = NewOperator("Sx", NF, IndexUp_3d, IndexDn_3d)
OppSy_3d = NewOperator("Sy", NF, IndexUp_3d, IndexDn_3d)
OppSz_3d = NewOperator("Sz", NF, IndexUp_3d, IndexDn_3d)
OppSsqr_3d = NewOperator("Ssqr", NF, IndexUp_3d, IndexDn_3d)
OppLx_3d = NewOperator("Lx", NF, IndexUp_3d, IndexDn_3d)
OppLy_3d = NewOperator("Ly", NF, IndexUp_3d, IndexDn_3d)
OppLz_3d = NewOperator("Lz", NF, IndexUp_3d, IndexDn_3d)
OppLsqr_3d = NewOperator("Lsqr", NF, IndexUp_3d, IndexDn_3d)
OppJx_3d = NewOperator("Jx", NF, IndexUp_3d, IndexDn_3d)
OppJy_3d = NewOperator("Jy", NF, IndexUp_3d, IndexDn_3d)
OppJz_3d = NewOperator("Jz", NF, IndexUp_3d, IndexDn_3d)
OppJsqr_3d = NewOperator("Jsqr", NF, IndexUp_3d, IndexDn_3d)
Oppldots_3d = NewOperator("ldots", NF, IndexUp_3d, IndexDn_3d)

Oppcldots = NewOperator("ldots", NF, IndexUp_2p, IndexDn_2p)

OppF0_3d = NewOperator("U", NF, IndexUp_3d, IndexDn_3d, {{1,0,0}})
OppF2_3d = NewOperator("U", NF, IndexUp_3d, IndexDn_3d, {{0,1,0}})
OppF4_3d = NewOperator("U", NF, IndexUp_3d, IndexDn_3d, {{0,0,1}})
OppUpdF0 = NewOperator("U", NF, IndexUp_2p, IndexDn_2p, IndexUp_3d, IndexDn_3d, {{1,0}}, {{0,0}})
OppUpdF2 = NewOperator("U", NF, IndexUp_2p, IndexDn_2p, IndexUp_3d, IndexDn_3d, {{0,1}}, {{0,0}})
OppUpdG1 = NewOperator("U", NF, IndexUp_2p, IndexDn_2p, IndexUp_3d, IndexDn_3d, {{0,0}}, {{1,0}})
OppUpdG3 = NewOperator("U", NF, IndexUp_2p, IndexDn_2p, IndexUp_3d, IndexDn_3d, {{0,0}}, {{0,1}})

OppN_2p = NewOperator("Number", NF, IndexUp_2p, IndexUp_2p, {{1,1,1}})
    + NewOperator("Number", NF, IndexDn_2p, IndexDn_2p, {{1,1,1}})
OppN_3d = NewOperator("Number", NF, IndexUp_3d, IndexUp_3d, {{1,1,1,1,1}})
    + NewOperator("Number", NF, IndexDn_3d, IndexDn_3d, {{1,1,1,1,1}})

if ModelType == "LigandField" then
    OppSx_Ld = NewOperator("Sx", NF, IndexUp_Ld, IndexDn_Ld)
    OppSy_Ld = NewOperator("Sy", NF, IndexUp_Ld, IndexDn_Ld)
    OppSz_Ld = NewOperator("Sz", NF, IndexUp_Ld, IndexDn_Ld)
    OppSsqr_Ld = NewOperator("Ssqr", NF, IndexUp_Ld, IndexDn_Ld)
    OppLx_Ld = NewOperator("Lx", NF, IndexUp_Ld, IndexDn_Ld)
    OppLy_Ld = NewOperator("Ly", NF, IndexUp_Ld, IndexDn_Ld)
    OppLz_Ld = NewOperator("Lz", NF, IndexUp_Ld, IndexDn_Ld)
    OppLsqr_Ld = NewOperator("Lsqr", NF, IndexUp_Ld, IndexDn_Ld)
    OppJx_Ld = NewOperator("Jx", NF, IndexUp_Ld, IndexDn_Ld)
    OppJy_Ld = NewOperator("Jy", NF, IndexUp_Ld, IndexDn_Ld)
    OppJz_Ld = NewOperator("Jz", NF, IndexUp_Ld, IndexDn_Ld)
    OppJsqr_Ld = NewOperator("Jsqr", NF, IndexUp_Ld, IndexDn_Ld)
    OppN_Ld = NewOperator("Number", NF, IndexUp_Ld, IndexUp_Ld, {{1,1,1,1,1}})
        + NewOperator("Number", NF, IndexDn_Ld, IndexDn_Ld, {{1,1,1,1,1}})

    OppSx = OppSx_3d + OppSx_Ld
    OppSy = OppSy_3d + OppSy_Ld
    OppSz = OppSz_3d + OppSz_Ld
    OppSsqr = OppSx * OppSx + OppSy * OppSy + OppSz * OppSz
    OppLx = OppLx_3d + OppLx_Ld
    OppLy = OppLy_3d + OppLy_Ld
    OppLz = OppLz_3d + OppLz_Ld
    OppLsqr = OppLx * OppLx + OppLy * OppLy + OppLz * OppLz
    OppJx = OppJx_3d + OppJx_Ld
    OppJy = OppJy_3d + OppJy_Ld
    OppJz = OppJz_3d + OppJz_Ld
    OppJsqr = OppJx * OppJx + OppJy * OppJy + OppJz * OppJz
else
    OppSx = OppSx_3d
    OppSy = OppSy_3d
    OppSz = OppSz_3d
    OppSsqr = OppSsqr_3d
    OppLx = OppLx_3d
    OppLy = OppLy_3d
    OppLz = OppLz_3d
    OppLsqr = OppLsqr_3d
    OppJx = OppJx_3d
    OppJy = OppJy_3d
    OppJz = OppJz_3d
    OppJsqr = OppJsqr_3d
end

-- Dipole transition operators ----------------------------------------------
local t = math.sqrt(1/2)
local Akm
Akm = {{{{1,-1,t}},{{1,1,-t}}}}
TXASx = NewOperator("CF", NF, IndexUp_3d, IndexDn_3d, IndexUp_2p, IndexDn_2p, Akm)
Akm = {{{{1,-1,t*I}},{{1,1,t*I}}}}
TXASy = NewOperator("CF", NF, IndexUp_3d, IndexDn_3d, IndexUp_2p, IndexDn_2p, Akm)
Akm = {{{{1,0,1}}}}
TXASz = NewOperator("CF", NF, IndexUp_3d, IndexDn_3d, IndexUp_2p, IndexDn_2p, Akm)
TXASr = t * (TXASx - I * TXASy)
TXASl = -t * (TXASx + I * TXASy)

-- One-particle crystal field / ligand field --------------------------------
local Rotation = BuildRotationMatrix(ModelType)
local RotationInverse = Inverse(Rotation)

local E3d = Get3dOrbitalEnergies()
OppCF_3d = BuildDiagonalShellOperator(IndexUp_3d, IndexDn_3d, E3d, RotationInverse)

if ModelType == "LigandField" then
    local ELd = GetLigandOrbitalEnergies()
    local VHyb = GetHybridizationEnergies()
    OppCF_Ld = BuildDiagonalShellOperator(IndexUp_Ld, IndexDn_Ld, ELd, RotationInverse)
    OppHybrid = BuildHybridizationOperator(IndexUp_3d, IndexDn_3d, IndexUp_Ld, IndexDn_Ld, VHyb, RotationInverse)
else
    OppCF_Ld = 0
    OppHybrid = 0
end

-- Initial and final state Hamiltonians -------------------------------------
if ModelType == "LigandField" then
    local ed = (10 * Delta - nd * (19 + nd) * Udd / 2) / (10 + nd)
    local eL = nd * (((1 + nd) * Udd / 2) - Delta) / (10 + nd)
    local epfinal = (10 * Delta + (1 + nd) * (nd * Udd / 2 - (10 + nd) * Upd)) / (16 + nd)
    local edfinal = (10 * Delta - nd * (31 + nd) * Udd / 2 - 90 * Upd) / (16 + nd)
    local eLfinal = (((1 + nd) * (nd * Udd / 2 + 6 * Upd)) - (6 + nd) * Delta) / (16 + nd)

    Hamiltonian = F0dd * OppF0_3d + F2dd * OppF2_3d + F4dd * OppF4_3d
        + zeta_3d * Oppldots_3d
        + Bz * (2 * OppSz_3d + OppLz_3d)
        + Hex * OppSz_3d
        + OppCF_3d + OppCF_Ld + OppHybrid
        + ed * OppN_3d + eL * OppN_Ld

    XASHamiltonian = F0dd * OppF0_3d + F2dd * OppF2_3d + F4dd * OppF4_3d
        + zeta_3d * Oppldots_3d
        + Bz * (2 * OppSz_3d + OppLz_3d)
        + Hex * OppSz_3d
        + OppCF_3d + OppCF_Ld + OppHybrid
        + edfinal * OppN_3d + eLfinal * OppN_Ld + epfinal * OppN_2p
        + zeta_2p * Oppcldots
        + F0pd * OppUpdF0 + F2pd * OppUpdF2 + G1pd * OppUpdG1 + G3pd * OppUpdG3

    StartRestrictions = {{NF, NB, {{"000000 1111111111 0000000000", nd, nd}}, {{"111111 0000000000 1111111111", 16, 16}}}}
else
    Hamiltonian = F0dd * OppF0_3d + F2dd * OppF2_3d + F4dd * OppF4_3d
        + zeta_3d * Oppldots_3d
        + Bz * (2 * OppSz_3d + OppLz_3d)
        + Hex * OppSz_3d
        + OppCF_3d

    XASHamiltonian = Hamiltonian
        + zeta_2p * Oppcldots
        + F0pd * OppUpdF0 + F2pd * OppUpdF2 + G1pd * OppUpdG1 + G3pd * OppUpdG3

    StartRestrictions = {{NF, NB, {{"111111 0000000000", 6, 6}}, {{"000000 1111111111", nd, nd}}}}
end

-- Initial-state multiplets --------------------------------------------------
psiList = Eigensystem(Hamiltonian, StartRestrictions, Npsi)
Weights = BoltzmannWeights(psiList, Hamiltonian, Temperature)

OppList = {{Hamiltonian, OppSsqr, OppLsqr, OppJsqr, OppSz_3d, OppLz_3d, Oppldots_3d, OppN_3d}}
OppLabels = {{"<E>", "<S^2>", "<L^2>", "<J^2>", "<Sz_3d>", "<Lz_3d>", "<l.s>", "<N_3d>"}}
if ModelType == "LigandField" then
    table.insert(OppList, OppN_Ld)
    table.insert(OppLabels, "<N_Ld>")
end

print("Computed " .. tostring(#psiList) .. " initial-state multiplets.")
print("Initial-state expectation values:")
io.write("state ")
for i = 1, #OppLabels do
    io.write(string.format("%10s ", OppLabels[i]))
end
io.write("    weight\\n")
for i = 1, #psiList do
    io.write(string.format("%5d ", i))
    for j = 1, #OppList do
        local value = Chop(psiList[i] * OppList[j] * psiList[i])
        io.write(string.format("%10.4f ", value))
    end
    io.write(string.format(" %10.6f\\n", Weights[i]))
end

-- XAS spectra ---------------------------------------------------------------
XASSpectra = CreateSpectra(XASHamiltonian, {{TXASz, TXASr, TXASl}}, psiList,
    {{{{"Emin", Emin}}, {{"Emax", Emax}}, {{"NE", NE}}, {{"Gamma", InternalGamma}}}})
XASSpectra.Broaden(PostGaussianFWHM, PostLorentzianFWHM)

local IsoWeights = SpectraWeightsForPolarizations(Weights, 3)
XASIsoSpectra = Spectra.Sum(XASSpectra, IsoWeights)

XASSpectra.Print({{{{"file","Quanty_Ledge_XAS_All.dat"}}}})
XASIsoSpectra.Print({{{{"file","Quanty_Ledge_XAS_Iso.dat"}}}})

print("Finished Quanty L-edge XAS calculation.")
print("Initial states come from Eigensystem(Hamiltonian, ...).")
print("Final-state multiplet structure is sampled through the XASHamiltonian resolvent in CreateSpectra.")
print("Use the same Hamiltonian definitions as the basis for future RIXS extensions.")
"""

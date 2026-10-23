#!/usr/bin/env python3
"""
thesis_analysis.py  -  complete analysis code in ONE file
==========================================================

MSc thesis: "Bioelectric Design of Charged Hydrogels: Fixed-Charge Density as a Regulator of
the Local Ionic Microenvironment and Human Mesenchymal Stem/Stromal Cell Fate"
Augustine Odibo, Department of Chemical Engineering, Villanova University, 2026.

Implements every quantitative method in Chapter 3 and regenerates the tables/figures of
Chapter 4:

    Calculators   Table 3.1, 3.2, 4.5, Eq. 3.6, Sec. 3.7, Donnan / Debye / GHK / Flory-Rehner
    Rheology      GP, FN, t90, G'p, G'LVR, gamma_c, G'w, slope n, nu, xi (Sec. 3.3.3)
                  -> Tables 4.1-4.4, Figs 4.4-4.12, design map, stiffness-matched pairs
    IPNs          G'LVR, yield strain, swelling, EIS conductivity -> Figs 4.1-4.3, 4.13, 4.14
    Printers      swelling + gel fraction, two-way ANOVA -> Table 4.6, Fig. A.2
    Assays        BSA/BCA, PicoGreen, ALP (Eq. 3.7), release, MTT, BMP-2 ELISA (4PL)
    Statistics    Grubbs, Welch + Holm, two-way ANOVA (log), one-way ANOVA + Tukey

REQUIREMENTS
    pip install numpy scipy pandas statsmodels matplotlib openpyxl tabulate

USAGE
    python thesis_analysis.py                 # make SYNTHETIC demo data (if none) + run everything
    python thesis_analysis.py all --demo      # regenerate demo data, then run everything
    python thesis_analysis.py test            # self-check against numbers printed in the thesis
    python thesis_analysis.py calculators [--recipe AMPS 15 20]
    python thesis_analysis.py matrix   --manifest data/formulation_matrix/manifest.csv
    python thesis_analysis.py ipn      --data data/ipn --thickness 0.2 --area 0.8
    python thesis_analysis.py printer  --data data/printer_comparison
    python thesis_analysis.py assays   --data data/assays
    python thesis_analysis.py demo     --out data
    python thesis_analysis.py <command> -h    # options for any command

Outputs go to results/<command>/ (CSV tables, report.md, 300-dpi PNG figures).

IMPORTANT: the demo data are SYNTHETIC (simulated from the summary values in the thesis) so
the pipeline can run end-to-end. Replace data/ with your real TRIOS / CHI exports and balance
sheets (layouts described in each command's help text) to analyse the actual experiments.

NOTES FROM CHECKING THE CODE AGAINST THE THESIS
  1. Sec. 3.2.2 says MBAA came from a 10 % w/v stock, but Table 3.1 volumes match a 4 % w/v
     (40 mg/mL) stock exactly. Set MBAA_STOCK_MG_PER_ML below if your stock differs.
  2. DEAEAAm volumes in Table 3.1 match a 50 % w/v stock, not neat monomer.
  3. Table 4.5, 6.25 uL row: 10 x 238 / 250.5 = 9.50 % w/v PEGDA (table prints 9.52 %).
"""
from __future__ import annotations

import argparse
import io as _io
import itertools
import math
import re
import sys
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as st
from scipy.optimize import brentq, curve_fit
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt



# ============================================================================================
# THEORY
# ============================================================================================
# Physical-chemistry relations used in Chapter 2 and Sec. 3.3.3.
#
# * Eq. 2.1  Goldman-Hodgkin-Katz resting membrane potential
# * Eq. 2.2  Donnan partitioning of mobile ions into a charged gel
# * Eq. 2.3  Donnan potential, and the Donnan osmotic pressure Pi_D
# * Eq. 2.4  Debye length of a 1:1 electrolyte
# * Eq. 2.5  Flory-Rehner equilibrium swelling (solved for phi or nu)
# * Sec. 3.3.3  rubber-elasticity chain density nu ~ G'/RT and mesh size xi ~ (kT/G')^(1/3)
R = 8.314462618        # J/(mol K)
F = 96485.33212        # C/mol
KB = 1.380649e-23      # J/K
NA = 6.02214076e23


def thermal_voltage(T: float = 298.15) -> float:
    """RT/F in volts (25.69 mV at 25 C)."""
    return R * T / F


# ----------------------------------------------------------------------------
# Donnan / Debye
# ----------------------------------------------------------------------------
def donnan_concentrations(c_fixed_mM: float, c_salt_mM: float, sign: int = -1):
    """Mobile-ion concentrations inside a gel with fixed-charge density c_F (Eq. 2.2).

    Donnan condition c+ c- = cs^2 and electroneutrality.
    sign = -1 for an anionic network (counter-ion = cation), +1 for cationic.

    Returns (c_counter, c_co) in mM.
    """
    cf = abs(c_fixed_mM)
    c_counter = cf / 2 + np.sqrt(cf ** 2 / 4 + c_salt_mM ** 2)
    c_co = c_salt_mM ** 2 / c_counter
    return c_counter, c_co


def donnan_potential_mV(c_fixed_mM: float, c_salt_mM: float, sign: int = -1,
                        T: float = 298.15) -> float:
    """psi_D = (RT/F) ln(cs / c+)  (Eq. 2.3). Negative for anionic gels."""
    c_counter, _ = donnan_concentrations(c_fixed_mM, c_salt_mM, sign)
    psi = thermal_voltage(T) * np.log(c_salt_mM / c_counter) * 1000
    return psi if sign < 0 else -psi


def donnan_osmotic_pressure_kPa(c_fixed_mM: float, c_salt_mM: float, T: float = 298.15) -> float:
    """Pi_D = RT (c+ + c- - 2 cs). mM == mol/m^3, so result is Pa -> /1000 for kPa."""
    cp, cm = donnan_concentrations(c_fixed_mM, c_salt_mM)
    return R * T * (cp + cm - 2 * c_salt_mM) / 1000


def debye_length_nm(ionic_strength_M: float) -> float:
    """kappa^-1 ~ 0.304 / sqrt(I) nm for a 1:1 electrolyte at 25 C (Eq. 2.4)."""
    return 0.304 / np.sqrt(ionic_strength_M)


def ionic_strength(conc_M: dict, charges: dict) -> float:
    """I = 1/2 sum c_i z_i^2."""
    return 0.5 * sum(conc_M[k] * charges[k] ** 2 for k in conc_M)


def fixed_charge_from_formulation(conc_pct_wv: float, mw: float, swelling_volume_factor: float = 1.0,
                                  charges_per_monomer: int = 1) -> float:
    """Nominal fixed-charge density (mM) of an as-prepared gel, diluted by subsequent swelling.

    10 % w/v AMPS-Na (229.23 g/mol) -> 436 mM in the as-prepared state.
    """
    return conc_pct_wv * 10 / mw * 1000 * charges_per_monomer / swelling_volume_factor


# ----------------------------------------------------------------------------
# GHK
# ----------------------------------------------------------------------------
def ghk_vmem_mV(P: dict, c_out: dict, c_in: dict, T: float = 310.15) -> float:
    """Goldman-Hodgkin-Katz voltage equation (Eq. 2.1) for K+, Na+, Cl-.

    P, c_out, c_in : dicts with keys 'K', 'Na', 'Cl' (relative permeabilities; mM).
    """
    num = P["K"] * c_out["K"] + P["Na"] * c_out["Na"] + P["Cl"] * c_in["Cl"]
    den = P["K"] * c_in["K"] + P["Na"] * c_in["Na"] + P["Cl"] * c_out["Cl"]
    return thermal_voltage(T) * np.log(num / den) * 1000


def vmem_in_donnan_phase(c_fixed_mM: float, sign: int, P: dict, c_bath: dict, c_in: dict,
                         T: float = 310.15) -> float:
    """Illustrative Vmem of a cell sitting inside the Donnan phase of a charged gel.

    Cations (K, Na) are scaled by c+/cs and anions (Cl) by c-/cs (or the reverse for a
    cationic gel), using the total 1:1 salt concentration as cs. This links Eq. 2.2 to
    Eq. 2.1 and is the quantitative form of the thesis hypothesis.
    """
    cs = c_bath["Na"] + c_bath["K"]
    c_counter, c_co = donnan_concentrations(c_fixed_mM, cs)
    f_cat, f_an = (c_counter / cs, c_co / cs) if sign < 0 else (c_co / cs, c_counter / cs)
    c_local = {"K": c_bath["K"] * f_cat, "Na": c_bath["Na"] * f_cat, "Cl": c_bath["Cl"] * f_an}
    return ghk_vmem_mV(P, c_local, c_in, T)


# ----------------------------------------------------------------------------
# Rubber elasticity / Flory-Rehner
# ----------------------------------------------------------------------------
def chain_density_mol_m3(G_Pa, T: float = 298.0):
    """nu ~ G'/RT (ideal affine network)."""
    return np.asarray(G_Pa) / (R * T)


def mesh_size_nm(G_Pa, T: float = 298.0):
    """xi ~ (kB T / G')^(1/3), in nm."""
    return (KB * T / np.asarray(G_Pa, dtype=float)) ** (1 / 3) * 1e9


def concentration_scaling_exponent(G_low, G_high, c_low=10.0, c_high=20.0):
    """Apparent exponent a in G' ~ c^a (a ~ 2.25 for an ideal network in good solvent)."""
    return np.log(G_high / G_low) / np.log(c_high / c_low)


def flory_rehner_nu(phi: float, chi: float, V1: float = 18.0e-6) -> float:
    """Chain density (mol/m^3) from equilibrium polymer volume fraction (Eq. 2.5)."""
    lhs = -(np.log(1 - phi) + phi + chi * phi ** 2)
    return lhs / (V1 * (phi ** (1 / 3) - phi / 2))


def flory_rehner_phi(nu_mol_m3: float, chi: float, V1: float = 18.0e-6,
                     pi_donnan_Pa: float = 0.0, T: float = 298.15) -> float:
    """Solve Eq. 2.5 for phi, optionally adding a Donnan osmotic pressure to the mixing term.

    Mixing + ionic = elastic:
        -(RT/V1)[ln(1-phi)+phi+chi phi^2] + Pi_D = RT nu (phi^(1/3) - phi/2)
    """
    def f(phi):
        mix = -(R * T / V1) * (np.log(1 - phi) + phi + chi * phi ** 2)
        el = R * T * nu_mol_m3 * (phi ** (1 / 3) - phi / 2)
        return mix + pi_donnan_Pa - el
    grid = np.logspace(-8, np.log10(0.999), 400)
    vals = np.array([f(g) for g in grid])
    idx = np.where(np.sign(vals[:-1]) != np.sign(vals[1:]))[0]
    if idx.size == 0:
        raise ValueError("No equilibrium: Donnan pressure exceeds the elastic restoring pressure "
                         "at all phi (gel would dissolve / swell without bound).")
    i = idx[-1]
    return brentq(f, grid[i], grid[i + 1])


def mass_swelling_from_phi(phi: float, rho_polymer: float = 1.12, rho_solvent: float = 1.0) -> float:
    """Q_m = Ws/Wd from polymer volume fraction."""
    return 1 + (1 / phi - 1) * rho_solvent / rho_polymer



# ============================================================================================
# FORMULATION
# ============================================================================================
# Precursor formulation and geometry calculators.
#
# Reproduces
# ----------
# * Table 3.1  - single-network formulation matrix (2,000 uL per formulation)
# * Table 3.2  - number of gels in the experimental design
# * Sec. 3.2.2 - spacer-cast disc volume (V = pi r^2 h)
# * Sec. 3.2.4 / Table 4.5 - redox (APS / iron gluconate) dilutions, C = C_stock V_i / V_total
# * Eq. 3.6    - free volume and exposed surface area of the micropillar mold
# * Sec. 3.7   - sensitivity of the BCA depletion method for protein adsorption
#
# Note on the MBAA stock
# ----------------------
# The text of Sec. 3.2.2 states that MBAA was added from a 10 % w/v stock, but the
# volumes printed in Table 3.1 are reproduced exactly only with a 4 % w/v (40 mg/mL)
# stock. Likewise, the DEAEAAm volumes (400 / 800 uL for 0.2 / 0.4 g) correspond to a
# 50 % w/v stock rather than neat monomer. Defaults below reproduce Table 3.1; change
# `MBAA_STOCK_MG_PER_ML` / `Monomer.stock_mg_per_ml` if your bench stocks differ.
# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
MW_MBAA = 154.17          # g/mol, N,N'-methylenebis(acrylamide)
MBAA_STOCK_MG_PER_ML = 40.0   # reproduces Table 3.1 (see module docstring)
LAP_STOCK_PCT = 2.0       # % w/v
LAP_STOCK_UL = 100.0      # uL per 2 mL formulation -> 0.1 % w/v final
TOTAL_UL = 2000.0


@dataclass(frozen=True)
class Monomer:
    name: str
    charge: str
    mw: float | None              # g/mol (None for the crosslinking macromer PEGDA)
    stock_mg_per_ml: float        # mg of monomer per mL of stock as dispensed
    crosslinked_with_mbaa: bool = True


# Monomer stocks as used in Table 3.1
#   PEGDA Mn 700, neat, density 1.12 g/mL           -> 1120 mg/mL
#   AMPS-Na 50 % w/w aq., solution density ~1.2055   -> 602.8 mg AMPS per mL
#   DEAEAAm (stock giving 400 uL per 0.2 g)         -> 500 mg/mL
#   DMAAPS aqueous stock (500 uL per 0.2 g)         -> 400 mg/mL
MONOMERS: dict[str, Monomer] = {
    "PEGDA": Monomer("PEGDA", "neutral", None, 1120.0, crosslinked_with_mbaa=False),
    "AMPS": Monomer("AMPS", "anionic", 229.23, 0.5 * 1205.5),
    "DEAEAAm": Monomer("DEAEAAm", "cationic", 170.25, 500.0),
    "DMAAPS": Monomer("DMAAPS", "zwitterionic", 278.37, 400.0),
}


def formulation_recipe(monomer: str, conc_pct_wv: float, ratio: float | None = None,
                       total_ul: float = TOTAL_UL,
                       mbaa_stock_mg_per_ml: float = MBAA_STOCK_MG_PER_ML,
                       lap_stock_ul: float = LAP_STOCK_UL) -> dict:
    """Volumes (uL) of each stock needed for one formulation.

    Parameters
    ----------
    monomer : key of MONOMERS ("PEGDA", "AMPS", "DEAEAAm", "DMAAPS")
    conc_pct_wv : polymer concentration, % w/v (g per 100 mL)
    ratio : monomer-to-MBAA molar ratio (e.g. 10 for 10:1); ignored for PEGDA
    """
    m = MONOMERS[monomer]
    mass_mg = conc_pct_wv / 100.0 * total_ul          # g/100 mL * uL  == mg
    v_monomer = mass_mg / m.stock_mg_per_ml * 1000.0  # uL

    v_mbaa = 0.0
    mmol_mbaa = 0.0
    if m.crosslinked_with_mbaa:
        if ratio is None:
            raise ValueError(f"{monomer} needs a monomer:MBAA ratio")
        mmol_monomer = mass_mg / m.mw
        mmol_mbaa = mmol_monomer / ratio
        v_mbaa = mmol_mbaa * MW_MBAA / mbaa_stock_mg_per_ml * 1000.0

    v_diw = total_ul - v_monomer - v_mbaa - lap_stock_ul
    if v_diw < 0:
        raise ValueError(f"Stocks exceed total volume for {monomer} {conc_pct_wv}% {ratio}:1")
    return {
        "monomer": monomer,
        "charge": m.charge,
        "conc_pct_wv": conc_pct_wv,
        "ratio": f"{ratio:g}:1" if m.crosslinked_with_mbaa else "—",
        "monomer_stock_uL": round(v_monomer, 1),
        "MBAA_stock_uL": round(v_mbaa, 1) if m.crosslinked_with_mbaa else None,
        "MBAA_mM_final": round(mmol_mbaa / (total_ul * 1e-6), 2) if m.crosslinked_with_mbaa else None,
        "LAP_stock_uL": lap_stock_ul,
        "LAP_final_pct_wv": LAP_STOCK_PCT * lap_stock_ul / total_ul,
        "DIW_uL": round(v_diw, 1),
    }


def table_3_1(concs=(10, 20), ratios=(10, 25, 50), **kw) -> pd.DataFrame:
    """Rebuild Table 3.1 for every monomer / concentration / ratio."""
    rows = []
    for name, m in MONOMERS.items():
        for c in concs:
            if not m.crosslinked_with_mbaa:
                rows.append(formulation_recipe(name, c, **kw))
            else:
                for r in ratios:
                    rows.append(formulation_recipe(name, c, r, **kw))
    return pd.DataFrame(rows)


def table_3_2(n_per_test: int = 4, tests=("GF/SR", "DMA", "EIS")) -> pd.DataFrame:
    """Number of gels in the experimental design (Table 3.2)."""
    rows = []
    for name, m in MONOMERS.items():
        n_cond = 2 * (1 if not m.crosslinked_with_mbaa else 3)
        rows.append({"hydrogel": f"{name} ({m.charge})", "conditions": n_cond,
                     **{t: n_per_test for t in tests},
                     "total_gels": n_cond * n_per_test * len(tests)})
    df = pd.DataFrame(rows)
    df.loc[len(df)] = {"hydrogel": "Total", "total_gels": df["total_gels"].sum()}
    return df


# ----------------------------------------------------------------------------
# Geometry
# ----------------------------------------------------------------------------
def disc_volume_ul(diameter_mm: float = 8.0, thickness_mm: float = 1.0) -> float:
    """V = pi r^2 h, in uL (1 mm^3 = 1 uL). 8 mm x 1 mm -> 50.3 uL."""
    return math.pi * (diameter_mm / 2) ** 2 * thickness_mm


def rheometer_gap_um(volume_ul: float = 50.0, plate_diameter_mm: float = 8.0) -> float:
    """Gap that a loaded volume fills between parallel plates (50 uL, 8 mm -> ~995 um)."""
    return volume_ul / (math.pi * (plate_diameter_mm / 2) ** 2) * 1000.0


def precursor_needed_ul(n_gels: int, volume_per_gel_ul: float | None = None,
                        overage: float = 0.0) -> float:
    volume_per_gel_ul = volume_per_gel_ul or disc_volume_ul()
    return n_gels * volume_per_gel_ul * (1 + overage)


@dataclass
class MicropillarWell:
    """Single-post well of the microtissue mold (Sec. 3.7). Dimensions in mm."""
    R: float = 1.5    # well radius (3.0 mm diameter)
    H: float = 1.5    # well depth
    r1: float = 1.3   # upper post section radius
    h1: float = 0.7
    r2: float = 0.9   # post base radius
    h2: float = 0.3

    @property
    def cavity_ul(self):
        return math.pi * self.R ** 2 * self.H

    @property
    def post_ul(self):
        return math.pi * self.r1 ** 2 * self.h1 + math.pi * self.r2 ** 2 * self.h2

    @property
    def free_volume_ul(self):
        """Eq. 3.6: V_free = pi R^2 H - (pi r1^2 h1 + pi r2^2 h2)."""
        return self.cavity_ul - self.post_ul

    @property
    def surface_area_mm2(self):
        """Exposed hydrogel area: top annulus + outer wall + post walls + step + base."""
        pi = math.pi
        top_annulus = pi * (self.R ** 2 - self.r1 ** 2)
        outer_wall = 2 * pi * self.R * self.H
        post_walls = 2 * pi * self.r1 * self.h1 + 2 * pi * self.r2 * self.h2
        step = pi * abs(self.r1 ** 2 - self.r2 ** 2)
        base = pi * (self.R ** 2 - self.r2 ** 2)
        return top_annulus + outer_wall + post_walls + step + base

    def summary(self, n_wells: int = 6, prep_volume_ul: float = 75.0) -> dict:
        return {
            "cavity_uL": self.cavity_ul,
            "post_uL": self.post_ul,
            "free_volume_uL": self.free_volume_ul,
            "mold_volume_uL": n_wells * self.free_volume_ul,
            "prep_volume_uL": prep_volume_ul,
            "pipetting_overage_pct": 100 * (prep_volume_ul / (n_wells * self.free_volume_ul) - 1),
            "area_per_well_cm2": self.surface_area_mm2 / 100,
            "area_per_mold_cm2": n_wells * self.surface_area_mm2 / 100,
        }


def adsorption_sensitivity(area_cm2: float = 0.306, gamma_ng_cm2: float = 50.0,
                           c0_ug_ml: float = 100.0, v_incubation_ul: float = 200.0,
                           v_elution_ul: float = 100.0,
                           bca_range_ug_ml=(20.0, 2000.0)) -> dict:
    """Sensitivity of indirect-depletion vs direct-elution for protein adsorption (Sec. 3.7).

    M_adsorbed = (C_initial - C_supernatant) * V ;  Gamma = M_adsorbed / A
    """
    m_ads_ng = gamma_ng_cm2 * area_cm2
    m_added_ug = c0_ug_ml * v_incubation_ul / 1000
    c_sup = (m_added_ug - m_ads_ng / 1000) / (v_incubation_ul / 1000)
    c_elute = (m_ads_ng / 1000) / (v_elution_ul / 1000)
    lo, hi = bca_range_ug_ml
    return {
        "protein_added_ug": m_added_ug,
        "adsorbed_ng": m_ads_ng,
        "supernatant_ug_ml": c_sup,
        "depletion_pct": 100 * (c0_ug_ml - c_sup) / c0_ug_ml,
        "supernatant_in_BCA_range": lo <= c_sup <= hi,
        "direct_elution_ug_ml": c_elute,
        "elution_in_BCA_range": lo <= c_elute <= hi,
    }


# ----------------------------------------------------------------------------
# Redox initiation (Sec. 3.2.4, Table 4.5)
# ----------------------------------------------------------------------------
def redox_dilution(v_each_ul: float, v_pegda_ul: float = 238.0, c_stock_mM: float = 100.0,
                   pegda_pct: float = 10.0) -> dict:
    """Final APS and IG concentration when `v_each_ul` of each stock is added.

    C = C_stock * V_initiator / V_total,  V_total = V_PEGDA + 2 V_initiator
    """
    v_total = v_pegda_ul + 2 * v_each_ul
    return {
        "APS_IG_added_uL_each": v_each_ul,
        "V_total_uL": v_total,
        "final_mM_each": c_stock_mM * v_each_ul / v_total,
        "final_PEGDA_pct_wv": pegda_pct * v_pegda_ul / v_total,
    }


def table_4_5(volumes=(2.0, 2.5, 6.25, 12.5, 25.0), bulk=(60.0, 2000.0)) -> pd.DataFrame:
    rows = [redox_dilution(v) for v in volumes]
    rows.append(redox_dilution(bulk[0], v_pegda_ul=bulk[1]))
    return pd.DataFrame(rows)


def stock_mass_g(c_mM: float, mw: float, v_ml: float) -> float:
    """Mass to weigh for a stock (100 mM APS, 2 mL, 228.2 g/mol -> 0.0456 g)."""
    return c_mM / 1000 * mw * v_ml / 1000


def youngs_from_shear(G: float, poisson: float = 0.5) -> float:
    """E = 2 G (1 + v); ~3 G for incompressible hydrogels (Sec. 4.2.5)."""
    return 2 * G * (1 + poisson)



# ============================================================================================
# IO
# ============================================================================================
# Loaders for instrument exports.
#
# TRIOS (TA Instruments HR20)
# ---------------------------
# TRIOS can export to .txt (tab-separated, with "[step]" blocks), .csv or .xlsx. Column
# names vary with the TRIOS version and export template, e.g. "Storage modulus (Pa)",
# "Storage modulus G' (Pa)", "G' (Pa)". `load_trios` reads any of these and returns a
# DataFrame with *canonical* column names:
#
#     time_s, Gp_Pa, Gpp_Pa, strain_pct, stress_Pa, omega_rad_s, freq_Hz, tan_delta, step
#
# Missing columns are simply absent. If your export uses an unusual header, add it to
# COLUMN_ALIASES.
#
# The analysis functions also accept plain tidy CSVs with the canonical names, which is
# what `scripts/make_demo_data.py` writes.
#
# CHI 700E (impedance)
# --------------------
# CHI text exports contain a header block followed by a table
# "Freq/Hz, Z'/ohm, Z"/ohm, Z/ohm, Phase/deg". `load_chi_impedance` returns
# freq_Hz, Zre_ohm, Zim_ohm (sign convention: Zim negative for capacitive response),
# Zmag_ohm, phase_deg.
COLUMN_ALIASES = {
    "time_s": ["step time", "time", "test time", "time (s)", "step time (s)"],
    "Gp_Pa": ["storage modulus", "g'", "g′", "gp", "storage modulus g'"],
    "Gpp_Pa": ["loss modulus", "g''", "g\"", "g″", "gpp", "loss modulus g''"],
    "strain_pct": ["oscillation strain", "strain", "osc. strain", "oscillation strain (%)"],
    "stress_Pa": ["oscillation stress", "stress", "osc. stress"],
    "omega_rad_s": ["angular frequency", "ang. frequency", "omega"],
    "freq_Hz": ["frequency", "oscillation frequency"],
    "tan_delta": ["tan(delta)", "tan delta", "tan(δ)", "loss tangent"],
    "torque_uNm": ["oscillation torque", "torque"],
    "normal_force_N": ["normal force", "axial force"],
}

_UNIT_RE = re.compile(r"\s*[\(\[].*?[\)\]]\s*")


def _clean(name: str) -> str:
    return _UNIT_RE.sub("", str(name)).strip().lower()


def _unit(name: str) -> str:
    m = re.search(r"[\(\[](.*?)[\)\]]", str(name))
    return m.group(1).strip() if m else ""


def canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename instrument columns to canonical names and convert units (kPa->Pa, MPa->Pa, ms->s)."""
    if all(c in COLUMN_ALIASES or c == "step" for c in df.columns):
        return df
    out = {}
    used = set()
    for col in df.columns:
        key = _clean(col)
        for canon, aliases in COLUMN_ALIASES.items():
            if canon in used:
                continue
            if key == canon.lower() or key in aliases:
                s = pd.to_numeric(df[col], errors="coerce")
                unit = _unit(col).lower()
                if canon.endswith("_Pa"):
                    s = s * {"kpa": 1e3, "mpa": 1e6}.get(unit, 1.0)
                if canon == "time_s" and unit in ("min",):
                    s = s * 60
                if canon == "strain_pct" and unit in ("", "-", "1") and s.max() <= 20:
                    pass  # assume already % ; TRIOS default is %
                if canon == "omega_rad_s" and unit == "hz":
                    canon = "freq_Hz"
                out[canon] = s
                used.add(canon)
                break
        if key == "step" and "step" not in out:
            out["step"] = df[col]
    res = pd.DataFrame(out)
    if "omega_rad_s" not in res and "freq_Hz" in res:
        res["omega_rad_s"] = 2 * np.pi * res["freq_Hz"]
    if "freq_Hz" not in res and "omega_rad_s" in res:
        res["freq_Hz"] = res["omega_rad_s"] / (2 * np.pi)
    return res


def _read_trios_txt(path: Path) -> pd.DataFrame:
    """Parse a TRIOS text export with one or more [step] blocks."""
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    blocks = re.split(r"^\[step\]\s*$", text, flags=re.MULTILINE | re.IGNORECASE)
    if len(blocks) == 1:  # plain table
        return pd.read_csv(_io.StringIO(text), sep=None, engine="python")
    frames = []
    for i, b in enumerate(blocks[1:], start=1):
        lines = [ln for ln in b.strip().splitlines() if ln.strip()]
        if len(lines) < 3:
            continue
        step_name = lines[0].strip()
        header = lines[1].split("\t")
        # line 2 is units in many TRIOS exports
        start = 2
        units = lines[2].split("\t")
        if not any(re.match(r"^-?\d", u.strip()) for u in units if u.strip()):
            header = [f"{h} ({u})" if u.strip() else h for h, u in zip(header, units)]
            start = 3
        data = pd.read_csv(_io.StringIO("\n".join(lines[start:])), sep="\t", header=None)
        data = data.iloc[:, :len(header)]
        data.columns = header[: data.shape[1]]
        data["step"] = f"{i}: {step_name}"
        frames.append(data)
    return pd.concat(frames, ignore_index=True)


def load_trios(path, sheet=None) -> pd.DataFrame:
    """Load a TRIOS export (.txt/.csv/.xlsx) and return canonical columns."""
    path = Path(path)
    suf = path.suffix.lower()
    if suf in (".xlsx", ".xls"):
        raw = pd.read_excel(path, sheet_name=sheet or 0)
        # TRIOS xlsx often has a units row directly under the header
        first = raw.iloc[0].astype(str)
        if not any(re.match(r"^-?\d", v) for v in first):
            raw.columns = [f"{c} ({u})" if u and u != "nan" else c for c, u in zip(raw.columns, first)]
            raw = raw.iloc[1:].reset_index(drop=True)
    elif suf == ".txt":
        raw = _read_trios_txt(path)
    else:
        raw = pd.read_csv(path)
    df = canonicalize_columns(raw)
    return df.dropna(how="all").reset_index(drop=True)


def split_steps(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Split a multi-step TRIOS file (time -> amplitude -> frequency sweep) by step."""
    if "step" not in df:
        return {"all": df}
    return {k: g.drop(columns="step").reset_index(drop=True) for k, g in df.groupby("step", sort=False)}


def classify_step(df: pd.DataFrame) -> str:
    """Guess whether a step is a 'time', 'amplitude' or 'frequency' sweep from what varies."""
    def spread(col):
        if col not in df or df[col].dropna().size < 3:
            return 0.0
        v = df[col].dropna().to_numpy()
        v = v[v > 0]
        return np.log10(v.max() / v.min()) if v.size else 0.0
    s = {"amplitude": spread("strain_pct"), "frequency": spread("omega_rad_s")}
    if max(s.values()) < 0.5:
        return "time"
    return max(s, key=s.get)


def load_chi_impedance(path) -> pd.DataFrame:
    """Load a CHI 700E C. impedance text export."""
    lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    start = None
    for i, ln in enumerate(lines):
        if re.match(r"^\s*freq", ln, flags=re.IGNORECASE):
            start = i
            break
    if start is None:
        raise ValueError(f"No 'Freq/Hz' header found in {path}")
    data = []
    for ln in lines[start + 1:]:
        parts = [p for p in re.split(r"[,\t ]+", ln.strip()) if p]
        if len(parts) < 3:
            continue
        try:
            data.append([float(p) for p in parts[:5]])
        except ValueError:
            continue
    arr = np.array(data)
    df = pd.DataFrame({"freq_Hz": arr[:, 0], "Zre_ohm": arr[:, 1], "Zim_ohm": arr[:, 2]})
    df["Zmag_ohm"] = arr[:, 3] if arr.shape[1] > 3 else np.hypot(df.Zre_ohm, df.Zim_ohm)
    df["phase_deg"] = arr[:, 4] if arr.shape[1] > 4 else np.degrees(np.arctan2(df.Zim_ohm, df.Zre_ohm))
    # CHI reports Z" as negative for capacitive behaviour; enforce that convention
    if (df.Zim_ohm > 0).mean() > 0.8:
        df["Zim_ohm"] = -df["Zim_ohm"]
    return df.sort_values("freq_Hz").reset_index(drop=True)


def load_table(path) -> pd.DataFrame:
    path = Path(path)
    return pd.read_excel(path) if path.suffix.lower() in (".xlsx", ".xls") else pd.read_csv(path)



# ============================================================================================
# RHEOLOGY
# ============================================================================================
# Rheological parameter extraction (Sec. 3.3, definitions in Sec. 3.3.3).
#
# Time sweep (in situ photorheology, 1 % strain, 1 Hz, UV 20 mW/cm^2)
#     GP   gel point: first time after UV at which G' > G''
#     FN   full network: first time after UV at which G' changed by <= 1 % over three
#          consecutive points
#     G'p  plateau modulus: mean G' of the final five points (peak G' also reported,
#          relevant for the 20 % PEGDA overshoot)
#     t90  rise time: from the onset of the G' rise (first point with G' > 10x the pre-cure
#          baseline) to the time G' first reaches 90 % of G'p
# Amplitude sweep (1 Hz, 0.01-100 %+ strain)
#     G'LVR  mean G' of the first five points
#     gc     critical strain: first strain at which G' < 0.9 G'LVR, interpolated on a log
#            strain axis; right-censored (> max strain) when never reached
#     yield  (IPN, Sec. 3.3.1) strain at which stress-strain deviates from linearity
# Frequency sweep (1 % strain, 0.01-100 Hz)
#     G'w   mean G' between 0.1 and 10 rad/s
#     n     slope of log G' vs log w in the same window (points > 100 rad/s excluded)
# Network
#     nu = G'p / RT,  xi = (kB T / G'p)^(1/3)  at 298 K
# ----------------------------------------------------------------------------
# Time sweep
# ----------------------------------------------------------------------------
@dataclass
class TimeSweepResult:
    uv_on_s: float
    GP_s: float                 # relative to UV on
    FN_s: float                 # relative to UV on
    t90_s: float
    onset_s: float              # relative to UV on
    Gp_plateau_Pa: float
    Gp_peak_Pa: float
    baseline_Pa: float
    overshoot: bool
    sample_dt_s: float          # median sampling interval (GP resolution)

    def as_dict(self):
        return asdict(self)


def detect_uv_on(df: pd.DataFrame, factor: float = 10.0, n_baseline: int = 3) -> float:
    """Estimate shutter-opening time if it was not logged: the sample before the first
    point whose G' exceeds `factor` x the median of the first `n_baseline` points.
    Prefer the logged time whenever it is available (Sec. 3.3.2)."""
    t = df["time_s"].to_numpy()
    g = df["Gp_Pa"].to_numpy()
    base = np.nanmedian(g[:n_baseline])
    idx = np.argmax(g > factor * max(base, 1e-12))
    return float(t[max(idx - 1, 0)])


def gel_point(t_rel: np.ndarray, Gp: np.ndarray, Gpp: np.ndarray, interpolate: bool = False) -> float:
    """First time (relative to UV) at which G' > G''. NaN if never crossed.

    With interpolate=False (thesis default) the sampled time is returned, so GP is
    resolved only to the sampling interval (~6 s)."""
    after = t_rel > 0
    cross = np.where(after & (Gp > Gpp))[0]
    if cross.size == 0:
        return np.nan
    i = cross[0]
    if not interpolate or i == 0 or not after[i - 1]:
        return float(t_rel[i])
    # log-linear interpolation of tan(delta) = 1
    d0 = np.log(Gp[i - 1] / Gpp[i - 1])
    d1 = np.log(Gp[i] / Gpp[i])
    return float(t_rel[i - 1] + (t_rel[i] - t_rel[i - 1]) * (-d0) / (d1 - d0))


def full_network_time(t_rel: np.ndarray, Gp: np.ndarray, tol: float = 0.01, window: int = 3,
                      mode: str = "consecutive", start_after: float | None = None) -> float:
    """First time after UV at which G' changed by <= tol over `window` consecutive points.

    mode="consecutive": every point-to-point relative change inside the window <= tol
    mode="span"       : (max - min) / min inside the window <= tol
    start_after       : ignore windows before this time (defaults to the gel point, so
                        the flat liquid baseline is never mistaken for a plateau)
    Returns the time of the first point of the window.
    """
    start = 0.0 if start_after is None else start_after
    idx = np.where(t_rel >= start)[0]
    for i in idx:
        if i + window > len(Gp):
            break
        w = Gp[i:i + window]
        if np.any(~np.isfinite(w)) or np.any(w <= 0):
            continue
        if mode == "span":
            ok = (w.max() - w.min()) / w.min() <= tol
        else:
            ok = np.all(np.abs(np.diff(w)) / w[:-1] <= tol)
        if ok:
            return float(t_rel[i])
    return np.nan


def analyze_time_sweep(df: pd.DataFrame, uv_on_s: float | None = None, n_plateau: int = 5,
                       onset_factor: float = 10.0, fn_tol: float = 0.01, fn_window: int = 3,
                       fn_mode: str = "consecutive", interpolate_gp: bool = False,
                       overshoot_threshold: float = 1.25) -> TimeSweepResult:
    """Extract GP, FN, t90, G'p from one replicate time sweep.

    df needs time_s, Gp_Pa, Gpp_Pa. uv_on_s is the logged shutter-opening time
    (experiment time); if None it is estimated with `detect_uv_on`.
    """
    df = df.dropna(subset=["time_s", "Gp_Pa"]).sort_values("time_s")
    if uv_on_s is None:
        uv_on_s = detect_uv_on(df, onset_factor)
        warnings.warn(f"UV shutter time not given; estimated {uv_on_s:.1f} s")
    t = df["time_s"].to_numpy(float)
    Gp = df["Gp_Pa"].to_numpy(float)
    Gpp = df["Gpp_Pa"].to_numpy(float) if "Gpp_Pa" in df else np.full_like(Gp, np.nan)
    t_rel = t - uv_on_s

    pre = Gp[t_rel <= 0]
    baseline = float(np.nanmedian(pre)) if pre.size else float(Gp[0])

    GP = gel_point(t_rel, Gp, Gpp, interpolate_gp)
    plateau = float(np.nanmean(Gp[-n_plateau:]))
    peak = float(np.nanmax(Gp[t_rel > 0])) if np.any(t_rel > 0) else np.nan

    on = np.where((t_rel > 0) & (Gp > onset_factor * max(baseline, 1e-12)))[0]
    onset = float(t_rel[on[0]]) if on.size else np.nan
    reach = np.where((t_rel > 0) & (Gp >= 0.9 * plateau))[0]
    t90 = float(t_rel[reach[0]] - onset) if (reach.size and on.size) else np.nan

    FN = full_network_time(t_rel, Gp, fn_tol, fn_window, fn_mode,
                           start_after=GP if np.isfinite(GP) else 0.0)
    return TimeSweepResult(uv_on_s=float(uv_on_s), GP_s=GP, FN_s=FN, t90_s=t90, onset_s=onset,
                           Gp_plateau_Pa=plateau, Gp_peak_Pa=peak, baseline_Pa=baseline,
                           overshoot=bool(peak > overshoot_threshold * plateau),
                           sample_dt_s=float(np.median(np.diff(t))))


# ----------------------------------------------------------------------------
# Amplitude sweep
# ----------------------------------------------------------------------------
@dataclass
class AmplitudeSweepResult:
    G_LVR_Pa: float
    gamma_c_pct: float
    gamma_c_censored: bool          # True -> reported as "> max strain"
    max_strain_pct: float
    gamma_half_pct: float           # strain at which G' = 0.5 G'LVR (NaN if not reached)
    yield_strain_pct: float         # stress-strain linearity criterion (IPN definition)

    def as_dict(self):
        return asdict(self)


def _log_interp_crossing(x, y, level):
    """First x at which y drops below `level`, interpolated linearly in log10(x)."""
    below = np.where(y < level)[0]
    if below.size == 0:
        return np.nan
    i = below[0]
    if i == 0:
        return float(x[0])
    lx0, lx1 = np.log10(x[i - 1]), np.log10(x[i])
    y0, y1 = y[i - 1], y[i]
    frac = (y0 - level) / (y0 - y1) if y0 != y1 else 0.0
    return float(10 ** (lx0 + frac * (lx1 - lx0)))


def stress_strain_yield(strain_pct, stress_Pa=None, Gp_Pa=None, n_lvr: int = 5,
                        tol: float = 0.05) -> float:
    """Strain at which the stress-strain response deviates from linearity by > tol.

    Linear reference sigma = G'LVR * gamma; if stress is not exported, sigma ~ |G*| gamma
    is approximated by G' gamma. Interpolated on a log-strain axis.
    """
    g = np.asarray(strain_pct, float) / 100
    if stress_Pa is None:
        stress = np.asarray(Gp_Pa, float) * g
    else:
        stress = np.asarray(stress_Pa, float)
    slope = np.nanmedian(stress[:n_lvr] / g[:n_lvr])
    ratio = stress / (slope * g)
    return _log_interp_crossing(np.asarray(strain_pct, float), ratio, 1 - tol)


def analyze_amplitude_sweep(df: pd.DataFrame, n_lvr: int = 5, drop: float = 0.10,
                            yield_tol: float = 0.05) -> AmplitudeSweepResult:
    df = df.dropna(subset=["strain_pct", "Gp_Pa"]).sort_values("strain_pct")
    df = df[df["strain_pct"] > 0]
    x = df["strain_pct"].to_numpy(float)
    y = df["Gp_Pa"].to_numpy(float)
    G_lvr = float(np.nanmean(y[:n_lvr]))
    gc = _log_interp_crossing(x, y, (1 - drop) * G_lvr)
    censored = not np.isfinite(gc)
    stress = df["stress_Pa"].to_numpy(float) if "stress_Pa" in df else None
    return AmplitudeSweepResult(
        G_LVR_Pa=G_lvr,
        gamma_c_pct=float(x.max()) if censored else gc,
        gamma_c_censored=censored,
        max_strain_pct=float(x.max()),
        gamma_half_pct=_log_interp_crossing(x, y, 0.5 * G_lvr),
        yield_strain_pct=stress_strain_yield(x, stress, y, n_lvr, yield_tol),
    )


# ----------------------------------------------------------------------------
# Frequency sweep
# ----------------------------------------------------------------------------
@dataclass
class FrequencySweepResult:
    G_omega_Pa: float
    slope_n: float
    slope_r2: float
    n_points: int

    def as_dict(self):
        return asdict(self)


def analyze_frequency_sweep(df: pd.DataFrame, window=(0.1, 10.0),
                            inertia_cutoff: float = 100.0) -> FrequencySweepResult:
    df = df.dropna(subset=["omega_rad_s", "Gp_Pa"])
    df = df[(df["omega_rad_s"] <= inertia_cutoff) & (df["Gp_Pa"] > 0)]
    sel = df[(df["omega_rad_s"] >= window[0]) & (df["omega_rad_s"] <= window[1])]
    if len(sel) < 2:
        return FrequencySweepResult(np.nan, np.nan, np.nan, len(sel))
    lx = np.log10(sel["omega_rad_s"].to_numpy(float))
    ly = np.log10(sel["Gp_Pa"].to_numpy(float))
    n, b = np.polyfit(lx, ly, 1)
    pred = n * lx + b
    ss_res = np.sum((ly - pred) ** 2)
    ss_tot = np.sum((ly - ly.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return FrequencySweepResult(float(sel["Gp_Pa"].mean()), float(n), float(r2), len(sel))


# ----------------------------------------------------------------------------
# Replicate handling and summaries
# ----------------------------------------------------------------------------
def find_duplicate_replicates(curves: dict[str, pd.DataFrame], col: str = "Gp_Pa") -> list[tuple]:
    """Detect replicates whose exported curves are identical (cf. 10 % PEGDA rep 3 = rep 1)."""
    keys = list(curves)
    dups = []
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            ca, cb = curves[a][col].to_numpy(), curves[b][col].to_numpy()
            if ca.shape == cb.shape and np.allclose(ca, cb, equal_nan=True):
                dups.append((a, b))
    return dups


def network_parameters(Gp_Pa: float, T: float = 298.0) -> dict:
    return {"nu_mol_m3": float(chain_density_mol_m3(Gp_Pa, T)), "xi_nm": float(mesh_size_nm(Gp_Pa, T)),
            "E_kPa_approx": 3 * Gp_Pa / 1000}


def mean_sd(x, digits=None) -> str:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return "—"
    m, s = x.mean(), (x.std(ddof=1) if x.size > 1 else 0.0)
    if digits is None:
        digits = 2 if abs(m) < 10 else (1 if abs(m) < 100 else 0)
    return f"{m:.{digits}f} ± {s:.{digits}f}"


def median_range(x, censored=None, digits=None) -> str:
    """'median [min-max]' as in Table 4.3; '> max (no yield)' if all censored."""
    x = np.asarray(x, float)
    c = np.zeros_like(x, bool) if censored is None else np.asarray(censored, bool)
    if c.all():
        return f"> {x.min():.0f} (no yield)"
    med = np.median(x)
    fmt = lambda v: f"{v:.{digits}f}" if digits is not None else (f"{v:.2g}" if v < 10 else f"{v:.0f}")
    return f"{fmt(med)} [{fmt(x.min())}–{fmt(x.max())}]" + (" †" if c.any() else "")


def average_curves(curves: list[pd.DataFrame], x: str, y: str = "Gp_Pa", n: int = 200,
                   log_x: bool = True, x_grid=None) -> pd.DataFrame:
    """Mean ± SD of replicate curves on a common grid (log-interpolated), for figures."""
    xs = [c[x].to_numpy(float) for c in curves]
    lo = max(np.nanmin(v[v > 0] if log_x else v) for v in xs)
    hi = min(np.nanmax(v) for v in xs)
    if x_grid is None:
        x_grid = np.logspace(np.log10(lo), np.log10(hi), n) if log_x else np.linspace(lo, hi, n)
    Y = []
    for c in curves:
        c = c.dropna(subset=[x, y]).sort_values(x)
        cx, cy = c[x].to_numpy(float), c[y].to_numpy(float)
        if log_x:
            m = (cx > 0) & (cy > 0)
            Y.append(10 ** np.interp(np.log10(x_grid), np.log10(cx[m]), np.log10(cy[m]),
                                     left=np.nan, right=np.nan))
        else:
            m = cy > 0
            Y.append(10 ** np.interp(x_grid, cx[m], np.log10(cy[m]), left=np.nan, right=np.nan))
    Y = np.vstack(Y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(Y, 0)
        sd = np.nanstd(Y, 0, ddof=1) if len(curves) > 1 else np.zeros_like(mean)
    return pd.DataFrame({x: x_grid, "mean": mean, "sd": sd})



# ============================================================================================
# GRAVIMETRY
# ============================================================================================
# Equilibrium swelling and gel fraction (Sec. 3.4-3.5).
#
#     SR      = Ws / Wd                          (Eq. 3.1)
#     EWC (%) = (Ws - Wd) / Ws x 100             (Eq. 3.2)
#     GF (%)  = Wf / Wi x 100                    (Eq. 3.3)
#     SF (%)  = (Wi - Wf) / Wi x 100 = 100 - GF  (Eq. 3.4)
#
# PBS salt correction
# -------------------
# When a gel swollen in PBS is lyophilised, the dissolved salts of the imbibed buffer stay
# behind and inflate Wd. The salt mass is estimated from the mass of water removed
# (Ws - Wd) and the dissolved-solids content of PBS, then subtracted from Wd before
# Eq. 3.1 is applied. 1x PBS (Gibco 10010, pH 7.4) contains ~9.6 g/L dissolved solids
# (NaCl 8.0, KCl 0.2, Na2HPO4 1.15 (anhydrous basis), KH2PO4 0.2 g/L); adjust
# `PBS_SOLIDS_G_PER_L` for your formulation.
PBS_SOLIDS_G_PER_L = 9.55
WATER_DENSITY = 0.997  # g/mL at 25 C


def salt_mass(Ws, Wd, solids_g_per_L: float = PBS_SOLIDS_G_PER_L):
    """Residual salt mass (same units as Ws/Wd) left by evaporated buffer.

    Mass of water removed = Ws - Wd_measured. Buffer solids per gram of water
    ~ solids_g_per_L / (1000 * rho_water).
    """
    water = np.asarray(Ws, float) - np.asarray(Wd, float)
    per_g_water = solids_g_per_L / (1000 * WATER_DENSITY)
    # Wd_measured = polymer + salt, water = Ws - polymer - salt  -> salt = water * per_g_water
    return water * per_g_water


def swelling(Ws, Wd, medium: str = "DIW", solids_g_per_L: float = PBS_SOLIDS_G_PER_L) -> pd.DataFrame:
    """Swelling ratio and EWC, with salt correction for saline media."""
    Ws = np.asarray(Ws, float)
    Wd = np.asarray(Wd, float)
    salt = salt_mass(Ws, Wd, solids_g_per_L) if medium.upper() in ("PBS", "DPBS", "SALINE") else 0 * Ws
    Wp = Wd - salt
    return pd.DataFrame({
        "Ws": Ws, "Wd_measured": Wd, "salt": salt, "Wd_polymer": Wp,
        "SR": Ws / Wp,
        "SR_uncorrected": Ws / Wd,
        "EWC_pct": (Ws - Wp) / Ws * 100,
    })


def gel_fraction(Wi, Wf) -> pd.DataFrame:
    Wi = np.asarray(Wi, float)
    Wf = np.asarray(Wf, float)
    gf = Wf / Wi * 100
    return pd.DataFrame({"Wi": Wi, "Wf": Wf, "GF_pct": gf, "SF_pct": 100 - gf})


def is_constant_mass(masses, rel_tol: float = 0.005) -> bool:
    """Lyophilisation end-point check: last two weighings differ by < rel_tol."""
    m = np.asarray(masses, float)
    return m.size >= 2 and abs(m[-1] - m[-2]) / m[-2] < rel_tol


def analyze_gravimetry_table(df: pd.DataFrame) -> pd.DataFrame:
    """Row-wise analysis of a tidy sheet.

    Expected columns: group, medium, replicate, Ws, Wd   (swelling)
               and/or group, replicate, Wi, Wf           (gel fraction)
    Extra columns (e.g. printer) are carried through.
    """
    out = df.copy()
    if {"Ws", "Wd"}.issubset(df.columns):
        res = []
        for _, r in df.iterrows():
            s = swelling([r.Ws], [r.Wd], r.get("medium", "DIW")).iloc[0]
            res.append(s[["salt", "Wd_polymer", "SR", "SR_uncorrected", "EWC_pct"]])
        out = pd.concat([out.reset_index(drop=True), pd.DataFrame(res).reset_index(drop=True)], axis=1)
    if {"Wi", "Wf"}.issubset(df.columns):
        g = gel_fraction(df["Wi"], df["Wf"])
        out["GF_pct"] = g["GF_pct"].to_numpy()
        out["SF_pct"] = g["SF_pct"].to_numpy()
    return out



# ============================================================================================
# EIS
# ============================================================================================
# Ionic conductivity from electrochemical impedance spectroscopy (Sec. 3.6).
#
# CHI 700E, A.C. impedance, 0 V initial potential, 20 mV amplitude, 1 Hz - 1 MHz,
# 10 points/decade, 2 s quiet time.
#
# Bulk resistance R_bulk is taken from the *low-frequency* real-axis region of the Nyquist
# plot (Z' vs -Z''), because the high-frequency intercept in this configuration reflects
# series and contact resistance rather than ion transport through the gel [213]. Then
#
#     sigma = L / (R_bulk A)      (Eq. 3.5),     rho = 1 / sigma
#
# Methods available in `bulk_resistance`:
#     "lowfreq_intercept"  linear extrapolation of the lowest-frequency Nyquist points to
#                          -Z'' = 0 (default, thesis method)
#     "lowfreq_plateau"    mean |Z| over the low-frequency Bode plateau (1-10 Hz)
#     "min_imag"           Z' at the point where |Z''| is smallest within the low-f region
#     "hf_intercept"       high-frequency intercept (series/contact resistance; for comparison)
def bulk_resistance(df: pd.DataFrame, method: str = "lowfreq_intercept",
                    lowf_band=(1.0, 10.0), n_fit: int = 4) -> float:
    d = df.sort_values("freq_Hz")
    f = d["freq_Hz"].to_numpy(float)
    zr = d["Zre_ohm"].to_numpy(float)
    zi = d["Zim_ohm"].to_numpy(float)          # negative for capacitive
    nzi = -zi

    if method == "lowfreq_plateau":
        m = (f >= lowf_band[0]) & (f <= lowf_band[1])
        return float(np.mean(np.hypot(zr[m], zi[m])))
    if method == "min_imag":
        m = (f >= lowf_band[0]) & (f <= lowf_band[1])
        return float(zr[m][np.argmin(np.abs(nzi[m]))])
    if method == "hf_intercept":
        k = np.argsort(f)[::-1][:n_fit]
        p = np.polyfit(nzi[k], zr[k], 1)
        return float(np.polyval(p, 0.0))
    if method == "lowfreq_intercept":
        k = np.argsort(f)[:n_fit]
        if np.ptp(nzi[k]) < 1e-9 * max(1.0, np.abs(zr[k]).max()):
            return float(zr[k].mean())
        p = np.polyfit(nzi[k], zr[k], 1)           # Z' as a function of -Z''
        return float(np.polyval(p, 0.0))
    raise ValueError(method)


def conductivity(R_bulk_ohm, thickness_cm: float, area_cm2: float) -> dict:
    """sigma (mS/cm) and rho (ohm m) from Eq. 3.5."""
    R = np.asarray(R_bulk_ohm, float)
    sigma_S_cm = thickness_cm / (R * area_cm2)
    return {"R_bulk_ohm": R, "sigma_mS_cm": sigma_S_cm * 1e3,
            "rho_ohm_m": 1 / (sigma_S_cm * 100)}


def cell_constant_from_reference(R_ref_ohm: float, sigma_ref_mS_cm: float) -> float:
    """Calibrate L/A (cm^-1) with a reference solution of known conductivity (e.g. PBS ~15 mS/cm
    or a KCl standard): L/A = sigma_ref * R_ref."""
    return sigma_ref_mS_cm * 1e-3 * R_ref_ohm


def analyze_spectra(spectra: dict[str, pd.DataFrame], thickness_cm: float, area_cm2: float,
                    method: str = "lowfreq_intercept", **kw) -> pd.DataFrame:
    rows = []
    for name, df in spectra.items():
        R = bulk_resistance(df, method, **kw)
        c = conductivity(R, thickness_cm, area_cm2)
        rows.append({"sample": name, "R_bulk_ohm": R, "sigma_mS_cm": float(c["sigma_mS_cm"]),
                     "rho_ohm_m": float(c["rho_ohm_m"]),
                     "R_hf_ohm": bulk_resistance(df, "hf_intercept")})
    return pd.DataFrame(rows)


def simulate_spectrum(R_bulk: float, R_hf: float = 20.0, tau_s: float = 1e-4, alpha: float = 0.8,
                      f_min: float = 1.0, f_max: float = 1e6, ppd: int = 10, noise: float = 0.0,
                      rng=None) -> pd.DataFrame:
    """Synthetic spectrum with a low-f plateau at R_bulk and a high-f plateau at R_hf
    (Cole element), matching the shape described in Sec. 4.4. For demo/testing only."""
    rng = np.random.default_rng(rng)
    n = int(round(np.log10(f_max / f_min) * ppd)) + 1
    f = np.logspace(np.log10(f_max), np.log10(f_min), n)
    w = 2 * np.pi * f
    Z = R_hf + (R_bulk - R_hf) / (1 + (1j * w * tau_s) ** alpha)
    Z = Z * (1 + noise * rng.standard_normal(n))
    return pd.DataFrame({"freq_Hz": f, "Zre_ohm": Z.real, "Zim_ohm": Z.imag,
                         "Zmag_ohm": np.abs(Z), "phase_deg": np.degrees(np.angle(Z))})



# ============================================================================================
# STATS
# ============================================================================================
# Statistical analysis (Sec. 3.14).
#
# * mean ± SD of n = 4 independently prepared gels
# * two-sided Grubbs outlier test, alpha = 0.05 (G_crit = 1.481 for n = 4)
# * two-way ANOVA (concentration x ratio, with interaction) on log-transformed moduli and
#   critical strains, run separately for each chemistry  -> Table 4.4
# * planned pairwise comparisons by Welch's unequal-variance t-test, Holm-corrected over the
#   four comparisons within each chemistry
# * one-way ANOVA with Tukey's HSD (IPN data, Figs 4.1, 4.3, 4.13)
# * two-way ANOVA printer x medium followed by Welch's t-tests (Table 4.6)
def stars(p: float) -> str:
    if not np.isfinite(p):
        return ""
    return "****" if p < 1e-4 else "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "ns"


def fmt_p(p: float) -> str:
    return "p < 0.0001" if p < 1e-4 else f"p = {p:.2g}" if p < 0.01 else f"p = {p:.2f}"


# ----------------------------------------------------------------------------
# Outliers
# ----------------------------------------------------------------------------
def grubbs_critical(n: int, alpha: float = 0.05) -> float:
    """Two-sided Grubbs critical value."""
    t = st.t.ppf(1 - alpha / (2 * n), n - 2)
    return (n - 1) / np.sqrt(n) * np.sqrt(t ** 2 / (n - 2 + t ** 2))


def grubbs_test(x, alpha: float = 0.05) -> dict:
    """Single-pass two-sided Grubbs test. Returns G, G_crit, index of suspect, is_outlier."""
    x = np.asarray(x, float)
    n = x.size
    if n < 3:
        return {"G": np.nan, "G_crit": np.nan, "index": None, "value": None, "outlier": False}
    dev = np.abs(x - x.mean())
    i = int(np.argmax(dev))
    G = dev[i] / x.std(ddof=1)
    Gc = grubbs_critical(n, alpha)
    return {"G": float(G), "G_crit": float(Gc), "index": i, "value": float(x[i]), "outlier": bool(G > Gc)}


# ----------------------------------------------------------------------------
# t-tests
# ----------------------------------------------------------------------------
def welch(a, b, log: bool = False) -> dict:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if log:
        a, b = np.log10(a), np.log10(b)
    t, p = st.ttest_ind(a, b, equal_var=False)
    return {"t": float(t), "p": float(p)}


def planned_comparisons(df: pd.DataFrame, response: str, conc_col: str = "conc",
                        ratio_col: str = "ratio", log: bool = True, method: str = "holm") -> pd.DataFrame:
    """Ratio effect at each concentration and concentration effect at each ratio
    (four Welch tests), Holm-adjusted within the chemistry."""
    rows = []
    concs = sorted(df[conc_col].unique())
    ratios = sorted(df[ratio_col].unique())
    for c in concs:
        r0, r1 = ratios[:2]
        a = df[(df[conc_col] == c) & (df[ratio_col] == r0)][response].dropna()
        b = df[(df[conc_col] == c) & (df[ratio_col] == r1)][response].dropna()
        rows.append({"comparison": f"{c}%: {r0}:1 vs {r1}:1", **welch(a, b, log),
                     "mean_a": a.mean(), "mean_b": b.mean()})
    for r in ratios:
        c0, c1 = concs[:2]
        a = df[(df[conc_col] == c0) & (df[ratio_col] == r)][response].dropna()
        b = df[(df[conc_col] == c1) & (df[ratio_col] == r)][response].dropna()
        rows.append({"comparison": f"{r}:1: {c0}% vs {c1}%", **welch(a, b, log),
                     "mean_a": a.mean(), "mean_b": b.mean()})
    out = pd.DataFrame(rows)
    out["fold"] = out[["mean_a", "mean_b"]].max(axis=1) / out[["mean_a", "mean_b"]].min(axis=1)
    out["p_adj"] = multipletests(out["p"], method=method)[1]
    out["sig"] = out["p_adj"].map(stars)
    return out


# ----------------------------------------------------------------------------
# ANOVA
# ----------------------------------------------------------------------------
def two_way_anova(df: pd.DataFrame, response: str, a: str = "conc", b: str = "ratio",
                  log: bool = True, typ: int = 2) -> pd.DataFrame:
    """Two-way ANOVA with interaction. Returns F, p, df for each effect."""
    d = df[[response, a, b]].dropna().copy()
    if log:
        d = d[d[response] > 0]
        d["_y"] = np.log10(d[response])
    else:
        d["_y"] = d[response]
    model = smf.ols(f"_y ~ C({a}) * C({b})", data=d).fit()
    tab = anova_lm(model, typ=typ)
    resid_df = tab.loc["Residual", "df"]
    out = tab.drop(index="Residual").rename(index={f"C({a})": a, f"C({b})": b, f"C({a}):C({b})": f"{a}:{b}"})
    out = out.rename(columns={"PR(>F)": "p"})
    out["df_resid"] = resid_df
    out["sig"] = out["p"].map(stars)
    return out[["df", "df_resid", "F", "p", "sig"]]


def anova_cell(row) -> str:
    return f"F = {row.F:.3g}, {fmt_p(row.p)}"


def one_way_anova_tukey(df: pd.DataFrame, response: str, group: str = "group", alpha: float = 0.05):
    """One-way ANOVA followed by Tukey's HSD (GraphPad-equivalent)."""
    d = df[[response, group]].dropna()
    samples = [g[response].to_numpy() for _, g in d.groupby(group, sort=False)]
    F, p = st.f_oneway(*samples)
    tk = pairwise_tukeyhsd(d[response], d[group], alpha=alpha)
    tk_df = pd.DataFrame(tk.summary().data[1:], columns=tk.summary().data[0])
    # exact p-values (statsmodels rounds in summary)
    tk_df["p-adj"] = tk.pvalues
    tk_df["sig"] = tk_df["p-adj"].map(stars)
    return {"F": float(F), "p": float(p), "df_between": len(samples) - 1,
            "df_within": len(d) - len(samples), "tukey": tk_df}


def all_pairs_welch(df: pd.DataFrame, response: str, group: str, log: bool = False,
                    method: str | None = None) -> pd.DataFrame:
    rows = []
    for g1, g2 in itertools.combinations(df[group].unique(), 2):
        a = df[df[group] == g1][response]
        b = df[df[group] == g2][response]
        rows.append({"a": g1, "b": g2, **welch(a, b, log)})
    out = pd.DataFrame(rows)
    if method:
        out["p_adj"] = multipletests(out["p"], method=method)[1]
    return out


# ----------------------------------------------------------------------------
# Descriptives
# ----------------------------------------------------------------------------
def describe(df: pd.DataFrame, response: str, by) -> pd.DataFrame:
    g = df.groupby(by, sort=False)[response]
    return pd.DataFrame({"n": g.count(), "mean": g.mean(), "sd": g.std(ddof=1),
                         "median": g.median(), "min": g.min(), "max": g.max()})


def tost_welch(a, b, margin: float, log: bool = True) -> dict:
    """Two one-sided tests for equivalence (suggested in Sec. 4.2.5 for matched pairs).

    margin is on the analysed scale; with log=True, margin=log10(1.25) tests whether the
    ratio of geometric means lies within 0.8-1.25.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    if log:
        a, b = np.log10(a), np.log10(b)
    d = a.mean() - b.mean()
    va, vb = a.var(ddof=1) / a.size, b.var(ddof=1) / b.size
    se = np.sqrt(va + vb)
    dof = (va + vb) ** 2 / (va ** 2 / (a.size - 1) + vb ** 2 / (b.size - 1))
    p_low = 1 - st.t.cdf((d + margin) / se, dof)
    p_high = st.t.cdf((d - margin) / se, dof)
    return {"diff": d, "p_lower": p_low, "p_upper": p_high, "p_tost": max(p_low, p_high),
            "equivalent": max(p_low, p_high) < 0.05}



# ============================================================================================
# ASSAYS
# ============================================================================================
# Biochemical assay analysis.
#
# * Standard curves (BCA / BSA, PicoGreen / lambda-DNA): linear fit, R^2, inverse
#   prediction, range checking and dilution correction (Sec. 3.8, 3.11.4, Fig. 4.15)
# * Cumulative release with full-volume replacement at 1, 6, 12, 24, 72 h (Sec. 3.8)
# * ALP activity: slope of A405 vs time over the linear region, normalised to protein,
#   nmol p-nitrophenol / min / mg protein  (Eq. 3.7)
# * MTT: blank (cell-free hydrogel) correction, reference-wavelength subtraction, % control
# * ELISA (BMP-2 DuoSet): four-parameter logistic standard curve and back-calculation
EPS_PNP = 18500.0  # M^-1 cm^-1, p-nitrophenolate at 405 nm


# ----------------------------------------------------------------------------
# Linear standard curves
# ----------------------------------------------------------------------------
@dataclass
class LinearStandard:
    slope: float
    intercept: float
    r2: float
    x_min: float
    x_max: float
    y_min: float
    y_max: float

    def predict_conc(self, signal, dilution=1.0, flag=True):
        """Inverse prediction x = (y - b) / m, times the dilution factor.

        Returns (conc, in_range). Signals above/below the standards are flagged so the
        sample can be diluted and re-assayed (Sec. 4.5)."""
        y = np.asarray(signal, float)
        x = (y - self.intercept) / self.slope * np.asarray(dilution, float)
        in_range = (y >= self.y_min) & (y <= self.y_max)
        return (x, in_range) if flag else x

    def label(self):
        return f"y = {self.slope:.4g}x + {self.intercept:.4g}, R² = {self.r2:.3f}"


def fit_linear_standard(conc, signal, blank=None, through_origin: bool = False) -> LinearStandard:
    """Fit a standard curve. If `blank` is given it is subtracted from the signals first.
    Replicate wells can be passed as repeated concentrations."""
    x = np.asarray(conc, float)
    y = np.asarray(signal, float) - (np.mean(blank) if blank is not None else 0.0)
    if through_origin:
        m = np.sum(x * y) / np.sum(x * x)
        b = 0.0
    else:
        m, b = np.polyfit(x, y, 1)
    pred = m * x + b
    r2 = 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)
    return LinearStandard(float(m), float(b), float(r2), x.min(), x.max(), y.min(), y.max())


def suggest_dilution(signal, std: LinearStandard, target_frac: float = 0.6) -> int:
    """Smallest integer dilution that brings an over-range sample to ~60 % of the top standard."""
    conc = (np.asarray(signal, float) - std.intercept) / std.slope
    d = np.ceil(conc / (target_frac * std.x_max))
    return int(max(1, np.nanmax(d)))


# ----------------------------------------------------------------------------
# Release
# ----------------------------------------------------------------------------
def cumulative_release(times_h, conc_mg_ml, volume_ml: float, loaded_mg: float | None = None,
                       full_replacement: bool = True, sample_volume_ml: float | None = None) -> pd.DataFrame:
    """Cumulative protein release.

    full_replacement=True (thesis protocol): the whole buffer volume is collected and
    replaced, so each sample contains only what was released in that interval:
        M_cum(t_n) = sum_i C_i V
    Otherwise (partial sampling of volume v from V):
        M_cum(t_n) = C_n V + v sum_{i<n} C_i
    """
    t = np.asarray(times_h, float)
    c = np.asarray(conc_mg_ml, float)
    if full_replacement:
        m_int = c * volume_ml
        m_cum = np.cumsum(m_int)
    else:
        v = sample_volume_ml
        m_cum = c * volume_ml + v * np.concatenate([[0], np.cumsum(c)[:-1]])
        m_int = np.diff(np.concatenate([[0], m_cum]))
    out = pd.DataFrame({"time_h": t, "conc_mg_ml": c, "released_interval_mg": m_int,
                        "cumulative_mg": m_cum})
    if loaded_mg:
        out["cumulative_pct"] = 100 * m_cum / loaded_mg
    return out


def korsmeyer_peppas(t, k, n):
    return k * t ** n


def fit_release_kinetics(times_h, frac_released) -> dict:
    """Fit Mt/Minf = k t^n to the first 60 % of release (n ~ 0.5 Fickian for slabs)."""
    t = np.asarray(times_h, float)
    f = np.asarray(frac_released, float)
    m = (f <= 0.6) & (t > 0)
    (k, n), _ = curve_fit(korsmeyer_peppas, t[m], f[m], p0=[0.1, 0.5], maxfev=10000)
    return {"k": k, "n": n}


# ----------------------------------------------------------------------------
# ALP
# ----------------------------------------------------------------------------
def kinetic_slope(time_min, absorbance, min_points: int = 4, r2_min: float = 0.98) -> dict:
    """Slope of the linear region of A405 vs time.

    Starts with all points and drops late points (substrate depletion / plateau) until
    R^2 >= r2_min or only `min_points` remain."""
    t = np.asarray(time_min, float)
    a = np.asarray(absorbance, float)
    for end in range(len(t), min_points - 1, -1):
        res = st.linregress(t[:end], a[:end])
        if res.rvalue ** 2 >= r2_min:
            break
    return {"slope_per_min": res.slope, "r2": res.rvalue ** 2, "n_points": end}


def alp_activity(slope_per_min=None, delta_A=None, t_min=None, blank_slope: float = 0.0,
                 Vt_ml: float = 0.150, Vs_ml: float = 0.050, path_cm: float | None = None,
                 protein_mg_ml=None, eps: float = EPS_PNP):
    """ALP activity in nmol pNP / min / mg protein (Eq. 3.7).

        activity = (dA/dt) Vt 1e6 / (eps d Vs C_protein)

    Give either the kinetic slope (A/min) or an end-point delta_A over t_min.
    Vt/Vs default to 150 uL total / 50 uL lysate in a 48-well plate (Sec. 3.12).
    path_cm is the liquid height in the well; if None it is computed for 150 uL in a
    48-well plate (11.0 mm well diameter -> ~0.16 cm). Use the plate reader's
    pathlength correction if available.
    """
    if path_cm is None:
        path_cm = well_pathlength_cm(Vt_ml * 1000, 11.0)
    if slope_per_min is None:
        slope_per_min = np.asarray(delta_A, float) / t_min
    rate = np.asarray(slope_per_min, float) - blank_slope          # A/min
    # mol/L/min = rate/(eps d); x Vt(L) -> mol/min; x1e9 -> nmol/min; / (Vs(mL) C mg/mL) -> per mg
    nmol_per_min = rate / (eps * path_cm) * (Vt_ml / 1000) * 1e9
    if protein_mg_ml is None:
        return nmol_per_min
    return nmol_per_min / (Vs_ml * np.asarray(protein_mg_ml, float))


def well_pathlength_cm(volume_ul: float, well_diameter_mm: float) -> float:
    """Liquid height in a flat-bottom well: d = V / (pi r^2)."""
    r_cm = well_diameter_mm / 20
    return volume_ul / 1000 / (np.pi * r_cm ** 2)


# ----------------------------------------------------------------------------
# MTT
# ----------------------------------------------------------------------------
def mtt_viability(A570, A_ref=None, blank570=None, blank_ref=None, control=None) -> pd.DataFrame:
    """Reference-wavelength and cell-free-hydrogel blank correction; % of control if given."""
    a = np.asarray(A570, float) - (np.asarray(A_ref, float) if A_ref is not None else 0.0)
    b = 0.0
    if blank570 is not None:
        b = np.mean(np.asarray(blank570, float) - (np.asarray(blank_ref, float) if blank_ref is not None else 0.0))
    corr = a - b
    out = pd.DataFrame({"A_corrected": corr})
    if control is not None:
        out["pct_control"] = 100 * corr / np.mean(control)
    return out


# ----------------------------------------------------------------------------
# ELISA 4PL
# ----------------------------------------------------------------------------
def four_pl(x, a, b, c, d):
    """a = response at zero, d = response at infinity, c = EC50, b = slope."""
    return d + (a - d) / (1 + (x / c) ** b)


def fit_4pl(conc, od):
    x = np.asarray(conc, float)
    y = np.asarray(od, float)
    p0 = [y.min(), 1.0, np.median(x[x > 0]), y.max()]
    popt, _ = curve_fit(four_pl, np.clip(x, 1e-9, None), y, p0=p0, maxfev=20000)
    pred = four_pl(np.clip(x, 1e-9, None), *popt)
    r2 = 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)
    return {"params": popt, "r2": r2}


def inverse_4pl(od, params, dilution=1.0):
    a, b, c, d = params
    y = np.asarray(od, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        x = c * ((a - d) / (y - d) - 1) ** (1 / b)
    return x * dilution



# ============================================================================================
# PLOTTING
# ============================================================================================
# Figure functions matching the layout of the thesis figures.
#
# bar_with_points   Figs 4.1, 4.3, 4.13, A.1, A.2  (mean ± SD bars, individual gels, brackets)
# sweep_mean_sd     Figs 4.2, 4.4-4.12             (mean ± SD curves on log axes)
# nyquist_bode      Fig. 4.14
# standard_curve    Fig. 4.15
CHEM_COLORS = {
    "PEGDA": "#6b6b6b", "neutral": "#6b6b6b",
    "AMPS": "#2a6fb0", "anionic": "#2a6fb0",
    "DEAEAAm": "#c8452f", "cationic": "#c8452f",
    "zwitterionic": "#3c9a5f", "DMAAPS": "#3c9a5f",
}
SERIES_STYLES = ["-", "--", "-.", ":"]
SERIES_COLORS = ["#1b4f8a", "#5b9bd5", "#b8381f", "#f08a5d", "#3c9a5f", "#7f7f7f"]


def _style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
        "axes.spines.right": False, "axes.linewidth": 0.9, "savefig.dpi": 300,
        "legend.frameon": False,
    })


def color_for(name: str, i: int = 0) -> str:
    for k, c in CHEM_COLORS.items():
        if k.lower() in str(name).lower():
            return c
    return SERIES_COLORS[i % len(SERIES_COLORS)]


def bar_with_points(df: pd.DataFrame, response: str, group: str, order=None, ylabel: str = "",
                    title: str = "", pairs: pd.DataFrame | None = None, show_ns: bool = False,
                    scale: float = 1.0, ax=None, rng=0):
    """Bars = mean ± SD, dots = individual gels; `pairs` is a Tukey table with columns
    group1, group2, p-adj (as returned by stats.one_way_anova_tukey)."""
    _style()
    order = order or list(df[group].unique())
    if ax is None:
        _, ax = plt.subplots(figsize=(1.1 * len(order) + 1.5, 3.6))
    r = np.random.default_rng(rng)
    tops = []
    for i, g in enumerate(order):
        v = df.loc[df[group] == g, response].dropna().to_numpy() * scale
        m, s = v.mean(), v.std(ddof=1) if v.size > 1 else 0
        c = color_for(g, i)
        ax.bar(i, m, 0.62, color=c, alpha=0.35, edgecolor=c, linewidth=1.2)
        ax.errorbar(i, m, s, color="k", capsize=4, lw=1)
        ax.scatter(i + r.uniform(-0.12, 0.12, v.size), v, s=22, color=c, edgecolor="k", lw=0.5, zorder=3)
        tops.append(max(m + s, v.max()))
    ax.set_xticks(range(len(order)), order, rotation=0)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, fontsize=10)
    if pairs is not None and len(pairs):
        ymax = max(tops)
        h = 0.10 * ymax
        level = 0
        pairs = pairs.assign(_span=[abs(order.index(a) - order.index(b))
                                    for a, b in zip(pairs["group1"], pairs["group2"])]).sort_values("_span")
        for _, p in pairs.iterrows():
            lab = stars(p["p-adj"])
            if lab == "ns" and not show_ns:
                continue
            a, b = order.index(p["group1"]), order.index(p["group2"])
            y = ymax + h * (0.6 + level)
            ax.plot([a, a, b, b], [y - h * 0.25, y, y, y - h * 0.25], c="k", lw=0.8)
            ax.text((a + b) / 2, y + h * 0.02, lab, ha="center", va="bottom", fontsize=8)
            level += 1
        ax.set_ylim(0, ymax + h * (level + 2))
    ax.figure.tight_layout()
    return ax


def sweep_mean_sd(groups: dict[str, list[pd.DataFrame]], x: str, y: str = "Gp_Pa",
                  xlabel: str = "", ylabel: str = "G′ (kPa)", log_x: bool = True, log_y: bool = True,
                  y_scale: float = 1e-3, also: str | None = None, ax=None, title: str = "",
                  xlim=None, x_grid=None):
    """Mean ± SD of replicate curves for each formulation. `also="Gpp_Pa"` overlays G''
    with dashed lines (Fig. 4.2)."""
    _style()
    if ax is None:
        _, ax = plt.subplots(figsize=(5.2, 3.8))
    for i, (name, curves) in enumerate(groups.items()):
        c = SERIES_COLORS[i % len(SERIES_COLORS)]
        for col, ls, suffix in [(y, "-", "G′"), (also, "--", "G″")]:
            if col is None:
                continue
            avg = average_curves(curves, x, col, log_x=log_x, x_grid=x_grid)
            m, s = avg["mean"] * y_scale, avg["sd"] * y_scale
            lab = f"{name}" + (f" {suffix}" if also else "")
            ax.plot(avg[x], m, ls, color=c, lw=1.6, label=lab)
            ax.fill_between(avg[x], np.clip(m - s, m * 0.05, None), m + s, color=c, alpha=0.18, lw=0)
    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
    if xlim:
        ax.set_xlim(*xlim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    ax.figure.tight_layout()
    return ax


def time_sweeps(groups: dict[str, list[pd.DataFrame]], **kw):
    """Fig. 4.4-4.6: G' vs experiment time; curves interpolated on a common linear time grid."""
    tmax = min(min(c["time_s"].max() for c in v) for v in groups.values())
    grid = np.linspace(0, tmax, 300)
    return sweep_mean_sd(groups, "time_s", xlabel="Time (s)", log_x=False, x_grid=grid, **kw)


def nyquist_bode(spectra: dict[str, pd.DataFrame], title: str = ""):
    _style()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.8))
    for i, (name, d) in enumerate(spectra.items()):
        c = SERIES_COLORS[i % len(SERIES_COLORS)]
        a1.plot(d["Zre_ohm"], -d["Zim_ohm"], "o-", ms=3, lw=1, color=c, label=name)
        a2.loglog(d["freq_Hz"], d["Zmag_ohm"], "o-", ms=3, lw=1, color=c, label=name)
    a1.set_xlabel("Z′ (Ω)")
    a1.set_ylabel("−Z″ (Ω)")
    a1.set_title("(A) Nyquist", fontsize=10, loc="left")
    a2.set_xlabel("Frequency (Hz)")
    a2.set_ylabel("|Z| (Ω)")
    a2.set_title("(B) Bode", fontsize=10, loc="left")
    a1.legend(fontsize=8)
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    return fig


def standard_curve(conc, signal, std, xlabel="BSA (µg/mL)", ylabel="Signal", ax=None):
    _style()
    if ax is None:
        _, ax = plt.subplots(figsize=(4.4, 3.6))
    x = np.asarray(conc, float)
    ax.scatter(x, signal, color="#1b4f8a", edgecolor="k", lw=0.5, zorder=3)
    xx = np.linspace(0, x.max(), 50)
    ax.plot(xx, std.slope * xx + std.intercept, color="#b8381f", lw=1.4, label=std.label())
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8, loc="upper left")
    ax.figure.tight_layout()
    return ax


def design_map(summary: pd.DataFrame, x="Gp_kPa", y="gamma_c_pct", label="formulation", ax=None):
    """Stiffness vs strain-tolerance map (Sec. 4.2.5)."""
    _style()
    if ax is None:
        _, ax = plt.subplots(figsize=(5.2, 4))
    for i, r in summary.iterrows():
        c = color_for(r[label], i)
        ax.scatter(r[x], r[y], s=60, color=c, edgecolor="k", lw=0.6, zorder=3,
                   marker="^" if r.get("gamma_c_censored", False) else "o")
        ax.annotate(r[label], (r[x], r[y]), fontsize=7, xytext=(4, 3), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Plateau modulus G′p (kPa)")
    ax.set_ylabel("Critical strain γc (%)")
    ax.figure.tight_layout()
    return ax



# ============================================================================================
# MAKE_DEMO_DATA
# ============================================================================================
DEMO_MAIN_DOC = """Generate a SYNTHETIC demo dataset with the same structure as the real data.

The curves are simulated from the summary values reported in the thesis (Tables 4.2,
4.3, 4.6, Sec. 4.1, 4.3.1, 4.4) so every analysis script can be run end-to-end and the
outputs compared with the thesis. They are NOT the measured data. Replace the files in
data/ with your TRIOS / CHI exports and balance sheets (same manifest layout) to analyse
the real experiments.

    python scripts/make_demo_data.py --out data"""

#!/usr/bin/env python3


Z4 = np.array([-1.1619, -0.3873, 0.3873, 1.1619])  # mean 0, sample SD 1 for n = 4


def exact(mean, sd, rng):
    """Four values with exactly the given mean and sample SD (random order)."""
    return mean + sd * rng.permutation(Z4)


# ---------------------------------------------------------------------------
# Formulation matrix parameters (Tables 4.2 and 4.3)
# ---------------------------------------------------------------------------
FORMULATIONS = [
    # name, chem, conc, ratio, uv_on (4), FN (4), t90 mean, Gp mean/sd (kPa), GLVR mean/sd, Gw mean/sd,
    # slope n, gamma_c replicate values (%), max strain, test length (s), peak (kPa) or None
    ("PEGDA 10%", "PEGDA", 10, None, [48.46] * 4, [73, 55, 73, 49], 12.2, (50.1, 7.4), (48.7, 5.9),
     (10.4, 2.9), 0.023, [4.1, 10.5, 11.5, 12], 100, 320, None),
    ("PEGDA 20%", "PEGDA", 20, None, [66.7, 48.46, 48.46, 48.46], [134, 85, 73, 61], 6.0, (191, 61), (163, 44),
     (50.5, 32.2), 0.106, [0.3, 0.38, 0.42, 0.5], 100, 320, (404, 74)),
    ("DEAEAAm 10% 10:1", "DEAEAAm", 10, 10, [84.82] * 4, [79, 91, 79, 79], 42.5, (1.67, 0.63), (1.72, 0.60),
     (1.70, 0.78), -0.052, [131, 140, 146, 242], 1000, 500, None),
    ("DEAEAAm 10% 25:1", "DEAEAAm", 10, 25, [84.82, 84.82, 84.82, 6.0], [91, 85, 85, 73], 51.5, (1.08, 0.18),
     (1.08, 0.12), (0.86, 0.18), 0.006, [1.0, 190, 204, 309], 1000, 500, None),
    ("DEAEAAm 20% 10:1", "DEAEAAm", 20, 10, [84.82, 84.82, 84.82, 5.95], [127, 133, 134, 109], 113.7, (30.6, 5.3),
     (30.6, 5.3), (23.5, 6.6), 0.016, [17, 38, 42, 47], 100, 500, None),
    ("DEAEAAm 20% 25:1", "DEAEAAm", 20, 25, [84.82, 90.92, 109, 133], [103, 97, 99, 97], 67.0, (9.60, 1.17),
     (9.63, 1.26), (9.15, 0.50), 0.009, [115, 116, 120, 218], 1000, 500, None),
    ("AMPS 10% 10:1", "AMPS", 10, 10, [85, 85, 79, 85], [170, 127, 127, 152], 98.5, (2.51, 0.37), None,
     (1.51, 0.81), 0.019, [0.6, 1.5, 3.0, 279], 1000, 320, None),
    ("AMPS 10% 25:1", "AMPS", 10, 25, [61, 36, 12, 24], [61, 61, 61, 62], 21.3, (0.53, 0.07), (0.50, 0.10),
     (0.52, 0.06), 0.009, [1e6] * 4, 280, 320, None),
    ("AMPS 20% 10:1", "AMPS", 20, 10, [85, 79, 18, 12], [158, 164, 103, 115], 106.2, (24.5, 7.4), (29.1, 6.8),
     (26.4, 5.9), -0.001, [0.4, 30, 52, 72], 1000, 320, None),
    ("AMPS 20% 25:1", "AMPS", 20, 25, [36.4, 36.4, 18.2, 18.2], [91, 91, 85, 91], 42.5, (24.2, 1.4), (24.0, 1.4),
     (13.5, 5.6), 0.009, [84, 88, 92, 119], 1000, 320, None),
]


def time_sweep(uv_on, FN_target, t90, Gp_Pa, length, rng, peak_Pa=None, dt=6.06):
    """Two-component rise (fast + slow creep) tuned so that FN ~ FN_target."""
    phase = rng.uniform(0.0, 0.25)
    t = np.arange(0, length + 1e-9, dt)
    t = t - (t[np.searchsorted(t, uv_on)] - uv_on) + phase   # put a sample ~6 s after UV
    t = t[t >= 0]
    tr = t - uv_on
    t_ind = 1.0                                   # short O2-inhibition period after shutter opens
    tau1 = max(t90 * rng.uniform(0.9, 1.1) - t_ind, 2.0) / np.log(10)

    def model(w, tau2):
        x = np.clip(tr - t_ind, 0, None)
        g = Gp_Pa * ((1 - w) * (1 - np.exp(-x / tau1)) + w * (1 - np.exp(-x / tau2)))
        if peak_Pa:
            g = g + (peak_Pa - Gp_Pa) * np.exp(-x / 70.0) * (1 - np.exp(-x / 3.0)) * 1.25
        return g

    base = min(20.0, Gp_Pa / 300)
    best, best_err = (0.05, 10.0), np.inf
    for w in np.linspace(0.0, 0.45, 19):
        for tau2 in np.linspace(3, 150, 50):
            g = np.where(tr > t_ind, model(w, tau2), base)
            fn = full_network_time(tr, g, start_after=1.0)
            reach = tr[(tr > 0) & (g >= 0.9 * g[-5:].mean())]
            t90_sim = reach[0] if reach.size else np.inf
            err = (abs(fn - FN_target) if np.isfinite(fn) else 1e3) + 0.5 * abs(t90_sim - t90)
            if err < best_err:
                best, best_err = (w, tau2), err
    Gp = model(*best)
    pre = tr <= t_ind
    Gp[pre] = 10 ** rng.normal(np.log10(base), 0.4, pre.sum())    # liquid, near torque limit
    Gp[~pre] *= 1 + 0.0007 * rng.standard_normal((~pre).sum())
    Gpp = np.where(pre, Gp * 10 ** rng.normal(0.6, 0.2, t.size), 0.02 * Gp + 5)
    Gpp[~pre] = np.minimum(Gpp[~pre], 0.5 * Gp[~pre])
    return pd.DataFrame({"time_s": np.round(t, 2), "Gp_Pa": Gp, "Gpp_Pa": Gpp})


def amp_sweep(G_lvr, gamma_c, max_strain, rng, m=2.0, ppd=10):
    n = int(np.log10(max_strain / 0.01) * ppd) + 1
    g = np.logspace(-2, np.log10(max_strain), n)
    g0 = gamma_c * 9 ** (1 / m)
    Gp = G_lvr / (1 + (g / g0) ** m)
    Gp *= 1 + 0.01 * rng.standard_normal(n)
    Gp[:5] = G_lvr * (1 + 0.004 * rng.standard_normal(5))
    Gp[:5] *= G_lvr / Gp[:5].mean()
    bump = np.exp(-(np.log10(g / max(gamma_c, 1e-3))) ** 2 / 0.4)
    Gpp = 0.03 * G_lvr * (1 + 2.5 * bump) * np.minimum(1, (Gp / G_lvr) ** 0.3)
    stress = np.hypot(Gp, Gpp) * g / 100
    return pd.DataFrame({"strain_pct": g, "Gp_Pa": Gp, "Gpp_Pa": Gpp, "stress_Pa": stress})


def freq_sweep(G_omega, n_slope, rng, soft=False):
    w = np.logspace(np.log10(0.0628), np.log10(628), 41)
    Gp = G_omega * w ** n_slope
    Gp /= Gp[(w >= 0.1) & (w <= 10)].mean() / G_omega
    Gp *= 1 + 0.01 * rng.standard_normal(w.size)
    Gp *= 1 + 8 * (w / 300) ** 2.5                                  # inertia artefact
    if soft:
        lo = w < 0.1
        Gp[lo] *= 10 ** rng.normal(0, 0.25, lo.sum())
    Gpp = 0.03 * Gp
    return pd.DataFrame({"omega_rad_s": w, "freq_Hz": w / (2 * np.pi), "Gp_Pa": Gp, "Gpp_Pa": Gpp})


def make_formulation_matrix(root: Path, rng):
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for (name, chem, conc, ratio, uv, FN, t90, Gp, Glvr, Gw, nslope, gcs, gmax, length, peak) in FORMULATIONS:
        slug = name.replace(" ", "_").replace("%", "pct").replace(":", "-")
        (root / slug).mkdir(exist_ok=True)
        Gp_rep = exact(*Gp, rng) * 1e3
        if Glvr is None:   # AMPS 10% 10:1: three normal + one 22.5 kPa outlier (Sec. 4.2.3)
            Glvr_rep = np.array([1.9, 2.15, 2.4, 22.5]) * 1e3
        else:
            Glvr_rep = exact(*Glvr, rng) * 1e3
        Gw_rep = np.abs(exact(*Gw, rng)) * 1e3
        pk_rep = exact(*peak, rng) * 1e3 if peak else [None] * 4
        gmax_rep = [gmax * rng.uniform(0.96, 1.07)] * 4 if gmax < 1000 else [gmax] * 4
        for r in range(4):
            ts = time_sweep(uv[r], FN[r], t90, Gp_rep[r], length, rng, pk_rep[r])
            if name == "PEGDA 10%" and r == 2:
                ts = pd.read_csv(root / slug / "rep1_time.csv")       # exported duplicate (Sec. 4.11)
            ts.to_csv(root / slug / f"rep{r + 1}_time.csv", index=False)
            amp_sweep(Glvr_rep[r], gcs[r], gmax_rep[r], rng).to_csv(root / slug / f"rep{r + 1}_amplitude.csv", index=False)
            freq_sweep(Gw_rep[r], nslope, rng, soft=Gw[0] < 3).to_csv(root / slug / f"rep{r + 1}_frequency.csv", index=False)
            rows.append({"formulation": name, "chemistry": chem, "conc": conc, "ratio": ratio if ratio else "",
                         "replicate": r + 1, "uv_on_s": uv[r],
                         "time_file": f"{slug}/rep{r + 1}_time.csv",
                         "amplitude_file": f"{slug}/rep{r + 1}_amplitude.csv",
                         "frequency_file": f"{slug}/rep{r + 1}_frequency.csv"})
    pd.DataFrame(rows).to_csv(root / "manifest.csv", index=False)


# ---------------------------------------------------------------------------
# IPNs (Sec. 4.1, 4.3.1, 4.4)
# ---------------------------------------------------------------------------
IPN = {  # G'LVR kPa (mean, sd), yield %, SR (mean, sd), R_bulk ohm (mean, sd)
    "AMPS": ((40.5, 3.0), (0.35, 0.08), (5.5, 0.2), (17.9, 0.6)),
    "Cationic": ((43.0, 3.5), (0.15, 0.04), (3.5, 0.15), (34.8, 1.1)),
    "Zwitterionic": ((37.2, 2.8), (0.30, 0.08), (8.3, 0.2), (196.0, 6.1)),
    "PEGDA": ((41.0, 3.0), (0.60, 0.15), (7.2, 0.2), (52.6, 1.7)),
}
CELL_CONSTANT = 0.2499  # L/A in cm^-1 implied by R_bulk = 196 ohm -> 1.275 mS/cm


def make_ipn(root: Path, rng):
    for sub in ("rheology", "eis"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    rows_r, rows_s, rows_e = [], [], []
    for name, (G, y, SR, Rb) in IPN.items():
        Gs = exact(*G, rng) * 1e3
        ys = np.abs(exact(*y, rng))
        SRs = exact(*SR, rng)
        Rs = exact(*Rb, rng)
        for r in range(4):
            # yield (5 % deviation) ~ gamma_c (10 % drop) * (0.05/0.10)^(1/m)*... calibrate by m = 1.2
            m = 1.2
            gamma_c = ys[r] * (np.log(1 / 0.95) / np.log(1 / 0.9)) ** (-1 / m)
            df = amp_sweep(Gs[r], gamma_c, 100, rng, m=m)
            # zwitterionic loses modulus most steeply (Fig. 4.2)
            if name == "Zwitterionic":
                df["Gp_Pa"] = df["Gp_Pa"] / (1 + (df["strain_pct"] / 8) ** 3)
            f = f"rheology/{name}_rep{r + 1}_amplitude.csv"
            df.to_csv(root / f, index=False)
            rows_r.append({"group": name, "replicate": r + 1, "amplitude_file": f})
            Wd_polymer = rng.uniform(4.0, 5.0)                       # mg
            water = Wd_polymer * (SRs[r] - 1) / (1 + PBS_SOLIDS_G_PER_L / (1000 * WATER_DENSITY))
            salt = water * PBS_SOLIDS_G_PER_L / (1000 * WATER_DENSITY)
            Ws = Wd_polymer * SRs[r]
            rows_s.append({"group": name, "medium": "PBS", "replicate": r + 1,
                           "Ws": round(Ws, 4), "Wd": round(Wd_polymer + salt, 4)})
            spec = simulate_spectrum(Rs[r], R_hf=0.1 * Rs[r], tau_s=10 ** rng.uniform(-5.2, -4.8), noise=0.003, rng=rng)
            f = f"eis/{name}_rep{r + 1}.txt"
            with open(root / f, "w") as fh:
                fh.write("A.C. Impedance\nInstrument Model:  CHI700E\nInit E (V) = 0\n"
                         "Amplitude (V) = 0.02\nHigh Frequency (Hz) = 1e+6\nLow Frequency (Hz) = 1\n\n")
                fh.write('Freq/Hz, Z\'/ohm, Z"/ohm, Z/ohm, Phase/deg\n')
                for _, p in spec.iterrows():
                    fh.write(f"{p.freq_Hz:.3e}, {p.Zre_ohm:.3e}, {p.Zim_ohm:.3e}, {p.Zmag_ohm:.3e}, {p.phase_deg:.2f}\n")
            rows_e.append({"group": name, "replicate": r + 1, "file": f})
    pd.DataFrame(rows_r).to_csv(root / "rheology_manifest.csv", index=False)
    pd.DataFrame(rows_s).to_csv(root / "swelling.csv", index=False)
    pd.DataFrame(rows_e).to_csv(root / "eis_manifest.csv", index=False)
    (root / "eis_geometry.csv").write_text(f"thickness_cm,area_cm2\n0.2,{0.2 / CELL_CONSTANT:.4f}\n")


# ---------------------------------------------------------------------------
# Printer comparison (Table 4.6)
# ---------------------------------------------------------------------------
def make_printer(root: Path, rng):
    root.mkdir(parents=True, exist_ok=True)
    sw, gf = [], []
    targets = {("Formlabs", "PBS"): (6.18, 0.30), ("Formlabs", "DIW"): (6.57, 0.69),
               ("Phrozen", "PBS"): (5.60, 0.47), ("Phrozen", "DIW"): (6.63, 0.57)}
    for (pr, med), (m, s) in targets.items():
        for r, SR in enumerate(exact(m, s, rng)):
            Wp = rng.uniform(5.5, 6.5)
            Ws = Wp * SR                                   # swollen mass incl. dissolved salt
            k = PBS_SOLIDS_G_PER_L / (1000 * WATER_DENSITY) if med == "PBS" else 0.0
            salt = (Ws - Wp) / (1 + k) * k
            sw.append({"printer": pr, "medium": med, "replicate": r + 1,
                       "Ws": round(Ws, 4), "Wd": round(Wp + salt, 4)})
    for pr, (m, s) in {"Formlabs": (94.2, 2.2), "Phrozen": (93.2, 3.2)}.items():
        for r, g in enumerate(exact(m, s, rng)):
            Wi = rng.uniform(5.5, 6.5)
            gf.append({"printer": pr, "replicate": r + 1, "Wi": round(Wi, 4), "Wf": round(Wi * g / 100, 4)})
    pd.DataFrame(sw).to_csv(root / "swelling.csv", index=False)
    pd.DataFrame(gf).to_csv(root / "gel_fraction.csv", index=False)


# ---------------------------------------------------------------------------
# Cell assays (Sec. 3.8, 3.11, 3.12, 3.13)
# ---------------------------------------------------------------------------
def make_assays(root: Path, rng):
    root.mkdir(parents=True, exist_ok=True)
    # BSA standard (Fig. 4.15) with low-end scatter
    x = np.repeat([0, 12.5, 25, 50, 100, 150, 200], 3)
    y = 0.5428 * x + 7.0489 + rng.normal(0, 1, x.size) * (6 + 0.02 * x)
    pd.DataFrame({"conc_ug_ml": x, "signal": y}).to_csv(root / "bsa_standard.csv", index=False)
    # ALP kinetic reads (A405 every 5 min, 7 reads) and lysate protein signal
    rows = []
    for day, rate, prot in [(0, 0.004, 40), (7, 0.010, 70), (14, 0.018, 110), (21, 0.012, 160)]:
        for rep in range(1, 4):
            r_ = rate * rng.uniform(0.85, 1.15)
            p_sig = 0.5428 * prot * rng.uniform(0.9, 1.1) + 7.0489
            dil = 1
            if p_sig > 0.5428 * 200 + 7:          # over range -> diluted and re-read
                dil = 2
                p_sig = 0.5428 * prot / 2 * rng.uniform(0.9, 1.1) + 7.0489
            for k in range(7):
                t = 5 * k
                A = 0.05 + r_ * t * (1 - 0.004 * t) + rng.normal(0, 0.002)
                rows.append({"day": day, "replicate": rep, "time_min": t, "A405": A,
                             "protein_signal": p_sig, "protein_dilution": dil})
    pd.DataFrame(rows).to_csv(root / "alp_kinetics.csv", index=False)
    # BCA release (full replacement)
    t = [1, 6, 12, 24, 72]
    rel = []
    for g, k in [("AMPS", 0.08), ("Cationic", 0.05), ("Zwitterionic", 0.12), ("PEGDA", 0.10)]:
        for rep in range(1, 4):
            frac = 1 - np.exp(-k * np.array(t) ** 0.6)
            inc = np.diff(np.concatenate([[0], frac])) * 0.5  # 0.5 mg loaded
            A = 0.1 + 1.6 * inc / 1.0 * rng.uniform(0.95, 1.05, len(t))
            for ti, a in zip(t, A):
                rel.append({"group": g, "replicate": rep, "time_h": ti, "A562": a, "dilution": 1})
    pd.DataFrame(rel).to_csv(root / "bca_release.csv", index=False)
    bstd = pd.DataFrame({"conc_mg_ml": np.repeat([0, 0.0625, 0.125, 0.25, 0.5], 3)})
    bstd["A562"] = 0.1 + 1.6 * bstd.conc_mg_ml + rng.normal(0, 0.01, len(bstd))
    bstd.to_csv(root / "bca_standard.csv", index=False)
    # PicoGreen DNA on scaffold
    dstd = pd.DataFrame({"conc_ug_ml": np.repeat([0, 2, 5, 10, 20, 40], 3)})
    dstd["RFU"] = 150 + 980 * dstd.conc_ug_ml + rng.normal(0, 200, len(dstd))
    dstd.to_csv(root / "picogreen_standard.csv", index=False)
    dna = []
    for day, mu in [(1, 3.0), (21, 6.0), (28, 14.0), (35, 14.5)]:
        for med in ("expansion", "osteogenic"):
            for rep in range(1, 4):
                dna.append({"day": day, "medium": med, "replicate": rep,
                            "RFU": 150 + 980 * mu * rng.uniform(0.85, 1.15)})
    pd.DataFrame(dna).to_csv(root / "picogreen_samples.csv", index=False)
    # BMP-2 ELISA
    std = np.array([0, 31.3, 62.5, 125, 250, 500, 1000, 2000])
    od = 0.05 + (2.6 - 0.05) / (1 + (np.clip(std, 1e-3, None) / 600) ** -1.2)
    pd.DataFrame({"conc_pg_ml": np.repeat(std, 2), "OD450": np.repeat(od, 2) * rng.uniform(0.97, 1.03, 16)}
                 ).to_csv(root / "elisa_standard.csv", index=False)
    el = []
    for day in (1, 3, 7, 14, 21):
        for rep in range(1, 4):
            c = rng.uniform(30, 90)
            o = 0.05 + (2.6 - 0.05) / (1 + (c / 600) ** -1.2)
            el.append({"day": day, "replicate": rep, "OD450": o})
    pd.DataFrame(el).to_csv(root / "elisa_samples.csv", index=False)


def demo_main(argv=None):
    ap = argparse.ArgumentParser(description=DEMO_MAIN_DOC, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data")
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args(argv)
    rng = np.random.default_rng(a.seed)
    out = Path(a.out)
    make_formulation_matrix(out / "formulation_matrix", rng)
    make_ipn(out / "ipn", rng)
    make_printer(out / "printer_comparison", rng)
    make_assays(out / "assays", rng)
    (out / "README_SYNTHETIC.txt").write_text(
        "These files are SYNTHETIC, simulated from the summary values reported in the thesis.\n"
        "They exist only to exercise the analysis pipeline. Replace them with real exports.\n")
    print(f"Synthetic demo data written to {out.resolve()}")



# ============================================================================================
# RUN_CALCULATORS
# ============================================================================================
CALCULATORS_MAIN_DOC = """Bench calculators and theory checks (Sec. 2.5.1, 3.2, 3.7, 4.2.5, Table 3.1, 3.2, 4.5).

    python scripts/run_calculators.py --out results/calculators
    python scripts/run_calculators.py --recipe AMPS 15 20      # one custom formulation"""

#!/usr/bin/env python3


def calculators_main(argv=None):
    ap = argparse.ArgumentParser(description=CALCULATORS_MAIN_DOC, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="results/calculators")
    ap.add_argument("--recipe", nargs="+", metavar=("MONOMER", "CONC"),
                    help="monomer conc [ratio], e.g. --recipe DEAEAAm 15 25")
    ap.add_argument("--mbaa-stock", type=float, default=MBAA_STOCK_MG_PER_ML,
                    help="MBAA stock, mg/mL (40 reproduces Table 3.1)")
    a = ap.parse_args(argv)
    pd.set_option("display.width", 200, "display.max_columns", 20)

    if a.recipe:
        mon, conc = a.recipe[0], float(a.recipe[1])
        ratio = float(a.recipe[2]) if len(a.recipe) > 2 else None
        r = formulation_recipe(mon, conc, ratio, mbaa_stock_mg_per_ml=a.mbaa_stock)
        print(pd.Series(r).to_string())
        return

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t31 = table_3_1(mbaa_stock_mg_per_ml=a.mbaa_stock)
    t32 = table_3_2()
    t45 = table_4_5()
    well = MicropillarWell()
    mold = pd.Series(well.summary())
    ads = pd.Series(adsorption_sensitivity(area_cm2=well.surface_area_mm2 / 100))

    donnan = []
    for cs in (0.0001, 15, 150):
        for cf in (50, fixed_charge_from_formulation(10, 229.23)):
            cp, cm = donnan_concentrations(cf, cs)
            donnan.append({"c_salt_mM": cs, "c_fixed_mM": round(cf, 1), "c+_mM": cp, "c-_mM": cm,
                           "enrichment_c+/cs": cp / cs, "psi_D_mV": donnan_potential_mV(cf, cs),
                           "Pi_D_kPa": donnan_osmotic_pressure_kPa(cf, cs)})
    donnan = pd.DataFrame(donnan)

    # Illustrative Vmem shift for a cell inside the Donnan phase (hypothesis, Sec. 1.2 / 2.5.1)
    P = {"K": 1.0, "Na": 0.05, "Cl": 0.45}
    bath = {"K": 5.4, "Na": 144.0, "Cl": 125.0}
    cyto = {"K": 140.0, "Na": 12.0, "Cl": 30.0}
    vm = [{"gel": "no gel (medium)", "Vmem_mV": ghk_vmem_mV(P, bath, cyto)}]
    for cf in (10, 50, 150):
        vm.append({"gel": f"anionic, cF = {cf} mM", "Vmem_mV": vmem_in_donnan_phase(cf, -1, P, bath, cyto)})
        vm.append({"gel": f"cationic, cF = {cf} mM", "Vmem_mV": vmem_in_donnan_phase(cf, +1, P, bath, cyto)})
    vm = pd.DataFrame(vm)

    misc = pd.Series({
        "disc volume 8 mm x 1 mm (uL)": disc_volume_ul(),
        "rheometer gap for 50 uL on 8 mm plates (um)": rheometer_gap_um(),
        "precursor for 240 gels (mL)": precursor_needed_ul(240) / 1000,
        "APS for 2 mL of 100 mM (g)": stock_mass_g(100, 228.2, 2),
        "iron gluconate for 2 mL of 100 mM (g)": stock_mass_g(100, 446.1, 2),
        "Debye length, I = 0.16 M (nm)": debye_length_nm(0.16),
        "nominal sulfonate, 10 % AMPS-Na (mM)": fixed_charge_from_formulation(10, 229.23),
    })

    for name, t in [("table_3_1", t31), ("table_3_2", t32), ("table_4_5", t45), ("donnan", donnan), ("vmem_ghk", vm)]:
        t.to_csv(out / f"{name}.csv", index=False)
    mold.to_csv(out / "microtissue_mold.csv", header=["value"])
    ads.to_csv(out / "adsorption_sensitivity.csv", header=["value"])
    misc.to_csv(out / "misc.csv", header=["value"])

    for title, t in [("Table 3.1", t31), ("Table 3.2", t32), ("Table 4.5", t45)]:
        print(f"\n=== {title} ===\n{t.to_string(index=False)}")
    print(f"\n=== Microtissue mold (Eq. 3.6) ===\n{mold.round(3).to_string()}")
    print(f"\n=== Adsorption sensitivity (Sec. 3.7) ===\n{ads.to_string()}")
    print(f"\n=== Donnan equilibrium (Eq. 2.2-2.3) ===\n{donnan.round(3).to_string(index=False)}")
    print(f"\n=== Illustrative GHK Vmem in the Donnan phase ===\n{vm.round(2).to_string(index=False)}")
    print(f"\n=== Misc ===\n{misc.round(4).to_string()}")
    print(f"\nResults written to {out.resolve()}")



# ============================================================================================
# RUN_FORMULATION_MATRIX
# ============================================================================================
MATRIX_MAIN_DOC = """Formulation-matrix photorheology analysis (Sec. 3.3.2-3.3.3, 4.2).

Input: a manifest CSV with one row per replicate:

    formulation, chemistry, conc, ratio, replicate, uv_on_s, time_file, amplitude_file, frequency_file

  * chemistry : PEGDA / AMPS / DEAEAAm (any label; PEGDA rows have an empty ratio)
  * uv_on_s   : logged UV shutter-opening time (experiment time, s)
  * *_file    : paths (relative to the manifest) of the TRIOS exports (.txt/.csv/.xlsx)
                OR a single `file` column pointing to one multi-step TRIOS export; the
                time/amplitude/frequency steps are then detected automatically.

Outputs (in --out):
  replicate_parameters.csv       every extracted parameter, every gel
  table_4_1_gelation.csv         GP, FN, t90
  table_4_2_replicates.csv       replicate GP / FN / shutter times
  table_4_3_viscoelastic.csv     G'p, G'LVR, G'w, n, gc, nu, xi
  table_4_4_anova.csv            two-way ANOVA per chemistry
  planned_comparisons.csv        Welch + Holm
  matched_pairs.csv, scaling_exponents.csv, qc_log.txt
  figures/fig_4_04 ... fig_4_12, design_map
  report.md

    python scripts/run_formulation_matrix.py --manifest data/formulation_matrix/manifest.csv --out results/formulation_matrix"""

#!/usr/bin/env python3


RESPONSES = {  # column -> (label, log-transform)
    "Gp_plateau_Pa": ("G′p (time sweep)", True),
    "G_LVR_Pa": ("G′LVR (amplitude sweep)", True),
    "G_omega_Pa": ("G′ω (frequency sweep)", True),
    "FN_s": ("FN", False),
    "gamma_c_pct": ("γc", True),
}


def load_replicate(row, base: Path):
    curves = {}
    if isinstance(row.get("file"), str) and row["file"]:
        df = load_trios(base / row["file"])
        for _, step in split_steps(df).items():
            curves.setdefault(classify_step(step), step)
    for kind in ("time", "amplitude", "frequency"):
        f = row.get(f"{kind}_file")
        if isinstance(f, str) and f:
            curves[kind] = load_trios(base / f)
    return curves


def analyse(manifest: Path, n_plateau=5, fn_tol=0.01, gc_drop=0.10, grubbs_alpha=0.05):
    base = manifest.parent
    man = pd.read_csv(manifest)
    man["ratio"] = pd.to_numeric(man["ratio"], errors="coerce")
    qc, rows, curves = [], [], {}
    for form, grp in man.groupby("formulation", sort=False):
        reps = {}
        for _, r in grp.iterrows():
            reps[int(r.replicate)] = (r, load_replicate(r, base))
        # duplicate exports (Sec. 4.11)
        ts = {k: v[1]["time"] for k, v in reps.items() if "time" in v[1]}
        dup_drop = set()
        for a, b in find_duplicate_replicates(ts):
            qc.append(f"{form}: time-sweep replicate {b} is an exact duplicate of replicate {a}; "
                      f"replicate {b} excluded from time-sweep metrics.")
            dup_drop.add(b)
        curves[form] = {"time": [], "amplitude": [], "frequency": []}
        for k, (r, c) in reps.items():
            rec = {"formulation": form, "chemistry": r.chemistry, "conc": r.conc, "ratio": r.ratio, "replicate": k}
            if "time" in c and k not in dup_drop:
                res = analyze_time_sweep(c["time"], r.get("uv_on_s"), n_plateau=n_plateau, fn_tol=fn_tol)
                rec.update(res.as_dict())
                curves[form]["time"].append(c["time"])
            if "amplitude" in c:
                rec.update(analyze_amplitude_sweep(c["amplitude"], drop=gc_drop).as_dict())
                curves[form]["amplitude"].append(c["amplitude"])
            if "frequency" in c:
                fr = analyze_frequency_sweep(c["frequency"])
                rec.update({"G_omega_Pa": fr.G_omega_Pa, "slope_n": fr.slope_n, "slope_r2": fr.slope_r2})
                curves[form]["frequency"].append(c["frequency"])
            rows.append(rec)
    df = pd.DataFrame(rows)

    # Grubbs on G'LVR within each formulation -> flag (excluded from the LVR mean only)
    df["G_LVR_outlier"] = False
    for form, g in df.groupby("formulation", sort=False):
        t = grubbs_test(g["G_LVR_Pa"].dropna(), grubbs_alpha)
        if t["outlier"]:
            idx = g["G_LVR_Pa"].dropna().index[t["index"]]
            df.loc[idx, "G_LVR_outlier"] = True
            qc.append(f"{form}: G'LVR = {t['value'] / 1e3:.3g} kPa is a Grubbs outlier "
                      f"(G = {t['G']:.3f} > Gcrit = {t['G_crit']:.3f}, alpha = {grubbs_alpha}); excluded from LVR mean.")
    return df, curves, qc


# ---------------------------------------------------------------------------
def table_4_1(df):
    out = []
    for form, g in df.groupby("formulation", sort=False):
        gp = g["GP_s"].dropna()
        out.append({"Formulation": form,
                    "Gel point after UV (s)": f"≤ {g['sample_dt_s'].median():.0f} ({gp.mean():.1f})" if len(gp) else "—",
                    "Full network, FN (s)": mean_sd(g["FN_s"], 1),
                    "Rise time, t90 (s)": mean_sd(g["t90_s"], 1),
                    "n": int(g["FN_s"].notna().sum())})
    return pd.DataFrame(out)


def table_4_2(df):
    out = []
    for form, g in df.groupby("formulation", sort=False):
        g = g.sort_values("replicate")
        out.append({"Formulation": form,
                    "GP replicates (s)": ", ".join(f"{v:.1f}" for v in g["GP_s"].dropna()),
                    "GP mean ± SD (s)": mean_sd(g["GP_s"], 2),
                    "FN replicates (s)": ", ".join(f"{v:.0f}" for v in g["FN_s"].dropna()),
                    "FN mean ± SD (s)": mean_sd(g["FN_s"], 1),
                    "UV shutter opened at (s)": ", ".join(f"{v:g}" for v in g["uv_on_s"].dropna())})
    return pd.DataFrame(out)


def table_4_3(df):
    out = []
    for form, g in df.groupby("formulation", sort=False):
        gp = g["Gp_plateau_Pa"].dropna() / 1e3
        lvr_ok = g.loc[~g["G_LVR_outlier"], "G_LVR_Pa"] / 1e3
        gpcell = mean_sd(gp)
        if g["overshoot"].fillna(False).astype(bool).any():
            gpcell += f" (peak {mean_sd(g['Gp_peak_Pa'].dropna() / 1e3, 0)})"
        lvr = mean_sd(lvr_ok) + ("*" if g["G_LVR_outlier"].any() else "")
        net = network_parameters(gp.mean() * 1e3)
        out.append({"Formulation": form, "G′p, time sweep (kPa)": gpcell,
                    "G′LVR, amplitude sweep (kPa)": lvr,
                    "G′ω, 0.1–10 rad/s (kPa)": mean_sd(g["G_omega_Pa"] / 1e3),
                    "Slope n": f"{g['slope_n'].mean():.3f}",
                    "γc, median [range] (%)": median_range(g["gamma_c_pct"], g["gamma_c_censored"]),
                    "ν (mol/m³)": f"{net['nu_mol_m3']:.3g}", "ξ (nm)": f"{net['xi_nm']:.2g}",
                    "E ≈ 3G′p (kPa)": f"{net['E_kPa_approx']:.3g}"})
    return pd.DataFrame(out)


def table_4_4(df):
    rows = []
    charged = df[df["ratio"].notna()]
    for chem, g in charged.groupby("chemistry", sort=False):
        for col, (label, log) in RESPONSES.items():
            if g[col].notna().sum() < 6:
                continue
            a = two_way_anova(g, col, "conc", "ratio", log=log)
            rows.append({"Response": label, "Chemistry": chem,
                         "Concentration": anova_cell(a.loc["conc"]),
                         "Ratio": anova_cell(a.loc["ratio"]),
                         "Interaction": anova_cell(a.loc["conc:ratio"]),
                         "df": f"({a.loc['conc', 'df']:.0f}, {a.loc['conc', 'df_resid']:.0f})"})
    return pd.DataFrame(rows).sort_values(["Response", "Chemistry"], kind="stable")


def planned(df):
    out = []
    charged = df[df["ratio"].notna()]
    for chem, g in charged.groupby("chemistry", sort=False):
        for col, (label, log) in RESPONSES.items():
            pc = planned_comparisons(g, col, "conc", "ratio", log=log)
            pc.insert(0, "response", label)
            pc.insert(0, "chemistry", chem)
            out.append(pc)
    peg = df[df["ratio"].isna()]
    for col, (label, log) in RESPONSES.items():
        concs = sorted(peg["conc"].unique())
        if len(concs) == 2:
            a = peg[peg.conc == concs[0]][col].dropna()
            b = peg[peg.conc == concs[1]][col].dropna()
            w = welch(a, b, log)
            out.append(pd.DataFrame([{"chemistry": "PEGDA", "response": label,
                                      "comparison": f"{concs[0]}% vs {concs[1]}%", **w,
                                      "mean_a": a.mean(), "mean_b": b.mean(), "p_adj": w["p"],
                                      "sig": stars(w["p"])}]))
    return pd.concat(out, ignore_index=True)


def matched_pairs(df, col="Gp_plateau_Pa"):
    """Oppositely charged formulations at equal conc and ratio (Sec. 4.2.5)."""
    out = []
    ch = df[df["ratio"].notna()]
    chems = list(ch["chemistry"].unique())
    if len(chems) < 2:
        return pd.DataFrame()
    for (c, r), g in ch.groupby(["conc", "ratio"]):
        a = g[g.chemistry == chems[0]][col].dropna()
        b = g[g.chemistry == chems[1]][col].dropna()
        if len(a) < 2 or len(b) < 2:
            continue
        w = welch(a, b, log=True)          # moduli are log-transformed (Sec. 3.14)
        tost = tost_welch(a, b, margin=np.log10(1.5))
        out.append({"conc": c, "ratio": f"{r:g}:1",
                    chems[0] + " (kPa)": mean_sd(a / 1e3), chems[1] + " (kPa)": mean_sd(b / 1e3),
                    "Welch p": w["p"], "not different (p>0.05)": w["p"] > 0.05,
                    "TOST p (±1.5-fold)": tost["p_tost"], "equivalent": tost["equivalent"],
                    "E range (kPa)": f"{3 * min(a.mean(), b.mean()) / 1e3:.0f}–{3 * max(a.mean(), b.mean()) / 1e3:.0f}"})
    return pd.DataFrame(out)


def scaling(df):
    out = []
    for (chem, ratio), g in df.groupby(["chemistry", df["ratio"].fillna(0)]):
        m = g.groupby("conc")["Gp_plateau_Pa"].mean()
        if len(m) == 2:
            lo, hi = m.index.min(), m.index.max()
            out.append({"chemistry": chem, "ratio": f"{ratio:g}:1" if ratio else "—",
                        "fold_change": m[hi] / m[lo],
                        "exponent_a": concentration_scaling_exponent(m[lo], m[hi], lo, hi)})
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
def figures(df, curves, figdir: Path):
    figdir.mkdir(parents=True, exist_ok=True)
    chems = list(dict.fromkeys(df["chemistry"]))
    numbers = {"DEAEAAm": (4, 7, 10), "AMPS": (5, 8, 11), "PEGDA": (6, 9, 12)}
    for chem in chems:
        forms = list(dict.fromkeys(df.loc[df.chemistry == chem, "formulation"]))
        nt, na, nf = numbers.get(chem, (0, 0, 0))
        tg = {f: curves[f]["time"] for f in forms if curves[f]["time"]}
        if tg:
            ax = time_sweeps(tg, title=f"{chem}: photopolymerization time sweeps")
            ax.figure.savefig(figdir / f"fig_4_{nt:02d}_time_{chem}.png")
            plt.close(ax.figure)
        ag = {f: curves[f]["amplitude"] for f in forms if curves[f]["amplitude"]}
        if ag:
            ax = sweep_mean_sd(ag, "strain_pct", xlabel="Oscillation strain (%)",
                                 title=f"{chem}: amplitude sweeps")
            ax.figure.savefig(figdir / f"fig_4_{na:02d}_amplitude_{chem}.png")
            plt.close(ax.figure)
        fg = {f: curves[f]["frequency"] for f in forms if curves[f]["frequency"]}
        if fg:
            ax = sweep_mean_sd(fg, "omega_rad_s", xlabel="Angular frequency (rad/s)",
                                 title=f"{chem}: frequency sweeps")
            ax.axvspan(100, 1000, color="0.85", alpha=0.6, lw=0)
            ax.axvspan(0.1, 10, color="#ffe9a8", alpha=0.35, lw=0)
            ax.figure.savefig(figdir / f"fig_4_{nf:02d}_frequency_{chem}.png")
            plt.close(ax.figure)
    summ = df.groupby("formulation", sort=False).agg(
        Gp_kPa=("Gp_plateau_Pa", lambda v: v.mean() / 1e3),
        gamma_c_pct=("gamma_c_pct", "median"),
        gamma_c_censored=("gamma_c_censored", "all")).reset_index()
    ax = design_map(summ)
    ax.figure.savefig(figdir / "design_map.png")
    plt.close(ax.figure)


def write_report(out: Path, tabs: dict, qc: list[str]):
    lines = ["# Formulation-matrix rheology: analysis report", ""]
    if qc:
        lines += ["## QC notes", ""] + [f"- {q}" for q in qc] + [""]
    for title, t in tabs.items():
        lines += [f"## {title}", "", t.to_markdown(index=False, floatfmt=".3g"), ""]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")


def matrix_main(argv=None):
    ap = argparse.ArgumentParser(description=MATRIX_MAIN_DOC, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default="data/formulation_matrix/manifest.csv")
    ap.add_argument("--out", default="results/formulation_matrix")
    ap.add_argument("--n-plateau", type=int, default=5)
    ap.add_argument("--fn-tol", type=float, default=0.01)
    ap.add_argument("--gc-drop", type=float, default=0.10)
    ap.add_argument("--no-figures", action="store_true")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    df, curves, qc = analyse(Path(a.manifest), a.n_plateau, a.fn_tol, a.gc_drop)
    df.to_csv(out / "replicate_parameters.csv", index=False)
    tabs = {
        "Table 4.1 Gelation kinetics": table_4_1(df),
        "Table 4.2 Replicate GP, FN and UV shutter times": table_4_2(df),
        "Table 4.3 Viscoelastic properties and network parameters": table_4_3(df),
        "Table 4.4 Two-way ANOVA (log-transformed moduli and γc)": table_4_4(df),
        "Planned comparisons (Welch, Holm within chemistry)": planned(df),
        "Stiffness-matched oppositely charged pairs": matched_pairs(df),
        "Concentration-scaling exponents (G′ ∝ c^a)": scaling(df),
    }
    names = ["table_4_1_gelation", "table_4_2_replicates", "table_4_3_viscoelastic", "table_4_4_anova",
             "planned_comparisons", "matched_pairs", "scaling_exponents"]
    for n, t in zip(names, tabs.values()):
        t.to_csv(out / f"{n}.csv", index=False)
    (out / "qc_log.txt").write_text("\n".join(qc) + "\n", encoding="utf-8")
    if not a.no_figures:
        figures(df, curves, out / "figures")
    write_report(out, tabs, qc)
    pd.set_option("display.width", 200, "display.max_columns", 20)
    for t, d in list(tabs.items())[:4]:
        print(f"\n=== {t} ===\n{d.to_string(index=False)}")
    print("\nQC:\n  " + "\n  ".join(qc))
    print(f"\nResults written to {out.resolve()}")



# ============================================================================================
# RUN_IPN_ANALYSIS
# ============================================================================================
IPN_MAIN_DOC = """Charged IPN analysis (Sec. 4.1, 4.3.1, 4.4).

Inputs in --data (default data/ipn):
  rheology_manifest.csv   group, replicate, amplitude_file      (37 C amplitude sweeps)
  swelling.csv            group, medium, replicate, Ws, Wd      (PBS, 37 C)
  eis_manifest.csv        group, replicate, file                (CHI 700E exports)
  eis_geometry.csv        thickness_cm, area_cm2                (or pass --thickness/--area)

Outputs:
  ipn_rheology.csv, ipn_swelling.csv, ipn_conductivity.csv, ipn_stats.txt
  figures/fig_4_01_modulus.png, fig_4_02_amplitude.png, fig_4_03_yield.png,
          fig_4_13_swelling.png, fig_4_14_eis_<group>.png

Statistics: one-way ANOVA + Tukey HSD for G'LVR, yield strain and swelling ratio."""

#!/usr/bin/env python3


ORDER = ["AMPS", "Cationic", "Zwitterionic", "PEGDA"]


def anova_block(name, df, col, order, log_lines):
    res = one_way_anova_tukey(df, col, "group")
    log_lines.append(f"\n## {name}\nOne-way ANOVA: F({res['df_between']},{res['df_within']}) = {res['F']:.3g}, "
                     f"{fmt_p(res['p'])}\nTukey HSD:\n{res['tukey'].to_string(index=False)}")
    return res


def ipn_main(argv=None):
    ap = argparse.ArgumentParser(description=IPN_MAIN_DOC, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/ipn")
    ap.add_argument("--out", default="results/ipn")
    ap.add_argument("--thickness", type=float, help="sample thickness L (cm)")
    ap.add_argument("--area", type=float, help="electrode contact area A (cm^2)")
    ap.add_argument("--eis-method", default="lowfreq_intercept",
                    choices=["lowfreq_intercept", "lowfreq_plateau", "min_imag", "hf_intercept"])
    ap.add_argument("--yield-tol", type=float, default=0.05,
                    help="fractional deviation from linear stress-strain defining yield")
    a = ap.parse_args(argv)
    data, out = Path(a.data), Path(a.out)
    fig = out / "figures"
    fig.mkdir(parents=True, exist_ok=True)
    log = ["# Charged IPN statistics"]

    # ---------------- rheology ----------------
    man = pd.read_csv(data / "rheology_manifest.csv")
    order = [g for g in ORDER if g in set(man.group)] + [g for g in man.group.unique() if g not in ORDER]
    rows, curves = [], {g: [] for g in order}
    for _, r in man.iterrows():
        d = load_trios(data / r.amplitude_file)
        res = analyze_amplitude_sweep(d, yield_tol=a.yield_tol)
        rows.append({"group": r.group, "replicate": r.replicate, "G_LVR_kPa": res.G_LVR_Pa / 1e3,
                     "yield_strain_pct": res.yield_strain_pct, "gamma_c_pct": res.gamma_c_pct})
        curves[r.group].append(d)
    rheo = pd.DataFrame(rows)
    rheo.to_csv(out / "ipn_rheology.csv", index=False)
    r1 = anova_block("G'LVR (Fig. 4.1)", rheo, "G_LVR_kPa", order, log)
    r3 = anova_block("Yield strain (Fig. 4.3)", rheo, "yield_strain_pct", order, log)
    ax = bar_with_points(rheo, "G_LVR_kPa", "group", order, "Storage modulus G′ (kPa)",
                           pairs=r1["tukey"], show_ns=False)
    ax.figure.savefig(fig / "fig_4_01_modulus.png"); plt.close(ax.figure)
    ax = bar_with_points(rheo, "yield_strain_pct", "group", order, "Yield strain (%)", pairs=r3["tukey"])
    ax.figure.savefig(fig / "fig_4_03_yield.png"); plt.close(ax.figure)
    ax = sweep_mean_sd(curves, "strain_pct", also="Gpp_Pa", xlabel="Oscillation strain (%)",
                         title="Charged IPNs, 1 Hz, 37 °C")
    ax.figure.savefig(fig / "fig_4_02_amplitude.png"); plt.close(ax.figure)

    # ---------------- swelling ----------------
    sw = analyze_gravimetry_table(pd.read_csv(data / "swelling.csv"))
    sw.to_csv(out / "ipn_swelling.csv", index=False)
    r13 = anova_block("Swelling ratio in PBS (Fig. 4.13)", sw, "SR", order, log)
    ax = bar_with_points(sw, "SR", "group", order, "Swelling ratio (Ws/Wd)", pairs=r13["tukey"])
    ax.figure.savefig(fig / "fig_4_13_swelling.png"); plt.close(ax.figure)

    # ---------------- EIS ----------------
    geo = pd.read_csv(data / "eis_geometry.csv").iloc[0] if (data / "eis_geometry.csv").exists() else {}
    L = a.thickness or geo.get("thickness_cm")
    A = a.area or geo.get("area_cm2")
    em = pd.read_csv(data / "eis_manifest.csv")
    cond = []
    for g, grp in em.groupby("group", sort=False):
        spectra = {f"{g[:2].lower()}{r.replicate}": load_chi_impedance(data / r.file) for _, r in grp.iterrows()}
        c = analyze_spectra(spectra, L, A, a.eis_method)
        c.insert(0, "group", g)
        cond.append(c)
        f = nyquist_bode(spectra, title=f"{g} IPN in PBS")
        f.savefig(fig / f"fig_4_14_eis_{g}.png"); plt.close(f)
    cond = pd.concat(cond, ignore_index=True)
    cond.to_csv(out / "ipn_conductivity.csv", index=False)

    # ---------------- summary ----------------
    summ = pd.concat([
        describe(rheo, "G_LVR_kPa", "group")[["n", "mean", "sd"]].add_prefix("G'LVR_kPa_"),
        describe(rheo, "yield_strain_pct", "group")[["mean", "sd"]].add_prefix("yield_pct_"),
        describe(sw, "SR", "group")[["mean", "sd"]].add_prefix("SR_"),
        describe(sw, "EWC_pct", "group")[["mean", "sd"]].add_prefix("EWC_"),
        describe(cond, "sigma_mS_cm", "group")[["mean", "sd"]].add_prefix("sigma_mS_cm_"),
        describe(cond, "R_bulk_ohm", "group")[["mean", "sd"]].add_prefix("Rbulk_ohm_"),
        describe(cond, "rho_ohm_m", "group")[["mean", "sd"]].add_prefix("rho_ohm_m_"),
    ], axis=1).reindex(order)
    summ.to_csv(out / "ipn_summary.csv")
    rc = one_way_anova_tukey(cond, "sigma_mS_cm", "group") if cond.groupby("group").size().min() > 1 else None
    if rc:
        log.append(f"\n## Conductivity\nOne-way ANOVA: F = {rc['F']:.3g}, {fmt_p(rc['p'])}\n{rc['tukey'].to_string(index=False)}")
    (out / "ipn_stats.txt").write_text("\n".join(log) + "\n", encoding="utf-8")
    pd.set_option("display.width", 200, "display.max_columns", 30)
    print(summ.round(3).to_string())
    print("\n".join(log))
    print(f"\nResults written to {out.resolve()}")



# ============================================================================================
# RUN_PRINTER_COMPARISON
# ============================================================================================
PRINTER_MAIN_DOC = """Printed 10 % PEGDA reproducibility: Formlabs vs Phrozen (Sec. 4.3.2, Table 4.6, Figs A.1-A.2).

Inputs (--data, default data/printer_comparison):
  swelling.csv       printer, medium (PBS/DIW), replicate, Ws, Wd   (independent discs per medium)
  gel_fraction.csv   printer, replicate, Wi, Wf

Statistics: two-way ANOVA (printer x medium) on SR, followed by Welch's t-tests;
Welch's t-test on gel fraction."""

#!/usr/bin/env python3


def printer_main(argv=None):
    ap = argparse.ArgumentParser(description=PRINTER_MAIN_DOC, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/printer_comparison")
    ap.add_argument("--out", default="results/printer_comparison")
    a = ap.parse_args(argv)
    data, out = Path(a.data), Path(a.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    sw = analyze_gravimetry_table(pd.read_csv(data / "swelling.csv"))
    gf = analyze_gravimetry_table(pd.read_csv(data / "gel_fraction.csv"))
    sw.to_csv(out / "swelling_results.csv", index=False)
    gf.to_csv(out / "gel_fraction_results.csv", index=False)

    rows = []
    for pr in sw.printer.unique():
        r = {"Printer": pr}
        for med in ("PBS", "DIW"):
            d = sw[(sw.printer == pr) & (sw.medium == med)]
            r[f"SR in {med}"] = mean_sd(d.SR, 2)
            r[f"EWC in {med} (%)"] = mean_sd(d.EWC_pct, 1)
        g = gf[gf.printer == pr]
        r["Gel fraction (%)"] = mean_sd(g.GF_pct, 1)
        r["Sol fraction (%)"] = mean_sd(g.SF_pct, 1)
        rows.append(r)
    t46 = pd.DataFrame(rows)
    t46.to_csv(out / "table_4_6.csv", index=False)

    an = two_way_anova(sw, "SR", "printer", "medium", log=False)
    pw = []
    for med in sw.medium.unique():
        d = sw[sw.medium == med]
        prs = d.printer.unique()
        pw.append({"comparison": f"{med}: {prs[0]} vs {prs[1]}",
                   **welch(d[d.printer == prs[0]].SR, d[d.printer == prs[1]].SR)})
    for pr in sw.printer.unique():
        d = sw[sw.printer == pr]
        pw.append({"comparison": f"{pr}: PBS vs DIW", **welch(d[d.medium == "PBS"].SR, d[d.medium == "DIW"].SR)})
    pw = pd.DataFrame(pw)
    prs = gf.printer.unique()
    gfp = welch(gf[gf.printer == prs[0]].GF_pct, gf[gf.printer == prs[1]].GF_pct)

    txt = [f"Two-way ANOVA (printer x medium) on SR:\n{an.to_string()}",
           f"\nWelch t-tests on SR:\n{pw.to_string(index=False)}",
           f"\nGel fraction {prs[0]} vs {prs[1]}: Welch t = {gfp['t']:.3f}, p = {gfp['p']:.3f}"]
    (out / "stats.txt").write_text("\n".join(txt) + "\n", encoding="utf-8")

    ax = bar_with_points(gf, "GF_pct", "printer", list(prs), "Gel fraction (%)")
    ax.set_ylim(0, 105)
    ax.figure.savefig(out / "figures" / "fig_A_2_gel_fraction.png"); plt.close(ax.figure)
    sw["printer_medium"] = sw.printer + "\n" + sw.medium
    ax = bar_with_points(sw, "SR", "printer_medium", None, "Swelling ratio (Ws/Wd)")
    ax.figure.savefig(out / "figures" / "swelling_printer_medium.png"); plt.close(ax.figure)

    print(t46.to_string(index=False))
    print("\n".join(txt))
    print(f"\nResults written to {out.resolve()}")



# ============================================================================================
# RUN_CELL_ASSAYS
# ============================================================================================
ASSAYS_MAIN_DOC = """Cell / biochemical assay analysis (Sec. 3.8, 3.11.4, 3.11.6, 3.12, 3.13; Sec. 4.5, 4.6, 4.9, 4.10).

Inputs in --data (default data/assays); any missing file is skipped:
  bsa_standard.csv        conc_ug_ml, signal                         (Fig. 4.15)
  alp_kinetics.csv        day, replicate, time_min, A405, protein_signal, protein_dilution
  bca_standard.csv        conc_mg_ml, A562
  bca_release.csv         group, replicate, time_h, A562, dilution
  picogreen_standard.csv  conc_ug_ml, RFU
  picogreen_samples.csv   day, medium, replicate, RFU
  elisa_standard.csv      conc_pg_ml, OD450
  elisa_samples.csv       day, replicate, OD450
  mtt.csv                 group, day, replicate, A570, A_ref, is_blank (optional)

The ALP output gives the numbers needed for the bracketed placeholders in Sec. 4.5."""

#!/usr/bin/env python3


RELEASE_VOLUME_ML = 1.0     # buffer volume per hydrogel (edit to your protocol)


def assays_main(argv=None):
    ap = argparse.ArgumentParser(description=ASSAYS_MAIN_DOC, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/assays")
    ap.add_argument("--out", default="results/assays")
    ap.add_argument("--release-volume-ml", type=float, default=RELEASE_VOLUME_ML)
    a = ap.parse_args(argv)
    d, out = Path(a.data), Path(a.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    log = []

    # ---------------- BSA standard + ALP ----------------
    if (d / "bsa_standard.csv").exists():
        bsa = pd.read_csv(d / "bsa_standard.csv")
        blank = bsa.loc[bsa.conc_ug_ml == 0, "signal"]
        std = fit_linear_standard(bsa.conc_ug_ml, bsa.signal)
        log.append(f"BSA standard: {std.label()}  (range {std.x_min:g}-{std.x_max:g} ug/mL)")
        ax = standard_curve(bsa.conc_ug_ml, bsa.signal, std, ylabel="Signal (RFU)")
        ax.figure.savefig(out / "figures" / "fig_4_15_bsa_standard.png"); plt.close(ax.figure)

        if (d / "alp_kinetics.csv").exists():
            alp = pd.read_csv(d / "alp_kinetics.csv")
            rows = []
            for (day, rep), g in alp.groupby(["day", "replicate"]):
                k = kinetic_slope(g.time_min, g.A405)
                sig = g.protein_signal.iloc[0]
                dil = g.get("protein_dilution", pd.Series([1])).iloc[0]
                prot_ug, ok = std.predict_conc(sig, dil)
                prot = float(prot_ug) / 1000      # mg/mL
                act = alp_activity(slope_per_min=k["slope_per_min"], protein_mg_ml=prot)
                rows.append({"day": day, "replicate": rep, "slope_A_per_min": k["slope_per_min"],
                             "slope_r2": k["r2"], "points_used": k["n_points"], "protein_mg_ml": prot,
                             "protein_dilution": dil, "protein_in_range": bool(ok),
                             "ALP_nmol_min_mg": float(act)})
            alp_res = pd.DataFrame(rows)
            alp_res.to_csv(out / "alp_results.csv", index=False)
            over = alp_res[~alp_res.protein_in_range]
            if len(over):
                log.append(f"WARNING: {len(over)} lysate(s) outside the BSA standard range -> dilute and re-assay "
                           f"(suggested dilution 1:{suggest_dilution(over.protein_mg_ml * 1000, std)})")
            summ = alp_res.groupby("day").agg(protein=("protein_mg_ml", mean_sd),
                                              ALP=("ALP_nmol_min_mg", lambda v: mean_sd(v, 1)))
            groups = [g.ALP_nmol_min_mg.values for _, g in alp_res.groupby("day")]
            F, p = st.f_oneway(*groups)
            tk = one_way_anova_tukey(alp_res.assign(day=alp_res.day.map(lambda x: f"d{x:02d}")), "ALP_nmol_min_mg", "day")
            peak_day = alp_res.groupby("day").ALP_nmol_min_mg.mean().idxmax()
            log.append("\nALP (Sec. 4.5 placeholders):\n" + summ.to_string()
                       + f"\nPeak ALP at day {peak_day}. One-way ANOVA across days: F = {F:.3g}, {fmt_p(p)}"
                       + f"\nTukey:\n{tk['tukey'].to_string(index=False)}")
            ax = bar_with_points(alp_res.assign(day=alp_res.day.map(lambda x: f"Day {x}")),
                                   "ALP_nmol_min_mg", "day", None, "ALP (nmol pNP/min/mg protein)")
            ax.figure.savefig(out / "figures" / "alp_activity.png"); plt.close(ax.figure)

    # ---------------- BCA release ----------------
    if (d / "bca_standard.csv").exists() and (d / "bca_release.csv").exists():
        bs = pd.read_csv(d / "bca_standard.csv")
        bstd = fit_linear_standard(bs.conc_mg_ml, bs.A562)
        rel = pd.read_csv(d / "bca_release.csv")
        rows = []
        for (grp, rep), g in rel.groupby(["group", "replicate"], sort=False):
            g = g.sort_values("time_h")
            c, ok = bstd.predict_conc(g.A562, g.dilution)
            cr = cumulative_release(g.time_h, np.clip(c, 0, None), a.release_volume_ml)
            cr.insert(0, "replicate", rep)
            cr.insert(0, "group", grp)
            cr["in_range"] = ok
            rows.append(cr)
        rel_res = pd.concat(rows, ignore_index=True)
        rel_res.to_csv(out / "release_results.csv", index=False)
        log.append(f"\nBCA standard: {bstd.label()}")
        fig, ax = plt.subplots(figsize=(5, 3.6))
        for i, (grp, g) in enumerate(rel_res.groupby("group", sort=False)):
            m = g.groupby("time_h").cumulative_mg.agg(["mean", "std"])
            ax.errorbar(m.index, m["mean"] * 1000, m["std"] * 1000, marker="o", capsize=3,
                        color=color_for(grp, i), label=grp)
        ax.set_xlabel("Time (h)"); ax.set_ylabel("Cumulative protein released (µg)"); ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(out / "figures" / "protein_release.png"); plt.close(fig)

    # ---------------- PicoGreen DNA ----------------
    if (d / "picogreen_standard.csv").exists() and (d / "picogreen_samples.csv").exists():
        ps = pd.read_csv(d / "picogreen_standard.csv")
        pstd = fit_linear_standard(ps.conc_ug_ml, ps.RFU)
        dna = pd.read_csv(d / "picogreen_samples.csv")
        dna["DNA_ug_ml"], dna["in_range"] = pstd.predict_conc(dna.RFU)
        dna.to_csv(out / "dna_results.csv", index=False)
        log.append(f"\nPicoGreen standard: {pstd.label()}")
        for med, g in dna.groupby("medium"):
            r = one_way_anova_tukey(g.assign(day=g.day.astype(str)), "DNA_ug_ml", "day")
            log.append(f"DNA, {med} medium: one-way ANOVA F = {r['F']:.3g}, {fmt_p(r['p'])}\n"
                       + g.groupby("day").DNA_ug_ml.agg(mean_sd).to_string())

    # ---------------- BMP-2 ELISA ----------------
    if (d / "elisa_standard.csv").exists() and (d / "elisa_samples.csv").exists():
        es = pd.read_csv(d / "elisa_standard.csv")
        fit = fit_4pl(es.conc_pg_ml, es.OD450)
        el = pd.read_csv(d / "elisa_samples.csv")
        el["BMP2_pg_ml"] = inverse_4pl(el.OD450, fit["params"])
        el.to_csv(out / "elisa_results.csv", index=False)
        r = one_way_anova_tukey(el.assign(day=el.day.astype(str)), "BMP2_pg_ml", "day")
        log.append(f"\nBMP-2 ELISA 4PL fit R² = {fit['r2']:.4f}; max sample = {el.BMP2_pg_ml.max():.1f} pg/mL "
                   f"({'all < 100 pg/mL' if el.BMP2_pg_ml.max() < 100 else 'some >= 100 pg/mL'}); "
                   f"change over time: F = {r['F']:.3g}, {fmt_p(r['p'])}")

    # ---------------- MTT ----------------
    if (d / "mtt.csv").exists():
        m = pd.read_csv(d / "mtt.csv")
        res = []
        for (grp, day), g in m.groupby(["group", "day"]):
            blank = g[g.get("is_blank", False) == True]  # noqa: E712
            cells = g[g.get("is_blank", False) != True]  # noqa: E712
            v = mtt_viability(cells.A570, cells.get("A_ref"), blank.A570 if len(blank) else None,
                                blank.get("A_ref") if len(blank) else None)
            res.append(cells.assign(A_corrected=v.A_corrected.values))
        pd.concat(res).to_csv(out / "mtt_results.csv", index=False)

    (out / "assay_log.txt").write_text("\n".join(log) + "\n", encoding="utf-8")
    print("\n".join(log))
    print(f"\nResults written to {out.resolve()}")



# ============================================================================================
# SELF-TESTS (python thesis_analysis.py test)
# ============================================================================================
class _Approx:
    def __init__(self, expected, rel=None, abs=None):
        self.e, self.rel, self.abs = expected, rel, abs

    def __eq__(self, other):
        tol = max(self.abs or 0.0, (self.rel if self.rel is not None else (1e-6 if self.abs is None else 0.0)) * abs(self.e))
        return bool(abs(float(other) - float(self.e)) <= tol)

    def __repr__(self):
        return f"{self.e} ± {self.abs or self.rel}"


class _Mark:
    @staticmethod
    def parametrize(names, values):
        def deco(f):
            f._params = (names, list(values))
            return f
        return deco


class pytest:  # minimal stand-in so the tests run without pytest installed
    approx = staticmethod(lambda e, rel=None, abs=None: _Approx(e, rel, abs))
    mark = _Mark()




# ============================================================================================
# TEST CASES
# ============================================================================================
# Regression tests: the code must reproduce numbers printed in the thesis.
#
#     python -m pytest -q
# ---------------------------------------------------------------- Table 3.1
TABLE_3_1 = {  # (monomer, conc, ratio): (monomer uL, MBAA uL, DIW uL)
    ("PEGDA", 10, None): (178.6, None, 1721.4), ("PEGDA", 20, None): (357.1, None, 1542.9),
    ("AMPS", 10, 10): (331.8, 336.3, 1231.9), ("AMPS", 10, 25): (331.8, 134.5, 1433.7),
    ("AMPS", 10, 50): (331.8, 67.3, 1500.9), ("AMPS", 20, 10): (663.6, 672.6, 563.8),
    ("AMPS", 20, 25): (663.6, 269.0, 967.4), ("AMPS", 20, 50): (663.6, 134.5, 1101.9),
    ("DEAEAAm", 10, 10): (400.0, 452.8, 1047.2), ("DEAEAAm", 10, 25): (400.0, 181.1, 1318.9),
    ("DEAEAAm", 10, 50): (400.0, 90.6, 1409.4), ("DEAEAAm", 20, 10): (800.0, 905.6, 194.4),
    ("DEAEAAm", 20, 25): (800.0, 362.2, 737.8), ("DEAEAAm", 20, 50): (800.0, 181.1, 918.9),
    ("DMAAPS", 10, 10): (500.0, 276.9, 1123.1), ("DMAAPS", 10, 25): (500.0, 110.8, 1289.2),
    ("DMAAPS", 10, 50): (500.0, 55.4, 1344.6), ("DMAAPS", 20, 10): (1000.0, 553.8, 346.2),
    ("DMAAPS", 20, 25): (1000.0, 221.5, 678.5), ("DMAAPS", 20, 50): (1000.0, 110.8, 789.2),
}


@pytest.mark.parametrize("key,expected", TABLE_3_1.items())
def test_table_3_1(key, expected):
    r = formulation_recipe(*key)
    assert r["monomer_stock_uL"] == pytest.approx(expected[0], abs=0.15)
    if expected[1] is not None:
        assert r["MBAA_stock_uL"] == pytest.approx(expected[1], abs=0.15)
    assert r["DIW_uL"] == pytest.approx(expected[2], abs=0.25)
    assert r["LAP_final_pct_wv"] == pytest.approx(0.1)


def test_table_3_2_total():
    assert table_3_2().iloc[-1]["total_gels"] == 240


# ---------------------------------------------------------------- geometry
def test_disc_and_gap():
    assert disc_volume_ul() == pytest.approx(50.3, abs=0.05)
    assert rheometer_gap_um() == pytest.approx(995, abs=5)


def test_mold_eq_3_6():
    w = MicropillarWell()
    assert w.cavity_ul == pytest.approx(10.60, abs=0.01)
    assert w.post_ul == pytest.approx(3.72 + 0.76, abs=0.01)
    assert w.free_volume_ul == pytest.approx(6.12, abs=0.01)
    assert 6 * w.free_volume_ul == pytest.approx(36.7, abs=0.05)
    assert w.surface_area_mm2 == pytest.approx(30.6, abs=0.05)


def test_adsorption_sensitivity():
    s = adsorption_sensitivity()
    assert s["adsorbed_ng"] == pytest.approx(15.3, abs=0.05)
    assert s["supernatant_ug_ml"] == pytest.approx(99.92, abs=0.01)
    assert s["direct_elution_ug_ml"] == pytest.approx(0.15, abs=0.01)
    assert s["supernatant_in_BCA_range"] and not s["elution_in_BCA_range"]


# ---------------------------------------------------------------- Table 4.5
# NB: Table 4.5 prints 9.52 % for the 6.25 uL row; 10 x 238 / 250.5 = 9.50 % (thesis typo)
@pytest.mark.parametrize("v,mM,peg", [(2.0, 0.83, 9.83), (2.5, 1.03, 9.79), (6.25, 2.50, 9.50),
                                      (12.5, 4.75, 9.05), (25.0, 8.68, 8.26)])
def test_table_4_5(v, mM, peg):
    r = redox_dilution(v)
    assert r["final_mM_each"] == pytest.approx(mM, abs=0.01)
    assert r["final_PEGDA_pct_wv"] == pytest.approx(peg, abs=0.015)


def test_bulk_redox():
    r = redox_dilution(60, v_pegda_ul=2000)
    assert r["final_mM_each"] == pytest.approx(2.83, abs=0.01)
    assert r["final_PEGDA_pct_wv"] == pytest.approx(9.43, abs=0.01)
    assert stock_mass_g(100, 228.2, 2) == pytest.approx(0.0456, abs=1e-4)
    assert stock_mass_g(100, 446.1, 2) == pytest.approx(0.0892, abs=1e-4)


# ---------------------------------------------------------------- theory (Sec. 2.5.1)
def test_donnan_worked_example():
    cp, _ = donnan_concentrations(50, 150)
    assert cp == pytest.approx(177, abs=0.5)
    assert donnan_potential_mV(50, 150) == pytest.approx(-4, abs=0.5)
    cp2, _ = donnan_concentrations(440, 150)
    assert cp2 == pytest.approx(480, rel=0.02)
    assert donnan_potential_mV(440, 150) == pytest.approx(-30, abs=1)
    assert donnan_potential_mV(50, 1e-3) < -90          # "order of -100 mV" in water


def test_debye_and_sulfonate():
    assert debye_length_nm(0.16) == pytest.approx(0.76, abs=0.02)
    assert fixed_charge_from_formulation(10, 229.23) == pytest.approx(436, abs=1)


def test_network_parameters_table_4_3():
    # (G'p kPa, nu, xi) from Table 4.3
    for G, nu, xi in [(50.1, 20.2, 4.3), (191, 76.9, 2.8), (1.67, 0.67, 13.5), (0.53, 0.21, 19.8),
                      (30.6, 12.3, 5.1), (24.5, 9.90, 5.5)]:
        assert chain_density_mol_m3(G * 1e3) == pytest.approx(nu, rel=0.02)
        assert mesh_size_nm(G * 1e3) == pytest.approx(xi, abs=0.15)


def test_scaling_exponent_pegda():
    assert concentration_scaling_exponent(50.1, 191) == pytest.approx(1.9, abs=0.05)


def test_youngs():
    assert youngs_from_shear(1.0) == pytest.approx(3.0)


def test_flory_rehner_roundtrip():
    nu = flory_rehner_nu(0.1, 0.45)
    assert flory_rehner_phi(nu, 0.45) == pytest.approx(0.1, rel=1e-6)
    assert flory_rehner_phi(nu, 0.45, pi_donnan_Pa=1e3) < 0.1    # Donnan pressure swells the gel


# ---------------------------------------------------------------- statistics
def test_grubbs_critical_value():
    assert grubbs_critical(4) == pytest.approx(1.481, abs=0.001)


def test_grubbs_amps_outlier():
    t = grubbs_test([1.9, 2.15, 2.4, 22.5])
    assert t["outlier"] and t["value"] == 22.5
    assert t["G"] == pytest.approx(1.500, abs=0.001)


Z4 = np.array([-1.1619, -0.3873, 0.3873, 1.1619])


def _group(m, s):
    return m + s * Z4


def test_printer_two_way_anova_table_4_6():
    """Two-way ANOVA depends only on cell means/SDs when balanced -> must match Table 4.6."""
    rows = []
    for (pr, med), (m, s) in {("F", "PBS"): (6.18, 0.30), ("F", "DIW"): (6.57, 0.69),
                              ("P", "PBS"): (5.60, 0.47), ("P", "DIW"): (6.63, 0.57)}.items():
        rows += [{"printer": pr, "medium": med, "SR": v} for v in _group(m, s)]
    a = two_way_anova(pd.DataFrame(rows), "SR", "printer", "medium", log=False)
    assert a.loc["printer", "F"] == pytest.approx(0.96, abs=0.03)
    assert a.loc["printer", "p"] == pytest.approx(0.35, abs=0.01)
    assert a.loc["medium", "F"] == pytest.approx(7.19, abs=0.1)
    assert a.loc["medium", "p"] == pytest.approx(0.020, abs=0.002)
    assert a.loc["printer:medium", "p"] == pytest.approx(0.25, abs=0.01)


def test_gel_fraction_welch():
    w = welch(_group(94.2, 2.2), _group(93.2, 3.2))
    assert w["p"] == pytest.approx(0.62, abs=0.01)


def test_matched_pair_welch_20pct():
    # AMPS 20% 10:1 (24.5 ± 7.4) vs DEAEAAm 20% 10:1 (30.6 ± 5.3), thesis p = 0.25.
    # On the raw scale Welch depends only on mean/SD/n (p = 0.233); the thesis tests
    # log-transformed moduli, whose p depends on the individual gels, so only the
    # conclusion (not significant) can be checked from the summary values.
    w = welch(_group(24.5, 7.4), _group(30.6, 5.3), log=True)
    assert 0.15 < w["p"] < 0.35


def test_matched_pair_welch_10pct():
    w = welch(_group(2.51, 0.37), _group(1.67, 0.63))
    assert w["p"] == pytest.approx(0.07, abs=0.01)


def test_holm():
    from statsmodels.stats.multitest import multipletests
    p = multipletests([0.01, 0.04, 0.03, 0.5], method="holm")[1]
    assert p[0] == pytest.approx(0.04)


# ---------------------------------------------------------------- rheology definitions
def _ts(dt=6.0, uv=30.0, tau=15.0, Gp=1000.0):
    t = np.arange(0, 300, dt)
    x = np.clip(t - uv, 0, None)
    G = np.where(t > uv, Gp * (1 - np.exp(-x / tau)), 1.0)
    Gpp = np.where(t > uv, 0.05 * G, 5.0)
    return pd.DataFrame({"time_s": t, "Gp_Pa": G, "Gpp_Pa": Gpp})


def test_time_sweep_definitions():
    r = analyze_time_sweep(_ts(), uv_on_s=30.0)
    assert r.GP_s == pytest.approx(6.0)                    # first sample after UV
    assert r.Gp_plateau_Pa == pytest.approx(1000, rel=1e-3)
    # t90: onset at first point > 10x baseline (6 s), 90 % at -tau ln(0.1)=34.5 -> sample 36 s
    assert r.onset_s == pytest.approx(6.0)
    assert r.t90_s == pytest.approx(36 - 6)
    # FN: consecutive changes <= 1 %  -> exp(-x/15)(1-exp(-6/15)) <= 0.01*(...)
    assert 50 <= r.FN_s <= 70


def test_amplitude_gamma_c_log_interp():
    g = np.logspace(-2, 2, 41)
    G = 1000 / (1 + (g / 10.0) ** 2)                  # 0.9 G_LVR at g = 10/3
    r = analyze_amplitude_sweep(pd.DataFrame({"strain_pct": g, "Gp_Pa": G}))
    assert r.gamma_c_pct == pytest.approx(10 / 3, rel=0.03)
    assert not r.gamma_c_censored


def test_amplitude_censored():
    g = np.logspace(-2, np.log10(270), 40)
    r = analyze_amplitude_sweep(pd.DataFrame({"strain_pct": g, "Gp_Pa": np.full_like(g, 500)}))
    assert r.gamma_c_censored and r.gamma_c_pct == pytest.approx(270)


def test_frequency_slope_and_inertia_exclusion():
    w = np.logspace(np.log10(0.063), np.log10(628), 41)
    G = 1000 * w ** 0.02 * (1 + 10 * (w / 300) ** 3)
    r = analyze_frequency_sweep(pd.DataFrame({"omega_rad_s": w, "Gp_Pa": G}))
    assert r.slope_n == pytest.approx(0.02, abs=0.005)


def test_duplicate_detection():
    a = _ts()
    assert find_duplicate_replicates({1: a, 2: _ts(tau=20), 3: a.copy()}) == [(1, 3)]


# ---------------------------------------------------------------- gravimetry / EIS / assays
def test_swelling_and_gel_fraction():
    s = swelling([10.0], [1.0], "DIW").iloc[0]
    assert s.SR == 10 and s.EWC_pct == pytest.approx(90)
    p = swelling([10.0], [1.0], "PBS").iloc[0]
    assert p.SR > s.SR                                  # salt removed from dry mass
    g = gel_fraction([10.0], [9.4]).iloc[0]
    assert g.GF_pct == pytest.approx(94) and g.SF_pct == pytest.approx(6)


def test_conductivity_zwitterionic():
    c = conductivity(196.0, thickness_cm=0.2, area_cm2=0.2 / 0.2499)
    assert float(c["sigma_mS_cm"]) == pytest.approx(1.275, abs=0.005)
    assert float(c["rho_ohm_m"]) == pytest.approx(7.85, abs=0.03)


def test_bulk_resistance_lowfreq():
    spec = simulate_spectrum(196.0, R_hf=20.0)
    assert bulk_resistance(spec) == pytest.approx(196.0, rel=0.02)
    assert bulk_resistance(spec, "hf_intercept") < 40


def test_alp_units():
    # 0.0185 A/min, d = 1 cm, Vt = 1 mL, Vs = 1 mL, 1 mg/mL -> 1 umol/L/min * 1 mL = 1 nmol/min
    a = alp_activity(slope_per_min=0.0185, Vt_ml=1, Vs_ml=1, path_cm=1, protein_mg_ml=1)
    assert a == pytest.approx(1.0)


def test_standard_curve_and_4pl():
    x = np.array([0, 25, 50, 100, 200.0])
    std = fit_linear_standard(x, 0.5428 * x + 7.0489)
    assert std.slope == pytest.approx(0.5428) and std.r2 == pytest.approx(1)
    conc, ok = std.predict_conc([7.0489 + 0.5428 * 80, 500])
    assert conc[0] == pytest.approx(80) and ok[0] and not ok[1]
    c = np.array([31.3, 62.5, 125, 250, 500, 1000, 2000])
    od = four_pl(c, 0.05, 1.2, 600, 2.6)
    f = fit_4pl(c, od)
    assert inverse_4pl(four_pl(80, *f["params"]), f["params"]) == pytest.approx(80, rel=1e-3)



def run_selftest(argv=None):
    tests = [(n, f) for n, f in globals().items() if n.startswith("test_") and callable(f)]
    passed, failed = 0, []
    for name, f in tests:
        cases = [()]
        if hasattr(f, "_params"):
            names, values = f._params
            multi = "," in names
            cases = [v if multi else (v,) for v in values]
        for c in cases:
            try:
                f(*c)
                passed += 1
            except Exception as e:  # noqa: BLE001
                failed.append(f"{name}{c if c else ''}: {type(e).__name__} {e}")
    print(f"{passed} passed, {len(failed)} failed")
    for msg in failed:
        print("  FAIL", msg)
    return 1 if failed else 0


# ============================================================================================
# COMMAND LINE
# ============================================================================================
def run_everything(argv=None):
    ap = argparse.ArgumentParser(description="Run the full pipeline")
    ap.add_argument("--demo", action="store_true", help="regenerate the SYNTHETIC demo data first")
    a = ap.parse_args(argv)
    if a.demo or not Path("data/formulation_matrix/manifest.csv").exists():
        print("Generating SYNTHETIC demo data in ./data ...")
        demo_main(["--out", "data"])
    steps = [("Calculators (Tables 3.1, 3.2, 4.5; Eq. 3.6; Donnan)", calculators_main),
             ("Formulation-matrix rheology (Tables 4.1-4.4, Figs 4.4-4.12)", matrix_main),
             ("Charged IPNs (Figs 4.1-4.3, 4.13, 4.14; conductivity)", ipn_main),
             ("Printer comparison (Table 4.6, Fig. A.2)", printer_main),
             ("Cell assays (Fig. 4.15, ALP, release, DNA, BMP-2)", assays_main)]
    for title, fn in steps:
        print(f"\n{'=' * 80}\n{title}\n{'=' * 80}")
        try:
            fn([])
        except Exception as e:  # noqa: BLE001
            print(f"!! step failed: {title}: {type(e).__name__}: {e}")
    print(f"\nAll outputs in {Path('results').resolve()}")


COMMANDS = {
    "all": run_everything, "test": run_selftest, "demo": demo_main, "calculators": calculators_main,
    "matrix": matrix_main, "ipn": ipn_main, "printer": printer_main, "assays": assays_main,
}


def main():
    if len(sys.argv) < 2:
        return run_everything([])
    cmd, rest = sys.argv[1], sys.argv[2:]
    if cmd in ("-h", "--help") or cmd not in COMMANDS:
        print(__doc__)
        return 0 if cmd in ("-h", "--help") else 2
    return COMMANDS[cmd](rest)


if __name__ == "__main__":
    sys.exit(main() or 0)

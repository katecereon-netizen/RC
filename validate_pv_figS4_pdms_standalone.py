"""
validate_pv_figS4_pdms_standalone.py   (STANDALONE: numpy, pandas, scipy, matplotlib, openpyxl only)

Fig. S4 of Li, Ahmed & Ma, Sol. RRL 5 (2021) 2000735, SI (windshield night, G = 0), 18:00-24:00:
ventilated normal and PDMS-coated PV modules with the spectral radiative-cooling (RC) front surface
of PV_RC_model_equations.pdf.                                                                   -- [L], [M]

Model  5-node lumped balance Glass-EVA-Si-EVA-TPT of validate_pv_fig3.py (G = 0: no solar terms) -- [V]
Runs   (V = 0 behind the windshield; every layer starts at that module's first measured temperature)
    Normal, grey      glass grey, eps_g = 0.85                                                     [M] Eq. (19)
    Normal, spectral  Q_rad - Q_atm - Q_grd with the glass-front eps(lambda)                       [M] Eq. (20)
    PDMS,   spectral  same equations, PDMS-glass-front eps(lambda) only; the film adds no heat capacity
Front eps(lambda): 0.28-2.5 um alpha = 1 - tau - rho ([L] Fig. S1), linear bridge,
4-16.5 um module eps of [L] Fig. 2c, tail beyond.
Sky: T_sky = eps_sky**0.25 T_a, eps_sky from the zenith transmittance tau0(lambda) of [L] Fig. 2c
(TWC = 1000 atm-cm), used in Q_atm, the grey terms and the ground correlation alike.

Files: data.xlsx (sheet "Fig S4"), li2021_fig2_digitized.xlsx (sheets "Fig S1", "Fig 2c", "Fig 2c sky")
"""

from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

# ===========================================================================
# 1. PARAMETERS
# ===========================================================================

# --- Files -------------------------------------------------------------------
F_DATA   = Path(r"C:\Users\bildiran.shop\OneDrive\Desktop\RC\Case.7\Data\data.xlsx")
F_DIG    = F_DATA.parent / "li2021_fig2_digitized.xlsx"
SHEET    = "Fig S4"
OUT_DIR  = Path("output_figS4_pdms")

# --- Window / weather --------------------------------------------------------
T_START  = "2020-10-09 18:00"   # -    window start                                   [L] Fig. S4b
T_END    = "2020-10-10 00:00"   # -    window end
WIND     = 0.0                  # -    multiplier on measured wind (windshield)

# --- Geometry / view factors -------------------------------------------------
A_MOD    = 0.18 * 0.18          # m2   module area                                    [V]
PF       = 0.75                 # -    cell packing factor                            [V]
TILT     = np.radians(30.0)     # rad  module tilt                                    [V]
F_G_SKY  = 0.5 * (1 + np.cos(TILT))             # glass -> sky
F_G_GRD  = 0.5 * (1 - np.cos(TILT))             # glass -> ground
F_T_SKY  = 0.5 * (1 + np.cos(np.pi - TILT))     # TPT   -> sky
F_T_GRD  = 0.5 * (1 - np.cos(np.pi - TILT))     # TPT   -> ground

# --- Optical: thermal IR -----------------------------------------------------
EPS_GLASS = 0.85                # -      grey glass emissivity                        [V]
EPS_TPT   = 0.90                # -      grey TPT emissivity                          [V]
TWC       = 1000                # atm-cm sky water column (Fig. 2c orange fill)       [L] Secs. 2.2.2, 2.4
LAM_TAIL  = 15.5                # um     eps beyond 16.5 um = mean of 15.5-16.5 um
LAM_MAX   = 100.0               # um     upper limit of the Planck integrals
N_THETA   = 60                  # -      zenith-angle quadrature points               [L] Eq. (8)
T_TAB     = np.arange(250.0, 320.01, 0.25)   # K  temperature grid of the RC tables

# --- Thermal: layers glass, EVA1, Si, EVA2, TPT ------------------------------
GLASS = dict(k=2.0,   rho=2450.0, cp=500.0,  d=3.2e-3)    # [V]  k W/mK, rho kg/m3, cp J/kgK, d m
EVA   = dict(k=0.31,  rho=950.0,  cp=2090.0, d=0.5e-3)    # [V]
SI    = dict(k=130.0, rho=2330.0, cp=677.0,  d=0.19e-3)   # [V]
TPT   = dict(k=0.15,  rho=1200.0, cp=1250.0, d=0.3e-3)    # [V]

# --- Convection ([V], McAdams-type) and ground -------------------------------
H_TOP = (5.7, 3.8)              # W/m2K  h_top  = 5.7 + 3.8 V
H_BACK = (2.8, 3.0)             # W/m2K  h_back = 2.8 + 3.0 V
GRD   = (1.205, -0.201, -0.466) # T_grd = c0 T_a + c1 T_sky + c2 V, G = 0             [M] Eq. (5)

# --- Constants ---------------------------------------------------------------
SIGMA  = 5.670374419e-8         # W/m2K4
H_PL   = 6.62607015e-34         # J s
C_LIGHT = 2.99792458e8          # m/s
K_B    = 1.380649e-23           # J/K

# --- Plot style --------------------------------------------------------------
COLORS = {"amb": "#8C8C8C", "meas_N": "#1F6FB4", "meas_P": "#2A9D8F",
          "Normal, grey": "#9C6ADE", "Normal, spectral": "#E4572E", "PDMS, spectral": "#0B6E4F"}
STYLE  = {"font.family": "serif", "font.size": 11, "axes.labelsize": 12, "axes.titlesize": 12,
          "axes.titleweight": "bold", "xtick.direction": "in", "ytick.direction": "in",
          "grid.linestyle": "--", "grid.alpha": 0.35, "savefig.dpi": 300, "savefig.bbox": "tight"}

# --- Derived thermal constants -----------------------------------------------
_half = lambda m: m["d"] / (2 * m["k"] * A_MOD)                 # half-layer resistance, K/W
R_GE1  = _half(GLASS) + _half(EVA)                              # K/W  glass-EVA1
R_E1SI = _half(EVA) + _half(SI)                                 # K/W  EVA1-Si  (= Si-EVA2)
R_E2T  = _half(EVA) + _half(TPT)                                # K/W  EVA2-TPT
_cap   = lambda m: m["rho"] * m["cp"] * m["d"] * A_MOD          # heat capacity, J/K   [M] Eq. (1)
C_G, C_E, C_SI, C_T = _cap(GLASS), _cap(EVA), _cap(SI) * PF, _cap(TPT)

# ===========================================================================
# 2. EQUATIONS
# ===========================================================================

def planck(T, lam_um):
    """Spectral blackbody radiance B(T, lambda), W/m2/sr/um.   [L] below Eq. (5)"""
    lam = lam_um * 1e-6
    return 2 * H_PL * C_LIGHT ** 2 / lam ** 5 / np.expm1(H_PL * C_LIGHT / (lam * K_B * np.atleast_1d(T)[:, None])) * 1e-6


def eps_atm(tau0):
    """Directional atmospheric emissivity on the zenith grid, shape (N_THETA, n_lambda).   [M] Eq. (13)"""
    th = (np.arange(N_THETA) + 0.5) * 0.5 * np.pi / N_THETA
    return 1.0 - tau0[None, :] ** (1.0 / np.cos(th)[:, None]), th


def sky_table(lam, tau0):
    """Hemispherical sky emissivity on T_TAB: int int eps_atm B(T_a) 2 sin cos / int B(T_a) dlambda.   [L] Eq. (8)"""
    e, th = eps_atm(tau0)
    w = (2 * np.sin(th) * np.cos(th) * 0.5 * np.pi / N_THETA)[:, None]
    B = planck(T_TAB, lam)
    return np.trapezoid(B * (w * e).sum(0), lam, axis=1) / np.trapezoid(B, lam, axis=1)


def t_sky(T_a, eps_tab):
    """T_sky = eps_sky**0.25 T_a, K.   [V] Eq. (9) form"""
    return np.interp(T_a, T_TAB, eps_tab) ** 0.25 * T_a


def rc_tables(eps, lam, tau0):
    """Front-surface tables on T_TAB, W/m2:
        q_rad(T) = pi int I_B(T) eps dlambda                                         [M] Eq. (11)
        q_atm(T) = int I_B(T) K dlambda, K = 2 pi int sin cos eps eps_atm dtheta      [M] Eqs. (12), (14)"""
    e, th = eps_atm(tau0)
    K = eps * ((2 * np.pi * np.sin(th) * np.cos(th) * 0.5 * np.pi / N_THETA)[:, None] * e).sum(0)
    B = planck(T_TAB, lam)
    return dict(q_rad=np.pi * np.trapezoid(B * eps, lam, axis=1), q_atm=np.trapezoid(B * K, lam, axis=1))


def q_front(T_g, T_a, T_s, T_grd, tab):
    """Net front radiative loss per area, W/m2.
    grey (tab None):  eps_g sigma [F_sky (T_g^4 - T_sky^4) + F_grd (T_g^4 - T_grd^4)]   [M] Eqs. (8), (19)
    spectral:         Q_rad(T_g) - F_sky Q_atm(T_a) - F_grd Q_rad(T_grd), eps_grd = 1     [M] Eqs. (15)-(16)"""
    if tab is None:
        return EPS_GLASS * SIGMA * (F_G_SKY * (T_g ** 4 - T_s ** 4) + F_G_GRD * (T_g ** 4 - T_grd ** 4))
    q = lambda T: np.interp(T, T_TAB, tab["q_rad"])
    return q(T_g) - F_G_SKY * np.interp(T_a, T_TAB, tab["q_atm"]) - F_G_GRD * q(T_grd)


def thermal_equations(t, T, w, eps_tab, tab):
    """Energy balance of the five layers, G = 0.   [V] Eqs. (21)-(24); glass node with q_front   [M] Eq. (20)"""
    T_g, T_e1, T_si, T_e2, T_t = T
    T_a, V = np.interp(t, w["t"], w["T_a"]), WIND * np.interp(t, w["t"], w["V"])
    T_s = t_sky(T_a, eps_tab)
    T_grd = GRD[0] * T_a + GRD[1] * T_s + GRD[2] * V                                   # [M] Eq. (5)
    h_top, h_back = H_TOP[0] + H_TOP[1] * V, H_BACK[0] + H_BACK[1] * V
    back = A_MOD * (h_back * (T_t - T_a)
                    + EPS_TPT * SIGMA * (F_T_SKY * (T_t ** 4 - T_s ** 4) + F_T_GRD * (T_t ** 4 - T_grd ** 4)))
    return np.array([
        ((T_e1 - T_g) / R_GE1 - A_MOD * (h_top * (T_g - T_a) + q_front(T_g, T_a, T_s, T_grd, tab))) / C_G,
        ((T_si - T_e1) / R_E1SI - (T_e1 - T_g) / R_GE1) / C_E,
        (-(T_si - T_e1) / R_E1SI - (T_si - T_e2) / R_E1SI) / C_SI,
        ((T_si - T_e2) / R_E1SI - (T_e2 - T_t) / R_E2T) / C_E,
        ((T_e2 - T_t) / R_E2T - back) / C_T])


def error_metrics(model, meas):
    r = model - meas
    return dict(mae=np.mean(np.abs(r)), rmse=np.sqrt(np.mean(r ** 2)), mape=np.mean(np.abs(r / meas)) * 100,
                rmspe=np.sqrt(np.mean((r / meas) ** 2)) * 100, r2=1 - np.sum(r ** 2) / np.sum((meas - meas.mean()) ** 2))

# ===========================================================================
# 3. DATA AND SOLVING
# ===========================================================================

@lru_cache(maxsize=None)
def read_digitized():
    """Digitized curves of [L]: Fig. S1 (tau, rho), Fig. 2c (module eps), Fig. 2c sky (tau0)."""
    x = pd.ExcelFile(F_DIG)
    return x.parse("Fig S1"), x.parse("Fig 2c"), x.parse("Fig 2c sky")


def read_sky():
    """tau0(lambda) of [L] Fig. 2c at TWC on its own points, zero outside 4-16.5 um."""
    s = read_digitized()[2]
    lam = np.concatenate([np.arange(0.5, s.wavelength_um.iloc[0], 0.01), s.wavelength_um,
                          np.arange(s.wavelength_um.iloc[-1] + 0.05, LAM_MAX + 1e-9, 0.05)])
    return lam, np.interp(lam, s.wavelength_um, s[f"tau_{TWC}"], left=0.0, right=0.0)


def read_eps(lam):
    """Front eps(lambda) on lam: alpha = 1 - tau - rho of [L] Fig. S1 (Kirchhoff) -> bridge -> [L] Fig. 2c -> tail."""
    f, c, _ = read_digitized()
    eps = {}
    for case, (s1, col) in {"Normal": ("glass", "normal"), "PDMS": ("pdms", "pdms")}.items():
        alpha = 1.0 - f[f"tau_{s1}"] - f[f"rho_{s1}"]
        e = np.interp(lam, np.r_[f.wavelength_um, c.wavelength_um], np.r_[alpha, c[f"eps_{col}"]])
        tail = c[f"eps_{col}"][c.wavelength_um >= LAM_TAIL].mean()
        eps[case] = np.where(lam > c.wavelength_um.iloc[-1], tail, e)
    return eps


def read_figS4():
    """Weather (1 min) and measured normal / PDMS back-surface temperatures, 18:00-24:00."""
    d = pd.read_excel(F_DATA, sheet_name=SHEET)
    t0, t1 = pd.Timestamp(T_START), pd.Timestamp(T_END)
    win = lambda df: df[(df.t >= t0) & (df.t <= t1)]
    w_df = win(pd.DataFrame({"t": pd.to_datetime(d["Time.2"], errors="coerce"),
                             "T_a": pd.to_numeric(d["Ambient"], errors="coerce"),
                             "V": pd.to_numeric(d["wind speed"], errors="coerce")}).dropna())
    m = win(pd.DataFrame({"t": pd.to_datetime(d["Time"], errors="coerce"),
                          "Normal": d["normal module"], "PDMS": d["PDMS module"]}).dropna())
    hours = 18.0 + (m.t - t0).dt.total_seconds().to_numpy() / 3600.0
    meas = {c: (hours, m[c].to_numpy(float)) for c in ("Normal", "PDMS")}
    w = dict(t=(w_df.t - t0).dt.total_seconds().to_numpy(float), T_a=w_df.T_a.to_numpy() + 273.15,
             V=np.clip(w_df.V.to_numpy(), 0, None))
    return w, w_df, meas


def simulate(w, eps_tab, tab, T0_C):
    """BDF solve, every layer at T0_C initially; returns hours and the TPT temperature, degC."""
    t = w["t"]
    sol = solve_ivp(thermal_equations, (t[0], t[-1]), np.full(5, T0_C + 273.15), method="BDF",
                    t_eval=np.linspace(t[0], t[-1], max(int((t[-1] - t[0]) / 180.0), 24 * 20) + 1),
                    args=(w, eps_tab, tab), rtol=1e-7, atol=1e-6, max_step=300)
    return 18.0 + sol.t / 3600.0, sol.y[4] - 273.15

# ===========================================================================
# 4. PLOTTING
# ===========================================================================

def stats_text(m):
    return (f"MAE   = {m['mae']:.2f} °C\nRMSE  = {m['rmse']:.2f} °C\nMAPE  = {m['mape']:.1f} %\n"
            f"RMSPE = {m['rmspe']:.1f} %\nR²    = {m['r2']:.3f}")


def plot_results(w_df, meas, res, met, dT_meas):
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(8.0, 8.4), sharex=True, gridspec_kw=dict(height_ratios=(2.0, 1.0)))
    ax.plot(18.0 + (w_df.t - w_df.t.iloc[0]).dt.total_seconds() / 3600.0, w_df.T_a, color=COLORS["amb"], lw=1.0, label="Ambient")
    ax.plot(*meas["Normal"], color=COLORS["meas_N"], lw=2.4, alpha=0.45, label="Measured, normal module")
    ax.plot(*meas["PDMS"], color=COLORS["meas_P"], lw=2.4, alpha=0.45, label="Measured, PDMS module")
    for run, ls in (("Normal, grey", ":"), ("Normal, spectral", "-"), ("PDMS, spectral", "-")):
        ax.plot(*res[run], ls, color=COLORS[run], lw=1.5, label=f"Model, {run}")
    for x, run in ((0.40, "Normal, spectral"), (0.98, "PDMS, spectral")):
        ax.text(x, 0.80, f"{run}\n" + stats_text(met[run]), transform=ax.transAxes, fontsize=8, va="top", ha="right",
                multialignment="left", family="monospace", color=COLORS[run],
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#999999", alpha=0.92))
    ax.set_ylabel("Bottom-surface (TPT) temperature [°C]")
    ax.set_title("Fig. S4, windshield night — normal vs PDMS module (V = 0, initial T = measured)")
    ax.grid(True)

    h, dT = res["Normal, spectral"][0], res["Normal, spectral"][1] - res["PDMS, spectral"][1]
    bx.plot(*dT_meas, color="#555555", lw=2.4, alpha=0.45, label="Measured")
    bx.plot(h, dT, color=COLORS["PDMS, spectral"], lw=1.5, label="Model, spectral (same equations, eps differs)")
    bx.text(0.98, 0.08, f"mean: measured {np.mean(dT_meas[1]):.2f} K, model {np.mean(dT):.2f} K",
            transform=bx.transAxes, ha="right", fontsize=8.5)
    bx.set(xlabel="Time [h]", ylabel="T$_{normal}$ − T$_{PDMS}$ [K]", xlim=(18, 24), ylim=(0, None))
    bx.grid(True)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = bx.get_legend_handles_labels()
    fig.legend(h1 + h2, l1 + [f"ΔT {l}" for l in l2], loc="lower center", ncol=2, fontsize=8)
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    fig.savefig(OUT_DIR / "figS4_normal_vs_pdms.png")


def plot_optics(lam_sky, tau0):
    """Optical inputs over 0.28-100 um: (a) front eps, (b) tau/rho, (c) eps_PDMS - eps_glass with 290 K blackbody, (d) sky tau0."""
    lam = np.concatenate([np.arange(0.28, 4.0, 0.005), np.arange(4.0, 17.0, 0.01), np.arange(17.0, LAM_MAX + 1e-9, 0.1)])
    eps, (f, c, _) = read_eps(lam), read_digitized()
    col = {"Normal": "#1F6FB4", "PDMS": "#D62728"}
    fig, ax = plt.subplots(4, 1, figsize=(9, 13), sharex=True, gridspec_kw=dict(height_ratios=(1.6, 1, 1.2, 1)))

    regions = ((0.28, 2.5, "Fig. S1\n1 - τ - ρ", "#F2E6D0"), (2.5, 4.0, "bridge", "#EDEDED"),
               (4.0, c.wavelength_um.iloc[-1], "Fig. 2c", "#DCEBF5"), (c.wavelength_um.iloc[-1], LAM_MAX, "tail", "#EDEDED"))
    for a, tag in zip(ax, "abcd"):
        for x0, x1, _, rc in regions:
            a.axvspan(x0, x1, color=rc, alpha=0.55, lw=0)
        a.axvspan(8, 13, color="#F5C26B", alpha=0.35, lw=0)
        a.grid(True)
        a.text(0.005, 0.93, f"({tag})", transform=a.transAxes, fontweight="bold", va="top")
    for x0, x1, label, _ in regions:
        ax[0].text(np.sqrt(x0 * x1), 1.02, label, ha="center", va="bottom", fontsize=8, multialignment="center")
    ax[0].text(10.2, 0.03, "8–13 µm\nwindow", ha="center", fontsize=8)

    # (a) front emissivity
    for k, label in (("Normal", "Normal module (glass front)"), ("PDMS", "PDMS module (PDMS-glass front)")):
        ax[0].plot(lam, eps[k], color=col[k], lw=1.3, label=label)
    ax[0].set(ylabel="Emissivity / absorptivity ε(λ)", ylim=(0, 1.05))
    ax[0].legend(loc="center left", fontsize=8.5)

    # (b) tau, rho: Fig. S1 up to 2.5 um, then tau + rho = 1 - eps (rho = 1 - eps, tau = 0 from 4 um)
    for k, s1 in (("Normal", "glass"), ("PDMS", "pdms")):
        r_s1 = np.interp(lam, f.wavelength_um, f[f"rho_{s1}"])
        rho = np.where(lam <= f.wavelength_um.iloc[-1], r_s1, np.where(lam >= 4.0, 1 - eps[k], np.minimum(r_s1, 1 - eps[k])))
        ax[1].plot(lam, np.clip(1 - eps[k] - rho, 0, None), color=col[k], lw=1.2, label=f"τ {s1}")
        ax[1].plot(lam, rho, color=col[k], lw=1.2, ls="--", label=f"ρ {s1}")
    ax[1].set(ylabel="τ (solid), ρ (dashed)", ylim=(0, 1.0))
    ax[1].legend(loc="center right", fontsize=8, ncol=2)
    ax[1].text(0.995, 0.97, "≤ 2.5 µm: [L] Fig. S1;  > 2.5 µm: τ + ρ = 1 − ε,  ρ = 1 − ε, τ = 0 from 4 µm",
               transform=ax[1].transAxes, ha="right", va="top", fontsize=8)

    # (c) emissivity difference against the 290 K blackbody
    ax[2].plot(lam, eps["PDMS"] - eps["Normal"], color="k", lw=1.2, label="ε$_{PDMS}$ − ε$_{glass}$")
    ax[2].axhline(0, color="#666", lw=0.8)
    ax[2].set(ylabel="Δε", ylim=(-0.1, 0.45))
    twin, B = ax[2].twinx(), planck(290.0, lam)[0] * lam
    twin.fill_between(lam, B / B.max(), color="#C44E52", alpha=0.18, lw=0, label="290 K blackbody, λB(λ) (norm.)")
    twin.set(ylim=(0, 3), yticks=[])
    (h1, l1), (h2, l2) = ax[2].get_legend_handles_labels(), twin.get_legend_handles_labels()
    ax[2].legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8.5)

    # (d) sky transmittance
    ax[3].plot(lam, np.interp(lam, lam_sky, tau0), color="#7A3E9D", lw=0.8, label=f"τ$_0$(λ), TWC = {TWC} atm-cm")
    ticks = [0.3, 0.5, 1, 2, 4, 8, 13, 20, 50, 100]
    ax[3].set(ylabel="Sky transmittance τ$_0$", ylim=(0, 1.05), xscale="log", xlim=(0.28, LAM_MAX),
              xlabel="Wavelength [µm]", xticks=ticks, xticklabels=[str(t) for t in ticks])
    ax[3].legend(loc="upper right", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "pdms_optics_full_range.png")

# ===========================================================================

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(STYLE)

    w, w_df, meas = read_figS4()
    lam, tau0 = read_sky()
    eps_tab = sky_table(lam, tau0)
    tab = {c: rc_tables(e, lam, tau0) for c, e in read_eps(lam).items()}

    runs = {"Normal, grey": (None, "Normal"), "Normal, spectral": (tab["Normal"], "Normal"), "PDMS, spectral": (tab["PDMS"], "PDMS")}
    res = {r: simulate(w, eps_tab, tb, meas[c][1][0]) for r, (tb, c) in runs.items()}
    met = {r: error_metrics(np.interp(meas[c][0], *res[r]), meas[c][1]) for r, (_, c) in runs.items()}

    dT_meas = (meas["Normal"][0], meas["Normal"][1] - np.interp(meas["Normal"][0], *meas["PDMS"]))
    dT_model = np.interp(dT_meas[0], res["Normal, spectral"][0], res["Normal, spectral"][1] - res["PDMS, spectral"][1])

    table = pd.DataFrame(met).T
    print(f"initial T: normal {meas['Normal'][1][0]:.2f} degC, PDMS {meas['PDMS'][1][0]:.2f} degC")
    print("\nModel vs measured, [L] Fig. S4, 18:00-24:00 (normal runs vs normal module, PDMS run vs PDMS module)")
    print(table.to_string(float_format=lambda v: f"{v:8.3f}"))
    print(f"\nRC effect T_normal - T_PDMS: measured {dT_meas[1].mean():.3f} K, model {dT_model.mean():.3f} K, "
          f"RMSE {np.sqrt(np.mean((dT_model - dT_meas[1]) ** 2)):.3f} K")
    table.to_csv(OUT_DIR / "figS4_pdms_metrics.csv")

    plot_results(w_df, meas, res, met, dT_meas)
    plot_optics(lam, tau0)
    print(f"saved figures and metrics to {OUT_DIR}")
    plt.show()


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
solver.py
1D Euler (Sod shock tube) by Flux Reconstruction (CPR) with Gauss solution points
- FR correction: Radau-based (Huynh/Vincent) derivatives
- Spatial ops: barycentric Lagrange on Gauss-Legendre nodes, derivative matrix
- Time integration: SSP-RK3
- BC: transmissive (non-reflecting)
- Numerical flux: HLLC
- Outputs: multi-time PDFs (ρ, u, p, ε) + combined.pdf in results/
"""

import os
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from numpy.polynomial.legendre import leggauss, Legendre

# ========== Utilities: Barycentric / Lagrange ==========
def barycentric_weights(x):
    x = np.asarray(x, float)
    n = len(x)
    w = np.ones(n)
    for j in range(n):
        w[j] = 1.0 / np.prod(x[j] - np.delete(x, j))
    return w

def lagrange_eval_matrix(y, x, w, atol=1e-14, rtol=1e-12):
    y = np.atleast_1d(np.asarray(y, float))
    x = np.asarray(x, float); w = np.asarray(w, float)
    m, n = len(y), len(x)
    L = np.empty((m, n), dtype=float)
    for p, yp in enumerate(y):
        diff = yp - x
        hit = np.isclose(diff, 0.0, atol=atol, rtol=rtol)
        if np.any(hit):
            row = np.zeros(n, dtype=float)
            row[np.flatnonzero(hit)[0]] = 1.0
            L[p] = row
        else:
            t = w / diff
            L[p] = t / np.sum(t)
    return L

def derivative_matrix(x, w):
    x = np.asarray(x, float); w = np.asarray(w, float)
    n = len(x)
    D = np.zeros((n, n), dtype=float)
    for i in range(n):
        diff = x[i] - x
        for j in range(n):
            if j == i: continue
            D[i, j] = w[j] / (w[i] * diff[j])
        D[i, i] = -np.sum(D[i, :])
    return D

# ========== FR correction functions (Radau-based) ==========
def correction_derivatives_at(nodes, p):
    nodes = np.asarray(nodes, float)
    if p == 0:
        return np.full_like(nodes, -0.5), np.full_like(nodes, 0.5)
    Pp, Pm1 = Legendre.basis(p), Legendre.basis(p-1)
    gLB = ((-1)**p) * 0.5 * (Pp - Pm1)
    gRB = 0.5 * (Pp + Pm1)
    return gLB.deriv()(nodes), gRB.deriv()(nodes)

# ========== Euler helpers ==========
def cons_to_prim(U, gamma):
    rho = U[..., 0]
    m   = U[..., 1]
    E   = U[..., 2]
    rho_safe = np.clip(rho, 1e-14, None)
    u   = m / rho_safe
    p   = (gamma - 1.0) * (E - 0.5 * rho * u * u)
    return rho, u, p

def specific_internal_energy(U, gamma):
    rho, u, p = cons_to_prim(U, gamma)
    return p / (gamma - 1.0) / np.clip(rho, 1e-14, None)

def flux_euler(U, gamma):
    rho, u, p = cons_to_prim(U, gamma)
    m = rho * u
    F1 = m
    F2 = m * u + p
    F3 = (U[..., 2] + p) * u
    return np.stack([F1, F2, F3], axis=-1)

def max_wavespeed(U, gamma):
    rho, u, p = cons_to_prim(U, gamma)
    a = np.sqrt(np.clip(gamma * np.clip(p, 0.0, None) / np.clip(rho, 1e-14, None), 0.0, None))
    return float(np.max(np.abs(u) + a))

def hllc_flux(UL, UR, gamma):
    # left
    rL, uL, pL = cons_to_prim(UL, gamma)
    aL = np.sqrt(np.clip(gamma * np.clip(pL, 0.0, None) / np.clip(rL, 1e-14, None), 0.0, None))
    EL = UL[..., 2]
    FL = flux_euler(UL, gamma)
    # right
    rR, uR, pR = cons_to_prim(UR, gamma)
    aR = np.sqrt(np.clip(gamma * np.clip(pR, 0.0, None) / np.clip(rR, 1e-14, None), 0.0, None))
    ER = UR[..., 2]
    FR = flux_euler(UR, gamma)

    SL = np.minimum(uL - aL, uR - aR)
    SR = np.maximum(uL + aL, uR + aR)

    num = (pR - pL) + rL*uL*(SL - uL) - rR*uR*(SR - uR)
    den = rL*(SL - uL) - rR*(SR - uR)
    den = np.where(np.abs(den) < 1e-14, 1e-14, den)
    SM  = num / den

    pStarL = pL + rL*(SL - uL)*(SM - uL)
    pStarR = pR + rR*(SR - uR)*(SM - uR)

    rStarL = rL * (SL - uL) / (SL - SM)
    mStarL = rStarL * SM
    EStarL = ((SL - uL) * EL - pL*uL + pStarL*SM) / (SL - SM)

    rStarR = rR * (SR - uR) / (SR - SM)
    mStarR = rStarR * SM
    EStarR = ((SR - uR) * ER - pR*uR + pStarR*SM) / (SR - SM)

    UStarL = np.stack([rStarL, mStarL, EStarL], axis=-1)
    UStarR = np.stack([rStarR, mStarR, EStarR], axis=-1)

    F = np.empty_like(FL)
    mask_L   = (0.0 <= SL)
    mask_ML  = (SL <= 0.0) & (0.0 <= SM)
    mask_MR  = (SM <= 0.0) & (0.0 <= SR)
    mask_R   = (SR <= 0.0)

    F[mask_L]  = FL[mask_L]
    F[mask_R]  = FR[mask_R]
    F[mask_ML] = FL[mask_ML] + SL[mask_ML, None] * (UStarL[mask_ML] - UL[mask_ML])
    F[mask_MR] = FR[mask_MR] + SR[mask_MR, None] * (UStarR[mask_MR] - UR[mask_MR])
    return F

# ========== FR RHS for Euler ==========
def rhs_FR_euler(U, gamma, dx, D, dglb, dgrb, Lm1, Lp1):
    K, Np, _ = U.shape
    F = flux_euler(U, gamma)                          # (K,Np,3)
    dF_dxi = np.einsum('ij,kjm->kim', D, F)           # (K,Np,3)

    UL_int = np.einsum('kjm,j->km', U, Lm1)           # (K,3)
    UR_int = np.einsum('kjm,j->km', U, Lp1)           # (K,3)

    FL_int = flux_euler(UL_int, gamma)                # (K,3)
    FR_int = flux_euler(UR_int, gamma)                # (K,3)

    # Interfaces and states:
    # - Right face of cell k is the interface (k | k+1).
    # - UL_R[k] = left state at the right face (taken from cell k, right trace).
    # - UR_R[k] = right state at the right face (taken from cell k+1, left trace).
    # - Transmissive BC at the domain ends: use the interior trace of the boundary cell.

    # Right faces (k | k+1)
    UL_R = UR_int.copy()        # UL_R[k] := right trace of cell k  (left state at right face k|k+1)
    UR_R = np.empty_like(UL_R)  # UR_R[k] := right state at right face k|k+1
    UR_R[:-1] = UL_int[1:]      # for k = 0..K-2: UR_R[k] = left trace of cell k+1
    UR_R[-1]  = UR_int[-1]      # last face (K-1 | outside): transmissive → use right trace of last cell
    Fstar_R = hllc_flux(UL_R, UR_R, gamma)

    # Left faces (k-1 | k)
    UR_L = UL_int.copy()        # UR_L[k] := left trace of cell k   (right state at left face k-1|k)
    UL_L = np.empty_like(UR_L)  # UL_L[k] := left state at left face k-1|k
    UL_L[1:] = UR_int[:-1]      # for k = 1..K-1: UL_L[k] = right trace of cell k-1
    UL_L[0]  = UL_int[0]        # first face (outside | 0): transmissive → use left trace of first cell
    Fstar_L = hllc_flux(UL_L, UR_L, gamma)

    # For Periodic Boundary condition 
    # UR_R[-1] = UL_int[0]        # periodic: right face of last cell uses left trace of cell 0
    # UL_L[0]  = UR_int[-1]       # periodic: left face of cell 0 uses right trace of last cell
    
    corr_R = (Fstar_R - FR_int)[:, None, :] * dgrb[None, :, None]
    corr_L = (Fstar_L - FL_int)[:, None, :] * dglb[None, :, None]
    dUdt   = -(2.0 / dx) * (dF_dxi + corr_R + corr_L)
    return dUdt

# ========== Exact Riemann (Sod) ==========
def sod_exact_solution(x, t, gamma=1.4, x0=0.5,
                       left=(1.0, 0.0, 1.0), right=(0.125, 0.0, 0.1)):
    """
    Exact density/velocity/pressure for Sod shock tube at time t (Toro).
    Returns (rho, u, p) with same shape as x.
    """
    x = np.asarray(x, float)
    rhoL, uL, pL = left
    rhoR, uR, pR = right
    aL = np.sqrt(gamma * pL / rhoL)
    aR = np.sqrt(gamma * pR / rhoR)

    if t <= 0.0:
        rho = np.where(x < x0, rhoL, rhoR)
        u   = np.where(x < x0, uL,   uR)
        p   = np.where(x < x0, pL,   pR)
        return rho, u, p

    # f functions (shock/rarefaction)
    AL = 2.0 / ((gamma + 1.0) * rhoL); BL = (gamma - 1.0)/(gamma + 1.0) * pL
    AR = 2.0 / ((gamma + 1.0) * rhoR); BR = (gamma - 1.0)/(gamma + 1.0) * pR
    g1 = (gamma - 1.0) / (2.0 * gamma)
    g2 = (gamma + 1.0) / (2.0 * gamma)

    def f_branch(p, pK, aK, rhoK, A, B):
        rare = p <= pK
        fr  = (2.0 * aK / (gamma - 1.0)) * ((p / pK)**g1 - 1.0)
        dfr = (1.0 / (rhoK * aK)) * (p / pK)**(-g2)
        fs  = (p - pK) * np.sqrt(A / (p + B))
        dfs = np.sqrt(A / (p + B)) * (1.0 - 0.5*(p - pK)/(p + B))
        return np.where(rare, fr, fs), np.where(rare, dfr, dfs)

    def f_total(p):
        fL, dL = f_branch(p, pL, aL, rhoL, AL, BL)
        fR, dR = f_branch(p, pR, aR, rhoR, AR, BR)
        return fL + fR + (uR - uL), dL + dR

    # Newton for p*
    p = max(1e-8, 0.5*(pL + pR) - 0.125*(uR - uL)*(rhoL + rhoR)*(aL + aR))
    plo, phi = 1e-8, max(pL, pR)*100 + 1.0
    for _ in range(60):
        F, dF = f_total(p)
        dF = max(dF, 1e-14)
        p_new = p - F/dF
        if p_new < plo or p_new > phi:
            p_new = 0.5*(p + (plo if F > 0 else phi))
        if abs(p_new - p) < 1e-12 * (1 + p):
            p = p_new; break
        p = p_new
    p_star = max(p, 1e-10)
    fL, _ = f_branch(p_star, pL, aL, rhoL, AL, BL)
    fR, _ = f_branch(p_star, pR, aR, rhoR, AR, BR)
    u_star = 0.5*(uL + uR) + 0.5*(fR - fL)

    xi = (x - x0) / t

    # Left side
    if p_star > pL:  # left shock
        SL = uL - aL * np.sqrt((gamma+1.0)/(2.0*gamma) * (p_star/pL) + (gamma-1.0)/(2.0*gamma))
        rho_star_L = rhoL * ((p_star/pL + (gamma-1.0)/(gamma+1.0)) /
                             ((gamma-1.0)/(gamma+1.0) * p_star/pL + 1.0))
        uL_star, pL_star = u_star, p_star
    else:            # left rarefaction
        a_star_L = aL * (p_star/pL)**g1
        SHL = uL - aL
        STL = u_star - a_star_L
        rho_star_L = rhoL * (p_star/pL)**(1.0/gamma)
        uL_star, pL_star = u_star, p_star

    # Right side
    if p_star > pR:  # right shock
        SR = uR + aR * np.sqrt((gamma+1.0)/(2.0*gamma) * (p_star/pR) + (gamma-1.0)/(2.0*gamma))
        rho_star_R = rhoR * ((p_star/pR + (gamma-1.0)/(gamma+1.0)) /
                             ((gamma-1.0)/(gamma+1.0) * p_star/pR + 1.0))
        uR_star, pR_star = u_star, p_star
    else:            # right rarefaction
        a_star_R = aR * (p_star/pR)**g1
        SHR = uR + aR
        STR = u_star + a_star_R
        rho_star_R = rhoR * (p_star/pR)**(1.0/gamma)
        uR_star, pR_star = u_star, p_star

    # Allocate
    rho = np.empty_like(x); u = np.empty_like(x); p = np.empty_like(x)

    # Sample left branch
    if p_star > pL:
        mask_L = xi < SL
        mask_ML = (xi >= SL) & (xi < u_star)
        rho[mask_L], u[mask_L], p[mask_L] = rhoL, uL, pL
        rho[mask_ML], u[mask_ML], p[mask_ML] = rho_star_L, uL_star, pL_star
    else:
        SHL = uL - aL
        STL = u_star - a_star_L
        mask1 = xi < SHL
        mask2 = (xi >= SHL) & (xi <= STL)
        mask3 = (xi > STL) & (xi < u_star)
        rho[mask1], u[mask1], p[mask1] = rhoL, uL, pL
        if np.any(mask2):
            xi2 = xi[mask2]
            a = ((uL + 2*aL/(gamma-1.0) - xi2) * (gamma-1.0)) / (gamma+1.0)
            u_mid = uL + 2.0/(gamma-1.0) * (aL - a)
            p_mid = pL * (a/aL)**(2.0*gamma/(gamma-1.0))
            rho_mid = rhoL * (a/aL)**(2.0/(gamma-1.0))
            rho[mask2], u[mask2], p[mask2] = rho_mid, u_mid, p_mid
        rho[mask3], u[mask3], p[mask3] = rho_star_L, uL_star, pL_star

    # Sample right branch
    if p_star > pR:
        mask_MR = (xi >= u_star) & (xi <= SR)
        mask_Rg = xi > SR
        rho[mask_MR], u[mask_MR], p[mask_MR] = rho_star_R, uR_star, pR_star
        rho[mask_Rg], u[mask_Rg], p[mask_Rg] = rhoR, uR, pR
    else:
        SHR = uR + aR
        STR = u_star + a_star_R
        mask1 = (xi > u_star) & (xi < STR)
        mask2 = (xi >= STR) & (xi <= SHR)
        mask3 = xi > SHR
        rho[mask1], u[mask1], p[mask1] = rho_star_R, uR_star, pR_star
        if np.any(mask2):
            xi2 = xi[mask2]
            a = ((xi2 - uR + 2*aR/(gamma-1.0)) * (gamma-1.0)) / (gamma+1.0)
            u_mid = uR - 2.0/(gamma-1.0) * (aR - a)
            p_mid = pR * (a/aR)**(2.0*gamma/(gamma-1.0))
            rho_mid = rhoR * (a/aR)**(2.0/(gamma-1.0))
            rho[mask2], u[mask2], p[mask2] = rho_mid, u_mid, p_mid
        rho[mask3], u[mask3], p[mask3] = rhoR, uR, pR

    return rho, u, p

# ========== Class shells for API compatibility ==========
class InitialCondition:
    def __init__(self):
        pass
    def sod(self, x, gamma, x0, left_state, right_state):
        rhoL, uL, pL = left_state
        rhoR, uR, pR = right_state
        left = x < x0
        rho = np.where(left, rhoL, rhoR)
        u   = np.where(left, uL,   uR)
        p   = np.where(left, pL,   pR)
        E   = p/(gamma-1.0) + 0.5*rho*u*u
        return np.stack([rho, rho*u, E], axis=-1)


# ========== Main Solver ==========
class SodShockTubeSolver:
    def __init__(self, config):
        # Read config with fallbacks
        getf = config.get
        self.gamma   = float(getf("gamma", fallback="1.4"))
        self.length  = float(getf("length", fallback="1.0"))
        self.K       = int(getf("K", fallback="200"))
        self.p       = int(getf("p", fallback="2"))
        self.CFL     = float(getf("CFL", fallback="0.4"))
        self.Tfinal  = float(getf("Tfinal", fallback="0.2"))
        self.n_outputs = int(getf("n_outputs", fallback="5"))  # number of snapshots (incl. final)
        self.results_dir = getf("results_dir", fallback="results")
        os.makedirs(self.results_dir, exist_ok=True)

        # Initial discontinuity and states
        self.x0 = float(getf("x0", fallback="0.5"))
        self.left_state  = (
            float(getf("left_rho", fallback="1.0")),
            float(getf("left_u",   fallback="0.0")),
            float(getf("left_p",   fallback="1.0")),
        )
        self.right_state = (
            float(getf("right_rho", fallback="0.125")),
            float(getf("right_u",   fallback="0.0")),
            float(getf("right_p",   fallback="0.1")),
        )

        # Reference operators on [-1,1]
        self.Np = self.p + 1
        self.xi, _ = leggauss(self.Np)       # Gauss solution points
        self.bw   = barycentric_weights(self.xi)
        self.D    = derivative_matrix(self.xi, self.bw)
        self.Lm1  = lagrange_eval_matrix([-1.0], self.xi, self.bw)[0]
        self.Lp1  = lagrange_eval_matrix([+1.0], self.xi, self.bw)[0]
        self.dglb, self.dgrb = correction_derivatives_at(self.xi, self.p)

        # Mesh
        x_edges = np.linspace(0.0, self.length, self.K + 1)
        self.dx = x_edges[1] - x_edges[0]
        map_ref = lambda xl: xl + 0.5*(self.xi + 1.0)*self.dx
        self.x_nodes = np.vstack([map_ref(x_edges[k]) for k in range(self.K)])  # (K,Np)

    def run(self, init_cond: InitialCondition, log_file):
        gamma = self.gamma
        # Initial condition on nodes
        U = init_cond.sod(self.x_nodes, gamma, self.x0, self.left_state, self.right_state)
        U0 = U.copy()

        # Output schedule (include t=0 and t=Tfinal)
        nout = max(2, self.n_outputs)
        snap_times = np.linspace(0.0, self.Tfinal, nout)

        # Combined PDF
        combined_path = os.path.join(self.results_dir, "combined.pdf")
        pdf_combined = PdfPages(combined_path)

        # Helper to output plots
        def output_snapshot(U_now, t_now, tag):
            x_plot = self.x_nodes.reshape(-1)
            rho    = U_now[...,0].reshape(-1)
            u      = cons_to_prim(U_now, gamma)[1].reshape(-1)
            p      = cons_to_prim(U_now, gamma)[2].reshape(-1)
            eps    = specific_internal_energy(U_now, gamma).reshape(-1)

            idx = np.argsort(x_plot)
            x_plot = x_plot[idx]
            rho, u, p, eps = rho[idx], u[idx], p[idx], eps[idx]

            # exact solution
            rho_ex, u_ex, p_ex = sod_exact_solution(x_plot, t_now, gamma=gamma, x0=self.x0,
                                                    left=self.left_state, right=self.right_state)
            eps_ex = p_ex / ((gamma - 1.0) * np.ones_like(rho_ex)) / np.maximum(rho_ex, 1e-14)

            fig, axes = plt.subplots(2, 2, figsize=(9, 6))
            ax = axes.ravel()

            def one(ax, y_num, y_ex, title, ylabel):
                ax.plot(x_plot, y_ex, '-', lw=2, color='blue', label='Exact')
                ax.plot(x_plot, y_num, '-', lw=2, color='red', label='FR')
                ax.set_title(title)
                ax.set_xlabel('x'); ax.set_ylabel(ylabel)
                ax.grid(True); ax.legend()

            one(ax[0], rho, rho_ex, f"Density ρ (t={t_now:.4f})", "ρ")
            one(ax[1], u,   u_ex,   f"Velocity u (t={t_now:.4f})", "u")
            one(ax[2], p,   p_ex,   f"Pressure p (t={t_now:.4f})", "p")
            one(ax[3], eps, eps_ex, f"Internal energy ε (t={t_now:.4f})", "ε")

            fig.tight_layout()

            # Save individual and append to combined
            fname = os.path.join(self.results_dir, f"solution_{tag}.pdf")
            fig.savefig(fname)
            pdf_combined.savefig(fig)
            plt.close(fig)
            return fname

        # First snapshot at t=0
        output_snapshot(U0, 0.0, "t0000")

        # Time integration (SSP-RK3) with adaptive dt
        t = 0.0
        next_idx = 1  # next snap index in snap_times
        while t < self.Tfinal - 1e-14:
            smax = max_wavespeed(U, gamma)
            dt = self.CFL * self.dx / ((2*self.p + 1) * max(smax, 1e-14))
            # adjust to not overrun final time or next snapshot time
            t_target = min(self.Tfinal, snap_times[next_idx]) if next_idx < len(snap_times) else self.Tfinal
            dt = min(dt, t_target - t)

            k1 = rhs_FR_euler(U, gamma, self.dx, self.D, self.dglb, self.dgrb, self.Lm1, self.Lp1)
            U1 = U + dt * k1
            k2 = rhs_FR_euler(U1, gamma, self.dx, self.D, self.dglb, self.dgrb, self.Lm1, self.Lp1)
            U2 = 0.75 * U + 0.25 * (U1 + dt * k2)
            k3 = rhs_FR_euler(U2, gamma, self.dx, self.D, self.dglb, self.dgrb, self.Lm1, self.Lp1)
            U  = (1.0/3.0) * U + (2.0/3.0) * (U2 + dt * k3)
            t += dt

            # Output if reached snapshot time
            while next_idx < len(snap_times) and t >= snap_times[next_idx] - 1e-12:
                tag = f"t{int(round(snap_times[next_idx]*10000)):04d}"
                output_snapshot(U, snap_times[next_idx], tag)
                next_idx += 1

        # Finalize combined PDF
        pdf_combined.close()
        # Log
        log_path = os.path.join(self.results_dir, "simulation.log")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"Finished at t={t:.6f}\n")
            f.write(f"Combined PDF: {os.path.abspath(combined_path)}\n")

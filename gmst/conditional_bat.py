"""Joint Gaussian AR BAT: stationary gap likelihood and state-dependent noise."""
from dataclasses import dataclass, replace

import numpy as np

from gmst import bat
from gmst.conditional_ar import ObservationGrid, SparseDesign
from gmst.contracts import FloatArray, IntArray, Model, Panel, Prediction
from gmst.features import internal_split, role_idx


@dataclass(frozen=True, slots=True)
class ConditionalConfig:
    n_iter: int = 2000
    burn: int = 1000
    seed: int = 0
    conditional: bool = True
    half_life: float = 30.
    fixed_attention: bool = True
    ref: str = "op"  # reference day rule: "op" (B0′ op/daytype) or "kind" (B0_kind operating type)
    # noise state g: "op" (kind>0, 2 states), "kind" (operating kind, 4 states),
    # "slot" (σ_η per slot_states, 4 states; ρ and σ_u stay on "op")
    noise_state: str = "op"
    innovation: str = "gaussian"  # "t": η ~ N(0, σ²/λ), λ ~ Gamma(ν/2, rate ν/2) — Student-t with ν = nu
    nu: float = 5.

    def __post_init__(self) -> None:
        if not 0 <= self.burn < self.n_iter or self.half_life <= 0 or self.ref not in ("op", "kind"):
            message = "Require 0 <= burn < n_iter, positive half_life and ref in {'op', 'kind'}"
            raise ValueError(message)
        if self.noise_state not in ("op", "kind", "slot"):
            raise ValueError("Require noise_state in {'op', 'kind', 'slot'}")
        if not self.conditional and self.noise_state != "op":
            raise ValueError("noise_state must be 'op' when conditional=False")
        if self.innovation not in ("gaussian", "t") or not self.nu > 2:
            raise ValueError("Require innovation in {'gaussian', 't'} and nu > 2")


@dataclass(frozen=True, slots=True)
class ConditionalState:
    mean: bat.BATMeanState
    sigma_u: FloatArray
    sigma_eta: FloatArray
    rho: FloatArray
    acceptance: FloatArray
    C: FloatArray
    inner: "ConditionalState | None" = None
    noise_state: str = "op"
    nu: float | None = None  # Student-t innovation degrees of freedom; None = Gaussian


def t_precision_weights(z: FloatArray, omega: FloatArray, nu: float, rng: np.random.Generator) -> FloatArray:
    """λ | z ~ Gamma((ν+ω)/2, rate (ν+ω·z²)/2) for standardized innovations z under the ω-weighted likelihood
    (normal–gamma scale mixture)."""
    return rng.gamma((nu + omega) / 2, 2 / (nu + omega * z * z))


def slot_states(q: FloatArray) -> IntArray:
    """(D, 24) hourly plan → (D, 96) innovation state: 0 non-operating day, 1 producing, 2 not producing,
    3 transition (the slot's on/off flag changes within ±4 slots of the same day)."""
    on = np.repeat(np.nan_to_num(q) > 0, 4, axis=-1)
    window = np.lib.stride_tricks.sliding_window_view(np.pad(on, ((0, 0), (4, 4)), mode="edge"), 9, axis=1)
    state = np.where(window.max(axis=-1) != window.min(axis=-1), 3, np.where(on, 1, 2))
    return np.where(on.any(axis=1, keepdims=True), state, 0)


def fit_conditional(panel: Panel, train: IntArray, C: FloatArray,
                    config: ConditionalConfig) -> ConditionalState:
    days = train[np.isfinite(panel["Y"][train]).any(axis=1)]
    scale = float(np.nanstd(panel["Y"][days]))
    q_scale = float(np.nanmax(panel["X"]["생산량"][days])) or 1.
    Y = panel["Y"][days] / scale
    grid = ObservationGrid.from_targets(Y)
    dn, tn, prev = grid.day, grid.slot, grid.previous
    y = Y[dn, tn]
    kind = bat.kinds(panel, days)
    if config.noise_state == "kind":
        group, groups = kind, 4
    elif config.conditional:
        group, groups = (kind > 0).astype(np.int64), 2
    else:
        group, groups = np.zeros(len(days), np.int64), 1
    S = slot_states(panel["X"]["생산량"][days, ::4]) if config.noise_state == "slot" else None
    noise = group[dn] if S is None else S[dn, tn]  # σ_η state per observation
    if S is not None:  # span t−s+1..t of each pair inside one run of equal states
        run = np.cumsum(np.diff(S, axis=1, prepend=S[:, :1]) != 0, axis=1)
        homogeneous = run[dn, tn] == run[dn, np.where(grid.gap > 0, tn[prev] + 1, tn)]
    means = np.stack([np.nanmean(Y[kind == k], axis=0) if (kind == k).any()
                      else np.nanmean(Y, axis=0) for k in range(4)])
    means = np.where(np.isfinite(means), means, np.nanmean(Y))
    ref = bat._ref(panel, days, kind, means, scale, config.ref)[dn, tn]
    X = bat.channels(panel["X"]["생산량"][days, ::4], q_scale)
    rows = [np.flatnonzero(kind[dn] == k) for k in range(4)]
    local_prev = [np.searchsorted(m, prev[m]) for m in rows]
    omega_d = 2. ** (-(days.max() - days) / config.half_life)
    omega = omega_d[dn]
    rng = np.random.default_rng(config.seed)
    theta = np.tile(bat.PRIOR_MU, (4, 1))
    proposal_step = np.r_[bat.STEP[:3], .5, .5]
    active = slice(3, 5) if config.fixed_attention else slice(0, 5)
    n_proposed = 2 if config.fixed_attention else 5
    coef = np.column_stack([means, np.zeros(4)])
    tau = np.ones(4)
    rho = np.full(groups, .5)
    eta_var = np.full(groups if S is None else 4, .05)
    u_var = np.full(groups, .01)
    u = np.zeros(len(days))
    lam = np.ones(len(y))  # latent per-observation precision scale (Student-t); stays 1 for Gaussian
    rw = bat._rw2() + 1e-4 * np.eye(96)
    fixed_channels = ((X[:, dn] * bat.attention(bat.PRIOR_MU)[tn]).sum(axis=-1).T
                      if config.fixed_attention else None)

    def attended(th: FloatArray, m: IntArray) -> FloatArray:
        if fixed_channels is not None:
            return fixed_channels[m] @ np.exp(th[3:])
        return (X[:, dn[m]] * bat.attention(th)[tn[m]]).sum(axis=-1).T @ np.exp(th[3:])

    trans = np.zeros(len(y))
    for k in range(1, 4):
        trans[rows[k]] = attended(theta[k], rows[k])
    count = config.n_iter - config.burn
    idle_draws = np.empty((count, 4, 96))
    gamma_draws = np.empty((count, 4))
    theta_draws = np.empty((count, 4, 5))
    u_draws, eta_draws, rho_draws = (np.empty((count, len(v))) for v in (u_var, eta_var, rho))
    accepted, attempted = np.zeros(4 + groups), np.zeros(4 + groups)
    for it in range(config.n_iter):
        whitening = grid.weights(rho[group])
        if S is not None:
            whitening = whitening[0], np.sqrt(eta_var[noise] / grid.transition_variance(rho[group], eta_var[S]))
        phi, inv_sd = whitening
        precision = omega * lam / eta_var[noise]  # weighted likelihood: precision ω·λ/σ²
        target = grid.whiten(y - trans - u[dn], whitening)
        for k, m in enumerate(rows):
            columns = np.column_stack([tn[m], tn[prev[m]], np.full(len(m), 96)])
            values = np.column_stack([inv_sd[m], -phi[m] * inv_sd[m],
                                      (ref[m] - phi[m] * ref[prev[m]]) * inv_sd[m]])
            design = SparseDesign(columns, values)
            A, rhs = design.normal_equations(precision[m], target[m])
            A[:96, :96] += rw / tau[k]
            A[96, 96] += 1 / 25.
            L = np.linalg.cholesky(A)
            beta = np.linalg.solve(L.T, np.linalg.solve(L, rhs))
            if k > 0 and len(m):
                proposal = theta[k].copy()
                proposal[active] += proposal_step[active] * rng.standard_normal(n_proposed)
                proposed_trans = attended(proposal, m)
                delta = proposed_trans - trans[m]
                new_target = target[m] - (delta - phi[m] * delta[local_prev[k]]) * inv_sd[m]
                new_rhs = design.rhs(precision[m], new_target)
                new_beta = np.linalg.solve(L.T, np.linalg.solve(L, new_rhs))
                ratio = (.5 * (new_rhs @ new_beta - rhs @ beta)
                         -.5 * precision[m] @ (new_target ** 2 - target[m] ** 2)
                         -.5 * np.sum(((proposal - bat.PRIOR_MU) / bat.PRIOR_SD) ** 2
                                      - ((theta[k] - bat.PRIOR_MU) / bat.PRIOR_SD) ** 2))
                attempted[k] += 1
                if np.log(rng.uniform()) < ratio:
                    theta[k], trans[m], beta = proposal, proposed_trans, new_beta
                    accepted[k] += 1
            elif k > 0:
                theta[k, active] = rng.normal(bat.PRIOR_MU[active], bat.PRIOR_SD[active])
            coef[k] = beta + np.linalg.solve(L.T, rng.standard_normal(97))
        tau = 1 / rng.gamma(49., 1 / (.01 + .5 * np.einsum("kt,ts,ks->k", coef[:, :96], rw, coef[:, :96])))
        base = coef[kind[dn], tn] + coef[kind[dn], 96] * ref
        h = (1 - phi) * inv_sd
        z = grid.whiten(y - base - trans, whitening)
        u_precision = 1 / u_var[group] + np.bincount(dn, precision * h * h, minlength=len(days))
        u_mean = np.bincount(dn, precision * h * z, minlength=len(days)) / u_precision
        u = u_mean + rng.standard_normal(len(days)) / np.sqrt(u_precision)
        residual = y - base - trans - u[dn]
        if config.innovation == "t":
            lam = t_precision_weights(grid.whiten(residual, whitening) / np.sqrt(eta_var[noise]), omega, config.nu, rng)
        for g in range(groups):
            m = group[dn] == g
            proposed_rho = rho[g] + .04 * rng.standard_normal()
            attempted[4 + g] += 1
            if 0 <= proposed_rho < .98:
                old_phi, old_inv = phi[m], inv_sd[m]
                new_phi = np.where(grid.gap[m] > 0, proposed_rho ** grid.gap[m], 0.)
                new_inv = (np.sqrt((1 - proposed_rho ** 2) / (1 - new_phi ** 2)) if S is None else np.sqrt(
                    eta_var[noise[m]] / grid.transition_variance(np.full(len(days), proposed_rho), eta_var[S])[m]))
                old_z = (residual[m] - old_phi * residual[prev[m]]) * old_inv
                new_z = (residual[m] - new_phi * residual[prev[m]]) * new_inv
                ratio = omega[m] @ (np.log(new_inv / old_inv)
                                    - .5 * lam[m] * (new_z ** 2 - old_z ** 2) / eta_var[noise[m]])
                if np.log(rng.uniform()) < ratio:
                    rho[g], phi[m], inv_sd[m] = proposed_rho, new_phi, new_inv
                    accepted[4 + g] += 1
            if S is None:
                z_g = (residual[m] - phi[m] * residual[prev[m]]) * inv_sd[m]
                eta_var[g] = 1 / rng.gamma(2 + omega[m].sum() / 2,
                                          1 / (.01 + .5 * (omega[m] * lam[m]) @ z_g ** 2))
            ud = u[group == g]
            u_var[g] = 1 / rng.gamma(2 + len(ud) / 2, 1 / (.01 + .5 * ud @ ud))
        if S is not None:
            # Pairs whose span t−s+1..t stays in one state have variance σ²_s·c (conjugate IG update);
            # the rest mix states, so the conjugate draw is an independence proposal corrected by their likelihood.
            e = residual - phi * residual[prev]
            het = ~homogeneous
            for n in range(4):
                m = homogeneous & (noise == n)
                z_n = e[m] * inv_sd[m]
                proposal = eta_var.copy()
                proposal[n] = 1 / rng.gamma(2 + omega[m].sum() / 2, 1 / (.01 + .5 * (omega[m] * lam[m]) @ z_n ** 2))
                if het.any():
                    old_v, new_v = (grid.transition_variance(rho[group], v[S])[het] for v in (eta_var, proposal))
                    ratio = omega[het] @ (-.5 * np.log(new_v / old_v)
                                          - .5 * lam[het] * e[het] ** 2 * (1 / new_v - 1 / old_v))
                    if np.log(rng.uniform()) >= ratio:
                        continue
                eta_var = proposal
        if it >= config.burn:
            s = it - config.burn
            idle_draws[s], gamma_draws[s], theta_draws[s] = coef[:, :96], coef[:, 96], theta
            u_draws[s], eta_draws[s], rho_draws[s] = np.sqrt(u_var), np.sqrt(eta_var), rho
    weights = np.exp(theta_draws[:, :, 3:]).transpose(0, 2, 1)
    weights[:, :, 0] = 0
    mean = bat.BATMeanState(scale=scale, q_scale=q_scale, kind_mean=means, idle=idle_draws,
                            gam=gamma_draws, w=weights, theta=theta_draws, ref_rule=config.ref)
    acceptance = np.divide(accepted, attempted, out=np.full_like(accepted, np.nan), where=attempted > 0)
    return ConditionalState(mean, u_draws, eta_draws, rho_draws, acceptance, C, noise_state=config.noise_state,
                            nu=config.nu if config.innovation == "t" else None)


def raw_paths(state: ConditionalState, panel: Panel, d: int) -> FloatArray:
    q = np.nan_to_num(panel["X"]["생산량"][d, ::4])
    mu = bat.plan_curves(state.mean, panel, d, q)
    kind_g = bat.kind_of(q)
    g = kind_g if state.noise_state == "kind" else int(kind_g > 0) if state.rho.shape[1] == 2 else 0
    rng = np.random.default_rng(int(panel["dates"][d].strftime("%Y%m%d")))
    count = len(mu)
    r = state.rho[:, g]
    sd = state.sigma_eta[:, slot_states(q[None])[0]] if state.noise_state == "slot" else state.sigma_eta[:, g, None]
    noise = rng.standard_normal((count, 96)) * sd
    if state.nu is not None:  # Student-t innovations: fresh λ per slot; slot 0 keeps the Gaussian stationary scale
        noise /= np.sqrt(rng.gamma(state.nu / 2, 2 / state.nu, (count, 96)))
    noise[:, 0] /= np.sqrt(1 - r * r)
    for t in range(1, 96):
        noise[:, t] += r * noise[:, t - 1]
    noise += state.sigma_u[:, g, None] * rng.standard_normal((count, 1))
    return mu + noise * state.mean["scale"]


def predict_conditional(state: ConditionalState, panel: Panel, d: int) -> Prediction:
    return bat.from_paths(np.maximum(raw_paths(state, panel, d), 0.), state.C)


def conditional_model(config: ConditionalConfig) -> Model[ConditionalState]:
    def fit(panel: Panel, fold: str, C: FloatArray) -> ConditionalState:
        tr = role_idx(panel, fold, "train")
        cal = internal_split(panel, fold)[2]
        state = fit_conditional(panel, tr, C, config)
        inner = fit_conditional(panel, tr[tr < cal[0]] if len(cal) else tr, C, config)
        return replace(state, inner=inner)

    name = "BAT_conditional_gaussian" if config.conditional else "BAT_pooled_gaussian"
    name += "_fixed_attention" if config.fixed_attention else ""
    name += "" if config.noise_state == "op" else "_noise" + config.noise_state
    name += "_tinnov" if config.innovation == "t" else ""
    return {"name": name + ("" if config.ref == "op" else f"_{config.ref}ref"),
            "fit": fit, "predict": predict_conditional, "inner_state": lambda state: state.inner}

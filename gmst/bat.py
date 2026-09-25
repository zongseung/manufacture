"""BAT: Bayesian attention transfer — production plan → day curve, fitted by Gibbs + Metropolis.

y_{d,t} = idle_{k,t} + γ_k·ref_{d,t} + Σ_c w_{c,k}·Σ_h α_{t,h}(θ_k) x_c(q_{d,h}) + ε,  ε ~ Laplace
(predictive paths add a day-common N(0, σ_u²) shift plus within-day AR(1) noise with Laplace innovations,
 estimated from residuals: lag-1 residual autocorrelation is ≈0.96, so independent slot noise inflates the daily max)
x_on(q) = 1[q>0], x_log(q) = log1p(q)/log1p(q_max): power follows whether lines run far more than volume.
α(θ) = sparsemax_h(−|Δ_{t,h} − c|^β / σ)  (generalised-Gaussian locality, Welty et al. 2009 BDLM prior idea;
posterior over attention parameters as in Fan et al. 2020). k = operating type from the plan (0/1–12/13–19/20+ h).
"""
from collections.abc import Callable
from typing import Final, NotRequired, TypedDict

import numpy as np
import torch

from gmst.baselines import b0p_ref
from gmst.contracts import FloatArray, IntArray, Model, Panel, Prediction
from gmst.evaluate import TAUS
from gmst.features import internal_split, role_idx

KINDS: Final = 4
T_IDX: Final = np.arange(96)
# θ = (log σ, log β, c [hours], log w_on, log w_log): w > 0 keeps ŷ non-decreasing in the plan
PRIOR_MU: Final = np.array([np.log(1.5), np.log(2.0), 0.0, np.log(0.3), np.log(0.3)])
PRIOR_SD: Final = np.array([0.5, 0.5, 1.0, 1.5, 1.5])
# ponytail: 고정 랜덤워크 보폭, 수용률이 0.15–0.5 밖이면 burn-in 적응 보폭
STEP: Final = np.array([0.15, 0.15, 0.3, 0.1, 0.1])


class BATState(TypedDict):
    scale: float
    q_scale: float
    kind_mean: FloatArray  # (4, 96) scaled, fallback reference curve
    idle: FloatArray  # (S, 4, 96)
    gam: FloatArray  # (S, 4)
    w: FloatArray  # (S, 2, 4) channels on/log × kind
    theta: FloatArray  # (S, 4, 5)
    sigma_u: FloatArray  # (S,)
    lam: FloatArray  # (S,)
    rho: FloatArray  # (S,) within-day AR(1) coefficient of residuals
    b_eta: FloatArray  # (S,) Laplace scale of AR(1) innovations (scaled units)
    C: FloatArray
    inner_state: NotRequired["BATState"]


def sparsemax(z: FloatArray) -> FloatArray:
    """Row-wise sparsemax (Martins & Astudillo 2016): Euclidean projection onto the simplex."""
    zs = -np.sort(-z, axis=-1)
    k = np.arange(1, z.shape[-1] + 1)
    css = np.cumsum(zs, axis=-1)
    support = 1 + k * zs > css
    kz = support.sum(-1, keepdims=True)
    tau = (np.take_along_axis(css, kz - 1, -1) - 1) / kz
    return np.maximum(z - tau, 0.0)


def attention(theta: FloatArray) -> FloatArray:
    """(96, 24) weights: slot t attends to hours h near t, width σ, shape β, lag c (hours)."""
    sigma, beta, c = np.exp(theta[0]), np.exp(theta[1]), theta[2]
    gap = (T_IDX[:, None] - 4 * np.arange(24)[None] - 1.5) / 4 - c
    return sparsemax(-np.abs(gap) ** beta / sigma)


def groups_from_plan(prod_hourly_day: FloatArray) -> tuple[IntArray, IntArray]:
    """Group per 15-min slot (D×96) = operating type (by producing hours) × slot production on/off.

    Groups: 0 = type0 (no production); 2k−1 / 2k = type k ∈ {1,2,3} off / on. Parent = operating type.
    """
    on_hour = np.nan_to_num(np.asarray(prod_hourly_day, float)) > 0
    hours = on_hour.sum(1)
    kind = np.select([hours == 0, hours <= 12, hours <= 19], [0, 1, 2], 3)
    on = np.repeat(on_hour, 4, axis=1)
    group = np.where(kind[:, None] == 0, 0, 2 * kind[:, None] - 1 + on).astype(np.int64)
    return group, np.array([0, 1, 1, 2, 2, 3, 3], np.int64)


def kinds(panel: Panel, days: IntArray) -> IntArray:
    group, parent = groups_from_plan(panel["X"]["생산량"][days, ::4])
    return parent[group[:, 0]] if len(days) else np.zeros(0, np.int64)


def channels(q_raw: FloatArray, q_scale: float) -> FloatArray:
    """(…, 24) raw plan → (2, …, 24): on/off and log-volume, both non-decreasing in q."""
    q = np.nan_to_num(q_raw)
    return np.stack([(q > 0).astype(float), np.log1p(q) / np.log1p(q_scale)])


def _ref(panel: Panel, days: IntArray, kind: IntArray, kind_mean: FloatArray, scale: float) -> FloatArray:
    """Most recent complete same op/daytype day (B0′ reference), else the kind's training mean."""
    out = np.empty((len(days), 96))
    for i, d in enumerate(days):
        ref, _ = b0p_ref(panel, int(d))
        out[i] = panel["Y"][ref] / scale if ref is not None else kind_mean[kind[i]]
    return out


def _rw2(n: int = 96) -> FloatArray:
    D = np.roll(np.eye(n), -1, 1) - 2 * np.eye(n) + np.roll(np.eye(n), 1, 1)
    return D.T @ D


def fit_bat(panel: Panel, train_idx: IntArray, C: FloatArray, n_iter: int = 2000, burn: int = 1000,
            thin: int = 1, seed: int = 0, half_life: float | None = 30.0) -> BATState:
    Y = panel["Y"]
    days = train_idx[np.isfinite(Y[train_idx]).any(1)]
    scale = float(np.nanstd(Y[days]))
    q_scale = float(np.nanmax(panel["X"]["생산량"][days])) or 1.0
    kind = kinds(panel, days)
    Ys = Y[days] / scale
    kind_mean = np.stack([np.nanmean(Ys[kind == k], 0) if (kind == k).any() else np.nanmean(Ys, 0)
                          for k in range(KINDS)])
    ref = _ref(panel, days, kind, kind_mean, scale)
    X = channels(panel["X"]["생산량"][days, ::4], q_scale)  # (2, D, 24)
    obs = np.isfinite(Ys)
    dn, tn = np.nonzero(obs)
    y, kn = Ys[obs], kind[dn]
    j = kn * 96 + tn
    D, N, G = len(days), len(y), KINDS * 96
    P = G + KINDS  # linear block: idle (4×96) + γ per kind; attention and w go through Metropolis

    def attended(th: FloatArray, rows: IntArray) -> FloatArray:
        return (X[:, dn[rows]] * attention(th)[tn[rows]]).sum(-1).T  # (n, 2)

    rng = np.random.default_rng(seed)
    theta = np.tile(PRIOR_MU, (KINDS, 1))
    r = ref[dn, tn]
    trans = np.empty(N)  # Σ_c w_c Σ_h α x_c, the attention-transfer term
    for k in range(KINDS):
        m = np.flatnonzero(kn == k)
        trans[m] = attended(theta[k], m) @ np.exp(theta[k, 3:])
    coef = np.zeros(P)
    tau_rw = np.full(KINDS, 1.0)
    sig_u, lam2 = 0.1, 1.0
    RW = _rw2()
    out: dict[str, list] = {k: [] for k in ("idle", "gam", "w", "theta", "sigma_u", "lam", "rho", "b_eta")}
    nxt = np.flatnonzero((dn[1:] == dn[:-1]) & (tn[1:] == tn[:-1] + 1))  # consecutive observed slot pairs (i, i+1)
    # 반감기 가중 우도 (backbone.py와 같은 방식): 계절에 따라 수준이 움직여서, 전 기간 동일 가중이면
    # 최근 수준을 못 따라가고 날짜 이동 분산을 과대추정한다 (f1: 가정 27 kW vs 검증 7.3 kW).
    omega_d = np.ones(D) if half_life is None else 2.0 ** (-(days.max() - days) / half_life)
    omega = omega_d[dn]

    for it in range(n_iter):
        fit = coef[j] + coef[G + kn] * r + trans
        eps = np.maximum(np.abs(y - fit), 1e-8)
        lam = np.sqrt(lam2)
        wm = rng.wald(lam / eps, lam2)  # 1/v, Laplace scale mixture (Park & Casella 2008)
        lam2 = rng.gamma(1.0 + omega.sum(), 1.0 / (1.0 + 0.5 * np.sum(omega / wm)))
        wm = wm * omega  # 가중 우도 = 정밀도에 ω를 곱함

        # linear block (given attention): idle (4×96) + γ per kind — sparse normal equations
        target = y - trans
        A = np.zeros((P, P))
        A[np.arange(G), np.arange(G)] = np.bincount(j, wm, G)
        A[np.arange(G), G + np.arange(G) // 96] = np.bincount(j, wm * r, G)
        A[G + np.arange(KINDS), G + np.arange(KINDS)] = np.bincount(kn, wm * r * r, KINDS)
        A = np.triu(A) + np.triu(A, 1).T
        prior = np.zeros((P, P))
        for k in range(KINDS):
            prior[k * 96:(k + 1) * 96, k * 96:(k + 1) * 96] = RW / tau_rw[k] + 1e-4 * np.eye(96)
        prior[G:, G:] = np.eye(KINDS) / 25.0  # γ ~ N(0, 5²) in scaled units
        rhs = np.concatenate([np.bincount(j, wm * target, G), np.bincount(kn, wm * r * target, KINDS)])
        L = np.linalg.cholesky(A + prior)
        coef = np.linalg.solve(L.T, np.linalg.solve(L, rhs) + rng.standard_normal(P))
        idle = coef[:G].reshape(KINDS, 96)
        tau_rw = 1.0 / rng.gamma(1.0 + 48, 1.0 / (0.01 + 0.5 * np.einsum("kt,ts,ks->k", idle, RW, idle)))

        # attention shape + positive gains θ_k: random-walk Metropolis on the Gaussian (given 1/v) likelihood
        for k in range(1, KINDS):
            m = np.flatnonzero(kn == k)
            if not len(m):
                continue
            e = y[m] - coef[j[m]] - coef[G + k] * r[m]

            def logp(th: FloatArray, m: IntArray = m, e: FloatArray = e) -> tuple[float, FloatArray]:
                tk = attended(th, m) @ np.exp(th[3:])
                return (-0.5 * float(wm[m] @ (e - tk) ** 2)
                        - 0.5 * float((((th - PRIOR_MU) / PRIOR_SD) ** 2).sum()), tk)

            cur, _ = logp(theta[k])
            prop = theta[k] + STEP * rng.standard_normal(len(STEP))
            new, t_new = logp(prop)
            if np.log(rng.uniform()) < new - cur:
                theta[k], trans[m] = prop, t_new

        fit = coef[j] + coef[G + kn] * r + trans
        # 날짜 공통 변동은 예측 잡음에만 둔다: 잠재 u_d를 평균식에 넣으면 기준일 수준 정보를 흡수해
        # 예측(u=0)에서 사라진다 (f1 MAE 19.0 → 13.1로 확인).
        day_mean = np.bincount(dn, y - fit, D) / np.maximum(np.bincount(dn, minlength=D), 1)
        sig_u = 1.0 / rng.gamma(1.0 + omega_d.sum() / 2, 1.0 / (0.01 + 0.5 * omega_d @ day_mean ** 2))

        if it >= burn and (it - burn) % thin == 0:
            res = y - fit - day_mean[dn]
            a, b = res[nxt], res[nxt + 1]
            wp = omega[nxt]
            rho = float(np.clip((wp * a) @ b / ((wp * a) @ a), 0.0, 0.99))
            out["rho"].append(rho)
            out["b_eta"].append(float(wp @ np.abs(b - rho * a) / wp.sum()))  # weighted Laplace MLE scale
            for key, value in (("idle", idle), ("gam", coef[G:]), ("w", np.exp(theta[:, 3:]).T), ("theta", theta),
                               ("sigma_u", np.sqrt(sig_u)), ("lam", np.sqrt(lam2))):
                out[key].append(np.array(value, copy=True))
    return {"scale": scale, "q_scale": q_scale, "kind_mean": kind_mean, "C": C,
            **{key: np.array(v) for key, v in out.items()}}  # type: ignore[typeddict-item]


def _day_parts(state: BATState, panel: Panel, d: int) -> tuple[int, FloatArray, FloatArray]:
    k = int(kinds(panel, np.array([d]))[0])
    ref = _ref(panel, np.array([d]), np.array([k]), state["kind_mean"], state["scale"])[0]
    return k, ref, np.nan_to_num(panel["X"]["생산량"][d, ::4]) / state["q_scale"]


def transfer_draws(state: BATState, panel: Panel, d: int) -> tuple[FloatArray, FloatArray]:
    """Posterior mean curves (S, 96) at the day's plan and log-channel transfer ∂ŷ/∂q (S, 96, 24), kW.

    The on/off channel is a step in q, so the gradient only sees the log-volume channel; the
    re-scheduler keeps the operating window (rule R2), so on/off only changes at window edges.
    """
    k, ref, _ = _day_parts(state, panel, d)
    x = channels(panel["X"]["생산량"][d, ::4], state["q_scale"])  # (2, 24)
    alpha = np.array([attention(th[k]) for th in state["theta"]])  # (S, 96, 24)
    w = state["w"][:, :, k]  # (S, 2)
    mean = state["idle"][:, k] + state["gam"][:, k, None] * ref + np.einsum("sc,sth,ch->st", w, alpha, x)
    q = np.nan_to_num(panel["X"]["생산량"][d, ::4])
    dlog = 1.0 / ((1.0 + q) * np.log1p(state["q_scale"]))
    return mean * state["scale"], w[:, 1, None, None] * alpha * dlog * state["scale"]


def curve_fn(state: BATState, panel: Panel, d: int) -> Callable[[torch.Tensor], torch.Tensor]:
    """Posterior-mean curve ŷ(q), differentiable in the raw (24,) plan (on/off held at the day's plan)."""
    k, ref, _ = _day_parts(state, panel, d)
    on = torch.tensor(channels(panel["X"]["생산량"][d, ::4], state["q_scale"])[0])
    alpha = torch.tensor(np.array([attention(th[k]) for th in state["theta"]]))
    w = torch.tensor(state["w"][:, :, k])
    base = torch.tensor((state["idle"][:, k] + state["gam"][:, k, None] * ref).mean(0))
    denom = float(np.log1p(state["q_scale"]))

    def f(q: torch.Tensor) -> torch.Tensor:
        x = torch.stack([on, torch.log1p(q.to(torch.float64)) / denom])
        return (base + torch.einsum("sc,sth,ch->st", w, alpha, x).mean(0)) * state["scale"]
    return f


def from_paths(samples: FloatArray, C: FloatArray) -> Prediction:
    q = np.quantile(samples, TAUS, axis=0)
    M = samples.max(1)
    Mq = np.quantile(M, TAUS)
    return {"y_mean": samples.mean(0), "y_median": q[9], "q": q, "paths": samples,
            "M_hat_median": float(Mq[9]), "M_hat_mean": float(M.mean()),
            "peak_time_mode": int(np.bincount(samples.argmax(1), minlength=96).argmax()),
            "risk_raw": (M[:, None] > C).mean(0), "Mq": Mq}


def paths(state: BATState, panel: Panel, d: int) -> FloatArray:
    """Posterior predictive paths (S, 96) in kW: parameter draws + day effect + Laplace noise.

    Noise is deliberately left uncalibrated: inner calibration windows are systematically quieter than
    validation, so CRPS-calibrating picks κ_u=0.25 and under-covers (pooled 50/80/90 cov 0.51/0.67/0.73).
    """
    mean, _ = transfer_draws(state, panel, d)
    rng = np.random.default_rng(int(panel["dates"][d].strftime("%Y%m%d")))
    S = len(mean)
    rho, b = state["rho"], state["b_eta"]
    eta = rng.laplace(0, 1, (S, 96)) * b[:, None]
    e = np.empty((S, 96))
    # ponytail: 시작값을 정상분산 가우시안으로 근사, 경계 효과가 보이면 전일 말 잔차로 시작
    e[:, 0] = rng.standard_normal(S) * b * np.sqrt(2 / (1 - rho ** 2))
    for t in range(1, 96):
        e[:, t] = rho * e[:, t - 1] + eta[:, t]
    noise = state["sigma_u"][:, None] * rng.standard_normal((S, 1)) + e
    return np.maximum(mean + noise * state["scale"], 0.0)


def predict_bat(state: BATState, panel: Panel, d: int) -> Prediction:
    return from_paths(paths(state, panel, d), state["C"])


def _with_inner[S](fit: Callable[[Panel, IntArray, FloatArray], S]) -> Callable[[Panel, str, FloatArray], S]:
    def wrapped(panel: Panel, fold: str, C: FloatArray) -> S:
        tr = role_idx(panel, fold, "train")
        cal = internal_split(panel, fold)[2]
        state = fit(panel, tr, C)
        state["inner_state"] = fit(panel, tr[tr < cal[0]] if cal.size else tr, C)  # type: ignore[index]
        return state
    return wrapped


def bat_model(n_iter: int = 2000) -> Model[BATState]:
    return {"name": "BAT", "fit": _with_inner(lambda p, tr, C: fit_bat(p, tr, C, n_iter, n_iter // 2)),
            "predict": predict_bat, "inner_state": lambda state: state.get("inner_state")}

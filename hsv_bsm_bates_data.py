import numpy as np
import pandas as pd
from math import erf, sqrt, exp, log


# ============================================================
# 0. Utilities
# ============================================================

def norm_cdf(x):
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def safe_clip_price(price, lower_bound, upper_bound):
    """
    Clip observed noisy price into no-arbitrage bounds.
    """
    if not np.isfinite(price):
        return np.nan

    return min(max(price, lower_bound), upper_bound)


def option_bounds(S, K, T, r, q, cp_flag):
    """
    European option no-arbitrage lower and upper bounds.
    q is continuous dividend yield d.
    """
    cp = cp_flag.upper()

    if cp == "C":
        lower = max(S * np.exp(-q * T) - K * np.exp(-r * T), 0.0)
        upper = S * np.exp(-q * T)
    elif cp == "P":
        lower = max(K * np.exp(-r * T) - S * np.exp(-q * T), 0.0)
        upper = K * np.exp(-r * T)
    else:
        raise ValueError("cp_flag must be 'C' or 'P'.")

    return lower, upper


def add_noise_to_price(
    clean_price,
    S,
    K,
    T,
    r,
    q,
    cp_flag,
    epsilon,
    rng,
    noise_type="relative",
):
    """
    Add controlled noise to Bates clean price.

    noise_type:
        relative:
            price_obs = price_clean * (1 + epsilon * Z)
        absolute_norm:
            price_obs = price_clean + S * epsilon * Z
    """
    if epsilon <= 0:
        return clean_price

    z = rng.normal(0.0, 1.0)

    if noise_type == "relative":
        noisy_price = clean_price * (1.0 + epsilon * z)

    elif noise_type == "absolute_norm":
        noisy_price = clean_price + S * epsilon * z

    else:
        raise ValueError("noise_type must be 'relative' or 'absolute_norm'.")

    lower, upper = option_bounds(S, K, T, r, q, cp_flag)

    return safe_clip_price(noisy_price, lower, upper)


# ============================================================
# 1. Black-Scholes price
# ============================================================

def bs_price(S, K, T, r, q, sigma, cp_flag="C"):
    """
    European option price under Black-Scholes.

    q is continuous dividend yield d.
    """
    cp = cp_flag.upper()

    if T <= 0:
        if cp == "C":
            return max(S - K, 0.0)
        else:
            return max(K - S, 0.0)

    sigma = max(sigma, 1e-12)

    d1 = (
        np.log(S / K)
        + (r - q + 0.5 * sigma ** 2) * T
    ) / (sigma * np.sqrt(T))

    d2 = d1 - sigma * np.sqrt(T)

    if cp == "C":
        price = (
            S * np.exp(-q * T) * norm_cdf(d1)
            - K * np.exp(-r * T) * norm_cdf(d2)
        )
    elif cp == "P":
        price = (
            K * np.exp(-r * T) * norm_cdf(-d2)
            - S * np.exp(-q * T) * norm_cdf(-d1)
        )
    else:
        raise ValueError("cp_flag must be 'C' or 'P'.")

    return float(max(price, 0.0))


def bs_delta(S, K, T, r, q, sigma, cp_flag="C"):
    """
    European option delta under Black-Scholes.
    """
    cp = cp_flag.upper()

    if T <= 0:
        if cp == "C":
            return 1.0 if S > K else 0.0
        return -1.0 if S < K else 0.0

    sigma = max(sigma, 1e-12)
    d1 = (
        np.log(S / K)
        + (r - q + 0.5 * sigma ** 2) * T
    ) / (sigma * np.sqrt(T))

    if cp == "C":
        return float(np.exp(-q * T) * norm_cdf(d1))
    if cp == "P":
        return float(-np.exp(-q * T) * norm_cdf(-d1))
    raise ValueError("cp_flag must be 'C' or 'P'.")


# ============================================================
# 2. Heston characteristic function
# ============================================================

def heston_cf_logS(u, S0, T, r, q, v0, kappa, theta, xi, rho):
    """
    Characteristic function of log(S_T) under Heston.
    q is continuous dividend yield d.
    """
    u = np.asarray(u, dtype=np.complex128)
    i = 1j

    x0 = np.log(S0)

    d = np.sqrt(
        (rho * xi * i * u - kappa) ** 2
        + xi ** 2 * (i * u + u ** 2)
    )

    g = (
        kappa - rho * xi * i * u - d
    ) / (
        kappa - rho * xi * i * u + d
    )

    exp_neg_dT = np.exp(-d * T)

    C = (
        i * u * (x0 + (r - q) * T)
        + (kappa * theta / xi ** 2)
        * (
            (kappa - rho * xi * i * u - d) * T
            - 2.0 * np.log((1.0 - g * exp_neg_dT) / (1.0 - g))
        )
    )

    D = (
        (kappa - rho * xi * i * u - d)
        / xi ** 2
        * ((1.0 - exp_neg_dT) / (1.0 - g * exp_neg_dT))
    )

    return np.exp(C + D * v0)


# ============================================================
# 3. Bates characteristic function
# ============================================================

def bates_cf_logS(
    u,
    S0,
    T,
    r,
    q,
    v0,
    kappa,
    theta,
    xi,
    rho,
    lam,
    muJ,
    sigmaJ,
):
    """
    Characteristic function of log(S_T) under Bates model.

    Bates = Heston stochastic volatility + compound Poisson lognormal jumps.

    dS/S- = (r - q - lam * kbar) dt
            + sqrt(v) dW
            + (J - 1) dN

    log J ~ N(muJ, sigmaJ^2)
    kbar = E[J - 1] = exp(muJ + 0.5 sigmaJ^2) - 1
    """
    u = np.asarray(u, dtype=np.complex128)
    i = 1j

    kbar = np.exp(muJ + 0.5 * sigmaJ ** 2) - 1.0

    # Heston diffusion part with risk-neutral drift adjusted by jump compensator
    # r - q - lam*kbar == r - q_adjusted
    q_adjusted = q + lam * kbar

    cf_heston_part = heston_cf_logS(
        u=u,
        S0=S0,
        T=T,
        r=r,
        q=q_adjusted,
        v0=v0,
        kappa=kappa,
        theta=theta,
        xi=xi,
        rho=rho,
    )

    # Jump part: sum of compound Poisson log jumps
    jump_cf = np.exp(
        lam * T * (
            np.exp(i * u * muJ - 0.5 * sigmaJ ** 2 * u ** 2)
            - 1.0
        )
    )

    return cf_heston_part * jump_cf


# ============================================================
# 4. COS payoff coefficients
# ============================================================

def _chi_psi(k, a, b, c, d):
    """
    COS coefficients for exp(y) and 1 over interval [c, d].
    """
    k = np.asarray(k)
    u = k * np.pi / (b - a)

    expr1 = (
        np.cos(u * (d - a)) * np.exp(d)
        - np.cos(u * (c - a)) * np.exp(c)
    )

    expr2 = u * (
        np.sin(u * (d - a)) * np.exp(d)
        - np.sin(u * (c - a)) * np.exp(c)
    )

    chi = (expr1 + expr2) / (1.0 + u ** 2)

    psi = np.empty_like(u, dtype=float)
    psi[k == 0] = d - c

    idx = k != 0
    psi[idx] = (
        np.sin(u[idx] * (d - a))
        - np.sin(u[idx] * (c - a))
    ) / u[idx]

    return chi, psi


def cos_price_from_cf_logS(
    cf_logS_func,
    S0,
    K,
    T,
    r,
    q,
    cp_flag,
    c1,
    c2,
    N=256,
    L=12,
):
    """
    Generic COS pricer using characteristic function of log(S_T).
    """
    cp = cp_flag.upper()

    if T <= 0:
        if cp == "C":
            return max(S0 - K, 0.0)
        else:
            return max(K - S0, 0.0)

    c2 = max(float(c2), 1e-12)

    a = c1 - L * np.sqrt(c2)
    b = c1 + L * np.sqrt(c2)

    # payoff kink y=log(S/K)=0 should be inside [a,b]
    a = min(a, -1e-10)
    b = max(b, 1e-10)

    k = np.arange(N)
    u = k * np.pi / (b - a)

    # Characteristic function of y = log(S_T / K)
    phi_y = cf_logS_func(u) * np.exp(-1j * u * np.log(K))

    if cp == "C":
        # payoff: K * (exp(y) - 1)^+
        chi, psi = _chi_psi(k, a, b, 0.0, b)
        Vk = 2.0 / (b - a) * K * (chi - psi)

    elif cp == "P":
        # payoff: K * (1 - exp(y))^+
        chi, psi = _chi_psi(k, a, b, a, 0.0)
        Vk = 2.0 / (b - a) * K * (-chi + psi)

    else:
        raise ValueError("cp_flag must be 'C' or 'P'.")

    terms = np.real(phi_y * np.exp(-1j * u * a)) * Vk
    terms[0] *= 0.5

    price = np.exp(-r * T) * np.sum(terms)

    return float(price)


# ============================================================
# 5. Heston and Bates COS wrappers
# ============================================================

def heston_price_cos(
    S0,
    K,
    T,
    r,
    q,
    v0,
    kappa,
    theta,
    xi,
    rho,
    cp_flag="C",
    N=256,
    L=12,
):
    """
    Heston European option price by COS method.
    """
    mean_int_var = theta * T + (v0 - theta) * (1.0 - np.exp(-kappa * T)) / kappa

    c1 = np.log(S0 / K) + (r - q) * T - 0.5 * mean_int_var

    c2 = mean_int_var * (
        1.0 + 0.5 * xi ** 2 * T + 2.0 * abs(rho) * xi * np.sqrt(T)
    )

    cf = lambda u: heston_cf_logS(
        u=u,
        S0=S0,
        T=T,
        r=r,
        q=q,
        v0=v0,
        kappa=kappa,
        theta=theta,
        xi=xi,
        rho=rho,
    )

    return cos_price_from_cf_logS(
        cf_logS_func=cf,
        S0=S0,
        K=K,
        T=T,
        r=r,
        q=q,
        cp_flag=cp_flag,
        c1=c1,
        c2=c2,
        N=N,
        L=L,
    )


def bates_price_cos(
    S0,
    K,
    T,
    r,
    q,
    v0,
    kappa,
    theta,
    xi,
    rho,
    lam,
    muJ,
    sigmaJ,
    cp_flag="C",
    N=256,
    L=15,
):
    """
    Bates European option price by COS method.
    """
    kbar = np.exp(muJ + 0.5 * sigmaJ ** 2) - 1.0

    mean_int_var = theta * T + (v0 - theta) * (1.0 - np.exp(-kappa * T)) / kappa

    # approximate first cumulant of y = log(S_T / K)
    c1 = (
        np.log(S0 / K)
        + (r - q - lam * kbar) * T
        - 0.5 * mean_int_var
        + lam * T * muJ
    )

    # approximate second cumulant:
    # stochastic variance part + compound Poisson log-jump variance
    c2_heston = mean_int_var * (
        1.0 + 0.5 * xi ** 2 * T + 2.0 * abs(rho) * xi * np.sqrt(T)
    )

    c2_jump = lam * T * (sigmaJ ** 2 + muJ ** 2)

    c2 = c2_heston + c2_jump

    cf = lambda u: bates_cf_logS(
        u=u,
        S0=S0,
        T=T,
        r=r,
        q=q,
        v0=v0,
        kappa=kappa,
        theta=theta,
        xi=xi,
        rho=rho,
        lam=lam,
        muJ=muJ,
        sigmaJ=sigmaJ,
    )

    return cos_price_from_cf_logS(
        cf_logS_func=cf,
        S0=S0,
        K=K,
        T=T,
        r=r,
        q=q,
        cp_flag=cp_flag,
        c1=c1,
        c2=c2,
        N=N,
        L=L,
    )


# ============================================================
#  BS synthetic dataset
# ============================================================

def generate_bs_dataset(
    n_samples=100_000,
    seed=123,
    cp_flag="P",
    use_log_moneyness=True,
    filter_K_range=True,
):
    """
    Generate BS synthetic option prices.

    Columns are aligned with the BSM feature set:
        S, K, tau, r, d, IV1
    """
    rng = np.random.default_rng(seed)

    ranges = {
        "S": (0.5, 5.0),
        "K": (1e-4, 5.0),
        "tau": (1.0 / 252.0, 4.0),
        "r": (0.0, 0.1),
        "d": (0.0, 0.1),
        "m": (-0.5, 0.5),
        "sigma": (1e-4, 1.0),
    }

    rows = []
    attempts = 0
    max_attempts = int(n_samples * 10)

    while len(rows) < n_samples and attempts < max_attempts:
        attempts += 1

        S = rng.uniform(*ranges["S"])
        tau = rng.uniform(*ranges["tau"])
        r = rng.uniform(*ranges["r"])
        d = rng.uniform(*ranges["d"])
        sigma = rng.uniform(*ranges["sigma"])

        if use_log_moneyness:
            m = rng.uniform(*ranges["m"])
            F = S * np.exp((r - d) * tau)
            K = F * np.exp(m)

            if filter_K_range and not (ranges["K"][0] <= K <= ranges["K"][1]):
                continue
        else:
            K = rng.uniform(*ranges["K"])
            F = S * np.exp((r - d) * tau)
            m = np.log(K / F)

        price = bs_price(
            S=S,
            K=K,
            T=tau,
            r=r,
            q=d,
            sigma=sigma,
            cp_flag=cp_flag,
        )

        if not np.isfinite(price):
            continue

        lower, upper = option_bounds(S, K, tau, r, d, cp_flag)
        price = safe_clip_price(price, lower, upper)
        delta = bs_delta(
            S=S,
            K=K,
            T=tau,
            r=r,
            q=d,
            sigma=sigma,
            cp_flag=cp_flag,
        )

        rows.append(
            {
                "S": S,
                "K": K,
                "tau": tau,
                "r": r,
                "d": d,
                "m": m,
                "cp_flag": cp_flag.upper(),
                "IV1": sigma,
                "sigma": sigma,
                "y_put": price,
                "delta_put": delta,
                "epsilon": 0.0,
                "model": "BS",
            }
        )

    df = pd.DataFrame(rows)

    if len(df) < n_samples:
        print(f"Warning: only generated {len(df)} valid BS samples out of {n_samples} requested.")

    return df


# ============================================================
#  Heston synthetic dataset
# ============================================================

def generate_heston_dataset(
    n_samples=100_000,
    seed=123,
    cp_flag="P",
    N_COS=256,
    L_COS=12,
    enforce_feller=False,
):
    rng = np.random.default_rng(seed)

    rows = []

    # Ranges based on your BS table + Heston latent parameters
    ranges = {
        "S": (1e-4, 5.0),
        "K": (1e-4, 5.0),
        "tau": (1.0 / 252.0, 4.0),
        "r": (0.0, 0.1),
        "d": (0.0, 0.1),
        "v0": (1e-4, 1.0),
        "theta": (1e-4, 1.0),
        "kappa": (0.1, 5.0),
        "xi": (0.05, 2.0),
        "rho": (-0.95, 0.5),
    }

    attempts = 0
    max_attempts = int(n_samples * 5)

    while len(rows) < n_samples and attempts < max_attempts:
        attempts += 1

        S = rng.uniform(*ranges["S"])
        K = rng.uniform(*ranges["K"])
        tau = rng.uniform(*ranges["tau"])
        r = rng.uniform(*ranges["r"])
        d = rng.uniform(*ranges["d"])

        v0 = rng.uniform(*ranges["v0"])
        theta = rng.uniform(*ranges["theta"])
        kappa = rng.uniform(*ranges["kappa"])
        xi = rng.uniform(*ranges["xi"])
        rho = rng.uniform(*ranges["rho"])

        if enforce_feller:
            if 2.0 * kappa * theta < xi ** 2:
                continue

        try:
            price = heston_price_cos(
                S0=S,
                K=K,
                T=tau,
                r=r,
                q=d,
                v0=v0,
                kappa=kappa,
                theta=theta,
                xi=xi,
                rho=rho,
                cp_flag=cp_flag,
                N=N_COS,
                L=L_COS,
            )
            bump = max(1e-4 * S, 1e-5)
            S_down = max(S - bump, 1e-8)
            S_up = S + bump
            price_down = heston_price_cos(
                S0=S_down,
                K=K,
                T=tau,
                r=r,
                q=d,
                v0=v0,
                kappa=kappa,
                theta=theta,
                xi=xi,
                rho=rho,
                cp_flag=cp_flag,
                N=N_COS,
                L=L_COS,
            )
            price_up = heston_price_cos(
                S0=S_up,
                K=K,
                T=tau,
                r=r,
                q=d,
                v0=v0,
                kappa=kappa,
                theta=theta,
                xi=xi,
                rho=rho,
                cp_flag=cp_flag,
                N=N_COS,
                L=L_COS,
            )
            delta = (price_up - price_down) / (S_up - S_down)
        except Exception:
            continue

        # Basic no-arbitrage and numerical filters
        if not np.isfinite(price) or not np.isfinite(delta):
            continue

        if price < -1e-8:
            continue

        if cp_flag.upper() == "C":
            upper_bound = S * np.exp(-d * tau)
        else:
            upper_bound = K * np.exp(-r * tau)

        if price > upper_bound + 1e-6:
            continue

        price = max(price, 0.0)

        rows.append(
            {
                "S": S,
                "K": K,
                "tau": tau,
                "r": r,
                "d": d,
                "v0": v0,
                "theta": theta,
                "kappa": kappa,
                "xi": xi,
                "rho": rho,
                "cp_flag": cp_flag.upper(),
                "y_put": price,
                "delta_put": delta,
                "model": "Heston",
            }
        )

    df = pd.DataFrame(rows)

    if len(df) < n_samples:
        print(f"Warning: only generated {len(df)} valid samples out of {n_samples} requested.")

    return df



# ============================================================
# 7. Bates synthetic dataset with epsilon noise
# ============================================================

def generate_bates_dataset(
    n_samples=100_000,
    seed=456,
    cp_flag="C",
    epsilon=0.01,
    noise_type="relative",
    N_COS=256,
    L_COS=15,
    use_log_moneyness=True,
    filter_K_range=True,
    enforce_feller=False,
):
    """
    Generate Bates synthetic option prices.

    Bates = Heston + jumps.
    The observed target price can include epsilon-controlled noise.
    """
    rng = np.random.default_rng(seed)

    ranges = {
        # Contract variables
        "S": (0.5, 5.0),
        "K": (1e-4, 5.0),
        "tau": (1.0 / 252.0, 4.0),
        "r": (0.0, 0.1),
        "d": (0.0, 0.1),
        "m": (-0.5, 0.5),

        # Heston latent parameters
        "v0": (1e-4, 0.25),
        "theta": (1e-4, 0.25),
        "kappa": (0.3, 5.0),
        "xi": (0.05, 1.0),
        "rho": (-0.95, 0.2),

        # Bates jump parameters
        "lam": (0.0, 1.0),
        "muJ": (-0.15, 0.02),
        "sigmaJ": (0.01, 0.30),
    }

    rows = []
    attempts = 0
    max_attempts = int(n_samples * 20)

    while len(rows) < n_samples and attempts < max_attempts:
        attempts += 1

        S = rng.uniform(*ranges["S"])
        tau = rng.uniform(*ranges["tau"])
        r = rng.uniform(*ranges["r"])
        d = rng.uniform(*ranges["d"])

        if use_log_moneyness:
            m = rng.uniform(*ranges["m"])
            F = S * np.exp((r - d) * tau)
            K = F * np.exp(m)

            if filter_K_range and not (ranges["K"][0] <= K <= ranges["K"][1]):
                continue
        else:
            K = rng.uniform(*ranges["K"])
            F = S * np.exp((r - d) * tau)
            m = np.log(K / F)

        v0 = rng.uniform(*ranges["v0"])
        theta = rng.uniform(*ranges["theta"])
        kappa = rng.uniform(*ranges["kappa"])
        xi = rng.uniform(*ranges["xi"])
        rho = rng.uniform(*ranges["rho"])

        lam = rng.uniform(*ranges["lam"])
        muJ = rng.uniform(*ranges["muJ"])
        sigmaJ = rng.uniform(*ranges["sigmaJ"])

        if enforce_feller:
            if 2.0 * kappa * theta < xi ** 2:
                continue

        try:
            price_clean = bates_price_cos(
                S0=S,
                K=K,
                T=tau,
                r=r,
                q=d,
                v0=v0,
                kappa=kappa,
                theta=theta,
                xi=xi,
                rho=rho,
                lam=lam,
                muJ=muJ,
                sigmaJ=sigmaJ,
                cp_flag=cp_flag,
                N=N_COS,
                L=L_COS,
            )
        except Exception:
            continue

        if not np.isfinite(price_clean):
            continue

        lower, upper = option_bounds(S, K, tau, r, d, cp_flag)

        # If COS returns a small numerical violation, clip it.
        # If violation is too large, discard the sample.
        if price_clean < lower - 1e-5:
            continue

        if price_clean > upper + 1e-5:
            continue

        price_clean = safe_clip_price(price_clean, lower, upper)

        price_obs = add_noise_to_price(
            clean_price=price_clean,
            S=S,
            K=K,
            T=tau,
            r=r,
            q=d,
            cp_flag=cp_flag,
            epsilon=epsilon,
            rng=rng,
            noise_type=noise_type,
        )

        if not np.isfinite(price_obs):
            continue

        rows.append(
            {
                "S": S,
                "K": K,
                "tau": tau,
                "r": r,
                "d": d,
                "m": m,
                "cp_flag": cp_flag.upper(),

                # # No single IV1 in Bates; keep NaN for compatibility
                # "IV1": np.nan,
                # "sigma": np.nan,

                # Heston latent parameters
                "v0": v0,
                "theta": theta,
                "kappa": kappa,
                "xi": xi,
                "rho": rho,

                # Bates jump parameters
                "lam": lam,
                "muJ": muJ,
                "sigmaJ": sigmaJ,

                # clean and noisy target
                "price_clean": price_clean,
                "price": price_obs,
                # "price_norm_clean": price_clean / S,
                # "price_norm": price_obs / S,

                "epsilon": epsilon,
                "noise_type": noise_type,
                "model": "Bates",
            }
        )

    df = pd.DataFrame(rows)

    if len(df) < n_samples:
        print(f"Warning: only generated {len(df)} valid Bates samples out of {n_samples} requested.")

    return df
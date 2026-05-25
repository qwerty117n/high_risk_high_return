"""
MIDAS リスク・リターン分析（論文準拠版）
Ghysels, Santa-Clara, Valkanov (2003) の手法を日経225に適用

【元コードからの主な変更点】
  1. 推定方法  : OLS → 準最尤法 (Quasi-MLE)
  2. 被説明変数: 月次リターン → 月次超過リターン
  3. ラグ数    : K=60 → K=260（約1年）
  4. ウェイト  : Beta多項式 → 指数形式（論文の式(3)）
  5. 標準誤差  : OLS SE → Newey-West頑健標準誤差（12ヶ月ラグ）
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import json, warnings
warnings.filterwarnings('ignore')

MAX_LAG = 260  # 論文: 約1年分（260営業日）

# =====================================================
# 1. データ取得（実データ失敗時は日経225模擬データ）
# =====================================================
print("=" * 60)
print("データ取得中...")

try:
    import yfinance as yf
    nikkei = yf.download("^N225", start="2010-01-01",
                         auto_adjust=True, progress=False)
    daily_price = nikkei['Close'].squeeze().dropna()
    if len(daily_price) < 100:
        raise ValueError("データ不足")
    daily_ret = daily_price.pct_change().dropna()
    data_source = "日経225（実データ）"
except Exception:
    print("  実データ取得失敗 → 日経225模擬データを使用")
    np.random.seed(2024)
    N_DAYS = 3500  # 約14年
    # GARCH(1,1): 日経225の実証的パラメータに近い値
    omega_g, alpha_g, beta_g = 4e-6, 0.07, 0.91
    sig2 = np.zeros(N_DAYS)
    r    = np.zeros(N_DAYS)
    sig2[0] = omega_g / (1 - alpha_g - beta_g)
    for t in range(1, N_DAYS):
        sig2[t] = omega_g + alpha_g * r[t-1]**2 + beta_g * sig2[t-1]
        r[t] = np.sqrt(sig2[t]) * np.random.standard_normal()
    dates    = pd.date_range('2010-01-04', periods=N_DAYS, freq='B')
    daily_ret = pd.Series(r, index=dates)
    data_source = "日経225（模擬データ）"

# 月次リターン（日次から複利計算）
monthly_ret = daily_ret.resample('ME').apply(lambda x: (1+x).prod() - 1)

# 月次超過リターン
# ※ 日本の政策金利は2010年代ほぼゼロのため0で近似
# ※ 実運用時は日銀短期金利データを使用すること
rf_monthly  = pd.Series(0.0, index=monthly_ret.index)
excess_ret  = monthly_ret - rf_monthly

print(f"  データ: {data_source}")
print(f"  日次観測数: {len(daily_ret)},  月次観測数: {len(excess_ret)}")

# =====================================================
# 2. r²マトリックスを事前計算
# =====================================================
print("\nr²マトリックス構築中...")

month_ends  = excess_ret.index
daily_vals  = daily_ret.values.astype(np.float64)
daily_ts    = daily_ret.index.to_numpy(dtype='datetime64[D]')
n_months    = len(month_ends)

R2_mat = np.zeros((n_months, MAX_LAG))
cnt    = np.zeros(n_months, dtype=int)

for i, me in enumerate(month_ends):
    ms   = np.datetime64(me.to_period('M').to_timestamp(), 'D')
    mask = daily_ts < ms
    past = daily_vals[mask]
    c    = min(len(past), MAX_LAG)
    if c < 22:
        continue
    R2_mat[i, :c] = past[-c:][::-1]**2  # d=1が最も直近
    cnt[i] = c

print(f"  有効月数: {(cnt >= 22).sum()}")

# =====================================================
# 3. ウェイト関数（論文の式(3): 指数形式）
# =====================================================
def midas_weights(k1, k2, max_lag=MAX_LAG):
    """
    w_d(κ₁,κ₂) = exp{κ₁d + κ₂d²} / Σexp{κ₁i + κ₂i²}
    κ₂ < 0 が必須（重みをゼロに収束させるため）
    κ₁ < 0 を推奨（単調減衰）
    """
    d  = np.arange(1, max_lag + 1, dtype=np.float64)
    lw = k1 * d + k2 * d**2
    lw -= lw.max()   # 数値安定化（オーバーフロー防止）
    w  = np.exp(lw)
    return w / w.sum()

# =====================================================
# 4. MIDAS分散推定（論文の式(2)）
# =====================================================
def midas_var(k1, k2):
    """
    V_t^MIDAS = 22 × Σ_{d=1}^{260} w_d × r²_{t-d}
    """
    w = midas_weights(k1, k2)
    V = np.full(n_months, np.nan)

    full = cnt >= MAX_LAG
    if full.any():
        V[full] = 22.0 * (R2_mat[full] @ w)    # ベクトル化（高速）

    for i in np.where((cnt >= 22) & ~full)[0]:  # 初期の不足月
        c  = cnt[i]
        ww = w[:c] / w[:c].sum()
        V[i] = 22.0 * np.dot(ww, R2_mat[i, :c])
    return V

# =====================================================
# 5. 準最尤法（Quasi-MLE）
#    論文の式(4): R_{t+1} ~ N(μ + γV_t, V_t)
#    対数尤度: Σ[-0.5 log(V_t) - 0.5(R-μ-γV)²/V_t]
#
#    κ₁,κ₂を最適化し、μとγはWLSで解析的に解く
#    （プロファイル尤度アプローチ）
# =====================================================
excess_vals = excess_ret.values

def wls_estimate(k1, k2):
    """κ₁,κ₂固定時の μ,γ の準MLE（WLS）"""
    V_all = midas_var(k1, k2)
    R = excess_vals[1:]
    V = V_all[:-1]
    ok = np.isfinite(R) & np.isfinite(V) & (V > 0)
    R_, V_ = R[ok], V[ok]

    if len(R_) < 30:
        return None

    # WLS: weight = 1/V_t（正規分布の対数尤度から導出）
    w_  = 1.0 / V_
    X   = np.column_stack([np.ones(len(R_)), V_])
    XtW = (X * w_[:, None]).T  # (2, n)

    try:
        theta  = np.linalg.solve(XtW @ X, XtW @ R_)
        cov_th = np.linalg.inv(XtW @ X)
    except np.linalg.LinAlgError:
        return None

    mu_, gm_ = theta
    mean_    = mu_ + gm_ * V_
    ll       = -0.5 * np.log(V_) - 0.5 * (R_ - mean_)**2 / V_

    return {'theta': theta, 'cov': cov_th,
            'R': R_, 'V': V_, 'ok': ok,
            'llf': float(np.sum(ll)), 'n': int(ok.sum())}

def profile_neg_llf(kappa):
    k1, k2 = kappa
    if k2 >= 0 or k1 > 0:
        return 1e10
    res = wls_estimate(k1, k2)
    return 1e10 if res is None else -res['llf']

# =====================================================
# 6. 最適化（複数の初期値から最良解を選択）
# =====================================================
print("\nMIDAS最適化中...")

inits = [
    [-0.05, -5e-8], [-0.02, -1e-8], [-0.08, -1e-7],
    [-0.03, -3e-8], [-0.01, -5e-9], [-0.10, -2e-7],
]
best_res = None
best_val = np.inf

for i, p0 in enumerate(inits):
    print(f"  試行 {i+1}/{len(inits)}...", end=' ', flush=True)
    try:
        res = minimize(profile_neg_llf, p0, method='Nelder-Mead',
                       options={'maxiter': 8000, 'xatol': 1e-8, 'fatol': 1e-8})
        print(f"LLF={-res.fun:.2f}, κ₁={res.x[0]:.4f}, κ₂={res.x[1]:.2e}")
        if res.fun < best_val:
            best_val = res.fun
            best_res  = res
    except Exception as e:
        print(f"失敗: {e}")

k1_h, k2_h = best_res.x
result = wls_estimate(k1_h, k2_h)
mu_h, gamma_h = result['theta']
R_ok, V_ok, ok_mask = result['R'], result['V'], result['ok']

# =====================================================
# 7. Newey-West頑健標準誤差（論文はラグ12ヶ月）
# =====================================================
def newey_west_se(R, V, mu, gamma, n_lags=12):
    """
    サンドイッチ推定量: (X'WX)^{-1} × S_NW × (X'WX)^{-1}
    S_NW: Newey-West(1987)ロングラン分散
    """
    n  = len(R)
    wt = 1.0 / V
    X  = np.column_stack([np.ones(n), V])

    # スコア（各観測の勾配）
    resid  = R - mu - gamma * V
    scores = X * (wt * resid)[:, None]  # (n, 2)

    # Newey-West分散
    S = scores.T @ scores
    for lag in range(1, n_lags + 1):
        bartlett = 1.0 - lag / (n_lags + 1)
        Gl = scores[lag:].T @ scores[:-lag]
        S += bartlett * (Gl + Gl.T)
    S /= n

    # (X'WX)^{-1}
    XtW   = (X * wt[:, None]).T
    bread = np.linalg.inv(XtW @ X / n)

    sandwich = bread @ S @ bread / n
    return np.sqrt(np.diag(sandwich))

se_nw       = newey_west_se(R_ok, V_ok, mu_h, gamma_h, n_lags=12)
se_mu, se_gamma = se_nw
t_mu        = mu_h    / se_mu
t_gamma     = gamma_h / se_gamma

# =====================================================
# 8. 比較用: 元コードの手法（OLS, Beta weights, K=60）
# =====================================================
K_old = 60

def beta_weights_old(K, th1, th2):
    j = np.arange(1, K + 1)
    x = j / K
    w = (x ** (th1 - 1)) * ((1 - x) ** (th2 - 1))
    return w / w.sum()

# 元コードと同じ方法でラグ行列を再作成（K=60）
R2_mat_old = np.zeros((n_months, K_old))
cnt_old    = np.zeros(n_months, dtype=int)
for i, me in enumerate(month_ends):
    ms   = np.datetime64(me.to_period('M').to_timestamp(), 'D')
    mask = daily_ts < ms
    past = daily_vals[mask]
    c    = min(len(past), K_old)
    if c < K_old:
        continue
    R2_mat_old[i, :] = past[-K_old:][::-1]**2
    cnt_old[i] = K_old

# OLS推定（元コードに相当）
w_old = beta_weights_old(K_old, 1.0, 3.0)
V_old_all = np.full(n_months, np.nan)
full_old = cnt_old >= K_old
if full_old.any():
    V_old_all[full_old] = 22.0 * (R2_mat_old[full_old] @ w_old)

R_old = excess_vals[1:]
V_old = V_old_all[:-1]
ok_old = np.isfinite(R_old) & np.isfinite(V_old) & (V_old > 0)
R_o, V_o = R_old[ok_old], V_old[ok_old]

# OLS
X_o  = np.column_stack([np.ones(len(R_o)), V_o])
ols_theta = np.linalg.lstsq(X_o, R_o, rcond=None)[0]
ols_resid = R_o - X_o @ ols_theta
ols_se    = np.sqrt(np.var(ols_resid, ddof=2) * np.diag(np.linalg.inv(X_o.T @ X_o)))
ols_gamma, ols_t = ols_theta[1], ols_theta[1] / ols_se[1]
ols_R2    = np.corrcoef(V_o, R_o)[0,1]**2

# =====================================================
# 9. 予測力R²
# =====================================================
R2_R = np.corrcoef(V_ok, R_ok)[0, 1]**2

daily_period = daily_ret.index.to_period('M')
month_ends_ok = month_ends[1:][ok_mask]
rv_list = []
for me in month_ends_ok:
    p    = me.to_period('M')
    mask = daily_period == p
    rv_list.append(np.sum(daily_vals[mask]**2))
rv   = np.array(rv_list)
ok2  = rv > 0
R2_V = np.corrcoef(V_ok[ok2], rv[ok2])[0, 1]**2

# =====================================================
# 10. ウェイト累積分布
# =====================================================
w_opt = midas_weights(k1_h, k2_h)
cumw  = np.cumsum(w_opt)

# =====================================================
# 11. 結果保存（可視化用）
# =====================================================
results = {
    'mu': float(mu_h), 'se_mu': float(se_mu), 't_mu': float(t_mu),
    'gamma': float(gamma_h), 'se_gamma': float(se_gamma), 't_gamma': float(t_gamma),
    'kappa1': float(k1_h), 'kappa2': float(k2_h),
    'R2_R': float(R2_R), 'R2_V': float(R2_V),
    'llf': result['llf'], 'n_obs': result['n'],
    'cumw_1M': float(cumw[21]), 'cumw_2M': float(cumw[43]),
    'cumw_4M': float(cumw[87]), 'cumw_6M': float(cumw[129]),
    'ols_gamma': float(ols_gamma), 'ols_t': float(ols_t), 'ols_R2': float(ols_R2),
    'data_source': data_source,
}
with open('/home/claude/nikkei_results.json', 'w') as f:
    json.dump(results, f, indent=2)

pd.DataFrame({'lag': range(1, MAX_LAG+1), 'weight': w_opt}).to_csv(
    '/home/claude/nikkei_weights.csv', index=False)

# =====================================================
# 12. 結果表示
# =====================================================
print("\n" + "="*60)
print("  MIDAS 推定結果（論文準拠版）")
print(f"  データ: {data_source}")
print("="*60)

print(f"\n【変更点のまとめ】")
print(f"  {'':18} {'元コード':>12}  →  {'論文準拠':>12}")
print(f"  {'推定方法':<18} {'OLS':>12}  →  {'準最尤法(QML)':>12}")
print(f"  {'標準誤差':<18} {'通常SE':>12}  →  {'Newey-West SE':>12}")
print(f"  {'ラグ数(K)':<18} {'60日':>12}  →  {'260日':>12}")
print(f"  {'ウェイト関数':<18} {'Beta多項式':>12}  →  {'指数形式':>12}")

print(f"\n【推定パラメータ（論文準拠版）】")
print(f"  {'':12} {'推定値':>10} {'NW標準誤差':>12} {'t値':>10}")
print(f"  {'-'*46}")
print(f"  {'μ':<12} {mu_h:>10.5f} {se_mu:>12.5f} {t_mu:>10.3f}")
print(f"  {'γ（リスク係数）':<12} {gamma_h:>10.4f} {se_gamma:>12.4f} {t_gamma:>10.3f}")
print(f"  {'κ₁':<12} {k1_h:>10.5f}")
print(f"  {'κ₂':<12} {k2_h:>10.2e}")
print(f"  {'対数尤度':<12} {result['llf']:>10.3f}")
print(f"  {'観測数':<12} {result['n']:>10d}")

print(f"\n【元コードとの比較（γの推定値）】")
print(f"  {'':25} {'γ':>8} {'t値':>8} {'R²':>8}")
print(f"  {'-'*50}")
print(f"  {'論文準拠 (MIDAS QML)':<25} {gamma_h:>8.4f} {t_gamma:>8.3f} {R2_R*100:>7.2f}%")
print(f"  {'元コード (OLS, K=60)':<25} {ols_gamma:>8.4f} {ols_t:>8.3f} {ols_R2*100:>7.2f}%")
print(f"  {'論文の値（参考）':<25} {'4.007':>8} {'2.647':>8} {'2.40':>7}%")

print(f"\n【予測力】")
print(f"  R²（リターン予測）: {R2_R*100:.2f}%  （論文: 2.4%）")
print(f"  R²（分散予測）    : {R2_V*100:.2f}%  （論文: 8.2%）")

print(f"\n【MIDASウェイト累積分布】")
for d, lab, paper in [(22,'1M','26%'),(44,'2M','46%'),(88,'4M','75%'),(130,'6M','')]:
    note = f"← 論文: {paper}" if paper else ""
    print(f"  直前{lab}（{d:3d}日）: {cumw[d-1]:5.1%}  {note}")

print(f"\n【結論】")
if t_gamma > 1.96:
    print(f"  γ = {gamma_h:.3f} (t = {t_gamma:.3f})")
    print(f"  → リスク・リターントレードオフが5%水準で有意 ✓")
elif t_gamma > 1.64:
    print(f"  γ = {gamma_h:.3f} (t = {t_gamma:.3f})")
    print(f"  → 10%水準で有意（5%水準では非有意）")
else:
    print(f"  γ = {gamma_h:.3f} (t = {t_gamma:.3f})")
    print(f"  → 統計的に有意なトレードオフは確認されない")
print("="*60)

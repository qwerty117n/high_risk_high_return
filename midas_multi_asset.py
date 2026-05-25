"""
MIDAS リスク・リターン分析 — 複数資産対応版
Ghysels, Santa-Clara, Valkanov (2003) 論文準拠

【CSVフォーマット（2種類に対応）】
  ワイド形式（推奨）:
    date,TOPIX,SP500,Nikkei225
    2010-01-04,1000.0,1132.99,10654.3
    ...

  ロング形式:
    date,asset,price
    2010-01-04,TOPIX,1000.0
    2010-01-04,SP500,1132.99
    ...

使い方:
    python midas_multi_asset.py --csv prices.csv
    python midas_multi_asset.py --csv prices.csv --assets TOPIX SP500
    python midas_multi_asset.py --csv prices.csv --rf_annual 0.001
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import argparse, json, warnings, sys
warnings.filterwarnings('ignore')

# =====================================================
# 定数
# =====================================================
MAX_LAG  = 260   # 論文準拠: 約1年の日次データ
NW_LAGS  = 12    # Newey-West標準誤差のラグ数（月次）
MIN_OBS  = 60    # 最低必要月次観測数

# =====================================================
# A. CSVの読み込み・前処理
# =====================================================
def load_price_csv(filepath: str, date_col: str = 'date',
                   asset_col: str = None, price_col: str = None) -> pd.DataFrame:
    """
    CSVを読み込み、ワイド形式（日付×資産）のDataFrameに変換する。

    Parameters
    ----------
    filepath   : CSVファイルパス
    date_col   : 日付列の列名（デフォルト: 'date'）
    asset_col  : ロング形式の場合の資産名列（省略時はワイド形式と判断）
    price_col  : ロング形式の場合の価格列名
    """
    df = pd.read_csv(filepath, parse_dates=[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)
    df[date_col] = pd.to_datetime(df[date_col])

    # ロング形式の検出・変換
    if asset_col and price_col:
        df = df.pivot(index=date_col, columns=asset_col, values=price_col)
        df.index.name = 'date'
        df = df.reset_index()
    else:
        # 日付列以外がすべて資産価格と見なす
        df = df.set_index(date_col)

    # 数値列のみ残す
    df = df.select_dtypes(include=[np.number])
    df.index.name = 'date'
    return df

# =====================================================
# B. MIDAS コアルーティン（1資産）
# =====================================================

def build_r2_matrix(daily_ret: np.ndarray,
                    daily_ts: np.ndarray,
                    month_ends,
                    max_lag: int = MAX_LAG):
    """r²マトリックスを事前計算（計算コスト削減の要）"""
    n_months = len(month_ends)
    R2_mat   = np.zeros((n_months, max_lag))
    cnt      = np.zeros(n_months, dtype=int)

    for i, me in enumerate(month_ends):
        ms   = np.datetime64(me.to_period('M').to_timestamp(), 'D')
        mask = daily_ts < ms
        past = daily_ret[mask]
        c    = min(len(past), max_lag)
        if c < 22:
            continue
        R2_mat[i, :c] = past[-c:][::-1]**2  # d=1が最も直近
        cnt[i] = c
    return R2_mat, cnt


def midas_weights(k1: float, k2: float, max_lag: int = MAX_LAG) -> np.ndarray:
    """
    論文の式(3): 指数形式ウェイト関数
    w_d = exp{κ₁d + κ₂d²} / Σexp{κ₁i + κ₂i²}
    κ₂ < 0 が必須（重みを収束させるため）
    """
    d  = np.arange(1, max_lag + 1, dtype=np.float64)
    lw = k1 * d + k2 * d**2
    lw -= lw.max()
    w  = np.exp(lw)
    return w / w.sum()


def midas_var(k1, k2, R2_mat, cnt, max_lag=MAX_LAG):
    """論文の式(2): V_t^MIDAS = 22 × Σ w_d × r²_{t-d}"""
    w = midas_weights(k1, k2, max_lag)
    n_months = R2_mat.shape[0]
    V = np.full(n_months, np.nan)

    full = cnt >= max_lag
    if full.any():
        V[full] = 22.0 * (R2_mat[full] @ w)

    for i in np.where((cnt >= 22) & ~full)[0]:
        c  = cnt[i]
        ww = w[:c] / w[:c].sum()
        V[i] = 22.0 * np.dot(ww, R2_mat[i, :c])
    return V


def wls_estimate(k1, k2, excess_vals, R2_mat, cnt):
    """
    κ₁,κ₂固定時の μ,γ の準MLE (WLS)
    論文の式(4): R_{t+1} ~ N(μ + γV_t, V_t)
    → 重み 1/V_t の加重最小二乗問題
    """
    V_all = midas_var(k1, k2, R2_mat, cnt)
    R     = excess_vals[1:]
    V     = V_all[:-1]
    ok    = np.isfinite(R) & np.isfinite(V) & (V > 0)
    R_, V_ = R[ok], V[ok]

    if len(R_) < MIN_OBS // 2:
        return None

    w_  = 1.0 / V_
    X   = np.column_stack([np.ones(len(R_)), V_])
    XtW = (X * w_[:, None]).T
    try:
        theta  = np.linalg.solve(XtW @ X, XtW @ R_)
        cov_th = np.linalg.inv(XtW @ X)
    except np.linalg.LinAlgError:
        return None

    mu_, gm_ = theta
    ll       = -0.5 * np.log(V_) - 0.5 * (R_ - mu_ - gm_ * V_)**2 / V_
    return {'theta': theta, 'cov': cov_th,
            'R': R_, 'V': V_, 'ok': ok,
            'llf': float(np.sum(ll)), 'n': int(ok.sum())}


def newey_west_se(R, V, mu, gamma, n_lags=NW_LAGS):
    """
    論文準拠: Newey-West(1987) 頑健標準誤差
    サンドイッチ推定量: (X'WX)^{-1} S_NW (X'WX)^{-1}
    """
    n  = len(R)
    wt = 1.0 / V
    X  = np.column_stack([np.ones(n), V])

    resid  = R - mu - gamma * V
    scores = X * (wt * resid)[:, None]

    S = scores.T @ scores
    for lag in range(1, n_lags + 1):
        bartlett = 1.0 - lag / (n_lags + 1)
        Gl       = scores[lag:].T @ scores[:-lag]
        S       += bartlett * (Gl + Gl.T)
    S /= n

    XtW   = (X * wt[:, None]).T
    bread = np.linalg.inv(XtW @ X / n)
    sand  = bread @ S @ bread / n
    return np.sqrt(np.diag(sand))


def run_midas_single(asset_name: str,
                     daily_ret: np.ndarray,
                     daily_ts: np.ndarray,
                     excess_vals: np.ndarray,
                     month_ends,
                     verbose: bool = True) -> dict:
    """
    1資産についてMIDAS分析を実行し結果を返す。

    Returns
    -------
    dict with keys:
        asset, n_obs, mu, gamma, se_mu, se_gamma, t_mu, t_gamma,
        kappa1, kappa2, R2_R, R2_V, llf, cumw_*M, converged
    """
    if verbose:
        print(f"\n  [{asset_name}] 分析中...")

    # r²マトリックス構築
    R2_mat, cnt = build_r2_matrix(daily_ret, daily_ts, month_ends)
    valid_months = int((cnt >= 22).sum())

    if valid_months < MIN_OBS:
        print(f"    警告: 有効月数({valid_months})が不足しています（最低{MIN_OBS}必要）")
        return {'asset': asset_name, 'converged': False,
                'error': f'有効月数不足: {valid_months}'}

    # プロファイル尤度最適化
    def profile_neg_llf(kappa):
        k1, k2 = kappa
        if k2 >= 0 or k1 > 0:
            return 1e10
        res = wls_estimate(k1, k2, excess_vals, R2_mat, cnt)
        return 1e10 if res is None else -res['llf']

    inits = [
        [-0.05, -5e-8], [-0.02, -1e-8], [-0.08, -1e-7],
        [-0.03, -3e-8], [-0.01, -5e-9], [-0.10, -2e-7],
    ]
    best_res, best_val = None, np.inf
    for p0 in inits:
        try:
            res = minimize(profile_neg_llf, p0, method='Nelder-Mead',
                           options={'maxiter': 8000, 'xatol': 1e-8, 'fatol': 1e-8})
            if res.fun < best_val:
                best_val, best_res = res.fun, res
        except Exception:
            pass

    if best_res is None:
        return {'asset': asset_name, 'converged': False, 'error': '最適化失敗'}

    k1_h, k2_h = best_res.x
    est = wls_estimate(k1_h, k2_h, excess_vals, R2_mat, cnt)
    if est is None:
        return {'asset': asset_name, 'converged': False, 'error': '推定失敗'}

    mu_h, gamma_h = est['theta']
    R_ok, V_ok    = est['R'], est['V']

    # Newey-West標準誤差
    se_nw         = newey_west_se(R_ok, V_ok, mu_h, gamma_h)
    se_mu, se_gm  = se_nw

    # R²（リターン予測）
    R2_R = float(np.corrcoef(V_ok, R_ok)[0, 1]**2)

    # R²（分散予測）
    daily_period  = pd.DatetimeIndex(
        daily_ts.astype('datetime64[D]').astype(str)).to_period('M')
    month_ends_ok = month_ends[1:][est['ok']]
    rv_list = []
    for me in month_ends_ok:
        p = me.to_period('M')
        rv_list.append(float(np.sum(daily_ret[daily_period == p]**2)))
    rv    = np.array(rv_list)
    ok2   = rv > 0
    R2_V  = float(np.corrcoef(V_ok[ok2], rv[ok2])[0, 1]**2) if ok2.sum() > 2 else np.nan

    # ウェイト累積分布
    w_opt = midas_weights(k1_h, k2_h)
    cumw  = np.cumsum(w_opt)

    result = {
        'asset'    : asset_name,
        'converged': True,
        'n_obs'    : est['n'],
        'mu'       : float(mu_h),
        'gamma'    : float(gamma_h),
        'se_mu'    : float(se_mu),
        'se_gamma' : float(se_gm),
        't_mu'     : float(mu_h / se_mu),
        't_gamma'  : float(gamma_h / se_gm),
        'kappa1'   : float(k1_h),
        'kappa2'   : float(k2_h),
        'R2_R'     : R2_R,
        'R2_V'     : R2_V,
        'llf'      : est['llf'],
        'cumw_1M'  : float(cumw[21]),
        'cumw_2M'  : float(cumw[43]),
        'cumw_4M'  : float(cumw[87]),
        'cumw_6M'  : float(cumw[min(129, MAX_LAG-1)]),
        'weights'  : w_opt.tolist(),
    }

    if verbose:
        sig = "✓ 有意(5%)" if abs(result['t_gamma']) > 1.96 else \
              "△ 有意(10%)" if abs(result['t_gamma']) > 1.64 else "✗ 非有意"
        print(f"    γ={gamma_h:.4f}  t={result['t_gamma']:.3f}  "
              f"R²(ret)={R2_R*100:.2f}%  {sig}")

    return result


# =====================================================
# C. 複数資産ループ
# =====================================================
def run_midas_multi(price_df: pd.DataFrame,
                    assets: list = None,
                    rf_annual: float = 0.0) -> pd.DataFrame:
    """
    複数資産に対してMIDAS分析を実行し、結果DataFrameを返す。

    Parameters
    ----------
    price_df   : 日付インデックス × 資産価格のDataFrame
    assets     : 分析する資産名リスト（省略時は全列）
    rf_annual  : 年率リスクフリーレート（デフォルト: 0）
    """
    if assets is None:
        assets = list(price_df.columns)

    rf_monthly = (1 + rf_annual)**(1/12) - 1
    all_results = []

    print(f"\n{'='*60}")
    print(f"  MIDAS 複数資産分析（論文準拠版）")
    print(f"  分析対象: {assets}")
    print(f"  無リスク金利: 年率{rf_annual*100:.2f}%（月次換算: {rf_monthly*100:.4f}%）")
    print(f"{'='*60}")

    for asset in assets:
        if asset not in price_df.columns:
            print(f"\n  [{asset}] 列が見つかりません。スキップします。")
            continue

        # 日次リターン（単純リターン）
        prices     = price_df[asset].dropna()
        daily_ret  = prices.pct_change().dropna().values.astype(np.float64)
        daily_ts   = prices.pct_change().dropna().index.to_numpy(dtype='datetime64[D]')

        # 月次超過リターン
        prices_s      = prices.pct_change().dropna()
        prices_s.index = pd.DatetimeIndex(prices_s.index)
        monthly_ret   = prices_s.resample('ME').apply(
                            lambda x: (1 + x).prod() - 1)
        excess_ret    = monthly_ret - rf_monthly
        excess_vals   = excess_ret.values.astype(np.float64)
        month_ends    = excess_ret.index

        result = run_midas_single(
            asset_name  = asset,
            daily_ret   = daily_ret,
            daily_ts    = daily_ts,
            excess_vals = excess_vals,
            month_ends  = month_ends,
        )
        all_results.append(result)

    return all_results


# =====================================================
# D. 結果の整形・表示
# =====================================================
def print_summary(results: list):
    """結果を比較表形式で表示"""
    print(f"\n{'='*70}")
    print("  比較サマリー")
    print(f"{'='*70}")
    header = f"  {'資産':<14} {'γ':>8} {'t値':>8} {'p値':>8} {'R²(ret)':>9} {'R²(var)':>9} {'判定':>10}"
    print(header)
    print(f"  {'-'*66}")

    from scipy.stats import t as t_dist
    for r in results:
        if not r.get('converged'):
            print(f"  {r['asset']:<14} {'推定失敗':>50}")
            continue
        pval   = 2 * (1 - t_dist.cdf(abs(r['t_gamma']), df=r['n_obs']-2))
        sig    = "★★ p<5%" if pval < 0.05 else \
                 "★ p<10%"  if pval < 0.10 else "n.s."
        print(f"  {r['asset']:<14} {r['gamma']:>8.4f} {r['t_gamma']:>8.3f} "
              f"{pval:>8.4f} {r['R2_R']*100:>8.2f}% {r['R2_V']*100:>8.2f}%  {sig:>10}")

    print(f"\n  {'γ':>8}: リスク・リターン係数（正かつ有意 → トレードオフ確認）")
    print(f"  {'R²(ret)':>8}: MIDASボラティリティによるリターン予測力")
    print(f"  {'R²(var)':>8}: MIDASボラティリティによる実現分散予測力")
    print(f"{'='*70}")

    print(f"\n  MIDASウェイト累積分布（論文参考値: 1M=26%, 2M=46%, 4M=75%）")
    print(f"  {'資産':<14} {'1M':>8} {'2M':>8} {'4M':>8} {'6M':>8}")
    print(f"  {'-'*46}")
    for r in results:
        if not r.get('converged'):
            continue
        print(f"  {r['asset']:<14} {r['cumw_1M']*100:>7.1f}% "
              f"{r['cumw_2M']*100:>7.1f}% {r['cumw_4M']*100:>7.1f}% "
              f"{r['cumw_6M']*100:>7.1f}%")


def save_results(results: list, out_prefix: str = '/home/claude/midas_output'):
    """結果をJSON・CSVに保存"""
    # JSON（詳細）
    json_path = f"{out_prefix}.json"
    save_data = [{k: v for k, v in r.items() if k != 'weights'}
                 for r in results]
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)

    # CSV（サマリー）
    csv_path = f"{out_prefix}_summary.csv"
    rows = []
    for r in results:
        if not r.get('converged'):
            rows.append({'asset': r['asset'], 'converged': False})
            continue
        rows.append({
            'asset'    : r['asset'],
            'gamma'    : round(r['gamma'], 6),
            't_gamma'  : round(r['t_gamma'], 4),
            'se_gamma' : round(r['se_gamma'], 6),
            'mu'       : round(r['mu'], 6),
            't_mu'     : round(r['t_mu'], 4),
            'kappa1'   : round(r['kappa1'], 6),
            'kappa2'   : f"{r['kappa2']:.4e}",
            'R2_return': round(r['R2_R'] * 100, 4),
            'R2_variance': round(r['R2_V'] * 100, 4),
            'log_likelihood': round(r['llf'], 4),
            'n_obs'    : r['n_obs'],
            'cumw_1M'  : round(r['cumw_1M'] * 100, 2),
            'cumw_2M'  : round(r['cumw_2M'] * 100, 2),
            'cumw_4M'  : round(r['cumw_4M'] * 100, 2),
            'cumw_6M'  : round(r['cumw_6M'] * 100, 2),
        })
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"\n  結果保存: {json_path}")
    print(f"  結果保存: {csv_path}")


# =====================================================
# E. メイン実行
# =====================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MIDAS 複数資産分析')
    parser.add_argument('--csv',       default='prices.csv', help='入力CSVファイルパス')
    parser.add_argument('--assets',    nargs='+', default=None, help='分析する資産名（省略時: 全列）')
    parser.add_argument('--rf_annual', type=float, default=0.0, help='年率リスクフリーレート（例: 0.001 = 0.1%%）')
    parser.add_argument('--date_col',  default='date', help='日付列の列名')
    parser.add_argument('--out',       default='/home/claude/midas_output', help='出力ファイルのプレフィックス')
    args = parser.parse_args()

    # --- 読み込み ---
    print(f"CSVファイル読み込み: {args.csv}")
    try:
        price_df = load_price_csv(args.csv, date_col=args.date_col)
        print(f"  読込完了: {price_df.shape[0]}行 × {price_df.shape[1]}列")
        print(f"  資産一覧: {list(price_df.columns)}")
        print(f"  期間: {price_df.index[0]} ～ {price_df.index[-1]}")
    except Exception as e:
        print(f"  エラー: {e}")
        sys.exit(1)

    # --- 分析実行 ---
    results = run_midas_multi(
        price_df   = price_df,
        assets     = args.assets,
        rf_annual  = args.rf_annual,
    )

    # --- 結果表示・保存 ---
    print_summary(results)
    save_results(results, out_prefix=args.out)

# ============================================================
# French, Schwert & Stambaugh (1987) 汎用分析コード
# CSVファイル（日付・資産価格）から分析を実行する
# ============================================================

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


# ============================================================
# ★ 設定エリア：ここだけ変更すれば異なるデータに対応できる
# ============================================================

CSV_PATH   = "your_data.csv"   # CSVファイルのパス
DATE_COL   = "Date"            # 日付列の列名
PRICE_COL  = "S&P500"           # 価格列の列名
ASSET_NAME = "S&P500"          # グラフタイトル用の資産名
ARIMA_ORDER = (2, 0, 0)        # ARIMAの次数（基本はそのままでOK）

# ============================================================
# ★ CSVの形式について
# ============================================================
# 以下のような形式を想定しています。
#
#   Date,Close
#   1990-01-02,353.40
#   1990-01-03,350.54
#   1990-01-04,352.20
#   ...
#
# 列名が異なる場合は DATE_COL・PRICE_COL を変更してください。
# 例）"日付" と "終値" の場合：
#   DATE_COL  = "日付"
#   PRICE_COL = "終値"


# ============================================================
# Step 1：CSVの読み込みと日次対数リターンの計算
# ============================================================
print("Step 1：CSVを読み込んでいます...")

df_raw = pd.read_csv(
    CSV_PATH,
    parse_dates=[DATE_COL],   # 日付列を日付型に変換
    index_col=DATE_COL,       # 日付列をインデックスに設定
)

# 価格列だけ取り出す
prices = df_raw[PRICE_COL].dropna().sort_index()

# 日次対数リターンを計算
# log(今日の価格 / 昨日の価格)
daily = np.log(prices / prices.shift(1)).dropna()
daily.name = "log_ret"

print(f"  資産名：{ASSET_NAME}")
print(f"  期間　：{daily.index[0].date()} 〜 {daily.index[-1].date()}")
print(f"  日数　：{len(daily)} 営業日\n")


# ============================================================
# Step 2：月次実現分散（RV）の計算
# ============================================================
print("Step 2：月次実現分散を計算しています...")

def calc_realized_variance(daily_returns):
    """
    月内の日次リターンから実現分散を計算する
    RV = Σ r²_d  +  2 × Σ r_d × r_{d-1}
    """
    r = daily_returns.values
    squared_sum           = np.sum(r ** 2)
    covariance_correction = 2 * np.sum(r[1:] * r[:-1])
    return squared_sum + covariance_correction

# 月ごとにグループ化して実現分散を計算
monthly_rv  = (
    daily
    .groupby(pd.Grouper(freq="ME"))
    .apply(calc_realized_variance)
    .rename("RV")
)

# 月次リターン（日次対数リターンの月内合計）
monthly_ret = daily.resample("ME").sum().rename("Return")

print(f"  月数　：{len(monthly_rv)} ヶ月")
print(f"  RV平均：{monthly_rv.mean():.6f}\n")


# ============================================================
# Step 3：ARIMAで実現分散を分解
# ============================================================
print("Step 3：ARIMAで実現分散を分解しています...")

rv_clean     = monthly_rv.dropna()
arima_result = ARIMA(rv_clean, order=ARIMA_ORDER).fit()

sigma2 = arima_result.fittedvalues  # 予測可能な分散
u      = arima_result.resid         # 予測不可能な分散（サプライズ）

print(f"  ARIMAの次数：{ARIMA_ORDER}")
print(f"  AIC　：{arima_result.aic:.2f}\n")


# ============================================================
# Step 4：データを1つのDataFrameにまとめる
# ============================================================
data = pd.DataFrame({
    "Return" : monthly_ret,
    "sigma2" : sigma2,
    "u"      : u,
}).dropna()

print(f"Step 4：分析データを作成しました（{len(data)} ヶ月分）\n")


# ============================================================
# Step 5：回帰分析
# ============================================================

def run_regression(y, x, x_label):
    """
    OLS回帰を実行して結果を表示する汎用関数
    y      : 被説明変数（リターン）
    x      : 説明変数（分散）
    x_label: 説明変数の名前（表示用）
    """
    X      = sm.add_constant(x)
    result = sm.OLS(y, X).fit()

    coef  = result.params[x.name]
    pval  = result.pvalues[x.name]
    r2    = result.rsquared

    print(result.summary())
    print(f"  係数　：{coef:+.4f}")
    print(f"  p値　：{pval:.4f}")
    print(f"  R²　 ：{r2:.4f}")

    # 符号と有意性の自動判定
    significant = pval < 0.05
    if x_label == "予測可能分散":
        if coef > 0 and significant:
            print("  → ✅ ハイリスク・ハイリターン支持（正・有意）")
        elif coef < 0 and significant:
            print("  → ❌ ハイリスク・ハイリターン否定（負・有意）")
        else:
            print("  → ⚠️  非有意（明確な関係なし）")
    else:
        if coef < 0 and significant:
            print("  → ✅ ボラティリティフィードバック効果を確認（負・有意）")
        else:
            print("  → ⚠️  非有意（明確なフィードバック効果なし）")

    return result

print("=" * 55)
print("回帰①：予測可能な分散 → 当期リターン")
print("=" * 55)
reg1 = run_regression(data["Return"], data["sigma2"], "予測可能分散")

print("\n" + "=" * 55)
print("回帰②：予測不可能な分散 → 当期リターン")
print("=" * 55)
reg2 = run_regression(data["Return"], data["u"], "予測不可能分散")


# ============================================================
# Step 6：可視化
# ============================================================
print("\nStep 6：グラフを描画しています...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(f"FSS(1987) 再現分析：{ASSET_NAME}", fontsize=14)

# --- グラフ①：実現分散の時系列 ---
ax = axes[0, 0]
monthly_rv.plot(ax=ax, color="steelblue", linewidth=0.8)
ax.set_title("月次実現分散（RV）の推移")
ax.set_ylabel("実現分散")

# --- グラフ②：予測可能・不可能の分解 ---
ax = axes[0, 1]
sigma2.plot(ax=ax, label="予測可能（σ²）", color="steelblue", linewidth=0.8)
u.plot(ax=ax, label="予測不可能（u）", color="tomato", linewidth=0.8, alpha=0.7)
ax.axhline(0, color="black", linewidth=0.5)
ax.set_title("実現分散の分解")
ax.legend(fontsize=9)

# --- グラフ③：散布図（回帰①）---
ax = axes[1, 0]
ax.scatter(data["sigma2"], data["Return"], alpha=0.3, s=10, color="steelblue")
x_line = np.linspace(data["sigma2"].min(), data["sigma2"].max(), 100)
y_line = reg1.params["const"] + reg1.params["sigma2"] * x_line
ax.plot(x_line, y_line, color="red", linewidth=1.5)
ax.set_title(
    f"回帰①　予測可能分散 → リターン\n"
    f"β={reg1.params['sigma2']:.3f}　p={reg1.pvalues['sigma2']:.3f}"
)
ax.set_xlabel("予測可能な分散（σ²）")
ax.set_ylabel("月次リターン")
ax.axhline(0, color="gray", linewidth=0.5)

# --- グラフ④：散布図（回帰②）---
ax = axes[1, 1]
ax.scatter(data["u"], data["Return"], alpha=0.3, s=10, color="tomato")
x_line2 = np.linspace(data["u"].min(), data["u"].max(), 100)
y_line2  = reg2.params["const"] + reg2.params["u"] * x_line2
ax.plot(x_line2, y_line2, color="red", linewidth=1.5)
ax.set_title(
    f"回帰②　予測不可能分散 → リターン\n"
    f"γ={reg2.params['u']:.3f}　p={reg2.pvalues['u']:.4f}"
)
ax.set_xlabel("予測不可能な分散（u）")
ax.set_ylabel("月次リターン")
ax.axhline(0, color="gray", linewidth=0.5)

plt.tight_layout()
output_filename = f"fss1987_{ASSET_NAME}.png"
plt.savefig(output_filename, dpi=150, bbox_inches="tight")
plt.show()
print(f"  グラフを {output_filename} に保存しました")


# ============================================================
# 結果サマリー
# ============================================================
print("\n" + "=" * 55)
print(f"分析結果サマリー：{ASSET_NAME}")
print("=" * 55)
print(f"  β（予測可能リスク）  　= {reg1.params['sigma2']:+.4f}"
      f"　　p = {reg1.pvalues['sigma2']:.4f}")
print(f"  γ（予測不可能リスク）　= {reg2.params['u']:+.4f}"
      f"　　p = {reg2.pvalues['u']:.4f}")

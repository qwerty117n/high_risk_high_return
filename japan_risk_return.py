"""
================================================
日本株：ハイリスク・ハイリターン検証
クロスセクション回帰分析（2段階）
================================================
実行前にインストール:
    pip install yfinance pandas numpy scipy statsmodels matplotlib

実行:
    python japan_risk_return.py
"""

import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy import stats

# =============================================
# 設定（自由に変更してください）
# =============================================
TICKERS = [
    "7203.T",  # トヨタ
    "6758.T",  # ソニー
    "8306.T",  # 三菱UFJ
    "9984.T",  # ソフトバンクG
    "6861.T",  # キーエンス
    "8035.T",  # 東京エレクトロン
    "4063.T",  # 信越化学
    "7741.T",  # HOYA
    "9433.T",  # KDDI
    "4519.T",  # 中外製薬
    "8058.T",  # 三菱商事
    "6367.T",  # ダイキン
    "7267.T",  # ホンダ
    "9432.T",  # NTT
    "4502.T",  # 武田薬品
    "8316.T",  # 三井住友FG
    "6954.T",  # ファナック
    "4661.T",  # オリエンタルランド
    "2914.T",  # JT
    "9020.T",  # JR東日本
    "7751.T",  # キヤノン
    "6098.T",  # リクルート
    "4543.T",  # テルモ
    "8031.T",  # 三井物産
    "6702.T",  # 富士通
    "7011.T",  # 三菱重工
    "4568.T",  # 第一三共
    "9613.T",  # NTTデータ
    "3382.T",  # セブン&アイ
    "8411.T",  # みずほFG
]

NAME_MAP = {
    "7203.T":"トヨタ",        "6758.T":"ソニー",
    "8306.T":"三菱UFJ",       "9984.T":"ソフトバンクG",
    "6861.T":"キーエンス",    "8035.T":"東京エレクトロン",
    "4063.T":"信越化学",      "7741.T":"HOYA",
    "9433.T":"KDDI",          "4519.T":"中外製薬",
    "8058.T":"三菱商事",      "6367.T":"ダイキン",
    "7267.T":"ホンダ",        "9432.T":"NTT",
    "4502.T":"武田薬品",      "8316.T":"三井住友FG",
    "6954.T":"ファナック",    "4661.T":"OLC",
    "2914.T":"JT",            "9020.T":"JR東日本",
    "7751.T":"キヤノン",      "6098.T":"リクルート",
    "4543.T":"テルモ",        "8031.T":"三井物産",
    "6702.T":"富士通",        "7011.T":"三菱重工",
    "4568.T":"第一三共",      "9613.T":"NTTデータ",
    "3382.T":"セブン&アイ",   "8411.T":"みずほFG",
}

BENCHMARK = "^N225"       # ベンチマーク（日経225）
RF_ANNUAL = 0.001         # 無リスク金利（年率0.1%）
START     = "2019-01-01"
END       = "2024-12-31"

# =============================================
# 1. データ取得
# =============================================
print("="*55)
print("  日本株 ハイリスク・ハイリターン検証")
print("="*55)
print(f"\nデータ取得中... ({START} 〜 {END})")

all_tickers = TICKERS + [BENCHMARK]
raw = yf.download(all_tickers, start=START, end=END,
                  auto_adjust=True, progress=False)["Close"]
raw = raw.dropna(axis=1, thresh=int(len(raw) * 0.8))
available = [t for t in TICKERS if t in raw.columns]
print(f"取得成功: {len(available)} 銘柄\n")

returns   = raw[available].pct_change().dropna()
bench_ret = raw[BENCHMARK].pct_change().dropna()
rf_daily  = RF_ANNUAL / 252

# =============================================
# 2. 第1段階：時系列回帰でβを推定
# =============================================
print("第1段階：各銘柄のβを時系列回帰で推定中...")
results = {}
for ticker in available:
    r = returns[ticker].dropna()
    b = bench_ret.reindex(r.index).dropna()
    r = r.reindex(b.index)

    slope, intercept, *_ = stats.linregress(b - rf_daily, r - rf_daily)

    ann_ret = r.mean() * 252
    ann_vol = r.std() * np.sqrt(252)
    sharpe  = (ann_ret - RF_ANNUAL) / ann_vol if ann_vol > 0 else np.nan

    cum = (1 + r).cumprod()
    mdd = ((cum - cum.cummax()) / cum.cummax()).min()

    results[ticker] = {
        "名前":             NAME_MAP.get(ticker, ticker),
        "β":               round(slope, 3),
        "α_daily(%)":      round(intercept * 100, 4),
        "年率リターン(%)":  round(ann_ret * 100, 2),
        "年率ボラティリティ(%)": round(ann_vol * 100, 2),
        "シャープレシオ":   round(sharpe, 3),
        "最大DD(%)":       round(mdd * 100, 2),
    }

df = pd.DataFrame(results).T
for col in ["β","年率リターン(%)","年率ボラティリティ(%)","シャープレシオ","最大DD(%)"]:
    df[col] = pd.to_numeric(df[col])

df_show = df.set_index("名前").drop(columns=["α_daily(%)"])
print("\n【銘柄別指標（β昇順）】")
pd.set_option("display.float_format", "{:.2f}".format)
pd.set_option("display.width", 100)
print(df_show.sort_values("β").to_string())

# =============================================
# 3. 第2段階：クロスセクション回帰
# =============================================
print("\n" + "="*55)
print("  第2段階：クロスセクション回帰")
print("="*55)

x = df["β"].values.astype(float)
y = df["年率リターン(%)"].values.astype(float)
mask = ~(np.isnan(x) | np.isnan(y))
x, y = x[mask], y[mask]

slope2, intercept2, r_val, p_val, std_err = stats.linregress(x, y)

bench_ann  = bench_ret.mean() * 252
gamma_capm = (bench_ann - RF_ANNUAL) * 100

print(f"\n  回帰式: R = {intercept2:.3f} + {slope2:.3f} × β")
print(f"\n  推定γ̂（傾き）       : {slope2:.4f} %")
print(f"  推定α̂（切片）       : {intercept2:.4f} %")
print(f"  R²                  : {r_val**2:.4f}  （{r_val**2*100:.1f}% 説明）")
print(f"  p値                 : {p_val:.4f}")
print(f"  CAPMの予測γ         : {gamma_capm:.4f} %")
print(f"  無リスク金利（年率） : {RF_ANNUAL*100:.2f} %")

# β*計算
denom = gamma_capm - slope2
if abs(denom) > 1e-6:
    beta_star = (intercept2 - RF_ANNUAL * 100) / denom
else:
    beta_star = float('inf')

beta_min, beta_max = x.min(), x.max()
print(f"\n  β* （交差点）       : {beta_star:.3f}")
print(f"  銘柄βの範囲         : {beta_min:.2f} 〜 {beta_max:.2f}")

# ケース判定
if 0 < beta_star <= beta_max:
    case = f"ケースA：交差点(β*={beta_star:.2f})が第1象限内"
    case_detail = f"β≈{beta_star:.2f}を境に損得が逆転する"
elif beta_star <= 0:
    case = "ケースB：交差点がβ≤0の領域"
    case_detail = "観測範囲全域で実際リターンがCAPM予測を下回る傾向"
else:
    case = "交差点がβの観測範囲外（右側）"
    case_detail = "観測範囲内では実際リターンがCAPM予測を上回り続ける"

print(f"  ケース判定          : {case}")

# =============================================
# 4. 総合解釈
# =============================================
print("\n" + "="*55)
print("  総合解釈")
print("="*55)

if slope2 > 0 and p_val < 0.05:
    g_eval = "正かつ有意（p<0.05）→ ハイリスク・ハイリターンの傾向あり"
elif slope2 > 0 and p_val < 0.10:
    g_eval = "正だが弱い有意（p<0.10）→ 傾向はあるが確信度は低い"
elif slope2 > 0:
    g_eval = "正だが有意でない → 傾向は見えるが偶然の可能性を排除できない"
else:
    g_eval = "負かつ有意 → 逆転（低リスク・高リターン）"
print(f"  γ̂の評価   : {g_eval}")

ratio = slope2 / gamma_capm if abs(gamma_capm) > 1e-6 else float('inf')
if ratio < 0.5:
    s_eval = f"CAPMの予測の{ratio*100:.0f}%水準 → ハイリスクの割に報われていない（弱い成立）"
elif ratio < 1.3:
    s_eval = f"CAPMの予測の{ratio*100:.0f}%水準 → 概ね整合"
else:
    s_eval = f"CAPMの予測の{ratio*100:.0f}%水準 → リスクが強く報われている"
print(f"  γ̂の大きさ : {s_eval}")
print(f"  R²の評価   : {r_val**2:.4f} → βの説明力は{'低い' if r_val**2 < 0.1 else '中程度' if r_val**2 < 0.3 else '高い'}")
print(f"  β*の解釈   : {case_detail}")

# =============================================
# 5. 可視化
# =============================================
plt.rcParams["font.family"] = ["Hiragino Sans","Yu Gothic","Meiryo","sans-serif"]
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.patch.set_facecolor("#0f0f1a")

for ax in axes:
    ax.set_facecolor("#1a1a2e")
    ax.tick_params(colors="#e0e0e0")
    ax.xaxis.label.set_color("#e0e0e0")
    ax.yaxis.label.set_color("#e0e0e0")
    for spine in ax.spines.values():
        spine.set_edgecolor("#2a2a3e")
    ax.grid(color="#2a2a3e", linestyle="--", linewidth=0.5, alpha=0.7)

# --- 左：散布図＋回帰線 ---
ax1 = axes[0]
names = [NAME_MAP.get(t, t) for t in df.index]
ax1.scatter(x, y, color="#00d4ff", s=70, alpha=0.85,
            edgecolors="#ffffff", linewidths=0.4, zorder=3)
for i, name in enumerate([n for n, m in zip(
        [NAME_MAP.get(t,t) for t in df.index], mask) if m]):
    ax1.annotate(name, (x[i], y[i]), fontsize=7, color="#e0e0e0",
                 alpha=0.8, xytext=(4, 3), textcoords="offset points")

x_line = np.linspace(x.min() - 0.1, x.max() + 0.1, 100)
ax1.plot(x_line, slope2 * x_line + intercept2,
         color="#ff6b6b", linewidth=2, linestyle="--",
         label=f"実際の回帰線（γ̂={slope2:.2f}%）")
ax1.plot(x_line, gamma_capm * x_line + RF_ANNUAL * 100,
         color="#00d4ff", linewidth=1.5, linestyle=":",
         label=f"CAPMの予測（γ={gamma_capm:.2f}%）")

ax1.axhline(0, color="#555566", linewidth=0.8)
ax1.set_xlabel("β（市場感応度）")
ax1.set_ylabel("年率リターン (%)")
ax1.set_title("β vs 年率リターン（クロスセクション回帰）",
              color="#00d4ff", fontsize=11, fontweight="bold")
ax1.legend(fontsize=8, facecolor="#1a1a2e", labelcolor="#e0e0e0")

info = (f"γ̂ = {slope2:.3f}%  p = {p_val:.3f}\n"
        f"α̂ = {intercept2:.3f}%  R² = {r_val**2:.3f}\n"
        f"β* = {beta_star:.2f}")
ax1.text(0.03, 0.97, info, transform=ax1.transAxes,
         fontsize=8, color="#ffd700", verticalalignment="top",
         bbox=dict(boxstyle="round", facecolor="#0f0f1a", alpha=0.7))

# --- 右：β分位別平均リターン ---
ax2 = axes[1]
df_valid = df.dropna(subset=["β","年率リターン(%)"])
df_valid["分位"] = pd.qcut(df_valid["β"], q=4,
                            labels=["Q1\n低β","Q2","Q3","Q4\n高β"])
group = df_valid.groupby("分位", observed=True).agg(
    平均リターン=("年率リターン(%)", "mean"),
    平均シャープ=("シャープレシオ", "mean"),
)
colors = ["#00d4ff" if v >= 0 else "#ff6b6b"
          for v in group["平均リターン"]]
bars = ax2.bar(group.index, group["平均リターン"],
               color=colors, edgecolor="#ffffff",
               linewidth=0.4, width=0.55)
for bar, val in zip(bars, group["平均リターン"]):
    ax2.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + (0.3 if val >= 0 else -0.8),
             f"{val:.1f}%", ha="center", color="#e0e0e0", fontsize=9)

ax2_r = ax2.twinx()
ax2_r.plot(group.index, group["平均シャープ"],
           color="#ffd700", marker="D", linewidth=2,
           markersize=6, label="シャープレシオ")
ax2_r.set_ylabel("シャープレシオ", color="#ffd700", fontsize=9)
ax2_r.tick_params(colors="#ffd700")
ax2_r.legend(loc="upper right", fontsize=8,
             facecolor="#1a1a2e", labelcolor="#e0e0e0")
ax2.axhline(0, color="#555566", linewidth=0.8)
ax2.set_xlabel("β分位（Q1=低β〜Q4=高β）")
ax2.set_ylabel("平均年率リターン (%)")
ax2.set_title("β分位別 平均リターン（ポートフォリオソート）",
              color="#00d4ff", fontsize=11, fontweight="bold")

fig.suptitle(f"日本株 ハイリスク・ハイリターン検証  {START}〜{END}",
             color="#00d4ff", fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("japan_risk_return.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print("\n  → グラフ保存: japan_risk_return.png")
plt.show()

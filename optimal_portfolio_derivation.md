# 効用関数による最適投資比率の導出

## 共通の設定

- 現在の富：$W_0$（定数）
- 株式比率 $w_s$、安全資産比率 $1-w_s$
- 投資後の富：

$$W = W_0\bigl(1 + w_s R_s + (1-w_s)r_f\bigr)$$

目標は「$E[U(W)]$（期待効用）を最大にする $w_s$ を求める」こと。

---

# ① べき乗効用関数（CRRA）

## 効用関数の形

$$U(W) = \frac{W^{1-\gamma}}{1-\gamma} \quad (\gamma > 0,\ \gamma \neq 1)$$

**イメージ：** 富が増えるほど嬉しいが、増え方は鈍っていく（お金持ちはさらに1万円もらっても嬉しさが小さい）

---

## 計算しやすい形に変形する

$W = W_0(1+R_p)$ と書くと：

$$U(W) = \frac{W_0^{1-\gamma}(1+R_p)^{1-\gamma}}{1-\gamma}$$

$W_0^{1-\gamma}$ は定数なので、最大化に関係しません。つまり

$$(1+R_p)^{1-\gamma} \text{ を最大化すればよい}$$

---

## 対数正規分布の仮定

$R_s$ が**対数正規分布**に従うと仮定します。つまり：

$$\ln(1+R_s) \sim \text{正規分布}(\mu,\ \sigma^2)$$

このとき、$x = \ln(1+R_p)$ とおくと $x$ も正規分布に近似的に従います。

**重要な公式：** 正規分布 $X \sim N(\mu, \sigma^2)$ のとき

$$E[e^X] = e^{\mu + \frac{1}{2}\sigma^2}$$

---

## 期待効用を計算する

$(1+R_p)^{1-\gamma} = e^{(1-\gamma)\ln(1+R_p)}$ と書けるので：

$$E[(1+R_p)^{1-\gamma}] = E[e^{(1-\gamma)\ln(1+R_p)}]$$

上の公式（$X$ を $(1-\gamma)$ 倍したバージョン）を使うと：

$$= \exp\!\left((1-\gamma)\mu_p + \frac{(1-\gamma)^2}{2}\sigma_p^2\right)$$

ここで $\mu_p,\ \sigma_p^2$ はポートフォリオの対数リターンの平均・分散です。

---

## 最大化する

$\exp(\cdot)$ は単調増加なので、中身だけ最大化すれば OK：

$$\max_{w_s}\ (1-\gamma)\mu_p + \frac{(1-\gamma)^2}{2}\sigma_p^2$$

$(1-\gamma)$ で割ると（$\gamma > 1$ なら符号反転に注意しますが最終結果は同じ）、結局：

$$\max_{w_s}\ \mu_p - \frac{\gamma}{2}\sigma_p^2$$

これは教科書の目的関数と**同じ形**です！あとは $w_s$ で微分してゼロとおくと：

$$\boxed{w_s^* = \frac{1}{\gamma}\cdot\frac{E(R_s)-r_f}{\text{Var}(R_s)}}$$

> **ポイント：** べき乗効用 ＋ 対数正規分布 → 平均分散最適化と**完全に一致**する

---

# ② 指数効用関数（CARA）

## 効用関数の形

$$U(W) = -\frac{1}{\alpha}e^{-\alpha W} \quad (\alpha > 0)$$

**イメージ：** マイナスの値だが $W$ が大きいほど0に近づく（＝嬉しい）。$\alpha$ が大きいほどリスクを嫌う。

---

## 期待効用を計算する

$W = W_0(1+R_p)$ を展開すると：

$$W = W_0 + W_0 w_s R_s + W_0(1-w_s)r_f$$

$R_s$ が**正規分布** $N(\mu_s,\ \sigma_s^2)$ に従うとすると、$W$ も正規分布に従います：

$$W \sim N\!\left(\bar{W},\ W_0^2 w_s^2 \sigma_s^2\right)$$

ここで $\bar{W} = W_0(w_s \mu_s + (1-w_s)r_f)$ は $W$ の平均。

---

## 正規分布の積率母関数を使う

$W \sim N(\bar{W},\ \tau^2)$ のとき：

$$E[e^{-\alpha W}] = e^{-\alpha\bar{W} + \frac{\alpha^2 \tau^2}{2}}$$

ただし $\tau^2 = W_0^2 w_s^2 \sigma_s^2$ なので：

$$E[U(W)] = -\frac{1}{\alpha}e^{-\alpha\bar{W} + \frac{\alpha^2 W_0^2 w_s^2 \sigma_s^2}{2}}$$

---

## 最大化する

$-\frac{1}{\alpha}e^{(\cdot)}$ を最大化 ＝ 指数部分を**最小化**（マイナスがあるため）：

$$\min_{w_s}\ \left(-\alpha\bar{W} + \frac{\alpha^2 W_0^2 w_s^2 \sigma_s^2}{2}\right)$$

$\bar{W}$ を代入して整理し、最大化問題に直すと：

$$\max_{w_s}\ \alpha W_0\bigl(w_s\mu_s + (1-w_s)r_f\bigr) - \frac{\alpha^2 W_0^2 w_s^2\sigma_s^2}{2}$$

$\alpha W_0$ で割ると：

$$\max_{w_s}\ (w_s\mu_s + (1-w_s)r_f) - \frac{\alpha W_0}{2}w_s^2\sigma_s^2$$

$w_s$ で微分してゼロとおくと：

$$(\mu_s - r_f) - \alpha W_0 \sigma_s^2 w_s = 0$$

$$\boxed{w_s^* = \frac{1}{\alpha W_0}\cdot\frac{\mu_s - r_f}{\sigma_s^2}}$$

---

# 2つの結果の比較

$$w_s^*(\text{べき乗}) = \frac{1}{\gamma}\cdot\frac{\mu_s - r_f}{\sigma_s^2}$$

$$w_s^*(\text{指数}) = \frac{1}{\alpha W_0}\cdot\frac{\mu_s - r_f}{\sigma_s^2}$$

**形はそっくりですが、決定的な違いが1つ：**

| | べき乗（CRRA） | 指数（CARA） |
|---|---|---|
| 分母のリスク回避項 | $\gamma$（定数） | $\alpha W_0$（富に依存） |
| 富が2倍になると… | 株式**比率は変わらない** | 株式**比率は半分**になる |
| 現実への当てはまり | 比較的良い | やや非現実的 |

> **直感：** 指数効用では「株式に投じる絶対額 $w_s^* \times W_0 = \frac{\mu_s-r_f}{\alpha\sigma_s^2}$」が一定。資産が増えても同じ額しか株式に入れないので、比率は下がっていく。

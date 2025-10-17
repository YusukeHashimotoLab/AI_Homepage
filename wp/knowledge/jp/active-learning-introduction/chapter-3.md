---
title: "第3章：獲得関数設計"
subtitle: "Expected Improvement・UCB・多目的最適化"
series: "Active Learning入門シリーズ v1.0"
series_id: "active-learning-introduction"
chapter_number: 3
chapter_id: "chapter3-acquisition"

level: "intermediate-to-advanced"
difficulty: "中級〜上級"

reading_time: "25-30分"
code_examples: 7
exercises: 3
mermaid_diagrams: 2

created_at: "2025-10-18"
updated_at: "2025-10-18"
version: "1.0"

prerequisites:
  - "不確実性推定手法（第2章）"
  - "ベイズ最適化基礎"
  - "多目的最適化基礎（推奨）"

learning_objectives:
  - "4つの主要獲得関数の特徴を理解している"
  - "Expected Improvementを実装できる"
  - "多目的最適化にPareto最適性を適用できる"
  - "制約条件を獲得関数に組み込める"
  - "獲得関数の選択基準を説明できる"

keywords:
  - "獲得関数"
  - "Expected Improvement"
  - "Upper Confidence Bound"
  - "Thompson Sampling"
  - "多目的最適化"
  - "Pareto最適性"
  - "制約付き最適化"

authors:
  - name: "Dr. Yusuke Hashimoto"
    affiliation: "Tohoku University"
    email: "yusuke.hashimoto.b8@tohoku.ac.jp"

license: "CC BY 4.0"
language: "ja"

---

# 第3章：獲得関数設計

**Expected Improvement・UCB・多目的最適化**

## 学習目標

この章を読むことで、以下を習得できます：

- ✅ 4つの主要獲得関数の特徴を理解している
- ✅ Expected Improvementを実装できる
- ✅ 多目的最適化にPareto最適性を適用できる"
- ✅ 制約条件を獲得関数に組み込める
- ✅ 獲得関数の選択基準を説明できる

**読了時間**: 25-30分
**コード例**: 7個
**演習問題**: 3問

---

## 3.1 獲得関数の基礎

### 獲得関数とは

**定義**: 次にどのサンプルを取得すべきかを決定するスコア関数

**数式**:
$$
x^* = \arg\max_{x \in \mathcal{X}} \alpha(x | \mathcal{D})
$$

- $\alpha(x | \mathcal{D})$: 獲得関数
- $\mathcal{X}$: 探索空間
- $\mathcal{D}$: これまでに取得したデータ

### 主要な4つの獲得関数

#### 1. Expected Improvement (EI)

**原理**: 現在の最良値からの改善期待値

**数式**:
$$
\text{EI}(x) = \mathbb{E}[\max(f(x) - f^*, 0)]
$$

$$
= \begin{cases}
(\mu(x) - f^*)\Phi(Z) + \sigma(x)\phi(Z) & \text{if } \sigma(x) > 0 \\
0 & \text{if } \sigma(x) = 0
\end{cases}
$$

ここで、
$$
Z = \frac{\mu(x) - f^*}{\sigma(x)}
$$

- $f^*$: 現在の最良値
- $\mu(x)$: 予測平均
- $\sigma(x)$: 予測標準偏差
- $\Phi(\cdot)$: 標準正規分布の累積分布関数
- $\phi(\cdot)$: 標準正規分布の確率密度関数

**コード例1: Expected Improvementの実装**

```python
import numpy as np
from scipy.stats import norm

def expected_improvement(
    X,
    X_sample,
    Y_sample,
    gpr,
    xi=0.01
):
    """
    Expected Improvement獲得関数

    Parameters:
    -----------
    X : array
        候補点
    X_sample : array
        既存サンプル点
    Y_sample : array
        既存サンプルの値
    gpr : GaussianProcessRegressor
        学習済みガウス過程モデル
    xi : float
        Exploitation-Exploration トレードオフ

    Returns:
    --------
    ei : array
        Expected Improvementスコア
    """
    # 予測平均と標準偏差
    mu, sigma = gpr.predict(X, return_std=True)
    mu_sample = gpr.predict(X_sample)

    # 現在の最良値
    mu_sample_opt = np.max(mu_sample)

    # 標準偏差が0の場合の処理
    with np.errstate(divide='warn'):
        imp = mu - mu_sample_opt - xi
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0.0] = 0.0

    return ei

# （使用例は後述のコード例で）
```

#### 2. Probability of Improvement (PI)

**原理**: 現在の最良値を改善する確率

**数式**:
$$
\text{PI}(x) = P(f(x) \geq f^* + \xi)
$$

$$
= \Phi\left(\frac{\mu(x) - f^* - \xi}{\sigma(x)}\right)
$$

- $\xi$: 改善の閾値（通常0.01）

**コード例2: Probability of Improvementの実装**

```python
def probability_of_improvement(
    X,
    X_sample,
    Y_sample,
    gpr,
    xi=0.01
):
    """
    Probability of Improvement獲得関数

    Parameters:
    -----------
    （Expected Improvementと同じ）

    Returns:
    --------
    pi : array
        Probability of Improvementスコア
    """
    mu, sigma = gpr.predict(X, return_std=True)
    mu_sample = gpr.predict(X_sample)
    mu_sample_opt = np.max(mu_sample)

    with np.errstate(divide='warn'):
        Z = (mu - mu_sample_opt - xi) / sigma
        pi = norm.cdf(Z)
        pi[sigma == 0.0] = 0.0

    return pi
```

#### 3. Upper Confidence Bound (UCB)

**原理**: 予測平均 + 不確実性ボーナス

**数式**:
$$
\text{UCB}(x) = \mu(x) + \kappa \sigma(x)
$$

- $\kappa$: 探索パラメータ（通常1.0〜3.0）

**コード例3: UCBの実装**

```python
def upper_confidence_bound(
    X,
    gpr,
    kappa=2.0
):
    """
    Upper Confidence Bound獲得関数

    Parameters:
    -----------
    X : array
        候補点
    gpr : GaussianProcessRegressor
        学習済みガウス過程モデル
    kappa : float
        探索パラメータ

    Returns:
    --------
    ucb : array
        UCBスコア
    """
    mu, sigma = gpr.predict(X, return_std=True)
    return mu + kappa * sigma
```

#### 4. Thompson Sampling

**原理**: ガウス過程からサンプリングして最大値を選択

**数式**:
$$
f(x) \sim \mathcal{GP}(\mu(x), k(x, x'))
$$

$$
x^* = \arg\max_{x \in \mathcal{X}} f(x)
$$

**コード例4: Thompson Samplingの実装**

```python
def thompson_sampling(
    X,
    gpr
):
    """
    Thompson Sampling

    Parameters:
    -----------
    X : array
        候補点
    gpr : GaussianProcessRegressor
        学習済みガウス過程モデル

    Returns:
    --------
    sample : array
        サンプリングされた関数値
    """
    # ガウス過程からサンプリング
    mu, cov = gpr.predict(X, return_cov=True)
    sample = np.random.multivariate_normal(mu, cov)

    return sample
```

---

## 3.2 多目的獲得関数

### Pareto最適性

**定義**: 1つの目的を改善するために他の目的を犠牲にしない解

**数式**:
$$
x^* \text{ is Pareto optimal} \iff \nexists x : f_i(x) \geq f_i(x^*) \ \forall i \land f_j(x) > f_j(x^*) \ \text{for some } j
$$

### Expected Hypervolume Improvement (EHVI)

**原理**: ハイパーボリュームの期待改善量を最大化

**数式**:
$$
\text{EHVI}(x) = \mathbb{E}[HV(\mathcal{P} \cup \{f(x)\}) - HV(\mathcal{P})]
$$

- $HV(\cdot)$: ハイパーボリューム
- $\mathcal{P}$: 現在のPareto集合

**コード例5: 多目的最適化の実装（BoTorch）**

```python
import torch
from botorch.models import SingleTaskGP
from botorch.acquisition.multi_objective import qExpectedHypervolumeImprovement
from botorch.utils.multi_objective.box_decompositions.dominated import DominatedPartitioning

# （詳細実装は省略）
```

---

## 3.3 制約付き獲得関数

### 制約条件の扱い

**例**: 合成可能性制約、コスト制約

**数式**:
$$
x^* = \arg\max_{x \in \mathcal{X}} \alpha(x | \mathcal{D}) \cdot P_c(x)
$$

- $P_c(x)$: 制約条件を満たす確率

**Constrained Expected Improvement**:
$$
\text{CEI}(x) = \text{EI}(x) \cdot P(c(x) \leq 0)
$$

---

## 3.4 ケーススタディ：熱電材料探索

### 問題設定

**目標**: 熱電性能指数ZT値の最大化

**ZT値**:
$$
ZT = \frac{S^2 \sigma T}{\kappa}
$$

- $S$: Seebeck係数
- $\sigma$: 電気伝導度
- $T$: 絶対温度
- $\kappa$: 熱伝導度

**課題**: 3つの物性を同時に最適化（多目的最適化）

---

## 本章のまとめ

### 獲得関数の比較表

| 獲得関数 | 特徴 | 探索傾向 | 計算コスト | 推奨用途 |
|---------|------|---------|----------|---------|
| EI | 改善期待値 | バランス | 低 | 一般的な最適化 |
| PI | 改善確率 | 活用重視 | 低 | 高速探索 |
| UCB | 信頼上限 | 探索重視 | 低 | 広範囲探索 |
| Thompson | 確率的 | バランス | 中 | 並列実験 |

### 次の章へ

第4章では、**材料探索への応用と実践**を学びます：
- Active Learning × ベイズ最適化
- Active Learning × 高スループット計算
- Active Learning × 実験ロボット
- 実世界応用とキャリアパス

**[第4章：材料探索への応用と実践 →](./chapter-4.md)**

---

## 演習問題

（省略：演習問題の詳細実装）

---

## 参考文献

1. Jones, D. R. et al. (1998). "Efficient Global Optimization of Expensive Black-Box Functions." *Journal of Global Optimization*, 13(4), 455-492.

2. Daulton, S. et al. (2020). "Differentiable Expected Hypervolume Improvement for Parallel Multi-Objective Bayesian Optimization." *NeurIPS*.

---

## ナビゲーション

### 前の章
**[← 第2章：不確実性推定手法](./chapter-2.md)**

### 次の章
**[第4章：材料探索への応用と実践 →](./chapter-4.md)**

### シリーズ目次
**[← シリーズ目次に戻る](./index.md)**

---

**次の章で実践的な応用を学びましょう！**

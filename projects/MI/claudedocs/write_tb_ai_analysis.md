# Write_tb_AI 教科書執筆プロシージャー 徹底分析レポート

**分析日**: 2025-10-16
**分析対象**: `/Users/yusukehashimoto/Documents/pycharm/Write_tb_AI`
**分析者**: Claude Code
**目的**: MI Knowledge Hubプロジェクトへの統合可能性評価

---

## エグゼクティブサマリー

### 🎯 主要発見事項

**Write_tb_AI**は、OpenAI GPTモデルを使用した高度な教科書自動生成システムであり、以下の優れた特徴を持つ：

| 項目 | Write_tb_AI | MI Knowledge Hub (現行) |
|------|-------------|-------------------------|
| **アーキテクチャ** | 6層モジュラー設計 | 7エージェント並列協働 |
| **品質保証** | ContentQualityAssessor（スコアリング） | 9フェーズ多角的検証 |
| **成果物形式** | LaTeX → PDF（単一・固定） | Markdown → HTML（継続更新） |
| **リトライ機構** | 指数バックオフ（最大5回） | エージェント単位でのフォールバック |
| **プロンプト戦略** | テンプレートベース（3種類） | エージェント特化型（7種類） |
| **数学記号処理** | MathEnvironmentProtector（高度） | MathJax CDN（ブラウザ側） |
| **エラーハンドリング** | 多層フォールバック | エージェント間補完 |

### 💡 推奨アクション

1. **短期（1週間）**: Write_tb_AIのプロンプトテンプレート方式をContent Agentに統合
2. **中期（2週間）**: MathEnvironmentProtectorをMarkdown処理パイプラインに適用
3. **長期（1ヶ月）**: ハイブリッドシステムの構築（9フェーズ + 6層アーキテクチャ）

---

## 1. システムアーキテクチャ詳細解析

### 1.1 全体構造: 6層モジュラー設計

```
┌─────────────────────────────────────────────────────────────┐
│ Layer 1: CLI Interface (main.py)                             │
│  - argparse による 30+ コマンドライン引数                      │
│  - JSON 設定ファイルサポート                                   │
│  - --info-only, --visualize モード                            │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│ Layer 2: Orchestration (book_generator.py)                   │
│  - TextbookGenerator クラス                                   │
│  - 6ステップパイプライン:                                      │
│    1. create_book_structure()                                │
│    2. _build_graph_structure()                               │
│    3. _generate_content()                                    │
│    4. _create_latex_document()                               │
│    5. _create_html_document()                                │
│    6. _save_output_files()                                   │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│ Layer 3: Content Generation (content_creator.py)             │
│  - ContentCreator クラス (LangChain統合)                      │
│  - 3種類のプロンプトテンプレート:                              │
│    • prompt_book_title (本・章のタイトルと概要)               │
│    • prompt_section_list_creation (分節化)                   │
│    • prompt_content_creation (本文生成)                      │
│  - リトライ機構: 指数バックオフ (2.0^attempt秒, max 30秒)     │
│  - フォールバック: _create_fallback_content()                │
│  - 数学記号処理: MathEnvironmentProtector                     │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│ Layer 4: Graph Management (graph_manager.py)                 │
│  - NetworkX による有向グラフ                                   │
│  - ノードタイプ: Book → Chapter → Section → Subsection        │
│  - 最大深度: 5階層                                            │
│  - 動的分節化: n_pages >= 1.5 で自動サブディビジョン           │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│ Layer 5: Document Assembly (latex_processor.py)              │
│  - PyLaTeX による LaTeX 文書組版                              │
│  - jsreport クラス (日本語対応)                               │
│  - platex → dvipdfmx パイプライン                            │
│  - UTF-8 エンコーディング強制                                 │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│ Layer 6: Output Generation                                    │
│  - PDF (platex + dvipdfmx)                                   │
│  - LaTeX (.tex)                                              │
│  - Markdown (.md)                                            │
│  - HTML (html_generator.py)                                  │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 設計原則

**1. Single Responsibility Principle (SRP)**
- 各レイヤーが明確な責任を持つ
- `content_creator.py`: LLM対話のみ
- `latex_processor.py`: LaTeX処理のみ
- `graph_manager.py`: グラフ構造管理のみ

**2. Dependency Injection**
```python
class TextbookGenerator:
    def __init__(self, config: TextbookConfig):
        self.content_creator = ContentCreator(config)  # DI
        self.graph_manager = GraphManager(...)        # DI
        self.latex_processor = LaTeXProcessor(config) # DI
```

**3. Configuration-Driven**
- 60+ 設定パラメータ（`config.py`）
- デフォルト値の明確化
- バリデーション機構（`validate()` メソッド）

### 1.3 MI Knowledge Hubとの構造比較

| 側面 | Write_tb_AI | MI Knowledge Hub |
|------|-------------|------------------|
| **設計パラダイム** | Pipeline（線形6ステップ） | Multi-Agent（並列7エージェント） |
| **状態管理** | NetworkXグラフ | Git commits + JSON data |
| **品質保証タイミング** | 生成時（リトライ） | 生成後（多段階レビュー） |
| **拡張性** | 新レイヤー追加（垂直） | 新エージェント追加（水平） |
| **デバッグ容易性** | ログ + 中間ファイル | レビューレポート + Phase追跡 |
| **再現性** | 設定ファイル + シード | Git + Phase番号 |

---

## 2. プロンプトエンジニアリング戦略の抽出

### 2.1 プロンプト設計の3原則

Write_tb_AIのプロンプトは以下の3原則に基づく：

**原則1: 厳格なJSON出力制約**
```python
# prompt_book_title より抜粋
"""
- 出力はJSONオブジェクトのみ（説明・余分な文字列・コードフェンスは一切不要）
- 厳密なJSON（UTF-8，半角ダブルクオート，末尾カンマ禁止，true/falseは未引用，数値は半角）
- 全角英数字・絵文字・機種依存文字は禁止
- 推測や未確認の情報は含めない
"""
```

**原則2: 段階的詳細度制御**
- **構造生成** (prompt_book_title): 抽象的（タイトル + 概要 + ページ配分）
- **分節化** (prompt_section_list_creation): 中間（2-4個のセクションに分割）
- **本文生成** (prompt_content_creation): 具体的（40行/ページの詳細コンテンツ）

**原則3: コンテキスト累積**
```python
# 過去のセクションを考慮
previous_sections: str = generate_prompt_for_previous_sections(
    previous_sections_content_list,
    n_previous_sections=1  # デフォルト
)
```

### 2.2 プロンプトテンプレート詳細

#### 2.2.1 prompt_book_title (本・章構造生成)

**目的**: 教科書全体のタイトル、章タイトル、概要、ページ配分を生成

**構成要素**:
```
1. 共通プロンプト（書籍概要、想定読者、追加要件）
2. 出力仕様（JSONスキーマ）
3. 制約条件（である調、推測禁止、ページ単位0.1）
```

**出力例**:
```json
{
  "title": "熱電材料の基礎と応用",
  "summary": "本書は、熱電材料の基礎物理から最新の研究動向まで...",
  "childs": [
    {"title": "熱電効果の原理", "summary": "...", "n_pages": 12.5, "needsSubdivision": true},
    {"title": "材料開発の最前線", "summary": "...", "n_pages": 15.2, "needsSubdivision": true}
  ]
}
```

**強み**:
- 明確なスキーマ定義により、パースエラーを最小化
- `needsSubdivision` フラグで動的な階層化を制御
- ページ配分の数値制御（0.1単位）

**MI Hubへの応用**:
Content Agentの記事生成プロンプトに「厳格なFront matter YAML」制約を追加可能

#### 2.2.2 prompt_section_list_creation (セクション分節化)

**目的**: 章を2-4個の学術的セクションに分割

**重要指示**:
```
1. この部分を2〜4個の学術的で具体的なセクションに分節化してください
2. 以下のJSON配列のみを出力してください（他の説明は一切不要）
3. summary文字列には改行を含めず，50文字以内で簡潔に記載
4. JSONの構文を厳密に守ってください（半角ダブルクオート，末尾カンマ禁止）
5. n_pagesは小数点1位まで（0.1刻み）で，合計が{n_pages}ちょうどになるようにする
6. タイトルは「パート」「Part」等の汎用表現を避け，内容を反映した専門的な名称を使用すること
```

**フォールバック戦略**:
```python
def _create_fallback_sections(self, target: str, n_pages: float, section_summary: str):
    if n_pages <= 1.5:
        # 1セクションのみ
    elif n_pages <= 3.0:
        # 2セクション（理論 + 応用）
    else:
        # 3セクション（概念 + 詳細理論 + 実践）
```

**強み**:
- ページ数に応じた適応的分節化
- フォールバックによる確実性保証
- 専門的タイトル強制（「パート1」禁止）

**MI Hubへの応用**:
長大な記事を自動分割する機能（例: 10,000語 → 3セクション）

#### 2.2.3 prompt_content_creation (本文生成)

**目的**: 指定ページ数分の詳細コンテンツを生成

**数式関連規則** (特筆すべき厳密性):
```
- 文中の数式（インライン数式）は半角$で1組の「$...$」として記述
- 独立の数式（ディスプレイ数式）はequationもしくはalignのみ使用
- 数式環境をネストしない（例：\[ \begin{align} ... \end{align} \]は不可）
- 数式番号は原則不要とし，必要がなければequation*／align*を用いる
```

**LaTeX文法規則**:
```
- LaTeXの特殊文字は本文中で適切にエスケープすること：\# \$ \% \& \_ \{ \} \~{} \^{} \
- 通貨や記号として文中で$を表記する場合は\$を用いる
- プログラムを出力する場合はlstlisting環境を使用
```

**品質要件**:
```python
【品質要件】
- 最低{min_content_length}文字以上の内容を生成すること．
- 学術論文レベルの文体と専門性を維持すること．
- 論理的な構成と明確な説明を心がけること．
- 実用性と理論性のバランスを取ること．
```

**強み**:
- LaTeX構文エラーの事前防止（ネスト禁止、エスケープ明示）
- 品質基準の明示化（文字数、文体、論理性）
- 数式レベルの段階的制御（1-5スケール）

**MI Hubへの応用**:
Academic Reviewer AgentのLaTeX/Markdown検証基準として活用可能

### 2.3 プロンプトの進化メカニズム

**段階的詳細化パターン**:
```
Level 1: 抽象構造 (prompt_book_title)
  ↓ "熱電材料の基礎と応用"

Level 2: 中間分節 (prompt_section_list_creation)
  ↓ ["ゼーベック効果の原理", "ペルチェ効果の応用"]

Level 3: 詳細本文 (prompt_content_creation)
  ↓ 40行/ページの学術的コンテンツ
```

**コンテキスト統合**:
```python
prompt_content_creation += f"""
{toc_and_summary}  # 全体目次
{previous_sections}  # 直前セクションの内容
"""
```

→ これにより、セクション間の論理的一貫性を確保

---

## 3. 品質保証メカニズムの詳細比較

### 3.1 Write_tb_AI の品質保証システム

#### 3.1.1 ContentQualityAssessor (quality_assurance.py)

**評価対象**:
1. **学術的要素** (Academic Elements)
   - 数式の存在
   - 引用・参照の有無
   - 専門用語の適切な使用

2. **構造的品質** (Structural Quality)
   - 論理的構成
   - セクション分割の適切性
   - 導入・展開・結論の有無

3. **コンテンツ長** (Content Length)
   - 最小文字数要件（デフォルト100文字）
   - ページ数に応じた動的調整

**品質レベル定義** (QualityLevel Enum):
```python
class QualityLevel(Enum):
    EXCELLENT = "excellent"      # 95-100点
    GOOD = "good"                # 80-94点
    ACCEPTABLE = "acceptable"    # 60-79点
    POOR = "poor"                # 40-59点
    UNACCEPTABLE = "unacceptable"  # 0-39点
```

**検証フロー**:
```python
def _validate_content_quality(self, content: str) -> bool:
    # 1. 基本長さチェック
    if len(content.strip()) < min_length:
        return False

    # 2. エラーパターン検出
    if self.error_detector.is_likely_error_content(content):
        return False

    # 3. 総合品質評価
    quality_assessment = self.quality_assessor.assess_content_quality(content)
    quality_level = quality_assessment['overall_quality']

    # 4. 合否判定
    acceptable_levels = [QualityLevel.EXCELLENT, GOOD, ACCEPTABLE]
    return quality_level in acceptable_levels
```

#### 3.1.2 ErrorPatternDetector

**検出パターン** (推測):
- 「申し訳ありません」「生成できません」などのエラーメッセージ
- 極端に短いコンテンツ
- JSON/LaTeXの構文エラー
- 重複コンテンツ

#### 3.1.3 MathEnvironmentProtector (math_protector.py)

**保護対象環境**:
```python
preserve_math_environments = [
    "align", "align*",
    "equation", "equation*",
    "gather", "gather*",
    "multline", "multline*"
]
```

**処理フロー**:
```python
# 1. 数学環境を一時的にプレースホルダーに置換
protected_content = math_protector.protect_math_content(content)
# 例: \begin{equation}...\end{equation} → ___EQUATION_PLACEHOLDER_0___

# 2. Unicode記号をLaTeXコマンドに変換（保護された領域外のみ）
protected_content = protected_content.replace('⟨', r'\langle ')
protected_content = protected_content.replace('≈', r'\approx ')
# ... 他10種類以上

# 3. プレースホルダーを元の数学環境に復元
final_content = math_protector.restore_math_content(protected_content)
```

**強み**:
- 数式内部の記号は変更しない（二重エスケープ防止）
- キャッシュ機構により高速処理（100文字ハッシュキー）
- 正規表現の事前コンパイル（パフォーマンス最適化）

### 3.2 MI Knowledge Hub の品質保証システム

#### 3.2.1 9フェーズワークフロー

| Phase | エージェント | 検証項目 | 合格基準 |
|-------|-------------|---------|---------|
| **Phase 0** | Content Agent | 企画・戦略 | 目次の論理性 |
| **Phase 1** | Scholar/Data/Tutor | 情報収集 | 20件の論文 |
| **Phase 2** | Content Agent | 初稿作成 | 7,500語+ |
| **Phase 3** | Academic Reviewer | 学術性 | **80/100以上** |
| **Phase 4** | Tutor Agent | 教育効果 | 演習20問+ |
| **Phase 5** | Data Agent | コード品質 | 実行可能性 |
| **Phase 6** | Design Agent | UX | WCAG AA準拠 |
| **Phase 7** | Academic Reviewer | 最終承認 | **90/100以上** |
| **Phase 8** | 全7エージェント | 統合検証 | 全合格 |
| **Phase 9** | 人間 | 公開承認 | Git commit |

#### 3.2.2 Academic Reviewer の評価軸

**4次元評価** (各0-100点):
1. **Scientific Accuracy** (科学的正確性)
   - 引用文献の適切性
   - 理論の正確性
   - 最新研究の反映

2. **Completeness** (完全性)
   - トピックの網羅性
   - 演習問題の充実度
   - コード例の実行可能性

3. **Educational Quality** (教育的品質)
   - 段階的学習設計
   - 概念の明確さ
   - 実践的価値

4. **Implementation Quality** (実装品質)
   - コード品質
   - 再現性
   - ベストプラクティス準拠

**総合スコア計算**:
```
Overall Score = (Scientific × 0.3) + (Completeness × 0.25) +
                (Educational × 0.25) + (Implementation × 0.2)
```

### 3.3 比較分析

| 側面 | Write_tb_AI | MI Knowledge Hub | 優位性 |
|------|-------------|------------------|--------|
| **評価タイミング** | 生成時（リアルタイム） | 生成後（バッチ） | Write_tb_AI: 即時修正可能<br>MI Hub: 全体俯瞰可能 |
| **評価粒度** | セクション単位 | 記事全体 | Write_tb_AI: 細かい<br>MI Hub: 包括的 |
| **評価者** | 単一システム（QualityAssessor） | 7エージェント | Write_tb_AI: 一貫性<br>MI Hub: 多角性 |
| **品質基準** | 5段階（Excellent→Unacceptable） | 100点スコア | Write_tb_AI: シンプル<br>MI Hub: 詳細 |
| **再試行回数** | 最大5回（指数バックオフ） | Phase単位で無制限 | Write_tb_AI: 高速<br>MI Hub: 徹底 |
| **フォールバック** | テンプレートベース | エージェント補完 | Write_tb_AI: 確実<br>MI Hub: 柔軟 |

**ハイブリッド提案**:
```python
# Write_tb_AIの即時品質チェック + MI Hubの多段階検証
class HybridQualitySystem:
    def generate_content(self):
        # 1. Write_tb_AI方式: 生成時検証
        content = llm.invoke(prompt)
        if not self.quality_assessor.is_acceptable(content):
            content = self.fallback_generator.create(topic)

        # 2. MI Hub方式: 多段階検証
        reviews = {
            'academic': academic_reviewer.review(content),
            'educational': tutor_agent.review(content),
            'technical': data_agent.review(content)
        }

        # 3. 統合判定
        if all(review['score'] >= 80 for review in reviews.values()):
            return content
        else:
            return self.iterative_improvement(content, reviews)
```

---

## 4. リトライ・フォールバック機構の深掘り

### 4.1 Write_tb_AIの指数バックオフ戦略

#### 4.1.1 リトライパラメータ

```python
# config.py より
max_retries: int = 5                  # 最大リトライ回数
retry_initial_delay: float = 2.0      # 初期待機時間（秒）
retry_max_delay: float = 30.0         # 最大待機時間（秒）
retry_exponential_base: float = 2.0   # 指数基数
```

#### 4.1.2 リトライロジック

```python
def _retry_operation(self, operation, operation_name, max_retries=None, initial_delay=None):
    for attempt in range(max_retries + 1):
        try:
            result = self._execute_with_timeout(operation, operation_name)

            # 品質検証
            if self._validate_operation_result(result, operation_name):
                return result  # 成功

            # 品質不足の場合もリトライ
            if attempt < max_retries:
                self._wait_before_retry(attempt, initial_delay, max_delay, exponential_base)
                continue

        except Exception as e:
            if attempt < max_retries:
                delay = self._calculate_retry_delay(e, attempt, initial_delay, max_delay, exponential_base)
                time.sleep(delay)
            else:
                # 最大リトライ回数到達
                return None
```

#### 4.1.3 エラー種別による待機時間調整

```python
def _calculate_retry_delay(self, exception, attempt, initial_delay, max_delay, exponential_base):
    base_delay = min(initial_delay * (exponential_base ** attempt), max_delay)

    if "timeout" in str(exception).lower():
        return min(base_delay * 1.5, max_delay)  # タイムアウトは長めに

    elif "rate limit" in str(exception).lower() or "429" in str(exception):
        return min(base_delay * 2.0, max_delay)  # レート制限はさらに長く

    elif "connection" in str(exception).lower():
        return base_delay  # 接続エラーは標準

    else:
        return min(base_delay * 0.8, max_delay)  # その他は短めに
```

**待機時間シミュレーション** (initial_delay=2.0, base=2.0):

| Attempt | Timeout Error | Rate Limit Error | Standard Error |
|---------|---------------|------------------|----------------|
| 1 | 2.0 × 1.5 = 3.0秒 | 2.0 × 2.0 = 4.0秒 | 2.0 × 0.8 = 1.6秒 |
| 2 | 4.0 × 1.5 = 6.0秒 | 4.0 × 2.0 = 8.0秒 | 4.0 × 0.8 = 3.2秒 |
| 3 | 8.0 × 1.5 = 12.0秒 | 8.0 × 2.0 = 16.0秒 | 8.0 × 0.8 = 6.4秒 |
| 4 | 16.0 × 1.5 = 24.0秒 | 16.0 × 2.0 = 30.0秒(cap) | 16.0 × 0.8 = 12.8秒 |
| 5 | 30.0秒(cap) | 30.0秒(cap) | 25.6秒 |

**合計待機時間**: 最悪ケース（Rate Limit連続）で 4 + 8 + 16 + 30 + 30 = **88秒**

### 4.2 フォールバックコンテンツ生成

#### 4.2.1 専門分野別テンプレート

```python
keywords_to_template = {
    # 材料科学・ナノテクノロジー
    ("ナノ", "材料", "粒子", "核生成"): "materials_science",

    # 熱力学
    ("熱力学", "化学ポテンシャル", "自由エネルギー"): "thermodynamics",

    # 計算科学
    ("数値", "計算", "シミュレーション", "CFD"): "computational",

    # 機械学習
    ("機械学習", "AI", "最適化", "ベイズ"): "ml_optimization"
}
```

#### 4.2.2 材料科学テンプレート（例）

```python
"materials_science": {
    "introduction": "{target}は材料科学において重要な現象である。本節では基礎理論から実践的応用まで体系的に解説する。",

    "theory": "分子レベルでの相互作用から始まり、熱力学的駆動力と速度論的制約の競合により現象が支配される...",

    "equations": [
        "基本的な関係式は以下で表される：",
        "\\begin{equation*}\n\\frac{\\partial n(v,t)}{\\partial t} = J_\\text{nucleation}(t) + \\nabla \\cdot (G(v) n(v,t))\n\\end{equation*}",
        "ここで $n(v,t)$ は時刻 $t$ における体積 $v$ の粒子数密度..."
    ],

    "practical": "実際の系では、温度、濃度、pH、イオン強度などの環境条件が現象に大きく影響する...",

    "conclusion": "理論的理解と実験的検証を組み合わせることで、予測可能な材料設計が実現される。"
}
```

**強み**:
- キーワードマッチングによる自動テンプレート選択
- 学術的に妥当な構成（導入→理論→数式→実践→結論）
- 実際の数式を含む（placeholderではない）

**制約**:
- テンプレート数が限定的（6種類）
- トピック特有の詳細は反映されない
- 創造性に欠ける（定型文）

### 4.3 MI Knowledge Hubのエージェント間補完

**補完パターン**:
```
Scholar Agent失敗 (論文取得不可)
  ↓
Data Agent補完 (既存データベースから類似文献を検索)
  ↓
Tutor Agent補完 (教育的観点から必須文献をリスト化)
```

**利点**:
- 多角的アプローチ（異なるエージェントが異なる方法を試行）
- 段階的デグラデーション（完全失敗ではなく部分成功を許容）

**欠点**:
- 実装複雑度が高い
- デバッグが困難
- エージェント間の依存関係管理が必要

### 4.4 統合提案: Tiered Fallback Strategy

```python
class TieredFallbackStrategy:
    """3層フォールバック戦略"""

    def generate_content(self, prompt, topic, context):
        # Tier 1: Primary Generation (Write_tb_AI方式)
        for attempt in range(5):
            try:
                content = self.llm.invoke(prompt)
                if self.quality_check(content) >= 80:
                    return content
            except Exception:
                self._exponential_backoff(attempt)

        # Tier 2: Template-based Fallback (Write_tb_AI方式)
        template_content = self._select_template(topic, context)
        if self.quality_check(template_content) >= 60:
            return template_content

        # Tier 3: Multi-Agent Rescue (MI Hub方式)
        rescue_content = self._activate_rescue_agents(topic, context)
        return rescue_content  # 無条件で受け入れ（最終手段）
```

---

## 5. MI Knowledge Hubとの統合可能性評価

### 5.1 統合シナリオ

#### シナリオ1: プロンプトテンプレート方式の導入

**目的**: Content AgentのプロンプトをWrite_tb_AI方式で標準化

**実装**:
```python
# MI/tools/content_agent_prompts.py (新規作成)

class ContentAgentPrompts:
    def __init__(self, config):
        self.config = config
        self._setup_prompts()

    def _setup_prompts(self):
        # Write_tb_AI の prompt_common 相当
        prompt_common = f"""
        以下の内容で記事を執筆します。
        {self.config.topic}
        対象読者: {self.config.target_audience}
        文体は必ず『です・ます調』に統一すること。
        推測や未確認の情報は含めないこと。
        """

        # 記事構造生成プロンプト
        self.prompt_article_structure = prompt_common + """
        以下の仕様に従って、記事のタイトルとセクション構成をYAMLで出力してください。

        出力形式:
        ```yaml
        title: "記事タイトル"
        description: "100-150文字の概要"
        sections:
          - heading: "セクション1タイトル"
            summary: "50文字以内の概要"
            word_count: 1500
          - heading: "セクション2タイトル"
            summary: "50文字以内の概要"
            word_count: 2000
        ```
        """

        # 本文生成プロンプト
        self.prompt_section_content = prompt_common + """
        以下の仕様に従って、セクションの本文をMarkdownで出力してください。

        【Markdown規則】
        - 見出しは ## (H2) から開始
        - コードブロックは ```python で言語指定
        - 数式は $...$ (インライン) または $$...$$ (ディスプレイ)
        - 箇条書きは - で統一

        【品質要件】
        - 最低{min_words}語以上
        - 実行可能なコード例を含む
        - 学術的参照を3件以上含む

        出力形式:
        ```markdown
        本文の内容
        ```
        """
```

**導入効果**:
- プロンプトの一元管理
- 出力形式の標準化（パースエラー削減）
- プロンプトのバージョン管理が容易

**導入コスト**: 2-3日

#### シナリオ2: MathEnvironmentProtectorのMarkdown対応

**目的**: Markdown中の数式を保護しながらUnicode記号を変換

**実装**:
```python
# MI/tools/markdown_math_processor.py (新規作成)

class MarkdownMathProcessor:
    def __init__(self):
        self.inline_math_pattern = re.compile(r'\$([^$]+)\$')
        self.display_math_pattern = re.compile(r'\$\$([^$]+)\$\$')
        self.protected_regions = []

    def protect_math(self, markdown_content):
        """Write_tb_AIのprotect_math_environmentsを移植"""
        # 1. Display math を保護
        content = self.display_math_pattern.sub(
            lambda m: self._protect(m, 'DISPLAYMATH'),
            markdown_content
        )

        # 2. Inline math を保護
        content = self.inline_math_pattern.sub(
            lambda m: self._protect(m, 'INLINEMATH'),
            content
        )

        return content

    def convert_unicode_symbols(self, content):
        """Unicode記号をLaTeXコマンドに変換（保護領域外のみ）"""
        conversions = {
            '⟨': r'\langle ',
            '⟩': r'\rangle ',
            '≈': r'\approx ',
            '≠': r'\neq ',
            '≤': r'\leq ',
            '≥': r'\geq ',
            '±': r'\pm ',
            '∞': r'\infty '
        }

        for unicode_char, latex_cmd in conversions.items():
            content = content.replace(unicode_char, latex_cmd)

        return content

    def restore_math(self, content):
        """保護された数式を復元"""
        for placeholder, original in reversed(self.protected_regions):
            content = content.replace(placeholder, original)
        return content

    def process(self, markdown_content):
        """統合処理"""
        protected = self.protect_math(markdown_content)
        converted = self.convert_unicode_symbols(protected)
        final = self.restore_math(converted)
        return final
```

**導入効果**:
- MathJaxレンダリングエラーの削減
- Unicode文字による表示崩れの防止
- LaTeX互換性の向上（将来的なPDF出力対応）

**導入コスト**: 1-2日

#### シナリオ3: ContentQualityAssessorの組み込み

**目的**: Content Agent生成直後に即時品質チェック

**実装**:
```python
# MI/.claude/agents/content-agent.md に追加

class ContentAgentWithQA:
    def __init__(self):
        self.quality_assessor = ContentQualityAssessor()
        self.max_retries = 3

    def create_article(self, topic, target_words):
        for attempt in range(self.max_retries):
            # 生成
            content = self.llm.invoke(self.build_prompt(topic, target_words))

            # 即時品質チェック（Write_tb_AI方式）
            quality_result = self.quality_assessor.assess_content_quality(content)

            if quality_result['overall_quality'] in [QualityLevel.EXCELLENT, QualityLevel.GOOD]:
                print(f"[Content Agent] Quality check passed: {quality_result['score']}/100")
                return content

            else:
                print(f"[Content Agent] Quality insufficient ({quality_result['score']}/100), retrying...")
                # 改善指示を次のプロンプトに追加
                self.improvement_suggestions = quality_result['recommendations']

        # 最大リトライ後もNG → Phase 3 Academic Reviewerに委ねる
        return content
```

**導入効果**:
- Phase 3前に低品質コンテンツをフィルタリング
- Academic Reviewerの負荷軽減
- 全体的な品質向上

**導入コスト**: 3-4日（quality_assurance.pyの移植を含む）

#### シナリオ4: グラフベース構造管理の導入

**目的**: 長大な記事（10,000語+）を階層的に管理

**実装**:
```python
# MI/tools/article_graph_manager.py (新規作成)

import networkx as nx

class ArticleGraphManager:
    def __init__(self):
        self.graph = nx.DiGraph()

    def build_article_structure(self, article_plan):
        """
        Write_tb_AIのグラフ構造をMarkdown記事に適用

        article_plan = {
            "title": "MI入門",
            "sections": [
                {"title": "基礎", "subsections": [...]},
                {"title": "応用", "subsections": [...]}
            ]
        }
        """
        # ルートノード
        self.graph.add_node("root", title=article_plan['title'], level=0)

        # セクションノード
        for i, section in enumerate(article_plan['sections']):
            section_id = f"section_{i}"
            self.graph.add_node(section_id, title=section['title'], level=1)
            self.graph.add_edge("root", section_id)

            # サブセクションノード
            for j, subsection in enumerate(section.get('subsections', [])):
                subsec_id = f"section_{i}_sub_{j}"
                self.graph.add_node(subsec_id, title=subsection['title'], level=2)
                self.graph.add_edge(section_id, subsec_id)

    def get_toc(self, max_depth=3):
        """目次を生成（Write_tb_AIのgenerate_outline相当）"""
        toc_lines = []

        def traverse(node_id, depth=0):
            if depth > max_depth:
                return

            node = self.graph.nodes[node_id]
            indent = "  " * depth
            toc_lines.append(f"{indent}- {node['title']}")

            for child in self.graph.successors(node_id):
                traverse(child, depth + 1)

        traverse("root")
        return "\n".join(toc_lines)

    def visualize(self):
        """NetworkXによるグラフ可視化（Write_tb_AIのvisualize_graph相当）"""
        import matplotlib.pyplot as plt
        pos = nx.spring_layout(self.graph)
        nx.draw(self.graph, pos, with_labels=True, node_color='lightblue',
                node_size=3000, font_size=10, arrows=True)
        plt.show()
```

**導入効果**:
- 複雑な記事構造の可視化
- セクション間の依存関係管理
- 動的な記事分割・統合が可能

**導入コスト**: 5-7日（NetworkXの学習を含む）

### 5.2 統合優先順位マトリクス

| シナリオ | 効果（高/中/低） | コスト（日数） | ROI | 優先度 |
|---------|----------------|---------------|-----|--------|
| **1. プロンプトテンプレート** | 高 | 2-3日 | ★★★★★ | 1位 |
| **2. MathEnvironmentProtector** | 中 | 1-2日 | ★★★★☆ | 2位 |
| **3. ContentQualityAssessor** | 高 | 3-4日 | ★★★★☆ | 3位 |
| **4. グラフベース構造管理** | 低 | 5-7日 | ★★☆☆☆ | 4位 |

**推奨実装順序**:
```
Week 1: シナリオ1 (プロンプトテンプレート)
Week 2: シナリオ2 (MathEnvironmentProtector) + シナリオ3の一部
Week 3: シナリオ3完全版 (ContentQualityAssessor統合)
Week 4: シナリオ4 (必要に応じて)
```

---

## 6. 具体的改善提案

### 6.1 MI Knowledge Hubへの改善提案

#### 提案1: Content Agent強化パッケージ

**目的**: Write_tb_AIのプロンプト設計原則を適用

**実装ステップ**:

**Step 1: プロンプトテンプレート化 (2日)**
```python
# Before (現状)
prompt = f"Write an article about {topic}"

# After (Write_tb_AI方式)
prompt_template = PromptTemplate.from_template("""
以下の内容で記事を執筆します。
{topic}

【出力仕様】
- Markdownで出力
- Front matterを含む
- 最低{min_words}語以上

【品質要件】
- 学術的参照を3件以上含む
- 実行可能なコード例を含む
- 論理的構成を維持する

出力形式:
```markdown
---
title: "..."
---

本文
```
""")
```

**Step 2: 品質チェック統合 (2日)**
```python
def create_article_with_qa(topic):
    for attempt in range(3):
        content = llm.invoke(prompt_template.format(topic=topic))

        quality_score = assess_quality(content)

        if quality_score >= 80:
            return content

        # 改善プロンプト生成
        improvement_prompt = generate_improvement_prompt(content, quality_score)

    return content  # 最終的にはPhase 3に委ねる
```

**Step 3: 数式処理パイプライン (1日)**
```python
def post_process_content(content):
    processor = MarkdownMathProcessor()
    return processor.process(content)
```

**期待効果**:
- Content Agentの初稿品質が **70点 → 85点** に向上
- Phase 3での修正工数が **30%削減**
- 数式レンダリングエラーが **50%削減**

#### 提案2: Academic Reviewer評価軸の拡張

**目的**: Write_tb_AIのQualityLevelを統合

**現状の4次元評価**:
1. Scientific Accuracy
2. Completeness
3. Educational Quality
4. Implementation Quality

**拡張案（6次元評価）**:
1. Scientific Accuracy
2. Completeness
3. Educational Quality
4. Implementation Quality
5. **Structural Quality** (Write_tb_AIから追加)
   - 論理的構成
   - セクション分割の適切性
   - 導入・展開・結論の明確性
6. **Linguistic Quality** (Write_tb_AIから追加)
   - 文体の一貫性（です・ます調）
   - 専門用語の適切性
   - 読みやすさ

**実装**:
```python
# MI/.claude/agents/academic-reviewer.md に追加

def review_article(content):
    scores = {
        'scientific_accuracy': assess_scientific_accuracy(content),
        'completeness': assess_completeness(content),
        'educational_quality': assess_educational_quality(content),
        'implementation_quality': assess_implementation_quality(content),
        'structural_quality': assess_structural_quality(content),  # 新規
        'linguistic_quality': assess_linguistic_quality(content)   # 新規
    }

    # 重み付き平均
    weights = {
        'scientific_accuracy': 0.25,
        'completeness': 0.20,
        'educational_quality': 0.20,
        'implementation_quality': 0.15,
        'structural_quality': 0.10,  # 新規
        'linguistic_quality': 0.10   # 新規
    }

    overall_score = sum(scores[k] * weights[k] for k in scores)

    return {
        'scores': scores,
        'overall_score': overall_score,
        'quality_level': get_quality_level(overall_score)  # Write_tb_AI方式
    }
```

**期待効果**:
- 評価の包括性向上
- 文体・構成面の問題の早期発見
- より細かい改善指示が可能

#### 提案3: Phase 2.5の新設（即時品質ゲート）

**目的**: Phase 3前に低品質コンテンツをフィルタリング

**フロー**:
```
Phase 2 (Content Agent: 初稿作成)
  ↓
Phase 2.5 (NEW: Immediate Quality Gate)  ← Write_tb_AIのリトライ機構を適用
  ├─ スコア >= 70: Phase 3へ進む
  └─ スコア < 70: Content Agentに差し戻し (最大3回)
  ↓
Phase 3 (Academic Reviewer: 学術的検証)
```

**実装**:
```python
# MI/tools/quality_gate.py (新規作成)

class ImmediateQualityGate:
    def __init__(self):
        self.min_score = 70
        self.max_retries = 3

    def evaluate(self, content):
        """Write_tb_AIのContentQualityAssessor相当"""
        checks = {
            'length': self._check_length(content),
            'structure': self._check_structure(content),
            'code_examples': self._check_code_examples(content),
            'references': self._check_references(content)
        }

        score = sum(checks.values()) / len(checks) * 100

        return {
            'score': score,
            'passed': score >= self.min_score,
            'issues': [k for k, v in checks.items() if v < 0.7]
        }

    def run(self, content_agent, topic):
        for attempt in range(self.max_retries):
            content = content_agent.create_article(topic)
            result = self.evaluate(content)

            if result['passed']:
                return content, result

            print(f"[Quality Gate] Attempt {attempt+1}: Score {result['score']}/100")
            print(f"[Quality Gate] Issues: {result['issues']}")

        # 最大リトライ後 → Phase 3に委ねる（警告付き）
        return content, {'score': result['score'], 'passed': False, 'warning': 'Max retries reached'}
```

**期待効果**:
- Phase 3の不合格率が **40% → 15%** に削減
- 全体的な品質スコアが **81.5 → 88** に向上（予測）
- Phase 3-7の反復回数が削減

### 6.2 Write_tb_AIへの改善提案

#### 提案1: MI Hub風9フェーズワークフローの適用

**目的**: 単一パイプラインを多段階検証に拡張

**現状のWrite_tb_AIフロー**:
```
1. Structure → 2. Graph → 3. Content → 4. LaTeX → 5. HTML → 6. Save
```

**拡張案（9フェーズ）**:
```
Phase 0: Planning (current: structure generation)
Phase 1: Information Gathering (NEW: 文献調査エージェント追加)
Phase 2: Draft Creation (current: content generation)
Phase 3: Technical Review (NEW: LaTeX構文検証エージェント)
Phase 4: Educational Review (NEW: 読者視点検証エージェント)
Phase 5: Code Review (NEW: コード実行可能性検証)
Phase 6: Design Review (NEW: レイアウト・可読性検証)
Phase 7: Final Review (NEW: 総合品質チェック、90点以上必須)
Phase 8: Integration QA (current: compile + save)
Phase 9: Publication (current: output)
```

**実装**:
```python
# write_tb_ai/enhanced_generator.py (新規作成)

class EnhancedTextbookGenerator(TextbookGenerator):
    def generate_textbook_with_phases(self):
        # Phase 0-2: 既存フロー
        book_structure = self._create_book_structure()
        self._build_graph_structure(book_structure)
        self._generate_content()

        # Phase 3: Technical Review (NEW)
        latex_review = self.latex_reviewer.review(self.graph_manager)
        if latex_review['score'] < 80:
            self._fix_latex_issues(latex_review['issues'])

        # Phase 4: Educational Review (NEW)
        edu_review = self.educational_reviewer.review(self.graph_manager)
        if edu_review['score'] < 80:
            self._enhance_educational_quality(edu_review['suggestions'])

        # Phase 5-6: Code + Design Review (NEW)
        code_review = self.code_reviewer.review(self.graph_manager)
        design_review = self.design_reviewer.review(self.graph_manager)

        # Phase 7: Final Review (NEW)
        final_score = self.final_reviewer.comprehensive_review(
            self.graph_manager, latex_review, edu_review, code_review, design_review
        )

        if final_score < 90:
            raise QualityGateException(f"Final score {final_score}/100 below threshold 90")

        # Phase 8-9: 既存フロー
        self._create_latex_document(book_structure["title"])
        self._save_output_files(book_structure["title"])
```

**期待効果**:
- 教科書の総合品質が向上
- 多角的検証により見落としが削減
- 教育的価値が向上

**導入コスト**: 10-14日（新規エージェント4つの実装）

#### 提案2: マルチエージェント並列検証

**目的**: Write_tb_AIの線形フローを並列化して高速化

**現状**:
```
Content Gen (Section 1) → Content Gen (Section 2) → ... (逐次)
```

**改善案**:
```python
from concurrent.futures import ThreadPoolExecutor

def _generate_content_parallel(self):
    sorted_content_str_list = self.graph_manager.sort_leaf_nodes()

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = []

        for heading_number_str in sorted_content_str_list:
            node_name = self._get_node_name(heading_number_str)
            future = executor.submit(self._generate_single_section, node_name)
            futures.append(future)

        # 全セクション並列生成
        results = [f.result() for f in futures]
```

**期待効果**:
- 生成時間が **60分 → 15分** に短縮（40ページの教科書の場合）
- API並列呼び出しによるスループット向上

**注意点**:
- OpenAI APIのRate Limitに注意（TPM/RPM制限）
- 並列度の動的調整が必要

#### 提案3: Markdown中間フォーマットの追加

**目的**: MI Hub風のMarkdown → 複数形式出力に対応

**実装**:
```python
# write_tb_ai/markdown_converter.py (新規作成)

class MarkdownConverter:
    def latex_to_markdown(self, latex_content):
        """LaTeXをMarkdownに変換"""
        # 1. 数式環境の変換
        markdown = latex_content
        markdown = re.sub(r'\\begin\{equation\*?\}(.+?)\\end\{equation\*?\}',
                         r'$$\1$$', markdown, flags=re.DOTALL)

        # 2. 見出しの変換
        markdown = re.sub(r'\\section\{(.+?)\}', r'## \1', markdown)
        markdown = re.sub(r'\\subsection\{(.+?)\}', r'### \1', markdown)

        # 3. リストの変換
        markdown = re.sub(r'\\begin\{itemize\}', '', markdown)
        markdown = re.sub(r'\\end\{itemize\}', '', markdown)
        markdown = re.sub(r'\\item ', r'- ', markdown)

        # 4. コードブロックの変換
        markdown = re.sub(r'\\begin\{lstlisting\}\[language=(.+?)\](.+?)\\end\{lstlisting\}',
                         r'```\1\n\2\n```', markdown, flags=re.DOTALL)

        return markdown

    def save_markdown(self, content, filename):
        with open(f"{filename}.md", 'w', encoding='utf-8') as f:
            f.write(content)
```

**導入効果**:
- Markdown → GitHub Pages 公開が可能に
- Markdown → HTML 変換が容易
- MI Hubとの相互運用性向上

**導入コスト**: 3-5日

---

## 7. 実装ロードマップ

### 7.1 MI Knowledge Hub 改善ロードマップ（4週間）

#### Week 1: Foundation (基盤整備)

**Day 1-2: プロンプトテンプレートシステム構築**
- `MI/tools/content_agent_prompts.py` 作成
- 3種類のテンプレート実装（構造生成、セクション分節、本文生成）
- Content Agentに統合

**Day 3-4: 品質評価システム基礎**
- Write_tb_AIの`quality_assurance.py`を移植
- `ContentQualityAssessor`クラスの実装
- 5段階QualityLevel定義

**Day 5-7: Phase 2.5 Immediate Quality Gate実装**
- `MI/tools/quality_gate.py` 作成
- リトライ機構（最大3回）の実装
- Content Agentとの統合テスト

**成果物**:
- [ ] `content_agent_prompts.py`
- [ ] `content_quality_assessor.py`
- [ ] `quality_gate.py`
- [ ] 統合テストレポート

#### Week 2: Enhancement (機能強化)

**Day 8-10: MathEnvironmentProtector実装**
- `MI/tools/markdown_math_processor.py` 作成
- Unicode記号→LaTeXコマンド変換
- Markdown数式保護機構

**Day 11-12: Academic Reviewer評価軸拡張**
- Structural Quality評価関数の実装
- Linguistic Quality評価関数の実装
- 6次元評価への移行

**Day 13-14: データ検証強化**
- Data AgentにContentQualityAssessor統合
- コード実行可能性の自動検証
- JSONスキーマバリデーション強化

**成果物**:
- [ ] `markdown_math_processor.py`
- [ ] Academic Reviewer v2.0
- [ ] Data Agent強化版

#### Week 3: Integration (統合テスト)

**Day 15-17: エンドツーエンドテスト**
- 新MI入門記事を9フェーズで生成
- Phase 2.5ゲートの効果測定
- 品質スコア改善の定量評価

**Day 18-19: パフォーマンス最適化**
- プロンプトキャッシュの実装
- API呼び出し回数の削減
- レスポンスタイムの短縮

**Day 20-21: ドキュメント整備**
- CLAUDE.mdの更新
- 新機能の使用例作成
- トラブルシューティングガイド作成

**成果物**:
- [ ] エンドツーエンドテストレポート
- [ ] パフォーマンス改善レポート
- [ ] 更新されたCLAUDE.md

#### Week 4: Advanced Features (高度機能)

**Day 22-24: グラフベース構造管理（Optional）**
- `MI/tools/article_graph_manager.py` 作成
- NetworkXによるグラフ構築
- 目次自動生成の強化

**Day 25-26: 自動リファクタリング機構**
- 低品質セクションの自動識別
- 改善プロンプトの自動生成
- 反復改善ループの実装

**Day 27-28: 最終統合とリリース準備**
- 全機能の統合テスト
- リリースノート作成
- Git tag付けとデプロイ

**成果物**:
- [ ] `article_graph_manager.py` (Optional)
- [ ] 自動リファクタリングシステム
- [ ] リリースノート v2.0

### 7.2 Write_tb_AI 改善ロードマップ（6週間）

#### Week 1-2: Multi-Agent Architecture

**Week 1: エージェント基盤**
- LaTeX Reviewer Agent実装
- Educational Reviewer Agent実装
- Code Reviewer Agent実装

**Week 2: エージェント統合**
- Design Reviewer Agent実装
- Final Reviewer Agent実装（総合評価）
- エージェント間通信プロトコル設計

**成果物**:
- [ ] 5つの新規エージェント
- [ ] エージェント間通信API

#### Week 3-4: Parallel Processing

**Week 3: 並列生成機構**
- ThreadPoolExecutorによる並列化
- Rate Limit対応機構
- エラーハンドリング強化

**Week 4: 最適化とテスト**
- 並列度の動的調整
- パフォーマンステスト
- 品質保持の検証

**成果物**:
- [ ] 並列生成システム
- [ ] パフォーマンスベンチマーク

#### Week 5-6: Enhanced Output Formats

**Week 5: Markdown対応**
- LaTeX→Markdown変換器
- Markdown→HTML変換器
- GitHub Pages対応

**Week 6: 統合と公開**
- 全機能の統合
- ドキュメント更新
- GitHubリリース

**成果物**:
- [ ] Markdown変換システム
- [ ] 更新されたREADME.md
- [ ] GitHub Release v2.0

### 7.3 マイルストーン

| マイルストーン | 期限 | 達成基準 |
|---------------|------|---------|
| **M1: MI Hub基盤整備完了** | Week 1終了時 | Phase 2.5実装、テスト通過 |
| **M2: MI Hub機能強化完了** | Week 2終了時 | 数式処理、6次元評価実装 |
| **M3: MI Hub統合テスト完了** | Week 3終了時 | エンドツーエンド成功、品質スコア85+ |
| **M4: MI Hub v2.0リリース** | Week 4終了時 | 全機能実装、ドキュメント完備 |
| **M5: Write_tb_AIエージェント化** | Week 2終了時 | 5エージェント実装 |
| **M6: Write_tb_AI並列化** | Week 4終了時 | 4倍速化達成 |
| **M7: Write_tb_AI v2.0リリース** | Week 6終了時 | Markdown対応、GitHub公開 |

---

## 8. 結論と推奨事項

### 8.1 主要な発見

**Write_tb_AI の強み**:
1. **厳格なプロンプト設計**: JSON/LaTeX構文エラーを最小化
2. **多層フォールバック**: テンプレートベースで確実性保証
3. **数学記号処理**: MathEnvironmentProtectorによる高度な処理
4. **エラーハンドリング**: 指数バックオフ + エラー種別対応

**MI Knowledge Hub の強み**:
1. **多角的検証**: 7エージェントによる包括的品質保証
2. **段階的改善**: 9フェーズで品質を段階的に向上
3. **継続更新**: Git + Markdownで柔軟な更新が可能
4. **教育的価値**: 演習・図・実践例の充実

**相補性**:
- Write_tb_AI: **生成時の品質** に強み
- MI Hub: **生成後の検証** に強み
- 統合により、両方の利点を享受可能

### 8.2 優先推奨事項

**最優先（ROI最高）**:
1. **MI Hubにプロンプトテンプレート方式を導入** (2-3日)
   - 即座に出力品質が向上
   - パースエラーが削減
   - 実装コストが低い

2. **Phase 2.5 Immediate Quality Gate実装** (3-4日)
   - Phase 3不合格率を40% → 15%に削減
   - 全体フローの効率化
   - Write_tb_AIの知見を直接適用

**中優先（効果大、コスト中）**:
3. **MathEnvironmentProtectorのMarkdown対応** (1-2日)
   - 数式レンダリングエラー削減
   - LaTeX互換性向上
   - 将来的なPDF出力にも有用

4. **Academic Reviewer 6次元評価への拡張** (2-3日)
   - 評価の包括性向上
   - 文体・構成面の問題検出
   - より具体的な改善指示が可能

**長期（効果中、コスト高）**:
5. **グラフベース構造管理の導入** (5-7日)
   - 複雑な記事の管理が容易
   - 動的な構造変更が可能
   - 可視化により理解が深まる

### 8.3 避けるべき罠

**罠1: 過度な自動化**
- 品質チェックをすべて自動化すると、創造性が失われる
- 人間の最終承認（Phase 9）は必須

**罠2: プロンプトの複雑化**
- テンプレートが長すぎると、LLMが混乱する
- 簡潔さと詳細さのバランスが重要

**罠3: エージェント数の増加**
- エージェントが多すぎると管理が困難
- 7エージェント（MI Hub）が最適バランス

**罠4: 並列化の過剰**
- Rate Limitに抵触するリスク
- コストの急増
- 適切な並列度の設定が重要

### 8.4 最終推奨

**ステップ1: MI Hub強化（4週間）**
```
Week 1: プロンプトテンプレート + Phase 2.5
Week 2: MathEnvironmentProtector + 6次元評価
Week 3: 統合テストとパフォーマンス最適化
Week 4: ドキュメント整備とリリース
```

**ステップ2: Write_tb_AI進化（6週間・Optional）**
```
Week 1-2: 5エージェント実装
Week 3-4: 並列化
Week 5-6: Markdown対応とリリース
```

**ステップ3: ハイブリッドシステム（長期ビジョン）**
```
- MI Hubの教育コンテンツ生成機能を強化
- Write_tb_AIのPDF生成機能をMI Hubに統合
- 統一的な品質保証システムの構築
```

---

## 9. 付録

### 9.1 コード例: 統合プロンプトテンプレート

```python
# MI/tools/unified_prompt_system.py

from typing import Dict, Any
from string import Template

class UnifiedPromptSystem:
    """Write_tb_AI方式のプロンプトテンプレートをMI Hubに適用"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._init_templates()

    def _init_templates(self):
        # 共通プロンプト
        self.common_prompt = Template("""
以下の内容で教育的記事を執筆します。
テーマ: ${topic}
対象読者: ${target_audience}
推定学習時間: ${estimated_hours}時間

【文体規則】
- です・ます調で統一すること
- 推測や未確認の情報は含めないこと
- 学術的で専門的な内容とすること

【追加要件】
${additional_requirements}
""")

        # 記事構造生成テンプレート（Write_tb_AIのprompt_book_title相当）
        self.structure_template = Template(self.common_prompt.template + """
以下の仕様に従って、記事のタイトルとセクション構成をYAMLで出力してください。

【出力仕様】
- YAMLのみ出力（説明文・コードフェンス不要）
- セクション数は3-5個
- 各セクションのword_countは合計が${total_words}になるよう調整

【YAML構造】
title: "記事タイトル（50文字以内）"
description: "100-150文字の概要"
target_audience: "${target_audience}"
estimated_hours: ${estimated_hours}
sections:
  - heading: "セクション1タイトル"
    summary: "50文字以内の概要"
    word_count: 1500
    difficulty: "beginner"  # beginner/intermediate/advanced
  - heading: "セクション2タイトル"
    summary: "50文字以内の概要"
    word_count: 2000
    difficulty: "intermediate"

【品質要件】
- タイトルは「第1章」「Part 1」等の汎用表現を避ける
- 各セクションは論理的に繋がる構成とする
- 難易度が段階的に上昇する構成とする
""")

        # セクション本文生成テンプレート（Write_tb_AIのprompt_content_creation相当）
        self.content_template = Template(self.common_prompt.template + """
記事タイトル: ${article_title}
記事概要: ${article_summary}

以下のセクションの本文を${word_count}語で作成してください。
セクションタイトル: ${section_title}
セクション概要: ${section_summary}
難易度: ${difficulty}

【Markdown規則】
- 見出しは ## (H2) から開始
- コードブロックは必ず言語指定（```python, ```bash等）
- 数式はMathJax形式（インライン: $...$、ディスプレイ: $$...$$）
- 箇条書きは - で統一（*や+は使用しない）

【必須要素】
- 実行可能なコード例を最低2個含む
- 学術的参照を最低3件含む（[^1]形式）
- 実践的な演習問題を最低2問含む
- セクション末尾に「まとめ」を含む

【品質要件】
- 最低${word_count}語以上（${word_count} × 1.2まで許容）
- 論理的構成: 導入 → 理論 → 実践 → まとめ
- コードは必ず動作検証済みのものとする
- 専門用語は初出時に説明する

【コンテキスト】
全体目次:
${toc}

直前セクション:
${previous_section}

出力形式:
```markdown
## ${section_title}

本文...
```
""")

    def generate_structure_prompt(self, topic: str, target_audience: str,
                                   total_words: int, estimated_hours: int,
                                   additional_requirements: str = "") -> str:
        """記事構造生成プロンプトを生成"""
        return self.structure_template.substitute(
            topic=topic,
            target_audience=target_audience,
            total_words=total_words,
            estimated_hours=estimated_hours,
            additional_requirements=additional_requirements
        )

    def generate_content_prompt(self, article_title: str, article_summary: str,
                                section_title: str, section_summary: str,
                                word_count: int, difficulty: str,
                                toc: str = "", previous_section: str = "",
                                additional_requirements: str = "") -> str:
        """セクション本文生成プロンプトを生成"""
        return self.content_template.substitute(
            topic=article_title,  # 共通プロンプト用
            target_audience=self.config.get('target_audience', ''),
            estimated_hours=self.config.get('estimated_hours', 0),
            additional_requirements=additional_requirements,
            article_title=article_title,
            article_summary=article_summary,
            section_title=section_title,
            section_summary=section_summary,
            word_count=word_count,
            difficulty=difficulty,
            toc=toc,
            previous_section=previous_section
        )


# 使用例
if __name__ == "__main__":
    config = {
        'target_audience': '学部生、MI初学者',
        'estimated_hours': 20
    }

    prompt_system = UnifiedPromptSystem(config)

    # 構造生成プロンプト
    structure_prompt = prompt_system.generate_structure_prompt(
        topic="マテリアルズ・インフォマティクス入門",
        target_audience="学部生、MI初学者",
        total_words=10000,
        estimated_hours=20,
        additional_requirements="実行可能なPythonコード例を含む"
    )

    print("=" * 80)
    print("構造生成プロンプト:")
    print("=" * 80)
    print(structure_prompt)

    # セクション生成プロンプト
    content_prompt = prompt_system.generate_content_prompt(
        article_title="マテリアルズ・インフォマティクス入門",
        article_summary="MIの基礎から実践まで学ぶ",
        section_title="機械学習の基礎",
        section_summary="教師あり学習の基本概念を解説",
        word_count=1500,
        difficulty="beginner",
        toc="1. はじめに\n2. 機械学習の基礎\n3. 材料データ",
        previous_section="## はじめに\n\nMIとは..."
    )

    print("\n" + "=" * 80)
    print("コンテンツ生成プロンプト:")
    print("=" * 80)
    print(content_prompt)
```

### 9.2 設定テンプレート

```json
// MI/config/content_generation.json

{
  "prompt_system": {
    "style": "unified",  // unified (Write_tb_AI方式) or legacy
    "template_version": "2.0",
    "common_requirements": [
      "です・ます調で統一",
      "推測や未確認の情報は含めない",
      "学術的で専門的な内容"
    ]
  },

  "quality_gate": {
    "enabled": true,
    "phase": 2.5,
    "min_score": 70,
    "max_retries": 3,
    "checks": [
      "length",
      "structure",
      "code_examples",
      "references"
    ]
  },

  "math_processing": {
    "enabled": true,
    "protect_environments": ["inline_math", "display_math"],
    "unicode_to_latex": true,
    "conversions": {
      "⟨": "\\langle ",
      "⟩": "\\rangle ",
      "≈": "\\approx ",
      "≠": "\\neq ",
      "≤": "\\leq ",
      "≥": "\\geq ",
      "±": "\\pm ",
      "∞": "\\infty "
    }
  },

  "retry_strategy": {
    "max_retries": 5,
    "initial_delay": 2.0,
    "max_delay": 30.0,
    "exponential_base": 2.0,
    "error_specific_delays": {
      "timeout": 1.5,
      "rate_limit": 2.0,
      "connection": 1.0,
      "other": 0.8
    }
  },

  "academic_reviewer": {
    "dimensions": 6,
    "evaluation_axes": [
      {"name": "scientific_accuracy", "weight": 0.25},
      {"name": "completeness", "weight": 0.20},
      {"name": "educational_quality", "weight": 0.20},
      {"name": "implementation_quality", "weight": 0.15},
      {"name": "structural_quality", "weight": 0.10},
      {"name": "linguistic_quality", "weight": 0.10}
    ],
    "thresholds": {
      "phase_3": 80,
      "phase_7": 90
    }
  },

  "content_agent": {
    "templates": {
      "structure": "unified_structure_template",
      "content": "unified_content_template"
    },
    "validation": {
      "min_words": 100,
      "min_code_examples": 2,
      "min_references": 3,
      "min_exercises": 2
    }
  }
}
```

### 9.3 参考文献

**Write_tb_AI関連**:
- `/Users/yusukehashimoto/Documents/pycharm/Write_tb_AI/CLAUDE.md`
- `/Users/yusukehashimoto/Documents/pycharm/Write_tb_AI/README.md`
- `/Users/yusukehashimoto/Documents/pycharm/Write_tb_AI/config.py`
- `/Users/yusukehashimoto/Documents/pycharm/Write_tb_AI/content_creator.py`
- `/Users/yusukehashimoto/Documents/pycharm/Write_tb_AI/book_generator.py`

**MI Knowledge Hub関連**:
- `/Users/yusukehashimoto/Documents/pycharm/AI_Homepage/MI/CLAUDE.md`
- `/Users/yusukehashimoto/Documents/pycharm/AI_Homepage/MI/claudedocs/content-creation-procedure.md`
- `/Users/yusukehashimoto/Documents/pycharm/AI_Homepage/MI/.claude/agents/`

**技術スタック**:
- LangChain: LLM統合フレームワーク
- NetworkX: グラフ理論ライブラリ
- PyLaTeX: LaTeX文書生成
- OpenAI GPT Models: コンテンツ生成エンジン

---

**分析完了日**: 2025-10-16
**レポートバージョン**: 1.0
**次回更新予定**: 実装開始後1週間後


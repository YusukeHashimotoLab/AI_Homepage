# 高品質アウトプット生成ガイド

**Write_tb_AI × MI Knowledge Hub 統合ベストプラクティス**

このガイドは、Write_tb_AIとMI Knowledge Hubの両システムから抽出した、高品質コンテンツ生成のための実践的なコツを体系化したものです。

---

## 目次

1. [プロンプトエンジニアリングの原則](#1-プロンプトエンジニアリングの原則)
2. [品質保証の多層防御](#2-品質保証の多層防御)
3. [エラーハンドリング戦略](#3-エラーハンドリング戦略)
4. [構造化と一貫性](#4-構造化と一貫性)
5. [検証とフィードバックループ](#5-検証とフィードバックループ)
6. [実践チェックリスト](#6-実践チェックリスト)

---

## 1. プロンプトエンジニアリングの原則

### 1.1 テンプレートベース設計（Write_tb_AIの強み）

**原則**: プロンプトを再利用可能なテンプレートとして設計し、一貫性と品質を保証する。

#### ✅ Good Example

```python
TEMPLATE_CONTENT_CREATION = """
あなたは{subject}の専門家です。以下の要求に従って、高品質な教育コンテンツを生成してください。

# 要求事項
- トピック: {topic}
- 対象レベル: {level}
- 文字数: {min_words}〜{max_words}字
- 必須要素: {required_elements}

# 出力形式（厳密に従うこと）
{{
  "title": "セクションタイトル",
  "content": "本文（Markdown形式）",
  "code_examples": [
    {{"language": "python", "code": "...", "explanation": "..."}}
  ],
  "exercises": [
    {{"question": "...", "answer": "...", "difficulty": "beginner|intermediate|advanced"}}
  ],
  "references": ["参考文献1", "参考文献2"]
}}

# 品質基準
- 科学的正確性を最優先
- 実行可能なコード例を含む
- 段階的な説明（易→難）
- 具体例と抽象概念のバランス
"""

# 使用例
prompt = TEMPLATE_CONTENT_CREATION.format(
    subject="Materials Informatics",
    topic="Bayesian Optimization",
    level="intermediate",
    min_words=2000,
    max_words=3000,
    required_elements="コード例3つ以上、演習問題5つ以上"
)
```

#### ❌ Bad Example

```python
# 毎回異なるプロンプトを即興で作成
prompt = f"ベイズ最適化について書いて。中級者向けで、2000字くらい。"
# → 出力形式が不定、品質基準が不明確、パースエラーのリスク大
```

### 1.2 出力形式の厳密な制約（Write_tb_AIの核心）

**原則**: LLMに期待する出力形式をJSON/Markdown/LaTeXなど明確に指定し、パースエラーを防ぐ。

#### ✅ Good Example

```python
OUTPUT_FORMAT_CONSTRAINT = """
# 出力形式（必ず以下のJSON形式で出力してください）

```json
{
  "section_title": "セクションタイトル（30文字以内）",
  "content": "本文内容（Markdown形式、2000-3000字）",
  "subsections": [
    {
      "subtitle": "サブセクションタイトル",
      "content": "サブセクション本文"
    }
  ],
  "metadata": {
    "word_count": 2500,
    "difficulty": "intermediate",
    "estimated_reading_time": "10 minutes"
  }
}
```

**重要**: JSON以外の文字列（説明文、コメント等）は一切含めないでください。
"""
```

#### ❌ Bad Example

```python
# 出力形式の指定があいまい
prompt = "ベイズ最適化について説明してください。JSON形式で返してください。"
# → LLMが「Sure! Here's the JSON: {...}」のような説明文を追加してパースエラー
```

### 1.3 コンテキスト注入の最適化

**原則**: 必要な情報を段階的に提供し、プロンプトの長さと情報密度のバランスを取る。

#### ✅ Good Example（3段階注入）

```python
# Stage 1: 基本コンテキスト
base_context = """
Subject: Materials Informatics
Target Audience: Undergraduate students (3rd year)
Language: Japanese
"""

# Stage 2: ドメイン知識
domain_knowledge = """
Related Topics Already Covered:
- Machine Learning Basics
- Python Programming
- Linear Regression

Topics to Cover Next:
- Gaussian Processes
- Bayesian Optimization
"""

# Stage 3: 参考資料
reference_materials = """
Key References:
1. Frazier, P. I. (2018). A Tutorial on Bayesian Optimization. arXiv:1807.02811.
2. Shahriari, B. et al. (2016). Taking the Human Out of the Loop. IEEE.

Relevant Datasets:
- Materials Project (mp-xxx)
- OQMD (Open Quantum Materials Database)
"""

# 統合プロンプト
full_prompt = f"{base_context}\n\n{domain_knowledge}\n\n{reference_materials}\n\n{task_instruction}"
```

#### ❌ Bad Example

```python
# 一度にすべての情報を詰め込む
prompt = "ベイズ最適化について書いて。対象は学部3年生で、機械学習とPythonは知ってて、ガウス過程も説明して、Materials ProjectとOQMDのデータセットも紹介して、Frazier 2018とShahriari 2016を参考にして..."
# → プロンプトが長すぎ、LLMが重要な情報を見落とす
```

---

## 2. 品質保証の多層防御

### 2.1 Generation-Time品質チェック（Write_tb_AI方式）

**原則**: コンテンツ生成と同時にリアルタイムで品質を検証し、不合格時は即座にリトライ。

#### ✅ Good Example

```python
class ContentQualityAssessor:
    """リアルタイム品質評価"""

    def assess_quality(self, content: str, metadata: dict) -> tuple[str, dict]:
        """
        Returns:
            rating: "excellent" | "good" | "acceptable" | "poor" | "unacceptable"
            details: {"word_count": int, "code_examples": int, "issues": list}
        """
        issues = []

        # 1. 文字数チェック
        word_count = len(content)
        if word_count < metadata["min_words"]:
            issues.append(f"文字数不足: {word_count} < {metadata['min_words']}")

        # 2. コード例チェック
        code_blocks = re.findall(r'```[\s\S]*?```', content)
        if len(code_blocks) < metadata["min_code_examples"]:
            issues.append(f"コード例不足: {len(code_blocks)} < {metadata['min_code_examples']}")

        # 3. 数式チェック
        math_expressions = re.findall(r'\$.*?\$', content)
        if len(math_expressions) < metadata["min_math_expressions"]:
            issues.append(f"数式不足: {len(math_expressions)} < {metadata['min_math_expressions']}")

        # 4. 参考文献チェック
        references = re.findall(r'\[.*?\]\(.*?\)', content)
        if len(references) < metadata["min_references"]:
            issues.append(f"参考文献不足: {len(references)} < {metadata['min_references']}")

        # 5. 構造チェック
        headings = re.findall(r'^#{2,4}\s+.+$', content, re.MULTILINE)
        if len(headings) < 3:
            issues.append("セクション構造が不十分（見出しが3つ未満）")

        # 評価判定
        if len(issues) == 0:
            rating = "excellent"
        elif len(issues) <= 2:
            rating = "good"
        elif len(issues) <= 4:
            rating = "acceptable"
        elif len(issues) <= 6:
            rating = "poor"
        else:
            rating = "unacceptable"

        return rating, {
            "word_count": word_count,
            "code_examples": len(code_blocks),
            "math_expressions": len(math_expressions),
            "references": len(references),
            "headings": len(headings),
            "issues": issues
        }

# 使用例
def generate_with_quality_check(prompt: str, metadata: dict, max_retries: int = 3):
    assessor = ContentQualityAssessor()

    for attempt in range(max_retries):
        content = llm.generate(prompt)
        rating, details = assessor.assess_quality(content, metadata)

        if rating in ["excellent", "good", "acceptable"]:
            return content, rating, details

        # 不合格時のプロンプト改善
        prompt = f"{prompt}\n\n# 前回の問題点\n{chr(10).join(details['issues'])}\n\n上記を改善して再生成してください。"

    raise QualityError(f"品質基準を満たせませんでした（{max_retries}回試行）")
```

### 2.2 Post-Generation多角的検証（MI Hub方式）

**原則**: 生成後に複数の観点から段階的に検証し、反復改善を行う。

#### ✅ Good Example（9フェーズ品質保証）

```python
class MultiPhaseQualityAssurance:
    """9フェーズ品質保証システム"""

    def __init__(self):
        self.academic_reviewer = AcademicReviewer()
        self.design_agent = DesignAgent()
        self.data_agent = DataAgent()

    def phase_0_2_initial_draft(self, topic: str, level: str) -> str:
        """Phase 0-2: 初期ドラフト生成"""
        content = self.content_agent.create_draft(topic, level)
        return content

    def phase_3_academic_review_gate_1(self, content: str) -> tuple[int, dict]:
        """Phase 3: 学術的品質評価（第1ゲート、基準80点）"""
        review = self.academic_reviewer.review(content, gate="phase_3")

        # 4次元評価
        scores = {
            "scientific_accuracy": review["scientific_accuracy"],  # 0-100
            "completeness": review["completeness"],               # 0-100
            "educational_quality": review["educational_quality"], # 0-100
            "implementation_quality": review["implementation_quality"]  # 0-100
        }

        total_score = sum(scores.values()) / len(scores)

        if total_score < 80:
            raise QualityGateFailure(f"Phase 3不合格: {total_score:.1f} < 80.0", review)

        return total_score, review

    def phase_4_6_enhancement(self, content: str, review: dict) -> str:
        """Phase 4-6: コンテンツ強化"""
        # Phase 4: 参考文献追加
        content = self.data_agent.add_references(content, review["missing_references"])

        # Phase 5: コード例改善
        content = self.content_agent.enhance_code_examples(content)

        # Phase 6: UX/デザイン最適化
        content = self.design_agent.optimize_layout(content)
        content = self.design_agent.add_diagrams(content)

        return content

    def phase_7_academic_review_gate_2(self, content: str) -> tuple[int, dict]:
        """Phase 7: 学術的品質評価（第2ゲート、基準90点）"""
        review = self.academic_reviewer.review(content, gate="phase_7")
        total_score = sum(review["scores"].values()) / len(review["scores"])

        if total_score < 80:
            raise QualityGateFailure("MAJOR_REVISION required", review)
        elif total_score < 90:
            raise QualityGateFailure("MINOR_REVISION required", review)

        return total_score, review

    def execute_full_pipeline(self, topic: str, level: str) -> dict:
        """完全な9フェーズパイプライン実行"""
        # Phase 0-2
        content = self.phase_0_2_initial_draft(topic, level)

        # Phase 3（第1ゲート）
        try:
            score_3, review_3 = self.phase_3_academic_review_gate_1(content)
        except QualityGateFailure as e:
            # Phase 1に戻って再生成
            content = self.phase_0_2_initial_draft(topic, level)
            score_3, review_3 = self.phase_3_academic_review_gate_1(content)

        # Phase 4-6
        content = self.phase_4_6_enhancement(content, review_3)

        # Phase 7（第2ゲート）
        max_minor_revisions = 2
        for revision in range(max_minor_revisions):
            try:
                score_7, review_7 = self.phase_7_academic_review_gate_2(content)
                break
            except QualityGateFailure as e:
                if "MAJOR_REVISION" in str(e):
                    # Phase 1に戻る
                    content = self.phase_0_2_initial_draft(topic, level)
                    score_3, review_3 = self.phase_3_academic_review_gate_1(content)
                    content = self.phase_4_6_enhancement(content, review_3)
                else:
                    # MINOR_REVISION: Phase 4に戻る
                    content = self.phase_4_6_enhancement(content, e.review)

        # Phase 8-9: 最終チェックと公開
        return {
            "content": content,
            "phase_3_score": score_3,
            "phase_7_score": score_7,
            "reviews": [review_3, review_7]
        }
```

### 2.3 ハイブリッドアプローチ（推奨）

**原則**: Generation-TimeとPost-Generationの両方を組み合わせる。

```python
class HybridQualityAssurance:
    """ハイブリッド品質保証システム"""

    def generate_high_quality_content(self, topic: str) -> str:
        # Stage 1: Generation-Time品質チェック（Write_tb_AI方式）
        content = self._generate_with_realtime_check(topic)

        # Stage 2: Post-Generation多角的検証（MI Hub方式）
        content = self._multi_phase_validation(content)

        return content

    def _generate_with_realtime_check(self, topic: str) -> str:
        """リアルタイム品質チェック付き生成"""
        for attempt in range(5):
            content = self.llm.generate(self.prompt_template.format(topic=topic))
            rating, details = self.quality_assessor.assess(content)

            if rating in ["excellent", "good"]:
                return content

            # プロンプト改善
            self.prompt_template = self._improve_prompt(self.prompt_template, details)

        raise QualityError("Generation-time check failed")

    def _multi_phase_validation(self, content: str) -> str:
        """多角的検証と改善"""
        # Academic review
        academic_score = self.academic_reviewer.review(content)
        if academic_score < 80:
            content = self._improve_academic_quality(content)

        # Design review
        design_score = self.design_agent.review(content)
        if design_score < 90:
            content = self._improve_design(content)

        return content
```

---

## 3. エラーハンドリング戦略

### 3.1 指数バックオフリトライ（Write_tb_AIの核心）

**原則**: エラー種別に応じた待機時間で、最大N回まで指数関数的にリトライする。

#### ✅ Good Example

```python
import time
import random
from typing import Callable, TypeVar, Optional

T = TypeVar('T')

class ExponentialBackoffRetry:
    """指数バックオフリトライ戦略"""

    def __init__(
        self,
        max_retries: int = 5,
        initial_delay: float = 2.0,
        exponential_base: float = 2.0,
        max_delay: float = 30.0,
        jitter: bool = True
    ):
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.exponential_base = exponential_base
        self.max_delay = max_delay
        self.jitter = jitter

    def execute(self, operation: Callable[[], T], operation_name: str = "operation") -> T:
        """
        エラー種別別の遅延計算付きリトライ実行

        Args:
            operation: 実行する処理
            operation_name: 操作名（ログ用）

        Returns:
            操作の結果

        Raises:
            最後のエラー（max_retries回失敗後）
        """
        last_exception = None

        for attempt in range(self.max_retries):
            try:
                result = operation()

                if attempt > 0:
                    print(f"✅ {operation_name} succeeded on attempt {attempt + 1}")

                return result

            except Exception as e:
                last_exception = e

                if attempt == self.max_retries - 1:
                    # 最後の試行で失敗
                    break

                # エラー種別別の遅延計算
                delay = self._calculate_delay(e, attempt)

                print(f"⚠️ {operation_name} failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                print(f"   Retrying in {delay:.1f} seconds...")

                time.sleep(delay)

        # すべてのリトライ失敗
        raise RetryExhaustedError(
            f"{operation_name} failed after {self.max_retries} attempts"
        ) from last_exception

    def _calculate_delay(self, error: Exception, attempt: int) -> float:
        """エラー種別に応じた遅延時間計算"""

        # 基本遅延（指数バックオフ）
        base_delay = min(
            self.initial_delay * (self.exponential_base ** attempt),
            self.max_delay
        )

        # エラー種別別の係数
        if isinstance(error, RateLimitError):
            # レート制限エラー: 長めの待機
            multiplier = 2.0
        elif isinstance(error, TimeoutError):
            # タイムアウト: 中程度の待機
            multiplier = 1.5
        elif isinstance(error, ValidationError):
            # バリデーションエラー: 短めの待機
            multiplier = 0.5
        else:
            # その他のエラー: 標準待機
            multiplier = 1.0

        delay = base_delay * multiplier

        # ジッター追加（Thundering Herd Problem回避）
        if self.jitter:
            jitter_amount = delay * 0.1  # ±10%
            delay += random.uniform(-jitter_amount, jitter_amount)

        return min(delay, self.max_delay)


# 使用例
retry_manager = ExponentialBackoffRetry(
    max_retries=5,
    initial_delay=2.0,
    exponential_base=2.0,
    max_delay=30.0,
    jitter=True
)

def generate_content():
    return llm.generate(prompt)

try:
    content = retry_manager.execute(generate_content, "content_generation")
except RetryExhaustedError as e:
    # フォールバック処理
    content = fallback_content_generator.generate()
```

### 3.2 エラーパターン検出とプロンプト改善

**原則**: 繰り返し発生するエラーパターンを検出し、プロンプトを自動改善する。

```python
class ErrorPatternDetector:
    """エラーパターン検出器"""

    def __init__(self):
        self.error_history = []
        self.error_patterns = {
            "json_parse_error": {
                "pattern": r"JSONDecodeError|Expecting value|Invalid JSON",
                "fix": "出力形式制約を強化（JSON以外の文字を含めない）"
            },
            "incomplete_output": {
                "pattern": r"Incomplete|Truncated|...$",
                "fix": "最大トークン数を増やす、セクション分割を指示"
            },
            "math_expression_error": {
                "pattern": r"LaTeX error|Math rendering failed",
                "fix": "数式環境保護を有効化、エスケープ処理を追加"
            },
            "code_syntax_error": {
                "pattern": r"SyntaxError|IndentationError",
                "fix": "コード検証ステップを追加、実行可能性をチェック"
            }
        }

    def detect_pattern(self, error: Exception) -> Optional[dict]:
        """エラーパターン検出"""
        error_message = str(error)

        for pattern_name, pattern_info in self.error_patterns.items():
            if re.search(pattern_info["pattern"], error_message, re.IGNORECASE):
                return {
                    "pattern_name": pattern_name,
                    "fix": pattern_info["fix"],
                    "error_message": error_message
                }

        return None

    def improve_prompt(self, prompt: str, error: Exception) -> str:
        """エラーに基づくプロンプト改善"""
        pattern_info = self.detect_pattern(error)

        if not pattern_info:
            return prompt

        # パターン別のプロンプト改善
        if pattern_info["pattern_name"] == "json_parse_error":
            return f"""{prompt}

# 出力形式の厳密な遵守
**重要**: 以下のJSON形式のみを出力し、説明文やコメントは一切含めないでください。
```json
{{
  "content": "..."
}}
```
"""

        elif pattern_info["pattern_name"] == "incomplete_output":
            return f"""{prompt}

# 完全性の確保
- すべてのセクションを完全に記述してください
- 途中で終了せず、必ず結論まで書いてください
- 文字数制限内で最大限の情報を提供してください
"""

        elif pattern_info["pattern_name"] == "math_expression_error":
            return f"""{prompt}

# 数式の記述規則
- インライン数式: `$expression$`形式を使用
- ディスプレイ数式: `$$expression$$`形式を使用
- LaTeX特殊文字は必ずエスケープ（\\\\, \\_, \\^等）
"""

        return prompt
```

### 3.3 フォールバックメカニズム

**原則**: リトライが尽きた場合の代替手段を用意する。

```python
class FallbackContentGenerator:
    """フォールバック コンテンツ生成器"""

    def __init__(self):
        self.template_library = self._load_templates()

    def generate(self, topic: str, level: str) -> str:
        """テンプレートベースのフォールバック生成"""

        # 1. テンプレートマッチング
        template = self._match_template(topic, level)

        if template:
            # 2. テンプレートベース生成
            return self._generate_from_template(template, topic)

        # 3. 最小限のスケルトン生成
        return self._generate_skeleton(topic, level)

    def _generate_skeleton(self, topic: str, level: str) -> str:
        """最小限のコンテンツスケルトン"""
        return f"""# {topic}

## 概要

本セクションでは、{topic}について説明します。

## 主要概念

### 概念1

（詳細は後で追加）

### 概念2

（詳細は後で追加）

## 実装例

```python
# サンプルコード（後で拡充）
pass
```

## 演習問題

1. （問題1）
2. （問題2）

## 参考文献

- （参考文献を追加）

---

**注意**: このコンテンツは自動生成に失敗したため、スケルトンのみ提供されています。
人手による編集が必要です。
"""
```

---

## 4. 構造化と一貫性

### 4.1 階層的コンテンツ構造（Write_tb_AI方式）

**原則**: NetworkXなどのグラフ構造を使用し、コンテンツの階層関係を明示的に管理する。

```python
import networkx as nx

class HierarchicalContentStructure:
    """階層的コンテンツ構造管理"""

    def __init__(self):
        self.graph = nx.DiGraph()
        self.node_counter = 0

    def add_book(self, title: str) -> str:
        """書籍ノード追加"""
        node_id = f"book_{self.node_counter}"
        self.node_counter += 1

        self.graph.add_node(node_id, type="book", title=title, content="")
        return node_id

    def add_chapter(self, book_id: str, title: str) -> str:
        """章ノード追加"""
        node_id = f"chapter_{self.node_counter}"
        self.node_counter += 1

        self.graph.add_node(node_id, type="chapter", title=title, content="")
        self.graph.add_edge(book_id, node_id)
        return node_id

    def add_section(self, chapter_id: str, title: str, content: str = "") -> str:
        """セクションノード追加"""
        node_id = f"section_{self.node_counter}"
        self.node_counter += 1

        self.graph.add_node(node_id, type="section", title=title, content=content)
        self.graph.add_edge(chapter_id, node_id)
        return node_id

    def get_toc(self) -> dict:
        """目次生成"""
        toc = {}

        for book_id in self.graph.nodes():
            if self.graph.nodes[book_id]["type"] == "book":
                toc[book_id] = {
                    "title": self.graph.nodes[book_id]["title"],
                    "chapters": []
                }

                for chapter_id in self.graph.successors(book_id):
                    chapter_data = {
                        "title": self.graph.nodes[chapter_id]["title"],
                        "sections": []
                    }

                    for section_id in self.graph.successors(chapter_id):
                        section_data = {
                            "title": self.graph.nodes[section_id]["title"]
                        }
                        chapter_data["sections"].append(section_data)

                    toc[book_id]["chapters"].append(chapter_data)

        return toc

    def validate_structure(self) -> list[str]:
        """構造の整合性検証"""
        issues = []

        # 1. 孤立ノードチェック
        for node_id in nx.isolates(self.graph):
            issues.append(f"孤立ノード: {node_id}")

        # 2. 循環参照チェック
        if not nx.is_directed_acyclic_graph(self.graph):
            cycles = list(nx.simple_cycles(self.graph))
            issues.append(f"循環参照検出: {cycles}")

        # 3. 深さチェック
        for node_id in self.graph.nodes():
            if self.graph.nodes[node_id]["type"] == "book":
                paths = nx.single_source_shortest_path_length(self.graph, node_id)
                max_depth = max(paths.values()) if paths else 0

                if max_depth > 4:
                    issues.append(f"階層が深すぎる（{max_depth}層）: {node_id}")

        return issues

# 使用例
structure = HierarchicalContentStructure()

book_id = structure.add_book("Materials Informatics教科書")
ch1_id = structure.add_chapter(book_id, "第1章: 機械学習基礎")
sec1_1_id = structure.add_section(ch1_id, "1.1 回帰分析", content="...")
sec1_2_id = structure.add_section(ch1_id, "1.2 分類問題", content="...")

# 構造検証
issues = structure.validate_structure()
if issues:
    print("構造上の問題:", issues)
```

### 4.2 メタデータ駆動の一貫性確保

**原則**: すべてのコンテンツにメタデータを付与し、一貫性を自動検証する。

```python
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

@dataclass
class ContentMetadata:
    """コンテンツメタデータ"""
    id: str
    title: str
    level: str  # "beginner" | "intermediate" | "advanced"
    topic: str
    subtopics: List[str]
    prerequisites: List[str]
    word_count: int
    code_example_count: int
    exercise_count: int
    reference_count: int
    created_at: datetime
    updated_at: datetime
    quality_score: float
    review_status: str  # "draft" | "phase_3_approved" | "phase_7_approved" | "published"

    def validate(self) -> List[str]:
        """メタデータ検証"""
        issues = []

        # レベル別基準
        level_requirements = {
            "beginner": {"min_words": 1500, "min_code": 2, "min_exercises": 5},
            "intermediate": {"min_words": 2500, "min_code": 3, "min_exercises": 8},
            "advanced": {"min_words": 3500, "min_code": 5, "min_exercises": 10}
        }

        req = level_requirements[self.level]

        if self.word_count < req["min_words"]:
            issues.append(f"文字数不足: {self.word_count} < {req['min_words']}")

        if self.code_example_count < req["min_code"]:
            issues.append(f"コード例不足: {self.code_example_count} < {req['min_code']}")

        if self.exercise_count < req["min_exercises"]:
            issues.append(f"演習問題不足: {self.exercise_count} < {req['min_exercises']}")

        if self.reference_count < 3:
            issues.append(f"参考文献不足: {self.reference_count} < 3")

        return issues


class ContentRegistry:
    """コンテンツレジストリ（一貫性管理）"""

    def __init__(self):
        self.contents: dict[str, ContentMetadata] = {}

    def register(self, metadata: ContentMetadata):
        """コンテンツ登録"""
        # 検証
        issues = metadata.validate()
        if issues:
            raise ValidationError(f"メタデータ検証失敗: {issues}")

        self.contents[metadata.id] = metadata

    def check_prerequisites(self, content_id: str) -> List[str]:
        """前提条件チェック"""
        metadata = self.contents[content_id]
        missing = []

        for prereq_id in metadata.prerequisites:
            if prereq_id not in self.contents:
                missing.append(prereq_id)
            elif self.contents[prereq_id].review_status != "published":
                missing.append(f"{prereq_id} (未公開)")

        return missing

    def check_level_consistency(self) -> List[str]:
        """レベル一貫性チェック"""
        issues = []
        level_order = {"beginner": 0, "intermediate": 1, "advanced": 2}

        for content_id, metadata in self.contents.items():
            content_level = level_order[metadata.level]

            for prereq_id in metadata.prerequisites:
                if prereq_id in self.contents:
                    prereq_level = level_order[self.contents[prereq_id].level]

                    if prereq_level > content_level:
                        issues.append(
                            f"{content_id} ({metadata.level}) の前提条件 "
                            f"{prereq_id} ({self.contents[prereq_id].level}) "
                            f"のレベルが高すぎる"
                        )

        return issues
```

---

## 5. 検証とフィードバックループ

### 5.1 自動検証スイート

```python
class AutomatedValidationSuite:
    """自動検証スイート"""

    def __init__(self):
        self.validators = [
            self.validate_structure,
            self.validate_content_quality,
            self.validate_code_examples,
            self.validate_math_expressions,
            self.validate_references,
            self.validate_accessibility
        ]

    def run_all(self, content: str, metadata: dict) -> dict:
        """すべての検証を実行"""
        results = {}

        for validator in self.validators:
            validator_name = validator.__name__
            try:
                issues = validator(content, metadata)
                results[validator_name] = {
                    "status": "passed" if not issues else "failed",
                    "issues": issues
                }
            except Exception as e:
                results[validator_name] = {
                    "status": "error",
                    "error": str(e)
                }

        return results

    def validate_structure(self, content: str, metadata: dict) -> List[str]:
        """構造検証"""
        issues = []

        # 見出し階層チェック
        headings = re.findall(r'^(#{1,6})\s+(.+)$', content, re.MULTILINE)

        for i, (level, title) in enumerate(headings):
            if i > 0:
                prev_level = len(headings[i-1][0])
                curr_level = len(level)

                # 見出しレベルが2段階以上飛んでいないかチェック
                if curr_level - prev_level > 1:
                    issues.append(f"見出しレベルの飛躍: {prev_level} → {curr_level} at '{title}'")

        return issues

    def validate_code_examples(self, content: str, metadata: dict) -> List[str]:
        """コード例検証"""
        issues = []

        # コードブロック抽出
        code_blocks = re.findall(r'```(\w+)\n([\s\S]*?)```', content)

        for lang, code in code_blocks:
            if lang == "python":
                # Python構文チェック
                try:
                    compile(code, '<string>', 'exec')
                except SyntaxError as e:
                    issues.append(f"Python構文エラー: {e}")

        return issues

    def validate_math_expressions(self, content: str, metadata: dict) -> List[str]:
        """数式検証"""
        issues = []

        # 数式抽出
        inline_math = re.findall(r'\$([^\$]+)\$', content)
        display_math = re.findall(r'\$\$([^\$]+)\$\$', content)

        all_math = inline_math + display_math

        for expr in all_math:
            # 基本的なLaTeX構文チェック
            if expr.count('{') != expr.count('}'):
                issues.append(f"数式の括弧不一致: {expr[:50]}...")

            if expr.count('\\begin') != expr.count('\\end'):
                issues.append(f"数式環境の不一致: {expr[:50]}...")

        return issues

    def validate_references(self, content: str, metadata: dict) -> List[str]:
        """参考文献検証"""
        issues = []

        # 参考文献リンク抽出
        references = re.findall(r'\[([^\]]+)\]\(([^\)]+)\)', content)

        # DOIリンクチェック
        doi_pattern = r'https?://doi\.org/10\.\d{4,}'

        for text, url in references:
            if 'doi.org' in url and not re.match(doi_pattern, url):
                issues.append(f"不正なDOIリンク: {url}")

        return issues

    def validate_accessibility(self, content: str, metadata: dict) -> List[str]:
        """アクセシビリティ検証"""
        issues = []

        # 画像のalt属性チェック
        images = re.findall(r'!\[([^\]]*)\]\(([^\)]+)\)', content)

        for alt_text, url in images:
            if not alt_text.strip():
                issues.append(f"画像に代替テキストがありません: {url}")

        # 表のヘッダーチェック
        tables = re.findall(r'\|(.+)\|\n\|[-:\s|]+\|', content)

        for table in tables:
            if not table.strip():
                issues.append("表にヘッダー行がありません")

        return issues
```

### 5.2 継続的改善ループ

```python
class ContinuousImprovementLoop:
    """継続的改善ループ"""

    def __init__(self):
        self.validation_suite = AutomatedValidationSuite()
        self.improvement_history = []

    def improve_iteratively(
        self,
        content: str,
        metadata: dict,
        max_iterations: int = 3
    ) -> dict:
        """反復的改善"""

        for iteration in range(max_iterations):
            # 1. 検証実行
            validation_results = self.validation_suite.run_all(content, metadata)

            # 2. 問題集約
            all_issues = []
            for validator_name, result in validation_results.items():
                if result["status"] == "failed":
                    all_issues.extend(result["issues"])

            if not all_issues:
                # すべて合格
                return {
                    "status": "success",
                    "content": content,
                    "iterations": iteration + 1,
                    "history": self.improvement_history
                }

            # 3. 改善プロンプト生成
            improvement_prompt = self._generate_improvement_prompt(content, all_issues)

            # 4. コンテンツ改善
            improved_content = self.llm.generate(improvement_prompt)

            # 5. 履歴記録
            self.improvement_history.append({
                "iteration": iteration + 1,
                "issues": all_issues,
                "improvements": self._diff(content, improved_content)
            })

            content = improved_content

        # max_iterations到達
        return {
            "status": "partial_success",
            "content": content,
            "iterations": max_iterations,
            "remaining_issues": all_issues,
            "history": self.improvement_history
        }

    def _generate_improvement_prompt(self, content: str, issues: List[str]) -> str:
        """改善プロンプト生成"""
        return f"""以下のコンテンツに検出された問題を修正してください。

# 元のコンテンツ
{content}

# 検出された問題（優先度順）
{chr(10).join(f"{i+1}. {issue}" for i, issue in enumerate(issues))}

# 修正指示
- すべての問題を解決してください
- コンテンツの品質を維持しつつ、問題箇所のみを修正してください
- 修正後のコンテンツ全体を出力してください
"""

    def _diff(self, old_content: str, new_content: str) -> List[str]:
        """差分抽出（簡易版）"""
        old_lines = old_content.split('\n')
        new_lines = new_content.split('\n')

        changes = []
        for i, (old, new) in enumerate(zip(old_lines, new_lines)):
            if old != new:
                changes.append(f"Line {i+1}: {old} → {new}")

        return changes
```

---

## 6. 実践チェックリスト

### 6.1 コンテンツ生成前チェックリスト

```markdown
## コンテンツ生成前チェックリスト

### プロンプト設計
- [ ] テンプレートベース設計を採用
- [ ] 出力形式（JSON/Markdown/LaTeX）を明示
- [ ] 文字数・コード例数・演習数などの定量的要求を指定
- [ ] 品質基準（正確性・完全性・教育性）を明示
- [ ] 対象レベル（beginner/intermediate/advanced）を明確化

### メタデータ準備
- [ ] コンテンツID生成
- [ ] トピックとサブトピックの定義
- [ ] 前提条件の確認
- [ ] 参考文献リストの準備
- [ ] 想定文字数・コード例数の設定

### 環境準備
- [ ] LLM接続確認
- [ ] リトライマネージャー設定（max_retries, initial_delay等）
- [ ] 品質評価システム準備
- [ ] フォールバックメカニズム確認
```

### 6.2 コンテンツ生成中チェックリスト

```markdown
## コンテンツ生成中チェックリスト

### リアルタイム品質チェック
- [ ] 文字数が基準を満たしているか
- [ ] コード例が必要数含まれているか
- [ ] 数式が正しく記述されているか
- [ ] 参考文献が適切に引用されているか
- [ ] 構造（見出し階層）が適切か

### エラーハンドリング
- [ ] パースエラー検出時のプロンプト改善
- [ ] タイムアウト発生時のリトライ実行
- [ ] 品質基準未達時の再生成
- [ ] エラーパターンの記録

### 進捗監視
- [ ] 現在のリトライ回数確認
- [ ] 品質スコアのモニタリング
- [ ] 生成時間の追跡
```

### 6.3 コンテンツ生成後チェックリスト

```markdown
## コンテンツ生成後チェックリスト

### 多角的検証
- [ ] 構造検証（見出し階層、セクション構成）
- [ ] コンテンツ品質検証（正確性、完全性、教育性）
- [ ] コード例検証（構文チェック、実行可能性）
- [ ] 数式検証（LaTeX構文、表示確認）
- [ ] 参考文献検証（リンク有効性、DOI形式）
- [ ] アクセシビリティ検証（alt属性、表ヘッダー）

### Phase 3学術的品質評価
- [ ] 科学的正確性: 80点以上
- [ ] 完全性: 80点以上
- [ ] 教育品質: 80点以上
- [ ] 実装品質: 80点以上
- [ ] 総合スコア: 80点以上

### 改善サイクル
- [ ] 検出された問題の優先度付け
- [ ] 高優先度問題の修正
- [ ] 修正後の再検証
- [ ] Phase 7最終評価（90点以上）

### 公開前最終チェック
- [ ] メタデータ完全性確認
- [ ] Git commit準備
- [ ] レビュー履歴の保存
- [ ] 公開URL確認
```

---

## 7. まとめ

### 7.1 高品質アウトプットの6原則

1. **テンプレート駆動設計**: 再利用可能なプロンプトテンプレートで一貫性を確保
2. **多層品質保証**: Generation-Time + Post-Generation のハイブリッドアプローチ
3. **エラー耐性**: 指数バックオフリトライ + エラーパターン検出
4. **構造的一貫性**: 階層管理 + メタデータ駆動検証
5. **継続的改善**: 自動検証 + 反復的改善ループ
6. **明確な基準**: 定量的品質基準 + ゲート制御

### 7.2 Write_tb_AI vs MI Hub ベストプラクティス統合

| 要素 | Write_tb_AI | MI Hub | 推奨統合アプローチ |
|------|-------------|--------|-------------------|
| **プロンプト** | テンプレート方式 | 自由形式 | **テンプレート方式を採用** |
| **品質保証** | Generation-Time | Post-Generation | **両方を組み合わせ** |
| **リトライ** | 指数バックオフ（5回） | Agent-level fallback | **Write_tb_AI方式 + エージェントフォールバック** |
| **構造管理** | NetworkXグラフ | Markdown階層 | **グラフ + Markdown** |
| **数式処理** | MathEnvironmentProtector | 標準Markdown | **Protectorを移植** |
| **検証** | ContentQualityAssessor | 9フェーズ多角的検証 | **即時評価 + 段階的検証** |

### 7.3 次のステップ

**即座に実装可能（ROI高）**:
1. プロンプトテンプレートシステム導入（2-3日）
2. MathEnvironmentProtectorのMarkdown移植（1-2日）
3. Phase 2.5即時品質ゲート追加（3-4日）

**中期的実装（ROI中）**:
4. ContentQualityAssessorの統合（1週間）
5. 指数バックオフリトライの標準化（3-4日）
6. エラーパターン検出システム（1週間）

**長期的実装（基盤強化）**:
7. NetworkXベースの構造管理（2週間）
8. 自動検証スイートの完全実装（2週間）
9. 継続的改善ループの自動化（2週間）

---

**作成日**: 2025-10-16
**バージョン**: 1.0
**ベース分析**: `claudedocs/write_tb_ai_analysis.md`
**対象システム**: Write_tb_AI + MI Knowledge Hub

**推奨利用方法**:
1. 新規コンテンツ生成時の参照ガイド
2. 既存システムの品質改善計画策定
3. チーム内の品質基準共有
4. コードレビュー時のチェックリスト

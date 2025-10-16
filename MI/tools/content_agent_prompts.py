"""
Content Agent Prompt Templates

High-quality prompt templates for MI Knowledge Hub content generation.
Based on Write_tb_AI's template-based approach with strict output format constraints.

Author: Claude Code Subagents
Created: 2025-10-16
Version: 1.0
"""

from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class ContentRequirements:
    """コンテンツ要求仕様"""
    topic: str
    level: str  # "beginner" | "intermediate" | "advanced"
    target_audience: str
    min_words: int
    min_code_examples: int
    min_exercises: int
    min_references: int
    language: str = "Japanese"


class ContentAgentPrompts:
    """Content Agent用プロンプトテンプレート集"""

    @staticmethod
    def article_structure_template(requirements: ContentRequirements) -> str:
        """
        テンプレート1: 記事構造生成

        Write_tb_AIのprompt_book_titleに相当。
        記事の全体構造（章・セクション）を生成する。

        Args:
            requirements: コンテンツ要求仕様

        Returns:
            構造生成用プロンプト
        """
        level_descriptions = {
            "beginner": "学部2-3年生、MIの基礎概念を初めて学ぶ学生",
            "intermediate": "大学院生、基礎知識があり実践的手法を学ぶ学生",
            "advanced": "研究者、最先端技術や高度な理論を理解する専門家"
        }

        return f"""あなたはマテリアルズ・インフォマティクス（Materials Informatics）の専門家として、教育コンテンツの構造を設計します。

# 記事構成要求

**トピック**: {requirements.topic}
**対象レベル**: {requirements.level} ({level_descriptions[requirements.level]})
**対象読者**: {requirements.target_audience}
**言語**: {requirements.language}

# タスク

以下の要求を満たす記事構成を設計してください：

1. **学習目標の明確化**: 読者が記事を読み終えた後に習得できる具体的なスキル・知識
2. **章立ての設計**: 論理的な流れで段階的に理解を深められる構成
3. **セクション分割**: 各章を適切な粒度のセクションに分割

# 出力形式（必ず以下のJSON形式のみを出力）

**重要**: JSON以外の文字列（説明文、コメント等）は一切含めないでください。

```json
{{
  "title": "記事タイトル（50文字以内、日本語）",
  "subtitle": "サブタイトル（任意、80文字以内）",
  "learning_objectives": [
    "学習目標1（具体的な習得スキル）",
    "学習目標2",
    "学習目標3（最低3つ）"
  ],
  "chapters": [
    {{
      "chapter_number": 1,
      "chapter_title": "導入",
      "chapter_description": "章の概要（100文字程度）",
      "estimated_words": 500,
      "sections": [
        {{
          "section_number": 1,
          "section_title": "背景と動機",
          "section_description": "なぜこのトピックが重要か"
        }},
        {{
          "section_number": 2,
          "section_title": "本記事で学ぶこと",
          "section_description": "記事の全体像"
        }}
      ]
    }},
    {{
      "chapter_number": 2,
      "chapter_title": "理論的基礎",
      "chapter_description": "章の概要",
      "estimated_words": 1500,
      "sections": [
        {{
          "section_number": 1,
          "section_title": "基本概念",
          "section_description": "核となる理論"
        }},
        {{
          "section_number": 2,
          "section_title": "数学的定式化",
          "section_description": "数式による表現"
        }}
      ]
    }},
    {{
      "chapter_number": 3,
      "chapter_title": "実装と実践",
      "chapter_description": "章の概要",
      "estimated_words": 2000,
      "sections": [
        {{
          "section_number": 1,
          "section_title": "Pythonによる実装",
          "section_description": "コード例"
        }},
        {{
          "section_number": 2,
          "section_title": "実データでの検証",
          "section_description": "実践例"
        }}
      ]
    }},
    {{
      "chapter_number": 4,
      "chapter_title": "応用と発展",
      "chapter_description": "章の概要",
      "estimated_words": 1000,
      "sections": [
        {{
          "section_number": 1,
          "section_title": "応用事例",
          "section_description": "実際の研究例"
        }},
        {{
          "section_number": 2,
          "section_title": "今後の展望",
          "section_description": "発展的トピック"
        }}
      ]
    }}
  ],
  "metadata": {{
    "total_chapters": 4,
    "total_sections": 8,
    "estimated_total_words": 5000,
    "estimated_reading_time": "20 minutes"
  }}
}}
```

# 設計原則

1. **段階的複雑性**: 基礎 → 理論 → 実践 → 応用の流れ
2. **バランス**: 理論と実践の適切な配分（理論40%、実践40%、応用20%）
3. **明確性**: 各章・セクションの役割が明確
4. **完全性**: トピックを包括的にカバー

# レベル別要求

- **beginner**: 基礎概念を丁寧に、数式は最小限、コード例は詳細に
- **intermediate**: 理論と実践のバランス、数式による定式化、実践的コード
- **advanced**: 最先端研究、高度な数学的扱い、複雑な実装例

**出力**: 上記JSON形式のみ（説明文なし）
"""

    @staticmethod
    def section_detail_template(
        article_structure: Dict,
        chapter_number: int,
        section_number: int,
        requirements: ContentRequirements
    ) -> str:
        """
        テンプレート2: セクション詳細化

        Write_tb_AIのprompt_section_list_creationに相当。
        特定セクションの詳細構成を生成する。

        Args:
            article_structure: 記事全体構造（template 1の出力）
            chapter_number: 対象章番号
            section_number: 対象セクション番号
            requirements: コンテンツ要求仕様

        Returns:
            セクション詳細化プロンプト
        """
        chapter = article_structure["chapters"][chapter_number - 1]
        section = chapter["sections"][section_number - 1]

        return f"""あなたはマテリアルズ・インフォマティクスの専門家として、記事セクションの詳細構成を設計します。

# セクション情報

**記事タイトル**: {article_structure["title"]}
**章**: 第{chapter_number}章「{chapter["chapter_title"]}」
**セクション**: {chapter_number}.{section_number} 「{section["section_title"]}」
**セクション概要**: {section["section_description"]}
**対象レベル**: {requirements.level}

# タスク

このセクションの詳細構成を設計してください：

1. **サブセクション分割**: 内容を論理的な単位に分割
2. **要素配置**: 説明文、数式、コード例、図表の配置計画
3. **演習問題設計**: セクション理解を確認する問題

# 出力形式（必ず以下のJSON形式のみを出力）

```json
{{
  "section_title": "{section["section_title"]}",
  "subsections": [
    {{
      "subsection_number": 1,
      "subsection_title": "サブセクションタイトル",
      "content_elements": [
        {{
          "type": "text",
          "description": "導入説明（300字程度）"
        }},
        {{
          "type": "equation",
          "description": "主要な数式の説明",
          "latex": "$E = mc^2$のような形式で記述"
        }},
        {{
          "type": "code",
          "description": "コード例の目的",
          "language": "python",
          "code_purpose": "何を実装するか"
        }},
        {{
          "type": "diagram",
          "description": "図表の説明",
          "diagram_type": "flowchart|graph|table"
        }}
      ],
      "estimated_words": 500
    }}
  ],
  "exercises": [
    {{
      "exercise_number": 1,
      "question": "演習問題の内容",
      "difficulty": "easy|medium|hard",
      "answer_hint": "ヒント",
      "solution_outline": "解答の概要"
    }}
  ],
  "key_points": [
    "このセクションのキーポイント1",
    "キーポイント2"
  ],
  "prerequisites": [
    "前提知識1",
    "前提知識2"
  ],
  "references": [
    "参考文献1（著者、タイトル、年）",
    "参考文献2"
  ]
}}
```

# 設計原則

1. **論理的流れ**: 定義 → 説明 → 例 → 演習
2. **具体性**: 抽象概念には必ず具体例を添える
3. **実行可能性**: コード例は完全に実行可能
4. **検証可能性**: 演習問題で理解度を確認

# レベル別要求

- **beginner**:
  - サブセクションは小さく（1つ300-500字）
  - 数式は最小限、丁寧な説明
  - 演習は基本的（easy中心）

- **intermediate**:
  - サブセクションは中程度（1つ500-800字）
  - 数式と実装のバランス
  - 演習はmedium中心、一部hard

- **advanced**:
  - サブセクションは大きく（1つ800-1200字）
  - 高度な数学的扱い
  - 演習はhard中心、研究レベル

**出力**: 上記JSON形式のみ（説明文なし）
"""

    @staticmethod
    def content_generation_template(
        section_detail: Dict,
        requirements: ContentRequirements,
        context: Optional[Dict] = None
    ) -> str:
        """
        テンプレート3: コンテンツ生成

        Write_tb_AIのprompt_content_creationに相当。
        実際の本文、コード、演習を生成する。

        Args:
            section_detail: セクション詳細構成（template 2の出力）
            requirements: コンテンツ要求仕様
            context: 追加コンテキスト（参考文献、データセット等）

        Returns:
            コンテンツ生成プロンプト
        """
        context_info = ""
        if context:
            if "references" in context:
                context_info += f"\n\n# 参考文献\n"
                for ref in context["references"]:
                    context_info += f"- {ref}\n"
            if "datasets" in context:
                context_info += f"\n\n# 利用可能データセット\n"
                for ds in context["datasets"]:
                    context_info += f"- {ds}\n"

        return f"""あなたはマテリアルズ・インフォマティクスの専門家として、高品質な教育コンテンツを生成します。

# セクション情報

**セクションタイトル**: {section_detail["section_title"]}
**サブセクション数**: {len(section_detail["subsections"])}
**演習問題数**: {len(section_detail["exercises"])}
**対象レベル**: {requirements.level}
{context_info}

# タスク

以下の詳細構成に基づいて、完全なMarkdownコンテンツを生成してください。

## セクション詳細構成

{_format_section_detail(section_detail)}

# 出力形式（必ず以下のMarkdown形式のみを出力）

**重要**: Markdown以外の文字列（「以下がコンテンツです」等の説明文）は一切含めないでください。

```markdown
## {section_detail["section_title"]}

### サブセクション1のタイトル

本文（300-500字程度）。具体例を含め、段階的に説明する。

数式が必要な場合:
$$
E = mc^2
$$

インライン数式: $x = \\frac{{-b \\pm \\sqrt{{b^2-4ac}}}}{{2a}}$

コード例が必要な場合:

```python
import numpy as np
import matplotlib.pyplot as plt

# データ生成
x = np.linspace(0, 10, 100)
y = np.sin(x)

# プロット
plt.figure(figsize=(8, 6))
plt.plot(x, y, label='sin(x)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('正弦関数の例')
plt.legend()
plt.grid(True)
plt.show()
```

**コード解説**:
1. `numpy`でデータ生成
2. `matplotlib`で可視化
3. グリッド表示で見やすく

### 演習問題

**問題1** (難易度: easy)

以下の問いに答えてください。

(問題文)

<details>
<summary>ヒント</summary>

(ヒント)

</details>

<details>
<summary>解答例</summary>

(解答例)

```python
# 解答コード
```

</details>

### このセクションのまとめ

- キーポイント1
- キーポイント2
- キーポイント3
```

# 品質基準

## 必須要素
- [ ] 文字数: {requirements.min_words}字以上
- [ ] コード例: {requirements.min_code_examples}個以上（実行可能）
- [ ] 演習問題: {requirements.min_exercises}問以上
- [ ] 参考文献: {requirements.min_references}件以上

## コンテンツ品質
- [ ] **科学的正確性**: 最新の研究に基づく正確な情報
- [ ] **教育的配慮**: 段階的説明、具体例、図解
- [ ] **実行可能性**: すべてのコードが実際に動作
- [ ] **完全性**: トピックを包括的にカバー

## コード例の要件
1. **完全性**: コピペで実行可能（import文を含む）
2. **説明**: コメントと解説文で動作を明確化
3. **実用性**: 実際の問題解決に使える例
4. **エラーハンドリング**: 適切な例外処理

## 数式の要件
1. **正確性**: 数学的に正しい記述
2. **記法**: LaTeX形式（$ ... $または$$ ... $$）
3. **説明**: 数式の意味を文章で解説
4. **変数定義**: すべての変数の意味を明記

## 演習問題の要件
1. **難易度表示**: easy/medium/hardを明記
2. **段階的**: 易→難の順序
3. **ヒント**: 考え方の方向性を示す
4. **解答例**: 詳細な解答コード・解説

# レベル別ガイドライン

## beginner
- 専門用語には必ず説明を追加
- 数式は最小限、直感的説明を重視
- コード例は詳細なコメント付き
- 演習はeasy中心

## intermediate
- 理論と実践のバランス
- 数式による定式化
- 実践的なコード例
- 演習はmedium中心、一部hard

## advanced
- 最先端の研究内容
- 高度な数学的扱い
- 複雑な実装例
- 研究レベルの演習

# 注意事項

1. **参考文献の引用**: 必ず出典を明記（[著者, 年]形式）
2. **データセット参照**: 利用可能なデータセットを活用
3. **再現性**: コード例の環境（Python 3.10+, ライブラリバージョン）を明記
4. **アクセシビリティ**: 図表には代替テキスト、コード例には説明文

**出力**: 上記Markdown形式のみ（説明文なし）
"""


def _format_section_detail(section_detail: Dict) -> str:
    """セクション詳細をテキスト形式にフォーマット"""
    output = []

    for subsection in section_detail["subsections"]:
        output.append(f"\n### {subsection['subsection_title']}")
        output.append(f"推定文字数: {subsection['estimated_words']}字")
        output.append(f"\n要素構成:")

        for element in subsection["content_elements"]:
            if element["type"] == "text":
                output.append(f"  - 説明文: {element['description']}")
            elif element["type"] == "equation":
                output.append(f"  - 数式: {element['description']}")
                if "latex" in element:
                    output.append(f"    例: {element['latex']}")
            elif element["type"] == "code":
                output.append(f"  - コード例: {element['description']}")
                output.append(f"    目的: {element['code_purpose']}")
            elif element["type"] == "diagram":
                output.append(f"  - 図表: {element['description']}")
                output.append(f"    種類: {element['diagram_type']}")

    output.append(f"\n演習問題:")
    for exercise in section_detail["exercises"]:
        output.append(f"  {exercise['exercise_number']}. {exercise['question']}")
        output.append(f"     難易度: {exercise['difficulty']}")

    return "\n".join(output)


# Quick access functions for Content Agent

def get_structure_prompt(
    topic: str,
    level: str = "intermediate",
    target_audience: str = "undergraduate",
    min_words: int = 5000
) -> str:
    """記事構造生成プロンプトを取得"""
    requirements = ContentRequirements(
        topic=topic,
        level=level,
        target_audience=target_audience,
        min_words=min_words,
        min_code_examples=3,
        min_exercises=5,
        min_references=5
    )
    return ContentAgentPrompts.article_structure_template(requirements)


def get_section_detail_prompt(
    article_structure: Dict,
    chapter_number: int,
    section_number: int,
    level: str = "intermediate"
) -> str:
    """セクション詳細化プロンプトを取得"""
    requirements = ContentRequirements(
        topic=article_structure["title"],
        level=level,
        target_audience="undergraduate",
        min_words=500,
        min_code_examples=1,
        min_exercises=2,
        min_references=2
    )
    return ContentAgentPrompts.section_detail_template(
        article_structure, chapter_number, section_number, requirements
    )


def get_content_generation_prompt(
    section_detail: Dict,
    level: str = "intermediate",
    min_words: int = 2000,
    context: Optional[Dict] = None
) -> str:
    """コンテンツ生成プロンプトを取得"""
    requirements = ContentRequirements(
        topic=section_detail["section_title"],
        level=level,
        target_audience="undergraduate",
        min_words=min_words,
        min_code_examples=3,
        min_exercises=5,
        min_references=5
    )
    return ContentAgentPrompts.content_generation_template(
        section_detail, requirements, context
    )

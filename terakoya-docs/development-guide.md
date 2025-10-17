# AI寺子屋 - 開発ガイド

**作成日**: 2025年10月17日
**バージョン**: 2.0（汎用化版）

---

## プロジェクト概要

**AI寺子屋（AI Terakoya）**は、様々な学術トピックの教育および研究コミュニティプラットフォームです。

**主要な特徴:**
- 7つの専門Claude Code subagentsによるAI駆動型コンテンツ生成
- 質ゲート付き9フェーズ品質ワークフロー
- GitHub Pages上の静的サイト
- JSON/Markdownデータレイヤー
- モバイルファーストのレスポンシブデザイン
- **APIキー不要** - すべてClaude Codeセッション内で完結

---

## アーキテクチャ

### Claude Code Subagent-Based システム

このプロジェクトは**Claude Code subagents**を活用しています。すべてのコンテンツ生成、論文収集、品質保証はClaude Codeセッション内で完結します。

**メリット:**
- ✅ APIキー不要
- ✅ Claude Code環境内で完結
- ✅ ファイルシステムへの直接アクセス
- ✅ バージョン管理されたsubagent定義
- ✅ シームレスな協調作業
- ✅ レート制限なし

---

## プロジェクト構造

```
AI_Homepage/ (AI寺子屋プロジェクト)
├── .claude/
│   └── agents/              # 7つのsubagent定義（汎用）
│       ├── scholar-agent.md
│       ├── content-agent.md
│       ├── academic-reviewer.md
│       ├── tutor-agent.md
│       ├── data-agent.md
│       ├── design-agent.md
│       └── maintenance-agent.md
├── tools/                   # 汎用Pythonツール
│   ├── validate_data.py     # JSONデータ検証
│   ├── content_agent_prompts.py  # プロンプトテンプレート
│   ├── example_template_usage.py  # テンプレート使用例
│   ├── md_to_html.py        # 日本語Markdown→HTML変換
│   └── md_to_html_en.py     # 英語Markdown→HTML変換
├── terakoya-docs/           # 汎用ドキュメント
│   ├── content-creation-workflow.md  # 9フェーズワークフロー
│   └── development-guide.md  # このファイル
├── projects/                # トピック別プロジェクト
│   ├── MI/                  # マテリアルズ・インフォマティクス
│   │   ├── data/
│   │   ├── content/
│   │   └── reviews/
│   ├── NM/                  # ナノマテリアル
│   ├── PI/                  # プロセス・インフォマティクス
│   └── MLP/                 # 機械学習ポテンシャル
├── wp/
│   └── knowledge/
│       ├── jp/
│       │   ├── mi-introduction/
│       │   ├── nm-introduction/
│       │   ├── pi-introduction/
│       │   ├── mlp-introduction/
│       │   └── index.html   # AI寺子屋ポータル
│       └── en/
└── TERAKOYA.md              # プロジェクト全体ガイド
```

---

## 7つの専門Subagents

すべてのsubagentsは `.claude/agents/` ディレクトリに定義されています。

| Subagent | ファイル | 主な役割 | 使用ツール |
|----------|---------|---------|-----------|
| **Scholar Agent** | `scholar-agent.md` | 論文収集・要約 | Read, Write, Edit, Bash, WebSearch |
| **Content Agent** | `content-agent.md` | 記事生成（9フェーズ） | Read, Write, Edit, MultiEdit, Bash |
| **Academic Reviewer** | `academic-reviewer.md` | 品質保証（0-100点スコアリング） | Read, Write, Edit, Grep, Bash |
| **Tutor Agent** | `tutor-agent.md` | 対話的学習支援 | Read, Write, Grep, Bash |
| **Data Agent** | `data-agent.md` | データセット・ツール管理 | Read, Write, Edit, Bash, WebSearch |
| **Design Agent** | `design-agent.md` | UX最適化・アクセシビリティ | Read, Write, Edit, Grep, Bash |
| **Maintenance Agent** | `maintenance-agent.md` | 検証・監視 | Read, Write, Bash, Grep, Glob |

### Subagentの使用方法

```bash
# Task toolを使用してsubagentを呼び出す
Task(
    subagent_type="content-agent",
    description="Create tutorial article",
    prompt="Create a beginner-level article about [topic]"
)
```

---

## 重要なツール

### 1. データ検証ツール - `validate_data.py`

**目的**: JSON データファイルの構造と整合性を検証

**使用方法:**
```bash
python tools/validate_data.py
```

**出力例:**
```
====================================================
Data Validation
====================================================

Validating papers.json...
  ✅ Valid (25 entries)

Validating datasets.json...
  ✅ Valid (15 entries)

====================================================
Validation Summary
====================================================

✅ All validations passed!
```

### 2. プロンプトテンプレート - `content_agent_prompts.py`

**目的**: 高品質なコンテンツ生成のためのプロンプトテンプレート集

**主要テンプレート:**
1. **記事構造生成テンプレート** - 全体の章・セクション構成
2. **セクション詳細化テンプレート** - サブセクション分割と要素配置
3. **コンテンツ生成テンプレート** - Markdown本文とコード例生成

**使用方法:**
```python
from tools.content_agent_prompts import get_structure_prompt

prompt = get_structure_prompt(
    topic="Topic Name",
    level="intermediate",
    target_audience="graduate students",
    min_words=5000
)
```

### 3. Markdown→HTML変換ツール

- **`md_to_html.py`** - 日本語Markdown変換
- **`md_to_html_en.py`** - 英語Markdown変換

**使用方法:**
```bash
python tools/md_to_html.py
python tools/md_to_html_en.py
```

---

## 開発ワークフロー

### 新トピックプロジェクトの作成

#### Step 1: プロジェクトディレクトリ作成

```bash
mkdir -p projects/[TOPIC_NAME]/{data,content,reviews}
```

#### Step 2: データファイル初期化

```bash
# projects/[TOPIC_NAME]/data/ に以下を作成
touch papers.json datasets.json tutorials.json tools.json
```

**JSONスキーマ例（papers.json）:**
```json
[
  {
    "id": "paper_001",
    "title": "Paper title",
    "authors": ["Author A", "Author B"],
    "year": 2024,
    "journal": "Journal Name",
    "doi": "10.xxxx/xxxxx",
    "abstract": "Abstract text",
    "tags": ["tag1", "tag2"],
    "collected_at": "2025-10-17T12:00:00Z"
  }
]
```

#### Step 3: 論文収集

```bash
# Scholar Agentを使用
"Use scholar-agent to collect papers on '[topic] [keywords]' from the last 30 days"
```

#### Step 4: コンテンツ作成

```bash
# Content Agentを使用して9フェーズワークフローで記事作成
"Use content-agent to create a beginner-level article about [topic]"
```

**処理フロー:**
1. Phase 0-2: 初稿作成
2. Phase 3: Academic Reviewer第一次レビュー（≥80点）
3. Phase 4-6: 強化（図表、引用、コード）
4. Phase 7: Academic Reviewer最終レビュー（≥90点）
5. Phase 8-9: 公開準備

#### Step 5: Web公開用コンテンツ作成

```bash
# wp/knowledge/jp/[topic]-introduction/ ディレクトリ作成
mkdir -p wp/knowledge/jp/[topic]-introduction
mkdir -p wp/knowledge/en/[topic]-introduction

# Markdownファイル配置
# Markdown→HTML変換
python tools/md_to_html.py
python tools/md_to_html_en.py
```

---

## 品質基準

### パフォーマンス
- **Lighthouse Performance**: ≥95
- **Lighthouse Accessibility**: 100
- **First Contentful Paint**: <1.5s
- **Time to Interactive**: <3.0s

### アクセシビリティ
- WCAG 2.1 Level AA 準拠
- カラーコントラスト比 ≥4.5:1
- タッチターゲット ≥44px × 44px
- キーボードナビゲーション対応

### 学術コンテンツ品質
- 科学的正確性（Academic Reviewerによる検証済み）
- すべての次元で≥90点（公開時）
- 査読論文の参照推奨
- 実行可能なコード例
- 明確な学習目標

---

## 重要な注意事項

### Subagentの使用が必須

**以下のタスクは必ずSubagentを使用:**
- 記事作成・編集 → `content-agent`
- 品質レビュー → `academic-reviewer`
- 論文収集 → `scholar-agent`
- データ管理 → `data-agent`
- UX改善 → `design-agent`
- データ検証 → `maintenance-agent`
- 学習支援 → `tutor-agent`

**直接作業してよいタスク:**
- ファイル読み取り（Read tool）
- 検証スクリプト実行（`python tools/validate_data.py`）
- Git操作（commit, push）
- プロジェクトインフラ構築
- ドキュメント更新

### APIキー不要

このプロジェクトは**外部APIキーを一切必要としません**。すべてのsubagentsはClaude Codeセッション内で動作します。

---

## Git ワークフロー

```bash
# フィーチャーブランチを作成
git checkout -b feature/add-[topic]-article

# Subagentsを使用してコンテンツ生成

# 検証
python tools/validate_data.py

# コミット
git commit -m "Add [topic] article (academic-reviewer score: 92)"

# プッシュ
git push origin feature/add-[topic]-article
```

---

## トラブルシューティング

### Subagentが見つからない

```bash
# Agent定義ファイルの存在確認
ls .claude/agents/

# YAML frontmatterの確認
head .claude/agents/scholar-agent.md
```

### データ検証エラー

```bash
# 詳細な検証実行
python tools/validate_data.py

# JSONファイルの手動チェック
python -m json.tool projects/[TOPIC]/data/papers.json
```

### コンテンツが品質基準未達

1. `projects/[TOPIC]/reviews/` ディレクトリのレビューレポートを確認
2. すべてのHIGH優先度の問題に対処
3. 推奨事項を実装
4. Academic Reviewerに再提出

---

## 新トピック追加のクイックスタート

### 1. プロジェクト作成
```bash
mkdir -p projects/[TOPIC]/{data,content,reviews}
```

### 2. データ初期化
```bash
cd projects/[TOPIC]/data
echo '[]' > papers.json
echo '[]' > datasets.json
echo '[]' > tutorials.json
echo '[]' > tools.json
```

### 3. 論文収集
```
"Use scholar-agent to collect papers on '[topic] [keywords]'"
```

### 4. コンテンツ作成
```
"Use content-agent to create a beginner-level article about [topic]"
```

### 5. Web公開
```bash
mkdir -p wp/knowledge/jp/[topic]-introduction
mkdir -p wp/knowledge/en/[topic]-introduction
# Markdownファイル配置後
python tools/md_to_html.py
python tools/md_to_html_en.py
```

### 6. 検証
```bash
python tools/validate_data.py
```

### 7. Git コミット
```bash
git add .
git commit -m "Add [topic] project with comprehensive content"
git push
```

---

## リソース

### ドキュメント
- `TERAKOYA.md` - AI寺子屋プロジェクト全体ガイド
- `terakoya-docs/content-creation-workflow.md` - 9フェーズワークフロー詳細
- `terakoya-docs/development-guide.md` - このファイル

### Subagent定義
- `.claude/agents/*.md` - 各subagentの詳細仕様

---

## まとめ

AI寺子屋の開発環境は、以下の3つの柱で構成されています：

1. **7つの専門Subagents** - コンテンツ生成、品質保証、データ管理を自動化
2. **汎用Pythonツール** - データ検証、プロンプトテンプレート、HTML変換
3. **9フェーズ品質ワークフロー** - 学術レビューゲート付き高品質コンテンツ生成

**すべての操作はAPIキー不要でClaude Codeセッション内で完結します。**

新しいトピックを追加する際は、この構造に従って `projects/[TOPIC]/` ディレクトリを作成し、7つのsubagentsを活用してコンテンツを生成してください。

---

**最終更新**: 2025年10月17日
**作成者**: AI寺子屋開発チーム

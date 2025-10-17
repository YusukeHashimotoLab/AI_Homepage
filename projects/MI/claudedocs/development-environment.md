# MI Knowledge Hub - 開発環境まとめ

**作成日**: 2025年10月17日
**バージョン**: 1.0

---

## プロジェクト概要

**MI Knowledge Hub**は、マテリアルズ・インフォマティクス（Materials Informatics）の教育および研究コミュニティプラットフォームです。

**主要な特徴:**
- 7つの専門Claude Code subagentsによるAI駆動型コンテンツ生成
- 学術レビューゲート付き9フェーズ品質ワークフロー
- GitHub Pages上の静的サイト
- JSON/Markdownデータレイヤー
- モバイルファーストのレスポンシブデザイン
- **APIキー不要** - すべてClaude Codeセッション内で完結

---

## 重要な機能とツール

### 1. Claude Code Subagents（7つの専門エージェント）

すべてのsubagentsは`.claude/agents/`ディレクトリに定義されています。

| エージェント | ファイル | 主な役割 | 使用ツール |
|------------|---------|---------|-----------|
| **Scholar Agent** | `scholar-agent.md` | 論文収集・要約 | Read, Write, Edit, Bash, WebSearch |
| **Content Agent** | `content-agent.md` | 記事生成（9フェーズ） | Read, Write, Edit, MultiEdit, Bash |
| **Academic Reviewer** | `academic-reviewer.md` | 品質保証（0-100点スコアリング） | Read, Write, Edit, Grep, Bash |
| **Tutor Agent** | `tutor-agent.md` | 対話的学習支援 | Read, Write, Grep, Bash |
| **Data Agent** | `data-agent.md` | データセット・ツール管理 | Read, Write, Edit, Bash, WebSearch |
| **Design Agent** | `design-agent.md` | UX最適化・アクセシビリティ | Read, Write, Edit, Grep, Bash |
| **Maintenance Agent** | `maintenance-agent.md` | 検証・監視 | Read, Write, Bash, Grep, Glob |

**使用方法:**
```bash
# Task toolを使用してsubagentを呼び出す
Task(
    subagent_type="content-agent",
    description="Create MI article",
    prompt="Create a beginner-level article about Bayesian optimization in MI"
)
```

### 2. Pythonツール（`tools/`ディレクトリ）

#### 2.1 データ検証ツール - `validate_data.py`

**目的**: JSON データファイルの構造と整合性を検証

**機能:**
- papers.json, datasets.json, tutorials.json, tools.json の検証
- 必須フィールドのチェック
- 重複ID検出
- データ型検証

**使用方法:**
```bash
python tools/validate_data.py
```

**出力例:**
```
====================================================
MI Knowledge Hub - Data Validation
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

#### 2.2 コンテンツエージェント用プロンプトテンプレート - `content_agent_prompts.py`

**目的**: 高品質なコンテンツ生成のためのプロンプトテンプレート集

**主要なテンプレート:**

1. **記事構造生成テンプレート** (`article_structure_template`)
   - 記事全体の章・セクション構成を設計
   - 学習目標の明確化
   - 推定文字数・読了時間の算出

2. **セクション詳細化テンプレート** (`section_detail_template`)
   - サブセクション分割
   - 要素配置（説明文、数式、コード、図表）
   - 演習問題設計

3. **コンテンツ生成テンプレート** (`content_generation_template`)
   - 実際のMarkdown本文生成
   - 実行可能なPythonコード例
   - 演習問題とヒント・解答

**使用方法:**
```python
from tools.content_agent_prompts import get_structure_prompt

prompt = get_structure_prompt(
    topic="Bayesian Optimization in Materials Informatics",
    level="intermediate",
    target_audience="graduate students",
    min_words=5000
)
```

**品質基準:**
- 文字数: 指定文字数以上
- コード例: 実行可能（import文を含む）
- 演習問題: 難易度表示・ヒント・解答例
- 参考文献: 適切な引用形式

#### 2.3 テンプレート使用例 - `example_template_usage.py`

**目的**: プロンプトテンプレートの実際の使用例を示すサンプルコード

**内容:**
- 3つのテンプレートの具体的な使用方法
- レベル別（beginner/intermediate/advanced）の例
- JSON出力のパース方法

### 3. データ管理（`data/`ディレクトリ）

#### 3.1 JSONデータファイル

| ファイル | 内容 | 必須フィールド |
|---------|------|--------------|
| `papers.json` | 研究論文メタデータ | id, title, authors, year, journal, doi, abstract, tags, collected_at |
| `datasets.json` | データセット情報 | id, name, description, url, data_types, size, license, updated_at |
| `tutorials.json` | チュートリアル情報 | id, title, description, level, difficulty, estimated_time, notebook_url, topics, prerequisites |
| `tools.json` | ツール・ライブラリ情報 | id, name, description, url, category, language, license, tags |

**データスキーマ例（papers.json）:**
```json
{
  "id": "paper_001",
  "title": "Bayesian Optimization for Materials Discovery",
  "authors": ["Smith, J.", "Doe, A."],
  "year": 2024,
  "journal": "Nature Materials",
  "doi": "10.1038/nmat.xxxx",
  "abstract": "Abstract text...",
  "tags": ["bayesian-optimization", "materials-discovery"],
  "collected_at": "2025-10-15T12:00:00Z"
}
```

### 4. コンテンツ生成ワークフロー（9フェーズ）

**Phase 0-2: 初期ドラフト作成**
- Content Agent が記事構造・セクション・本文を生成

**Phase 3: 第一次学術レビュー（80点ゲート）**
- Academic Reviewer が0-100点でスコアリング
- 80点未満 → Phase 1に戻る（大幅修正）
- 80点以上 → 次フェーズへ

**Phase 4-6: 強化フェーズ**
- Content Agent + Design Agent + Data Agent が協力
- 図表追加、引用強化、アクセシビリティ改善

**Phase 7: 最終学術レビュー（90点ゲート）**
- Academic Reviewer が再スコアリング
- 90点以上 → 承認
- 80-89点 → Phase 4に戻る（軽微な修正）
- 80点未満 → Phase 1に戻る（大幅修正）

**Phase 8-9: 最終確認・公開**
- 最終チェック
- `content/basics/`, `content/methods/`, `content/advanced/`, `content/applications/` に配置

**品質次元（4つ）:**
1. Scientific Accuracy（科学的正確性）
2. Clarity & Accessibility（明確性・アクセシビリティ）
3. References & Citations（参考文献・引用）
4. Code Quality & Reproducibility（コード品質・再現性）

### 5. ディレクトリ構造

```
MI/
├── .claude/
│   ├── agents/              # 7つのsubagent定義
│   │   ├── scholar-agent.md
│   │   ├── content-agent.md
│   │   ├── academic-reviewer.md
│   │   ├── tutor-agent.md
│   │   ├── data-agent.md
│   │   ├── design-agent.md
│   │   └── maintenance-agent.md
│   └── settings.local.json  # Claude Code設定
├── assets/                  # CSS/JS/画像
├── content/                 # Markdown記事
│   ├── basics/              # 初級コンテンツ
│   ├── methods/             # 中級メソッド
│   ├── advanced/            # 上級トピック
│   └── applications/        # ケーススタディ
├── data/                    # JSONデータ
│   ├── papers.json
│   ├── datasets.json
│   ├── tutorials.json
│   └── tools.json
├── tools/                   # Python検証・テンプレート
│   ├── validate_data.py
│   ├── content_agent_prompts.py
│   └── example_template_usage.py
├── reviews/                 # 学術レビューレポート
├── claudedocs/              # プロジェクトドキュメント
│   ├── requirements.md
│   ├── content-creation-procedure.md
│   └── development-environment.md (このファイル)
├── CLAUDE.md                # Claude Code用プロジェクトガイド
├── README.md                # プロジェクト概要
└── requirements.txt         # Python依存関係
```

---

## 開発ワークフロー

### 1. 論文収集

```bash
# Scholar Agentを使用
"Use scholar-agent to collect papers on 'bayesian optimization materials' from the last 30 days"
```

**結果:**
- `data/papers.json` に新規論文追加
- 重複チェック済み
- メタデータ抽出済み

### 2. 教育コンテンツ作成

```bash
# Content Agentを使用
"Use content-agent to create a beginner-level article about Gaussian Process Regression"
```

**処理:**
1. Phase 0-2: 初期ドラフト生成
2. Phase 3: Academic Reviewerによる第一次レビュー（≥80点）
3. Phase 4-6: 強化（図表、引用、コード例）
4. Phase 7: Academic Reviewerによる最終レビュー（≥90点）
5. Phase 8-9: 公開準備

**出力:**
- `content/basics/gaussian_process_regression.md`
- `reviews/gaussian_process_regression_phase3_review.md`
- `reviews/gaussian_process_regression_phase7_review.md`

### 3. データセット・ツール追加

```bash
# Data Agentを使用
"Use data-agent to add OQMD database to datasets.json"
```

**結果:**
- `data/datasets.json` 更新
- メタデータ完全性チェック済み

### 4. データ検証

```bash
# Pythonツールを直接実行
python tools/validate_data.py
```

**チェック項目:**
- JSON構文エラー
- 必須フィールドの欠落
- 重複ID
- 空値警告

### 5. システムヘルスチェック

```bash
# Maintenance Agentを使用
"Use maintenance-agent to validate all data and check links"
```

**実行内容:**
- JSON構造検証
- 全URLのアクセシビリティチェック
- Lighthouseスコア確認
- 健全性レポート生成

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
- ドキュメント更新（CLAUDE.md, README.md）

### APIキー不要

このプロジェクトは**外部APIキーを一切必要としません**。すべてのsubagentsはClaude Codeセッション内で動作します。

### バージョン管理

すべての変更はGitで管理されています。

**推奨Gitワークフロー:**
```bash
# フィーチャーブランチを作成
git checkout -b feature/add-quantum-ml-article

# Subagentsを使用してコンテンツ生成

# 検証
python tools/validate_data.py

# コミット
git commit -m "Add quantum ML article (academic-reviewer score: 92)"

# プッシュ
git push origin feature/add-quantum-ml-article
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
python -m json.tool data/papers.json
```

### コンテンツが品質基準未達

1. `reviews/` ディレクトリのレビューレポートを確認
2. すべてのHIGH優先度の問題に対処
3. 推奨事項を実装
4. Academic Reviewerに再提出

---

## リソース

### ドキュメント
- `CLAUDE.md` - Claude Code用プロジェクトガイド（最も重要）
- `README.md` - クイックスタートガイド
- `claudedocs/requirements.md` - 包括的要件（v1.1）
- `claudedocs/content-creation-procedure.md` - 9フェーズワークフロー詳細

### Subagent定義
- `.claude/agents/*.md` - 各subagentの詳細仕様

---

## まとめ

MI Knowledge Hubの開発環境は、以下の3つの柱で構成されています：

1. **7つの専門Subagents** - コンテンツ生成、品質保証、データ管理を自動化
2. **Pythonツール** - データ検証、プロンプトテンプレート、テスト
3. **9フェーズ品質ワークフロー** - 学術レビューゲート付き高品質コンテンツ生成

**すべての操作はAPIキー不要でClaude Codeセッション内で完結します。**

---

**最終更新**: 2025年10月17日
**作成者**: Dr. Yusuke Hashimoto
**連絡先**: yusuke.hashimoto.b8@tohoku.ac.jp

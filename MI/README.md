# MI Knowledge Hub

マテリアルズ・インフォマティクス（Materials Informatics）の包括的な学習プラットフォーム

**Powered by Claude Code Subagents** - 7つの専門AIエージェントがコンテンツを自動生成・管理

---

## 🎯 プロジェクト概要

学生から産業界技術者まで、MI学習者と実践者のための包括的な知識サイト。
**Claude Code subagents**による自動コンテンツ生成が特徴で、**APIキー不要**で運用できます。

**主な機能:**
- 📚 段階的学習コンテンツ (入門→中級→応用)
- 🤖 7つの専門subagentsによる自動生成・品質管理
- 💻 実行可能なJupyterチュートリアル
- 📄 論文情報自動収集・要約
- 🗂️ データセット・ツール情報管理
- ✅ 厳格な学術レビュー (80点・90点の2段階ゲート)

---

## 🏗️ アーキテクチャ: Claude Code Subagent-Based

```
Claude Code Session内
  ├─ scholar-agent: 論文収集・要約
  ├─ content-agent: 記事生成（9フェーズワークフロー）
  ├─ academic-reviewer: 品質保証（0-100点採点）
  ├─ tutor-agent: 対話型学習サポート
  ├─ data-agent: データセット・ツール管理
  ├─ design-agent: UX最適化・アクセシビリティ
  └─ maintenance-agent: バリデーション・監視
     ↓
静的サイト (HTML/CSS/JS + GitHub Pages)
     ↓
データ層 (JSON/Markdown)
```

**重要:** 外部API不要。全てClaude Codeセッション内で完結。

---

## 🚀 クイックスタート

### 環境構築

```bash
# 1. リポジトリクローン
git clone <repository-url>
cd MI

# 2. Python環境構築（検証ツール用、オプション）
python -m venv .venv
source .venv/bin/activate  # macOS/Linux

# 3. 最小限の依存関係インストール
pip install -r requirements.txt

# 4. APIキー不要！ Claude Codeで開いてsubagentsを使用
```

### ローカルプレビュー

```bash
# 静的サイトをローカルで確認
python -m http.server 8000
# http://localhost:8000 にアクセス
```

### Subagentの使い方

Claude Codeセッションで以下のように指示します：

```
# 論文を収集
"Use scholar-agent to collect papers on bayesian optimization for materials"

# 記事を生成（9フェーズワークフロー自動実行）
"Use content-agent to create an intermediate article about Bayesian optimization"

# 品質レビュー
"Use academic-reviewer to review content/methods/bayesian_optimization.md"

# データ検証
"Use maintenance-agent to validate all data"

# 対話型学習
"Ask tutor-agent to explain Bayesian optimization"
```

---

## 📁 ディレクトリ構造

```
MI/
├── .claude/
│   └── agents/              # 7つのsubagent定義（Markdown + YAML）
│       ├── scholar-agent.md
│       ├── content-agent.md
│       ├── academic-reviewer.md
│       ├── tutor-agent.md
│       ├── data-agent.md
│       ├── design-agent.md
│       └── maintenance-agent.md
├── assets/                  # CSS/JS/Images
│   ├── css/                 # モバイルファースト responsive CSS
│   ├── js/                  # データ駆動型 JavaScript
│   └── images/
├── content/                 # Markdown記事
│   ├── basics/              # 入門コンテンツ
│   ├── methods/             # 中級手法
│   ├── advanced/            # 応用トピック
│   └── applications/        # ケーススタディ
├── data/                    # JSON データファイル
│   ├── papers.json
│   ├── datasets.json
│   ├── tutorials.json
│   └── tools.json
├── pages/                   # 静的HTMLページ
├── notebooks/               # Jupyterチュートリアル
├── tools/                   # 検証ユーティリティ
│   └── validate_data.py     # JSON検証スクリプト
├── reviews/                 # 学術レビューレポート
├── claudedocs/              # プロジェクト文書
│   ├── requirements.md
│   └── content-creation-procedure.md
├── index.html               # ホームページ
├── CLAUDE.md                # Claude Code向けガイド
└── requirements.txt         # 最小限の依存関係
```

---

## 🤖 7つのSubagent

| Agent | 役割 | 主なツール |
|-------|------|-----------|
| **scholar-agent** | 論文収集・要約 | Read, Write, Edit, WebSearch |
| **content-agent** | 記事生成（9フェーズ） | Read, Write, Edit, MultiEdit |
| **academic-reviewer** | 品質保証（0-100点採点） | Read, Write, Edit, Grep |
| **tutor-agent** | 対話型学習サポート | Read, Write, Grep |
| **data-agent** | データセット・ツール管理 | Read, Write, Edit, WebSearch |
| **design-agent** | UX最適化・アクセシビリティ | Read, Write, Edit, Grep |
| **maintenance-agent** | バリデーション・監視 | Read, Write, Bash, Grep, Glob |

各subagentの詳細は `.claude/agents/` 内のMarkdownファイルを参照。

---

## 📝 9フェーズ品質ワークフロー

content-agentによる記事生成は厳格な品質管理プロセスに従います：

```
Phase 0: 計画（学習目標・ターゲット定義）
  ↓
Phase 1-2: 初稿作成
  ↓
Phase 3: Academic Review #1 （≥80点必須）← ゲート
  ↓ (合格)
Phase 4-6: 改善・拡充（design-agent、data-agent協力）
  ↓
Phase 7: Academic Review #2 （≥90点必須）← ゲート
  ↓ (合格)
Phase 8-9: 最終チェック・公開
```

**品質ゲート:**
- Phase 3: 80点未満 → Phase 1へ戻る（大幅修正）
- Phase 7: 80-89点 → Phase 4へ戻る（軽微な修正）
- Phase 7: 90点以上 → 承認・公開

---

## ✅ データ検証

```bash
# 全JSONファイルを検証
python tools/validate_data.py

# 出力例:
# ============================================================
# MI Knowledge Hub - Data Validation
# ============================================================
#
# Validating papers.json...
#   ✅ Valid (3 entries)
#
# Validating datasets.json...
#   ✅ Valid (4 entries)
#
# Validating tutorials.json...
#   ✅ Valid (3 entries)
#
# Validating tools.json...
#   ✅ Valid (6 entries)
#
# ============================================================
# Validation Summary
# ============================================================
#
# ✅ All validations passed!
```

---

## 🎨 品質基準

### Performance Targets
- **Lighthouse Performance**: ≥95
- **Lighthouse Accessibility**: 100 (WCAG 2.1 Level AA)
- **First Contentful Paint**: <1.5s
- **Time to Interactive**: <3.0s

### Academic Quality
- 全記事: academic-reviwerスコア ≥90点
- 科学的正確性検証済み
- 査読論文参照
- 実行可能コード例

### Mobile Optimization
- タッチターゲット ≥44px × 44px (Apple HIG)
- レスポンシブデザイン (mobile-first)
- 横スクロールなし

---

## 📖 ドキュメント

**必読:**
1. **[CLAUDE.md](CLAUDE.md)** - Claude Code向けプロジェクトガイド
2. **[requirements.md](claudedocs/requirements.md)** - 包括的要件定義書
3. **[content-creation-procedure.md](claudedocs/content-creation-procedure.md)** - 9フェーズワークフロー詳細
4. **[.claude/agents/*.md](.claude/agents/)** - 各subagent仕様

---

## 🛠️ 技術スタック

**Frontend:**
- HTML5/CSS3/JavaScript (Vanilla)
- Mobile-First Responsive Design
- データ駆動型レンダリング（JSON → HTML）

**Content Generation:**
- Claude Code Subagents （APIキー不要）
- 9フェーズ品質ワークフロー
- 自動学術レビュー

**Development Tools:**
- Python 3.11+ （検証ツールのみ）
- Jupyter Notebooks （チュートリアル）

**Deployment:**
- GitHub Pages (static hosting)
- GitHub Actions (CI/CD)

---

## 🗓️ 開発フェーズ

- **Phase 1 (Month 1-2)**: 基盤構築、Subagent MVP ✅
- **Phase 2 (Month 3-5)**: コンテンツ作成、自動化
- **Phase 3 (Month 6)**: インタラクティブ機能
- **Phase 4 (Month 7+)**: 最適化・拡張

---

## 🤝 コントリビューション

1. Feature branchを作成
2. Claude Code subagentsでコンテンツ生成
3. academic-reviewerで品質確認（≥90点）
4. 検証: `python tools/validate_data.py`
5. Pull Requestを作成

---

## 📄 ライセンス

MIT License (予定)

---

## 📞 お問い合わせ

**Dr. Yusuke Hashimoto**
- **Email**: yusuke.hashimoto.b8@tohoku.ac.jp
- **Institution**: Tohoku University
- **Website**: [AI_Homepage](../en/index.html)

---

**開発期間**: 2025年10月〜 (6ヶ月間)
**Architecture Version**: 2.0 (Claude Code Subagent-based)
**Last Updated**: 2025-10-15

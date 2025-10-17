# AI寺子屋（AI Terakoya）プロジェクト

**バージョン**: 2.0
**最終更新**: 2025年10月17日

---

## プロジェクト概要

**AI寺子屋**は、様々な学術トピックの高品質な教育コンテンツを、Claude Code subagentsの協力により作成・公開するプラットフォームです。

### 特徴

- ✅ **7つの専門subagents** - Scholar, Content, Academic Reviewer, Tutor, Data, Design, Maintenance
- ✅ **9フェーズ品質ワークフロー** - 学術レビューゲート（80点/90点）付き
- ✅ **APIキー不要** - すべてClaude Codeセッション内で完結
- ✅ **トピック非依存** - MI, NM, PI, MLP など任意のトピックに対応
- ✅ **多言語対応** - 日本語・英語のコンテンツ生成
- ✅ **静的サイト** - GitHub Pages でホスティング

---

## プロジェクト構造

```
AI_Homepage/ (AI寺子屋)
├── .claude/
│   └── agents/              # 7つの汎用subagent定義
├── tools/                   # 汎用Pythonツール
│   ├── validate_data.py
│   ├── content_agent_prompts.py
│   ├── md_to_html.py
│   └── md_to_html_en.py
├── terakoya-docs/           # 汎用ドキュメント
│   ├── content-creation-workflow.md
│   └── development-guide.md
├── projects/                # トピック別プロジェクト
│   ├── MI/                  # マテリアルズ・インフォマティクス
│   ├── NM/                  # ナノマテリアル（将来）
│   ├── PI/                  # プロセス・インフォマティクス（将来）
│   └── MLP/                 # 機械学習ポテンシャル（将来）
├── wp/
│   ├── knowledge/           # 公開用Webコンテンツ
│   │   ├── jp/
│   │   │   ├── index.html   # AI寺子屋ポータル
│   │   │   ├── mi-introduction/
│   │   │   ├── nm-introduction/
│   │   │   ├── pi-introduction/
│   │   │   └── mlp-introduction/
│   │   └── en/
│   └── private/             # 研究室Webページ
└── TERAKOYA.md              # このファイル
```

---

## 主要コンポーネント

### 1. Subagents（`.claude/agents/`）

7つの専門subagentsがコンテンツ作成を協力して実行：

| Subagent | 役割 | 主要フェーズ |
|----------|------|------------|
| Scholar | 論文収集・要約・事実確認 | 0, 1, 3, 7, 8 |
| Content | 記事構造設計・執筆 | 0, 2 |
| Academic Reviewer | 学術レビュー・品質保証 | 3, 7, 8, 9 |
| Tutor | 教育的視点・FAQ作成 | 0, 1, 2, 4, 8 |
| Data | 実装検証・リソース管理 | 0, 1, 5, 8 |
| Design | UX最適化・アクセシビリティ | 0, 6, 8 |
| Maintenance | 技術的品質保証 | 5, 8, 9 |

**使用方法:**
```
"Use [subagent-name] to [task description]"

例:
"Use scholar-agent to collect papers on 'quantum computing materials' from the last 30 days"
"Use content-agent to create a beginner-level article about Bayesian optimization"
```

### 2. Pythonツール（`tools/`）

- **`validate_data.py`** - JSONデータ検証
- **`content_agent_prompts.py`** - 高品質プロンプトテンプレート
- **`example_template_usage.py`** - テンプレート使用例
- **`md_to_html.py`** - 日本語Markdown→HTML変換
- **`md_to_html_en.py`** - 英語Markdown→HTML変換

### 3. トピックプロジェクト（`projects/`）

各トピックは独立したディレクトリで管理：

```
projects/[TOPIC]/
├── data/
│   ├── papers.json
│   ├── datasets.json
│   ├── tutorials.json
│   └── tools.json
├── content/              # Markdown記事
├── reviews/              # Academic Reviewerレポート
└── claudedocs/           # トピック固有ドキュメント
```

---

## 9フェーズワークフロー

詳細は `terakoya-docs/content-creation-workflow.md` を参照。

```
Phase 0: 企画・準備 (30-60分)
  ↓
Phase 1: 多角的情報収集 (2-4時間)
  ↓
Phase 2: 初稿作成 (3-5時間)
  ↓
Phase 3: 学術レビュー第1サイクル (≥80点)
  ↓
Phase 4: 教育的レビュー
  ↓
Phase 5: 実装検証
  ↓
Phase 6: UX最適化
  ↓
Phase 7: 学術レビュー第2サイクル (≥90点)
  ↓
Phase 8: 統合品質保証
  ↓
Phase 9: 最終承認・公開
```

**品質ゲート:**
- Phase 3: Academic Review ≥ 80点
- Phase 7: Academic Review ≥ 90点
- Phase 8: 総合スコア ≥ 90点

---

## 新トピック追加方法

### Step 1: プロジェクトディレクトリ作成

```bash
mkdir -p projects/[TOPIC]/{data,content,reviews,claudedocs}
```

### Step 2: データファイル初期化

```bash
cd projects/[TOPIC]/data
echo '[]' > papers.json
echo '[]' > datasets.json
echo '[]' > tutorials.json
echo '[]' > tools.json
```

### Step 3: 論文収集

```
"Use scholar-agent to collect papers on '[topic] [keywords]' from the last 30 days"
```

### Step 4: コンテンツ作成

```
"Use content-agent to create a beginner-level article about [topic]"
```

**9フェーズワークフローが自動実行されます。**

### Step 5: Web公開用コンテンツ作成

```bash
# ディレクトリ作成
mkdir -p wp/knowledge/jp/[topic]-introduction
mkdir -p wp/knowledge/en/[topic]-introduction

# Markdown→HTML変換
python tools/md_to_html.py
python tools/md_to_html_en.py
```

### Step 6: 検証

```bash
python tools/validate_data.py
```

### Step 7: コミット

```bash
git add .
git commit -m "Add [topic] project"
git push
```

---

## 現在のトピック

### MI（マテリアルズ・インフォマティクス）

**ステータス**: 完了

**コンテンツ:**
- 4章構成の入門シリーズ
- 日本語・英語版
- 公開URL: https://yusukehashimotolab.github.io/wp/knowledge/jp/mi-introduction/

**プロジェクトディレクトリ:** `projects/MI/`

### NM（ナノマテリアル）

**ステータス**: 計画中

**次のステップ:**
1. `projects/NM/` ディレクトリ作成
2. Scholar Agentで論文収集
3. Content Agentで記事作成

### PI（プロセス・インフォマティクス）

**ステータス**: Web公開済み（記事のみ）

**次のステップ:**
1. `projects/PI/` ディレクトリ作成
2. 論文・データ収集

### MLP（機械学習ポテンシャル）

**ステータス**: Web公開済み（記事のみ）

**次のステップ:**
1. `projects/MLP/` ディレクトリ作成
2. 論文・データ収集

---

## 品質基準

### 学術的品質
- 論文引用: 15-20件（最新5年以内を70%以上）
- 事実誤認: ゼロ
- 数式エラー: ゼロ

### 教育的品質
- 学習目標達成率: 90%以上
- 演習問題: 各セクション1問以上
- FAQ: 20項目以上

### 実装品質
- コード実行成功率: 100%
- Pylintスコア: 9.0以上
- テスト環境: 4種類で検証済み

### UX品質
- Lighthouse Performance: ≥95
- Lighthouse Accessibility: 100
- Flesch Reading Ease: ≥60
- モバイル対応: 完全

---

## 重要なルール

### ✅ Subagentの使用が必須

以下のタスクは**必ずSubagentを使用**:
- 記事作成・編集 → `content-agent`
- 品質レビュー → `academic-reviewer`
- 論文収集 → `scholar-agent`
- データ管理 → `data-agent`
- UX改善 → `design-agent`
- データ検証 → `maintenance-agent`
- 学習支援 → `tutor-agent`

### ✅ 直接作業してよいタスク

- ファイル読み取り（Read tool）
- 検証スクリプト実行（`python tools/validate_data.py`）
- Git操作（commit, push）
- プロジェクトインフラ構築
- ドキュメント更新

### ✅ APIキー不要

このプロジェクトは**外部APIキーを一切必要としません**。すべてのsubagentsはClaude Codeセッション内で動作します。

---

## ドキュメント

### 必読ドキュメント

1. **`terakoya-docs/content-creation-workflow.md`**
   - 9フェーズワークフロー詳細
   - 各フェーズのタスク・品質ゲート

2. **`terakoya-docs/development-guide.md`**
   - 開発環境セットアップ
   - 新トピック追加方法
   - トラブルシューティング

### トピック固有ドキュメント

- `projects/MI/claudedocs/` - MIプロジェクト固有ドキュメント
- `MI/CLAUDE.md` - MIプロジェクトのClaude Codeガイド
- `MI/README.md` - MIプロジェクト概要

---

## Git ワークフロー

```bash
# フィーチャーブランチを作成
git checkout -b feature/add-[topic]-content

# Subagentsでコンテンツ生成

# 検証
python tools/validate_data.py

# コミット
git commit -m "Add [topic] content (academic-reviewer score: XX)"

# プッシュ
git push origin feature/add-[topic]-content
```

---

## トラブルシューティング

### Subagentが見つからない

```bash
ls .claude/agents/
# 7つのsubagent定義ファイルがあることを確認
```

### データ検証エラー

```bash
python tools/validate_data.py
# エラーメッセージに従って修正
```

### コンテンツが品質基準未達

1. `projects/[TOPIC]/reviews/` のレビューレポート確認
2. HIGH優先度の問題に対処
3. Academic Reviewerに再提出

---

## コントリビュート

新しいトピックを追加する場合：

1. **事前準備**: トピック名とスコープを決定
2. **プロジェクト作成**: `projects/[TOPIC]/` ディレクトリ作成
3. **コンテンツ生成**: 7つのsubagentsを活用
4. **品質確認**: 90点以上のスコア取得
5. **Web公開**: `wp/knowledge/` に配置
6. **Git commit**: 変更をコミット

---

## ライセンス

このプロジェクトは**CC BY 4.0**ライセンスの下で公開されています。

**許可:**
- ✅ 自由な閲覧・ダウンロード
- ✅ 教育利用（授業、勉強会等）
- ✅ 改変・派生物作成

**条件:**
- 📌 著者クレジット必須
- 📌 改変時は明示
- 📌 商用利用前に連絡

---

## 連絡先

**プロジェクトリーダー**: Dr. Yusuke Hashimoto
**所属**: 東北大学 学際科学フロンティア研究所
**Email**: yusuke.hashimoto.b8@tohoku.ac.jp

---

**AI寺子屋で、最高品質の教育コンテンツを共に創りましょう！**

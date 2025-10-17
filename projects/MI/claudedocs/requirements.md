# マテリアルズインフォマティクス知見サイト - 要件定義書

**プロジェクト名**: MI Knowledge Hub
**バージョン**: 1.1
**作成日**: 2025-10-15
**対象期間**: 6ヶ月 (Phase 1-3)
**ステークホルダー**: Dr. Yusuke Hashimoto (Tohoku University), 学生, 産業界技術者

---

## 1. プロジェクト概要

### 1.1 プロジェクトの背景

マテリアルズインフォマティクス(MI)は、材料科学とデータサイエンスを融合した急成長分野であり、教育リソースと実践的知識の体系的な集約が求められている。本プロジェクトは、学習者と実務者の両方に対応する包括的な知識プラットフォームを構築する。

### 1.2 プロジェクトの目的

- **教育**: MI基礎から応用までの段階的学習リソース提供
- **コミュニティ**: 学生・研究者・産業界の知識共有と交流促進
- **研究発信**: 東北大学研究室の成果とMI最新動向の発信
- **自動化**: AIエージェントによるコンテンツ生成・更新・保守の効率化

### 1.3 既存システムとの関係

- **AI_Homepage**: 研究室公式ホームページ(Dr. Hashimoto個人・研究室情報)
- **MI Knowledge Hub**: 独立運用の専門知識サイト(MIコミュニティ向け)
- **連携方式**: 相互リンク、デザインシステム共有、共通インフラ(GitHub Pages)

---

## 2. ターゲットユーザーとニーズ

### 2.1 ユーザーペルソナ

#### ペルソナ1: 学部生 (材料系)
- **背景**: Python基礎あり、機械学習は未経験、材料科学の基礎知識保有
- **目的**: MIの基礎理論学習、授業の予習・復習、卒業研究テーマ探索
- **ニーズ**:
  - わかりやすい入門コンテンツ
  - 実行可能なコード例(Jupyter Notebook)
  - 基礎的なデータセット
- **成功指標**: 基礎理論ページの完読、サンプルコードの実行成功

#### ペルソナ2: 修士生 (材料系・情報系)
- **背景**: Python中級、機械学習基礎あり、研究でMIを使い始める
- **目的**: 研究実装、論文執筆の参考、最新手法の理解
- **ニーズ**:
  - 実践的なチュートリアル
  - 論文とコードのセット
  - データセット・ツールの情報
- **成功指標**: チュートリアル完了、研究への応用

#### ペルソナ3: 産業界R&D担当者
- **背景**: 材料メーカー技術者、MI導入検討段階
- **目的**: 技術導入可能性調査、ROI評価、導入事例研究
- **ニーズ**:
  - 概要レベルの技術解説
  - 産業応用事例
  - ツール比較・評価
  - 相談窓口
- **成功指標**: 技術理解、導入判断材料の獲得

#### ペルソナ4: データサイエンティスト (材料分野興味)
- **背景**: 機械学習専門家、材料科学は学習中
- **目的**: ドメイン知識習得、MI手法の評価・適用
- **ニーズ**:
  - 材料科学の基礎(DS向け)
  - 実装詳細とコード
  - ベンチマークデータセット
- **成功指標**: 手法実装、論文レベルの理解

### 2.2 ユーザージャーニー

**学部生の典型的ジャーニー:**
```
1. 入門ページで概要把握
2. 基礎理論(機械学習)の学習
3. チュートリアル実行(Jupyter)
4. データセットで自己学習
5. コミュニティで質問
```

**産業界技術者の典型的ジャーニー:**
```
1. トップページで価値提案確認
2. 応用事例で実績確認
3. ツール比較で技術選定
4. 研究室情報で相談窓口確認
5. GitHub Discussionsで質問
```

---

## 3. 機能要件

### 3.1 コアコンテンツ機能

#### 3.1.1 学習パス提供
- **入門コース**: MI概要、必要な前提知識、学習ロードマップ
- **基礎理論**: 機械学習基礎、材料科学基礎、データ前処理
- **手法・技術**: 回帰/分類、ニューラルネット、ベイズ最適化、能動学習
- **応用事例**: 材料探索、プロセス最適化、逆設計
- **実装レベル**: 理論解説 + コードサンプル + 演習問題

#### 3.1.2 リソース提供
- **ツール紹介**: Python libraries (scikit-learn, PyTorch, matminer)
- **データセット**: 公開データセット一覧、使用方法、ライセンス情報
- **論文情報**: 重要論文リスト、自動収集された最新論文、要約
- **コード例**: GitHub連携、Jupyter Notebook、Google Colab実行

#### 3.1.3 インタラクティブ機能
- **Tutor Agent**: チャットUI、質問応答、学習支援
- **検索機能**: 全文検索、タグ検索、フィルタリング
- **Jupyter表示**: nbconvert変換、構文ハイライト、実行結果表示
- **Colab連携**: ワンクリックでColab起動

### 3.2 コミュニティ機能

#### 3.2.1 Q&Aフォーラム
- **実装**: GitHub Discussions統合
- **カテゴリ**: 基礎理論、実装、論文、キャリア
- **モデレーション**: 研究室メンバー

#### 3.2.2 イベント情報
- **セミナー告知**: 日時、講演者、アーカイブ
- **ワークショップ**: ハンズオン情報、参加申込
- **コミュニティリンク**: Discord/Slackへの誘導

#### 3.2.3 コード共有
- **GitHub連携**: リポジトリ一覧、スター数表示
- **コントリビューション**: プルリクエスト歓迎のサイン

### 3.3 研究室情報連携

- **About**: 研究室紹介、Dr. Hashimotoプロフィール
- **相互リンク**: AI_Homepageとの双方向リンク
- **お問い合わせ**: 共同研究、相談窓口、採用情報

---

## 4. コンテンツ構造

### 4.1 サイト構造 (8セクション)

```
MI Knowledge Hub
├── 1. 入門 (Introduction)
│   ├── MIとは
│   ├── 学習ロードマップ
│   ├── 前提知識チェック
│   └── よくある質問
│
├── 2. 基礎理論 (Fundamentals)
│   ├── 機械学習基礎
│   ├── 材料科学基礎 (DS向け)
│   ├── データ前処理
│   └── 特徴量エンジニアリング
│
├── 3. 手法・技術 (Methods)
│   ├── 教師あり学習 (回帰/分類)
│   ├── ニューラルネットワーク
│   ├── ベイズ最適化
│   ├── 能動学習
│   ├── 転移学習
│   └── 説明可能AI
│
├── 4. 応用事例 (Applications)
│   ├── 材料探索
│   ├── プロセス最適化
│   ├── 逆設計
│   ├── 物性予測
│   └── 産業事例
│
├── 5. ツール (Tools)
│   ├── Python環境構築
│   ├── 主要ライブラリ (scikit-learn, PyTorch, matminer)
│   ├── ツール比較表
│   └── インストールガイド
│
├── 6. データ (Datasets)
│   ├── 公開データセット一覧
│   ├── データ取得方法
│   ├── データ形式解説
│   └── ライセンス情報
│
├── 7. コミュニティ (Community)
│   ├── Q&Aフォーラム (GitHub Discussions)
│   ├── イベント情報
│   ├── Discord/Slack
│   └── コントリビューション
│
└── 8. 研究室情報 (Lab Info)
    ├── About (研究室紹介)
    ├── 研究トピック
    ├── お問い合わせ
    └── AI_Homepageへのリンク
```

### 4.2 コンテンツページ例

#### 例1: 「機械学習基礎」ページ
- **対象**: 学部生〜修士1年
- **前提知識**: Python基礎、線形代数初歩
- **学習時間**: 2-3時間
- **構成**:
  - 理論解説 (回帰/分類の概念)
  - 数式と直感的説明
  - scikit-learn入門コード
  - Jupyter Notebook (ダウンロード/Colab)
  - 演習問題 (3問)
  - 参考文献
- **関連ページ**: データ前処理、Pythonチュートリアル、データセット

#### 例2: 「材料探索の応用事例」ページ
- **対象**: 修士生、産業界技術者
- **構成**:
  - 事例概要 (どの材料、何を予測)
  - 使用手法とデータ
  - 結果と考察
  - 実装コード (GitHub)
  - 論文リンク
  - 産業への示唆
- **関連ページ**: ベイズ最適化、公開データセット

### 4.3 コンテンツ生成戦略

| コンテンツ種別 | 生成方式 | 更新頻度 | 担当Agent | 品質保証 |
|-------------|---------|---------|----------|----------|
| 基礎理論 | 人間執筆 | 年1回 | - | Academic Reviewer Agent (必須) |
| チュートリアル | 人間+AI (初稿AI) | 半年1回 | Content Agent | Academic Reviewer Agent (必須) |
| 論文要約 | AI自動生成 | 週次 | Scholar Agent | Academic Reviewer Agent (自動) |
| ニュース | AI生成+人間編集 | 随時 | Content Agent | Academic Reviewer Agent (推奨) |
| FAQ | AI生成 | 月次更新 | Tutor Agent | Academic Reviewer Agent (自動) |
| ツール情報 | 半自動 | 四半期1回 | Data Agent | Academic Reviewer Agent (推奨) |
| イベント情報 | 人間入力 | 随時 | - | - |

---

## 5. 技術要件

### 5.1 技術スタック

#### 5.1.1 フロントエンド
- **言語**: HTML5, CSS3, JavaScript (ES6+)
- **スタイル**: Vanilla CSS (CSS Variables for design tokens)
- **デザイン**: Mobile-First, Responsive (既存AI_Homepageデザインシステム継承)
- **Jupyter表示**: nbconvert (静的HTML変換)
- **チャットUI**: Vanilla JS + WebSocket or SSE

#### 5.1.2 バックエンド
- **言語**: Python 3.11+
- **LLM**: Claude API (Anthropic)
- **論文収集**: Scholarly library, arXiv API
- **自動化**: GitPython, GitHub Actions
- **非同期処理**: asyncio

#### 5.1.3 データ層
- **形式**: JSON (構造化データ), Markdown (記事), Jupyter Notebook (.ipynb)
- **ストレージ**: Git repository
- **スキーマ**: JSON Schema定義

#### 5.1.4 デプロイメント
- **ホスティング**: GitHub Pages (無料)
- **CI/CD**: GitHub Actions
- **ドメイン**: GitHub提供サブドメイン or カスタムドメイン
- **SSL**: GitHub Pages自動提供

#### 5.1.5 外部サービス
- **LLM API**: Claude API (Anthropic)
- **論文データ**: Google Scholar, arXiv
- **コミュニティ**: GitHub Discussions
- **分析**: Google Analytics (オプション)
- **Notebook実行**: Google Colab

### 5.2 システムアーキテクチャ

```
┌─────────────────────────────────────────────────────────┐
│  Frontend Layer (GitHub Pages)                          │
│  ├─ Static HTML/CSS/JS                                  │
│  ├─ JSON-driven rendering                               │
│  ├─ Jupyter Notebook viewer (nbconvert)                │
│  └─ Chat UI (Tutor Agent interface)                    │
└─────────────────────────────────────────────────────────┘
                          ↕ HTTPS
┌─────────────────────────────────────────────────────────┐
│  Multi-Agent Orchestrator (Local/GitHub Actions)       │
│  ├─ Scholar Agent: 論文収集・要約                        │
│  ├─ Content Agent: 記事生成・更新                        │
│  ├─ Tutor Agent: 対話学習支援                          │
│  ├─ Data Agent: データセット管理                        │
│  ├─ Design Agent: UX最適化・分析                        │
│  ├─ Maintenance Agent: 監視・保守                       │
│  └─ Academic Reviewer Agent: 学術レビュー・品質保証      │
└─────────────────────────────────────────────────────────┘
                          ↕
┌─────────────────────────────────────────────────────────┐
│  Data Layer (Git Repository)                            │
│  ├─ data/papers.json                                    │
│  ├─ data/tutorials.json                                 │
│  ├─ data/datasets.json                                  │
│  ├─ content/*.md (articles)                             │
│  └─ notebooks/*.ipynb (tutorials)                       │
└─────────────────────────────────────────────────────────┘
                          ↕
┌─────────────────────────────────────────────────────────┐
│  External APIs                                          │
│  ├─ Claude API (Anthropic)                              │
│  ├─ Google Scholar API                                  │
│  ├─ arXiv API                                            │
│  ├─ GitHub API                                          │
│  └─ Lighthouse API (Design Agent)                       │
└─────────────────────────────────────────────────────────┘
```

### 5.3 ディレクトリ構造

```
MI/
├── .github/
│   └── workflows/
│       ├── deploy.yml          # 自動デプロイ
│       ├── scholar-agent.yml   # 論文収集(週次)
│       └── maintenance.yml     # 監視・保守
│
├── agents/                     # マルチエージェントシステム
│   ├── __init__.py
│   ├── base_agent.py           # 基底クラス
│   ├── scholar_agent.py        # 論文収集
│   ├── content_agent.py        # 記事生成
│   ├── tutor_agent.py          # 対話学習
│   ├── data_agent.py           # データ管理
│   ├── design_agent.py         # UX最適化
│   ├── maintenance_agent.py    # 保守
│   ├── academic_reviewer_agent.py  # 学術レビュー
│   └── orchestrator.py         # エージェント調整
│
├── assets/
│   ├── css/
│   │   ├── variables.css       # デザイントークン
│   │   ├── reset.css
│   │   ├── base.css
│   │   ├── components.css
│   │   ├── layout.css
│   │   └── responsive.css
│   ├── js/
│   │   ├── main.js
│   │   ├── navigation.js
│   │   ├── data-loader.js      # JSON動的読み込み
│   │   ├── search.js           # 検索機能
│   │   └── chat-ui.js          # Tutor Agent UI
│   └── images/
│
├── data/                       # JSONデータ
│   ├── papers.json             # 論文情報
│   ├── tutorials.json          # チュートリアル
│   ├── datasets.json           # データセット
│   ├── tools.json              # ツール情報
│   ├── events.json             # イベント
│   └── faq.json                # FAQ
│
├── content/                    # Markdown記事
│   ├── introduction/
│   ├── fundamentals/
│   ├── methods/
│   ├── applications/
│   └── ...
│
├── notebooks/                  # Jupyter Notebooks
│   ├── basics/
│   ├── advanced/
│   └── applications/
│
├── pages/                      # 静的ページ
│   ├── index.html              # トップページ
│   ├── introduction.html
│   ├── fundamentals.html
│   ├── methods.html
│   ├── applications.html
│   ├── tools.html
│   ├── datasets.html
│   ├── community.html
│   └── about.html
│
├── claudedocs/                 # プロジェクト文書
│   ├── requirements.md         # 本書
│   ├── technical-design.md     # 技術設計書
│   └── design-system.md        # デザインシステム
│
├── tools/                      # 開発ツール
│   ├── convert_notebooks.py   # Jupyter→HTML変換
│   ├── generate_sitemap.py
│   └── validate_data.py        # JSONスキーマ検証
│
├── .env.example                # 環境変数テンプレート
├── .gitignore
├── README.md
└── requirements.txt            # Python依存関係
```

---

## 6. マルチエージェント設計

### 6.1 エージェント概要

本システムは、Claude Codeの`/agents`機能を活用した7つの専門エージェントで構成される。各エージェントは独立して動作し、Orchestratorが調整を行う。

### 6.2 各エージェントの仕様

#### 6.2.1 Scholar Agent (論文収集・要約)

**責任範囲:**
- Google Scholar / arXiv APIから論文自動収集
- 論文メタデータ抽出 (タイトル、著者、被引用数、DOI)
- Claude APIによる論文要約生成
- papers.jsonへの自動追加

**実行トリガー:**
- GitHub Actions (週次スケジュール実行)
- 手動コマンド: `python agents/scholar_agent.py --query "materials informatics"`

**入力:**
- 検索クエリ (キーワード)
- 収集期間 (過去1週間など)
- 収集件数上限

**出力:**
- papers.json更新
- 新規論文サマリー (Markdown)
- 実行ログ

**技術スタック:**
- Scholarly library (Google Scholar非公式API)
- arXiv API (公式)
- Claude API (要約生成)

**実装例:**
```python
class ScholarAgent:
    async def fetch_papers(self, query: str, days: int = 7) -> List[Paper]:
        # 1. Google Scholarから論文検索
        # 2. メタデータ抽出
        # 3. Claude APIで要約生成
        # 4. papers.jsonに追加
        pass
```

#### 6.2.2 Content Agent (記事生成・更新)

**責任範囲:**
- Markdown記事の自動生成 (初稿)
- 既存記事の更新・改善提案
- FAQ自動生成
- ニュース記事生成

**実行トリガー:**
- 手動コマンド: `python agents/content_agent.py --topic "ベイズ最適化"`
- Orchestrator経由

**入力:**
- トピック (例: "ベイズ最適化")
- 対象ユーザー (学部生/修士生/産業界)
- 記事タイプ (理論解説/チュートリアル/事例)

**出力:**
- Markdown記事 (content/ディレクトリ)
- Git commit (自動 or 人間レビュー待ち)

**実装例:**
```python
class ContentAgent:
    async def generate_article(self, topic: str, level: str) -> str:
        # 1. Claude APIでアウトライン生成
        # 2. 各セクション詳細化
        # 3. コード例生成 (必要に応じて)
        # 4. Markdown出力
        pass
```

#### 6.2.3 Tutor Agent (対話型学習支援)

**責任範囲:**
- ユーザー質問への応答
- 学習パス提案
- コード解説・デバッグ支援
- FAQ自動更新

**実行トリガー:**
- Webサイト上のチャットUI
- WebSocket/SSE経由でリアルタイム通信

**入力:**
- ユーザー質問 (自然言語)
- ユーザー履歴 (オプション、ローカルストレージ)

**出力:**
- 回答テキスト
- 推奨リンク (関連ページ)
- コード例 (必要に応じて)

**技術スタック:**
- Claude API (Conversational mode)
- RAG (Retrieval-Augmented Generation) - オプション
  - ベクトルDB: Chroma (ローカル) or Pinecone
  - Embedding: OpenAI or Cohere

**実装例:**
```python
class TutorAgent:
    async def answer_question(self, question: str, context: dict) -> str:
        # 1. 関連コンテンツ検索 (RAG)
        # 2. Claude APIで回答生成
        # 3. リンク推奨
        pass
```

**UI統合:**
- フロントエンド: assets/js/chat-ui.js
- バックエンド: WebSocketサーバー (オプション) or 静的API呼び出し

#### 6.2.4 Data Agent (データセット管理)

**責任範囲:**
- データセット情報の収集・整理
- データセットメタデータ管理
- ツール情報の更新
- データ品質チェック

**実行トリガー:**
- 手動コマンド: `python agents/data_agent.py --update-datasets`
- 四半期スケジュール (GitHub Actions)

**入力:**
- データセット名 or URL
- メタデータ (ライセンス、形式、サイズ)

**出力:**
- datasets.json更新
- tools.json更新

**実装例:**
```python
class DataAgent:
    async def add_dataset(self, name: str, url: str, metadata: dict):
        # 1. データセット情報検証
        # 2. メタデータ整形
        # 3. datasets.jsonに追加
        pass
```

#### 6.2.5 Design Agent (UX最適化)

**責任範囲:**
- 類似サイトのUX分析 (Playwright scraping)
- パフォーマンス測定 (Lighthouse)
- アクセシビリティ監査 (axe-core)
- レイアウト改善提案
- CSS最適化

**実行トリガー:**
- 月次スケジュール (GitHub Actions)
- 手動コマンド: `python agents/design_agent.py --analyze`

**入力:**
- 分析対象URL (自サイト + 競合サイト)
- 改善目標 (パフォーマンス/アクセシビリティ)

**出力:**
- 分析レポート (Markdown)
- 改善提案 (Issue自動作成)
- CSS最適化コード (オプション)

**技術スタック:**
- Playwright (Webスクレイピング、既存HP開発で使用実績)
- Lighthouse API (パフォーマンス)
- axe-core (アクセシビリティ)
- Beautiful Soup (HTML解析)

**実装例:**
```python
class DesignAgent:
    async def analyze_competitors(self, urls: List[str]) -> Report:
        # 1. Playwrightでサイト訪問
        # 2. Lighthouse実行
        # 3. レイアウト構造分析
        # 4. ベストプラクティス抽出
        # 5. 改善提案生成
        pass
```

#### 6.2.6 Maintenance Agent (監視・保守)

**責任範囲:**
- サイト稼働監視 (リンク切れチェック)
- パフォーマンス監視
- データ整合性チェック
- バックアップ管理
- セキュリティ監査

**実行トリガー:**
- 日次スケジュール (GitHub Actions)
- 手動コマンド: `python agents/maintenance_agent.py --check-all`

**入力:**
- サイトURL
- データファイルパス

**出力:**
- ヘルスチェックレポート
- エラー通知 (GitHub Issue or メール)
- 自動修正 (可能な場合)

**実装例:**
```python
class MaintenanceAgent:
    async def check_links(self) -> List[BrokenLink]:
        # 1. 全ページをクロール
        # 2. リンク検証
        # 3. 壊れたリンクレポート
        pass

    async def validate_data(self) -> List[ValidationError]:
        # 1. JSONスキーマ検証
        # 2. データ整合性チェック
        pass
```

#### 6.2.7 Academic Reviewer Agent (学術レビュー・品質保証)

**責任範囲:**
- コンテンツの科学的正確性検証
- 理論・数式・計算の正しさチェック
- 引用・参考文献の妥当性確認
- 教育的観点からの内容評価
- 難易度の適切性判定
- 改善提案とフィードバック

**実行トリガー:**
- Content Agent記事生成後の自動レビュー
- 人間が執筆した記事のレビュー依頼
- 定期的な既存コンテンツの再評価 (四半期)
- 手動コマンド: `python agents/academic_reviewer_agent.py --review content/methods/bayesian_optimization.md`

**入力:**
- レビュー対象のMarkdown記事 or Jupyter Notebook
- コンテキスト (対象読者レベル、トピック分野)
- レビュー観点 (正確性/完全性/教育的配慮)

**出力:**
- 品質評価レポート (Markdown)
  - 正確性スコア (0-100)
  - 問題点の具体的指摘 (理論的誤り、説明不足、誤解を招く表現)
  - 改善提案 (具体的な修正案)
  - 承認/要修正判定
- GitHub Issue自動作成 (要修正の場合)
- レビューログ (logs/reviews/ディレクトリ)

**技術スタック:**
- Claude API (専門家ペルソナ: 材料科学教授 + 機械学習専門家)
- 論文データベース参照 (arXiv, Google Scholar) - 事実確認用
- Jupyter Notebook実行環境 (nbconvert, Papermill) - コード検証用

**レビュー基準:**
1. **科学的正確性 (40点)**
   - 理論の正しさ
   - 数式の正確性
   - 用語使用の適切性
   - 引用の正確性

2. **完全性 (20点)**
   - 必要な前提知識の明示
   - 論理展開の連続性
   - 例外・制約条件の記載

3. **教育的配慮 (20点)**
   - 対象読者レベルへの適合
   - 説明の明瞭さ
   - 例示の適切性
   - 段階的な難易度設計

4. **実装品質 (20点)**
   - コードの正確性
   - 実行可能性
   - ベストプラクティス遵守

**実装例:**
```python
class AcademicReviewerAgent:
    async def review_article(self, file_path: str, context: dict) -> ReviewReport:
        # 1. コンテンツ読み込みと解析
        content = self.load_content(file_path)

        # 2. Claude APIで専門家レビュー実行
        review_prompt = self.build_review_prompt(content, context)
        review_result = await self.claude.review(review_prompt)

        # 3. スコアリングと判定
        score = self.calculate_score(review_result)
        decision = "approved" if score >= 80 else "needs_revision"

        # 4. レポート生成
        report = self.generate_report(review_result, score, decision)

        # 5. 要修正の場合Issue作成
        if decision == "needs_revision":
            await self.create_issue(file_path, report)

        return report

    async def verify_code(self, notebook_path: str) -> CodeReviewResult:
        # 1. Jupyter Notebook実行
        # 2. エラーチェック
        # 3. 出力検証
        # 4. ベストプラクティスチェック
        pass

    def build_review_prompt(self, content: str, context: dict) -> str:
        return f"""
        あなたはマテリアルズインフォマティクスの専門家として、教育コンテンツをレビューしてください。

        対象読者: {context['target_audience']}
        トピック: {context['topic']}

        コンテンツ:
        {content}

        以下の観点で評価してください:
        1. 科学的正確性: 理論、数式、用語の正しさ
        2. 完全性: 論理展開、前提知識の明示
        3. 教育的配慮: 対象読者への適合性、説明の明瞭さ
        4. 実装品質: コードの正確性、実行可能性

        各項目について問題点を具体的に指摘し、改善案を提示してください。
        """
```

**ワークフロー統合:**
```python
# Orchestrator内での利用例
async def content_creation_workflow(self, topic: str):
    # 1. Content Agent: 初稿生成
    draft = await self.content.generate_article(topic)

    # 2. Academic Reviewer Agent: レビュー
    review = await self.reviewer.review_article(draft.path, context={
        'target_audience': 'undergraduate',
        'topic': topic
    })

    # 3. 承認判定
    if review.decision == "approved":
        await self.commit_changes(f"Add article: {topic}")
    else:
        # 人間レビュー待ちフラグ
        await self.create_review_request(draft.path, review.report)
```

**品質保証プロセス:**
```
Content Agent (記事生成)
  ↓
Academic Reviewer Agent (自動レビュー)
  ├─ Score ≥ 80 → 自動承認 → 公開
  └─ Score < 80 → Issue作成 → 人間レビュー
                    ↓
                修正 → 再レビュー → 公開
```

### 6.3 Orchestrator (エージェント調整)

**役割:**
- エージェント間の依存関係管理
- 実行順序の制御
- エラーハンドリング
- ログ集約

**実装例:**
```python
class Orchestrator:
    async def run_weekly_update(self):
        # 1. Scholar Agent: 論文収集
        papers = await self.scholar.fetch_papers()

        # 2. Content Agent: 新規論文サマリー生成
        draft = await self.content.generate_news(papers)

        # 3. Academic Reviewer Agent: 品質チェック
        review = await self.reviewer.review_article(draft.path, {
            'target_audience': 'mixed',
            'topic': 'papers_summary'
        })

        # 4. レビュー結果判定
        if review.score >= 80:
            # 5. Data Agent: データ整合性チェック
            await self.data.validate_all()

            # 6. Git commit & push
            await self.commit_changes("Weekly update: papers and news")
        else:
            # 人間レビュー待ち
            await self.create_review_request(draft.path, review.report)
```

### 6.4 エージェント間通信

**通信方式:**
- 同期実行: 直接関数呼び出し (Python)
- 非同期実行: メッセージキュー (オプション、将来拡張)

**データ共有:**
- 共通データ層 (Git repository内のJSON/Markdown)
- 実行ログ (logs/ディレクトリ)

---

## 7. 非機能要件

### 7.1 パフォーマンス

- **ページ読み込み時間**: < 2秒 (4G mobile)
- **Lighthouse Performance Score**: ≥ 95
- **Time to Interactive (TTI)**: < 3秒
- **First Contentful Paint (FCP)**: < 1.5秒

### 7.2 アクセシビリティ

- **WCAG 2.1 Level AA準拠**
- **Lighthouse Accessibility Score**: 100
- **キーボードナビゲーション**: 全機能対応
- **スクリーンリーダー対応**: ARIA属性適切使用
- **色コントラスト**: 4.5:1以上 (通常テキスト)

### 7.3 レスポンシブデザイン

- **モバイルファースト設計**
- **ブレークポイント**:
  - Mobile: 0-767px
  - Tablet: 768-1023px
  - Desktop: 1024-1439px
  - Wide: 1440px+
- **タッチターゲット**: ≥ 44px × 44px (Apple HIG)
- **ホバー効果**: `@media (hover: hover)` 使用 (iPhoneダブルタップ問題回避)

### 7.4 セキュリティ

- **HTTPS**: GitHub Pages標準提供
- **API Key管理**: 環境変数 (.env, GitHub Secrets)
- **XSS対策**: ユーザー入力のサニタイゼーション (Tutor Agent)
- **CORS**: 適切なオリジン設定
- **依存関係**: 定期的な脆弱性スキャン (Dependabot)

### 7.5 SEO

- **メタタグ**: title, description, OGP
- **構造化データ**: Schema.org (Article, Dataset)
- **サイトマップ**: 自動生成 (tools/generate_sitemap.py)
- **robots.txt**: 適切なクロール設定
- **セマンティックHTML**: h1-h6階層、適切な要素使用

### 7.6 国際化 (i18n)

- **Phase 1**: 日本語のみ
- **Phase 2+**: 英語対応 (既存AI_Homepageの日英切替パターン踏襲)
- **将来**: AIによる多言語自動翻訳 (Content Agent拡張)

### 7.7 ブラウザサポート

- **Desktop**: Chrome, Firefox, Safari, Edge (最新版 + 1つ前のメジャーバージョン)
- **Mobile**: iOS Safari, Chrome (最新版)
- **必須機能**: ES6+, CSS Grid, Flexbox

### 7.8 可用性

- **稼働率目標**: 99.9% (GitHub Pages SLA依存)
- **バックアップ**: Git履歴 (全データバージョン管理)
- **復旧時間**: < 1時間 (Git revert)

---

## 8. 制約事項

### 8.1 技術的制約

- **ホスティング**: GitHub Pages無料プラン
  - 静的サイトのみ
  - 1GB容量制限
  - 月間100GBトラフィック制限
  - サーバーサイド処理不可 (API endpointsは外部サービス or GitHub Actions)
- **LLM API**: Claude API使用量制限
  - コスト管理: 月間予算設定
  - レート制限対応: リトライロジック
- **論文API**: Google Scholar非公式API
  - アクセス制限: スクレイピングポリシー遵守
  - 代替: arXiv公式API併用

### 8.2 リソース制約

- **開発期間**: 6ヶ月 (Phase 1-3)
- **開発体制**: 主に1名 (Dr. Hashimoto) + 学生サポート (オプション)
- **予算**: 無料範囲内 (GitHub, Claude API無料枠 or 低コスト)

### 8.3 運用制約

- **保守**: 自動化優先 (週次手動レビューは許容)
- **コンテンツ更新**: 基礎理論は年1回、動的コンテンツは自動更新
- **コミュニティモデレーション**: 研究室メンバーが対応

### 8.4 法的制約

- **著作権**: 引用ルール遵守、ライセンス明記
- **個人情報**: ユーザーデータ収集最小限 (Cookieなし、Analyticsはオプション)
- **論文利用**: フェアユース範囲内の要約のみ

---

## 9. 開発フェーズとマイルストーン

### Phase 1: Foundation (Month 1-2)

#### 目標
基盤構築とコアコンテンツの骨格作成

#### タスク
- [ ] プロジェクト構造構築
  - [ ] ディレクトリ構造作成
  - [ ] Git repository初期化
  - [ ] GitHub Pages設定
- [ ] デザインシステム移植
  - [ ] 既存AI_HomepageのCSS variables継承
  - [ ] 8セクション用レイアウト設計
  - [ ] モバイルファーストCSS実装
- [ ] 静的ページ作成
  - [ ] 8セクションのHTML雛形
  - [ ] ナビゲーション実装
  - [ ] データローダー実装 (data-loader.js)
- [ ] JSONスキーマ設計
  - [ ] papers.json
  - [ ] tutorials.json
  - [ ] datasets.json
  - [ ] tools.json
  - [ ] events.json
  - [ ] faq.json
- [ ] Agent基盤実装
  - [ ] base_agent.py
  - [ ] orchestrator.py
  - [ ] Scholar Agent (MVP)
  - [ ] Content Agent (MVP)

#### 成果物
- 動作する静的サイト (コンテンツは仮)
- GitHub Pages公開
- 2つのAgent動作確認

#### 成功基準
- [ ] 8セクションページがモバイル/デスクトップで正しく表示
- [ ] Lighthouse Performance ≥ 90, Accessibility = 100
- [ ] Scholar Agentが論文5件収集可能
- [ ] Content Agentが簡易記事生成可能

---

### Phase 2: Content (Month 3-5)

#### 目標
実際の教育コンテンツ作成と自動化

#### タスク
- [ ] 基礎理論コンテンツ
  - [ ] 入門ページ (MI概要、学習ロードマップ)
  - [ ] 機械学習基礎 (理論解説 + コード)
  - [ ] 材料科学基礎 (DS向け)
  - [ ] データ前処理・特徴量エンジニアリング
- [ ] 手法・技術コンテンツ
  - [ ] 教師あり学習 (回帰/分類)
  - [ ] ニューラルネットワーク
  - [ ] ベイズ最適化
  - [ ] 能動学習
- [ ] チュートリアル
  - [ ] Jupyter Notebook 5-10本作成
  - [ ] nbconvert自動変換スクリプト
  - [ ] Google Colab連携
- [ ] リソース情報
  - [ ] ツール紹介ページ (scikit-learn, PyTorch, matminer)
  - [ ] データセット一覧 (10-15件)
  - [ ] 論文情報 (Scholar Agent自動収集)
- [ ] 研究室情報連携
  - [ ] Aboutページ (AI_Homepageから移植)
  - [ ] 相互リンク設定
- [ ] Agent強化
  - [ ] Scholar Agent: 週次自動実行 (GitHub Actions)
  - [ ] Content Agent: FAQ自動生成
  - [ ] Data Agent実装 (データセット管理)

#### 成果物
- 30-40ページの実コンテンツ
- 5-10本のJupyterチュートリアル
- 論文情報自動更新システム

#### 成功基準
- [ ] 学部生が「機械学習基礎」を完了できる
- [ ] Jupyterチュートリアルが Colab で実行可能
- [ ] Scholar Agent が週次で論文10件追加
- [ ] 全ページが WCAG AA準拠

---

### Phase 3: Interactive (Month 6)

#### 目標
インタラクティブ機能とコミュニティ基盤

#### タスク
- [ ] Tutor Agent実装
  - [ ] Claude API統合
  - [ ] チャットUI実装 (assets/js/chat-ui.js)
  - [ ] RAG実装 (オプション)
- [ ] 検索機能
  - [ ] 全文検索実装 (assets/js/search.js)
  - [ ] タグフィルタリング
- [ ] コミュニティ統合
  - [ ] GitHub Discussions設定
  - [ ] イベント情報ページ
  - [ ] Discord/Slackリンク
- [ ] Design Agent実装
  - [ ] Playwright競合分析
  - [ ] Lighthouse自動監視
  - [ ] 改善提案自動Issue化
- [ ] Maintenance Agent実装
  - [ ] リンク切れチェック
  - [ ] データ整合性検証
  - [ ] 日次ヘルスチェック

#### 成果物
- 対話型学習支援システム
- 完全な検索機能
- コミュニティハブ
- 自動保守システム

#### 成功基準
- [ ] Tutor Agentが基礎的な質問に回答可能
- [ ] 検索が1秒以内に結果表示
- [ ] GitHub Discussionsに10件以上の投稿
- [ ] Design Agentが週次でレポート生成
- [ ] Maintenance Agentがエラー検出可能

---

### Phase 4: Optimization (Month 7+, 継続)

#### 目標
パフォーマンス最適化と機能拡張

#### タスク
- [ ] パフォーマンス最適化
  - [ ] 画像最適化 (WebP, lazy loading)
  - [ ] CSS/JS minify
  - [ ] Critical CSS inline
- [ ] SEO強化
  - [ ] メタタグ最適化
  - [ ] 構造化データ追加
  - [ ] サイトマップ自動生成
- [ ] コンテンツ拡充
  - [ ] 応用事例追加 (産業界向け)
  - [ ] 上級チュートリアル
  - [ ] ツール比較表
- [ ] Agent強化
  - [ ] Tutor Agent: RAG精度向上
  - [ ] Content Agent: 多言語対応
  - [ ] Design Agent: A/Bテスト機能
- [ ] 分析・改善
  - [ ] Google Analytics導入 (オプション)
  - [ ] ユーザーフィードバック収集
  - [ ] 継続的改善

#### 成果物
- Lighthouse Performance 95+
- 50+ページのコンテンツ
- 高度な自動化システム

---

## 10. 成功基準

### 10.1 定量的指標

| 指標 | 目標値 | 測定方法 |
|------|--------|---------|
| **ページビュー** | 500/月 (Phase 1終了時) | Google Analytics |
| **ユニークユーザー** | 200/月 | Google Analytics |
| **平均セッション時間** | > 3分 | Google Analytics |
| **直帰率** | < 60% | Google Analytics |
| **Lighthouse Performance** | ≥ 95 | Lighthouse CI |
| **Lighthouse Accessibility** | 100 | Lighthouse CI |
| **ページ読み込み時間** | < 2秒 | WebPageTest |
| **リンク切れ** | 0件 | Maintenance Agent |
| **コンテンツ数** | 40+ページ (Phase 3終了時) | 手動カウント |
| **Jupyterチュートリアル** | 10+本 | 手動カウント |
| **コミュニティ投稿** | 20+件 (Phase 3終了時) | GitHub Discussions |

### 10.2 定性的指標

- **学生からのフィードバック**: 「わかりやすい」「実践的」などのポジティブ評価
- **産業界からの問い合わせ**: 共同研究・相談の問い合わせ発生
- **外部リンク**: 他大学・企業サイトからの被リンク
- **SNS言及**: Twitter/Linkedinでの肯定的な言及

### 10.3 技術的成功基準

- **自動化率**: コンテンツ更新の70%以上がAgent経由
- **稼働率**: 99%以上 (GitHub Pages依存)
- **CI/CDパイプライン**: 全テスト通過
- **コードカバレッジ**: 80%以上 (Python Agents)

### 10.4 教育的成功基準

- **学習完了率**: 基礎コース完了者 50名以上 (Phase 3終了時)
- **チュートリアル実行**: Colab実行数 100回以上
- **質問応答**: Tutor Agentの回答満足度 70%以上

---

## 11. リスクと対策

### 11.1 技術的リスク

| リスク | 影響 | 確率 | 対策 |
|--------|------|------|------|
| **GitHub Pages容量超過** | 高 | 中 | 画像最適化、外部ストレージ検討 |
| **Claude API コスト超過** | 中 | 中 | 使用量監視、キャッシュ活用 |
| **Scholar API制限** | 中 | 高 | arXiv併用、レート制限対応 |
| **Tutor Agent精度不足** | 中 | 中 | RAG導入、FAQ充実化 |
| **パフォーマンス劣化** | 低 | 低 | 継続的監視、最適化 |

### 11.2 運用リスク

| リスク | 影響 | 確率 | 対策 |
|--------|------|------|------|
| **コンテンツ作成遅延** | 高 | 中 | Agent活用、段階的公開 |
| **保守負荷増大** | 中 | 中 | 自動化徹底、Maintenance Agent |
| **コミュニティ荒れ** | 低 | 低 | モデレーションルール、GitHub Discussions活用 |

### 11.3 ユーザー体験リスク

| リスク | 影響 | 確率 | 対策 |
|--------|------|------|------|
| **モバイルUX問題** | 高 | 低 | モバイルファースト徹底、実機テスト |
| **コンテンツの難易度ミスマッチ** | 中 | 中 | ユーザーレベル別コンテンツ、フィードバック収集 |
| **検索精度不足** | 中 | 中 | 検索アルゴリズム改善、タグ体系整備 |

---

## 12. 次のステップ

### 12.1 immediate Actions (今週中)

1. **プロジェクト構造構築**
   - ディレクトリ作成
   - Git repository初期化
   - requirements.txt作成

2. **技術設計書作成**
   - `claudedocs/technical-design.md`
   - データベーススキーマ詳細化
   - Agent API仕様

3. **デザインシステム作成**
   - `claudedocs/design-system.md`
   - CSS variables定義
   - コンポーネント仕様

### 12.2 Phase 1 Kickoff (来週〜)

1. **開発環境構築**
   - Python環境セットアップ
   - Claude API key取得
   - GitHub Actions設定

2. **MVP実装開始**
   - 静的ページ雛形
   - Scholar Agent基本機能
   - 初期コンテンツ作成

---

## 13. 付録

### 13.1 参考サイト

**MIコミュニティサイト:**
- Materials Project: https://materialsproject.org/
- NOMAD Repository: https://nomad-lab.eu/
- Matminer Docs: https://hackingmaterials.lbl.gov/matminer/

**教育サイト:**
- Coursera Machine Learning: https://www.coursera.org/learn/machine-learning
- Fast.ai: https://www.fast.ai/
- Kaggle Learn: https://www.kaggle.com/learn

**デザイン参考:**
- 既存AI_Homepage: 研究室公式サイト (デザインシステム踏襲)
- Material-UI: https://mui.com/
- Tailwind UI: https://tailwindui.com/

### 13.2 関連ツール・ライブラリ

**Python:**
- scholarly: Google Scholar API
- arxiv: arXiv API
- anthropic: Claude API
- beautifulsoup4: Webスクレイピング
- playwright: ブラウザ自動化
- gitpython: Git操作

**JavaScript:**
- nbviewer.js: Jupyter表示
- marked.js: Markdown→HTML
- highlight.js: 構文ハイライト

**CI/CD:**
- GitHub Actions: 自動化
- Lighthouse CI: パフォーマンス監視
- Dependabot: 依存関係更新

### 13.3 用語集

- **MI (Materials Informatics)**: マテリアルズインフォマティクス、材料科学とデータサイエンスの融合分野
- **RAG (Retrieval-Augmented Generation)**: 検索拡張生成、LLMの回答精度向上手法
- **nbconvert**: Jupyter Notebook形式変換ツール
- **SSG (Static Site Generator)**: 静的サイト生成ツール
- **CI/CD**: 継続的インテグレーション・デプロイメント

### 13.4 コンテンツ作成プロシージャー

高品質な記事を作成するための標準化されたワークフローは、別ドキュメントで詳細に定義されています:

**📄 [コンテンツ作成プロシージャー](content-creation-procedure.md)**

このプロシージャーは、7つのエージェントが協力して極めて質の高い教育コンテンツを作成するための9フェーズのワークフローを定義しています。

**主要フェーズ**:
- Phase 0: 企画・準備
- Phase 1: 多角的情報収集
- Phase 2: 初稿作成
- Phase 3: 学術レビュー第1サイクル (合格基準: 80点)
- Phase 4: 教育的レビュー
- Phase 5: 実装検証
- Phase 6: UX最適化
- Phase 7: 学術レビュー第2サイクル (合格基準: 90点)
- Phase 8: 統合品質保証
- Phase 9: 最終承認・公開

**品質目標**:
- 総合スコア: 90点以上 (100点満点)
- 学術的正確性: 論文引用15-20件、事実誤認ゼロ
- 教育的品質: 学習目標達成率90%以上
- 実装品質: コード実行成功率100%、Pylintスコア9.0以上
- UX品質: Lighthouse Performance 95+, Accessibility 100

**想定所要時間**: 15-20時間/記事

### 13.5 連絡先

- **プロジェクトオーナー**: Dr. Yusuke Hashimoto
- **Email**: yusuke.hashimoto.b8@tohoku.ac.jp
- **GitHub**: (リポジトリURL、作成後追記)

---

## 変更履歴

| バージョン | 日付 | 変更内容 | 担当者 |
|----------|------|---------|--------|
| 1.0 | 2025-10-15 | 初版作成 | Claude Code (brainstorming session) |
| 1.1 | 2025-10-15 | Academic Reviewer Agent追加 (学術レビュー・品質保証機能) | Claude Code |

---

**End of Requirements Document**

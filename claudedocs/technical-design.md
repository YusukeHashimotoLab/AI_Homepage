# 技術設計書

**プロジェクト名**: 橋本佑介研究室ホームページ再構築
**作成日**: 2025-10-14
**バージョン**: 1.0

---

## 1. システムアーキテクチャ

### 1.1 全体構成

```
┌─────────────────────────────────────────┐
│         ユーザー（ブラウザ）              │
└─────────────────┬───────────────────────┘
                  │
                  │ HTTPS
                  │
┌─────────────────▼───────────────────────┐
│       GitHub Pages (静的ホスティング)     │
│                                          │
│  ┌────────────────────────────────────┐ │
│  │  HTML/CSS/JavaScript               │ │
│  │  (レスポンシブ・モバイルファースト)  │ │
│  └────────────────────────────────────┘ │
│                                          │
│  ┌────────────────────────────────────┐ │
│  │  Data Files (JSON/YAML)            │ │
│  │  - papers.json                     │ │
│  │  - news.json                       │ │
│  │  - members.json                    │ │
│  └────────────────────────────────────┘ │
└──────────────────────────────────────────┘
                  ▲
                  │
                  │ Git Push
                  │
┌─────────────────┴───────────────────────┐
│      ローカル開発環境                     │
│                                          │
│  ┌────────────────────────────────────┐ │
│  │  AI更新ツール (Python)              │ │
│  │                                    │ │
│  │  ┌──────────────────────────────┐ │ │
│  │  │ 自然言語チャットIF            │ │ │
│  │  │ (Claude/OpenAI API)          │ │ │
│  │  └──────────────────────────────┘ │ │
│  │                                    │ │
│  │  ┌──────────────────────────────┐ │ │
│  │  │ 論文取得モジュール            │ │ │
│  │  │ (Scholar, ORCID, Crossref)   │ │ │
│  │  └──────────────────────────────┘ │ │
│  │                                    │ │
│  │  ┌──────────────────────────────┐ │ │
│  │  │ データ管理モジュール          │ │ │
│  │  │ (JSON/YAML読み書き)          │ │ │
│  │  └──────────────────────────────┘ │ │
│  └────────────────────────────────────┘ │
└──────────────────────────────────────────┘
```

### 1.2 データフロー

```
ユーザー指示（自然言語）
    ↓
AIチャットインターフェース
    ↓
Claude/OpenAI API（意図解釈）
    ↓
適切なアクションを決定
    ├→ 論文取得: Scholar API呼び出し
    ├→ ニュース追加: news.json更新
    ├→ メンバー追加: members.json更新
    └→ コンテンツ編集: HTML/Markdown更新
    ↓
データファイル更新
    ↓
（オプション）GitHub自動コミット
    ↓
GitHub Pages自動デプロイ
```

---

## 2. フロントエンド設計

### 2.1 技術選択

| 項目 | 技術 | 理由 |
|------|------|------|
| HTML | HTML5 | セマンティックHTML、アクセシビリティ |
| CSS | CSS3 + CSS Variables | モダンなスタイリング、テーマ切り替え |
| JavaScript | Vanilla JS (ES6+) | 軽量、フレームワーク不要 |
| ビルドツール | なし（静的） | シンプル、保守性高い |

### 2.2 ディレクトリ構造

```
AI_Homepage/
├── index.html                 # ルート（日本語リダイレクト）
├── en/                        # 英語版
│   ├── index.html
│   ├── research.html
│   ├── publications.html
│   ├── news.html
│   ├── members.html
│   ├── talks.html
│   ├── contact.html
│   └── links.html
├── jp/                        # 日本語版
│   ├── index.html
│   ├── research.html
│   ├── publications.html
│   ├── news.html
│   ├── members.html
│   ├── talks.html
│   ├── contact.html
│   └── links.html
├── assets/
│   ├── css/
│   │   ├── variables.css      # CSS変数（カラー、サイズ等）
│   │   ├── reset.css          # CSSリセット
│   │   ├── base.css           # 基本スタイル
│   │   ├── components.css     # コンポーネント
│   │   ├── layout.css         # レイアウト
│   │   └── responsive.css     # レスポンシブ
│   ├── js/
│   │   ├── main.js            # メインロジック
│   │   ├── navigation.js      # ナビゲーション
│   │   └── data-loader.js     # データ読み込み
│   └── images/
│       └── (画像ファイル)
├── data/                      # データファイル
│   ├── papers.json            # 論文情報
│   ├── news.json              # ニュース
│   ├── members.json           # メンバー
│   ├── talks.json             # 講演実績
│   └── profile.json           # プロフィール
├── tools/                     # AI更新ツール
│   ├── ai_chat.py             # チャットインターフェース
│   ├── paper_fetcher.py       # 論文取得
│   ├── data_manager.py        # データ管理
│   ├── config.py              # 設定
│   └── requirements.txt       # Python依存関係
├── claudedocs/                # ドキュメント
│   ├── requirements.md
│   ├── technical-design.md
│   └── user-manual.md
├── README.md
└── .gitignore
```

### 2.3 レスポンシブデザイン戦略

#### ブレークポイント
```css
/* モバイル（デフォルト） */
@media (min-width: 0px) { ... }

/* タブレット */
@media (min-width: 768px) { ... }

/* デスクトップ */
@media (min-width: 1024px) { ... }

/* 大画面 */
@media (min-width: 1440px) { ... }
```

#### モバイルファースト原則
1. デフォルトスタイル = モバイル
2. メディアクエリで段階的に拡張
3. タッチ操作を最優先
4. 最小タップサイズ: 44px × 44px（Apple HIG準拠）

### 2.4 ナビゲーション設計

#### デスクトップ
```html
<header>
  <nav class="desktop-nav">
    <a href="index.html">Home</a>
    <a href="research.html">Research</a>
    <!-- ... -->
  </nav>
  <div class="lang-switcher">
    <a href="../en/">EN</a>
    <a href="../jp/">JP</a>
  </div>
</header>
```

#### モバイル
```html
<!-- ハンバーガーメニュー -->
<header>
  <button class="menu-toggle" aria-label="メニュー">☰</button>
  <nav class="mobile-nav" aria-hidden="true">
    <!-- メニュー項目 -->
  </nav>
</header>

<!-- ボトムナビゲーション -->
<nav class="bottom-nav">
  <a href="index.html">🏠 Home</a>
  <a href="research.html">🔬 Research</a>
  <a href="publications.html">📚 Papers</a>
  <a href="contact.html">✉️ Contact</a>
</nav>
```

### 2.5 データ駆動レンダリング

#### 例: 論文リストの動的生成
```javascript
// data-loader.js
async function loadPapers() {
  const response = await fetch('../data/papers.json');
  const papers = await response.json();

  const container = document.getElementById('papers-list');
  papers.forEach(paper => {
    const item = createPaperElement(paper);
    container.appendChild(item);
  });
}

function createPaperElement(paper) {
  return `
    <article class="paper">
      <h3>${paper.title}</h3>
      <p class="authors">${paper.authors.join(', ')}</p>
      <p class="journal">${paper.journal}, ${paper.year}</p>
      <a href="${paper.url}">View Paper</a>
    </article>
  `;
}
```

#### データ形式: papers.json
```json
[
  {
    "id": "paper001",
    "title": "Machine Learning for Materials Discovery",
    "authors": ["Hashimoto Y.", "Smith J."],
    "journal": "Nature Materials",
    "year": 2024,
    "doi": "10.1038/xxxxx",
    "url": "https://doi.org/10.1038/xxxxx",
    "citations": 42,
    "tags": ["machine-learning", "materials-informatics"]
  }
]
```

---

## 3. AI更新ツール設計

### 3.1 システム要件

- Python 3.9以上
- Claude API または OpenAI API
- インターネット接続（API呼び出し用）

### 3.2 コアモジュール

#### 3.2.1 チャットインターフェース (`ai_chat.py`)

```python
"""
自然言語チャットインターフェースのメインモジュール
"""

import anthropic
from typing import Dict, Any

class AIContentManager:
    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.data_manager = DataManager()
        self.paper_fetcher = PaperFetcher()

    def process_command(self, user_input: str) -> str:
        """
        自然言語の指示を解釈し、適切なアクションを実行

        Args:
            user_input: ユーザーの自然言語入力

        Returns:
            実行結果のメッセージ
        """
        # Claude APIで意図を解釈
        intent = self._parse_intent(user_input)

        # アクションルーティング
        if intent['action'] == 'add_news':
            return self._add_news(intent['params'])
        elif intent['action'] == 'add_member':
            return self._add_member(intent['params'])
        elif intent['action'] == 'update_papers':
            return self._update_papers(intent['params'])
        else:
            return "申し訳ありません。その操作は理解できませんでした。"

    def _parse_intent(self, user_input: str) -> Dict[str, Any]:
        """Claude APIで自然言語を構造化データに変換"""
        prompt = f"""
        以下のユーザー指示を解釈し、JSON形式で返してください。

        ユーザー指示: {user_input}

        返すJSON:
        {{
            "action": "add_news | add_member | update_papers | edit_content",
            "params": {{
                // アクションに必要なパラメータ
            }}
        }}
        """

        response = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )

        import json
        return json.loads(response.content[0].text)
```

#### 3.2.2 データ管理 (`data_manager.py`)

```python
"""
JSON/YAMLデータファイルの読み書きを管理
"""

import json
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

class DataManager:
    def __init__(self, data_dir: str = "../data"):
        self.data_dir = Path(data_dir)

    def load_data(self, filename: str) -> Any:
        """データファイルを読み込み"""
        filepath = self.data_dir / filename

        if filename.endswith('.json'):
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        elif filename.endswith('.yaml') or filename.endswith('.yml'):
            with open(filepath, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)

    def save_data(self, filename: str, data: Any) -> None:
        """データファイルに保存"""
        filepath = self.data_dir / filename

        if filename.endswith('.json'):
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        elif filename.endswith('.yaml') or filename.endswith('.yml'):
            with open(filepath, 'w', encoding='utf-8') as f:
                yaml.dump(data, f, allow_unicode=True)

    def add_news(self, title: str, content: str, date: str = None) -> str:
        """ニュースを追加"""
        news_list = self.load_data('news.json')

        new_entry = {
            'id': f"news{len(news_list) + 1:03d}",
            'date': date or datetime.now().strftime('%Y-%m-%d'),
            'title': title,
            'content': content
        }

        news_list.insert(0, new_entry)  # 最新を先頭に
        self.save_data('news.json', news_list)

        return f"ニュース「{title}」を追加しました。"

    def add_member(self, name: str, role: str, research: str) -> str:
        """メンバーを追加"""
        members = self.load_data('members.json')

        new_member = {
            'id': f"member{len(members) + 1:03d}",
            'name': name,
            'role': role,
            'research': research
        }

        members.append(new_member)
        self.save_data('members.json', members)

        return f"メンバー「{name}」を追加しました。"
```

#### 3.2.3 論文取得 (`paper_fetcher.py`)

```python
"""
Google Scholar, ORCID, Crossrefから論文情報を取得
"""

import requests
from typing import List, Dict
from scholarly import scholarly  # Google Scholar API

class PaperFetcher:
    def __init__(self):
        self.data_manager = DataManager()

    def fetch_from_scholar(self, author_name: str) -> List[Dict]:
        """Google Scholarから論文を取得"""
        search_query = scholarly.search_author(author_name)
        author = next(search_query)
        author_filled = scholarly.fill(author)

        papers = []
        for pub in author_filled['publications']:
            pub_filled = scholarly.fill(pub)
            papers.append({
                'title': pub_filled['bib']['title'],
                'authors': pub_filled['bib'].get('author', []),
                'year': pub_filled['bib'].get('pub_year'),
                'journal': pub_filled['bib'].get('venue'),
                'citations': pub_filled.get('num_citations', 0),
                'url': pub_filled.get('pub_url')
            })

        return papers

    def update_papers(self) -> str:
        """既存の論文リストを更新"""
        papers = self.fetch_from_scholar("Yusuke Hashimoto")
        self.data_manager.save_data('papers.json', papers)

        return f"{len(papers)}件の論文情報を更新しました。"
```

### 3.3 使用例

#### ターミナルから実行
```bash
$ cd tools
$ python ai_chat.py

=== AI Content Manager ===
あなた: ニュースを追加して。タイトルは「新論文発表」、内容は「Nature Materialsに論文が掲載されました」

AI: ニュース「新論文発表」を追加しました。news.jsonに保存されています。

あなた: メンバーページに田中太郎を追加して。役職は博士課程、研究テーマは機械学習です。

AI: メンバー「田中太郎」を追加しました。members.jsonに保存されています。

あなた: 最新の論文情報を取得して

AI: Google Scholarから論文情報を取得中...
25件の論文情報を更新しました。
```

### 3.4 設定ファイル (`config.py`)

```python
"""
設定情報を管理
"""

import os
from dotenv import load_dotenv

load_dotenv()

# API キー
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# データディレクトリ
DATA_DIR = '../data'

# GitHub 自動コミット設定
AUTO_COMMIT = False
COMMIT_MESSAGE_TEMPLATE = "Update content via AI tool"

# デフォルト設定
DEFAULT_LANGUAGE = 'jp'
SUPPORTED_LANGUAGES = ['jp', 'en']
```

### 3.5 依存関係 (`requirements.txt`)

```txt
anthropic>=0.18.0
openai>=1.0.0
requests>=2.31.0
scholarly>=1.7.11
python-dotenv>=1.0.0
pyyaml>=6.0.1
```

---

## 4. パフォーマンス最適化

### 4.1 最適化戦略

| 項目 | 施策 | 効果 |
|------|------|------|
| CSS | ミニファイ、重複削除 | ファイルサイズ削減 |
| JavaScript | 必要最小限、遅延読み込み | 初期表示高速化 |
| 画像 | WebP使用、遅延読み込み | データ転送量削減 |
| フォント | system-ui優先、サブセット化 | 読み込み時間短縮 |
| HTML | セマンティック、無駄な要素削減 | レンダリング高速化 |

### 4.2 Lighthouse目標スコア

- Performance: 95+
- Accessibility: 100
- Best Practices: 100
- SEO: 100

---

## 5. セキュリティ

### 5.1 セキュリティ対策

1. **HTTPS必須**
   - GitHub Pagesで自動対応

2. **XSS対策**
   - ユーザー入力のサニタイズ
   - Content Security Policy設定

3. **APIキー管理**
   - `.env`ファイルで管理（.gitignore登録）
   - 環境変数から読み込み

4. **お問い合わせフォーム**
   - Formspree使用（スパム対策込み）
   - reCAPTCHA v3実装

### 5.2 `.gitignore`

```
# Python
__pycache__/
*.py[cod]
.venv/
venv/

# 環境変数
.env
.env.local

# IDE
.vscode/
.idea/

# OS
.DS_Store
Thumbs.db

# ビルド成果物
dist/
build/
```

---

## 6. デプロイメント

### 6.1 GitHub Pages設定

1. リポジトリ設定
   - Settings → Pages
   - Source: `main` branch
   - Directory: `/` (root)

2. カスタムドメイン（オプション）
   - CNAME設定
   - DNS設定

### 6.2 自動デプロイ（GitHub Actions）

```yaml
# .github/workflows/deploy.yml
name: Deploy to GitHub Pages

on:
  push:
    branches: [ main ]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Deploy to GitHub Pages
        uses: peaceiris/actions-gh-pages@v3
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: ./
```

---

## 7. テスト戦略

### 7.1 テスト項目

#### モバイル対応テスト
- [ ] iPhone Safari (iOS 15+)
- [ ] iPhone Chrome
- [ ] Android Chrome
- [ ] タブレット（iPad, Android）

#### 機能テスト
- [ ] ナビゲーション動作
- [ ] リンク・ボタンのタップ
- [ ] データ読み込み
- [ ] 言語切り替え
- [ ] お問い合わせフォーム

#### パフォーマンステスト
- [ ] Lighthouse スコア
- [ ] ページ読み込み時間
- [ ] Core Web Vitals

#### アクセシビリティテスト
- [ ] キーボードナビゲーション
- [ ] スクリーンリーダー
- [ ] カラーコントラスト

### 7.2 テストツール

- Chrome DevTools（Device Mode）
- Lighthouse
- WAVE（アクセシビリティ）
- 実機テスト（iPhone, Android）

---

## 8. 保守・運用

### 8.1 定期メンテナンス

| 項目 | 頻度 | 内容 |
|------|------|------|
| 論文情報更新 | 月1回 | AIツールで自動取得 |
| ニュース追加 | 随時 | チャットインターフェースで追加 |
| セキュリティ更新 | 四半期 | 依存ライブラリの更新 |
| バックアップ | 毎週 | Git自動バックアップ |

### 8.2 トラブルシューティング

#### 問題: AIツールが動作しない
- APIキーの確認（.env）
- Python依存関係のインストール確認
- ネットワーク接続確認

#### 問題: データが表示されない
- JSON形式の確認（バリデーション）
- ファイルパスの確認
- ブラウザコンソールでエラー確認

---

## 9. 次のステップ

1. デザインシステムの調査・策定
2. プロトタイプ作成
3. コンテンツ移行
4. AI更新ツール実装
5. テスト・検証
6. デプロイ

---

## 変更履歴

| 日付 | バージョン | 変更内容 | 担当 |
|------|-----------|---------|------|
| 2025-10-14 | 1.0 | 初版作成 | Claude |

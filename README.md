# 橋本佑介研究室ホームページ

東北大学 学際科学フロンティア研究所 橋本佑介特任准教授の研究公知用ホームページ

## プロジェクト概要

iPhone完全対応のモバイルファーストなデザインと、AI駆動のコンテンツ管理システムを備えた研究室ホームページです。

### 主な特徴

- ✅ **完全なモバイル対応**: iPhoneでのタップ問題を解決、44px以上のタッチターゲット
- ✅ **AI更新システム**: 自然言語でコンテンツ更新が可能
- ✅ **多言語対応**: 日本語・英語の完全サポート
- ✅ **モダンCSS**: CSS変数、モバイルファースト、アクセシビリティ準拠
- ✅ **静的サイト**: GitHub Pages対応、高速読み込み

## クイックスタート

### ローカルで確認

```bash
# ローカルサーバーを起動
python -m http.server 8000

# ブラウザでアクセス
open http://localhost:8000
```

### モバイルテスト

Chrome DevToolsでテスト：
1. F12で開発者ツールを開く
2. Ctrl+Shift+M (Mac: Cmd+Shift+M) でデバイスモードに切り替え
3. iPhone SE または iPhone 12 を選択
4. タップ動作、ナビゲーション、ボトムナビを確認

## プロジェクト構造

```
AI_Homepage/
├── index.html              # ルート（日本語へリダイレクト）
├── en/                     # 英語版ページ
│   └── index.html
├── jp/                     # 日本語版ページ
│   └── index.html
├── assets/
│   ├── css/
│   │   ├── variables.css   # デザイントークン（色、スペーシング等）
│   │   ├── reset.css       # モダンCSSリセット
│   │   ├── base.css        # 基本スタイル
│   │   ├── components.css  # コンポーネント（ボタン、カード等）
│   │   └── responsive.css  # レスポンシブ対応
│   ├── js/                 # JavaScript（今後追加予定）
│   └── images/            # 画像ファイル
├── data/                   # JSONデータ（今後追加予定）
├── tools/                  # AI更新ツール（今後実装予定）
└── claudedocs/             # 設計ドキュメント
    ├── requirements.md
    ├── technical-design.md
    └── design-system.md
```

## 開発ガイドライン

### CSS設計

**モバイルファースト**
```css
/* デフォルト = モバイル（0-768px） */
.element { ... }

/* タブレット（768px+） */
@media (min-width: 768px) { ... }

/* デスクトップ（1024px+） */
@media (min-width: 1024px) { ... }
```

**iPhone対応の重要ポイント**
```css
/* ❌ 間違い: iPhoneでダブルタップが必要になる */
nav a:hover { background: #34495e; }

/* ✅ 正解: hoverデバイスのみに適用 */
@media (hover: hover) and (pointer: fine) {
  nav a:hover { background: #34495e; }
}
```

### デザイントークン

すべてのデザイン値は `assets/css/variables.css` で定義：

```css
/* 色 */
--color-primary-900: #1a252f;
--color-primary-700: #2c3e50;

/* スペーシング（8pxグリッド） */
--spacing-4: 1rem;  /* 16px */
--spacing-6: 1.5rem;  /* 24px */

/* タイポグラフィ */
--font-size-base: 1rem;  /* 16px */
--font-size-xl: 1.25rem;  /* 20px */
```

## AI更新ツール（実装予定）

自然言語でコンテンツを更新できるPythonツールを実装予定：

```bash
# 仮想環境の作成
python -m venv .venv
source .venv/bin/activate  # macOS/Linux

# 依存関係のインストール（実装後）
pip install -r tools/requirements.txt

# AIチャットインターフェース起動（実装後）
cd tools
python ai_chat.py
```

**使用例**：
```
あなた: ニュースを追加。タイトル「新論文発表」
AI: ニュースを追加しました。

あなた: 最新の論文情報を取得して
AI: 25件の論文情報を更新しました。
```

## ドキュメント

詳細な設計情報は `/claudedocs/` を参照：

- **`requirements.md`**: 要件定義、機能一覧、成功基準
- **`technical-design.md`**: アーキテクチャ、実装詳細
- **`design-system.md`**: カラーパレット、タイポグラフィ、コンポーネント仕様
- **`CLAUDE.md`**: Claude Code用の開発ガイド

## テスト

### 必須テスト項目

**iPhone Safari（最優先）**
- [ ] すべてのリンク・ボタンがシングルタップで動作
- [ ] ダブルタップ不要
- [ ] ボトムナビゲーションが表示・機能
- [ ] 横スクロールなし

**パフォーマンス**
- [ ] Lighthouse Performance: 95+
- [ ] Lighthouse Accessibility: 100
- [ ] ページ読み込み < 2秒（4G）

**アクセシビリティ**
- [ ] キーボードナビゲーション（Tab, Enter, Space）
- [ ] スクリーンリーダー互換
- [ ] カラーコントラスト ≥ 4.5:1

## デプロイ

GitHub Pagesへの自動デプロイ：

```bash
git add .
git commit -m "Update content"
git push origin main

# GitHub Pagesが自動的にデプロイ
```

デプロイURL: `https://yusukehashimotolab.github.io/`

## ライセンス & 連絡先

**研究者**: 橋本 佑介
**所属**: 東北大学 学際科学フロンティア研究所
**役職**: 特任准教授
**Email**: yusuke.hashimoto.b8@tohoku.ac.jp

---

**開発開始**: 2025-10-14
**バージョン**: 1.0.0
**ステータス**: 開発中（基本構造完成、コンテンツ移行・AI更新ツール実装予定）

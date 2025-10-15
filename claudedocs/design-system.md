# デザインシステム設計書

**プロジェクト名**: 橋本佑介研究室ホームページ
**作成日**: 2025-10-14
**バージョン**: 1.0

---

## 1. デザインコンセプト

### 1.1 コアバリュー

**"シンプル・明快・アクセシブル"**

- **シンプル**: 余計な装飾を排除し、コンテンツに集中
- **明快**: 情報が探しやすく、理解しやすい構造
- **アクセシブル**: すべてのユーザーが利用できるデザイン

### 1.2 デザイン原則

1. **コンテンツファースト**: デザインはコンテンツを引き立てる役割
2. **ホワイトスペース**: 余白を活用し、読みやすさを向上
3. **一貫性**: 全ページで統一されたルックアンドフィール
4. **レスポンシブ**: すべてのデバイスで快適な体験
5. **パフォーマンス**: 高速な読み込みと滑らかな動作

### 1.3 参考: 2025年のトレンド

調査結果から採用する要素：
- ✅ ミニマリストデザイン（アカデミック向け）
- ✅ ホワイトスペースの活用
- ✅ モバイルファースト
- ✅ モダンCSS（Grid, Container Queries）
- ✅ アクセシビリティ重視

---

## 2. カラーシステム

### 2.1 カラーパレット

#### プライマリーカラー（メインブランド）

```css
/* ダークネイビー系（落ち着いた学術的印象） */
--color-primary-900: #1a252f;  /* 最も濃い */
--color-primary-700: #2c3e50;  /* メイン */
--color-primary-500: #34495e;  /* アクセント */
--color-primary-300: #5d6d7e;  /* ライト */
```

#### ニュートラルカラー（背景・テキスト）

```css
/* グレースケール */
--color-neutral-900: #2c3e50;  /* テキスト濃 */
--color-neutral-700: #5d6d7e;  /* テキスト中 */
--color-neutral-500: #95a5a6;  /* テキスト薄 */
--color-neutral-300: #bdc3c7;  /* ボーダー */
--color-neutral-200: #ecf0f1;  /* 背景薄 */
--color-neutral-100: #f8f9fa;  /* 背景 */
--color-neutral-50: #ffffff;   /* ホワイト */
```

#### アクセントカラー（ポイント使用）

```css
/* アクセント（控えめな青緑） */
--color-accent-500: #3498db;   /* リンク、強調 */
--color-accent-300: #5dade2;   /* hover */

/* セマンティックカラー */
--color-success: #27ae60;      /* 成功 */
--color-warning: #f39c12;      /* 警告 */
--color-error: #e74c3c;        /* エラー */
--color-info: #3498db;         /* 情報 */
```

### 2.2 カラー使用ガイドライン

| 用途 | カラー | 例 |
|------|--------|-----|
| ヘッダー背景 | primary-900 | #1a252f |
| 本文テキスト | neutral-900 | #2c3e50 |
| 見出し | primary-700 | #2c3e50 |
| リンク | accent-500 | #3498db |
| リンクhover | accent-300 | #5dade2 |
| 背景（メイン） | neutral-50 | #ffffff |
| 背景（セクション） | neutral-100 | #f8f9fa |
| ボーダー | neutral-300 | #bdc3c7 |

### 2.3 アクセシビリティ基準

WCAGコントラスト比:
- 通常テキスト: 4.5:1以上
- 大きなテキスト: 3:1以上
- アクティブUI: 3:1以上

---

## 3. タイポグラフィ

### 3.1 フォントスタック

#### 日本語

```css
--font-family-jp:
  "Hiragino Kaku Gothic ProN",  /* macOS */
  "Hiragino Sans",               /* macOS 10.11+ */
  "Noto Sans JP",                /* Web Font */
  "Yu Gothic Medium",            /* Windows */
  "Meiryo",                      /* Windows 旧 */
  sans-serif;
```

#### 英語

```css
--font-family-en:
  "Inter",                       /* Web Font */
  -apple-system,                 /* macOS/iOS */
  BlinkMacSystemFont,            /* Chrome on macOS */
  "Segoe UI",                    /* Windows */
  "Roboto",                      /* Android */
  "Helvetica Neue",              /* macOS fallback */
  Arial,
  sans-serif;
```

#### モノスペース（コード）

```css
--font-family-mono:
  "SF Mono",                     /* macOS */
  "Monaco",
  "Consolas",                    /* Windows */
  "Liberation Mono",
  "Courier New",
  monospace;
```

### 3.2 フォントサイズ

#### デスクトップ

```css
--font-size-xs: 0.75rem;    /* 12px */
--font-size-sm: 0.875rem;   /* 14px */
--font-size-base: 1rem;     /* 16px - ベース */
--font-size-lg: 1.125rem;   /* 18px */
--font-size-xl: 1.25rem;    /* 20px */
--font-size-2xl: 1.5rem;    /* 24px */
--font-size-3xl: 1.875rem;  /* 30px */
--font-size-4xl: 2.25rem;   /* 36px */
--font-size-5xl: 3rem;      /* 48px */
```

#### モバイル（1rem = 16px）

```css
--font-size-mobile-xs: 0.75rem;   /* 12px */
--font-size-mobile-sm: 0.875rem;  /* 14px */
--font-size-mobile-base: 1rem;    /* 16px */
--font-size-mobile-lg: 1.125rem;  /* 18px */
--font-size-mobile-xl: 1.25rem;   /* 20px */
--font-size-mobile-2xl: 1.5rem;   /* 24px */
--font-size-mobile-3xl: 2rem;     /* 32px */
```

### 3.3 行間（Line Height）

```css
--line-height-tight: 1.25;    /* 見出し用 */
--line-height-normal: 1.5;    /* 本文用（デフォルト） */
--line-height-relaxed: 1.75;  /* 読みやすさ重視 */
--line-height-loose: 2;       /* 詩的な表現 */
```

### 3.4 フォントウェイト

```css
--font-weight-normal: 400;
--font-weight-medium: 500;
--font-weight-semibold: 600;
--font-weight-bold: 700;
```

### 3.5 タイポグラフィスケール

| 要素 | サイズ（Desktop） | サイズ（Mobile） | Weight | 用途 |
|------|------------------|-----------------|--------|------|
| H1 | 3rem (48px) | 2rem (32px) | 700 | ページタイトル |
| H2 | 2.25rem (36px) | 1.5rem (24px) | 600 | セクション見出し |
| H3 | 1.875rem (30px) | 1.25rem (20px) | 600 | サブセクション |
| H4 | 1.5rem (24px) | 1.125rem (18px) | 600 | 小見出し |
| Body | 1rem (16px) | 1rem (16px) | 400 | 本文 |
| Small | 0.875rem (14px) | 0.875rem (14px) | 400 | 注釈 |
| Caption | 0.75rem (12px) | 0.75rem (12px) | 400 | キャプション |

---

## 4. スペーシング（余白）

### 4.1 スペーシングスケール

8pxベースのスケール（8の倍数で統一）

```css
--spacing-0: 0;
--spacing-1: 0.25rem;   /* 4px */
--spacing-2: 0.5rem;    /* 8px */
--spacing-3: 0.75rem;   /* 12px */
--spacing-4: 1rem;      /* 16px */
--spacing-5: 1.25rem;   /* 20px */
--spacing-6: 1.5rem;    /* 24px */
--spacing-8: 2rem;      /* 32px */
--spacing-10: 2.5rem;   /* 40px */
--spacing-12: 3rem;     /* 48px */
--spacing-16: 4rem;     /* 64px */
--spacing-20: 5rem;     /* 80px */
--spacing-24: 6rem;     /* 96px */
```

### 4.2 使用ガイドライン

| 用途 | スペーシング | サイズ |
|------|-------------|--------|
| セクション間 | spacing-16〜24 | 64〜96px |
| コンポーネント間 | spacing-8〜12 | 32〜48px |
| 要素間 | spacing-4〜6 | 16〜24px |
| 小要素間 | spacing-2〜3 | 8〜12px |
| パディング（ボタン） | spacing-3 spacing-6 | 12px 24px |

---

## 5. レイアウトシステム

### 5.1 コンテナ幅

```css
--container-xs: 480px;
--container-sm: 640px;
--container-md: 768px;
--container-lg: 1024px;
--container-xl: 1280px;
--container-2xl: 1440px;

/* メインコンテナ */
--container-max-width: 1200px;
```

### 5.2 グリッドシステム

12カラムグリッド（CSS Grid使用）

```css
.grid {
  display: grid;
  grid-template-columns: repeat(12, 1fr);
  gap: var(--spacing-6);
}

/* レスポンシブ例 */
.col-12 { grid-column: span 12; }  /* 全幅 */
.col-6 { grid-column: span 6; }    /* 半分 */
.col-4 { grid-column: span 4; }    /* 1/3 */
.col-3 { grid-column: span 3; }    /* 1/4 */

@media (max-width: 768px) {
  .col-6, .col-4, .col-3 {
    grid-column: span 12;  /* モバイルは全幅 */
  }
}
```

### 5.3 ブレークポイント

```css
/* モバイル（デフォルト） */
@media (min-width: 0px) { ... }

/* タブレット */
@media (min-width: 768px) {
  --breakpoint: 'tablet';
}

/* ノートPC */
@media (min-width: 1024px) {
  --breakpoint: 'desktop';
}

/* デスクトップ */
@media (min-width: 1440px) {
  --breakpoint: 'wide';
}
```

---

## 6. コンポーネント

### 6.1 ボタン

#### プライマリーボタン

```css
.btn-primary {
  padding: var(--spacing-3) var(--spacing-6);
  background: var(--color-primary-700);
  color: white;
  border: 2px solid var(--color-primary-700);
  border-radius: 4px;
  font-weight: var(--font-weight-medium);
  min-height: 44px;  /* タッチ対応 */
  transition: all 0.3s ease;
}

.btn-primary:hover {
  background: var(--color-primary-500);
  border-color: var(--color-primary-500);
}

.btn-primary:focus {
  outline: 2px solid var(--color-accent-500);
  outline-offset: 2px;
}
```

#### セカンダリーボタン

```css
.btn-secondary {
  padding: var(--spacing-3) var(--spacing-6);
  background: transparent;
  color: var(--color-primary-700);
  border: 2px solid var(--color-primary-700);
  border-radius: 4px;
  font-weight: var(--font-weight-medium);
  min-height: 44px;
}

.btn-secondary:hover {
  background: var(--color-primary-700);
  color: white;
}
```

### 6.2 カード

```css
.card {
  background: white;
  border: 1px solid var(--color-neutral-300);
  border-radius: 8px;
  padding: var(--spacing-6);
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
  transition: all 0.3s ease;
}

.card:hover {
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.12);
  transform: translateY(-4px);
}
```

### 6.3 ナビゲーション

#### デスクトップヘッダー

```css
header {
  position: sticky;
  top: 0;
  background: var(--color-primary-900);
  color: white;
  padding: var(--spacing-4) var(--spacing-6);
  z-index: 100;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

nav a {
  color: white;
  text-decoration: none;
  padding: var(--spacing-2) var(--spacing-4);
  border-radius: 4px;
  transition: background 0.2s;
}

nav a:hover,
nav a[aria-current="page"] {
  background: rgba(255, 255, 255, 0.1);
}
```

#### モバイルボトムナビゲーション

```css
.bottom-nav {
  position: fixed;
  bottom: 0;
  left: 0;
  right: 0;
  background: white;
  border-top: 1px solid var(--color-neutral-300);
  display: none;  /* デスクトップは非表示 */
  z-index: 100;
  box-shadow: 0 -2px 8px rgba(0, 0, 0, 0.08);
}

@media (max-width: 768px) {
  .bottom-nav {
    display: flex;
    justify-content: space-around;
  }

  .bottom-nav a {
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: var(--spacing-2);
    min-width: 44px;
    min-height: 44px;
    color: var(--color-neutral-700);
    text-decoration: none;
    font-size: var(--font-size-xs);
  }

  .bottom-nav a[aria-current="page"] {
    color: var(--color-primary-700);
    font-weight: var(--font-weight-medium);
  }
}
```

### 6.4 フォーム

```css
input, textarea, select {
  width: 100%;
  padding: var(--spacing-3);
  border: 2px solid var(--color-neutral-300);
  border-radius: 4px;
  font-size: var(--font-size-base);
  font-family: inherit;
  transition: border-color 0.2s;
}

input:focus, textarea:focus, select:focus {
  outline: none;
  border-color: var(--color-accent-500);
  box-shadow: 0 0 0 3px rgba(52, 152, 219, 0.1);
}

label {
  display: block;
  font-weight: var(--font-weight-medium);
  margin-bottom: var(--spacing-2);
  color: var(--color-neutral-900);
}
```

---

## 7. アニメーション

### 7.1 トランジション

```css
/* 基本トランジション */
--transition-fast: 0.15s ease;
--transition-base: 0.3s ease;
--transition-slow: 0.5s ease;

/* イージング関数 */
--ease-in-out: cubic-bezier(0.4, 0, 0.2, 1);
--ease-out: cubic-bezier(0, 0, 0.2, 1);
--ease-in: cubic-bezier(0.4, 0, 1, 1);
```

### 7.2 使用ガイドライン

- ホバー効果: 0.15〜0.3秒
- ページ遷移: 0.3〜0.5秒
- モーダル表示: 0.3秒
- 無効化: `prefers-reduced-motion` 対応必須

```css
@media (prefers-reduced-motion: reduce) {
  *,
  *::before,
  *::after {
    animation-duration: 0.01ms !important;
    transition-duration: 0.01ms !important;
  }
}
```

---

## 8. アイコン

### 8.1 アイコンシステム

絵文字を活用（軽量、Unicode標準）

| 用途 | アイコン | コード |
|------|---------|--------|
| ホーム | 🏠 | U+1F3E0 |
| 研究 | 🔬 | U+1F52C |
| 論文 | 📚 | U+1F4DA |
| ニュース | 📰 | U+1F4F0 |
| メンバー | 👥 | U+1F465 |
| お問い合わせ | ✉️ | U+2709 |
| リンク | 🔗 | U+1F517 |
| ダウンロード | 📥 | U+1F4E5 |

### 8.2 代替案

- Font Awesome（軽量版）
- Heroicons（Tailwind製）
- Material Icons

---

## 9. レスポンシブ画像

### 9.1 画像最適化

```html
<picture>
  <source
    srcset="image-mobile.webp"
    media="(max-width: 768px)"
    type="image/webp">
  <source
    srcset="image-desktop.webp"
    media="(min-width: 769px)"
    type="image/webp">
  <img
    src="image-fallback.jpg"
    alt="説明文"
    loading="lazy"
    width="800"
    height="600">
</picture>
```

### 9.2 画像サイズガイドライン

| デバイス | 最大幅 | フォーマット |
|---------|-------|------------|
| モバイル | 800px | WebP/JPEG |
| タブレット | 1200px | WebP/JPEG |
| デスクトップ | 1920px | WebP/JPEG |

---

## 10. アクセシビリティ

### 10.1 必須対応

- [ ] セマンティックHTML（h1-h6, nav, main, article）
- [ ] キーボードナビゲーション（Tab, Enter, Space）
- [ ] フォーカス表示（outline: 2px solid）
- [ ] カラーコントラスト（WCAG AA: 4.5:1以上）
- [ ] スクリーンリーダー対応（aria-label, role）
- [ ] スキップリンク（"Skip to content"）

### 10.2 ARIA属性

```html
<!-- ナビゲーション -->
<nav role="navigation" aria-label="メインナビゲーション">

<!-- 現在のページ -->
<a href="index.html" aria-current="page">ホーム</a>

<!-- ボタン -->
<button aria-label="メニューを開く" aria-expanded="false">

<!-- モーダル -->
<div role="dialog" aria-modal="true" aria-labelledby="modal-title">
```

---

## 11. ダークモード（オプション）

### 11.1 CSS変数の切り替え

```css
:root {
  --bg-primary: #ffffff;
  --text-primary: #2c3e50;
}

@media (prefers-color-scheme: dark) {
  :root {
    --bg-primary: #1a252f;
    --text-primary: #ecf0f1;
  }
}

/* 手動切り替え */
[data-theme="dark"] {
  --bg-primary: #1a252f;
  --text-primary: #ecf0f1;
}
```

---

## 12. パフォーマンス

### 12.1 CSS最適化

- CSSファイルは1つに統合（HTTP/2対応）
- 不要なセレクタ削除
- ミニファイ（圧縮）
- クリティカルCSSのインライン化

### 12.2 フォント最適化

```css
/* システムフォント優先 */
font-family: -apple-system, BlinkMacSystemFont, sans-serif;

/* Web Fontは必要最小限 */
@font-face {
  font-family: 'Inter';
  src: url('inter-subset.woff2') format('woff2');
  font-display: swap;  /* FOUT防止 */
}
```

---

## 13. 実装チェックリスト

### 基本
- [ ] CSS変数定義（colors, spacing, typography）
- [ ] リセットCSS適用
- [ ] レスポンシブグリッド実装
- [ ] モバイルファーストCSS

### コンポーネント
- [ ] ヘッダーナビゲーション
- [ ] ボトムナビゲーション（モバイル）
- [ ] ボタン（primary, secondary）
- [ ] カード
- [ ] フォーム要素

### アクセシビリティ
- [ ] セマンティックHTML
- [ ] キーボードナビゲーション
- [ ] ARIA属性
- [ ] カラーコントラスト検証

### パフォーマンス
- [ ] 遅延画像読み込み
- [ ] WebP対応
- [ ] CSSミニファイ
- [ ] Lighthouse 90+

---

## 変更履歴

| 日付 | バージョン | 変更内容 | 担当 |
|------|-----------|---------|------|
| 2025-10-14 | 1.0 | 初版作成 | Claude |

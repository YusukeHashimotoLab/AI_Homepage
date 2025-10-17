# 高品質記事作成プロシージャー

**目的**: 7つのエージェントが協力し、学術的に正確で教育的に優れた最高品質の記事を作成する

**品質方針**: 科学的正確性 > 教育的効果 > 実装品質 > UX最適化

**想定所要時間**: 15-20時間/記事

**目標品質スコア**: 90点以上 (100点満点)

---

## プロシージャー全体フロー

```
Phase 0: 企画・準備 (30-60分)
  ↓
Phase 1: 多角的情報収集 (2-4時間)
  ↓
Phase 2: 初稿作成 (3-5時間)
  ↓
Phase 3: 学術レビュー第1サイクル (2-3時間)
  ↓ [Score < 80] → Phase 2へ戻る
  ↓ [Score ≥ 80]
Phase 4: 教育的レビュー (1-2時間)
  ↓
Phase 5: 実装検証 (2-3時間)
  ↓
Phase 6: UX最適化 (1-2時間)
  ↓
Phase 7: 学術レビュー第2サイクル (1-2時間)
  ↓ [Score < 90] → Phase 4へ戻る
  ↓ [Score ≥ 90]
Phase 8: 統合品質保証 (1-2時間)
  ↓
Phase 9: 最終承認・公開 (30分)
```

---

## Phase 0: 企画・準備

**参加エージェント**: 全エージェント

**目的**: トピックの明確化と執筆戦略の策定

### タスク

#### Orchestrator
- 対象読者レベル決定 (学部生/修士生/産業界)
- 前提知識の範囲特定
- 学習目標の明確化 (3-5個)
- 記事タイプ決定 (理論解説/チュートリアル/応用事例)

#### Scholar Agent
- トピックの最新研究動向確認
- 重要論文リスト作成 (Top 10-20)
- 研究の歴史的背景整理

#### Data Agent
- 関連データセット確認
- 利用可能なコード例調査
- ツール・ライブラリの最新版確認

#### Tutor Agent
- よくある質問・誤解の特定
- 学習者がつまずきやすいポイント予測
- 段階的学習パスの設計

#### Design Agent
- 競合サイトの優れた記事分析 (3-5サイト)
- ベストプラクティスの抽出
- UI/UXパターンの推奨

### 成果物
- 記事企画書 (Markdown, 1000-2000語)
- 参考文献リスト (BibTeX)
- 学習目標リスト
- コンテンツ構造案 (アウトライン)

### 品質ゲート
- [ ] 学習目標が明確に定義されている
- [ ] 参考論文が10件以上収集されている
- [ ] 対象読者レベルが明確

---

## Phase 1: 多角的情報収集

**参加エージェント**: Scholar, Data, Tutor

**目的**: 最高品質のソース情報を徹底的に収集

### 1.1 Scholar Agent: 文献調査

**タスク**:
- Google Scholar: 最新論文20件収集 (過去5年)
- arXiv: プレプリント論文確認
- 重要論文の詳細読解・要約 (各500-1000語)
- 引用関係の分析 (被引用数上位)
- 研究手法の比較分析

**出力**:
- 論文要約集 (papers_summary.md)
- 重要な数式・定理リスト
- 研究の系譜図
- コンセンサス vs 議論のあるポイント

### 1.2 Data Agent: 実装リソース収集

**タスク**:
- GitHub: スター数上位の実装コード調査
- データセット: ベンチマークデータ収集
- ツール比較: ライブラリの長所短所整理
- コード品質評価: ベストプラクティス抽出

**出力**:
- 推奨ツールリスト (比較表付き)
- サンプルコード集
- データセット詳細 (形式、サイズ、ライセンス)

### 1.3 Tutor Agent: 教育的視点収集

**タスク**:
- Stack Overflow: よくある質問分析
- GitHub Discussions: 学習者の疑問点収集
- 教科書・教材レビュー: 説明アプローチ分析
- 前提知識ギャップ特定

**出力**:
- FAQ候補リスト (20-30項目)
- 誤解しやすいポイント集
- 段階的説明の推奨構造

### 品質ゲート
- [ ] 論文収集: 20件以上 (過去3年が70%以上)
- [ ] 実装コード: 3つ以上の実行可能な例
- [ ] FAQ: 20項目以上

---

## Phase 2: 初稿作成

**参加エージェント**: Content Agent (主導), Scholar, Tutor

**目的**: 収集情報を統合し、高品質な初稿を作成

### 2.1 Content Agent: 構造設計と執筆

**推奨構造**:
```markdown
# [記事タイトル]

## 1. イントロダクション
- 動機 (なぜこのトピックが重要か)
- 学習目標 (この記事で学べること)
- 前提知識 (必要な予備知識)

## 2. 背景知識
- 基礎概念の復習
- 関連する理論の説明

## 3. 理論解説
- 主要概念の詳細説明
- 数式と直感的説明の組み合わせ
- 図表による視覚化

## 4. 実装
- アルゴリズムの疑似コード
- Pythonによる実装例
- コードの詳細解説

## 5. 応用例
- 実世界での利用シーン
- ケーススタディ
- 結果の解釈

## 6. 演習問題
- 基礎問題 (理解度確認)
- 応用問題 (実装力確認)

## 7. まとめ
- 要点の再確認
- 次のステップ (発展的学習)
- 参考文献

## 参考文献
```

**執筆ガイドライン**:
- 文字数: 5,000-8,000語
- 数式: LaTeX記法で正確に記述
- コード: 詳細なコメント付き
- 図表: 説明的なキャプション
- 引用: 必ず出典明示

### 2.2 Scholar Agent: リアルタイム事実確認

**並行タスク**:
- Content Agent執筆中に事実確認
- 論文引用の正確性チェック
- 数式の出典確認
- 用語定義の標準性確認

### 2.3 Tutor Agent: 教育的配慮チェック

**並行タスク**:
- 前提知識の明示は十分か
- 説明順序は段階的か
- 例示は適切か
- 難易度は対象読者に適合か

### 成果物
- 初稿 (Markdown, 5000-8000語)
- 実装コード (Jupyter Notebook)
- 図表素材
- 参考文献リスト (BibTeX)

### 品質ゲート
- [ ] 文字数: 5,000語以上
- [ ] セクション: 7つ以上
- [ ] コード例: 3つ以上
- [ ] 参考文献: 10件以上

---

## Phase 3: 学術レビュー第1サイクル

**参加エージェント**: Academic Reviewer Agent (主導), Scholar

**目的**: 科学的正確性の徹底検証

### レビュー観点 (100点満点)

#### 1. 科学的正確性 (40点)
- [ ] 理論の正しさ: 定義、定理、証明
- [ ] 数式の正確性: 記法、導出過程、単位
- [ ] 用語使用: 標準用語、略語の定義
- [ ] 引用の正確性: 出典明示、引用スタイル
- [ ] 最新性: 最近の研究成果の反映

**検証方法**:
- 論文原文との照合 (Scholar Agent支援)
- 数式の再導出
- 定義の教科書的定義との比較
- arXiv/Google Scholarでの事実確認

#### 2. 完全性 (20点)
- [ ] 前提知識の明示: 必要な予備知識リスト
- [ ] 論理展開: 飛躍のない連続性
- [ ] 例外・制約: 適用範囲の明確化
- [ ] 反例: 成立しないケースの説明
- [ ] 比較: 類似手法との違い

#### 3. 教育的配慮 (20点)
- [ ] 対象読者適合: 難易度の適切性
- [ ] 説明明瞭さ: 専門用語の説明、比喩
- [ ] 例示: 具体例、図表、コード
- [ ] 段階性: 簡単→複雑への移行

#### 4. 実装品質 (20点)
- [ ] コード正確性: 構文、ロジック
- [ ] 実行可能性: エラーなく実行完了
- [ ] ベストプラクティス: PEP8, 型ヒント
- [ ] 再現性: 環境依存の排除

### レビュー結果フォーマット

```markdown
# Academic Review Report - Round 1

## スコア
- 科学的正確性: X / 40
- 完全性: Y / 20
- 教育的配慮: Z / 20
- 実装品質: W / 20
- **総合スコア: N / 100**

## Critical Issues (必須修正)
1. [理論的誤り] 問題の詳細と修正案
2. [数式エラー] 問題の詳細と修正案

## Recommendations (推奨改善)
1. [説明不足] 改善提案
2. [例示追加] 改善提案

## 判定
- [ ] 承認 (Score ≥ 80)
- [ ] 要修正 (Score < 80)
```

### 判定基準
- **Score ≥ 90**: 優秀、Phase 4へ
- **Score 80-89**: 良好、軽微な修正後Phase 4へ
- **Score < 80**: 要大幅修正、Phase 2へ戻る (最大3サイクル)

### 品質ゲート
- [ ] Academic Review Score ≥ 80
- [ ] Critical Issues: 全て解決

---

## Phase 4: 教育的レビュー

**参加エージェント**: Tutor Agent (主導), Academic Reviewer

**目的**: 学習効果の最大化

### レビュー項目

#### 4.1 学習曲線の滑らかさ
- [ ] 難易度の急激な変化はないか
- [ ] 新しい概念の導入は段階的か (1セクションあたり3個以下)
- [ ] 前のセクションとの接続は明確か

**改善提案例**:
- トランジション文の追加
- 難易度グラフ作成
- セクション分割

#### 4.2 認知負荷の最適化
- [ ] 長文パラグラフの有無 (推奨: 5文以下)
- [ ] 複雑な数式に言葉での説明があるか
- [ ] 一度に提示される情報量は適切か

**改善提案例**:
- 箇条書きへの変換
- 図表による視覚化
- セクション細分化

#### 4.3 インタラクティブ性
- [ ] 読者に考えさせる質問の挿入
- [ ] 演習問題の配置 (各セクション末)
- [ ] 実行可能なコード例

**追加要素**:
- "考えてみよう" コラム
- "やってみよう" 演習
- "よくある間違い" 注意喚起

#### 4.4 メタ認知支援
- [ ] セクション冒頭: 学習目標明示
- [ ] セクション末: 要点まとめ
- [ ] 記事末: 理解度チェックリスト
- [ ] 次のステップ: 発展的学習への道標

### 成果物
- 教育的レビューレポート
- 改善提案リスト (優先度付き)
- 追加コンテンツ案 (コラム、演習問題)

### 品質ゲート
- [ ] 全改善提案に対応
- [ ] 演習問題: 各セクション1問以上

---

## Phase 5: 実装検証

**参加エージェント**: Data Agent (主導), Maintenance Agent

**目的**: コードの完全性と実行可能性の保証

### 5.1 コード品質検証

#### レベル1: 構文・スタイル
```bash
# 自動実行ツール
pylint notebook.py --rcfile=.pylintrc
black notebook.py --check
mypy notebook.py
isort notebook.py --check

# 目標
# - Pylintスコア: 9.0以上
# - 型ヒント付与率: 100%
```

#### レベル2: 実行可能性
**テスト環境**:
1. Python 3.11 (最新)
2. Python 3.9 (最小サポート)
3. Google Colab環境
4. クリーンな仮想環境

**チェック項目**:
- [ ] 全セル実行エラーなし
- [ ] 実行時間 < 10分/ノートブック
- [ ] メモリ使用量 < 2GB

#### レベル3: 再現性
- [ ] requirements.txtの完全性
- [ ] ランダムシード固定
- [ ] データパス相対化
- [ ] プラットフォーム依存性排除

#### レベル4: 教育的コード品質
- [ ] コメント率: 30%以上
- [ ] 関数名: 説明的
- [ ] マジックナンバー排除
- [ ] 複雑度: McCabe < 10

### 5.2 セキュリティチェック (Maintenance Agent)
```bash
# 依存関係スキャン
safety check
pip-audit

# ライセンス確認
pip-licenses
```

### 成果物
- コード品質レポート
- 実行テスト結果 (4環境)
- セキュリティレポート
- 修正後のJupyter Notebook

### 品質ゲート
- [ ] Pylintスコア: 9.0以上
- [ ] 全環境で実行成功
- [ ] セキュリティ脆弱性: なし

---

## Phase 6: UX最適化

**参加エージェント**: Design Agent

**目的**: 読みやすさと視覚的魅力の最大化

### 6.1 可読性分析
```python
# 指標測定
from textstat import textstat

flesch_reading_ease = textstat.flesch_reading_ease(text)
# 目標: 60以上

avg_sentence_length = textstat.avg_sentence_length(text)
# 目標: 20語以下
```

**改善アクション**:
- 長文の分割提案
- 箇条書き化提案
- 読みやすい言い回しへの変換

### 6.2 視覚的階層
- [ ] 見出しレベル (h1-h6) の適切な使用
- [ ] セクション区切りの明確さ
- [ ] 強調 (太字/イタリック) の一貫性

### 6.3 図表の最適化
- [ ] 図表の解像度 (推奨: 150dpi以上)
- [ ] キャプションの説明性
- [ ] 本文との連携 (図1参照)
- [ ] カラーユニバーサルデザイン

### 6.4 モバイル最適化
- [ ] 数式のモバイル表示確認
- [ ] コードブロックのスクロール
- [ ] 画像のレスポンシブ対応
- [ ] タップターゲットサイズ (44px)

### 6.5 競合サイト比較
**分析対象**:
- Materials Project blog
- Towards Data Science (MI記事)
- Fast.ai ドキュメント

**ベンチマーキング**:
- レイアウトパターン抽出
- インタラクティブ要素の分析
- ナビゲーション設計の評価

### 成果物
- UX改善レポート
- CSS改善提案
- 図表再作成リスト
- モバイル最適化チェックリスト

### 品質ゲート
- [ ] Flesch Reading Ease: 60以上
- [ ] 平均文長: 20語以下
- [ ] 図表品質: 全て最適化済み

---

## Phase 7: 学術レビュー第2サイクル

**参加エージェント**: Academic Reviewer Agent, Scholar

**目的**: 全修正後の最終学術検証

### Phase 3との差分
- Phase 3-6の全修正反映確認
- 新たに追加されたコンテンツの検証
- 全体整合性の確認

### 厳格化された基準
```
Phase 3: 初稿レビュー (合格基準: 80点)
Phase 7: 最終レビュー (合格基準: 90点)
```

**追加チェック**:
- [ ] 修正漏れの有無
- [ ] 新たに導入されたエラーの確認
- [ ] 全体の流れの最終確認
- [ ] 参考文献リストの完全性

### レビュー結果フォーマット
```markdown
# Academic Review Report - Round 2 (Final)

## スコア改善
- Round 1: X / 100
- Round 2: Y / 100
- 改善幅: +Z点

## 修正の質評価
- Critical Issues: 全て解決 ✅/❌
- Recommendations: N%対応 ✅/❌

## 最終判定
- [ ] 承認 (Score ≥ 90)
- [ ] 条件付き承認 (Score 85-89)
- [ ] 再修正 (Score < 85)
```

### 判定基準
- **Score ≥ 90**: Phase 8へ進む
- **Score 85-89**: 軽微な修正後Phase 8へ
- **Score < 85**: Phase 4へ戻り再修正 (最大2サイクル)

### 品質ゲート
- [ ] Academic Review Score ≥ 90
- [ ] 全Critical Issues解決

---

## Phase 8: 統合品質保証

**参加エージェント**: 全エージェント

**目的**: 全エージェント総動員での最終チェック

### 8.1 並行最終チェック

#### Scholar Agent: 最新性確認
- [ ] Phase 0以降の新規論文確認
- [ ] 内容の古さがないか最終チェック
- [ ] 参考文献リストの最終更新

#### Content Agent: 統一性確認
- [ ] 用語の一貫性 (表記ゆれチェック)
- [ ] トーン・スタイルの統一
- [ ] 文末表現の統一 (です/ます調)
- [ ] 記号使用の一貫性

#### Tutor Agent: 最終教育的チェック
- [ ] 学習目標の達成度確認
- [ ] 演習問題の適切性
- [ ] 理解度チェックリストの完全性
- [ ] 次のステップガイドの充実度

#### Data Agent: リソース最終確認
- [ ] 全リンクの有効性確認
- [ ] データセットURLの生存確認
- [ ] GitHubリポジトリの最新版確認
- [ ] ライセンス情報の最終確認

#### Design Agent: 最終UXチェック
```bash
# Lighthouseスコア測定
lighthouse https://localhost:8000/article.html --view

# 目標
# - Performance: 95以上
# - Accessibility: 100
```

- [ ] モバイル実機確認
- [ ] ブラウザ互換性テスト
- [ ] 印刷プレビュー確認

#### Maintenance Agent: 技術的品質保証
- [ ] 全リンク切れチェック (自動)
- [ ] 画像ファイルサイズ最適化
- [ ] メタデータ完全性 (title, description, OGP)
- [ ] sitemap.xml更新

#### Academic Reviewer Agent: 最終署名
- [ ] 全エージェントのチェック結果レビュー
- [ ] 残存リスクの評価
- [ ] 公開承認の最終判断
- [ ] 品質保証証明書発行

### 8.2 統合レポート生成

```markdown
# 統合品質保証レポート

## 記事情報
- タイトル: [記事タイトル]
- 対象読者: [レベル]
- 文字数: X語
- 所要学習時間: Y時間
- コード例: N個 (全て実行確認済み)

## 品質スコア
| エージェント | 観点 | スコア | 判定 |
|------------|------|--------|------|
| Academic Reviewer | 学術的正確性 | XX/100 | ✅ |
| Tutor | 教育的効果 | XX/100 | ✅ |
| Data | 実装品質 | XX/100 | ✅ |
| Design | UX品質 | XX/100 | ✅ |
| Maintenance | 技術品質 | XX/100 | ✅ |
| **総合** | **全体評価** | **XX/100** | **✅** |

## プロセス統計
- 総所要時間: X時間
- レビューサイクル: N回
- 修正回数: N回
- 参考論文: N件
- 実装検証: 4環境でテスト完了

## 最終承認
- [X] 公開承認
- 承認者: Academic Reviewer Agent
- 承認日時: YYYY-MM-DD HH:MM
- 品質保証証明書: QA-YYYY-MM-DD-NNN
```

### 品質ゲート
- [ ] 全エージェント承認
- [ ] 総合スコア: 90以上
- [ ] Lighthouseスコア: Performance 95+, Accessibility 100

---

## Phase 9: 最終承認・公開

**参加エージェント**: Orchestrator, Maintenance Agent

**目的**: 人間による最終確認と公開実行

### 9.1 人間レビュー (Dr. Hashimoto)

**レビュー内容**:
```
確認項目:
□ 統合品質保証レポート確認
□ 記事本文の通読 (または抜粋)
□ Jupyter Notebook動作確認 (1つ選択実行)
□ デザインプレビュー (モバイル/デスクトップ)
□ メタ情報確認 (著者名、日付、ライセンス)

承認判断:
- 承認: Phase 9.2へ進む
- 条件付き承認: 軽微な修正後公開
- 差し戻し: 該当Phaseへ戻る
```

### 9.2 公開実行 (Orchestrator)

```bash
# Git commit
git add content/methods/[topic].md
git add notebooks/[topic].ipynb
git add data/tutorials.json

git commit -m "$(cat <<'EOF'
Add comprehensive guide: [Title]

- Target audience: [Level]
- Quality score: XX/100 (7 agents review)
- Includes: Theory, implementation, exercises
- Verified: 4 environments tested
- Review cycles: N rounds

Co-Authored-By: Scholar Agent <agent@mi-knowledge-hub>
Co-Authored-By: Content Agent <agent@mi-knowledge-hub>
Co-Authored-By: Academic Reviewer Agent <agent@mi-knowledge-hub>
Co-Authored-By: Tutor Agent <agent@mi-knowledge-hub>
Co-Authored-By: Data Agent <agent@mi-knowledge-hub>
Co-Authored-By: Design Agent <agent@mi-knowledge-hub>
Co-Authored-By: Maintenance Agent <agent@mi-knowledge-hub>
EOF
)"

# Push to GitHub Pages
git push origin main

# 公開確認
curl -I https://mi-knowledge-hub.github.io/methods/[topic]
```

### 9.3 公開後モニタリング (Maintenance Agent)

**初期24時間モニタリング**:
- [ ] リンク切れ発生: なし
- [ ] ページ読み込み時間: < 2秒
- [ ] エラー発生: なし
- [ ] モバイル表示: 正常
- [ ] Colab連携: 正常

---

## 品質保証マトリックス

### 各Phaseの品質ゲート一覧

| Phase | 合格基準 | 不合格時アクション |
|-------|---------|-----------------|
| Phase 0 | 学習目標明確、論文10件以上 | 追加企画会議 |
| Phase 1 | 論文20件以上、FAQ20項目以上 | 追加調査実行 |
| Phase 2 | 文字数5000語以上、コード3例以上 | 不足セクション追加 |
| Phase 3 | Academic Review ≥ 80点 | Phase 2へ戻る |
| Phase 4 | 全改善提案対応 | 未対応項目の実施 |
| Phase 5 | コード実行成功率100% | バグ修正 |
| Phase 6 | Flesch 60+, Lighthouse Accessibility 100 | UX改善 |
| Phase 7 | Academic Review ≥ 90点 | Phase 4へ戻る |
| Phase 8 | 全エージェント承認、総合90+ | 該当Phaseへ戻る |
| Phase 9 | 人間承認 | 該当Phaseへ戻る |

### レビューサイクルの制限

```
Phase 2 ⇄ Phase 3: 最大3サイクル
Phase 4 ⇄ Phase 7: 最大2サイクル

制限超過の場合:
→ トピック再検討 (Phase 0へ戻る)
→ または人間による大幅修正
```

---

## 期待される品質レベル

### 学術的品質
- 論文引用: 15-20件 (最新5年以内を70%以上)
- 事実誤認: ゼロ
- 数式エラー: ゼロ
- 用語不統一: ゼロ

### 教育的品質
- 学習目標達成率: 90%以上 (ユーザーテスト)
- 理解度: 対象読者の80%が完了可能
- 演習問題: 各セクション1問以上
- FAQ: 20項目以上

### 実装品質
- コード実行成功率: 100%
- Pylintスコア: 9.0以上
- テスト環境: 4種類で検証済み
- ドキュメント: 全関数にdocstring

### UX品質
- Lighthouse Performance: 95以上
- Lighthouse Accessibility: 100
- Flesch Reading Ease: 60以上
- モバイル対応: 完全

---

## プロセス改善のフィードバックループ

### Phase 9後の振り返り (Post-Mortem)

**全エージェント参加**:

#### 議題
1. **所要時間レビュー**
   - ボトルネックの特定
   - 効率化可能なポイント

2. **品質スコアの分析**
   - 低スコア項目の原因分析
   - 改善策の提案

3. **プロセスの問題点**
   - エージェント間連携の課題
   - 情報共有の改善点

4. **ベストプラクティスの抽出**
   - 今回うまくいった手法
   - 次回記事への適用

#### 成果物
- プロセス改善提案書 (process_improvement_YYYYMMDD.md)
- 次回記事への推奨事項
- エージェント設定の最適化案

---

## クイックリファレンス

### エージェント役割早見表

| Agent | 主な役割 | 主要タスク |
|-------|---------|----------|
| **Scholar** | 文献調査 | 論文収集・要約、事実確認 |
| **Content** | 執筆 | 初稿作成、構造設計 |
| **Tutor** | 教育最適化 | 学習効果分析、FAQ作成 |
| **Data** | 実装検証 | コード品質、リソース管理 |
| **Design** | UX最適化 | 可読性、視覚デザイン |
| **Maintenance** | 技術保守 | セキュリティ、リンクチェック |
| **Academic Reviewer** | 品質保証 | 学術レビュー、承認判定 |

### 重要な判定基準

```
Phase 3: Academic Review ≥ 80点 → Phase 4へ
Phase 7: Academic Review ≥ 90点 → Phase 8へ
Phase 8: 総合スコア ≥ 90点 → Phase 9へ
```

### 緊急時の対応

**品質基準に達しない場合**:
- 3サイクル試行後も80点未満 → トピック見直し
- 致命的な誤りの発見 → 即座にPhase 2へ戻る
- 技術的実装不可能 → Phase 0で代替アプローチ検討

---

## 付録: 実行コマンド例

### Scholar Agent: 論文収集
```bash
python agents/scholar_agent.py \
  --query "materials informatics bayesian optimization" \
  --days 1825 \
  --max-results 20 \
  --output papers_summary.md
```

### Content Agent: 記事生成
```bash
python agents/content_agent.py \
  --topic "bayesian_optimization" \
  --level "undergraduate" \
  --type "tutorial" \
  --output content/methods/bayesian_optimization.md
```

### Academic Reviewer Agent: レビュー実行
```bash
python agents/academic_reviewer_agent.py \
  --review content/methods/bayesian_optimization.md \
  --context '{"target_audience": "undergraduate", "topic": "bayesian_optimization"}' \
  --output reviews/bayesian_optimization_review_round1.md
```

### Data Agent: コード検証
```bash
python agents/data_agent.py \
  --verify notebooks/bayesian_optimization.ipynb \
  --environments python311,python39,colab \
  --output verification_report.md
```

### Design Agent: UX分析
```bash
python agents/design_agent.py \
  --analyze http://localhost:8000/methods/bayesian_optimization \
  --competitors materialsproject.org,towardsdatascience.com \
  --output ux_analysis.md
```

### Maintenance Agent: 最終チェック
```bash
python agents/maintenance_agent.py \
  --check-all \
  --path content/methods/bayesian_optimization.md \
  --output final_check.md
```

---

**Document Version**: 1.0
**Last Updated**: 2025-10-15
**Maintained By**: MI Knowledge Hub Development Team

# AI寺子屋 - 高品質教育コンテンツ作成ワークフロー

**目的**: 7つのClaude Code subagentsが協力し、学術的に正確で教育的に優れた最高品質の記事を作成する

**品質方針**: 科学的正確性 > 教育的効果 > 実装品質 > UX最適化

**想定所要時間**: 15-20時間/記事（トピックによって変動）

**目標品質スコア**: 90点以上 (100点満点)

---

## ワークフロー全体図

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

## 7つのSubagentsの役割

| Agent | 主な役割 | 主要フェーズ |
|-------|---------|------------|
| **Scholar Agent** | 論文収集・要約・事実確認 | Phase 0, 1, 3, 7, 8 |
| **Content Agent** | 記事構造設計・執筆 | Phase 0, 2 |
| **Academic Reviewer** | 学術レビュー・品質保証 | Phase 3, 7, 8, 9 |
| **Tutor Agent** | 教育的視点・FAQ作成 | Phase 0, 1, 2, 4, 8 |
| **Data Agent** | 実装検証・リソース管理 | Phase 0, 1, 5, 8 |
| **Design Agent** | UX最適化・アクセシビリティ | Phase 0, 6, 8 |
| **Maintenance Agent** | 技術的品質保証・リンクチェック | Phase 5, 8, 9 |

---

## Phase 0: 企画・準備

**参加Subagents**: 全Subagents

**目的**: トピックの明確化と執筆戦略の策定

### タスク

#### Content Agent（主導）
- 対象読者レベル決定（初級/中級/上級）
- 前提知識の範囲特定
- 学習目標の明確化（3-5個）
- 記事タイプ決定（理論解説/チュートリアル/応用事例）

#### Scholar Agent
- トピックの最新研究動向確認
- 重要論文リスト作成（Top 10-20）
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
- 関連サイトの優れた記事分析（3-5サイト）
- ベストプラクティスの抽出
- UI/UXパターンの推奨

### 成果物
- 記事企画書（Markdown, 1000-2000語）
- 参考文献リスト（BibTeX）
- 学習目標リスト
- コンテンツ構造案（アウトライン）

### 品質ゲート
- [ ] 学習目標が明確に定義されている
- [ ] 参考論文が10件以上収集されている
- [ ] 対象読者レベルが明確

---

## Phase 1: 多角的情報収集

**参加Subagents**: Scholar, Data, Tutor

**目的**: 最高品質のソース情報を徹底的に収集

### 1.1 Scholar Agent: 文献調査

**タスク**:
- Google Scholar: 最新論文20件収集（過去5年）
- 重要論文の詳細読解・要約（各500-1000語）
- 引用関係の分析（被引用数上位）
- 研究手法の比較分析

**出力**:
- 論文要約集（papers_summary.md）
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
- 推奨ツールリスト（比較表付き）
- サンプルコード集
- データセット詳細（形式、サイズ、ライセンス）

### 1.3 Tutor Agent: 教育的視点収集

**タスク**:
- Stack Overflow: よくある質問分析
- GitHub Discussions: 学習者の疑問点収集
- 教科書・教材レビュー: 説明アプローチ分析
- 前提知識ギャップ特定

**出力**:
- FAQ候補リスト（20-30項目）
- 誤解しやすいポイント集
- 段階的説明の推奨構造

### 品質ゲート
- [ ] 論文収集: 20件以上（過去3年が70%以上）
- [ ] 実装コード: 3つ以上の実行可能な例
- [ ] FAQ: 20項目以上

---

## Phase 2: 初稿作成

**参加Subagents**: Content Agent（主導）, Scholar, Tutor

**目的**: 収集情報を統合し、高品質な初稿を作成

### 2.1 Content Agent: 構造設計と執筆

**推奨構造**:
```markdown
# [記事タイトル]

## 1. イントロダクション
- 動機（なぜこのトピックが重要か）
- 学習目標（この記事で学べること）
- 前提知識（必要な予備知識）

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
- 基礎問題（理解度確認）
- 応用問題（実装力確認）

## 7. まとめ
- 要点の再確認
- 次のステップ（発展的学習）
- 参考文献
```

**執筆ガイドライン**:
- 文字数: 5,000-8,000語
- 数式: LaTeX記法で正確に記述
- コード: 詳細なコメント付き
- 図表: 説明的なキャプション
- 引用: 必ず出典明示

### 成果物
- 初稿（Markdown, 5000-8000語）
- 実装コード（Jupyter Notebook）
- 図表素材
- 参考文献リスト（BibTeX）

### 品質ゲート
- [ ] 文字数: 5,000語以上
- [ ] セクション: 7つ以上
- [ ] コード例: 3つ以上
- [ ] 参考文献: 10件以上

---

## Phase 3: 学術レビュー第1サイクル（80点ゲート）

**参加Subagents**: Academic Reviewer（主導）, Scholar

**目的**: 科学的正確性の徹底検証

### レビュー観点（100点満点）

#### 1. 科学的正確性（40点）
- [ ] 理論の正しさ: 定義、定理、証明
- [ ] 数式の正確性: 記法、導出過程、単位
- [ ] 用語使用: 標準用語、略語の定義
- [ ] 引用の正確性: 出典明示、引用スタイル
- [ ] 最新性: 最近の研究成果の反映

#### 2. 完全性（20点）
- [ ] 前提知識の明示
- [ ] 論理展開の連続性
- [ ] 例外・制約の明確化
- [ ] 反例の説明
- [ ] 類似手法との比較

#### 3. 教育的配慮（20点）
- [ ] 対象読者適合性
- [ ] 説明の明瞭さ
- [ ] 具体例・図表・コード
- [ ] 段階的な複雑性

#### 4. 実装品質（20点）
- [ ] コード正確性
- [ ] 実行可能性
- [ ] ベストプラクティス遵守
- [ ] 再現性

### 判定基準
- **Score ≥ 90**: 優秀、Phase 4へ
- **Score 80-89**: 良好、軽微な修正後Phase 4へ
- **Score < 80**: 要大幅修正、Phase 2へ戻る（最大3サイクル）

### 品質ゲート
- [ ] Academic Review Score ≥ 80
- [ ] Critical Issues: 全て解決

---

## Phase 4: 教育的レビュー

**参加Subagents**: Tutor Agent（主導）, Academic Reviewer

**目的**: 学習効果の最大化

### レビュー項目

#### 4.1 学習曲線の滑らかさ
- [ ] 難易度の急激な変化はないか
- [ ] 新しい概念の導入は段階的か（1セクションあたり3個以下）
- [ ] 前のセクションとの接続は明確か

#### 4.2 認知負荷の最適化
- [ ] 長文パラグラフの有無（推奨: 5文以下）
- [ ] 複雑な数式に言葉での説明があるか
- [ ] 一度に提示される情報量は適切か

#### 4.3 インタラクティブ性
- [ ] 読者に考えさせる質問の挿入
- [ ] 演習問題の配置（各セクション末）
- [ ] 実行可能なコード例

#### 4.4 メタ認知支援
- [ ] セクション冒頭: 学習目標明示
- [ ] セクション末: 要点まとめ
- [ ] 記事末: 理解度チェックリスト
- [ ] 次のステップ: 発展的学習への道標

### 品質ゲート
- [ ] 全改善提案に対応
- [ ] 演習問題: 各セクション1問以上

---

## Phase 5: 実装検証

**参加Subagents**: Data Agent（主導）, Maintenance Agent

**目的**: コードの完全性と実行可能性の保証

### 5.1 コード品質検証

#### レベル1: 構文・スタイル
```bash
pylint notebook.py --rcfile=.pylintrc
black notebook.py --check
mypy notebook.py
```

**目標**:
- Pylintスコア: 9.0以上
- 型ヒント付与率: 100%

#### レベル2: 実行可能性
**テスト環境**:
1. Python 3.11（最新）
2. Python 3.9（最小サポート）
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

### 品質ゲート
- [ ] Pylintスコア: 9.0以上
- [ ] 全環境で実行成功
- [ ] セキュリティ脆弱性: なし

---

## Phase 6: UX最適化

**参加Subagents**: Design Agent

**目的**: 読みやすさと視覚的魅力の最大化

### 6.1 可読性分析

**Flesch Reading Ease目標**: 60以上
**平均文長目標**: 20語以下

### 6.2 視覚的階層
- [ ] 見出しレベル（h1-h6）の適切な使用
- [ ] セクション区切りの明確さ
- [ ] 強調（太字/イタリック）の一貫性

### 6.3 図表の最適化
- [ ] 図表の解像度（推奨: 150dpi以上）
- [ ] キャプションの説明性
- [ ] 本文との連携（図1参照）
- [ ] カラーユニバーサルデザイン

### 6.4 モバイル最適化
- [ ] 数式のモバイル表示確認
- [ ] コードブロックのスクロール
- [ ] 画像のレスポンシブ対応
- [ ] タップターゲットサイズ（44px）

### 品質ゲート
- [ ] Flesch Reading Ease: 60以上
- [ ] 平均文長: 20語以下
- [ ] 図表品質: 全て最適化済み

---

## Phase 7: 学術レビュー第2サイクル（90点ゲート）

**参加Subagents**: Academic Reviewer, Scholar

**目的**: 全修正後の最終学術検証

### Phase 3との差分
- Phase 3-6の全修正反映確認
- 新たに追加されたコンテンツの検証
- 全体整合性の確認

### 厳格化された基準
```
Phase 3: 初稿レビュー（合格基準: 80点）
Phase 7: 最終レビュー（合格基準: 90点）
```

### 判定基準
- **Score ≥ 90**: Phase 8へ進む
- **Score 85-89**: 軽微な修正後Phase 8へ
- **Score < 85**: Phase 4へ戻り再修正（最大2サイクル）

### 品質ゲート
- [ ] Academic Review Score ≥ 90
- [ ] 全Critical Issues解決

---

## Phase 8: 統合品質保証

**参加Subagents**: 全Subagents

**目的**: 全Subagent総動員での最終チェック

### 8.1 並行最終チェック

#### Scholar Agent: 最新性確認
- [ ] Phase 0以降の新規論文確認
- [ ] 内容の古さがないか最終チェック
- [ ] 参考文献リストの最終更新

#### Content Agent: 統一性確認
- [ ] 用語の一貫性（表記ゆれチェック）
- [ ] トーン・スタイルの統一
- [ ] 文末表現の統一
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
```

**目標**:
- Performance: 95以上
- Accessibility: 100

#### Maintenance Agent: 技術的品質保証
- [ ] 全リンク切れチェック（自動）
- [ ] 画像ファイルサイズ最適化
- [ ] メタデータ完全性（title, description, OGP）

#### Academic Reviewer: 最終署名
- [ ] 全Subagentsのチェック結果レビュー
- [ ] 残存リスクの評価
- [ ] 公開承認の最終判断
- [ ] 品質保証証明書発行

### 品質ゲート
- [ ] 全Subagents承認
- [ ] 総合スコア: 90以上
- [ ] Lighthouseスコア: Performance 95+, Accessibility 100

---

## Phase 9: 最終承認・公開

**参加Subagents**: Maintenance Agent

**目的**: 最終確認と公開実行

### 9.1 公開実行

```bash
# Git commit
git add content/[topic].md
git add notebooks/[topic].ipynb
git add data/tutorials.json

git commit -m "$(cat <<'EOF'
Add comprehensive guide: [Title]

- Target audience: [Level]
- Quality score: XX/100 (7 subagents review)
- Includes: Theory, implementation, exercises
- Verified: 4 environments tested
- Review cycles: N rounds

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Scholar Agent <agent@ai-terakoya>
Co-Authored-By: Content Agent <agent@ai-terakoya>
Co-Authored-By: Academic Reviewer Agent <agent@ai-terakoya>
Co-Authored-By: Tutor Agent <agent@ai-terakoya>
Co-Authored-By: Data Agent <agent@ai-terakoya>
Co-Authored-By: Design Agent <agent@ai-terakoya>
Co-Authored-By: Maintenance Agent <agent@ai-terakoya>
EOF
)"

# Push to GitHub Pages
git push origin main
```

### 9.2 公開後モニタリング（Maintenance Agent）

**初期24時間モニタリング**:
- [ ] リンク切れ発生: なし
- [ ] ページ読み込み時間: < 2秒
- [ ] エラー発生: なし
- [ ] モバイル表示: 正常

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
| Phase 8 | 全Subagents承認、総合90+ | 該当Phaseへ戻る |
| Phase 9 | 最終確認完了 | 該当Phaseへ戻る |

### レビューサイクルの制限

```
Phase 2 ⇄ Phase 3: 最大3サイクル
Phase 4 ⇄ Phase 7: 最大2サイクル

制限超過の場合:
→ トピック再検討（Phase 0へ戻る）
→ または大幅修正
```

---

## 期待される品質レベル

### 学術的品質
- 論文引用: 15-20件（最新5年以内を70%以上）
- 事実誤認: ゼロ
- 数式エラー: ゼロ
- 用語不統一: ゼロ

### 教育的品質
- 学習目標達成率: 90%以上
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

## クイックリファレンス

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

**Document Version**: 2.0（汎用化版）
**Last Updated**: 2025-10-17
**Maintained By**: AI寺子屋開発チーム

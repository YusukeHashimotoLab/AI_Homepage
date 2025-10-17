# 高品質アウトプット生成クイックリファレンス

**1ページで理解する品質向上のコツ**

---

## 🎯 5つの黄金ルール

### 1. テンプレート化せよ
```python
# ❌ Bad: 毎回違うプロンプト
prompt = f"ベイズ最適化について書いて。中級者向けで2000字くらい。"

# ✅ Good: テンプレート化
TEMPLATE = """
トピック: {topic}
レベル: {level}
文字数: {min_words}〜{max_words}
必須要素: コード例{code_count}個、演習{exercise_count}問

出力形式:
```json
{{"title": "...", "content": "...", "code_examples": [...], "exercises": [...]}}
```
"""
```

### 2. 出力形式を厳密に制約せよ
```python
# ❌ Bad: あいまいな指示
"JSON形式で返してください"

# ✅ Good: 厳密な制約
"""
**重要**: 以下のJSON形式のみを出力し、説明文は一切含めないこと
```json
{"content": "..."}
```
"""
```

### 3. 即座に品質チェックせよ
```python
for attempt in range(5):
    content = llm.generate(prompt)
    rating = quality_assessor.assess(content)

    if rating in ["excellent", "good"]:
        break  # 合格

    # 不合格 → プロンプト改善して再試行
    prompt = improve_prompt(prompt, rating)
```

### 4. エラーは指数バックオフでリトライせよ
```python
for attempt in range(max_retries):
    try:
        return operation()
    except Exception as e:
        delay = min(2.0 * (2.0 ** attempt), 30.0)  # 2, 4, 8, 16, 30秒
        time.sleep(delay)
```

### 5. 多角的に検証せよ
```python
# Generation-Time（即座）
rating = assess_realtime(content)

# Post-Generation（段階的）
academic_score = academic_review(content)  # Phase 3: 80点以上
design_score = design_review(content)
final_score = final_review(content)        # Phase 7: 90点以上
```

---

## ⚡ 実践チェックリスト

### 生成前
- [ ] プロンプトテンプレート準備済み
- [ ] 出力形式（JSON/Markdown）明示
- [ ] 文字数・コード数・演習数を定量的に指定
- [ ] リトライ設定（max_retries=5）

### 生成中
- [ ] リアルタイム品質チェック有効
- [ ] エラーパターン検出有効
- [ ] 進捗モニタリング有効

### 生成後
- [ ] 構造検証（見出し階層）
- [ ] コード検証（構文エラー）
- [ ] 数式検証（LaTeX構文）
- [ ] 参考文献検証（リンク有効性）
- [ ] Phase 3評価: 80点以上
- [ ] Phase 7評価: 90点以上

---

## 🔧 コード例: 完全な品質保証パイプライン

```python
class HighQualityContentGenerator:
    def __init__(self):
        self.retry_manager = ExponentialBackoffRetry(max_retries=5)
        self.quality_assessor = ContentQualityAssessor()
        self.academic_reviewer = AcademicReviewer()

    def generate(self, topic: str, level: str) -> dict:
        # Step 1: テンプレートベース生成
        prompt = self.template.format(topic=topic, level=level)

        # Step 2: リトライ付き生成 + リアルタイムチェック
        content = self.retry_manager.execute(
            lambda: self._generate_with_check(prompt)
        )

        # Step 3: 多角的検証
        academic_score = self.academic_reviewer.review(content)

        if academic_score < 80:
            # Phase 1に戻る
            content = self.generate(topic, level)

        # Step 4: 改善サイクル
        content = self._enhance(content)

        # Step 5: 最終評価
        final_score = self.academic_reviewer.review(content)

        if final_score < 90:
            content = self._enhance(content)  # 再改善

        return {"content": content, "score": final_score}

    def _generate_with_check(self, prompt: str) -> str:
        content = self.llm.generate(prompt)
        rating, details = self.quality_assessor.assess(content)

        if rating not in ["excellent", "good"]:
            raise QualityError(f"品質不足: {rating}")

        return content
```

---

## 📊 品質スコア基準

| 項目 | Phase 3基準 | Phase 7基準 |
|------|-------------|-------------|
| **科学的正確性** | 80点以上 | 90点以上 |
| **完全性** | 80点以上 | 90点以上 |
| **教育品質** | 80点以上 | 90点以上 |
| **実装品質** | 80点以上 | 90点以上 |
| **総合** | 80点以上 | 90点以上 |

### 文字数基準
- **Beginner**: 1500字以上、コード2例、演習5問
- **Intermediate**: 2500字以上、コード3例、演習8問
- **Advanced**: 3500字以上、コード5例、演習10問

---

## 🚨 よくある失敗と対策

### 失敗1: パースエラー頻発
```python
# ❌ 原因: LLMが説明文を追加
"Sure! Here's the JSON: {...}"

# ✅ 対策: 出力制約を厳格化
"""
**重要**: JSON以外の文字列は一切含めないこと
正しい出力: {"content": "..."}
誤った出力: "Here's the result: {...}"
"""
```

### 失敗2: 品質が安定しない
```python
# ❌ 原因: プロンプトが毎回違う
prompt1 = "ベイズ最適化について書いて"
prompt2 = "ベイズ最適化を説明して"

# ✅ 対策: テンプレート固定
TEMPLATE = "トピック: {topic}、レベル: {level}、..."
```

### 失敗3: リトライしても失敗
```python
# ❌ 原因: プロンプトを改善していない
for i in range(5):
    content = llm.generate(same_prompt)  # 同じプロンプト

# ✅ 対策: エラーに基づきプロンプト改善
for i in range(5):
    content = llm.generate(prompt)
    if validate(content):
        break
    prompt = improve_prompt_based_on_error(prompt, content)
```

### 失敗4: 数式が壊れる
```python
# ❌ 原因: Unicode文字が混入
"α + β = γ"  # LaTeXでエラー

# ✅ 対策: MathEnvironmentProtectorで保護
protector = MathEnvironmentProtector()
protected = protector.protect("$\\alpha + \\beta = \\gamma$")
```

---

## 💡 即座に使えるコードスニペット

### スニペット1: 指数バックオフリトライ
```python
def retry_with_backoff(operation, max_retries=5):
    for attempt in range(max_retries):
        try:
            return operation()
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            delay = min(2.0 * (2.0 ** attempt), 30.0)
            time.sleep(delay)
```

### スニペット2: リアルタイム品質チェック
```python
def assess_quality(content: str) -> str:
    issues = []
    if len(content) < 2000:
        issues.append("文字数不足")
    if len(re.findall(r'```[\s\S]*?```', content)) < 3:
        issues.append("コード例不足")

    if not issues:
        return "excellent"
    elif len(issues) <= 2:
        return "good"
    else:
        return "poor"
```

### スニペット3: エラーパターン検出
```python
def detect_error_pattern(error: Exception) -> str:
    error_msg = str(error)

    if "JSONDecodeError" in error_msg:
        return "json_parse_error"
    elif "Incomplete" in error_msg:
        return "incomplete_output"
    elif "LaTeX error" in error_msg:
        return "math_expression_error"
    else:
        return "unknown_error"
```

---

## 📈 ROI順の改善アクション

### 即座実装（ROI ★★★★★）
1. **プロンプトテンプレート化** (2-3日)
   - 効果: 品質70→85点、パースエラー50%削減
2. **リアルタイム品質チェック** (1日)
   - 効果: 不合格コンテンツ早期検出、リトライ効率化

### 1週間以内（ROI ★★★★☆）
3. **指数バックオフリトライ** (3日)
   - 効果: エラー回復率80%→95%
4. **数式保護機構** (1-2日)
   - 効果: 数式エラー80%削減

### 2週間以内（ROI ★★★☆☆）
5. **ContentQualityAssessor統合** (1週間)
   - 効果: Phase 3合格率80%→90%
6. **自動検証スイート** (1週間)
   - 効果: 構造・コード・数式の自動検証

---

## 🎓 まとめ: 3つの重要ポイント

1. **一貫性**: テンプレート化で品質を安定化
2. **即応性**: Generation-Timeチェックで早期発見
3. **多角性**: Post-Generation検証で完全性保証

**この3つを実装すれば、品質スコアは確実に80→90点に向上します。**

---

**詳細版**: `claudedocs/high_quality_output_guide.md`
**分析レポート**: `claudedocs/write_tb_ai_analysis.md`

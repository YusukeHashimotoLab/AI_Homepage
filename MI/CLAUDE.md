# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Project Overview

This is the **MI Knowledge Hub** - a comprehensive Materials Informatics educational and community platform powered by **Claude Code subagents**. The project features AI-driven content generation through 7 specialized subagents, automatic paper collection, and interactive learning support.

**Key Characteristics:**
- 7 specialized Claude Code subagents collaborating on content creation
- Quality-first approach with academic review gates (80-point and 90-point thresholds)
- Static site on GitHub Pages with JSON/Markdown data layer
- Mobile-first responsive design with WCAG 2.1 Level AA accessibility
- **NO API keys required** - everything runs within Claude Code sessions

---

## Architecture: Claude Code Subagent-Based

### Core Philosophy

This project leverages **Claude Code subagents** instead of external API clients. All content generation, paper collection, and quality assurance happen within Claude Code sessions using the built-in tools (Read, Write, Edit, Bash, Grep, WebSearch).

**Benefits:**
- ✅ No API keys or external services required
- ✅ All agents run within secure Claude Code environment
- ✅ Direct access to file system and tools
- ✅ Version controlled agent definitions (.claude/agents/)
- ✅ Seamless collaboration between agents
- ✅ No rate limiting or API costs

### 7 Specialized Subagents

Located in `.claude/agents/`, each agent is a Markdown file with YAML frontmatter defining its role, tools, and behavior.

| Agent | File | Primary Role | Key Tools |
|-------|------|--------------|-----------|
| **Scholar Agent** | `scholar-agent.md` | Paper collection & summarization | Read, Write, Edit, Bash, WebSearch |
| **Content Agent** | `content-agent.md` | Article generation (9-phase workflow) | Read, Write, Edit, MultiEdit, Bash |
| **Academic Reviewer** | `academic-reviewer.md` | Quality assurance (0-100 scoring) | Read, Write, Edit, Grep, Bash |
| **Tutor Agent** | `tutor-agent.md` | Interactive learning support | Read, Write, Grep, Bash |
| **Data Agent** | `data-agent.md` | Dataset & tool management | Read, Write, Edit, Bash, WebSearch |
| **Design Agent** | `design-agent.md` | UX optimization & accessibility | Read, Write, Edit, Grep, Bash |
| **Maintenance Agent** | `maintenance-agent.md` | Validation & monitoring | Read, Write, Bash, Grep, Glob |

---

## Common Workflows

### 1. Collecting Recent Papers

```
User: "Use scholar-agent to collect recent papers on bayesian optimization for materials"

scholar-agent:
1. Searches using WebSearch
2. Extracts metadata (DOI, authors, abstract)
3. Checks data/papers.json for duplicates
4. Adds new papers with proper structure
5. Reports: "Added 5 new papers, skipped 2 duplicates"
```

### 2. Creating Educational Content (9-Phase Workflow)

```
User: "Use content-agent to create an intermediate-level article about Bayesian optimization"

Phase 0-2: content-agent drafts initial article
  ↓
Phase 3: academic-reviewer reviews (must score ≥80)
  → If fail: Return to Phase 1
  → If pass: Continue
  ↓
Phase 4-6: content-agent enhances with help from design-agent, data-agent
  ↓
Phase 7: academic-reviewer reviews again (must score ≥90)
  → If 80-89: Minor revision (return to Phase 4)
  → If <80: Major revision (return to Phase 1)
  → If ≥90: Approved
  ↓
Phase 8-9: Final checks and publication to content/methods/
```

### 3. Interactive Learning Support

```
User: "Ask tutor-agent to explain Bayesian optimization"

tutor-agent:
Uses Socratic dialogue:
"Before I explain, let me ask: もし実験のコストが非常に高い場合、
どのように次の実験条件を選びますか？"

[Guides through discovery learning with hints and questions]
```

### 4. Data Validation & Maintenance

```
User: "Use maintenance-agent to validate all data"

maintenance-agent:
1. Validates JSON structure (papers, datasets, tutorials, tools)
2. Checks all URLs for accessibility
3. Runs quality metrics (Lighthouse scores)
4. Generates health report
5. Reports issues with priority levels
```

---

## Directory Structure

```
MI/
├── .claude/
│   └── agents/              # 7 subagent definitions (Markdown + YAML)
│       ├── scholar-agent.md
│       ├── content-agent.md
│       ├── academic-reviewer.md
│       ├── tutor-agent.md
│       ├── data-agent.md
│       ├── design-agent.md
│       └── maintenance-agent.md
├── assets/                  # CSS/JS/Images
├── content/                 # Markdown articles
│   ├── basics/              # Beginner content
│   ├── methods/             # Intermediate methods
│   ├── advanced/            # Advanced topics
│   └── applications/        # Case studies
├── data/                    # JSON data files
│   ├── papers.json
│   ├── datasets.json
│   ├── tutorials.json
│   └── tools.json
├── pages/                   # Static HTML pages
├── notebooks/               # Jupyter tutorials
├── tools/                   # Validation utilities (Python)
│   ├── validate_data.py     # JSON validation script
│   └── __init__.py
├── reviews/                 # Academic review reports
├── claudedocs/              # Project documentation
│   ├── requirements.md
│   └── content-creation-procedure.md
├── index.html               # Homepage
├── CLAUDE.md                # This file
└── requirements.txt         # Minimal dependencies
```

---

## Development Setup

### Prerequisites

**Required:**
- Python 3.11+ (for validation utilities and Jupyter notebooks)
- Claude Code (for subagent execution)

**Optional:**
- Jupyter (for tutorial notebooks)

### Installation

```bash
# 1. Setup Python environment
python -m venv .venv
source .venv/bin/activate  # macOS/Linux

# 2. Install minimal dependencies
pip install -r requirements.txt

# 3. No API keys needed! Everything runs in Claude Code
```

### Local Preview

```bash
# Start local server
python -m http.server 8000

# Visit: http://localhost:8000
```

---

## Content Creation Process

### Using Subagents in Claude Code

**Explicit Invocation:**
```
"Use content-agent to create an article on ベイズ最適化"
"Ask scholar-agent to collect papers from the last 7 days"
"Have academic-reviewer review content/methods/bayesian_optimization.md"
```

**Automatic Delegation:**
Claude Code will automatically delegate to appropriate subagents based on context.

### Quality Assurance: Academic Review Gates

The **academic-reviewer** subagent enforces quality through scoring (0-100):

**Phase 3 Gate (After Initial Draft):**
- Minimum required: 80/100
- Dimensions: Scientific accuracy, clarity, references, accessibility
- Fail → Major revision (return to Phase 1)

**Phase 7 Gate (After Enhancement):**
- Minimum required: 90/100
- All dimensions must be excellent
- Fail 80-89 → Minor revision (return to Phase 4)
- Fail <80 → Major revision (return to Phase 1)

**Review Report Structure:**
- Detailed scores for each dimension
- Specific issues with line numbers
- Prioritized improvement recommendations
- Clear decision: APPROVE / MINOR_REVISION / MAJOR_REVISION

---

## Data Management

### JSON Data Schemas

**papers.json:**
```json
{
  "id": "paper_001",
  "title": "Paper title",
  "authors": ["Author A", "Author B"],
  "year": 2024,
  "journal": "Journal Name",
  "doi": "10.xxxx/xxxxx",
  "abstract": "Abstract text",
  "tags": ["tag1", "tag2"],
  "collected_at": "2025-10-15T12:00:00Z"
}
```

**datasets.json, tutorials.json, tools.json:**
See data-agent documentation in `.claude/agents/data-agent.md` for complete schemas.

### Data Validation

```bash
# Validate all JSON files
python tools/validate_data.py

# Output:
# ✅ papers.json (153 entries, no errors)
# ✅ datasets.json (48 entries, no errors)
# ⚠️ tutorials.json (2 missing fields)
# ✅ tools.json (67 entries, no errors)
```

---

## Quality Standards

### Performance Targets
- **Lighthouse Performance**: ≥95
- **Lighthouse Accessibility**: 100
- **First Contentful Paint**: <1.5s
- **Time to Interactive**: <3.0s

### Accessibility Requirements
- WCAG 2.1 Level AA compliance
- Color contrast ratio ≥4.5:1
- Touch targets ≥44px × 44px (Apple HIG)
- Keyboard navigation support
- Screen reader compatibility

### Academic Content Quality
- Scientific accuracy verified by academic-reviewer
- All dimensions scored ≥90 at publication
- Peer-reviewed references preferred
- Executable code examples
- Clear learning objectives

---

## Common Tasks

### Collecting Papers

```
User: "Use scholar-agent to collect papers on 'materials informatics machine learning' from the last 30 days"

scholar-agent will:
1. Search using WebSearch
2. Extract and validate metadata
3. Update data/papers.json
4. Report results
```

### Creating Content

```
User: "Use content-agent to create a beginner article about MI basics"

Process:
1. content-agent drafts (Phase 0-2)
2. academic-reviewer scores (Phase 3, must be ≥80)
3. content-agent enhances (Phase 4-6)
4. academic-reviewer scores again (Phase 7, must be ≥90)
5. Published to content/basics/
```

### Updating Datasets

```
User: "Use data-agent to add a new dataset about crystal structures"

data-agent will:
1. Prompt for complete metadata
2. Validate required fields
3. Check for duplicates
4. Add to data/datasets.json
5. Verify JSON integrity
```

### Validating System Health

```
User: "Use maintenance-agent to validate all data and check links"

maintenance-agent will:
1. Validate JSON structure
2. Check all URLs
3. Run quality metrics
4. Generate health report
5. Prioritize issues
```

---

## Collaboration Patterns

### Scholar + Content Agent
```
Content creation needs recent research context
  → scholar-agent collects papers
  → content-agent references in article
```

### Content + Academic Reviewer Agent
```
Article draft completed
  → academic-reviewer scores (Phase 3)
  → content-agent revises based on feedback
  → academic-reviewer scores again (Phase 7)
```

### Content + Design Agent
```
Article structure complete
  → design-agent adds diagrams, improves formatting
  → design-agent checks accessibility
  → content-agent incorporates improvements
```

### Data + Maintenance Agent
```
New data added
  → maintenance-agent validates structure
  → maintenance-agent checks URLs
  → data-agent fixes reported issues
```

---

## Troubleshooting

### Subagent Not Found

If a subagent is not recognized:
```bash
# Check that agent file exists
ls .claude/agents/

# Verify YAML frontmatter
head .claude/agents/scholar-agent.md
```

### Validation Failures

```bash
# Run validation script
python tools/validate_data.py

# Fix reported issues in JSON files
# Re-validate until all pass
```

### Content Below Quality Threshold

If academic-reviewer rejects content:
1. Read the review report in reviews/
2. Address all HIGH priority issues
3. Incorporate recommendations
4. Re-submit for review

---

## Git Workflow

```bash
# Always work on feature branches
git checkout -b feature/add-quantum-ml-article

# Create content using subagents
# (All changes tracked automatically)

# Validate before committing
python tools/validate_data.py

# Commit with descriptive message
git commit -m "Add quantum ML article (academic-reviewer score: 92)"

# Push and create PR
git push origin feature/add-quantum-ml-article
```

---

## Key Differences from Original Design

### What Changed

**Before (Python API-based):**
- Required ANTHROPIC_API_KEY
- Python agents with anthropic library
- External API calls for each generation
- Complex async/await patterns
- Rate limiting concerns

**Now (Claude Code Subagent-based):**
- No API keys required
- Subagents run within Claude Code
- Direct tool access (Read, Write, Edit, Bash)
- Synchronous, session-based execution
- No rate limits within session

### What Stayed the Same

- 9-phase quality workflow
- Academic review gates (80/90 points)
- Static HTML/CSS/JS frontend
- JSON data layer
- Mobile-first responsive design
- WCAG 2.1 Level AA accessibility

---

## Documentation Reference

**Essential reading:**

1. **`claudedocs/requirements.md`** - Comprehensive requirements (v1.1)
2. **`claudedocs/content-creation-procedure.md`** - 9-phase quality workflow
3. **`.claude/agents/*.md`** - Individual subagent documentation
4. **`README.md`** - Quick start guide

---

## Contact

**Project Lead**: Dr. Yusuke Hashimoto
**Email**: yusuke.hashimoto.b8@tohoku.ac.jp
**Institution**: Tohoku University

---

**Last Updated**: 2025-10-15
**CLAUDE.md Version**: 2.0 (Subagent Architecture)

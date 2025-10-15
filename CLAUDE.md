# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Project Overview

This is a **research laboratory homepage** project for Dr. Yusuke Hashimoto (Tohoku University). The project is being rebuilt from scratch to solve iPhone compatibility issues and implement AI-powered content management.

**Key Goals:**
- Complete iPhone/mobile compatibility (fixing tap/navigation issues)
- AI-driven content updates via natural language chat interface
- Minimalist academic design with white space emphasis
- Static site hosted on GitHub Pages with data-driven rendering

---

## Before You Start

**IMPORTANT: Always refer to the detailed design documentation in `/claudedocs/` before implementing:**

- **`claudedocs/requirements.md`** - Read this first to understand:
  - Project goals and success criteria
  - Target users and their needs
  - All required features and pages
  - Non-functional requirements (performance, accessibility, SEO)
  - Technical stack and constraints

- **`claudedocs/technical-design.md`** - Reference for implementation details:
  - System architecture and data flow
  - Directory structure and file organization
  - AI content management system design
  - Security considerations and deployment strategy
  - Performance optimization techniques

- **`claudedocs/design-system.md`** - Essential for all UI work:
  - Complete color palette and design tokens
  - Typography scale and font stacks
  - Spacing system (8px grid)
  - Component specifications (buttons, cards, forms)
  - Responsive breakpoints and layout patterns
  - Accessibility requirements

**When to consult these documents:**
- Before starting any new feature implementation
- When making design decisions (colors, spacing, typography)
- When implementing responsive layouts
- When adding new components
- When troubleshooting design or architecture issues
- When onboarding to the project

---

## Architecture

### Two-Part System

1. **Static Website** (HTML/CSS/JavaScript)
   - Mobile-first responsive design
   - Data-driven rendering from JSON files
   - Bilingual support (Japanese/English)
   - Deployed to GitHub Pages

2. **AI Content Management Tool** (Python)
   - Natural language chat interface powered by Claude API
   - Automated paper fetching from Google Scholar/ORCID
   - JSON data file management
   - Runs locally, updates pushed via Git

### Data Flow

```
User command (natural language)
  ↓
Claude API interprets intent
  ↓
Python tool executes action
  ├→ Fetch papers from Scholar
  ├→ Add news to news.json
  ├→ Add members to members.json
  └→ Update HTML/JSON files
  ↓
Git commit & push
  ↓
GitHub Pages auto-deploy
```

---

## Directory Structure

```
AI_Homepage/
├── index.html              # Root redirect to jp/
├── en/                     # English pages
│   ├── index.html
│   ├── research.html
│   ├── publications.html
│   ├── news.html
│   ├── members.html
│   ├── talks.html
│   ├── contact.html
│   └── links.html
├── jp/                     # Japanese pages (same structure as en/)
├── assets/
│   ├── css/
│   │   ├── variables.css   # Design tokens (colors, spacing)
│   │   ├── reset.css
│   │   ├── base.css
│   │   ├── components.css
│   │   ├── layout.css
│   │   └── responsive.css
│   ├── js/
│   │   ├── main.js
│   │   ├── navigation.js
│   │   └── data-loader.js  # Fetches and renders JSON data
│   └── images/
├── data/                   # JSON data files
│   ├── papers.json
│   ├── news.json
│   ├── members.json
│   ├── talks.json
│   └── profile.json
├── tools/                  # AI content management
│   ├── ai_chat.py          # Main chat interface
│   ├── data_manager.py     # JSON CRUD operations
│   ├── paper_fetcher.py    # Scholar/ORCID integration
│   ├── config.py
│   └── requirements.txt
└── claudedocs/             # Design documentation
    ├── requirements.md
    ├── technical-design.md
    └── design-system.md
```

---

## Development Commands

### Python Environment

```bash
# Activate virtual environment
source .venv/bin/activate  # macOS/Linux
.venv\Scripts\activate     # Windows

# Install dependencies
pip install -r tools/requirements.txt
```

### AI Content Management

```bash
# Run interactive chat interface
cd tools
python ai_chat.py

# Example interactions:
# "ニュースを追加して。タイトルは「新論文発表」"
# "最新の論文情報を取得して"
# "メンバーに田中太郎を追加。博士課程、研究は機械学習"
```

### Testing

```bash
# Start local server for testing
python -m http.server 8000
# Visit: http://localhost:8000

# Mobile testing with Chrome DevTools
# F12 → Toggle device toolbar (Ctrl+Shift+M)
# Select iPhone SE or iPhone 12

# Check Lighthouse scores
# Chrome DevTools → Lighthouse tab
# Target: Performance 95+, Accessibility 100
```

### Deployment

```bash
# Manual deployment (Git push triggers auto-deploy)
git add .
git commit -m "Update content"
git push origin main

# GitHub Pages automatically deploys from main branch
```

---

## Design System

### CSS Architecture

**Mobile-First Approach:**
- Default styles target mobile (0-768px)
- Progressive enhancement via media queries
- Minimum tap target: 44px × 44px (Apple HIG)

**CSS Variables Pattern:**
```css
/* variables.css defines all design tokens */
--color-primary-900: #1a252f;
--color-primary-700: #2c3e50;
--spacing-4: 1rem;  /* 16px */
--font-size-base: 1rem;  /* 16px */
```

**Breakpoints:**
```css
@media (min-width: 768px) { /* Tablet */ }
@media (min-width: 1024px) { /* Desktop */ }
@media (min-width: 1440px) { /* Wide */ }
```

### iPhone Compatibility Critical Points

1. **Hover Effects:**
   ```css
   /* ONLY apply hover on devices with hover capability */
   @media (hover: hover) and (pointer: fine) {
     nav a:hover { background: #4a5f7a; }
   }
   ```

2. **Touch Targets:**
   - All clickable elements ≥ 44px × 44px
   - Adequate spacing between tap targets (8px minimum)

3. **Navigation:**
   - Desktop: Header navigation
   - Mobile: Bottom navigation (fixed, 4 main items)
   - Both use `min-height: 44px` for accessibility

### Data-Driven Rendering

JavaScript dynamically loads JSON data and renders HTML:

```javascript
// data-loader.js pattern
async function loadPapers() {
  const response = await fetch('../data/papers.json');
  const papers = await response.json();

  const container = document.getElementById('papers-list');
  papers.forEach(paper => {
    container.appendChild(createPaperElement(paper));
  });
}
```

**When adding new content:**
1. Update JSON file (via AI tool or manually)
2. Reload page (JavaScript renders automatically)
3. No HTML editing required

---

## AI Content Management System

### Core Components

**AIContentManager (`ai_chat.py`):**
- Accepts natural language commands
- Uses Claude API to parse intent into structured actions
- Routes to appropriate data management functions

**DataManager (`data_manager.py`):**
- CRUD operations for JSON files
- Handles news, members, papers, talks
- Atomic write operations with UTF-8 encoding

**PaperFetcher (`paper_fetcher.py`):**
- Google Scholar API integration
- Fetches publication metadata (title, authors, citations)
- Updates `data/papers.json`

### Environment Setup

Required `.env` file in `tools/`:
```
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...  # Optional alternative
```

**Security:** Never commit `.env` to Git (already in `.gitignore`)

### Adding New AI Actions

To extend the AI tool with new capabilities:

1. **Define action in `ai_chat.py`:**
   ```python
   elif intent['action'] == 'your_new_action':
       return self._your_new_method(intent['params'])
   ```

2. **Implement in `data_manager.py`:**
   ```python
   def your_new_method(self, ...):
       data = self.load_data('filename.json')
       # Modify data
       self.save_data('filename.json', data)
       return "Success message"
   ```

3. **Update intent parsing prompt** to include new action type

---

## Content Migration from OutReach

The old site is at `/Users/yusukehashimoto/Documents/pycharm/OutReach`.

**Migration Process:**
1. Extract content from old HTML files
2. Convert to JSON format (`data/*.json`)
3. Copy images to `assets/images/`
4. Verify rendering in new data-driven templates

**Priority Migration:**
- Profile information → `data/profile.json`
- Publications → `data/papers.json`
- News → `data/news.json`
- Members → `data/members.json`

---

## Testing Requirements

### Mobile Testing Checklist

**iPhone Safari (Primary Target):**
- [ ] All links/buttons respond to single tap
- [ ] Navigation works without double-tap
- [ ] Bottom nav visible and functional
- [ ] No horizontal scroll
- [ ] Forms submit correctly

**Performance Targets:**
- [ ] Lighthouse Performance: 95+
- [ ] Lighthouse Accessibility: 100
- [ ] Page load < 2 seconds (4G mobile)

**Accessibility:**
- [ ] Keyboard navigation (Tab, Enter, Space)
- [ ] Screen reader compatible (aria-labels)
- [ ] Color contrast ≥ 4.5:1 (WCAG AA)
- [ ] Skip-to-content link functional

---

## Common Workflows

### Adding News Article

**Via AI Tool:**
```bash
cd tools
python ai_chat.py
> ニュースを追加。タイトル「研究成果発表」、内容「Nature Materialsに論文掲載」
```

**Manual:**
```bash
# Edit data/news.json
{
  "id": "news001",
  "date": "2025-10-14",
  "title": "研究成果発表",
  "content": "Nature Materialsに論文が掲載されました"
}
```

### Updating Publications

**Via AI Tool:**
```bash
python ai_chat.py
> 最新の論文情報を取得して
# Fetches from Google Scholar, updates papers.json
```

**Manual:**
```bash
# Edit data/papers.json with publication metadata
```

### Creating New Page

1. Copy template from existing page (e.g., `jp/research.html`)
2. Update page-specific content
3. Add navigation link in header
4. Test mobile/desktop layouts
5. Verify data-loader.js integration if using JSON data

---

## Code Style & Patterns

### HTML

- Semantic HTML5 elements (`<nav>`, `<main>`, `<article>`)
- ARIA attributes for accessibility
- Language attributes on `<html>` tag
- Meta viewport with `viewport-fit=cover` for iPhone notch

### CSS

- CSS Variables for all design tokens
- Mobile-first media queries
- `@media (hover: hover)` for hover effects
- BEM naming convention for components

### JavaScript

- Vanilla ES6+ (no frameworks)
- Async/await for data loading
- Error handling for fetch failures
- Progressive enhancement (works without JS for critical content)

### Python

- Type hints for function signatures
- Docstrings for all public methods
- UTF-8 encoding for JSON files
- Environment variables for API keys

---

## Critical Implementation Notes

### iPhone Tap Issues (Top Priority)

The original site had double-tap requirements on iPhone. **Root cause:** CSS hover states without hover capability detection.

**Solution (already implemented in design):**
```css
/* ❌ WRONG - causes double-tap on iPhone */
nav a:hover { background: #34495e; }

/* ✅ CORRECT - only on hover-capable devices */
@media (hover: hover) and (pointer: fine) {
  nav a:hover { background: #34495e; }
}
```

Apply this pattern to ALL interactive elements (links, buttons, cards).

### Data File Integrity

**All JSON files must:**
- Use UTF-8 encoding (ensure_ascii=False in Python)
- Include `id` field for each item
- Maintain consistent structure
- Be validated before committing

**Backup strategy:**
- Git tracks all JSON changes
- Test data modifications locally before pushing
- Keep old data structure during migrations

### Performance Optimization

**Critical CSS:**
- Inline critical CSS in `<head>` for above-fold content
- Load non-critical CSS with `<link rel="stylesheet">`

**Images:**
- Use WebP with JPEG fallback
- Lazy loading: `loading="lazy"` attribute
- Responsive images with `<picture>` element

**Fonts:**
- System fonts first (instant rendering)
- Web fonts with `font-display: swap`

---

## Documentation Reference

Full design specifications in:
- `claudedocs/requirements.md` - Feature requirements and scope
- `claudedocs/technical-design.md` - Architecture and implementation details
- `claudedocs/design-system.md` - Visual design tokens and patterns

---

## Troubleshooting

### AI Tool Not Working

1. Check `.env` file exists in `tools/` with valid API key
2. Verify Python dependencies: `pip install -r tools/requirements.txt`
3. Check network connection for API calls
4. Review error messages in terminal

### Data Not Displaying

1. Open browser console (F12) for JavaScript errors
2. Verify JSON file path in `data-loader.js`
3. Validate JSON syntax: `python -m json.tool data/papers.json`
4. Check fetch() CORS issues with local server

### Mobile Layout Issues

1. Test in Chrome DevTools device mode first
2. Check viewport meta tag in HTML
3. Verify media query breakpoints match design system
4. Test on real iPhone (Safari and Chrome) for final validation

---

## Project Status

**Current Phase:** Requirements and design complete, ready for implementation

**Next Steps:**
1. Create project structure and base files
2. Implement mobile-first HTML/CSS
3. Build data-driven JavaScript rendering
4. Migrate content from OutReach
5. Implement AI content management tool
6. Test on iPhone devices
7. Deploy to GitHub Pages

---

## Task Completion Checklist

**Before marking any task as complete, verify ALL applicable items:**

### General Requirements (All Tasks)
- [ ] Code follows project style guidelines (see Code Style & Patterns section)
- [ ] Documentation updated if adding new features or changing behavior
- [ ] No hardcoded values (use CSS variables for design tokens)
- [ ] Console has no errors or warnings
- [ ] Git commit message is descriptive

### Frontend Tasks (HTML/CSS/JS)
- [ ] Mobile-first implementation verified
- [ ] Tested on iPhone Safari (primary target)
- [ ] Tested on desktop browsers (Chrome, Firefox, Safari)
- [ ] All hover effects use `@media (hover: hover)`
- [ ] Touch targets ≥ 44px × 44px
- [ ] No horizontal scroll on mobile
- [ ] Keyboard navigation works (Tab, Enter, Space)
- [ ] ARIA attributes added where needed
- [ ] Color contrast meets WCAG AA (4.5:1)
- [ ] Images have alt text
- [ ] Forms have labels
- [ ] Lighthouse scores meet targets (Performance 95+, Accessibility 100)

### CSS Specific
- [ ] CSS variables used from `variables.css`
- [ ] Mobile styles defined first (no media query)
- [ ] Tablet styles in `@media (min-width: 768px)`
- [ ] Desktop styles in `@media (min-width: 1024px)`
- [ ] No duplicate selectors or rules
- [ ] Follows BEM naming convention for components

### JavaScript Specific
- [ ] Uses async/await for asynchronous operations
- [ ] Error handling implemented (try/catch for fetch)
- [ ] No global variables pollution
- [ ] Progressive enhancement (critical content works without JS)
- [ ] JSON data validated before use

### Python/AI Tool Tasks
- [ ] Type hints for function signatures
- [ ] Docstrings for all public methods
- [ ] UTF-8 encoding for file operations (ensure_ascii=False)
- [ ] API keys loaded from environment variables (.env)
- [ ] Error messages are user-friendly
- [ ] JSON output is properly formatted (indent=2)
- [ ] Works with virtual environment dependencies

### Data/Content Tasks
- [ ] JSON files are valid (validate with `python -m json.tool`)
- [ ] All items have unique `id` field
- [ ] UTF-8 encoding preserved
- [ ] No sensitive information in JSON files
- [ ] Backup exists (Git commit before major changes)

### Design Compliance
- [ ] Matches design tokens in `claudedocs/design-system.md`
- [ ] Colors from approved palette only
- [ ] Spacing uses 8px grid system
- [ ] Typography uses defined scale
- [ ] Components follow specifications

### Testing Checklist
- [ ] Works in local test server (`python -m http.server 8000`)
- [ ] Mobile DevTools testing passed
- [ ] Real device testing passed (if available)
- [ ] No console errors in browser
- [ ] Page loads in < 2 seconds (4G simulation)

### Pre-Commit Checklist
- [ ] `.gitignore` excludes sensitive files (.env, .DS_Store, etc.)
- [ ] No API keys or credentials in code
- [ ] No debug console.log statements
- [ ] No commented-out code blocks
- [ ] File paths use correct relative paths
- [ ] Links are not broken

### Documentation Updates Required When:
- Adding new pages → Update this CLAUDE.md and navigation
- Adding new AI actions → Update `tools/` documentation
- Changing data structure → Update JSON schema examples
- Adding dependencies → Update `requirements.txt` or `package.json`
- Changing deployment process → Update deployment section

**After completing all checks, mark the task as done and document any deviations or issues in commit message.**

---

**Contact:** Yusuke Hashimoto - yusuke.hashimoto.b8@tohoku.ac.jp

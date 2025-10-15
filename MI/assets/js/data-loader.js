/**
 * Data Loader JavaScript - MI Knowledge Hub
 * Handles dynamic loading and rendering of JSON data
 */

(function () {
  'use strict';

  const { fetchJSON, formatDate, truncateText, sanitizeHTML, showError, showLoading } =
    window.MIKnowledgeHub.utils;

  // ========================================
  // Paper Rendering
  // ========================================

  /**
   * Create paper card element
   * @param {Object} paper - Paper data
   * @returns {HTMLElement} Paper card element
   */
  function createPaperCard(paper) {
    const article = document.createElement('article');
    article.className = 'paper-card';
    article.setAttribute('role', 'listitem');

    const title = document.createElement('h3');
    title.className = 'paper-title';
    title.textContent = paper.title;

    const meta = document.createElement('div');
    meta.className = 'paper-meta';
    meta.textContent = `${paper.authors.join(', ')} - ${paper.journal} (${paper.year})`;

    const abstract = document.createElement('p');
    abstract.className = 'paper-abstract';
    abstract.textContent = truncateText(paper.abstract || '', 200);

    const tags = document.createElement('div');
    tags.className = 'tags';
    if (paper.tags) {
      paper.tags.forEach((tag) => {
        const tagSpan = document.createElement('span');
        tagSpan.className = 'tag';
        tagSpan.textContent = tag;
        tags.appendChild(tagSpan);
      });
    }

    const link = document.createElement('a');
    link.href = `https://doi.org/${paper.doi}`;
    link.target = '_blank';
    link.rel = 'noopener noreferrer';
    link.className = 'btn btn-text';
    link.textContent = '論文を読む →';

    article.appendChild(title);
    article.appendChild(meta);
    article.appendChild(abstract);
    if (paper.tags && paper.tags.length > 0) {
      article.appendChild(tags);
    }
    article.appendChild(link);

    return article;
  }

  /**
   * Load and render papers
   * @param {string} containerId - Container element ID
   * @param {number} limit - Maximum number of papers to display
   */
  async function loadPapers(containerId, limit = null) {
    const container = document.getElementById(containerId);
    if (!container) return;

    showLoading(container);

    try {
      const papers = await fetchJSON('../data/papers.json');

      if (!papers || papers.length === 0) {
        container.innerHTML = '<p class="loading">論文がありません。</p>';
        return;
      }

      // Sort by year (newest first)
      papers.sort((a, b) => b.year - a.year);

      // Limit results if specified
      const displayPapers = limit ? papers.slice(0, limit) : papers;

      container.innerHTML = '';
      displayPapers.forEach((paper) => {
        container.appendChild(createPaperCard(paper));
      });
    } catch (error) {
      console.error('Error loading papers:', error);
      showError('論文の読み込みに失敗しました。', container);
    }
  }

  // ========================================
  // Dataset Rendering
  // ========================================

  /**
   * Create dataset card element
   * @param {Object} dataset - Dataset data
   * @returns {HTMLElement} Dataset card element
   */
  function createDatasetCard(dataset) {
    const article = document.createElement('article');
    article.className = 'card';

    const title = document.createElement('h3');
    title.className = 'card-title';
    title.textContent = dataset.name;

    const description = document.createElement('p');
    description.className = 'card-description';
    description.textContent = dataset.description;

    const meta = document.createElement('div');
    meta.className = 'paper-meta';
    meta.innerHTML = `
      <strong>データ種類:</strong> ${dataset.data_types.join(', ')}<br>
      <strong>サイズ:</strong> ${dataset.size}<br>
      <strong>ライセンス:</strong> ${dataset.license}
    `;

    const link = document.createElement('a');
    link.href = dataset.url;
    link.target = '_blank';
    link.rel = 'noopener noreferrer';
    link.className = 'btn btn-secondary';
    link.textContent = 'データセットを見る';

    article.appendChild(title);
    article.appendChild(description);
    article.appendChild(meta);
    article.appendChild(link);

    return article;
  }

  /**
   * Load and render datasets
   * @param {string} containerId - Container element ID
   */
  async function loadDatasets(containerId) {
    const container = document.getElementById(containerId);
    if (!container) return;

    showLoading(container);

    try {
      const datasets = await fetchJSON('../data/datasets.json');

      if (!datasets || datasets.length === 0) {
        container.innerHTML = '<p class="loading">データセットがありません。</p>';
        return;
      }

      container.innerHTML = '';
      datasets.forEach((dataset) => {
        container.appendChild(createDatasetCard(dataset));
      });
    } catch (error) {
      console.error('Error loading datasets:', error);
      showError('データセットの読み込みに失敗しました。', container);
    }
  }

  // ========================================
  // Tutorial Rendering
  // ========================================

  /**
   * Create tutorial card element
   * @param {Object} tutorial - Tutorial data
   * @returns {HTMLElement} Tutorial card element
   */
  function createTutorialCard(tutorial) {
    const article = document.createElement('article');
    article.className = 'learning-path-card';

    const level = document.createElement('span');
    level.className = 'learning-path-level';
    level.textContent = tutorial.level;

    const title = document.createElement('h3');
    title.className = 'learning-path-title';
    title.textContent = tutorial.title;

    const description = document.createElement('p');
    description.className = 'learning-path-description';
    description.textContent = tutorial.description;

    const link = document.createElement('a');
    link.href = tutorial.notebook_url;
    link.className = 'btn btn-primary';
    link.textContent = 'チュートリアルを開く';
    link.style.marginTop = 'var(--spacing-4)';

    article.appendChild(level);
    article.appendChild(title);
    article.appendChild(description);
    article.appendChild(link);

    return article;
  }

  /**
   * Load and render tutorials
   * @param {string} containerId - Container element ID
   */
  async function loadTutorials(containerId) {
    const container = document.getElementById(containerId);
    if (!container) return;

    showLoading(container);

    try {
      const tutorials = await fetchJSON('../data/tutorials.json');

      if (!tutorials || tutorials.length === 0) {
        container.innerHTML = '<p class="loading">チュートリアルがありません。</p>';
        return;
      }

      container.innerHTML = '';
      tutorials.forEach((tutorial) => {
        container.appendChild(createTutorialCard(tutorial));
      });
    } catch (error) {
      console.error('Error loading tutorials:', error);
      showError('チュートリアルの読み込みに失敗しました。', container);
    }
  }

  // ========================================
  // Content Article Rendering
  // ========================================

  /**
   * Load and render markdown content articles
   * @param {string} containerId - Container element ID
   * @param {string} category - Content category (basics, methods, advanced, applications)
   */
  async function loadContentArticles(containerId, category) {
    const container = document.getElementById(containerId);
    if (!container) return;

    showLoading(container);

    try {
      // In a static site, we'd need to maintain an index of articles
      // For now, this is a placeholder for the content structure
      const articleIndex = await fetchJSON(`../data/${category}_index.json`);

      if (!articleIndex || articleIndex.length === 0) {
        container.innerHTML = '<p class="loading">コンテンツがありません。</p>';
        return;
      }

      container.innerHTML = '';
      articleIndex.forEach((article) => {
        const articleElement = createArticlePreview(article);
        container.appendChild(articleElement);
      });
    } catch (error) {
      console.error('Error loading content articles:', error);
      showError('コンテンツの読み込みに失敗しました。', container);
    }
  }

  /**
   * Create article preview element
   * @param {Object} article - Article metadata
   * @returns {HTMLElement} Article preview element
   */
  function createArticlePreview(article) {
    const section = document.createElement('section');
    section.className = 'article';
    section.id = article.id;

    const header = document.createElement('div');
    header.className = 'article-header';

    const title = document.createElement('h2');
    title.className = 'article-title';
    title.textContent = article.title;

    const meta = document.createElement('div');
    meta.className = 'article-meta';
    meta.innerHTML = `
      <span>レベル: ${article.level}</span>
      <span>更新日: ${formatDate(article.updated_at)}</span>
    `;

    header.appendChild(title);
    header.appendChild(meta);

    const body = document.createElement('div');
    body.className = 'article-body';
    body.innerHTML = article.preview_html || `<p>${article.description}</p>`;

    const readMore = document.createElement('a');
    readMore.href = article.url;
    readMore.className = 'btn btn-text';
    readMore.textContent = '続きを読む →';

    section.appendChild(header);
    section.appendChild(body);
    section.appendChild(readMore);

    return section;
  }

  // ========================================
  // Tool Rendering
  // ========================================

  /**
   * Load and render tools
   * @param {string} containerId - Container element ID
   */
  async function loadTools(containerId) {
    const container = document.getElementById(containerId);
    if (!container) return;

    showLoading(container);

    try {
      const tools = await fetchJSON('../data/tools.json');

      if (!tools || tools.length === 0) {
        container.innerHTML = '<p class="loading">ツールがありません。</p>';
        return;
      }

      container.innerHTML = '';
      tools.forEach((tool) => {
        const card = createToolCard(tool);
        container.appendChild(card);
      });
    } catch (error) {
      console.error('Error loading tools:', error);
      showError('ツールの読み込みに失敗しました。', container);
    }
  }

  /**
   * Create tool card element
   * @param {Object} tool - Tool data
   * @returns {HTMLElement} Tool card element
   */
  function createToolCard(tool) {
    const article = document.createElement('article');
    article.className = 'card';

    const title = document.createElement('h3');
    title.className = 'card-title';
    title.textContent = tool.name;

    const description = document.createElement('p');
    description.className = 'card-description';
    description.textContent = tool.description;

    const meta = document.createElement('div');
    meta.className = 'paper-meta';
    meta.innerHTML = `
      <strong>カテゴリ:</strong> ${tool.category}<br>
      <strong>言語:</strong> ${tool.language}
    `;

    const link = document.createElement('a');
    link.href = tool.url;
    link.target = '_blank';
    link.rel = 'noopener noreferrer';
    link.className = 'btn btn-secondary';
    link.textContent = 'ツールを見る';

    article.appendChild(title);
    article.appendChild(description);
    article.appendChild(meta);
    article.appendChild(link);

    return article;
  }

  // ========================================
  // Auto-load based on page
  // ========================================

  /**
   * Automatically load data based on page structure
   */
  function autoLoadData() {
    // Recent papers on home page
    if (document.getElementById('recent-papers-list')) {
      loadPapers('recent-papers-list', 6);
    }

    // All papers on papers page
    if (document.getElementById('papers-list')) {
      loadPapers('papers-list');
    }

    // Datasets page
    if (document.getElementById('datasets-list')) {
      loadDatasets('datasets-list');
    }

    // Tutorials page
    if (document.getElementById('tutorials-list')) {
      loadTutorials('tutorials-list');
    }

    // Tools page
    if (document.getElementById('tools-list')) {
      loadTools('tools-list');
    }

    // Content articles (basics, methods, etc.)
    const contentContainer = document.getElementById('content-articles-list');
    if (contentContainer) {
      const pageCategory = detectPageCategory();
      if (pageCategory) {
        loadContentArticles('content-articles-list', pageCategory);
      }
    }
  }

  /**
   * Detect page category from URL
   * @returns {string|null} Category name
   */
  function detectPageCategory() {
    const path = window.location.pathname;
    if (path.includes('basics')) return 'basics';
    if (path.includes('methods')) return 'methods';
    if (path.includes('advanced')) return 'advanced';
    if (path.includes('applications')) return 'applications';
    return null;
  }

  // ========================================
  // Initialization
  // ========================================

  /**
   * Initialize data loader
   */
  function init() {
    autoLoadData();
    console.log('Data loader initialized');
  }

  // Run initialization when DOM is ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

  // Export functions for manual use
  window.MIKnowledgeHub.dataLoader = {
    loadPapers,
    loadDatasets,
    loadTutorials,
    loadContentArticles,
    loadTools,
  };
})();

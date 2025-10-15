/**
 * Main JavaScript - MI Knowledge Hub
 * Core functionality and utilities
 */

(function () {
  'use strict';

  // ========================================
  // Configuration
  // ========================================

  const CONFIG = {
    dataPath: '/data/',
    apiEndpoint: null, // No backend API for static site
    animationDuration: 250,
    debounceDelay: 300,
  };

  // ========================================
  // Utility Functions
  // ========================================

  /**
   * Debounce function to limit function execution rate
   * @param {Function} func - Function to debounce
   * @param {number} wait - Wait time in milliseconds
   * @returns {Function} Debounced function
   */
  function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
      const later = () => {
        clearTimeout(timeout);
        func(...args);
      };
      clearTimeout(timeout);
      timeout = setTimeout(later, wait);
    };
  }

  /**
   * Throttle function to limit function execution rate
   * @param {Function} func - Function to throttle
   * @param {number} limit - Time limit in milliseconds
   * @returns {Function} Throttled function
   */
  function throttle(func, limit) {
    let inThrottle;
    return function executedFunction(...args) {
      if (!inThrottle) {
        func.apply(this, args);
        inThrottle = true;
        setTimeout(() => (inThrottle = false), limit);
      }
    };
  }

  /**
   * Fetch JSON data from file
   * @param {string} path - Path to JSON file
   * @returns {Promise<Object>} Parsed JSON data
   */
  async function fetchJSON(path) {
    try {
      const response = await fetch(path);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      return data;
    } catch (error) {
      console.error('Error fetching JSON:', error);
      throw error;
    }
  }

  /**
   * Format date to Japanese locale
   * @param {string} dateString - ISO date string
   * @returns {string} Formatted date
   */
  function formatDate(dateString) {
    const date = new Date(dateString);
    return date.toLocaleDateString('ja-JP', {
      year: 'numeric',
      month: 'long',
      day: 'numeric',
    });
  }

  /**
   * Truncate text to specified length
   * @param {string} text - Text to truncate
   * @param {number} maxLength - Maximum length
   * @returns {string} Truncated text with ellipsis
   */
  function truncateText(text, maxLength) {
    if (text.length <= maxLength) return text;
    return text.substring(0, maxLength).trim() + '...';
  }

  /**
   * Sanitize HTML to prevent XSS
   * @param {string} html - HTML string to sanitize
   * @returns {string} Sanitized HTML
   */
  function sanitizeHTML(html) {
    const temp = document.createElement('div');
    temp.textContent = html;
    return temp.innerHTML;
  }

  /**
   * Show error message to user
   * @param {string} message - Error message
   * @param {HTMLElement} container - Container element
   */
  function showError(message, container) {
    const errorDiv = document.createElement('div');
    errorDiv.className = 'error';
    errorDiv.textContent = message;
    container.innerHTML = '';
    container.appendChild(errorDiv);
  }

  /**
   * Show loading state
   * @param {HTMLElement} container - Container element
   */
  function showLoading(container) {
    const loadingDiv = document.createElement('p');
    loadingDiv.className = 'loading';
    loadingDiv.textContent = '読み込み中...';
    container.innerHTML = '';
    container.appendChild(loadingDiv);
  }

  // ========================================
  // Smooth Scroll
  // ========================================

  /**
   * Initialize smooth scroll for anchor links
   */
  function initSmoothScroll() {
    document.querySelectorAll('a[href^="#"]').forEach((anchor) => {
      anchor.addEventListener('click', function (e) {
        const targetId = this.getAttribute('href');
        if (targetId === '#') return;

        const targetElement = document.querySelector(targetId);
        if (targetElement) {
          e.preventDefault();
          targetElement.scrollIntoView({
            behavior: 'smooth',
            block: 'start',
          });

          // Update URL without jumping
          if (history.pushState) {
            history.pushState(null, null, targetId);
          }
        }
      });
    });
  }

  // ========================================
  // Keyboard Navigation
  // ========================================

  /**
   * Enhance keyboard navigation for interactive elements
   */
  function initKeyboardNavigation() {
    // Handle Enter key on card elements
    document.querySelectorAll('.card, .paper-card, .learning-path-card').forEach((card) => {
      card.setAttribute('tabindex', '0');
      card.addEventListener('keydown', function (e) {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          const link = this.querySelector('a');
          if (link) link.click();
        }
      });
    });
  }

  // ========================================
  // Intersection Observer (Lazy Loading)
  // ========================================

  /**
   * Initialize Intersection Observer for lazy loading
   */
  function initIntersectionObserver() {
    if ('IntersectionObserver' in window) {
      const observerOptions = {
        root: null,
        rootMargin: '50px',
        threshold: 0.1,
      };

      const observer = new IntersectionObserver((entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            entry.target.classList.add('is-visible');
            observer.unobserve(entry.target);
          }
        });
      }, observerOptions);

      // Observe all lazy-load elements
      document.querySelectorAll('.lazy-load').forEach((element) => {
        observer.observe(element);
      });
    }
  }

  // ========================================
  // Form Validation
  // ========================================

  /**
   * Basic form validation
   * @param {HTMLFormElement} form - Form to validate
   * @returns {boolean} Validation result
   */
  function validateForm(form) {
    const inputs = form.querySelectorAll('input[required], textarea[required]');
    let isValid = true;

    inputs.forEach((input) => {
      if (!input.value.trim()) {
        input.classList.add('error');
        isValid = false;
      } else {
        input.classList.remove('error');
      }
    });

    return isValid;
  }

  // ========================================
  // Local Storage Helper
  // ========================================

  const storage = {
    get(key) {
      try {
        const item = localStorage.getItem(key);
        return item ? JSON.parse(item) : null;
      } catch (error) {
        console.error('Error reading from localStorage:', error);
        return null;
      }
    },

    set(key, value) {
      try {
        localStorage.setItem(key, JSON.stringify(value));
        return true;
      } catch (error) {
        console.error('Error writing to localStorage:', error);
        return false;
      }
    },

    remove(key) {
      try {
        localStorage.removeItem(key);
        return true;
      } catch (error) {
        console.error('Error removing from localStorage:', error);
        return false;
      }
    },
  };

  // ========================================
  // Initialization
  // ========================================

  /**
   * Initialize all components when DOM is ready
   */
  function init() {
    initSmoothScroll();
    initKeyboardNavigation();
    initIntersectionObserver();

    console.log('MI Knowledge Hub initialized');
  }

  // Run initialization when DOM is ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

  // ========================================
  // Export utilities to global scope
  // ========================================

  window.MIKnowledgeHub = {
    config: CONFIG,
    utils: {
      debounce,
      throttle,
      fetchJSON,
      formatDate,
      truncateText,
      sanitizeHTML,
      showError,
      showLoading,
      validateForm,
    },
    storage,
  };
})();

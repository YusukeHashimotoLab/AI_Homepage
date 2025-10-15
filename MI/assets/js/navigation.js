/**
 * Navigation JavaScript - MI Knowledge Hub
 * Handles navigation interactions and active state management
 */

(function () {
  'use strict';

  // ========================================
  // Active Navigation State
  // ========================================

  /**
   * Update active navigation link based on current page
   */
  function updateActiveNavigation() {
    const currentPath = window.location.pathname;
    const navLinks = document.querySelectorAll('.nav-link, .nav-mobile-item');

    navLinks.forEach((link) => {
      const linkPath = new URL(link.href, window.location.origin).pathname;

      // Remove active class from all links
      link.classList.remove('active');
      link.removeAttribute('aria-current');

      // Add active class to matching link
      if (currentPath === linkPath || (currentPath === '/' && linkPath.includes('index.html'))) {
        link.classList.add('active');
        link.setAttribute('aria-current', 'page');
      }
    });
  }

  // ========================================
  // Mobile Navigation Toggle (Future Enhancement)
  // ========================================

  /**
   * Toggle mobile menu visibility
   * Note: Currently not needed as we use bottom navigation
   * Kept for future implementation of hamburger menu
   */
  function initMobileMenuToggle() {
    const menuToggle = document.querySelector('.menu-toggle');
    const mobileMenu = document.querySelector('.mobile-menu');

    if (menuToggle && mobileMenu) {
      menuToggle.addEventListener('click', () => {
        const isOpen = mobileMenu.classList.toggle('is-open');
        menuToggle.setAttribute('aria-expanded', isOpen);
        menuToggle.setAttribute('aria-label', isOpen ? 'メニューを閉じる' : 'メニューを開く');

        // Prevent body scroll when menu is open
        document.body.style.overflow = isOpen ? 'hidden' : '';
      });

      // Close menu when clicking outside
      document.addEventListener('click', (e) => {
        if (!mobileMenu.contains(e.target) && !menuToggle.contains(e.target)) {
          mobileMenu.classList.remove('is-open');
          menuToggle.setAttribute('aria-expanded', 'false');
          document.body.style.overflow = '';
        }
      });

      // Close menu on escape key
      document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && mobileMenu.classList.contains('is-open')) {
          mobileMenu.classList.remove('is-open');
          menuToggle.setAttribute('aria-expanded', 'false');
          document.body.style.overflow = '';
        }
      });
    }
  }

  // ========================================
  // Sidebar Navigation
  // ========================================

  /**
   * Update active state for sidebar navigation based on scroll position
   */
  function initSidebarNavigation() {
    const sidebarLinks = document.querySelectorAll('.sidebar-link');
    if (sidebarLinks.length === 0) return;

    // Get all section IDs from sidebar links
    const sections = Array.from(sidebarLinks)
      .map((link) => {
        const href = link.getAttribute('href');
        if (href && href.startsWith('#')) {
          return document.querySelector(href);
        }
        return null;
      })
      .filter(Boolean);

    if (sections.length === 0) return;

    /**
     * Update active sidebar link based on scroll position
     */
    function updateSidebarActive() {
      const scrollPosition = window.scrollY + 100; // Offset for header

      let activeSection = null;

      sections.forEach((section) => {
        const sectionTop = section.offsetTop;
        const sectionBottom = sectionTop + section.offsetHeight;

        if (scrollPosition >= sectionTop && scrollPosition < sectionBottom) {
          activeSection = section;
        }
      });

      // Update sidebar links
      sidebarLinks.forEach((link) => {
        link.classList.remove('active');
        const href = link.getAttribute('href');
        if (activeSection && href === `#${activeSection.id}`) {
          link.classList.add('active');
        }
      });
    }

    // Throttle scroll event for performance
    const throttledUpdate = window.MIKnowledgeHub.utils.throttle(updateSidebarActive, 100);
    window.addEventListener('scroll', throttledUpdate);

    // Initial update
    updateSidebarActive();
  }

  // ========================================
  // Header Scroll Behavior
  // ========================================

  /**
   * Add scroll-based styling to header
   */
  function initHeaderScrollBehavior() {
    const header = document.querySelector('.header');
    if (!header) return;

    let lastScrollTop = 0;
    const scrollThreshold = 50;

    /**
     * Handle scroll events for header
     */
    function handleScroll() {
      const scrollTop = window.pageYOffset || document.documentElement.scrollTop;

      // Add shadow when scrolled
      if (scrollTop > scrollThreshold) {
        header.classList.add('is-scrolled');
      } else {
        header.classList.remove('is-scrolled');
      }

      // Hide header on scroll down, show on scroll up (optional)
      // if (scrollTop > lastScrollTop && scrollTop > scrollThreshold) {
      //   header.classList.add('is-hidden');
      // } else {
      //   header.classList.remove('is-hidden');
      // }

      lastScrollTop = scrollTop <= 0 ? 0 : scrollTop;
    }

    // Throttle scroll event
    const throttledScroll = window.MIKnowledgeHub.utils.throttle(handleScroll, 100);
    window.addEventListener('scroll', throttledScroll);

    // Initial check
    handleScroll();
  }

  // ========================================
  // Breadcrumb Navigation (Future Enhancement)
  // ========================================

  /**
   * Generate breadcrumb navigation based on current page
   */
  function generateBreadcrumbs() {
    const breadcrumbContainer = document.querySelector('.breadcrumbs');
    if (!breadcrumbContainer) return;

    const path = window.location.pathname;
    const segments = path.split('/').filter(Boolean);

    const breadcrumbs = [
      { label: 'ホーム', url: '/index.html' },
    ];

    // Build breadcrumb path
    let currentPath = '';
    segments.forEach((segment, index) => {
      currentPath += '/' + segment;
      const isLast = index === segments.length - 1;

      // Format segment label
      let label = segment
        .replace('.html', '')
        .replace(/-/g, ' ')
        .replace(/\b\w/g, (l) => l.toUpperCase());

      breadcrumbs.push({
        label,
        url: isLast ? null : currentPath,
      });
    });

    // Render breadcrumbs
    const breadcrumbHTML = breadcrumbs
      .map((crumb, index) => {
        if (crumb.url) {
          return `<a href="${crumb.url}" class="breadcrumb-link">${crumb.label}</a>`;
        } else {
          return `<span class="breadcrumb-current">${crumb.label}</span>`;
        }
      })
      .join('<span class="breadcrumb-separator">/</span>');

    breadcrumbContainer.innerHTML = breadcrumbHTML;
  }

  // ========================================
  // Keyboard Shortcuts
  // ========================================

  /**
   * Initialize keyboard shortcuts for navigation
   */
  function initKeyboardShortcuts() {
    document.addEventListener('keydown', (e) => {
      // Skip if user is typing in an input
      if (e.target.matches('input, textarea')) return;

      // Slash key (/) - Focus search (future implementation)
      if (e.key === '/' && !e.shiftKey && !e.ctrlKey && !e.metaKey) {
        e.preventDefault();
        const searchInput = document.querySelector('.search-input');
        if (searchInput) {
          searchInput.focus();
        }
      }

      // Escape key - Clear search or close modals
      if (e.key === 'Escape') {
        const searchInput = document.querySelector('.search-input');
        if (searchInput && document.activeElement === searchInput) {
          searchInput.value = '';
          searchInput.blur();
        }
      }
    });
  }

  // ========================================
  // Back to Top Button (Future Enhancement)
  // ========================================

  /**
   * Show/hide back to top button based on scroll position
   */
  function initBackToTop() {
    const backToTopButton = document.querySelector('.back-to-top');
    if (!backToTopButton) return;

    function toggleBackToTop() {
      if (window.scrollY > 500) {
        backToTopButton.classList.add('is-visible');
      } else {
        backToTopButton.classList.remove('is-visible');
      }
    }

    // Throttle scroll event
    const throttledToggle = window.MIKnowledgeHub.utils.throttle(toggleBackToTop, 100);
    window.addEventListener('scroll', throttledToggle);

    // Handle click
    backToTopButton.addEventListener('click', () => {
      window.scrollTo({
        top: 0,
        behavior: 'smooth',
      });
    });

    // Initial check
    toggleBackToTop();
  }

  // ========================================
  // Initialization
  // ========================================

  /**
   * Initialize navigation components
   */
  function init() {
    updateActiveNavigation();
    initMobileMenuToggle();
    initSidebarNavigation();
    initHeaderScrollBehavior();
    initKeyboardShortcuts();
    initBackToTop();

    console.log('Navigation initialized');
  }

  // Run initialization when DOM is ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();

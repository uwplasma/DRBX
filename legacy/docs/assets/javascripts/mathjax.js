// MathJax configuration for mkdocs-material + pymdownx.arithmatex.
//
// Notes:
// - The arithmatex "generic" mode leaves TeX delimiters (\(...\), \[...\]) in the HTML.
// - MathJax typesets these at runtime.
// - We trigger an explicit typeset both on initial page load and on mkdocs-material navigation
//   events (if "instant navigation" is enabled later).

window.MathJax = {
  tex: {
    inlineMath: [['$', '$'], ['\\(', '\\)']],
    displayMath: [['$$', '$$'], ['\\[', '\\]']],
    processEscapes: true,
  },
  svg: {
    // Share a single global font cache to reduce page weight and re-typesetting overhead.
    fontCache: 'global',
  },
  options: {
    skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code'],
  },
  startup: {
    typeset: false,
    ready: () => {
      // eslint-disable-next-line no-undef
      MathJax.startup.defaultReady();

      // eslint-disable-next-line no-undef
      const typeset = () => MathJax.typesetPromise();

      // Initial load.
      typeset();

      // mkdocs-material "instant navigation" hook (if enabled).
      if (typeof window.document$ !== 'undefined') {
        // eslint-disable-next-line no-undef
        window.document$.subscribe(() => {
          typeset();
        });
      }
    },
  },
};

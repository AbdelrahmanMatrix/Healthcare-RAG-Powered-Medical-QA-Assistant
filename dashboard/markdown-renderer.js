/**
 * renderMarkdown — Markdown to HTML renderer.
 *
 * Delegates to the battle-tested `marked` library.
 * This wrapper exists so the dashboard and tests use the same entry point.
 *
 * In the browser: `marked` is loaded via CDN (<head>) as a global.
 * In Node.js:    `marked` is resolved from node_modules (devDependency).
 *
 * XSS protection: Uses marked's built-in renderer.html hook to escape raw HTML
 * blocks/inline HTML. Unlike pre-escaping the input (which causes double-escaping
 * inside code blocks), this only escapes raw HTML that marked identifies as
 * separate HTML tokens — code blocks, bold, tables, etc. are unaffected.
 *
 * @param {string} text — Markdown text to render.
 * @returns {string} — Rendered HTML string.
 */
function renderMarkdown(text) {
  if (!text) return '';

  // Browser: marked is a global set by the CDN <script> tag
  // Node: marked is resolved via require() by the test file
  if (typeof marked !== 'undefined' && typeof marked.parse === 'function') {
    var r = new marked.Renderer();
    r.html = escapeHtmlToken;
    r.htmlInline = escapeHtmlToken;
    return marked.parse(text, { renderer: r });
  }

  // Fallback (shouldn't happen if dependencies are loaded correctly)
  console.warn('marked.parse() not available — falling back to HTML-escaped text');
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/\n/g, '<br>');
}

/**
 * Escape raw HTML tokens to prevent XSS (used by marked renderer hooks).
 *
 * In marked v15+, the html/htmlInline renderer methods receive a token object
 * (with .text, .raw, .block properties), not a plain string. This function
 * handles both formats for compatibility.
 */
function escapeHtmlToken(arg) {
  var text = (typeof arg === 'string') ? arg : (arg.text || arg.raw || '');
  return text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

// Export for Node.js testing
if (typeof module !== 'undefined' && module.exports) {
  const marked = require('marked');
  module.exports = {
    renderMarkdown(text) {
      if (!text) return '';
      // Marked handles code block escaping natively; we escape raw HTML for XSS
      var r = new marked.Renderer();
      r.html = escapeHtmlToken;
      r.htmlInline = escapeHtmlToken;
      return marked.parse(text, { renderer: r });
    },
  };
}

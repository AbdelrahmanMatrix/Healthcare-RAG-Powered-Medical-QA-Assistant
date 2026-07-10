/**
 * renderMarkdown — Simple Markdown to HTML renderer.
 *
 * Converts a subset of markdown syntax to HTML:
 *   - Fenced code blocks (```lang or ~~~lang) → <pre><code>
 *   - Inline code (`code`) → <code>
 *   - Bold (**text**) → <strong>
 *   - Pipe tables → <table class="rag-table">
 *   - Unordered lists (- and *) → <ul><li>
 *   - Newlines → <br>
 *
 * Designed for the Healthcare RAG dashboard. Input is plain text
 * (not HTML) and output is safe HTML (entities escaped at entry).
 *
 * @param {string} text — Markdown text to render.
 * @returns {string} — Rendered HTML string.
 */
function renderMarkdown(text) {
  if (!text) return '';
  // Escape HTML first
  var html = text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

  // ── Step 0a: Extract fenced code blocks ──────────────────────────
  // Replace ```lang...``` blocks with placeholders so other markdown
  // processing doesn't mangle them. Restore at the end with <pre><code>.
  var codeBlocks = [];
  var codeBlockIndex = 0;
  html = html.replace(/(```|~~~)(\w*)\s*\n?([\s\S]*?)\n?\1/g, function(match, fence, lang, code) {
    var idx = codeBlockIndex++;
    codeBlocks[idx] = {lang: lang || '', code: code};
    return '~~CODEBLOCK_' + idx + '~~';
  });

  // ── Step 0b: Convert inline code ─────────────────────────────────
  // Single backtick inline code: `text`
  html = html.replace(/`([^`]+)`/g, '<code>$1</code>');

  // Step 1: Convert **bold** (do this early so it works everywhere)
  html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');

  // Step 2: Convert markdown tables: | col1 | col2 | ...
  // First, collect all table lines and replace them as a group
  var lines = html.split('\n');
  var tableLines = [];
  var inTable = false;
  var resultLines = [];

  function flushTable() {
    if (tableLines.length < 2) {
      // Not enough lines for a table — put them back as-is
      resultLines = resultLines.concat(tableLines);
      tableLines = [];
      return;
    }
    // Filter out separator rows (rows where ALL non-empty cells are just dashes/colons/spaces)
    var dataLines = tableLines.filter(function(line) {
      var cells = line.replace(/^\||\|$/g, '').split('|').map(function(c) { return c.trim(); });
      var nonEmpty = cells.filter(function(c) { return c.length > 0; });
      // If every non-empty cell matches the separator pattern, this is a separator row
      var allSeparators = nonEmpty.every(function(c) { return /^:?-+:?$/.test(c); });
      return !allSeparators;
    });

    if (dataLines.length === 0) {
      tableLines = [];
      return;
    }

    var rows = dataLines.map(function(line) {
      var cells = line.replace(/^\||\|$/g, '').split('|').map(function(c) {
        return c.trim().replace(/^:?-+:?$/, '').trim();
      });
      var rowHtml = cells.map(function(c) {
        if (!c) return '<td></td>';
        return '<td>' + c + '</td>';
      }).join('');
      return '<tr>' + rowHtml + '</tr>';
    });

    resultLines.push('<table class="rag-table">' + rows.join('') + '</table>');
    tableLines = [];
  }

  for (var i = 0; i < lines.length; i++) {
    var line = lines[i];
    var trimmed = line.trim();
    if (/^\|.*\|$/.test(trimmed)) {
      tableLines.push(trimmed);
      inTable = true;
    } else {
      if (inTable) {
        flushTable();
        inTable = false;
      }
      resultLines.push(line);
    }
  }
  // Flush any remaining table at end
  if (inTable) flushTable();

  html = resultLines.join('\n');

  // Step 3: Convert lines starting with - or * to list items
  lines = html.split('\n');
  var listLines = [];
  var inList = false;
  resultLines = [];

  function flushList() {
    if (listLines.length > 0) {
      resultLines.push('<ul>' + listLines.join('') + '</ul>');
      listLines = [];
    }
  }

  for (var i = 0; i < lines.length; i++) {
    var line = lines[i];
    var match = line.match(/^\s*[-*]\s+(.+)$/);
    if (match) {
      listLines.push('<li>' + match[1] + '</li>');
      inList = true;
    } else {
      if (inList) {
        flushList();
        inList = false;
      }
      resultLines.push(line);
    }
  }
  if (inList) flushList();

  html = resultLines.join('\n');

  // Step 4: Convert remaining newlines to <br>
  html = html.replace(/\n/g, '<br>');

  // ── Step 5: Restore fenced code blocks ────────────────────────────
  html = html.replace(/~~CODEBLOCK_(\d+)~~/g, function(match, idx) {
    var block = codeBlocks[parseInt(idx)];
    if (!block) return match;
    var code = block.code
      .replace(/&amp;/g, '&')
      .replace(/&lt;/g, '<')
      .replace(/&gt;/g, '>');
    var langClass = block.lang ? ' class="language-' + block.lang + '"' : '';
    var langLabel = block.lang ? '<span class="lang-label">' + block.lang + '</span>' : '';
    return '<pre>' + langLabel + '<code' + langClass + '>' + code + '</code></pre>';
  });

  return html;
}

// Export for Node.js testing
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { renderMarkdown };
}

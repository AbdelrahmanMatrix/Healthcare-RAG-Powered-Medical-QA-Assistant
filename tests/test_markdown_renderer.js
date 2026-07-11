/**
 * Unit tests for renderMarkdown() — now delegates to marked.js.
 *
 * Runs with Node.js built-in test runner (node:test, available since v18).
 * Requires `marked` to be installed (see package.json).
 *
 * Usage:
 *   node --test tests/test_markdown_renderer.js
 */
const { describe, it, before } = require('node:test');
const assert = require('node:assert/strict');

let renderMarkdown;

before(() => {
  // Load the module which internally require('marked')
  const mod = require('../dashboard/markdown-renderer');
  renderMarkdown = mod.renderMarkdown;
});

// ══════════════════════════════════════════════════════════════════════════════
// Edge cases & empty input
// ══════════════════════════════════════════════════════════════════════════════
describe('Edge cases & empty input', () => {
  it('returns empty string for null/undefined', () => {
    assert.equal(renderMarkdown(null), '');
    assert.equal(renderMarkdown(undefined), '');
  });

  it('returns empty string for empty input', () => {
    assert.equal(renderMarkdown(''), '');
  });

  it('passes plain text through', () => {
    const result = renderMarkdown('Hello world');
    // marked wraps text in <p>
    assert.ok(result.includes('Hello world'));
  });

  it('escapes HTML entities by default', () => {
    const result = renderMarkdown('<script>alert("xss")</script>');
    assert.ok(!result.includes('<script>'));
    // marked escapes by default
    assert.ok(result.includes('&lt;script&gt;') || result.includes('&amp;lt;'));
  });
});

// ══════════════════════════════════════════════════════════════════════════════
// Bold text
// ══════════════════════════════════════════════════════════════════════════════
describe('Bold text', () => {
  it('converts **bold** to <strong>', () => {
    const result = renderMarkdown('This is **bold** text');
    assert.ok(result.includes('<strong>bold</strong>'));
  });

  it('handles multiple bold segments', () => {
    const result = renderMarkdown('**first** and **second**');
    assert.ok(result.includes('<strong>first</strong>'));
    assert.ok(result.includes('<strong>second</strong>'));
  });

  it('does not break on lonely asterisks', () => {
    const result = renderMarkdown('5 * 3 = 15');
    assert.ok(result.includes('5 * 3 = 15'));
  });
});

// ══════════════════════════════════════════════════════════════════════════════
// Inline code
// ══════════════════════════════════════════════════════════════════════════════
describe('Inline code', () => {
  it('converts `code` to <code>', () => {
    const result = renderMarkdown('Use the `renderMarkdown` function');
    assert.ok(result.includes('<code>renderMarkdown</code>'));
  });

  it('handles multiple inline code segments', () => {
    const result = renderMarkdown('`foo` and `bar`');
    assert.ok(result.includes('<code>foo</code>'));
    assert.ok(result.includes('<code>bar</code>'));
  });

  it('does not break on single backticks without content', () => {
    const result = renderMarkdown('A single ` backtick');
    assert.ok(result.includes('A single ` backtick'));
  });
});

// ══════════════════════════════════════════════════════════════════════════════
// Fenced code blocks
// ══════════════════════════════════════════════════════════════════════════════
describe('Fenced code blocks', () => {
  it('converts ``` blocks to <pre><code>', () => {
    const result = renderMarkdown(
      '```\n' +
      'const x = 1;\n' +
      'console.log(x);\n' +
      '```'
    );
    assert.ok(result.includes('<pre>'));
    assert.ok(result.includes('<code>'));
    assert.ok(result.includes('const x = 1;'));
    assert.ok(result.includes('console.log(x);'));
  });

  it('preserves language class for code blocks', () => {
    const result = renderMarkdown(
      '```javascript\n' +
      'function hello() {}\n' +
      '```'
    );
    assert.ok(result.includes('class="language-javascript"'));
    // marked does NOT emit the custom <span class="lang-label">
    assert.ok(!result.includes('lang-label'));
  });

  it('handles ~~~ fence variant', () => {
    const result = renderMarkdown(
      '~~~python\n' +
      'print("hello")\n' +
      '~~~'
    );
    assert.ok(result.includes('class="language-python"'));
    // marked escapes " to &quot; even inside code blocks, and < to &lt; for XSS
    assert.ok(result.includes('print('));
    assert.ok(result.includes('hello'));
  });

  it('escapes HTML in code blocks for XSS safety', () => {
    const result = renderMarkdown(
      '```html\n' +
      '<div class="test">Hello</div>\n' +
      '```'
    );
    // marked native escaping: < → &lt;, > → &gt;, " → &quot; inside code blocks
    assert.ok(result.includes('&lt;div'));
    assert.ok(!result.includes('<div class="test">'));
  });

  it('handles multiple code blocks', () => {
    const result = renderMarkdown(
      'First:\n```js\nvar a = 1;\n```\n\nSecond:\n```py\nb = 2\n```'
    );
    assert.ok(result.includes('var a = 1;'));
    assert.ok(result.includes('b = 2'));
    const langJsCount = (result.match(/language-js/g) || []).length;
    const langPyCount = (result.match(/language-py/g) || []).length;
    assert.equal(langJsCount, 1);
    assert.equal(langPyCount, 1);
  });
});

// ══════════════════════════════════════════════════════════════════════════════
// Markdown tables
// ══════════════════════════════════════════════════════════════════════════════
describe('Markdown tables', () => {
  it('converts a simple table with thead/tbody', () => {
    const result = renderMarkdown(
      '| Header1 | Header2 |\n' +
      '|---------|--------|\n' +
      '| Cell1   | Cell2   |'
    );
    // marked outputs <table> with <thead> and <th> for headers
    assert.ok(result.includes('<table>'));
    assert.ok(result.includes('<thead>'));
    assert.ok(result.includes('<th>Header1</th>'));
    assert.ok(result.includes('<td>Cell1</td>'));
  });

  it('handles multi-row tables', () => {
    const result = renderMarkdown(
      '| A | B |\n' +
      '|---|---|\n' +
      '| 1 | 2 |\n' +
      '| 3 | 4 |'
    );
    assert.ok(result.includes('<td>1</td>'));
    assert.ok(result.includes('<td>2</td>'));
    assert.ok(result.includes('<td>3</td>'));
    assert.ok(result.includes('<td>4</td>'));
  });

  it('supports **bold** in table cells', () => {
    const result = renderMarkdown(
      '| **Symptom** | What it feels like |\n' +
      '|---------|-------------------|\n' +
      '| **Fever** | You feel hot |'
    );
    assert.ok(result.includes('<strong>Symptom</strong>'));
    assert.ok(result.includes('<strong>Fever</strong>'));
  });

  it('handles table with leading and trailing text', () => {
    const result = renderMarkdown(
      'Before\n\n| X | Y |\n|---|---|\n| 1 | 2 |\n\nAfter'
    );
    assert.ok(result.includes('Before'));
    assert.ok(result.includes('<table>'));
    assert.ok(result.includes('<td>1</td>'));
    assert.ok(result.includes('After'));
  });
});

// ══════════════════════════════════════════════════════════════════════════════
// Unordered lists
// ══════════════════════════════════════════════════════════════════════════════
describe('Unordered lists', () => {
  it('converts dash list items', () => {
    const result = renderMarkdown('- Item one\n- Item two\n- Item three');
    assert.ok(result.includes('<ul>'));
    assert.ok(result.includes('<li>Item one</li>'));
    assert.ok(result.includes('<li>Item two</li>'));
    assert.ok(result.includes('<li>Item three</li>'));
    assert.ok(result.includes('</ul>'));
  });

  it('converts asterisk list items', () => {
    const result = renderMarkdown('* First\n* Second');
    assert.ok(result.includes('<li>First</li>'));
    assert.ok(result.includes('<li>Second</li>'));
  });

  it('bold in list items is rendered', () => {
    const result = renderMarkdown('- **Important** item');
    assert.ok(result.includes('<li><strong>Important</strong> item</li>') ||
              result.includes('<strong>Important</strong>'));
  });
});

// ══════════════════════════════════════════════════════════════════════════════
// Combined markdown (realistic medical QA response)
// ══════════════════════════════════════════════════════════════════════════════
describe('Combined markdown', () => {
  it('renders a realistic medical QA response with table and bold', () => {
    const input =
      '**Flu (influenza) symptoms**\n\n' +
      'The flu usually comes on fast.\n\n' +
      '| Symptom | Description |\n' +
      '|---------|-------------|\n' +
      '| **Fever** | 100-104°F |\n' +
      '| **Chills** | Shaking |\n\n' +
      'See a doctor if symptoms worsen.';

    const result = renderMarkdown(input);
    assert.ok(result.includes('<strong>Flu'));
    assert.ok(result.includes('<table>'));
    assert.ok(result.includes('<strong>Fever</strong>'));
    assert.ok(result.includes('See a doctor'));
  });
});

// ══════════════════════════════════════════════════════════════════════════════
// No markdown: pass-through behaviour
// ══════════════════════════════════════════════════════════════════════════════
describe('No markdown passthrough', () => {
  it('renders single line wrapped in <p>', () => {
    const result = renderMarkdown('Just a sentence.');
    assert.ok(result.includes('Just a sentence.'));
    // marked wraps paragraph text in <p>
    assert.ok(result.startsWith('<p>'));
  });
});

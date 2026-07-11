/**
 * Unit tests for renderMarkdown() — the simple markdown renderer used
 * in the Healthcare RAG dashboard.
 *
 * Runs with Node.js built-in test runner (node:test, available since v18).
 * Usage:
 *   node --test tests/test_markdown_renderer.js
 *   # or: node --test (discovers all test/*.js files)
 */

const { describe, it } = require('node:test');
const assert = require('node:assert/strict');
const { renderMarkdown } = require('../dashboard/markdown-renderer');

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

  it('passes plain text through with <br> for newlines', () => {
    const result = renderMarkdown('Hello world');
    assert.equal(result, 'Hello world');
  });

  it('converts newlines to <br>', () => {
    const result = renderMarkdown('Line one\nLine two');
    assert.equal(result, 'Line one<br>Line two');
  });

  it('escapes HTML entities', () => {
    const result = renderMarkdown('<script>alert("xss")</script>');
    assert.ok(result.includes('&lt;script&gt;'));
    assert.ok(result.includes('&quot;'));
    assert.ok(!result.includes('<script>'));
    assert.ok(!result.includes('alert'));
  });
});

// ══════════════════════════════════════════════════════════════════════════════
// Bold text
// ══════════════════════════════════════════════════════════════════════════════
describe('Bold text', () => {
  it('converts **bold** to <strong>', () => {
    const result = renderMarkdown('This is **bold** text');
    assert.equal(result, 'This is <strong>bold</strong> text');
  });

  it('handles multiple bold segments', () => {
    const result = renderMarkdown('**first** and **second**');
    assert.equal(result, '<strong>first</strong> and <strong>second</strong>');
  });

  it('handles bold inside table cells', () => {
    const result = renderMarkdown(
      '| Symptom | Description |\n' +
      '|---------|-------------|\n' +
      '| **Fever** | High temperature |'
    );
    assert.ok(result.includes('<strong>Fever</strong>'));
    assert.ok(result.includes('<td>Fever</td>') || result.includes('<strong>Fever</strong>'));
  });

  it('does not break on lonely asterisks', () => {
    const result = renderMarkdown('5 * 3 = 15');
    assert.equal(result, '5 * 3 = 15');
  });
});

// ══════════════════════════════════════════════════════════════════════════════
// Inline code
// ══════════════════════════════════════════════════════════════════════════════
describe('Inline code', () => {
  it('converts `code` to <code>', () => {
    const result = renderMarkdown('Use the `renderMarkdown` function');
    assert.equal(result, 'Use the <code>renderMarkdown</code> function');
  });

  it('handles multiple inline code segments', () => {
    const result = renderMarkdown('`foo` and `bar`');
    assert.equal(result, '<code>foo</code> and <code>bar</code>');
  });

  it('does not break on single backticks without content', () => {
    const result = renderMarkdown('A single ` backtick');
    assert.equal(result, 'A single ` backtick');
  });

  it('inline code escapes HTML entities (no revert like fenced blocks)', () => {
    const result = renderMarkdown('Use `<div>` tags');
    // Inline code does NOT revert entities, so &lt; stays as &lt;
    assert.ok(result.includes('<code>'));
    assert.ok(result.includes('&lt;div&gt;'));
    assert.ok(!result.includes('<div>'));
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
    assert.ok(result.includes('<span class="lang-label">javascript</span>'));
  });

  it('handles ~~~ fence variant', () => {
    const result = renderMarkdown(
      '~~~python\n' +
      'print("hello")\n' +
      '~~~'
    );
    assert.ok(result.includes('class="language-python"'));
    assert.ok(result.includes('print("hello")'));
  });

  it('handles multiple code blocks', () => {
    const result = renderMarkdown(
      'First:\n```js\nvar a = 1;\n```\n\nSecond:\n```py\nb = 2\n```'
    );
    assert.ok(result.includes('var a = 1;'));
    assert.ok(result.includes('b = 2'));
    assert.ok(result.match(/language-js/g));
    assert.ok(result.match(/language-py/g));
  });

  it('preserves content with HTML-like syntax', () => {
    const result = renderMarkdown(
      '```html\n' +
      '<div class="test">Hello</div>\n' +
      '```'
    );
    // HTML entities in code blocks should be reverted to raw text
    assert.ok(result.includes('<div class="test">Hello</div>'));
    assert.ok(!result.includes('&lt;'));  // content inside <pre><code> should not be escaped
  });

  it('code block content not processed for bold/tables/lists', () => {
    const result = renderMarkdown(
      '```\n' +
      '**not bold**\n' +
      '- not a list\n' +
      '```'
    );
    assert.ok(result.includes('**not bold**'));
    assert.ok(result.includes('- not a list'));
    assert.ok(!result.includes('<strong>not bold</strong>'));
  });

  it('handles code block at end of string', () => {
    const result = renderMarkdown(
      'Some text\n```\ncode block\n```'
    );
    assert.ok(result.includes('code block'));
  });

  it('handles empty code block (no content)', () => {
    const result = renderMarkdown(
      '```\n' +
      '```'
    );
    // Empty code blocks should produce empty <pre><code>
    assert.ok(result.includes('<pre>'));
    assert.ok(result.includes('<code>'));
    assert.ok(result.includes('</code></pre>'));
  });

  it('handles empty code block with language specifier', () => {
    const result = renderMarkdown(
      '```python\n' +
      '```'
    );
    assert.ok(result.includes('class="language-python"'));
    assert.ok(result.includes('<span class="lang-label">python</span>'));
  });
    const result = renderMarkdown(
      '```\n' +
      '```'
    );
    // Empty code blocks should produce empty <pre><code>
    assert.ok(result.includes('<pre>'));
    assert.ok(result.includes('<code>'));
    assert.ok(result.includes('</code></pre>'));
  });

  it('handles consecutive fenced code blocks', () => {
    const result = renderMarkdown(
      '```js\n' +
      'var a = 1;\n' +
      '```\n' +
      '```py\n' +
      'b = 2\n' +
      '```'
    );
    assert.ok(result.includes('var a = 1;'));
    assert.ok(result.includes('b = 2'));
    // Should have exactly 2 code blocks
    const preCount = (result.match(/<pre>/g) || []).length;
    assert.equal(preCount, 2);
    const langJsCount = (result.match(/language-js/g) || []).length;
    const langPyCount = (result.match(/language-py/g) || []).length;
    assert.equal(langJsCount, 1);
    assert.equal(langPyCount, 1);
  });

  it('handles code block at start of string', () => {
    const result = renderMarkdown(
      '```\ncode block\n```\nSome text'
    );
    assert.ok(result.includes('code block'));
    assert.ok(result.includes('Some text'));
  });
});

// ══════════════════════════════════════════════════════════════════════════════
// Markdown tables
// ══════════════════════════════════════════════════════════════════════════════
describe('Markdown tables', () => {
  it('converts a simple table', () => {
    const result = renderMarkdown(
      '| Header1 | Header2 |\n' +
      '|---------|--------|\n' +
      '| Cell1   | Cell2   |'
    );
    assert.ok(result.includes('<table class="rag-table">'));
    assert.ok(result.includes('<td>Header1</td>'));
    assert.ok(result.includes('<td>Cell1</td>'));
    // Separator row should be filtered out
    const separatorMatch = result.match(/---/g);
    assert.equal(separatorMatch, null);
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

  it('filters separator rows completely', () => {
    const result = renderMarkdown(
      '| Name | Value |\n' +
      '|------|-------|\n' +
      '| Foo  | Bar   |'
    );
    // Should have exactly 2 <tr> (header + data), not 3 (header + separator + data)
    const trCount = (result.match(/<tr>/g) || []).length;
    assert.equal(trCount, 2);
  });

  it('handles tables with alignment markers', () => {
    const result = renderMarkdown(
      '| Left | Center | Right |\n' +
      '|:-----|:------:|------:|\n' +
      '| L    | C      | R     |'
    );
    assert.ok(result.includes('<td>L</td>'));
    assert.ok(result.includes('<td>C</td>'));
    assert.ok(result.includes('<td>R</td>'));
  });

  it('handles table with trailing text', () => {
    const result = renderMarkdown(
      '| Col1 | Col2 |\n' +
      '|------|------|\n' +
      '| A    | B    |\n' +
      '\nSome text after table'
    );
    assert.ok(result.includes('<table'));
    assert.ok(result.includes('Some text after table'));
  });

  it('handles table with leading text', () => {
    const result = renderMarkdown(
      'Before\n| X | Y |\n|---|---|\n| 1 | 2 |'
    );
    assert.ok(result.includes('Before'));
    assert.ok(result.includes('<table'));
    assert.ok(result.includes('<td>1</td>'));
  });

  it('tables with empty cells produce empty <td> elements', () => {
    const result = renderMarkdown(
      '| A | B | C |\n' +
      '|---|---|---|\n' +
      '| X |   | Z |'
    );
    assert.ok(result.includes('<td>X</td>'));
    assert.ok(result.includes('<td></td>'));
    assert.ok(result.includes('<td>Z</td>'));
  });

  it('single pipe line is not treated as table', () => {
    const result = renderMarkdown('| just one row |');
    // Not enough rows for a table — should be passed through
    assert.equal(result.includes('<table'), false);
    assert.ok(result.includes('just one row'));
  });

  it('tables with **bold** in cells render correctly', () => {
    const result = renderMarkdown(
      '| Symptom | What it feels like |\n' +
      '|---------|-------------------|\n' +
      '| **Fever** | You feel hot |'
    );
    // Bold inside table should still work: **Fever** → <strong>Fever</strong>
    assert.ok(result.includes('<strong>Fever</strong>') || result.includes('<td><strong>Fever</strong></td>'));
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

  it('handles indented list items', () => {
    const result = renderMarkdown('  - Indented item');
    assert.ok(result.includes('<li>Indented item</li>'));
  });

  it('separates adjacent lists with text between them', () => {
    const result = renderMarkdown('- List A item\n\nSome text\n\n- List B item');
    // These should be two separate <ul> lists, not merged
    const ulCount = (result.match(/<ul>/g) || []).length;
    assert.equal(ulCount, 2);
  });

  it('bold in list items is rendered', () => {
    const result = renderMarkdown('- **Important** item');
    assert.ok(result.includes('<li><strong>Important</strong> item</li>'));
  });
});

// ══════════════════════════════════════════════════════════════════════════════
// Combined markdown
// ══════════════════════════════════════════════════════════════════════════════
describe('Combined markdown', () => {
  it('renders a realistic medical QA response with table and bold', () => {
    const input =
      '**Flu (influenza) symptoms** – what to look for\n' +
      'The flu usually comes on fast.\n\n' +
      '| Symptom | Description |\n' +
      '|---------|-------------|\n' +
      '| **Fever** | 100-104°F |\n' +
      '| **Chills** | Shaking |\n\n' +
      'See a doctor if symptoms worsen.';

    const result = renderMarkdown(input);
    assert.ok(result.includes('<strong>Flu'));
    assert.ok(result.includes('<table class="rag-table">'));
    assert.ok(result.includes('<td>Fever</td>') || result.includes('<strong>Fever</strong>'));
    assert.ok(result.includes('See a doctor'));
  });

  it('renders text with code block, table, and list', () => {
    const input =
      '## Example\n' +
      '```bash\n' +
      'echo "hello"\n' +
      '```\n' +
      'Results:\n' +
      '| Tool | Status |\n' +
      '|------|--------|\n' +
      '| A    | ✅     |\n' +
      '- Done\n' +
      '- Checked';

    const result = renderMarkdown(input);
    assert.ok(result.includes('<pre>'));
    assert.ok(result.includes('<code>'));
    assert.ok(result.includes('echo "hello"'));
    assert.ok(result.includes('<table'));
    assert.ok(result.includes('<td>A</td>'));
    assert.ok(result.includes('<ul>'));
    assert.ok(result.includes('<li>Done</li>'));
  });
});

// ══════════════════════════════════════════════════════════════════════════════
// No markdown: pass-through behaviour
// ══════════════════════════════════════════════════════════════════════════════
describe('No markdown passthrough', () => {
  it('return single line unchanged', () => {
    assert.equal(renderMarkdown('Just a sentence.'), 'Just a sentence.');
  });

  it('converts newlines but not plain words to tags', () => {
    const result = renderMarkdown('Line1\nLine2\nLine3');
    assert.equal(result, 'Line1<br>Line2<br>Line3');
  });
});

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

test('regression: no color swatch is drawn translucent, so a level\'s color is shown as measured', () => {
    // before: extrapolated level swatches had opacity 0.8, which blended the panel into them, and dog-ages' levels
    // 6–8 showed lighter than level 5 though each is darker
    const css = readFileSync(new URL('../styles.css', import.meta.url), 'utf8').replace(/\/\*[\s\S]*?\*\//g, '');
    const rules = [...css.matchAll(/([^{}]+)\{([^}]*)\}/g)].filter(m => /\.swatch\b/.test(m[1]));
    assert.ok(rules.length > 0, 'swatch rules found');
    for (const [, selector, body] of rules)
        assert.ok(!/\bopacity\s*:/.test(body), `${selector.trim()} sets an opacity`);
});

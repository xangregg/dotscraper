import { test } from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, readFileSync } from 'node:fs';
import { IMAGE_DIR, INDEX, indexText } from '../tools/images-index.mjs';

// The test images are local (git ignores them), so a fresh clone has neither the images nor the index.
test('bench-local/index.json matches the images present', { skip: !existsSync(INDEX) && 'no local index' }, () => {
    assert.equal(readFileSync(INDEX, 'utf8'), indexText(IMAGE_DIR), 'run npm run images:index');
});

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createRng } from '../src/core/rng.js';
import { hungarian, matchDots } from '../src/core/metrics.js';

function bruteForceMin(cost) {
    const n = cost.length;
    let best = Infinity;
    const perm = [...Array(n).keys()];
    const permute = k => {
        if (k === n) {
            best = Math.min(best, perm.reduce((s, j, i) => s + cost[i][j], 0));
            return;
        }
        for (let i = k; i < n; i++) {
            [perm[k], perm[i]] = [perm[i], perm[k]];
            permute(k + 1);
            [perm[k], perm[i]] = [perm[i], perm[k]];
        }
    };
    permute(0);
    return best;
}

test('hungarian finds the minimum-cost assignment (against brute force, n ≤ 6)', () => {
    const rng = createRng(5);
    for (let trial = 0; trial < 40; trial++) {
        const n = 1 + Math.floor(rng.uniform() * 6);
        const cost = Array.from({ length: n }, () => Array.from({ length: n }, () => Math.round(100 * rng.uniform())));
        const a = hungarian(cost);
        assert.equal(new Set(a).size, n);
        assert.equal(a.reduce((s, j, i) => s + cost[i][j], 0), bruteForceMin(cost));
    }
});

test('matchDots: perfect, missing and extra dots', () => {
    const truth = [{ x: 10, y: 10 }, { x: 50, y: 10 }, { x: 90, y: 10 }];
    let m = matchDots(truth, truth.map(p => ({ x: p.x + 0.3, y: p.y - 0.4 })), 5);
    assert.equal(m.pairs.length, 3);
    assert.equal(m.precision, 1);
    assert.equal(m.recall, 1);
    assert.ok(Math.abs(m.rmse - 0.5) < 1e-12);
    m = matchDots(truth, [{ x: 10, y: 10 }, { x: 200, y: 200 }], 5);
    assert.deepEqual(m.missed, [1, 2]);
    assert.deepEqual(m.extra, [1]);
    assert.equal(m.precision, 0.5);
    assert.equal(m.recall, 1 / 3);
    assert.equal(m.countError, -1);
});

test('matchDots pairs a dense cluster optimally where greedy nearest-first would not', () => {
    // greedy pairs truth 0 with found 0 (distance 1) and leaves truth 1 unmatched (found 1 is 6 away, beyond tol);
    // optimal pairs truth 0–found 1 (3) and truth 1–found 0 (3)
    const truth = [{ x: 0, y: 0 }, { x: 4, y: 0 }];
    const found = [{ x: 1, y: 0 }, { x: -3, y: 0 }];
    const m = matchDots(truth, found, 4);
    assert.equal(m.pairs.length, 2);
    assert.equal(m.recall, 1);
});

test('matchDots: coincident truth dots need two found dots', () => {
    const truth = [{ x: 20, y: 20 }, { x: 20, y: 20 }];
    assert.equal(matchDots(truth, [{ x: 20.5, y: 20 }], 5).recall, 0.5);
    assert.equal(matchDots(truth, [{ x: 20.5, y: 20 }, { x: 19.5, y: 20 }], 5).recall, 1);
});

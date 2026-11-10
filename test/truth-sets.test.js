// Truth sets: one data set per directory (data.csv in data units) rendered by JMP in many styles, with the plot
// area in the header (bench1: 100 dots, bench2: 204, bench3: 204 as a beeswarm). Each set is skipped when absent.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { existsSync } from 'node:fs';
import { detectAxes } from '../src/core/axes.js';
import { matchDots } from '../src/core/metrics.js';
import { readImage } from '../tools/png.mjs';
import { readTruthCsv, truthPixels, plotAreaMapping } from '../tools/truth-csv.mjs';
import { extractAll } from '../tools/bench.mjs';
import { estimateStyle } from '../src/core/style.js';

const dir = set => new URL(`../${set}/`, import.meta.url).pathname;
const skip = set => !existsSync(dir(set) + 'data.csv') && `${set} absent`;
const FIT_TOL = 0.05; // px RMSE against the plot-area mapping
const TICK_SNAP_TOL = 0.5; // px: JMP draws ticks on whole pixels

function score(set, name) {
    const truth = readTruthCsv(dir(set) + 'data.csv');
    const px = truthPixels(truth);
    const results = extractAll(readImage(dir(set) + name), /\.jpg$/.test(name));
    const found = results.flatMap(r => r.fit.dots);
    return { truth, px, results, found, m: matchDots(px.dots, found, 1) };
}

test('truth sets: rows, coincident dots, plot areas, lanes', { skip: skip('bench1') || skip('bench2') || skip('bench3') }, () => {
    const [b1, b2, b3] = ['bench1', 'bench2', 'bench3'].map(s => readTruthCsv(dir(s) + 'data.csv'));
    assert.equal(b1.rows.length, 100);
    const keys = b1.rows.map(r => `${r.x},${r.y}`);
    assert.equal(keys.length - new Set(keys).size, 1);
    assert.equal(b2.rows.length, 204);
    assert.deepEqual(b1.plotArea, { left: 70, top: 30, width: 546, height: 436 });
    assert.equal(b1.lanes || b2.lanes, false);
    // bench3: x is the group, ticks 1 and 2, and its plot area is 6 px taller
    assert.equal(b3.lanes, true);
    assert.equal(b3.plotArea.height, 442);
    assert.deepEqual(truthPixels(b3).laneCenters, [206.75, 479.25]);
});

test('truth sets: detected ticks fall on the plot-area mapping', { skip: skip('bench1') || skip('bench3') }, () => {
    for (const [set, name] of [['bench1', 'tiny-dots.png'], ['bench3', 'dots-swarm.png']]) {
        const truth = readTruthCsv(dir(set) + 'data.csv');
        const map = plotAreaMapping(truth);
        const ax = detectAxes(readImage(dir(set) + name));
        const labeled = ax.y.ticks.filter(t => t.label).map(t => t.pos);
        const want = [...truth.yTicks].sort((a, b) => b - a).map(map.y);
        assert.equal(labeled.length, want.length, `${set}/${name} y`);
        labeled.forEach((p, i) => assert.ok(Math.abs(p - want[i]) <= TICK_SNAP_TOL, `${set}/${name} y tick ${p} vs ${want[i]}`));
        if (!truth.lanes) {
            const xs = ax.x.ticks.map(t => t.pos);
            assert.equal(xs.length, truth.xTicks.length);
            xs.forEach((p, i) => assert.ok(Math.abs(p - map.x(truth.xTicks[i])) <= TICK_SNAP_TOL, `${set}/${name} x tick ${p}`));
        }
    }
});

test('bench1 tiny and small dots: every dot, the coincident pair too', { skip: skip('bench1') }, () => {
    for (const name of ['tiny-dots.png', 'small-dots.png']) {
        const { found, m } = score('bench1', name);
        assert.equal(found.length, 100, name);
        assert.equal(m.recall, 1, name);
        assert.ok(m.rmse < FIT_TOL, `${name}: rmse ${m.rmse}`);
    }
});

test('bench1 smoother: no dots on the black text of a 50% black family', { skip: skip('bench1') }, () => {
    const { found, m } = score('bench1', 'smoother.png');
    assert.equal(found.length, 100);
    assert.equal(m.recall, 1);
});

test('regression: JPEG dots are found (chroma restored, noise-tolerant thresholds)', { skip: skip('bench1') }, () => {
    // before: "No dot family found" (6 of 3246 pixels inside isolated dots had a flat 3×3 neighborhood)
    assert.equal(estimateStyle(readImage(dir('bench1') + 'medium-dots.jpg')).style, null);
    const { found, m, results } = score('bench1', 'medium-dots.jpg');
    assert.equal(results.length, 1);
    assert.ok(Math.abs(results[0].est.style.alpha - 0.5) < 0.02, `α ${results[0].est.style.alpha}`);
    assert.equal(found.length, 100);
    assert.equal(m.recall, 1);
    assert.ok(m.rmse < 2 * FIT_TOL, `rmse ${m.rmse}`);
});

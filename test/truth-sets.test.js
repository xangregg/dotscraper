// Truth sets: one data set per directory (data.csv in data units) rendered by JMP in many styles, with the plot
// area in the header (bench1: 100 dots, bench2: 204, bench3: 204 as a beeswarm). Each set is skipped when absent.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, mkdtempSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { detectAxes } from '../src/core/axes.js';
import { matchDots } from '../src/core/metrics.js';
import { readImage } from '../tools/png.mjs';
import { laneCentersFromDots, readTruthCsv, truthPixels, truthPixelsByTicks, plotAreaMapping } from '../tools/truth-csv.mjs';
import { extractAll } from '../src/core/extract.js';
import { laneRow } from '../tools/bench.mjs';
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

test('regression: label glyphs below the x axis aren’t dots (bench2 tiny-dots: the zeros of "100")', { skip: skip('bench2') }, () => {
    // before: 207 found, the 3 extras at (614.7, 482.5), (615.1, 484.3), (623.6, 482.9), 17 px below the axis
    const { found, m } = score('bench2', 'tiny-dots.png');
    assert.equal(found.length, 204);
    assert.equal(m.recall, 1);
});

test('truth CSV: category-name ticks become lanes 1, 2, … in the listed order', () => {
    const path = join(mkdtempSync(join(tmpdir(), 'truth-')), 'named.csv');
    writeFileSync(path, '# x ticks: early, late\n# y ticks: 0, 50, 100\n# x scale: linear, with jitter\n# y scale: linear\n'
        + 'group,speed\nlate,37.5\nearly,31.4\n');
    const truth = readTruthCsv(path);
    assert.deepEqual(truth.xTicks, [1, 2]);
    assert.deepEqual(truth.yTicks, [0, 50, 100]);
    assert.deepEqual(truth.rows.map(r => [r.x, r.y]), [[2, 37.5], [1, 31.4]]);
    assert.equal(truth.lanes, true);
    assert.equal(truth.xScale.jitter, true);
    writeFileSync(path, '# x ticks: early, late\n# y ticks: 0, 100\ngroup,speed\nmiddle,3\n');
    assert.throws(() => readTruthCsv(path), /"middle" isn't one of the tick names/);
});

test('truth CSV by ticks: the header pairs with the labeled ticks when minor ticks are detected too', () => {
    const path = join(mkdtempSync(join(tmpdir(), 'truth-')), 'minor.csv');
    writeFileSync(path, '# x ticks: 0, 10\n# y ticks: 0, 100, 200\n# x scale: linear\n# y scale: linear\nx,y\n5,100\n');
    const truth = readTruthCsv(path);
    const tick = (pos, label) => ({ pos, label: label ? [0, 0, 1, 1] : null });
    const axes = {
        x: { ticks: [tick(100, true), tick(200, true)] },
        y: { ticks: [tick(50, true), tick(100, false), tick(150, true), tick(200, false), tick(250, true)] },
    };
    const px = truthPixelsByTicks(truth, axes);
    assert.deepEqual([px.dots[0].x, px.dots[0].y], [150, 150]);
    // neither all ticks nor the labeled ones match the header's count
    axes.y.ticks[0].label = null;
    assert.match(truthPixelsByTicks(truth, axes).error, /ticks detected/);
});

test('truth CSV by ticks: ticks as printed (1K, 10M), log scales, and each row\'s group', () => {
    // before (wages2025): "1K" was a category name, "log" no scale, and the group was dropped, so the bench pooled
    // every family
    const path = join(mkdtempSync(join(tmpdir(), 'truth-')), 'log.csv');
    writeFileSync(path, '# x ticks: 100, 1K, 10K\n# y ticks: 10, 20, 40\n# x scale: log\n# y scale: log\nn,wage,group\n1000,20,3\n316.2,28.28,5\n');
    const truth = readTruthCsv(path);
    assert.deepEqual(truth.xTicks, [100, 1000, 10000]);
    const tick = pos => ({ pos, label: [0, 0, 1, 1] });
    const px = truthPixelsByTicks(truth, { x: { ticks: [tick(100), tick(200), tick(300)] }, y: { ticks: [tick(500), tick(400), tick(300)] } });
    assert.deepEqual(px.dots.map(d => d.group), ['3', '5']);
    assert.deepEqual([px.dots[0].x, px.dots[0].y], [200, 400]);
    // halfway in the log between ticks
    assert.ok(Math.abs(px.dots[1].x - 150) < 0.01 && Math.abs(px.dots[1].y - 350) < 0.05, `${px.dots[1].x}, ${px.dots[1].y}`);
});

test('truth CSV by ticks: lane data without x ticks takes its lanes from the fitted dots', () => {
    const path = join(mkdtempSync(join(tmpdir(), 'truth-')), 'lanes.csv');
    // the first row is lane 2 (hair1): lanes still go left to right
    writeFileSync(path, '# x ticks: 1, 2\n# y ticks: 0, 10\n# x scale: linear, with jitter\n# y scale: linear\nx,y\n2,5\n1,5\n1,8\n');
    const truth = readTruthCsv(path);
    const px = truthPixelsByTicks(truth, { x: null, y: { ticks: [{ pos: 300, label: [] }, { pos: 100, label: [] }] } });
    assert.equal(px.lanesFromDots, true);
    assert.equal(px.laneCenters, null);
    assert.ok(px.dots.every(d => Number.isNaN(d.x)));
    // two jittered strips, one stray dot each side of the gap between them
    const xs = [48, 50, 53, 55, 51, 148, 150, 152, 155];
    assert.deepEqual(laneCentersFromDots(xs, 2), [51, 151]);
    assert.equal(laneCentersFromDots([10], 2), null);
    // lane 1 (left, x ≈ 51) has dots at y 5 and 8 (px 200, 140); lane 2 at y 5
    const dot = (x, y) => ({ x, y, flags: [] });
    const results = [{ fit: { radius: 4, dots: [dot(50, 200), dot(52, 140), dot(150, 200)] } }];
    const r = laneRow('lanes', px.dots, laneCentersFromDots(results[0].fit.dots.map(d => d.x), 2), results, 0);
    assert.equal(r.precision, 1);
    assert.equal(r.recall, 1);
});

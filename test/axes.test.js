import { test } from 'node:test';
import assert from 'node:assert/strict';
import { existsSync } from 'node:fs';
import { createRng } from '../src/core/rng.js';
import { detectAxes, fitScale, toData, toPixel, resolution, guessValues, plotBounds } from '../src/core/axes.js';
import { synthGraph, scatterPoints, labelBlocks } from '../src/core/synth.js';
import { readPng } from '../tools/png.mjs';

const near = (a, b, tol, what) => assert.ok(Math.abs(a - b) <= tol, `${what}: ${a} vs ${b} (±${tol})`);

function plot(extra = {}) {
    const rng = createRng(31);
    return {
        width: 420, height: 340, background: [255, 255, 255], blur: 0.5,
        families: [{ color: [50, 161, 73], alpha: 0.3, radius: 9, points: scatterPoints(rng, { n: 40, box: [80, 30, 400, 270] }) }],
        axes: {
            x: { pos: 290.5, from: 59, to: 405, ticks: [80, 140, 200, 260, 320, 380] },
            y: { pos: 60.5, from: 20, to: 291.5, ticks: [270, 220, 170, 120, 70] },
        },
        ...extra,
    };
}

test('synthetic L axes: lines, ticks (0.2 px) and label boxes are found', () => {
    const spec = plot();
    const { image } = synthGraph(spec);
    const ax = detectAxes(image);
    near(ax.x.line.pos, 290.5, 0.2, 'x axis');
    near(ax.y.line.pos, 60.5, 0.2, 'y axis');
    assert.equal(ax.x.ticks.length, 6);
    assert.equal(ax.y.ticks.length, 5);
    spec.axes.x.ticks.forEach((t, i) => near(ax.x.ticks[i].pos, t, 0.2, 'x tick'));
    [...spec.axes.y.ticks].sort((a, b) => a - b).forEach((t, i) => near(ax.y.ticks[i].pos, t, 0.2, 'y tick'));
    assert.ok(ax.x.regular && ax.y.regular);
    near(ax.x.spacing, 60, 0.05, 'x spacing');
    // every tick's label box covers its block (within 2 px of antialiasing and blur)
    const blocks = labelBlocks(spec.axes);
    for (const t of [...ax.x.ticks, ...ax.y.ticks]) {
        assert.ok(t.label, `tick at ${t.pos} has a label`);
        const b = blocks.find(k => k[0] - 2 <= t.label[0] && t.label[2] <= k[2] + 2 && k[1] - 2 <= t.label[1] && t.label[3] <= k[3] + 2);
        assert.ok(b, `label ${t.label} matches a block`);
    }
});

test('regression: mid-gray axes and ticks (JMP draws them at luminance 165) are axes, not gridlines', () => {
    // crisp one-pixel lines, as JMP draws them (blurred, the line would be lighter than JMP's)
    const spec = plot({ blur: 0 });
    const centered = a => ({ ...a, ticks: a.ticks.map(t => t + 0.5) });
    spec.axes = { ...spec.axes, color: [165, 165, 165], width: 1, x: centered(spec.axes.x), y: centered(spec.axes.y) };
    const ax = detectAxes(synthGraph(spec).image);
    assert.ok(ax.x?.line && ax.y?.line, 'both axis lines');
    assert.equal(ax.x.ticks.length, 6);
    assert.equal(ax.y.ticks.length, 5);
    assert.ok([...ax.x.ticks, ...ax.y.ticks].every(t => t.label));
});

test('no y axis line: light gridlines and a dark baseline in step with them stand in as y ticks', () => {
    const spec = plot({
        gridlines: [100, 160, 220].map(y => ({ points: [[40, y], [410, y]], width: 2, color: [225, 225, 225] })),
        axes: { x: { pos: 280, from: 40, to: 410, ticks: [80, 160, 240, 320, 400] } },
    });
    const ax = detectAxes(synthGraph(spec).image);
    assert.ok(ax.y, 'y ticks found');
    assert.deepEqual(ax.y.ticks.map(t => t.source), ['grid', 'grid', 'grid', 'grid']);
    [100, 160, 220, 280].forEach((y, i) => near(ax.y.ticks[i].pos, y, 0.3, 'gridline'));
});

test('fitScale, toData, toPixel and resolution: linear and log', () => {
    const lin = fitScale([{ pos: 100, value: 0 }, { pos: 200, value: 5 }, { pos: 300, value: 10 }], 'linear');
    near(lin.b, 0.05, 1e-12, 'slope');
    near(toData(lin, 250), 7.5, 1e-12, 'toData');
    near(toPixel(lin, 7.5), 250, 1e-9, 'toPixel');
    near(resolution(lin, 0), 0.05, 1e-12, 'resolution');
    assert.ok(lin.maxResidualPx < 1e-9);
    // a mistyped value shows up as a residual in px
    const bad = fitScale([{ pos: 100, value: 0 }, { pos: 200, value: 5 }, { pos: 300, value: 12 }], 'linear');
    assert.ok(bad.maxResidualPx > 10);
    const log = fitScale([{ pos: 400, value: 1 }, { pos: 300, value: 10 }, { pos: 200, value: 100 }], 'log');
    near(toData(log, 350), Math.sqrt(10), 1e-9, 'log midpoint');
    near(resolution(log, 300), 10 * Math.LN10 / 100, 1e-9, 'log resolution');
    assert.equal(fitScale([{ pos: 1, value: 3 }], 'linear'), null);
    assert.equal(fitScale([{ pos: 1, value: -1 }, { pos: 2, value: 10 }], 'log'), null);
});

test('guessValues fills the other ticks at the typed precision', () => {
    const ticks = [{ pos: 850.1, value: 0, text: '0' }, { pos: 702.7, value: 20, text: '20' }, { pos: 556.4, value: null }, { pos: 112.1, value: null }];
    assert.deepEqual(guessValues(ticks, 'linear'), [null, null, 40, 100]);
    const dec = [{ pos: 0, value: 3.1, text: '3.10' }, { pos: 10, value: 3.2, text: '3.20' }, { pos: 20, value: null }];
    assert.deepEqual(guessValues(dec, 'linear'), [null, null, 3.3]);
    const log = [{ pos: 0, value: 64, text: '64' }, { pos: 10, value: 32, text: '32' }, { pos: 30, value: null }];
    assert.deepEqual(guessValues(log, 'log'), [null, null, 8]);
});

const TESTGRAPHS = new URL('../testgraphs/', import.meta.url).pathname;
const has = name => existsSync(TESTGRAPHS + name);

test('dog-ages: 6 regular ticks per axis, with labels', { skip: !has('dog-ages.png') && 'local test image absent' }, () => {
    const ax = detectAxes(readPng(TESTGRAPHS + 'dog-ages.png'));
    assert.equal(ax.x.ticks.length, 6);
    assert.equal(ax.y.ticks.length, 6);
    assert.ok(ax.x.regular && ax.y.regular);
    near(ax.x.spacing, 88.94, 0.1, 'x spacing');
    near(ax.y.spacing, 147.6, 0.1, 'y spacing');
    assert.ok([...ax.x.ticks, ...ax.y.ticks].every(t => t.label));
    // the isolated dot at (714.1, 508.0) px is at about (14.5, 46.4) in data
    const x = fitScale(ax.x.ticks.map((t, i) => ({ pos: t.pos, value: 3 * i })), 'linear');
    const y = fitScale([...ax.y.ticks].reverse().map((t, i) => ({ pos: t.pos, value: 20 * i })), 'linear');
    near(toData(x, 714.1), 14.48, 0.02, 'x data');
    near(toData(y, 508.0), 46.40, 0.05, 'y data');
});

test('regressions on the other test images', { skip: !has('pathogen-trade-time-scatter.png') && 'local test images absent' }, () => {
    // light gridline stubs inside pathogen's frame are not ticks; its x labels are separate words
    const p = detectAxes(readPng(TESTGRAPHS + 'pathogen-trade-time-scatter.png'));
    assert.equal(p.x.ticks.length, 5);
    assert.ok(p.x.ticks.every(t => t.label && t.label[2] - t.label[0] < 60));
    // Laser's zero line crosses its y axis and is the 0 tick
    const l = detectAxes(readPng(TESTGRAPHS + 'Laser-Asyllable-Maturity.png'));
    assert.equal(l.y.ticks.length, 6);
    // economist: no y axis line; three gridlines and the zero baseline, each labeled
    const e = detectAxes(readPng(TESTGRAPHS + 'economist-canada-unfriendly-chart.png'));
    assert.equal(e.y.ticks.length, 4);
    assert.ok(e.y.ticks.every(t => t.source === 'grid' && t.label));
    // all four labels are the right-aligned numbers, not a dot rim at the zero line's left end
    assert.ok(e.y.ticks.every(t => t.label[0] > 700), JSON.stringify(e.y.ticks.map(t => t.label)));
    // its baseline bounds the plot below: the dots above it, labeling 3 ticks against its 2 year labels below, once
    // made it the plot's top and masked every dot
    assert.deepEqual(plotBounds(e), { left: null, right: null, top: null, bottom: e.x.line.pos });
    // thin antialiased axes count as dark (cantons, beeswarm-boxplot)
    assert.equal(detectAxes(readPng(TESTGRAPHS + 'cantons-haemocytes-dotplot.png')).y.ticks.length, 5);
    const bb = detectAxes(readPng(TESTGRAPHS + 'beeswarm-boxplot.png'));
    assert.equal(bb.y.ticks.length, 9);
    // its gray significance bracket (luminance 160) has three unlabeled "ticks" but is not an x axis
    assert.equal(bb.x, null);
});

test('regression: the native-size dog-ages "100" label keeps its "1"', { skip: !has('dog-ages-actual-size.png') && 'local test image absent' }, () => {
    const ax = detectAxes(readPng(TESTGRAPHS + 'dog-ages-actual-size.png'));
    const top = ax.y.ticks[0].label;
    const zero = ax.y.ticks[ax.y.ticks.length - 1].label;
    // three digits against one, in the same font
    assert.ok(top[2] - top[0] > 2.5 * (zero[2] - zero[0]), `100: ${top}, 0: ${zero}`);
});

test('plotBounds: an axis bounds the plot on the side away from its labels; gridline axes and mixed labels don’t', () => {
    const tick = (pos, label) => ({ pos, source: 'tick', label });
    const xBelow = { line: { dir: 'h', pos: 300.5 }, ticks: [tick(80, [70, 310, 90, 322]), tick(140, [130, 310, 150, 322])] };
    const yLeft = { line: { dir: 'v', pos: 60.5 }, ticks: [tick(270, [30, 264, 52, 276]), tick(220, [30, 214, 52, 226])] };
    assert.deepEqual(plotBounds({ x: xBelow, y: yLeft }), { left: 60.5, right: null, top: null, bottom: 300.5 });
    // labels above the x axis (an axis at the top) and right of the y axis
    const xAbove = { line: { dir: 'h', pos: 20.5 }, ticks: [tick(80, [70, 2, 90, 14]), tick(140, [130, 2, 150, 14])] };
    const yRight = { line: { dir: 'v', pos: 400.5 }, ticks: [tick(270, [410, 264, 430, 276]), tick(220, [410, 214, 430, 226])] };
    assert.deepEqual(plotBounds({ x: xAbove, y: yRight }), { left: null, right: 400.5, top: 20.5, bottom: null });
    // labels on both sides, one labeled tick, gridlines, no axis
    const mixed = { line: { dir: 'h', pos: 300.5 }, ticks: [tick(80, [70, 310, 90, 322]), tick(140, [130, 280, 150, 292])] };
    const oneLabel = { line: { dir: 'h', pos: 300.5 }, ticks: [tick(80, [70, 310, 90, 322]), tick(140, null)] };
    const grid = { line: { dir: 'v', pos: 60.5 }, ticks: [{ pos: 270, source: 'grid', label: [30, 264, 52, 276] }, { pos: 220, source: 'grid', label: [30, 214, 52, 226] }] };
    const none = { left: null, right: null, top: null, bottom: null };
    assert.deepEqual(plotBounds({ x: mixed, y: grid }), none);
    assert.deepEqual(plotBounds({ x: oneLabel, y: null }), none);
});

test('regression: ticks drawn into the plot have their labels outside it, and dots by the ticks aren\'t labels', () => {
    // before (crystal1, Nature's frames): labels were sought on the ticks' side only, so dots near two y ticks
    // became their "labels", the plot was bounded on the wrong side and every dot was masked
    const spec = plot({ blur: 0 });
    spec.axes = { ...spec.axes, inward: true };
    const dot = (x, y) => ({ x, y });
    spec.families = [{ color: [104, 93, 184], alpha: 1, radius: 9, points: [dot(84, 270), dot(82, 170), dot(86, 70), dot(140, 266), dot(260, 268), dot(380, 265)] }];
    const ax = detectAxes(synthGraph(spec).image);
    assert.equal(ax.x.ticks.length, 6);
    assert.equal(ax.y.ticks.length, 5);
    const blocks = labelBlocks(spec.axes);
    for (const t of [...ax.x.ticks, ...ax.y.ticks]) {
        assert.ok(t.label, `tick at ${t.pos} has a label`);
        assert.ok(blocks.some(k => k[0] - 2 <= t.label[0] && t.label[2] <= k[2] + 2 && k[1] - 2 <= t.label[1] && t.label[3] <= k[3] + 2), `label ${t.label} is a label block`);
    }
    const bounds = plotBounds(ax);
    near(bounds.left, 60.5, 0.2, 'left bound');
    near(bounds.bottom, 290.5, 0.2, 'bottom bound');
    assert.equal(bounds.right, null);
    assert.equal(bounds.top, null);
});

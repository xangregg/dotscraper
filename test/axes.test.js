import { test } from 'node:test';
import assert from 'node:assert/strict';
import { existsSync } from 'node:fs';
import { createRng } from '../src/core/rng.js';
import { detectAxes, fitScale, toData, toPixel, resolution, guessValues, plotBounds } from '../src/core/axes.js';
import { synthGraph, scatterPoints, labelBlocks } from '../src/core/synth.js';
import { readPng, readImage } from '../tools/png.mjs';

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

test('regression: dark navy ticks and labels (wages2025: 24, 37, 50, chroma 26) are axes', () => {
    // before: gray ink was chroma 24 or less, so no tick was found and the gridlines stood in, unlabeled (crisp, as
    // wages2025's: blurred, the edges' mixes with white are gray enough)
    const spec = plot({ blur: 0 });
    const centered = a => ({ ...a, ticks: a.ticks.map(t => t + 0.5) });
    spec.axes = { ...spec.axes, color: [24, 37, 50], width: 1, x: centered(spec.axes.x), y: centered(spec.axes.y) };
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

test('regression: wages2025, navy ticks on gray axis lines, and the x axis\'s corner label isn\'t a y label', { skip: !has('wages2025.png') && 'local test image absent' }, () => {
    // before: no tick on either axis (navy, chroma 26); then the y axis's labels were all dropped: the x label "100",
    // under the corner, reached into their search beside the corner tick at gap 0, and they lay 19 px out
    const ax = detectAxes(readPng(new URL('../testgraphs/wages2025.png', import.meta.url).pathname));
    for (const k of ['x', 'y']) {
        assert.ok(ax[k]?.line, `${k} axis line`);
        assert.equal(ax[k].ticks.filter(t => t.label).length, 6, `${k}: ${ax[k].ticks.map(t => t.pos.toFixed(0))}`);
    }
    assert.ok(ax.y.ticks.every(t => t.label[2] < ax.y.line.pos), 'y labels left of the y axis');
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

// A white image with a y axis whose bottom tick is hidden by a baseline running right from the axis's end (bench-nature
// cells1: bars standing on a line from the y axis's "0"), every tick labeled by a block left of it.
function cornerImage(baselineAtEnd) {
    const width = 200;
    const height = 260;
    const data = new Uint8ClampedArray(4 * width * height).fill(255);
    const rect = (x0, y0, x1, y1) => {
        for (let y = y0; y < y1; y++) {
            for (let x = x0; x < x1; x++)
                data.fill(0, 4 * (y * width + x), 4 * (y * width + x) + 3);
        }
    };
    const axisX = 60;
    const ticks = [40, 90, 140, 190];
    const end = baselineAtEnd ? 240 : 228;
    rect(axisX, 30, axisX + 2, end + 1);
    for (const t of ticks)
        rect(axisX - 6, t, axisX, t + 2);
    rect(axisX, 240, 190, 242); // the baseline, 2 px thick, from the axis to the right
    for (const t of [...ticks, 240])
        rect(30, t - 4, 44, t + 6);
    return { width, height, data };
}

test('regression: an exponent is of its label\'s box, which is its own tick\'s and not the minor tick\'s before it', () => {
    // before (primates1's log x axis): "10" was boxed without its raised 2, and that box's center was nearer the
    // minor tick 8 px before the labeled one. Blocks stand in for the number and, raised after it, its exponent,
    // the two together centered on the tick.
    const spec = plot({ families: [], blur: 0 });
    spec.axes = { ...spec.axes, x: { ...spec.axes.x, ticks: [112, 120, 232, 240, 352, 360] }, label: [0, 0] };
    const { image } = synthGraph(spec);
    const rect = (x0, y0, x1, y1) => {
        for (let y = y0; y < y1; y++) {
            for (let x = x0; x < x1; x++)
                image.data.fill(0, 4 * (y * image.width + x), 4 * (y * image.width + x) + 3);
        }
    };
    for (const t of [120, 240, 360]) {
        rect(t - 18, 308, t + 6, 324);
        rect(t + 9, 302, t + 18, 313);
    }
    for (const t of spec.axes.y.ticks)
        rect(30, t - 5, 44, t + 5);
    const ax = detectAxes(image);
    assert.deepEqual(ax.x.ticks.map(t => t.label), [null, [102, 302, 138, 324], null, [222, 302, 258, 324], null, [342, 302, 378, 324]]);
});

test('regression: a label above and beside a taller glyph is not its exponent', () => {
    // pathogen: the y label "32" lies right of a part of the rotated axis title and wholly above it; taken for its
    // exponent it joined the title, and its tick lost its label
    const image = cornerImage(true);
    for (let y = 98; y < 140; y++) {
        for (let x = 10; x < 24; x++)
            image.data.fill(0, 4 * (y * image.width + x), 4 * (y * image.width + x) + 3);
    }
    const ax = detectAxes(image);
    assert.deepEqual(ax.y.ticks.map(t => t.label), [40, 90, 140, 190, 240].map(t => [30, t - 4, 44, t + 6]));
});

test('a baseline from the end of the y axis is its bottom tick when it continues the spacing', () => {
    const ax = detectAxes(cornerImage(true));
    assert.equal(ax.y.ticks.length, 5, JSON.stringify(ax.y.ticks.map(t => t.pos)));
    near(ax.y.ticks[4].pos, 241, 0.5, 'corner tick');
    assert.ok(ax.y.ticks.every(t => t.label));
    assert.ok(ax.y.regular);
    // an axis ending short of the baseline doesn't get it
    assert.equal(detectAxes(cornerImage(false)).y.ticks.length, 4);
});

const NATURE = new URL('../bench-nature/', import.meta.url).pathname;
const CELLS1 = 'cells1-10.1038-s41586-026-10946-1.png';

test('regression: cells1, the "0" tick hidden by the bars\' baseline is found and labeled', { skip: !existsSync(NATURE + CELLS1) && 'bench-nature cells1 absent' }, () => {
    const ax = detectAxes(readPng(NATURE + CELLS1));
    const labeled = ax.y.ticks.filter(t => t.label);
    assert.equal(labeled.length, 4, JSON.stringify(ax.y.ticks.map(t => [t.pos, !!t.label])));
    near(labeled[3].pos, 1031.8, 0.5, '0 tick');
});

const PLOS = new URL('../bench-plos/', import.meta.url).pathname;
const BOXDOTS1 = 'boxdots1-10.1371-journal.pone.0361404.png';

test('regression: boxdots1, a filled box\'s edges with dots along them are no x axis', { skip: !existsSync(PLOS + BOXDOTS1) && 'bench-plos boxdots1 absent' }, () => {
    // before: the green box's lower edge (y 618.5) was the x axis, its "ticks" dots touching it (4 of 14 px at their
    // sides) and its "labels" dots; without that rule its other edge (y 640.3), whose dot "labels" lie beyond the
    // box's fill. The plot's top was set there and every dot above it masked. The figure's tick labels are cropped.
    const ax = detectAxes(readPng(PLOS + BOXDOTS1));
    assert.equal(ax.x, null, `x axis at ${ax.x?.line?.pos}`);
    assert.deepEqual(plotBounds(ax), { left: null, right: null, top: null, bottom: null });
});

// a graph with two columns of dots over an x axis that has labels and no tick marks, and a y axis with ticks
function categoryPlot(xTicks = [150, 310]) {
    const rng = createRng(7);
    return {
        width: 420, height: 360, background: [255, 255, 255], blur: 0.5,
        families: [{
            color: [70, 120, 220], alpha: 1, radius: 5,
            points: xTicks.flatMap(x => scatterPoints(rng, { n: 20, box: [x - 30, 40, x + 30, 270] })),
        }],
        axes: {
            x: { pos: 290.5, from: 59, to: 405, ticks: xTicks, labelsOnly: true },
            // the bottom tick is at the corner: its label lies beside the x axis line's end, beyond the y axis
            y: { pos: 60.5, from: 20, to: 291.5, ticks: [290.5, 230, 170, 110, 50] },
        },
    };
}

// a solid black block, as an axis title's
function block(image, [x0, y0, x1, y1]) {
    for (let y = y0; y < y1; y++) {
        for (let x = x0; x < x1; x++)
            image.data.set([0, 0, 0, 255], 4 * (y * image.width + x));
    }
}

test('an axis with labels and no tick marks: the labels are its positions; a title under them is not', () => {
    const spec = categoryPlot();
    const { image } = synthGraph(spec);
    // the axis's title, a row below the labels
    block(image, [200, 322, 260, 332]);
    const ax = detectAxes(image);
    assert.ok(ax.x, 'an x axis');
    near(ax.x.line.pos, 290.5, 0.5, 'the x axis line');
    assert.deepEqual(ax.x.ticks.map(t => t.source), ['label', 'label']);
    assert.deepEqual(ax.x.ticks.map(t => t.length), [0, 0]);
    ax.x.ticks.forEach((t, i) => near(t.pos, spec.axes.x.ticks[i], 1, `position ${i}`));
    const blocks = labelBlocks(spec.axes);
    for (const t of ax.x.ticks)
        assert.ok(blocks.some(k => k[0] - 2 <= t.label[0] && t.label[2] <= k[2] + 2 && k[1] - 2 <= t.label[1] && t.label[3] <= k[3] + 2), `label box ${t.label}`);
    // the y axis is as it was, its corner label its own
    assert.equal(ax.y.ticks.filter(t => t.label).length, 5);
    assert.ok(ax.y.ticks.every(t => t.source === 'tick'));
    // the line bounds the plot: its labels are below it
    assert.deepEqual(plotBounds(ax), { left: ax.y.line.pos, right: null, top: null, bottom: ax.x.line.pos });
});

test('no category axis from one word under a line, or from a line that isn\'t at the other axis\'s end', () => {
    // one label: an axis title, not a category
    assert.equal(detectAxes(synthGraph(categoryPlot([230])).image).x, null);
    // no label at all
    const bare = categoryPlot();
    bare.axes.label = [0, 0];
    bare.axes.x.ticks = [];
    assert.equal(detectAxes(synthGraph(bare).image).x, null);
    // the labeled line is up in the plot, not at the y axis's end: a bracket or a box's edge with text under it
    const spec = categoryPlot();
    spec.axes.x.pos = 180.5;
    spec.families[0].points = spec.families[0].points.filter(p => p.y < 160);
    assert.equal(detectAxes(synthGraph(spec).image).x, null);
});

const BENCH3 = new URL('../bench3/', import.meta.url).pathname;

test('regression: bench3, the category axis under the swarms has no tick marks', { skip: !existsSync(BENCH3 + 'dots-swarm.png') && 'bench3 absent' }, () => {
    // before: no x axis (an axis needed tick marks), so the plot had no bottom side and the axes panel no x
    for (const name of ['dots-swarm.png', 'dots-packed.jpg', 'dots-swarm-grid.png']) {
        const ax = detectAxes(readImage(BENCH3 + name));
        assert.ok(ax.x, `${name}: an x axis`);
        assert.deepEqual(ax.x.ticks.map(t => t.source), ['label', 'label'], name);
        // the lanes are at 206.5 and 479.5 px (data.csv's plot area and scale); the labels "1" and "2" are centered
        // 1.5 and 1 px left of them, their ink not being centered on their advance
        near(ax.x.ticks[0].pos, 206.5, 2, `${name}: lane 1`);
        near(ax.x.ticks[1].pos, 479.5, 2, `${name}: lane 2`);
        // "group", the title under them, is no third category
        assert.equal(ax.x.ticks.length, 2, name);
        assert.equal(ax.y.ticks.filter(t => t.label).length, 7, `${name}: y labels`);
        near(plotBounds(ax).bottom, 471.5, 0.6, `${name}: the plot's bottom`);
    }
});

test('regression: cells1, the bars\' names under their baseline are the x axis\'s two categories', { skip: !existsSync(NATURE + CELLS1) && 'bench-nature cells1 absent' }, () => {
    const ax = detectAxes(readPng(NATURE + CELLS1));
    assert.deepEqual(ax.x?.ticks.map(t => t.source), ['label', 'label']);
    // "L5PTs" and "GrCs", under the two bars (centered about 430 and 661 px)
    near(ax.x.ticks[0].pos, 430, 6, 'L5PTs');
    near(ax.x.ticks[1].pos, 661, 6, 'GrCs');
});

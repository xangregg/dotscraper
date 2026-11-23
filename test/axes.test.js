import { test } from 'node:test';
import assert from 'node:assert/strict';
import { existsSync } from 'node:fs';
import { createRng } from '../src/core/rng.js';
import { detectAxes, backgroundColors, panelEdges, fitScale, toData, toPixel, resolution, guessValues, plotBounds } from '../src/core/axes.js';
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
    // the gray band under its names is no gridlines in pieces: a row of it is pieces of light ink between the
    // letters, each as thick as the band (three more "lines" at 2175, 2287 and 2297 without the thickness)
    assert.deepEqual(bb.lines.filter(l => l.dir === 'h' && l.pos > 2000).map(l => Math.round(l.pos * 10) / 10), [2276.5, 2298.5, 2331.5]);
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

function fill(image, [x0, y0, x1, y1], color) {
    for (let y = y0; y < y1; y++) {
        for (let x = x0; x < x1; x++)
            image.data.set([...color, 255], 4 * (y * image.width + x));
    }
}

const PANEL = [60, 20, 420, 290]; // px, [x0, y0, x1, y1)
const PANEL_X_TICKS = [120, 200, 280, 360];
const PANEL_Y_TICKS = [70, 130, 190, 250];

// A plot as ggplot2's default theme draws it: a gray panel on a white page, white gridlines across it, no axis
// lines, and dark tick marks 2 px wide leaving the panel's bottom and left edges, with labels beyond them. The row
// under the panel and the column left of it are its color blended with the page's, as an edge between pixels is,
// and a tick's first pixel there is lighter than the rest of it. Dots are red discs.
//
// `apart`: as ggplot2 drew one 7 in wide at 300 dpi. The y ticks stand 2 px off the panel, and the edge's pixel
// between is white, the end of the gridline at the tick; the gridlines have rims between their color and the
// panel's, drawn here in the 40 px beside the left edge; and the x axis has a tick with its label just left of the
// panel's corner.
function panelPlot({ ticks = true, apart = false } = {}) {
    const image = { width: 440, height: 340, data: new Uint8ClampedArray(4 * 440 * 340).fill(255) };
    const [x0, y0, x1, y1] = PANEL;
    fill(image, PANEL, [230, 230, 230]);
    fill(image, [x0, y1, x1, y1 + 1], [244, 244, 244]);
    fill(image, [x0 - 1, y0, x0, y1], [240, 240, 240]);
    for (const x of PANEL_X_TICKS) {
        fill(image, [x - 1, y0, x + 1, y1], [255, 255, 255]);
        if (ticks) {
            fill(image, [x - 1, y1, x + 1, y1 + 1], [194, 194, 194]);
            fill(image, [x - 1, y1 + 1, x + 1, y1 + 6], [51, 51, 51]);
        }
        fill(image, [x - 8, y1 + 10, x + 8, y1 + 20], [77, 77, 77]);
    }
    for (const y of PANEL_Y_TICKS) {
        fill(image, [x0, y - 1, x1, y + 1], [255, 255, 255]);
        if (ticks && apart) {
            fill(image, [x0 - 1, y - 1, x0, y + 1], [255, 255, 255]);
            fill(image, [x0 - 8, y - 1, x0 - 3, y + 1], [51, 51, 51]);
        }
        else if (ticks) {
            fill(image, [x0 - 1, y - 1, x0, y + 1], [158, 158, 158]);
            fill(image, [x0 - 6, y - 1, x0 - 1, y + 1], [51, 51, 51]);
        }
        // (apart: the top tick has no label, so that the page has fewer labels than the panel has rims)
        if (!apart || y !== PANEL_Y_TICKS[0])
            fill(image, [x0 - 30, y - 5, x0 - 12, y + 5], [77, 77, 77]);
    }
    if (apart) {
        for (const y of PANEL_Y_TICKS) {
            fill(image, [x0, y - 2, x0 + 40, y - 1], [242, 242, 242]);
            fill(image, [x0, y + 1, x0 + 40, y + 2], [242, 242, 242]);
        }
        fill(image, [x0 - 4, y1, x0 - 2, y1 + 6], [51, 51, 51]);
        fill(image, [x0 - 11, y1 + 10, x0 + 5, y1 + 20], [77, 77, 77]);
    }
    const rng = createRng(11);
    for (const { x: cx, y: cy } of scatterPoints(rng, { n: 30, box: [x0 + 10, y0 + 10, x1 - 10, y1 - 10] })) {
        for (let y = Math.floor(cy - 5); y <= cy + 5; y++) {
            for (let x = Math.floor(cx - 5); x <= cx + 5; x++) {
                if (Math.hypot(x + 0.5 - cx, y + 0.5 - cy) <= 5)
                    image.data.set([214, 39, 40, 255], 4 * (y * image.width + x));
            }
        }
    }
    return image;
}

test('a gray panel with tick marks at its edges and no axis lines: the edges are the axes', () => {
    const image = panelPlot();
    const ax = detectAxes(image);
    assert.ok(ax.x.line.edge && ax.y.line.edge);
    assert.deepEqual([ax.x.line.pos, ax.x.line.from, ax.x.line.to, ax.x.line.thickness], [290, 60, 420, 0]);
    assert.deepEqual([ax.y.line.pos, ax.y.line.from, ax.y.line.to], [60, 20, 290]);
    assert.deepEqual(ax.x.ticks.map(t => t.source), ['tick', 'tick', 'tick', 'tick']);
    PANEL_X_TICKS.forEach((x, i) => near(ax.x.ticks[i].pos, x, 0.01, 'x tick'));
    PANEL_Y_TICKS.forEach((y, i) => near(ax.y.ticks[i].pos, y, 0.01, 'y tick'));
    assert.deepEqual(ax.x.ticks.map(t => t.label), PANEL_X_TICKS.map(x => [x - 8, 300, x + 8, 310]));
    assert.deepEqual(ax.y.ticks.map(t => t.label), PANEL_Y_TICKS.map(y => [30, y - 5, 48, y + 5]));
    // the panel's edges bound the plot, as axis lines do
    assert.deepEqual(plotBounds(ax), { left: 60, right: null, top: null, bottom: 290 });
    // the edges are no lines of ink
    assert.ok(ax.lines.every(l => !l.edge));
});

test('panelEdges: a filled rectangle that is not the page; not a band, an ellipse or the page itself', () => {
    const image = panelPlot();
    const edges = panelEdges(image, backgroundColors(image));
    assert.deepEqual(edges.map(l => [l.dir, l.pos, l.from, l.to]), [['h', 20, 60, 420], ['h', 290, 60, 420], ['v', 60, 20, 290], ['v', 420, 20, 290]]);
    // an ellipse of one color, a confidence region's fill, as wide as a panel: its first row is a sliver of it
    const ellipse = { width: 440, height: 340, data: new Uint8ClampedArray(4 * 440 * 340).fill(255) };
    for (let y = 0; y < 340; y++) {
        for (let x = 0; x < 440; x++) {
            if (((x - 220) / 180) ** 2 + ((y - 170) / 130) ** 2 <= 1)
                ellipse.data.set([213, 221, 240, 255], 4 * (y * 440 + x));
        }
    }
    assert.equal(backgroundColors(ellipse).length, 2);
    assert.deepEqual(panelEdges(ellipse, backgroundColors(ellipse)), []);
    // a band that fans out from its middle to full height at both ends (a fit's confidence band): two straight
    // sides, its first and last columns, and no straight top or bottom. And the same turned a quarter.
    for (const turned of [false, true]) {
        const fan = { width: 440, height: 340, data: new Uint8ClampedArray(4 * 440 * 340).fill(255) };
        for (let y = 0; y < 340; y++) {
            for (let x = 0; x < 440; x++) {
                const [u, v] = [Math.abs(x - 220) / 180, Math.abs(y - 170) / 130];
                if (u <= 1 && v <= 1 && (turned ? u <= v : v <= u))
                    fan.data.set([205, 213, 232, 255], 4 * (y * 440 + x));
            }
        }
        assert.equal(backgroundColors(fan).length, 2);
        assert.deepEqual(panelEdges(fan, backgroundColors(fan)), [], turned ? 'turned' : 'fan');
    }
    // a page of one color has no panel
    assert.deepEqual(panelEdges(synthGraph(plot()).image, [[255, 255, 255]]), []);
});

test('a panel without tick marks gives no axis, and dark axis lines come before a panel\'s edges', () => {
    // labels alone beside a panel: nothing says where on its edge a value is
    const bare = detectAxes(panelPlot({ ticks: false }));
    assert.ok(!bare.x?.line && !bare.y?.line);
    // a panel inside axis lines of their own: the lines and their ticks leave its edges too, as strokes do
    const spec = plot();
    const { image } = synthGraph(spec);
    const framed = { width: image.width, height: image.height, data: new Uint8ClampedArray(image.data) };
    for (let p = 0; p < framed.width * framed.height; p++) {
        const [x, y] = [p % framed.width, Math.floor(p / framed.width)];
        if (x >= 62 && x < 405 && y >= 20 && y < 289 && framed.data[4 * p] === 255 && framed.data[4 * p + 1] === 255 && framed.data[4 * p + 2] === 255)
            framed.data.set([240, 240, 233, 255], 4 * p);
    }
    assert.equal(panelEdges(framed, backgroundColors(framed)).length, 4);
    const ax = detectAxes(framed);
    assert.ok(!ax.x.line.edge && !ax.y.line.edge);
    near(ax.x.line.pos, 290.5, 0.2, 'x axis');
    near(ax.y.line.pos, 60.5, 0.2, 'y axis');
    assert.equal(ax.x.ticks.length, 6);
});

const BENCH_R = new URL('../bench-r/', import.meta.url).pathname;

test('bench-r: ggplot2\'s default theme at 72, 110 and 300 dpi, the ticks at the gray panel\'s edges', { skip: !existsSync(BENCH_R + 'gg-gray.png') && 'bench-r absent' }, () => {
    for (const [name, xGap, yGap] of [['gg-gray-small', 110.8, 63.6], ['gg-gray', 169.3, 97.2], ['gg-gray-hi', 377.4, 225.9]]) {
        const ax = detectAxes(readImage(`${BENCH_R}${name}.png`));
        assert.ok(ax.x.line.edge && ax.y.line.edge, name);
        // 25, 50, 75 below and 10 to 40 beside: evenly spaced, each with its label
        assert.equal(ax.x.ticks.length, 3, name);
        assert.equal(ax.y.ticks.length, 4, name);
        assert.ok(ax.x.regular && ax.y.regular, name);
        near(ax.x.spacing, xGap, 0.5, `${name} x spacing`);
        near(ax.y.spacing, yGap, 0.5, `${name} y spacing`);
        assert.ok([...ax.x.ticks, ...ax.y.ticks].every(t => t.label && t.source === 'tick'), name);
    }
});

test('regression: ticks that stand off a panel\'s edge, gridline rims inside it, and a tick at its corner', () => {
    // ggplot2's gray theme, 7 in wide at 300 dpi: no y tick touched the left edge, so there was no y axis; with the
    // ticks found, the rims of the gridlines inside the panel were their labels where they outnumbered the page's,
    // and the plot was bounded on the wrong side; and the x tick at the corner, seen in part, was a fifth y tick
    // and the x axis's first, 2 px off its place
    const ax = detectAxes(panelPlot({ apart: true }));
    assert.ok(ax.x.line.edge && ax.y.line.edge);
    assert.equal(ax.y.ticks.length, 4);
    PANEL_Y_TICKS.forEach((y, i) => near(ax.y.ticks[i].pos, y, 0.01, 'y tick'));
    // a tick's length is from the edge: its 5 px and the 3 between
    assert.deepEqual(ax.y.ticks.map(t => t.length), [8, 8, 8, 8]);
    assert.deepEqual(ax.y.ticks.map(t => t.label), PANEL_Y_TICKS.map((y, i) => (i ? [30, y - 5, 48, y + 5] : null)));
    assert.deepEqual(plotBounds(ax), { left: 60, right: null, top: null, bottom: 290 });
    // the corner's tick is the x axis's first, with its label
    assert.equal(ax.x.ticks.length, 5);
    near(ax.x.ticks[0].pos, 57, 0.01, 'corner tick');
    assert.deepEqual(ax.x.ticks[0].label, [49, 300, 65, 310]);
});

const GRID_Y = [40, 110, 180, 250, 320]; // px, the gridlines' rows (each 3 px thick, about these)
const GRID_X = [80, 520]; // px, where the gridlines start and end
const YEARS_X = [110, 205, 300, 395, 490]; // px, the middles of the labels under the last gridline

function disc(image, cx, cy, r, color) {
    for (let y = Math.floor(cy - r); y <= cy + r; y++) {
        for (let x = Math.floor(cx - r); x <= cx + r; x++) {
            if (Math.hypot(x + 0.5 - cx, y + 0.5 - cy) <= r)
                image.data.set([...color, 255], 4 * (y * image.width + x));
        }
    }
}

// A chart with no axis lines and no tick marks, as song-cents and ggplot2's theme_minimal draw them: light gridlines
// with a gray label off the left end of each, level with it; labels in a row under the last gridline; a title over
// the first gridline's start; a dashed orange reference line with an orange label between two gridlines. Blue dots
// with white strokes lie over the gridlines and leave no run of 30% of the width on the second and the fourth; one
// covers the fourth near its start, leaving a 14 px stub before it; and a gray ring sits on the first gridline.
function gridPlot() {
    const image = { width: 560, height: 400, data: new Uint8ClampedArray(4 * 560 * 400).fill(255) };
    const gray = [88, 88, 88];
    for (const y of GRID_Y) {
        fill(image, [GRID_X[0], y - 1, GRID_X[1], y + 2], [229, 229, 229]);
        fill(image, [52, y - 6, 74, y + 7], gray);
    }
    // a light stroke 1 px thick off the third gridline's end, in its middle row: no piece of the line, which is 3
    fill(image, [530, GRID_Y[2], 542, GRID_Y[2] + 1], [229, 229, 229]);
    fill(image, [GRID_X[0], 12, 250, 26], gray);
    for (let x = GRID_X[0]; x < GRID_X[1]; x += 14)
        fill(image, [x, 214, x + 8, 216], [223, 105, 59]);
    fill(image, [60, 209, 74, 222], [223, 105, 59]);
    for (const x of YEARS_X)
        fill(image, [x - 17, 345, x + 17, 358], gray);
    // the ring on the first gridline: gray, 2 px wide
    for (let y = 30; y < 62; y++) {
        for (let x = 140; x < 172; x++) {
            const d = Math.hypot(x + 0.5 - 156, y + 0.5 - 46);
            if (d <= 10 && d > 8)
                image.data.set([119, 119, 106, 255], 4 * (y * image.width + x));
        }
    }
    const dots = [];
    for (const y of [GRID_Y[1], GRID_Y[3]]) {
        for (let x = y === GRID_Y[3] ? 106 : 100; x < GRID_X[1] - 12; x += 60)
            dots.push([x, y + (x % 3) - 1]);
    }
    const rng = createRng(3);
    for (const { x, y } of scatterPoints(rng, { n: 30, box: [100, 50, 500, 310] }))
        dots.push([x, y]);
    for (const [x, y] of dots) {
        disc(image, x, y, 12, [255, 255, 255]);
        disc(image, x, y, 10, [93, 146, 205]);
    }
    return image;
}

test('gridlines under dots, their labels off their ends, and labels under the last one: both axes, no lines or ticks', () => {
    const image = gridPlot();
    const ax = detectAxes(image);
    // every gridline, the two broken by a row of dots too, from its start to its end
    const grid = ax.lines.filter(l => l.dir === 'h' && !l.dark);
    assert.deepEqual(grid.map(l => [Math.round(l.pos * 10) / 10, l.from, l.to]), GRID_Y.map(y => [y + 0.5, GRID_X[0], GRID_X[1]]));
    assert.deepEqual(ax.y.ticks.map(t => t.source), ['grid', 'grid', 'grid', 'grid', 'grid']);
    // each with the label off its end, not the title over the first, the ring on it, the stub of the fourth, or the
    // reference line's own label
    assert.deepEqual(ax.y.ticks.map(t => t.label), GRID_Y.map(y => [52, y - 6, 74, y + 7]));
    // the labels under the last gridline are the x axis's positions
    assert.deepEqual(ax.x.ticks.map(t => [t.pos, t.source]), YEARS_X.map(x => [x, 'label']));
    near(ax.x.line.pos, GRID_Y[4] + 0.5, 0.05, 'the last gridline');
    // the plot ends where the gridlines start, and at the last gridline
    assert.ok(ax.y.line.edge);
    const bounds = plotBounds(ax);
    assert.deepEqual([bounds.left, bounds.right, bounds.top], [GRID_X[0], null, null]);
    near(bounds.bottom, GRID_Y[4] + 0.5, 0.05, 'bottom');
});

test('regression: ink beside a gridline\'s end is no label of it, and gridlines without labels have no labels under them', () => {
    // gridlines with a row of words under the last, and `mark` at each one's start
    const scene = mark => {
        const image = { width: 560, height: 400, data: new Uint8ClampedArray(4 * 560 * 400).fill(255) };
        for (const y of GRID_Y) {
            fill(image, [GRID_X[0], y - 1, GRID_X[1], y + 2], [229, 229, 229]);
            mark(image, y);
        }
        for (const x of YEARS_X)
            fill(image, [x - 17, 345, x + 17, 358], [88, 88, 88]);
        return detectAxes(image);
    };
    const none = { left: null, right: null, top: null, bottom: null };
    // no labels: a slanted light curve is such "gridlines" too, a pixel or two apart (an opaque test's scene: 11 of
    // them), and the dots under the last were an x axis of labels that ended the plot there, above every dot
    const bare = scene(() => {});
    assert.deepEqual(bare.y.ticks.map(t => [t.source, t.label]), GRID_Y.map(() => ['grid', null]));
    assert.equal(bare.x, null);
    assert.deepEqual(plotBounds(bare), none);
    // gray ink off each line's start that begins level with the line and lies below it: a label level with a line
    // has some of itself on either side. In a synthetic swarm two "gridlines" 2 px apart took slivers of a dot for
    // their labels, and the plot ended at the dot.
    const beside = scene((image, y) => fill(image, [60, y, 76, y + 7], [88, 88, 88]));
    assert.deepEqual(beside.y.ticks.map(t => t.label), GRID_Y.map(() => null));
    assert.equal(beside.x, null);
    assert.deepEqual(plotBounds(beside), none);
    // the same ink a few rows higher is the lines' labels, and the words under the last line the x axis's
    const level = scene((image, y) => fill(image, [60, y - 3, 76, y + 4], [88, 88, 88]));
    assert.deepEqual(level.y.ticks.map(t => t.label), GRID_Y.map(y => [60, y - 3, 76, y + 4]));
    assert.deepEqual(level.x.ticks.map(t => t.pos), YEARS_X);
    assert.equal(plotBounds(level).left, GRID_X[0]);
    // labels over the lines' far ends, along them (economist): the gridlines' labels, and no end of the plot
    const along = scene((image, y) => fill(image, [GRID_X[1] - 30, y - 17, GRID_X[1] - 4, y - 4], [88, 88, 88]));
    assert.deepEqual(along.y.ticks.map(t => t.label), GRID_Y.map(y => [GRID_X[1] - 30, y - 17, GRID_X[1] - 4, y - 4]));
    assert.equal(along.y.line, null);
    assert.deepEqual(plotBounds(along), none);
});

test('regression: a row through light gray dots is no line in pieces', () => {
    // nicest language: rows through its packed gray dots are pieces of light ink too, chords that fill most of
    // their span, and three of them were "gridlines" and a y axis. A chord's ink across is as tall as the dot there.
    const image = { width: 560, height: 300, data: new Uint8ClampedArray(4 * 560 * 300).fill(255) };
    for (let x = 60; x < 500; x += 22)
        disc(image, x, 150 + (x % 5) - 2, 8, [200, 200, 200]);
    assert.deepEqual(detectAxes(image).lines, []);
    // … in touching pairs, in a larger image, where a line may be as thick as the dots are: the ink across is not
    // of one thickness along a pair's chord
    const large = { width: 1500, height: 1000, data: new Uint8ClampedArray(4 * 1500 * 1000).fill(255) };
    for (let x = 200; x < 1300; x += 42) {
        disc(large, x, 500, 9, [200, 200, 200]);
        disc(large, x + 18, 500, 9, [200, 200, 200]);
    }
    assert.deepEqual(detectAxes(large).lines, []);
});

test('a light line in pieces is thin and fills its span; a dark line under dots keeps its run', () => {
    const blank = () => ({ width: 560, height: 300, data: new Uint8ClampedArray(4 * 560 * 300).fill(255) });
    // a light gray ellipse in stripes: every row is long pieces of light ink, each as thick as the ellipse there
    // (and the ellipse under 5% of the image, or its color would be a background's and no ink)
    const region = blank();
    for (let y = 0; y < 300; y++) {
        for (let x = 0; x < 560; x++) {
            if (((x - 280) / 220) ** 2 + ((y - 150) / 14) ** 2 <= 1 && x % 40 >= 8)
                region.data.set([215, 215, 215, 255], 4 * (y * 560 + x));
        }
    }
    assert.deepEqual(detectAxes(region).lines, []);
    // a slanted light band in stripes: its rows are pieces that start and end 12 px farther along from one row to
    // the next, a few rows alike at a time, and each piece as thick as the band
    const band = blank();
    for (let y = 130; y < 150; y++) {
        const x0 = 40 + 12 * (y - 130);
        for (let x = x0; x < x0 + 280; x++) {
            if (x % 45 < 39)
                band.data.set([215, 215, 215, 255], 4 * (y * 560 + x));
        }
    }
    assert.deepEqual(detectAxes(band).lines.filter(l => l.dir === 'h'), []);
    // a light bar 30 px thick, in stripes of 29 px, too short for pieces, but for three rows in its middle, where
    // every second gap is filled: those rows' pieces are flat and fill their span, and the ink across them is the
    // bar's, thicker than a line (beeswarm-boxplot: rows of the box's light band, between the dots over it). The
    // image is tall enough that the bar is under 5% of it, and no background.
    const bar = { width: 1500, height: 700, data: new Uint8ClampedArray(4 * 1500 * 700).fill(255) };
    for (let x = 100; x < 1400; x++) {
        if (x % 33 < 29)
            fill(bar, [x, 140, x + 1, 170], [215, 215, 215]);
        else if (x % 66 < 33)
            fill(bar, [x, 154, x + 1, 157], [215, 215, 215]);
    }
    assert.deepEqual(detectAxes(bar).lines, []);
    // two short light rules in one row, far apart: enough ink between them for a line, over little of their span
    const rules = blank();
    fill(rules, [40, 149, 130, 152], [229, 229, 229]);
    fill(rules, [430, 149, 520, 152], [229, 229, 229]);
    assert.deepEqual(detectAxes(rules).lines, []);
    // a dark line with light edges, dots over it: it is its longest run, as before, whatever its light edges' pieces
    // add up to
    const dark = blank();
    fill(dark, [40, 149, 520, 152], [200, 200, 200]);
    fill(dark, [40, 150, 520, 151], [60, 60, 60]);
    for (const x of [70, 120, 170, 420, 470])
        disc(dark, x, 150, 10, [93, 146, 205]);
    const lines = detectAxes(dark).lines;
    assert.deepEqual(lines.map(l => [l.dir, l.from, l.to, l.thickness, l.dark]), [['h', 180, 410, 3, true]]);
});

const GRADER1 = 'grader1-10.1371-journal.pone.0361887.png';

test('regression: grader1, gridlines both ways with their labels off their ends (theme_minimal)', { skip: !existsSync(PLOS + GRADER1) && 'bench-plos grader1 absent' }, () => {
    // before: the labels were scraps (a gridline's label was looked for above or below the line), and no axis had
    // values. Each gridline has its label now, and the plot ends where the gridlines do.
    const ax = detectAxes(readImage(PLOS + GRADER1));
    for (const axis of [ax.x, ax.y]) {
        assert.deepEqual(axis.ticks.map(t => t.source), ['grid', 'grid', 'grid', 'grid']);
        assert.ok(axis.ticks.every(t => t.label), 'every gridline has a label');
        assert.ok(axis.line.edge);
    }
    // the y labels left of the lines' starts, the x labels under the lines' ends
    assert.ok(ax.y.ticks.every(t => t.label[2] <= ax.y.line.pos));
    assert.ok(ax.x.ticks.every(t => t.label[1] >= ax.x.line.pos));
    assert.deepEqual(plotBounds(ax), { left: 96, right: null, top: null, bottom: 1154 });
});

test('regression: song-cents, seven gridlines under 325 dots, labels off their ends, years under the last', { skip: !has('song-cents.png') && 'local test image absent' }, () => {
    // before: no x axis; five of the seven gridlines (the 20 and 10 lines have no run of 30% of the width), labeled
    // with the title, a dot's rim and scraps
    const ax = detectAxes(readImage(TESTGRAPHS + 'song-cents.png'));
    assert.equal(ax.y.ticks.length, 7);
    assert.ok(ax.y.regular, 'evenly spaced');
    near(ax.y.spacing, 142.15, 0.1, 'gridline spacing');
    assert.ok(ax.y.ticks.every(t => t.source === 'grid' && t.label && t.label[2] <= 143 && t.label[1] < t.pos && t.label[3] > t.pos), 'labels off the lines\' starts, level with them');
    // the dashed orange reference line at 7 is no gridline
    assert.ok(!ax.y.ticks.some(t => Math.abs(t.pos - 706) < 20));
    assert.deepEqual(ax.x.ticks.map(t => t.source), ['label', 'label', 'label', 'label', 'label', 'label', 'label']);
    assert.ok(ax.x.regular);
    near(ax.x.spacing, 323.9, 0.3, 'label spacing');
    const bounds = plotBounds(ax);
    assert.equal(bounds.left, 143);
    near(bounds.bottom, 905.4, 0.1, 'bottom');
});

// categoryPlot with black dots and a black zero line across it at one of the y axis's ticks, from the tick's outer
// end to the plot's right; in each column a dot rests on the line (a chord of it 5 px wide, as a tick is) with
// another above it, as beeswarms stand on a reference line
function zeroLinePlot() {
    const spec = categoryPlot();
    spec.families[0].color = [0, 0, 0];
    spec.families[0].points = spec.families[0].points.filter(p => Math.abs(p.y - 170) > 14);
    const { image } = synthGraph(spec);
    for (let y = 169; y < 171; y++) {
        for (let x = 53; x < 405; x++)
            image.data.set([0, 0, 0, 255], 4 * (y * image.width + x));
    }
    for (const x of spec.axes.x.ticks) {
        disc(image, x, 164.8, 5, [0, 0, 0]);
        disc(image, x, 149, 5, [0, 0, 0]);
    }
    return { spec, image };
}

test('regression: a zero line across the plot, with dots standing on it, is no axis when the plot has a line at its end (biomarker)', () => {
    // before: the dots on the line were its ticks and the dots above them the ticks' labels, so the zero line was the
    // x axis, the plot above it the labels' side, and every dot there was masked (biomarker: recall 0.47)
    const { spec, image } = zeroLinePlot();
    const ax = detectAxes(image);
    near(ax.x.line.pos, 290.5, 0.5, 'the x axis line');
    assert.deepEqual(ax.x.ticks.map(t => t.source), ['label', 'label']);
    ax.x.ticks.forEach((t, i) => near(t.pos, spec.axes.x.ticks[i], 1, `position ${i}`));
    assert.equal(ax.y.ticks.filter(t => t.label).length, 5);
    assert.deepEqual(plotBounds(ax), { left: ax.y.line.pos, right: null, top: null, bottom: ax.x.line.pos });
});

test('an x axis across the middle of the y axis, with nothing at the y axis\'s ends, stays the x axis', () => {
    const spec = plot();
    spec.axes.x = { ...spec.axes.x, pos: 170.5 };
    spec.axes.y = { ...spec.axes.y, ticks: [270, 220, 120, 70] };
    const ax = detectAxes(synthGraph(spec).image);
    near(ax.x.line.pos, 170.5, 0.2, 'x axis');
    assert.equal(ax.x.ticks.length, 6);
    assert.ok(ax.x.ticks.every(t => t.source === 'tick' && t.label));
});

test('regression: black marks over a twentieth of the image are ink, not a second page: black axes are found (mpo)', () => {
    // before: black, 5.6% of mpo (dots of 13 px, lines 7 px thick, bold labels), was a background color as a panel
    // is, so its axis lines were no ink: the y axis showed one tick and a bracket over the plot was the x axis
    const rng = createRng(31);
    const spec = plot({ families: [{ color: [0, 0, 0], alpha: 1, radius: 12, points: scatterPoints(rng, { n: 40, box: [80, 30, 400, 270] }) }] });
    const { image } = synthGraph(spec);
    assert.deepEqual(backgroundColors(image), [[255, 255, 255], [0, 0, 0]]);
    const ax = detectAxes(image);
    assert.deepEqual(ax.backgrounds, [[255, 255, 255]]);
    near(ax.x.line.pos, 290.5, 0.2, 'x axis');
    near(ax.y.line.pos, 60.5, 0.2, 'y axis');
    assert.equal(ax.x.ticks.filter(t => t.label).length, 6);
    assert.equal(ax.y.ticks.filter(t => t.label).length, 5);
});

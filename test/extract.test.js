// extract.js: the steps from an image to its dots, as the app takes them and as extractAll takes them at once.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { existsSync } from 'node:fs';
import { readImage } from '../tools/png.mjs';
import { scanFamilies, excludedBoxes, familyClips, maskOptions, fitInput, dotFills, extractAll } from '../src/core/extract.js';
import { synthGraph, SYNTH_PRESETS } from '../src/core/synth.js';
import { detectAxes, plotBounds, clipEdges } from '../src/core/axes.js';
import { buildMask } from '../src/core/layers.js';
import { fitDots } from '../src/core/fit.js';
import { matchDots } from '../src/core/metrics.js';
import { restoreChroma } from '../src/core/jpeg.js';
import { solidImage } from '../src/core/render.js';

const sparse = synthGraph(SYNTH_PRESETS.sparse(1));
const swarm = synthGraph(SYNTH_PRESETS.beeswarm(1));

function toRgba(im) {
    const data = new Uint8ClampedArray(4 * im.width * im.height).fill(255);
    for (let p = 0; p < im.width * im.height; p++)
        data.set(im.data.subarray(3 * p, 3 * p + 3), 4 * p);
    return { width: im.width, height: im.height, data };
}

test('scanFamilies: exact colors on the image as it is; noisy matching on its chroma restored', () => {
    const exact = scanFamilies(sparse.image);
    assert.equal(exact.work, sparse.image);
    assert.equal(exact.switched, false);
    assert.equal(exact.prepared.estimates.length, 1);
    assert.ok(Math.abs(exact.prepared.estimates[0].style.radius - 10) < 0.5, `radius ${exact.prepared.estimates[0].style.radius}`);
    const jpeg = scanFamilies(sparse.image, { jpeg: true });
    assert.deepEqual(jpeg.work, restoreChroma(sparse.image));
    assert.equal(jpeg.switched, false);
});

test('scanFamilies with tryNoisy: both scans run, each on half the progress, and exact colors keep a clean image', () => {
    const seen = [];
    const scan = scanFamilies(sparse.image, { tryNoisy: true, onProgress: (label, fraction) => seen.push({ label, fraction }) });
    // the noisy scan finds the same one family, not more
    assert.equal(scan.switched, false);
    assert.equal(scan.work, sparse.image);
    const noisy = seen.filter(s => s.label.startsWith('Trying noisy color matching: '));
    const first = seen.filter(s => !noisy.includes(s));
    assert.ok(first.length > 0 && noisy.length > 0);
    assert.ok(first.every(s => s.fraction >= 0 && s.fraction <= 0.5), 'the exact scan fills the first half');
    assert.ok(noisy.every(s => s.fraction >= 0.5 && s.fraction <= 1), 'the noisy scan the second');
    assert.equal(seen.at(-1).fraction, 1);
    for (let i = 1; i < seen.length; i++)
        assert.ok(seen[i].fraction >= seen[i - 1].fraction - 1e-12, `progress goes forward (${seen[i - 1].fraction} then ${seen[i].fraction})`);
});

test('maskOptions and fitInput: the axes\' bounds and clip edges with the boxes given; the estimate\'s style unless one is given', () => {
    const axes = detectAxes(sparse.image);
    const boxes = [[1, 2, 3, 4]];
    assert.deepEqual(maskOptions(axes, boxes), { bounds: plotBounds(axes), clips: clipEdges(axes), exclude: boxes, strokes: null });
    // with the family's clip edges (familyClips), those instead of the candidates
    const cuts = [{ ...clipEdges(axes)[0], cut: 7 }];
    assert.equal(maskOptions(axes, boxes, cuts).clips, cuts);
    // … and a stroke's pixels (stroke.js pageStroke), as given
    const strokes = new Uint8Array(4);
    assert.equal(maskOptions(axes, boxes, cuts, strokes).strokes, strokes);
    const est = scanFamilies(sparse.image).prepared.estimates[0];
    const mask = new Uint8Array(sparse.image.width * sparse.image.height);
    const input = fitInput(est.image, est, mask);
    assert.equal(input.image, est.image);
    assert.equal(input.style, est.style);
    assert.equal(input.mask, mask);
    assert.equal(input.line, est.familyMeasure.line);
    assert.equal(input.D, est.D);
    const style = { ...est.style, radius: 12 };
    assert.equal(fitInput(est.image, est, mask, style).style, style);
});

test('excludedBoxes: nothing to leave out of a graph without text; dotFills: none for a family of one color', () => {
    assert.deepEqual(excludedBoxes(sparse.image, 10), []);
    const [res] = extractAll(sparse.image);
    assert.equal(dotFills(sparse.image, res.est, res.fit.dots, res.fit.radius), null);
    assert.equal(res.fill, undefined);
});

test('extractAll finds the dots of a synthetic graph', () => {
    const results = extractAll(sparse.image);
    assert.equal(results.length, 1);
    assert.equal(results[0].jpegAuto, false);
    const m = matchDots(sparse.truth, results[0].fit.dots, 5);
    assert.ok(m.precision >= 0.95 && m.recall >= 0.95, `P ${m.precision} R ${m.recall}`);
});

test('extractAll is the app\'s steps taken in turn: scan, the clip edges, mask each family, fit it', () => {
    // three opaque families: each is masked against the others and fitted on its own
    const { image } = swarm;
    const all = extractAll(image);
    assert.equal(all.length, 3);
    const { work, prepared } = scanFamilies(image);
    const axes = detectAxes(image);
    const clips = familyClips(axes, prepared.estimates.map(e => ({ D: e.D, radius: e.style.radius })), work.width, work.height);
    assert.equal(clips.length, 3);
    prepared.estimates.forEach((est, k) => {
        const layers = buildMask(est.image, est, maskOptions(axes, excludedBoxes(work, est.style.radius), clips[k]));
        const fit = fitDots(fitInput(est.image, est, layers.mask));
        assert.deepEqual(all[k].est.style, est.style);
        assert.deepEqual(all[k].layers.mask, layers.mask);
        assert.deepEqual(all[k].fit.dots, fit.dots);
        assert.equal(all[k].fit.radius, fit.radius);
    });
    const found = all.flatMap(r => r.fit.dots);
    const m = matchDots(swarm.truth, found, 4);
    assert.ok(m.precision >= 0.9 && m.recall >= 0.9, `P ${m.precision} R ${m.recall} (${found.length} for ${swarm.truth.length})`);
});

test('extractAll says why when no family can be fitted', () => {
    const blank = toRgba(solidImage(120, 90, [255, 255, 255]));
    assert.throws(() => extractAll(blank), err => err instanceof Error && err.message.length > 0);
});

// A graph with blue dots in its plot and three orange ones under the x axis's labels, as a legend's markers or a
// note's bullets are: a family by its color and its round dots, with no dot among the data.
function withOutsideFamily() {
    return synthGraph({
        width: 520, height: 440, background: [255, 255, 255], blur: 0.5,
        families: [
            { color: [40, 90, 200], alpha: 1, radius: 7, points: [[120, 60], [180, 210], [240, 120], [300, 260], [360, 90], [420, 180], [150, 150], [330, 190], [460, 250], [210, 80]].map(([x, y]) => ({ x, y })) },
            { color: [230, 130, 20], alpha: 1, radius: 7, points: [[200, 400], [260, 400], [320, 400]].map(([x, y]) => ({ x, y })) },
        ],
        axes: {
            x: { pos: 300.5, from: 79, to: 500, ticks: [120, 220, 320, 420] },
            y: { pos: 80.5, from: 20, to: 301.5, ticks: [270, 200, 130, 60] },
        },
    }).image;
}

test('scanFamilies with the plot\'s bounds: a family with no ink inside the plot is left out', () => {
    const image = withOutsideFamily();
    const bounds = plotBounds(detectAxes(image));
    assert.deepEqual([bounds.left.toFixed(1), bounds.bottom.toFixed(1)], ['80.5', '300.5']);
    // (the colors as measured are a unit off those drawn)
    const colors = scan => scan.prepared.estimates.map(e => (e.style.color[0] > e.style.color[2] ? 'orange' : 'blue')).sort();
    // without bounds both colors are families
    assert.deepEqual(colors(scanFamilies(image)), ['blue', 'orange']);
    assert.deepEqual(colors(scanFamilies(image, { bounds })), ['blue']);
    // bounds that leave no family inside are doubted, not the families
    assert.equal(scanFamilies(image, { bounds: { left: null, right: null, top: null, bottom: 10 } }).prepared.estimates.length, 2);
    // extractAll finds the axes itself: the blue dots, and nothing under the axis
    const results = extractAll(image);
    assert.equal(results.length, 1);
    assert.equal(results[0].fit.dots.length, 10);
});

const BENCH_R = new URL('../bench-r/', import.meta.url).pathname;

test('regression: bench-r gg-gray-hi, the tick labels and tick marks are no translucent family beside the dots', { skip: !existsSync(BENCH_R + 'gg-gray-hi.png') && 'bench-r absent' }, () => {
    // ggplot2's gray theme at 300 dpi: text at 60 and ticks at 38 on the gray line from the panel's 230 fit two
    // levels of a translucent family, α 0.87, with no whole dot and no radius, and extractAll stopped on it. All of
    // its ink of half a dot or more is beyond the axes; inside the plot its count image gave 0.1 or less to the
    // gridlines' rims.
    const results = extractAll(readImage(BENCH_R + 'gg-gray-hi.png'));
    assert.equal(results.length, 1);
    assert.deepEqual(results[0].est.style.color.map(Math.round), [202, 16, 31]);
    assert.equal(results[0].fit.dots.length, 40);
});

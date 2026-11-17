import { test } from 'node:test';
import assert from 'node:assert/strict';
import { existsSync } from 'node:fs';
import { createRng } from '../src/core/rng.js';
import { estimateStyle, fitGeometric } from '../src/core/style.js';
import { synthGraph, SYNTH_PRESETS, scatterPoints } from '../src/core/synth.js';
import { readPng } from '../tools/png.mjs';
import { createDotModel, solidImage, toRgba } from '../src/core/render.js';

const near = (a, b, tol, what) => assert.ok(Math.abs(a - b) <= tol, `${what}: ${a} vs ${b} (±${tol})`);

test('fitGeometric recovers α and c from exact levels, and flags a single level as underdetermined', () => {
    const B = [255, 250, 245];
    const c = [40, 120, 200];
    const levels = Array.from({ length: 5 }, (_, n) => c.map((v, ch) => v + (B[ch] - v) * 0.7 ** n));
    const fit = fitGeometric(levels);
    near(fit.alpha, 0.3, 1e-6, 'α');
    for (let ch = 0; ch < 3; ch++)
        near(fit.c[ch], c[ch], 1e-3, 'c');
    assert.ok(fitGeometric(levels.slice(0, 2)).underdetermined);
});

test('fitGeometric with a missing level: counts 1, 2 and 4 fit exactly; read as 1, 2, 3 they do not', () => {
    const B = [255, 255, 255];
    const c = [239, 49, 70];
    const at = n => c.map((v, ch) => v + (B[ch] - v) * 0.5 ** n);
    const colors = [B, at(1), at(2), at(4)];
    const fit = fitGeometric(colors, [0, 1, 2, 4]);
    near(fit.alpha, 0.5, 1e-6, 'α');
    for (let ch = 0; ch < 3; ch++)
        near(fit.c[ch], c[ch], 1e-3, 'c');
    assert.ok(fitGeometric(colors).rms > 1, 'consecutive counts should not fit');
});

// presets: true α, c, r, and the blur the renderer should use (null where upsampling makes it only approximate)
const EXPECTED = {
    sparse: { alpha: 0.35, c: [40, 110, 200], radius: 10, blur: 0 },
    dense: { alpha: 0.3, c: [50, 160, 70], radius: 11, blur: 0.8 },
    curve: { alpha: 0.31, c: [50, 161, 73], radius: 24, blur: null },
};

test('synthetic presets: α within 0.005, c within 2 units, radius within 0.05 px, blur within 0.15 px', () => {
    for (const [name, want] of Object.entries(EXPECTED)) {
        const { image } = synthGraph(SYNTH_PRESETS[name](1));
        const est = estimateStyle(image);
        const s = est.style;
        near(s.alpha, want.alpha, 0.005, `${name} α`);
        for (let ch = 0; ch < 3; ch++)
            near(s.color[ch], want.c[ch], 2, `${name} c`);
        near(s.radius, want.radius, 0.05, `${name} radius`);
        if (want.blur !== null)
            near(s.blur, want.blur, 0.15, `${name} blur`);
        assert.ok(est.isolated.length >= 1);
    }
});

test('the level table starts with B and the measured levels, then continues geometrically', () => {
    const { image } = synthGraph(SYNTH_PRESETS.dense(1));
    const { style, background } = estimateStyle(image);
    assert.deepEqual(style.levels[0], background.color);
    const K = style.measuredLevels;
    const q = 1 - style.alpha;
    const next = style.levels[K].map((v, ch) => style.color[ch] + (v - style.color[ch]) * q);
    for (let ch = 0; ch < 3; ch++)
        near(style.levels[K + 1][ch], next[ch], 1e-9, 'extrapolated level');
});

test('two translucent families and an opaque line in one family’s color: both found, the line reported beyond', () => {
    const rng = createRng(9);
    const red = [246, 66, 60];
    const blue = [58, 77, 172];
    const box = [20, 20, 480, 330];
    const spec = {
        width: 500, height: 350, background: [255, 255, 255], blur: 0.7,
        families: [
            { color: blue, alpha: 0.29, radius: 8.4, points: scatterPoints(rng, { n: 60, box }) },
            { color: red, alpha: 0.3, radius: 8.4, points: scatterPoints(rng, { n: 90, box }) },
        ],
        lines: [{ points: [[10, 300], [250, 150], [490, 40]], width: 7, color: red }],
    };
    const est = estimateStyle(synthGraph(spec).image);
    const translucent = est.candidates.filter(c => c.translucent);
    assert.equal(translucent.length, 2);
    const redFam = translucent.find(c => c.fit.c[0] > 200);
    const blueFam = translucent.find(c => c.fit.c[2] > 150);
    near(redFam.fit.alpha, 0.3, 0.01, 'red α');
    near(blueFam.fit.alpha, 0.29, 0.01, 'blue α');
    assert.ok(redFam.beyond.some(L => Math.hypot(...L.color.map((v, ch) => v - red[ch])) < 3), 'red line found beyond the levels');
    // the larger family comes first
    assert.equal(translucent[0], redFam);
    // choosing the second family by index
    const second = estimateStyle(synthGraph(spec).image, { family: est.candidates.indexOf(blueFam) });
    near(second.style.alpha, 0.29, 0.01, 'chosen blue α');
});

test('gray text with two shades is not taken for a translucent family', () => {
    // two gray rectangles (text-like), no dots: their levels fit only with c far outside the RGB cube
    const image = { width: 200, height: 100, data: new Uint8ClampedArray(4 * 200 * 100).fill(255) };
    for (let y = 20; y < 80; y++) {
        for (let x = 20; x < 180; x++) {
            const v = x < 100 ? 85 : 0;
            image.data.set([v, v, v, 255], 4 * (y * 200 + x));
        }
    }
    const est = estimateStyle(image);
    assert.ok(est.candidates.every(c => !c.translucent));
});

const TESTGRAPHS = new URL('../testgraphs/', import.meta.url).pathname;
const has = name => existsSync(TESTGRAPHS + name);

test('dog-ages matches the phase 0 measurements', { skip: !has('dog-ages.png') && 'local test image absent' }, () => {
    const est = estimateStyle(readPng(TESTGRAPHS + 'dog-ages.png'));
    const s = est.style;
    near(s.alpha, 0.310, 0.003, 'α');
    near(s.radius, 24.04, 0.1, 'radius');
    assert.equal(s.measuredLevels, 5);
    assert.ok(est.isolated.length >= 4, `${est.isolated.length} isolated dots`);
    near(s.blur, 1.15, 0.15, 'blur');
});

test('economist: red and blue families, red first, its trend line beyond its levels', { skip: !has('economist-canada-unfriendly-chart.png') && 'local test image absent' }, () => {
    const est = estimateStyle(readPng(TESTGRAPHS + 'economist-canada-unfriendly-chart.png'));
    const translucent = est.candidates.filter(c => c.translucent);
    assert.equal(translucent.length, 2);
    near(translucent[0].fit.alpha, 0.302, 0.005, 'red α');
    near(translucent[1].fit.alpha, 0.289, 0.01, 'blue α');
    assert.ok(translucent[0].beyond.length >= 1);
    near(est.style.radius, 8.37, 0.1, 'red radius');
});

test('regression: translucent dots that never overlap form an opaque family with a two-entry table and a radius', () => {
    // sparse preset seed 3 has no overlapping dots: α can't be measured. Its table once repeated the level, so every
    // dot read as 13 deep; then its color (5.9% of the image) was taken for a background and the family was lost.
    const est = estimateStyle(synthGraph(SYNTH_PRESETS.sparse(3)).image);
    assert.equal(est.familyMeasure.kind, 'opaque');
    assert.equal(est.style.measuredLevels, 1);
    assert.equal(est.style.levels.length, 2);
    near(est.style.radius, 10, 0.05, 'radius');
});

test('regression: sparse translucent dots with one coincident pair are translucent, α from the pair', () => {
    // bench1 small-dots and tiny-dots: 50% black dots that overlap only in one coincident pair, whose second level
    // is a handful of flat pixels; read as opaque, the pair's darker pixels were "other" and both dots were lost
    for (const radius of [2.6, 3.8]) {
        const points = [];
        for (let y = 20; y < 180; y += 20) {
            for (let x = 20; x < 280; x += 20)
                points.push({ x: x + 0.3 * (y % 7), y: y + 0.2 * (x % 5) });
        }
        points.push({ ...points[17] });
        const spec = { width: 300, height: 200, background: [255, 255, 255], blur: 0.3, families: [{ color: [0, 0, 0], alpha: 0.5, radius, points }] };
        const est = estimateStyle(synthGraph(spec).image);
        assert.equal(est.familyMeasure.kind, 'translucent', `r ${radius}`);
        near(est.style.alpha, 0.5, 0.02, `α at r ${radius}`);
        near(est.style.radius, radius, 0.15, 'radius');
    }
});

test('a gray label on the line to opaque black dots is no level 1: the dots are an opaque family', () => {
    // molar1's colors: black dots (19, 20, 19) and a gray label (62, 62, 62), which fit levels 1 and 2 of a translucent
    // family at α 0.78 that no dot showed whole. The label's strokes, 4 px wide, lie apart from every dot, where a
    // real level 2 lies inside level 1.
    const W = 500;
    const H = 200;
    const model = createDotModel(solidImage(W, H, [255, 255, 255]), { color: [19, 20, 19], alpha: 1, radius: 7 });
    const dots = [];
    for (let k = 0; k < 24; k++)
        dots.push({ x: 20 + 20 * k, y: 30 + 24 * Math.sin(k / 3) });
    model.addAll(dots);
    const im = model.image;
    for (let s = 0; s < 20; s++) {
        const x0 = 20 + 24 * s;
        for (let y = 120; y < 180; y++) {
            for (let x = x0; x < x0 + 4; x++)
                im.data.set([62, 62, 62], 3 * (y * W + x));
        }
        for (let x = x0; x < x0 + 16; x++) {
            for (let y = 150; y < 154; y++)
                im.data.set([62, 62, 62], 3 * (y * W + x));
        }
    }
    const est = estimateStyle(toRgba(im));
    assert.equal(est.familyMeasure?.kind, 'opaque', est.notes.join(' '));
    near(est.style.radius, 7, 0.5, 'radius');
});

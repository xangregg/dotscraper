import { test } from 'node:test';
import assert from 'node:assert/strict';
import { existsSync } from 'node:fs';
import { createRng } from '../src/core/rng.js';
import { shapeRadius, siblingRadius } from '../src/core/opaque.js';
import { estimateStyle } from '../src/core/style.js';
import { synthGraph, SYNTH_PRESETS, scatterPoints } from '../src/core/synth.js';
import { readPng } from '../tools/png.mjs';
import { createDotModel, solidImage, toRgba } from '../src/core/render.js';

const near = (a, b, tol, what) => assert.ok(Math.abs(a - b) <= tol, `${what}: ${a} vs ${b} (±${tol})`);

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

// Translucent dots over a box plot's box drawn in their own hue, darker than one dot (protrusion2: the green box read
// as level 2, 6 units from the dots' own level 2). sRGB compositing, as JMP's.
function dotsOverBox(boxAlpha) {
    const W = 360;
    const H = 260;
    const fam = { color: [133, 216, 179], alpha: 0.29, radius: 12 };
    const bg = solidImage(W, H, [255, 255, 255]);
    const box = fam.color.map(v => 255 + boxAlpha * (v - 255));
    for (let y = 40; y < 220; y++) {
        for (let x = 200; x < 300; x++)
            bg.data.set(box, 3 * (y * W + x));
    }
    const dots = [];
    // lone dots, and overlapping pairs that show level 2 on the page
    for (let k = 0; k < 6; k++)
        dots.push({ x: 30 + 28 * k, y: 30 }, { x: 30 + 28 * k, y: 100 + 4 * k }, { x: 30 + 28 * k + 12, y: 108 + 4 * k });
    for (let k = 0; k < 7; k++)
        dots.push({ x: 222 + 9 * (k % 4), y: 60 + 24 * k }, { x: 268 - 7 * (k % 3), y: 70 + 22 * k });
    const model = createDotModel(bg, fam);
    model.addAll(dots);
    return { image: toRgba(model.image), box, fam };
}

test('a box under translucent dots in their own hue is a band, not one of their levels', () => {
    const { image, box } = dotsOverBox(0.45);
    const est = estimateStyle(image);
    const fam = est.familyMeasure;
    const dist = c => Math.hypot(...c.map((v, ch) => v - box[ch]));
    assert.ok(fam.band && dist(fam.band) <= 3, `band ${fam.band}`);
    assert.ok(fam.box, 'marked a box');
    assert.equal(fam.kind, 'translucent');
    assert.ok(fam.levels.every(L => dist(L.color) > 3), `levels ${fam.levels.map(L => L.color.map(Math.round))}`);
    // (one dot over the box, 181, 231, 209, lies on the line near level 3, 177, and joins the geometric fit: α 0.35)
    near(est.style.alpha, 0.29, 0.07, 'α');
    near(est.style.radius, 12, 0.5, 'radius');
});

// count image of discs of radius r at count 1, antialiased, with a NaN line two pixels wide through every center
function cutDiscs(W, H, centers, r) {
    const D = new Float32Array(W * H);
    for (const c of centers) {
        for (let y = 0; y < H; y++) {
            for (let x = 0; x < W; x++) {
                const v = Math.min(1, Math.max(0, r + 0.5 - Math.hypot(x + 0.5 - c.x, y + 0.5 - c.y)));
                D[y * W + x] = Math.max(D[y * W + x], v);
            }
        }
    }
    for (const c of centers) {
        for (let x = 0; x < W; x++) {
            D[Math.floor(c.y) * W + x] = NaN;
            D[(Math.floor(c.y) - 1) * W + x] = NaN;
        }
    }
    return D;
}

test('shapeRadius: dots cut through their centers by a masked line are whole dots, not halves', () => {
    // protrusion1: every dot on a whisker; the distance transform read 9.5–10 for 19.5, and Hough never scanned 19.5
    const centers = [40, 110, 180, 250].map(x => ({ x, y: 40 }));
    const D = cutDiscs(300, 80, centers, 19.5);
    const s = shapeRadius(D, 300, 80);
    assert.ok(s, 'a radius');
    // (the scan keeps the largest radius at which most centers show: these discs show at 18–21 alike)
    assert.ok(s.radius >= 18 && s.radius <= 21.5, `radius ${s.radius}, not half of 19.5`);
});

test('siblingRadius: another family\'s radius when its dots show at it, none when they don\'t', () => {
    const centers = [40, 110, 180, 250].map(x => ({ x, y: 40 }));
    const D = cutDiscs(300, 80, centers, 15);
    const s = siblingRadius(D, 300, 80, [10, 15, 21]);
    assert.equal(s?.radius, 15);
    assert.equal(siblingRadius(D, 300, 80, [8, 25]), null);
});

test('regression: nearly opaque dots a touch deeper in their outer halves have no ring to measure, so none is a whole dot (expression, noisy)', () => {
    // expression read with noisy matching: red dots at α 0.96, the outer half of each a level deeper; two passed as
    // whole outlined dots and gave a radius to this family and, as a sibling's, to a gray one: 1033 dots for 46
    const W = 200;
    const H = 120;
    const style = { color: [20, 40, 60], alpha: 0.9, radius: 8, ring: { inner: 0.55, alpha: 0.035 } };
    const dots = [];
    for (let y = 20; y < H - 10; y += 30) {
        for (let x = 20; x < W - 10; x += 30)
            dots.push({ x: x + 0.3, y: y + 0.6 });
    }
    dots.push({ ...dots[0] }, { ...dots[5] });
    const model = createDotModel(solidImage(W, H, [255, 255, 255]), style);
    model.addAll(dots);
    const im = toRgba(model.image);
    const est = estimateStyle({ ...im, data: Uint8ClampedArray.from(im.data, v => Math.round(v)) }, { seed: [43, 61, 79] });
    assert.ok(est.familyMeasure.translucent && est.style.alpha > 0.93, `α ${est.style.alpha}`);
    assert.equal(est.style.ring, undefined);
    assert.equal(est.isolated.length, 0);
    assert.equal(est.style.radius, null);
});

test('small blurred translucent dots with a ring inside the rim: α, radius and the ring\'s radii and opacity', () => {
    // as matplotlib draws them: the edge stroked in the fill's color and alpha, half over the fill
    const W = 420;
    const H = 300;
    const style = { color: [71, 116, 164], alpha: 0.44, radius: 6.8, blur: 0.5, ring: { inner: 0.51, outer: 0.76, alpha: 0.44 * 0.56 } };
    const rng = createRng(11);
    const dots = [];
    while (dots.length < 90)
        dots.push({ x: 20 + (W - 40) * rng.uniform(), y: 20 + (H - 40) * rng.uniform() });
    // coincident pairs, for flat pixels two layers deep
    dots.push(...dots.slice(0, 8).map(d => ({ ...d })));
    const model = createDotModel(solidImage(W, H, [255, 255, 255]), style);
    model.addAll(dots);
    const im = toRgba(model.image);
    const est = estimateStyle({ ...im, data: Uint8ClampedArray.from(im.data, v => Math.round(v)) });
    assert.equal(est.familyMeasure.kind, 'translucent');
    near(est.style.alpha, 0.44, 0.01, 'α');
    near(est.style.radius, 6.8, 0.1, 'radius');
    near(est.style.blur, 0.5, 0.15, 'blur');
    near(est.style.ring.inner, 0.51, 0.03, 'ring inner');
    near(est.style.ring.outer, 0.76, 0.03, 'ring outer');
    near(est.style.ring.alpha, 0.246, 0.04, 'ring opacity');
});

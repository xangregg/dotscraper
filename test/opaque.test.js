import { test } from 'node:test';
import assert from 'node:assert/strict';
import { existsSync } from 'node:fs';
import { createRng } from '../src/core/rng.js';
import { scanOf } from '../src/core/scan.js';
import { measureOpaque, familyRings, colorClusters, isDotFamily } from '../src/core/opaque.js';
import { estimateStyle, opaqueFamilyEstimate } from '../src/core/style.js';
import { buildMask } from '../src/core/layers.js';
import { fitDots } from '../src/core/fit.js';
import { synthGraph, SYNTH_PRESETS, scatterPoints } from '../src/core/synth.js';
import { matchDots } from '../src/core/metrics.js';
import { readPng, readImage } from '../tools/png.mjs';
import { createDotModel, solidImage, toRgba } from '../src/core/render.js';
import { estimateFamilies } from '../src/core/families.js';
import { laneCentersFromDots, readTruthCsv, truthPixels, truthPixelsByTicks } from '../tools/truth-csv.mjs';
import { readTruthJson, truthFromExport, isDotsExport } from '../tools/truth-json.mjs';
import { extractAll } from '../src/core/extract.js';
import { groupRow, laneRow } from '../tools/bench.mjs';
import { detectAxes } from '../src/core/axes.js';
import { outlinesInP } from '../src/core/outline.js';
import { restoreChroma } from '../src/core/jpeg.js';

const near = (a, b, tol, what) => assert.ok(Math.abs(a - b) <= tol, `${what}: ${a} vs ${b} (±${tol})`);

function fitFamily(image, k, candidates) {
    const est = estimateStyle(image, { family: k, candidates });
    const layers = buildMask(image, est);
    const fit = fitDots({ image, style: est.style, line: est.familyMeasure.line, mask: layers.mask, D: est.D, imgT: est.fitT, seeds: est.seeds });
    return { est, layers, fit };
}

test('synthetic swarm: three opaque families, their colors and radius; no family for the antialiased blends', () => {
    const spec = SYNTH_PRESETS.beeswarm(1);
    const est = estimateStyle(synthGraph(spec).image);
    // exactly three: maroon's blend with white once made a fourth "family"
    assert.equal(est.candidates.length, 3);
    for (const f of spec.families) {
        const c = est.candidates.find(k => Math.hypot(...k.seed.map((v, ch) => v - f.color[ch])) < 6);
        assert.ok(c, `family ${f.color} found`);
        assert.equal(c.kind, 'opaque');
        near(c.radius, 8, 0.15, 'radius');
    }
});

test('measureOpaque, a plain fill: 1 in its dots, 0 on the page, unknown on another family; radius, centers, coverage', () => {
    const spec = SYNTH_PRESETS.beeswarm(1);
    const { image, truth } = synthGraph(spec);
    const [first, ...rest] = spec.families;
    const fam = measureOpaque(scanOf(image), first.color, rest.map(f => f.color));
    const at = d => fam.D[Math.floor(d.y) * image.width + Math.floor(d.x)];
    const own = truth.filter(d => d.family === 0);
    // a dot's center is covered (the image has noise of 0.4 units), or unknown where the black median line crosses it
    const centers = own.map(at);
    const covered = centers.filter(v => v > 0.95).length;
    assert.ok(centers.every(v => v > 0.95 || Number.isNaN(v)), `own centers: ${centers.filter(v => !(v > 0.95)).slice(0, 5)}`);
    assert.ok(covered >= 0.8 * own.length, `${covered} of ${own.length} own centers covered`);
    assert.equal(fam.D[2 * image.width + 2], 0, 'the page');
    const theirs = truth.filter(d => d.family !== 0 && own.every(o => Math.hypot(o.x - d.x, o.y - d.y) > 3 * first.radius));
    assert.ok(theirs.length > 5 && theirs.every(d => Number.isNaN(at(d))), 'another family, away from this one');
    near(fam.radius, first.radius, 0.15, 'radius');
    assert.ok(fam.dots >= 0.7 * own.length && fam.dots <= own.length, `${fam.dots} centers for ${own.length} dots`);
    assert.ok(fam.coverage > 0.75, `coverage ${fam.coverage}`);
    assert.deepEqual(fam.background, [255, 255, 255]);
    assert.equal(fam.kind, 'opaque');
});

test('synthetic swarm: each family fitted exactly, no phantoms from other families’ rims', () => {
    for (const seed of [1, 2]) {
        const spec = SYNTH_PRESETS.beeswarm(seed);
        const { image, truth } = synthGraph(spec);
        const { candidates } = estimateStyle(image);
        candidates.forEach((c, k) => {
            const fam = spec.families.findIndex(f => Math.hypot(...f.color.map((v, ch) => v - c.seed[ch])) < 6);
            const { fit } = fitFamily(image, k, candidates);
            const m = matchDots(truth.filter(t => t.family === fam), fit.dots, 4);
            assert.equal(m.recall, 1, `seed ${seed} family ${fam} recall`);
            assert.equal(m.precision, 1, `seed ${seed} family ${fam} precision`);
            assert.ok(m.rmse < 0.1, `rmse ${m.rmse}`);
        });
    }
});

test('an opaque bar in the dot color is masked as thin; dots cut by another line are not', () => {
    const rng = createRng(41);
    const teal = [58, 152, 176];
    const points = scatterPoints(rng, { n: 30, box: [30, 30, 370, 270] });
    const spec = {
        width: 400, height: 300, background: [255, 255, 255], blur: 0.5,
        families: [{ color: teal, alpha: 1, radius: 5.7, points }],
        lines: [
            { points: [[100, 280], [100, 150], [300, 150], [300, 280]], width: 6, color: teal }, // a bar outline, Laser-like
            { points: [[20, 80], [380, 80]], width: 2, color: [0, 0, 0] }, // a black line across some dots
        ],
    };
    const { image } = synthGraph(spec);
    const est = estimateStyle(image);
    const { thin } = buildMask(image, est);
    // the bar's middle is thin
    let barCore = 0;
    let barThin = 0;
    for (let y = 160; y < 270; y++) {
        for (const x of [99, 100, 299, 300]) {
            barCore++;
            barThin += thin[y * 400 + x];
        }
    }
    assert.ok(barThin / barCore > 0.9, `${barThin} of ${barCore} bar pixels thin`);
    // dots crossed by the black line keep their halves (they were joined into "thin" marks once)
    for (const p of points.filter(q => Math.abs(q.y - 80) < 4)) {
        for (const dy of [-3, 3])
            assert.equal(thin[Math.round(p.y + dy) * 400 + Math.round(p.x)], 0, `half of the dot at ${p.x}, ${p.y}`);
    }
});

const BENCH_LOCAL = new URL('../bench-local/', import.meta.url).pathname;
const has = name => existsSync(BENCH_LOCAL + name);

test('families of the real images', { skip: !has('cantons-haemocytes-dotplot.png') && 'local test images absent' }, () => {
    const kinds = name => estimateStyle(readPng(BENCH_LOCAL + name)).candidates.map(c => c.kind);
    assert.deepEqual(kinds('dog-ages.png'), ['translucent']);
    assert.deepEqual(kinds('economist-canada-unfriendly-chart.png'), ['translucent', 'translucent']);
    const cantons = estimateStyle(readPng(BENCH_LOCAL + 'cantons-haemocytes-dotplot.png')).candidates;
    assert.equal(cantons.length, 4);
    assert.ok(cantons.every(c => Math.abs(c.radius - 6.37) < 0.1));
    assert.equal(kinds('Laser-Asyllable-Maturity.png').length, 3);
    // the gray dots share their color with antialiased text, which once made white their background and text their
    // radius; the red dots read as half of pink and once took pink dots as their own
    const nicest = estimateStyle(readPng(BENCH_LOCAL + 'nicest language.png')).candidates;
    assert.ok(nicest.length >= 4);
    assert.ok(nicest.some(c => Math.hypot(...c.seed.map((v, ch) => v - [202, 201, 191][ch])) < 6));
    // violin's dots sit on its shading and on white
    assert.equal(kinds('violin-beeswarm.png').length, 1);
});

test('regression: Laser’s black family fits its dots, not its text or bar', { skip: !has('Laser-Asyllable-Maturity.png') && 'local test image absent' }, () => {
    const image = readPng(BENCH_LOCAL + 'Laser-Asyllable-Maturity.png');
    const { candidates } = estimateStyle(image);
    const k = candidates.findIndex(c => c.seed.every(v => v < 40));
    const { fit } = fitFamily(image, k, candidates);
    // 23 black dots; text and the bar outline once drew 118
    near(fit.dots.length, 23, 1, 'black dots');
    // none below the plot (the category labels) or left of the y axis (the tick labels)
    assert.ok(fit.dots.every(d => d.y < 600 && d.x > 140), JSON.stringify(fit.dots.filter(d => d.y >= 600 || d.x <= 140)));
});

// JMP's outlined markers: each dot an opaque disc of the outline color with a smaller fill drawn over it, dot by dot
function outlinedImage(dots, fill, outline, r, width) {
    let im = solidImage(200, 120, [255, 255, 255]);
    for (const d of dots) {
        for (const [color, radius] of [[outline, r + width], [fill, r]]) {
            const model = createDotModel(im, { color, alpha: 1, radius });
            model.add(d.x, d.y);
            im = model.image;
        }
    }
    const rgba = toRgba(im);
    return { ...rgba, data: Uint8ClampedArray.from(rgba.data, v => Math.round(v)) };
}

test('outlined opaque dots: one family, fill and outline, measured at the outline’s radius', () => {
    const fill = [171, 8, 252];
    const outline = [105, 2, 156];
    const dots = [];
    for (let i = 0; i < 6; i++)
        dots.push({ x: 20 + i * 30, y: 30 }, { x: 25 + i * 30, y: 85 + (i % 2) * 6 });
    dots.push({ x: 37, y: 92 }); // overlapping the second row's first dot
    const image = outlinedImage(dots, fill, outline, 8.8, 1.7);
    const est = estimateStyle(image);
    assert.equal(est.candidates.length, 1, 'the outline is not a family of its own');
    const fam = est.candidates[0];
    assert.equal(fam.kind, 'opaque');
    assert.ok(fam.outline && Math.hypot(...fam.outline.map((v, ch) => v - outline[ch])) <= 6, `outline ${fam.outline}`);
    near(fam.radius, 10.5, 0.2, 'radius');
    const layers = buildMask(image, est);
    const fit = fitDots({ image, style: est.style, line: est.familyMeasure.line, mask: layers.mask, D: est.D, imgT: est.fitT });
    const m = matchDots(dots, fit.dots, 2);
    assert.equal(fit.dots.length, dots.length);
    assert.equal(m.recall, 1);
});

// White dots in black outlines on a white page (bench-nature hair1, tumour1), drawn in order so later dots hide
// earlier ones, beside a bar outlined in black (its inside is enclosed page too, but larger than a dot) and small
// rings like letters' counters.
function pageFilledImage(dots) {
    let im = solidImage(260, 160, [255, 255, 255]);
    const black = [20, 20, 20];
    const white = [255, 255, 255];
    const disc = (x, y, color, radius) => {
        const model = createDotModel(im, { color, alpha: 1, radius });
        model.add(x, y);
        im = model.image;
    };
    // the bar: a black rectangle with a white inside
    for (let y = 60; y < 150; y++) {
        for (let x = 200; x < 250; x++) {
            const edge = y < 62 || x < 202 || x >= 248;
            im.data.set(edge ? black : white, 3 * (y * 260 + x));
        }
    }
    for (const [x, y] of [[20, 140], [32, 140], [44, 140], [56, 140]]) {
        disc(x, y, black, 4.6);
        disc(x, y, white, 3.4);
    }
    for (const d of dots) {
        disc(d.x, d.y, black, 9.5);
        disc(d.x, d.y, white, 8);
    }
    const rgba = toRgba(im);
    return { ...rgba, data: Uint8ClampedArray.from(rgba.data, v => Math.round(v)) };
}

test('dots filled with the page color inside outlines: one family by their enclosed fills, fitted whole and hidden', () => {
    const dots = [];
    for (let i = 0; i < 6; i++)
        dots.push({ x: 20 + i * 30, y: 25 });
    dots.push({ x: 30, y: 75 }, { x: 44, y: 78 }, { x: 58, y: 75 }); // overlapping: the later dots on top
    dots.push({ x: 120, y: 80 }, { x: 225, y: 100 }, { x: 225, y: 140 }); // two in the bar, one touching its edge
    const image = pageFilledImage(dots);
    const est = estimateStyle(image);
    const k = est.candidates.findIndex(c => c.pageFill);
    assert.ok(k >= 0, `a page-filled family among ${est.candidates.map(c => c.seed)}`);
    const fam = est.candidates[k];
    assert.deepEqual(fam.seed.map(Math.round), [255, 255, 255]);
    // (read in a band with the outline's antialiased rims, so a little light)
    assert.ok(Math.hypot(...fam.outline.map((v, ch) => v - 20)) <= 25, `outline ${fam.outline}`);
    near(fam.radius, 9.5, 0.4, 'radius');
    const { fit } = fitFamily(image, k, est.candidates);
    const m = matchDots(dots, fit.dots, 1.5);
    assert.equal(m.recall, 1, `missed ${JSON.stringify(m.missed.map(i => dots[i]))}`);
    assert.equal(fit.dots.length, dots.length, `extra ${JSON.stringify(m.extra.map(j => fit.dots[j]))}`);
});

// Outlined white dots and "0"s, drawn in order on a canvas (supersampled 4 × 4 for the antialiasing): a dot is a
// black disc of radius R with a white one of rf over it; a 0 a black ellipse with a white one inside, its counter
// about half a fill's area; `bar` a gray rectangle drawn first, under the dots.
function outlinedScene({ dots, zeros = [], bar = null, width = 300, height = 130, R = 9.5, rf = 8 }) {
    const S = 4;
    const black = [20, 20, 20];
    const gray = [171, 168, 173];
    const sum = new Float64Array(3 * width * height);
    const color = (x, y) => {
        let c = [255, 255, 255];
        if (bar && x >= bar.x0 && x < bar.x1 && y >= bar.y0 && y < bar.y1)
            c = gray;
        for (const z of zeros) {
            const e = (a, b) => ((x - z.x) / (a * rf)) ** 2 + ((y - z.y) / (b * rf)) ** 2 <= 1;
            if (e(0.875, 1.3))
                c = e(0.56, 1) ? [255, 255, 255] : black;
        }
        for (const d of dots) {
            const r = Math.hypot(x - d.x, y - d.y);
            if (r <= R)
                c = r <= rf ? [255, 255, 255] : black;
        }
        return c;
    };
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            for (let sy = 0; sy < S; sy++) {
                for (let sx = 0; sx < S; sx++) {
                    const c = color(x + (sx + 0.5) / S, y + (sy + 0.5) / S);
                    for (let ch = 0; ch < 3; ch++)
                        sum[3 * (y * width + x) + ch] += c[ch];
                }
            }
        }
    }
    const data = new Uint8ClampedArray(4 * width * height);
    for (let p = 0; p < width * height; p++) {
        for (let ch = 0; ch < 3; ch++)
            data[4 * p + ch] = Math.round(sum[3 * p + ch] / (S * S));
        data[4 * p + 3] = 255;
    }
    return { width, height, data };
}

test('page-filled dots mostly hidden in rows: their crescents are fills, the counters of 0s are not', () => {
    // before: no family (two whole fills, under the three needed); with crescents taken at any Hough arc of a fifth,
    // the 0s' counters became dots too. Each row is drawn left to right, so its last dot is the only whole one and the
    // others show their left crescents. At hair2's scale (fill 26.6 px, outline 3.9, dots 14.6 apart): half as large,
    // the fit drops two middle dots, which change the union of discs by less than a dot costs.
    const rows = [[60, 74.6, 89.2, 103.8, 118.4], [300, 314.6, 329.2, 343.8]];
    const dots = rows.flatMap((xs, k) => xs.map(x => ({ x, y: 50 + 100 * k })));
    const zeros = [{ x: 470, y: 50 }, { x: 520, y: 50 }, { x: 570, y: 50 }, { x: 520, y: 150 }];
    const image = outlinedScene({ dots, zeros, width: 640, height: 200, R: 30.5, rf: 26.6 });
    const est = estimateStyle(image);
    const k = est.candidates.findIndex(c => c.pageFill);
    assert.ok(k >= 0, `a page-filled family among ${est.candidates.map(c => c.seed)}`);
    const { fit } = fitFamily(image, k, est.candidates);
    const m = matchDots(dots, fit.dots, 1.5);
    assert.equal(m.recall, 1, `missed ${JSON.stringify(m.missed.map(i => dots[i]))}`);
    assert.equal(fit.dots.length, dots.length, `extra ${JSON.stringify(m.extra.map(j => fit.dots[j]))}`);
});

test('page-filled dots over a gray bar: the outline mixes with the bar, which is no part of a dot', () => {
    // before: the gray (171), on the line from white to black, read as 0.36 of the outline out to the outline's
    // reach, so the dots over the bar read larger than those on the page (tumour1: the fit paired them)
    const dots = [{ x: 30, y: 40 }, { x: 60, y: 40 }, { x: 90, y: 40 }, { x: 160, y: 50 }, { x: 200, y: 70 }, { x: 240, y: 55 }];
    const bar = { x0: 140, x1: 260, y0: 30, y1: 125 };
    const image = outlinedScene({ dots, bar });
    const est = estimateStyle(image);
    const k = est.candidates.findIndex(c => c.pageFill);
    assert.ok(k >= 0, `a page-filled family among ${est.candidates.map(c => c.seed)}`);
    const fam = estimateStyle(image, { family: k, candidates: est.candidates });
    // on the bar, two pixels beyond a dot's outline
    for (const d of dots.slice(3)) {
        const p = Math.floor(d.y) * image.width + Math.floor(d.x + 11.5);
        assert.equal(fam.D[p], 0, `D beside (${d.x}, ${d.y})`);
    }
    const { fit } = fitFamily(image, k, est.candidates);
    const m = matchDots(dots, fit.dots, 1);
    assert.equal(m.recall, 1, `missed ${JSON.stringify(m.missed.map(i => dots[i]))}`);
    assert.equal(fit.dots.length, dots.length, `extra ${JSON.stringify(m.extra.map(j => fit.dots[j]))}`);
    near(fit.radius, 9.5, 0.3, 'radius');
});

// Shapes painted in order on a white canvas, supersampled S × S: discs { x, y, r, color } and thick segments
// { line: [x0, y0, x1, y1], w, color }, each painted over its bounding box only
function paintScene(width, height, shapes, S = 4) {
    const W = width * S;
    const H = height * S;
    const buf = new Uint8Array(3 * W * H).fill(255);
    for (const sh of shapes) {
        const [x0, y0, x1, y1] = sh.line ?? [sh.x, sh.y, sh.x, sh.y];
        const reach = sh.line ? sh.w / 2 : sh.r;
        const dx = x1 - x0;
        const dy = y1 - y0;
        const len2 = dx * dx + dy * dy;
        for (let Y = Math.max(0, Math.floor((Math.min(y0, y1) - reach) * S)); Y < Math.min(H, Math.ceil((Math.max(y0, y1) + reach) * S)); Y++) {
            for (let X = Math.max(0, Math.floor((Math.min(x0, x1) - reach) * S)); X < Math.min(W, Math.ceil((Math.max(x0, x1) + reach) * S)); X++) {
                const px = (X + 0.5) / S;
                const py = (Y + 0.5) / S;
                const t = len2 ? Math.max(0, Math.min(1, ((px - x0) * dx + (py - y0) * dy) / len2)) : 0;
                if (Math.hypot(px - x0 - t * dx, py - y0 - t * dy) <= reach)
                    buf.set(sh.color, 3 * (Y * W + X));
            }
        }
    }
    const data = new Uint8ClampedArray(4 * width * height);
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            for (let ch = 0; ch < 3; ch++) {
                let sum = 0;
                for (let sy = 0; sy < S; sy++) {
                    for (let sx = 0; sx < S; sx++)
                        sum += buf[3 * ((y * S + sy) * W + x * S + sx) + ch];
                }
                data[4 * (y * width + x) + ch] = Math.round(sum / (S * S));
            }
            data[4 * (y * width + x) + 3] = 255;
        }
    }
    return { width, height, data };
}

test('grays in black outlines beside black dots, after phase1: every family, curves of their own grays under them', () => {
    // before: 187 alone. 136's and 228's rings read as the page, their curves outweighing their dots; 82, between
    // black and 187, was a blend of 187 with its background, black; black, the outlines' color and text's, read its
    // periods' radius; one gray's two histogram peaks made two families. Dots of fill radius 17 in outlines 2.4 px
    // wide, as phase1's.
    const black = [20, 21, 20];
    const R = 19.4;
    const rf = 17;
    const grays = { 228: [228, 228, 228], 187: [187, 188, 188], 136: [136, 136, 136], 82: [82, 82, 81] };
    const dots = [];
    const shapes = [];
    const row = (key, y, xs) => xs.forEach(x => dots.push({ x, y, family: key }));
    // 228 and 136 sit on curves of their own gray, 13 px wide, far longer than their dots, from the image's edge (a
    // stub shorter than 3 r beside a dot reads as its fill, and the fit doubles that dot: phase1's 187 where its
    // curve meets its dots)
    shapes.push({ line: [0, 60, 880, 75], w: 13, color: grays[228] }, { line: [0, 300, 880, 285], w: 13, color: grays[136] });
    shapes.push({ line: [20, 420, 880, 420], w: 2, color: black });
    row(228, 64, [80, 200, 320, 440, 560]);
    row(187, 180, [80, 200, 320, 440, 560]);
    row(136, 296, [80, 200, 320, 440, 560]);
    row(82, 180, [680, 800]);
    row(82, 296, [680, 800]);
    row(82, 64, [680]);
    row('black', 420, [80, 200, 320, 440, 560, 680]);
    for (const d of dots) {
        if (d.family === 'black')
            shapes.push({ x: d.x, y: d.y, r: R, color: black });
        else
            shapes.push({ x: d.x, y: d.y, r: R, color: black }, { x: d.x, y: d.y, r: rf, color: grays[d.family] });
    }
    // text's periods, and a black block as a colorbar's (too narrow for a dot)
    for (let k = 0; k < 30; k++)
        shapes.push({ x: 30 + 28 * k, y: 500, r: 4.9, color: black });
    shapes.push({ line: [880, 120, 880, 320], w: 26, color: black });
    const image = paintScene(900, 530, shapes);
    const results = extractAll(image);
    const colorOf = r => r.est.style.color.map(Math.round);
    for (const [key, c] of [...Object.entries(grays), ['black', black]]) {
        const res = results.find(r => Math.hypot(...r.est.style.color.map((v, ch) => v - c[ch])) < 10);
        assert.ok(res, `family ${key} among ${results.map(colorOf).join('; ')}`);
        const want = dots.filter(d => String(d.family) === key);
        const m = matchDots(want, res.fit.dots, 2);
        assert.equal(m.recall, 1, `${key}: missed ${JSON.stringify(m.missed.map(i => want[i]))}`);
        assert.equal(res.fit.dots.length, want.length, `${key}: extra ${JSON.stringify(m.extra.map(j => res.fit.dots[j]))}`);
    }
});

const benchDir = set => new URL(`../${set}/`, import.meta.url).pathname;
const benchAbsent = (set, name) => !existsSync(benchDir(set) + name) && `${set}/${name} absent`;

function benchRow(set, name) {
    const truth = truthPixels(readTruthCsv(benchDir(set) + 'data.csv')).dots;
    const results = extractAll(readImage(benchDir(set) + name));
    const found = results.flatMap(r => r.fit.dots);
    // matched within half a radius, as the bench does
    return { found: found.length, m: matchDots(truth, found, 0.5 * results[0].fit.radius) };
}

test('regression: outlined opaque dots are a family (bench1 solid-opaque-stroke)', { skip: benchAbsent('bench1', 'solid-opaque-stroke.png') }, () => {
    // before: "No dot family found": every outline pixel read as neither the fill nor its background
    // 98 at first; 100–101 with splits (one true dot is fitted as two, 4.7 px off). Since the one-color fill plane
    // (outline.js), the outline's darker purple reads as a ring and the dots are fitted by their rings in drawing
    // order: 98, precision 1.000, recall 0.98, RMSE 0.034 px (the opaque path: 101, 0.990, 1.00, 0.551)
    const { found, m } = benchRow('bench1', 'solid-opaque-stroke.png');
    assert.ok(found >= 97 && found <= 101, `found ${found}`);
    assert.ok(m.recall >= 0.97, `recall ${m.recall}`);
    assert.ok(m.precision >= 0.99, `precision ${m.precision}`);
});

test('regression: bench2 solid-opaque-stroke', { skip: benchAbsent('bench2', 'solid-opaque-stroke.png') }, () => {
    // before: no family. 165 of 204: about 8 dots are under 0.1% visible, and dots in packed clusters are missed
    // because the boundaries between fill and outline inside a cluster aren't used
    // By its rings (outline.js): 172, precision 0.988, recall 0.833, RMSE 0.284 px (the opaque path with splits:
    // 179, 0.966, 0.848, 0.944; 165 before splits)
    const { found, m } = benchRow('bench2', 'solid-opaque-stroke.png');
    assert.ok(found >= 168, `found ${found}`);
    assert.ok(m.recall >= 0.82, `recall ${m.recall}`);
    assert.ok(m.precision >= 0.98, `precision ${m.precision}`);
});

test('regression: a cluster of one opaque color is not covered by too few dots (bench2 solid-opaque)', { skip: benchAbsent('bench2', 'solid-opaque.png') }, () => {
    // before: 180 of 204, each of several fitted dots stretched over two true ones; the truth's own dots, refined by
    // the fit, settle at 194 (about 10 are coincident or under 10% visible). Their union matched the image to 27 px,
    // so births found no site; splits replace such a dot by two
    const { found, m } = benchRow('bench2', 'solid-opaque.png');
    assert.ok(found >= 192 && found <= 200, `found ${found}`);
    assert.ok(m.recall >= 0.92, `recall ${m.recall}`);
});

test('regression: no outlined family made of a translucent family’s levels (economist)', { skip: !has('economist-canada-unfriendly-chart.png') && 'local test image absent' }, () => {
    // a red–blue lens inside red dots, with red's one-dot level around it, passed as an outlined opaque family
    const { estimates } = estimateFamilies(readPng(new URL('../bench-local/economist-canada-unfriendly-chart.png', import.meta.url).pathname));
    assert.deepEqual(estimates.map(e => e.familyMeasure.kind), ['translucent', 'translucent']);
});

// Fills inside one thin dark stroke (bench-nature engagement1's style): each dot drawn as a stroke disc, then its
// fill inside, in data order. The 0.8 px stroke never shows its own color at exact pixel coverage.
const STROKED = { width: 260, height: 200, radius: 7, stroke: 0.8, strokeColor: [20, 20, 60], fills: [[188, 186, 234], [4, 3, 87]] };

function strokedImage(seed) {
    const rng = createRng(seed);
    const { width, height, radius } = STROKED;
    const dots = [];
    while (dots.length < 30) {
        const d = { x: 20 + rng.uniform() * (width - 40), y: 20 + rng.uniform() * (height - 40), family: dots.length % 2 };
        // the first 16 apart, so each family has isolated dots; the rest anywhere
        if (dots.length < 16 && dots.some(e => Math.hypot(e.x - d.x, e.y - d.y) < 2 * radius + 6))
            continue;
        dots.push(d);
    }
    let img = solidImage(width, height, [255, 255, 255]);
    for (const d of dots) {
        for (const [color, r] of [[STROKED.strokeColor, radius], [STROKED.fills[d.family], radius - STROKED.stroke]]) {
            const m = createDotModel(img, { color, alpha: 1, radius: r, blur: 0 });
            m.add(d.x, d.y);
            img = m.image;
        }
    }
    return { image: toRgba(img), dots };
}

test('stroked fills: a light fill inside a dark stroke is a family, and each family keeps its own dots', () => {
    // before: one family, the dark one; the light dots' edges mix three colors (white, stroke, fill), which the
    // opaque test read as unknown, so none of them was found
    const { image, dots } = strokedImage(7);
    const results = extractAll(image, false);
    assert.equal(results.length, 2);
    const light = results.find(r => r.est.style.color[0] > 150);
    const dark = results.find(r => r.est.style.color[0] < 50);
    assert.ok(light?.est.familyMeasure.stroke, 'the light family is read with its stroke');
    for (const [res, k] of [[light, 0], [dark, 1]]) {
        const mine = matchDots(dots.filter(d => d.family === k), res.fit.dots, 2);
        const theirs = matchDots(dots.filter(d => d.family !== k), res.fit.dots, 2);
        assert.ok(mine.pairs.length >= 14, `family ${k}: ${mine.pairs.length} of 15`);
        assert.ok(theirs.pairs.length === 0, `family ${k} took ${theirs.pairs.length} of the other's dots`);
        near(res.fit.radius, STROKED.radius, 0.3, `family ${k} radius`);
    }
});

// Small fills in a black ring, as a low-resolution figure has them (bench-pmc funnel): fills of 3.7 px in dots of
// 5.5, and the plot's frame, 2 px wide, in the ring's black.
const SMALL_STROKED = { width: 600, height: 360, radius: 5.5, stroke: 1.8, strokeColor: [31, 28, 29], fill: [63, 129, 180], frame: [20, 20, 580, 340] };

function smallStrokedImage(seed) {
    const rng = createRng(seed);
    const { width, height, radius, frame } = SMALL_STROKED;
    const dots = [];
    while (dots.length < 60) {
        const d = { x: 40 + rng.uniform() * (width - 80), y: 40 + rng.uniform() * (height - 80) };
        if (dots.length < 50 && dots.some(e => Math.hypot(e.x - d.x, e.y - d.y) < 2 * radius + 8))
            continue;
        dots.push(d);
    }
    let img = solidImage(width, height, [255, 255, 255]);
    const [x0, y0, x1, y1] = frame;
    for (let y = y0; y < y1; y++) {
        for (let x = x0; x < x1; x++) {
            if (x < x0 + 2 || x >= x1 - 2 || y < y0 + 2 || y >= y1 - 2)
                img.data.set(SMALL_STROKED.strokeColor, 3 * (y * width + x));
        }
    }
    for (const d of dots) {
        for (const [color, r] of [[SMALL_STROKED.strokeColor, radius], [SMALL_STROKED.fill, radius - SMALL_STROKED.stroke]]) {
            const m = createDotModel(img, { color, alpha: 1, radius: r, blur: 0 });
            m.add(d.x, d.y);
            img = m.image;
        }
    }
    return { image: toRgba(img), dots };
}

test('small stroked fills are a family at their whole radius, and a 2 px frame in the stroke\'s color is no dot\'s', () => {
    // before: no family. The stroke reading took the fill's radius from the plain one, no family itself, whose
    // radius a scan for centers had raised past the whole dot (funnel: 6.33 px for fills of 3.6 in dots of 5.5), so
    // no center's inner disc was fill
    const { image, dots } = smallStrokedImage(3);
    const results = extractAll(image, false);
    assert.equal(results.length, 1);
    const [res] = results;
    assert.ok(res.est.familyMeasure.stroke, 'read with its stroke');
    near(res.fit.radius, SMALL_STROKED.radius, 0.4, 'radius');
    near(res.est.familyMeasure.fillRadius, SMALL_STROKED.radius - SMALL_STROKED.stroke, 0.5, 'fill radius');
    const m = matchDots(dots, res.fit.dots, 2);
    assert.ok(m.pairs.length >= 58, `${m.pairs.length} of 60 found`);
    assert.ok(res.fit.dots.length <= 61, `${res.fit.dots.length} dots`);
    // the frame reads as covered (it is the stroke's color), and the mask's thin marks were lines of 3 px or more:
    // every frame pixel away from the dots is thin now
    const { width, frame } = SMALL_STROKED;
    const [x0, y0, x1, y1] = frame;
    let onFrame = 0;
    let thin = 0;
    for (let x = x0 + 4; x < x1 - 4; x++) {
        for (const y of [y0, y0 + 1, y1 - 2, y1 - 1]) {
            onFrame++;
            thin += res.layers.thin[y * width + x];
        }
    }
    for (let y = y0 + 4; y < y1 - 4; y++) {
        for (const x of [x0, x0 + 1, x1 - 2, x1 - 1]) {
            onFrame++;
            thin += res.layers.thin[y * width + x];
        }
    }
    assert.equal(thin, onFrame, `${thin} of ${onFrame} frame pixels thin`);
});

const PMC = new URL('../bench-pmc/', import.meta.url).pathname;
const FUNNEL = 'funnel-10.1155-tbed-1096796';

test('regression: funnel, small blue fills in black rings are found, and none on the frame', { skip: !existsSync(`${PMC}${FUNNEL}.png`) && 'bench-pmc absent' }, () => {
    // before: no family; with the fills' own radius, 120 dots for 72, 45 of them strung along the right frame line
    const image = readImage(`${PMC}${FUNNEL}.png`);
    const truth = readTruthJson(`${PMC}truth/${FUNNEL}.json`).dots;
    const results = extractAll(image, false);
    assert.equal(results.length, 1);
    const found = results[0].fit.dots;
    const m = matchDots(truth, found, results[0].fit.radius / 2);
    assert.equal(m.recall, 1);
    assert.ok(found.length <= 75, `${found.length} dots for 72`);
    assert.ok(found.every(d => d.x < 1040), 'no dot on the right frame line');
});

const NATURE = new URL('../bench-nature/', import.meta.url).pathname;
const ENGAGEMENT = 'engagement1-10.1038-s41586-026-10967-w';

// the user's hand-checked dots (truth JSON, px)
// Written for the web capture; the PDF capture that replaced it (2× the pixels) was read as one scale-colored family
// fitted by its rings (22 of 49 dots) until stroked families found one at a time took precedence (familiesExplainInk)
test('regression: engagement1, lavender dots in a dark stroke are found, and the navy family no longer takes them', { skip: !(existsSync(`${NATURE}${ENGAGEMENT}.png`) && existsSync(`${NATURE}truth/${ENGAGEMENT}.json`)) && 'bench-nature engagement1 absent' }, () => {
    // before: one family (50, 55, 99) with 33 dots; no lavender dot found; the user's truth has 46 dots, 10 lavender
    const t = readTruthJson(`${NATURE}truth/${ENGAGEMENT}.json`);
    const light = new Set(t.families.filter(f => f.color[0] > 150).map(f => f.family)); // the light fill, whatever its label
    const truth = t.dots.map(d => ({ ...d, lavender: light.has(d.family) }));
    const results = extractAll(readImage(`${NATURE}${ENGAGEMENT}.png`), false);
    const lav = results.find(r => r.est.familyMeasure.stroke && r.est.style.color[0] > 150);
    assert.ok(lav, 'a stroked lavender family');
    const m = matchDots(truth.filter(t => t.lavender), lav.fit.dots, 3);
    assert.ok(m.pairs.length === 10 && m.extra.length <= 1, `lavender: ${m.pairs.length} of 10, ${m.extra.length} extra`);
    for (const r of results.filter(r => r !== lav))
        assert.equal(matchDots(truth.filter(t => t.lavender), r.fit.dots, 3).pairs.length, 0, 'another family took lavender dots');
});

const FANC2 = 'fanc2-10.1038-s41586-026-11035-z.png';

test('regression: fanc2, one gray family with black overlaps, and no family from its rims or decimal points', { skip: !existsSync(NATURE + FANC2) && 'bench-nature fanc2 absent' }, () => {
    // before: a rim-gray family at r 3.15 found 1105 "dots" on the lenses and text (136 s), and the gray family 38
    // of the 52 (every overlap is drawn in the text's black)
    const results = extractAll(readImage(NATURE + FANC2));
    assert.equal(results.length, 1, results.map(r => `${r.est.style.color.map(Math.round)} r ${r.est.style.radius}`).join('; '));
    near(results[0].fit.radius, 15.1, 0.3, 'radius');
    // the overlaps (one flat black) read as covered: 39 dots without that, 50 with it, of n = 52
    const n = results[0].fit.dots.length;
    assert.ok(n >= 48 && n <= 54, `${n} dots`);
});

const WAVE1 = 'wave1-10.1038-s41586-026-11145-8.png';

test('regression: wave1, decimal points beside regular dots are no family', { skip: !existsSync(NATURE + WAVE1) && 'bench-nature wave1 absent' }, () => {
    // before: a black family at r 3.23 (27 isolated decimal points, coverage 0.19) beside dots at r 14.9 found 887
    const { estimates } = estimateFamilies(readImage(NATURE + WAVE1));
    assert.equal(estimates.length, 3, estimates.map(e => `${e.style?.color.map(Math.round)} r ${e.style?.radius}`).join('; '));
    for (const e of estimates)
        near(e.style.radius, 14.9, 0.3, 'radius');
});

const HAIR2 = 'hair2-10.1038-s41586-026-11098-y.png';
const HAIR3 = 'hair3-10.1038-s41586-026-11098-y.png';

test('regression: hair3, pink fills in a black stroke are one family at their full radius, its five gray ones another; its decimal points are none', { skip: !existsSync(NATURE + HAIR3) && 'bench-nature hair3 absent' }, () => {
    // before: black decimal points (r 4.97, coverage 0.02) were the only family, 722 "dots"; the pink family's
    // isolated dots were outvoted by those specks (black is its stroke), its coverage diluted by all black ink, and
    // once found, its radius cut to 18 by the speck family's color excluded from its stroke.
    // The gray dots, five in one cluster, were no family: two centers in the black ring's reading, one of them strong,
    // where a family needed three; they are one as a color all in dots of the pink family's size
    const { estimates } = estimateFamilies(readImage(NATURE + HAIR3));
    assert.equal(estimates.length, 2, estimates.map(e => `${e.style?.color.map(Math.round)} r ${e.style?.radius}`).join('; '));
    assert.ok(Math.hypot(...estimates[0].style.color.map((v, ch) => v - [255, 175, 179][ch])) < 8);
    near(estimates[0].style.radius, 28.3, 0.5, 'radius');
    assert.ok(estimates[0].style.outline, 'outlined');
    assert.ok(Math.hypot(...estimates[1].style.color.map((v, ch) => v - [226, 225, 226][ch])) < 8);
    near(estimates[1].style.radius, 28.3, 1, 'the gray dots\' radius');
    assert.ok(estimates[1].style.outline, 'outlined');
});

test('regression: hair2, decimal points alone are no family; its clustered pink dots are one, by their outlines, and its white ones by their crescents', { skip: !existsSync(NATURE + HAIR2) && 'bench-nature hair2 absent' }, () => {
    // before: black specks (r 4.86, coverage 0.02, 4 isolated) made the only family, 170 "dots". Its nine pink dots
    // sit in two heavy clusters (one strong arc in the stroke model); the outline model finds them by their fills.
    // Its nine white-filled dots, the page's color, were no family: two rows with one whole dot each (two whole
    // fills, three needed; the crescents show under half their edge); then the fit, by the union, found 7.
    const image = readImage(NATURE + HAIR2);
    const results = extractAll(image);
    assert.equal(results.length, 2, results.map(r => `${r.est.style.color.map(Math.round)}`).join('; '));
    const pink = results.find(r => !r.est.familyMeasure.pageFill);
    assert.ok(Math.hypot(...pink.est.style.color.map((v, ch) => v - [255, 175, 179][ch])) < 8);
    near(pink.est.style.radius, 30.3, 0.8, 'radius');
    const white = results.find(r => r.est.familyMeasure.pageFill);
    assert.equal(white.fit.dots.length, 9, JSON.stringify(white.fit.dots.map(d => [Math.round(d.x), Math.round(d.y)])));
    // against the data (lanes 1–2 white, 3–4 pink): measured P 0.947 (the pink's one doubled dot), R 1.000
    const px = truthPixelsByTicks(readTruthCsv(NATURE + HAIR2.replace('.png', '.csv')), detectAxes(image));
    const centers = laneCentersFromDots(results.flatMap(r => r.fit.dots.map(d => d.x)), 4);
    const r = laneRow('hair2', px.dots, centers, results, 0);
    assert.ok(r.precision >= 0.9 && r.recall >= 0.95, `precision ${r.precision}, recall ${r.recall}, found ${r.found}`);
});

test('outlined grays in a row over a curve of one\'s own gray, and black dots on them: no dot for a stub of the curve or for an outline', () => {
    // phase1's top row, added to the scene above. Before: the dark gray curve showing for a few px between two dots
    // of another gray read as dot, hidden on its other sides, so the fit put a dark gray dot under those two for each
    // stub, and a second one where the curve leaves a dot; and a gray dot's black outline beside a black dot read as
    // black dot, which a second black disc a pixel off explained (phase1: 16 black dots fitted as 24).
    const black = [20, 21, 20];
    const R = 19.4;
    const rf = 17;
    const grays = { 228: [228, 228, 228], 187: [187, 188, 188], 136: [136, 136, 136], 82: [82, 82, 81] };
    const dots = [];
    const row = (key, y, xs) => xs.forEach(x => dots.push({ x, y, family: key }));
    const shapes = [{ line: [0, 60, 880, 75], w: 13, color: grays[228] }, { line: [0, 300, 880, 285], w: 13, color: grays[136] },
        { line: [20, 420, 880, 420], w: 2, color: black }];
    row(228, 64, [80, 200, 320, 440, 560]);
    row(187, 180, [80, 200, 320, 440, 560]);
    row(136, 296, [80, 200, 320, 440, 560]);
    row(82, 180, [680, 800]);
    row(82, 296, [680, 800]);
    row(82, 64, [680]);
    row('black', 420, [80, 200, 320, 440, 560, 680]);
    // the row, placed as the left of phase1's: a dark gray curve under mid grays 43 px apart, a few px of it showing
    // between them; dark gray dots over two of those; black dots over the mid grays' outlines. In drawing order.
    shapes.push({ line: [20, 560, 880, 560], w: 12, color: grays[82] });
    row(136, 560, [65, 108, 151, 195, 239, 280]);
    row(82, 560, [128, 257]);
    row('black', 560, [210, 292]);
    for (const d of dots) {
        shapes.push({ x: d.x, y: d.y, r: R, color: black });
        if (d.family !== 'black')
            shapes.push({ x: d.x, y: d.y, r: rf, color: grays[d.family] });
    }
    for (let k = 0; k < 30; k++)
        shapes.push({ x: 30 + 28 * k, y: 500, r: 4.9, color: black });
    shapes.push({ line: [880, 120, 880, 320], w: 26, color: black });
    const image = paintScene(900, 610, shapes);
    const results = extractAll(image);
    for (const [key, c] of [...Object.entries(grays), ['black', black]]) {
        const res = results.find(r => Math.hypot(...r.est.style.color.map((v, ch) => v - c[ch])) < 10);
        assert.ok(res, `family ${key} among ${results.map(r => r.est.style.color.map(Math.round)).join('; ')}`);
        const want = dots.filter(d => String(d.family) === key);
        const m = matchDots(want, res.fit.dots, 2);
        assert.equal(m.recall, 1, `${key}: missed ${JSON.stringify(m.missed.map(i => want[i]))}`);
        assert.equal(res.fit.dots.length, want.length, `${key}: extra ${JSON.stringify(m.extra.map(j => res.fit.dots[j]).map(d => [Math.round(d.x), Math.round(d.y), d.flags.join('+')]))}`);
    }
    // the grays' outlines beside a black dot (clear of every black disc by a pixel, within 4 px of one) are the
    // grays', but for the wedge where an outline runs into the black disc, as thick there as a dot's edge. Measured: 8
    // of 30 read as black dot; before, when the nearer of the two fills decided, 18.
    const blackD = results.find(r => Math.max(...r.est.style.color) < 40).est.D;
    const outlined = dots.filter(d => d.family !== 'black');
    const blacks = dots.filter(d => d.family === 'black');
    let beside = 0;
    let asBlack = 0;
    for (let y = 520; y < 600; y++) {
        for (let x = 0; x < 400; x++) {
            const onOutline = outlined.some(d => Math.abs(Math.hypot(x + 0.5 - d.x, y + 0.5 - d.y) - (R + rf) / 2) <= (R - rf) / 2 - 0.7);
            const toBlack = Math.min(...blacks.map(d => Math.hypot(x + 0.5 - d.x, y + 0.5 - d.y)));
            if (!onOutline || toBlack < R + 1 || toBlack > R + 4)
                continue;
            beside++;
            asBlack += blackD[y * image.width + x] > 0.5 ? 1 : 0;
        }
    }
    assert.ok(beside >= 10, `${beside} outline pixels beside the black dots`);
    assert.ok(asBlack <= 10, `${asBlack} of ${beside} outline pixels beside black dots read as black dot`);
});

const PHASE1 = 'phase1-10.1038-s41586-026-11163-6.png';

test('regression: phase1, all five families: four grays in black outlines and black dots', { skip: !existsSync(NATURE + PHASE1) && 'bench-nature phase1 absent' }, () => {
    // before: no family. White, gray and black are on one line, so the stroke model can't read a gray fill; the
    // outline model counted every black pixel (text, axes, a black family) as dot, was gated on black being thin over
    // the whole image, and the darker grays were taken for rims between the lighter one and its outline, then merged
    // with it as one translucent family over a black "region"; the same-gray curves behind the dots set the radius
    // (6.3 for 19). Then 2 of 5 (187 and 82, 50 for 97, R 0.299): 136's coverage 0.29, its curve's pixels counted;
    // 228's ring read as the page, its curve outweighing its dots; black measured as text (r 4.86, coverage 0.01).
    const image = readImage(NATURE + PHASE1);
    const results = extractAll(image);
    const estimates = results.map(r => r.est);
    const grays = estimates.filter(e => e.style?.outline && e.familyMeasure?.kind === 'opaque');
    const names = estimates.map(e => `${e.style?.color.map(Math.round)}`).join('; ');
    assert.equal(grays.length, 4, names);
    for (const e of grays) {
        near(e.style.radius, 19.4, 0.8, `radius of ${e.style.color.map(Math.round)}`);
        assert.ok(Math.max(...e.style.outline.color) < 40, 'a black outline');
    }
    const black = estimates.filter(e => !e.style?.outline && e.style && Math.max(...e.style.color) < 40);
    assert.equal(black.length, 1, names);
    near(black[0].style.radius, 19.4, 0.8, 'radius of black');
    // Against the hand truth. Before: 128 found for 97, P 0.625, 27 of the extras on the colorbar, beyond where the
    // x axis line ends: each family has one dot cut in half there, short of what a family must show to be clipped
    // (one passed). The cut is now read on the families together (layers.js plotClips): 101, P 0.812.
    // Then the top row, where the five families' dots lie over one another on their curves: stubs of a gray's own
    // curve between other dots (a hidden dot fitted for each), and the grays' outlines beside black dots (16 black
    // dots fitted as 24). Without them no fit is refitted larger for its close pairs: the radii ended at 20.2 and
    // 20.4 px for two families, and are now 19.2 to 19.8 (19.0 on the pixels). Measured: 85, P 0.965, R 0.845.
    const truth = readTruthJson(`${NATURE}truth/${PHASE1.replace('.png', '.json')}`);
    const r = groupRow('phase1', truth.dots, results, 0);
    assert.ok(r.recall >= 0.8 && r.precision >= 0.9, `precision ${r.precision}, recall ${r.recall}, found ${r.found}`);
    const found = results.flatMap(q => q.fit.dots);
    assert.ok(found.length <= 92, `${found.length} found`);
    for (const q of results)
        near(q.fit.radius, 19.4, 0.6, `fitted radius of ${q.est.style.color.map(Math.round)}`);
    const blackDots = results.find(q => Math.max(...q.est.style.color) < 40).fit.dots;
    assert.ok(blackDots.length <= 18, `${blackDots.length} black dots for 16`);
    // (the x axis line ends at 1289; the colorbar begins at 1313)
    assert.deepEqual(found.filter(d => d.x > 1300).map(d => [Math.round(d.x), Math.round(d.y)]), [], 'no dot on the colorbar');
});

test('regression: engagement1, five blues in one dark stroke are stroked families, not a scale; text is no family', { skip: !existsSync(`${NATURE}${ENGAGEMENT}.png`) && 'bench-nature engagement1 absent' }, () => {
    // before: read as one family on a color scale fitted by its rings (22 of 49); then, with the families found one
    // at a time preferred, black text's grays as a translucent family at r 3.5 (1362 "dots")
    const { estimates } = estimateFamilies(readImage(`${NATURE}${ENGAGEMENT}.png`));
    assert.ok(!estimates.some(e => e.scale), 'no scale reading');
    assert.ok(estimates.length >= 4, `${estimates.length} families`);
    for (const e of estimates)
        near(e.style.radius, 14.3, 0.6, `radius of ${e.style.color.map(Math.round)}`);
    // the gray of text's edges, 209, is within a cluster's reach of the lavender fills: read in the dark ring it
    // "had" their ten dots, all its thick pixels in discs of the families' size, and was a fifth family with no dot
    assert.ok(!estimates.some(e => Math.max(...e.style.color) - Math.min(...e.style.color) < 15), 'no gray family');
});

const KITE1 = 'kite1-10.1038-s41586-026-11194-z.png';

test('regression: kite1, cyan dots on a cyan line under a red band drawn over them', { skip: !existsSync(NATURE + KITE1) && 'bench-nature kite1 absent' }, () => {
    // before: 10 found for 25, red only. The cyan line, in the dots' color, set cyan's radius to its half width (3.5
    // for 12.3) and the scan never reached the dots; and the red dots, most inside the red band, made the band their
    // background, which families.js took for the page, so the band was no region and the cyan dots under it unseen
    const image = readImage(NATURE + KITE1);
    const results = extractAll(image);
    assert.equal(results.length, 2);
    for (const r of results)
        near(r.fit.radius, 12.3, 0.6, `radius of ${r.est.style.color.map(Math.round)}`);
    const csv = NATURE + KITE1.replace('.png', '.csv');
    if (!existsSync(csv))
        return;
    const px = truthPixelsByTicks(readTruthCsv(csv), detectAxes(results[0].jpegAuto ? restoreChroma(image) : image));
    const row = groupRow(KITE1, px.dots, results, 0);
    assert.ok(row.precision >= 0.88 && row.recall >= 0.92, `precision ${row.precision}, recall ${row.recall}, found ${row.found}`);
});

const MOLAR1 = 'molar1-10.1038-s41586-026-11021-5.png';

test('regression: molar1, black dots beside a gray label that fits as their level 1', { skip: !existsSync(NATURE + MOLAR1) && 'bench-nature molar1 absent' }, () => {
    // before: radius unknown. The "K_d = 119 nM" label (62, 62, 62) and the black dots (19, 20, 19) fit levels 1 and 2
    // of a translucent family at α 0.78, with no dot whole; none of the 191 black pieces lies near the gray
    const results = extractAll(readImage(NATURE + MOLAR1));
    assert.equal(results.length, 1);
    assert.equal(results[0].est.familyMeasure.kind, 'opaque');
    near(results[0].fit.radius, 9.6, 0.4, 'radius');
    const truthFile = NATURE + 'truth/' + MOLAR1.replace('.png', '.json');
    if (!existsSync(truthFile))
        return;
    const row = groupRow('molar1', readTruthJson(truthFile).dots, results, 0);
    assert.ok(row.precision >= 0.95 && row.recall >= 0.95, `P ${row.precision.toFixed(3)} R ${row.recall.toFixed(3)}, found ${row.found}`);
});

// Purple dots joined by a purple line, with black dots drawn over half of them (9 px off center) and black error bars
// down through those: delta1's layout. The black pixels at each seam read as purple's background, so the purple
// remnant was cut from the black dot over it, masked as part of the line, and a purple dot under black paid for every
// seam pixel it covered.
function blackOverPurple() {
    const W = 700;
    const H = 388;
    const R = 14;
    const PURPLE = [145, 77, 163];
    const BLACK = [19, 20, 19];
    const bg = solidImage(W, H, [255, 255, 255]);
    const purple = [];
    for (let k = 0; k < 10; k++)
        purple.push({ x: 40 + 48 * k, y: 120 + 20 * Math.sin(k) });
    const paint = (x, y, c) => bg.data.set(c, 3 * (Math.round(y) * W + Math.round(x)));
    for (let k = 0; k + 1 < purple.length; k++) {
        const [a, b] = [purple[k], purple[k + 1]];
        for (let t = 0; t <= 1; t += 0.002) {
            for (let dy = -1; dy <= 1; dy++) {
                for (let dx = -1; dx <= 1; dx++)
                    paint(a.x + t * (b.x - a.x) + dx, a.y + t * (b.y - a.y) + dy, PURPLE);
            }
        }
    }
    const under = createDotModel(bg, { color: PURPLE, alpha: 1, radius: R });
    under.addAll(purple);
    const covered = purple.filter((_, k) => k % 2);
    for (const d of covered) {
        for (let y = d.y - 40; y <= d.y + 22; y++) {
            for (let dx = -1; dx <= 1; dx++)
                under.image.data.set(BLACK, 3 * (Math.round(y) * W + Math.round(d.x) + dx));
        }
    }
    const black = [...covered.map(d => ({ x: d.x, y: d.y - 9 })), { x: 60, y: 40 }, { x: 200, y: 40 }, { x: 340, y: 40 }, { x: 480, y: 40 }];
    const over = createDotModel(under.image, { color: BLACK, alpha: 1, radius: R });
    over.addAll(black);
    return { image: toRgba(over.image), purple, black };
}

test('opaque dots partly under another family\'s are found: the seam is the upper family\'s edge, unknown for the lower', () => {
    const { image, purple, black } = blackOverPurple();
    const results = extractAll(image);
    const ofColor = c => results.find(r => Math.hypot(...r.est.style.color.map((v, ch) => v - c[ch])) < 20);
    const p = ofColor([145, 77, 163]);
    const b = ofColor([19, 20, 19]);
    assert.deepEqual(p.est.familyMeasure.under.map(c => c.map(Math.round)), [[19, 20, 19]], 'purple is under black');
    assert.deepEqual(b.est.familyMeasure.under, [], 'black is under nothing');
    // before: 5 of the 10 purple (none of those under black)
    for (const [r, truth, what] of [[p, purple, 'purple'], [b, black, 'black']]) {
        const m = matchDots(truth, r.fit.dots, 0.5 * r.fit.radius);
        assert.equal(m.pairs.length, truth.length, `${what}: ${m.pairs.length} of ${truth.length}, ${r.fit.dots.length} found`);
        assert.equal(r.fit.dots.length, truth.length, `${what}: ${r.fit.dots.length} found`);
    }
});

const DELTA1 = 'delta1-10.1038-s41586-026-10928-3.png';

test('regression: delta1, black dots joined by their line to the axes, beside as much black text', { skip: !existsSync(NATURE + DELTA1) && 'bench-nature delta1 absent' }, () => {
    // before: no black family. The dots, line, error bars and axes are one piece 14.1 px deep, 47% of the black; text
    // 3–5 px deep is 49%, so the radius scan went around the bold panel label "g" (8.1 px) and found no dots
    const results = extractAll(readImage(NATURE + DELTA1));
    const black = results.filter(r => r.est.familyMeasure.kind === 'opaque' && Math.max(...r.est.style.color) < 40);
    assert.equal(black.length, 1, `families ${results.map(r => r.est.style.color.map(Math.round)).join(' / ')}`);
    near(black[0].fit.radius, 14, 0.6, 'radius');
    const truthFile = NATURE + 'truth/' + DELTA1.replace('.png', '.json');
    if (!existsSync(truthFile))
        return;
    // (purple dots under black ones: 18 of 24 found before the drawing order; one almost wholly hidden is still missed)
    const row = groupRow('delta1', readTruthJson(truthFile).dots, results, 0);
    assert.ok(row.precision >= 0.92 && row.recall >= 0.95, `P ${row.precision.toFixed(3)} R ${row.recall.toFixed(3)}, found ${row.found}`);
});

const KITE2 = 'kite2-10.1038-s41586-026-11194-z.png';

test('regression: kite2, fills in rings of their darker shade beside a curve of the fill color', { skip: !existsSync(NATURE + KITE2) && 'bench-nature kite2 absent' }, () => {
    // before: no family. The curve (211, 136, 88) joins the fill's color cluster (209, 131, 82) and borders the page,
    // so the fill's surroundings were the page, not its rings (193, 88, 32, on the line from the page through the fill:
    // no stroke reading, and 3 px wide with the overlaps' lenses, so not thin for an outline); its radius was the
    // curve's half width, 6.9 for 19.5. The rings in the fill's darker shade now give the outline reading.
    const results = extractAll(readImage(NATURE + KITE2));
    assert.equal(results.length, 1);
    near(results[0].fit.radius, 19.5, 0.6, 'radius');
    const truthFile = NATURE + 'truth/' + KITE2.replace('.png', '.json');
    if (!existsSync(truthFile))
        return;
    const row = groupRow('kite2', readTruthJson(truthFile).dots, results, 0);
    assert.ok(row.precision >= 0.95 && row.recall >= 0.95, `P ${row.precision.toFixed(3)} R ${row.recall.toFixed(3)}, found ${row.found}`);
});

const CELLS1 = 'cells1-10.1038-s41586-026-10946-1.png';

test('regression: cells1, magenta dots over a gray bar and clustered are a family like the green, by their fills', { skip: !existsSync(NATURE + CELLS1) && 'bench-nature cells1 absent' }, () => {
    // before: only green. Magenta has no isolated whole dot (the gray connecting lines chain the dots, and most sit
    // over a gray bar), so its radius came from the distance transform (4.5 for 20), and its round fills were judged
    // in a window that took in their neighbors' fills, 2 px away across the outlines
    const { estimates } = estimateFamilies(readImage(NATURE + CELLS1));
    assert.equal(estimates.length, 2, estimates.map(e => `${e.style?.color.map(Math.round)}`).join('; '));
    const magenta = estimates.find(e => Math.hypot(...e.style.color.map((v, ch) => v - [196, 78, 174][ch])) < 10);
    assert.ok(magenta, 'a magenta family');
    near(magenta.style.radius, 20, 0.6, 'radius');
});

const DEGRADATION1 = 'degradation1-10.1038-s41586-026-10980-z.png';

test('regression: degradation1, mint and orange fills in gray outlines are two families, and the grays are none', { skip: !existsSync(NATURE + DEGRADATION1) && 'bench-nature degradation1 absent' }, () => {
    // before: one translucent gray "family" with no radius, its levels the gray line (166) and the outlines (136);
    // the fills' ring lay on its line, which ruled out the outline model for them
    const { estimates } = estimateFamilies(readImage(NATURE + DEGRADATION1));
    assert.equal(estimates.length, 2, estimates.map(e => `${e.style?.color.map(Math.round)} r ${e.style?.radius}`).join('; '));
    for (const target of [[136, 214, 185], [250, 148, 101]]) {
        const e = estimates.find(e => Math.hypot(...e.style.color.map((v, ch) => v - target[ch])) < 10);
        assert.ok(e, `a family of ${target}`);
        assert.ok(e.style.outline, 'outlined');
        near(e.style.radius, 21.7, 0.8, 'radius');
    }
});

test('regression: degradation1, overlapping orange dots are told apart by their outlines (fitted by rings)', { skip: !(existsSync(NATURE + DEGRADATION1) && existsSync(NATURE + 'truth/' + DEGRADATION1.replace('.png', '.json'))) && 'bench-nature degradation1 or its truth absent' }, () => {
    // before: fitted as a union of discs, where fill and outline alike are "dot", two overlapping dots of a family
    // were one blob: against the user's truth (70 dots with some interior visible, the 2 legend dots among them),
    // 58 found, P 0.966 R 0.800; the orange family by its rings, P 0.967 R 0.843. The mint's rings are too poorly
    // measured (rms 0.095) and stay a union.
    const truth = readTruthJson(NATURE + 'truth/' + DEGRADATION1.replace('.png', '.json'));
    const results = extractAll(readImage(NATURE + DEGRADATION1));
    const orange = results.find(r => r.est.style.color[0] > 200);
    assert.ok(orange.est.style.layered, 'the orange family is fitted by its rings');
    const row = groupRow('degradation1', truth.dots, results, 0);
    // (R 0.843 until a dot hidden under another at its place went with the deaths: the fit had two orange dots
    // 0.17 px apart at 828, 157, and the second matched the truth's dot at 838, 157, which is not found)
    assert.ok(row.recall >= 0.82 && row.precision >= 0.95, `P ${row.precision.toFixed(3)} R ${row.recall.toFixed(3)}`);
    const dots = orange.fit.dots;
    assert.ok(!dots.some((a, i) => dots.some((b, j) => j > i && Math.hypot(a.x - b.x, a.y - b.y) < 1)), 'no two orange dots at one place');
});

test('regression: beeswarm-boxplot, packed red and gray swarms are two families with their dots, gray not doubled', { skip: !has('beeswarm-boxplot.png') && 'local test image absent' }, () => {
    // before: no family. With exact colors JPEG fringes gave red a dark red "background"; with noisy colors the
    // colormap took gray plus its noise for a scale (one point held 97% of the votes); gray's radius came from the
    // distance transform, 6.2 for 9 (no dot isolated in a packed swarm), and at 8.2–8.7 the fit put two dots in
    // nearly every gray dot
    const results = extractAll(readImage(BENCH_LOCAL + 'beeswarm-boxplot.png'));
    assert.equal(results.length, 2, results.map(r => `${r.est.style.color.map(Math.round)}`).join('; '));
    const gray = results.find(r => r.est.style.color[0] < 200);
    const red = results.find(r => r.est.style.color[0] > 200);
    assert.ok(gray && red, 'a gray and a red family');
    assert.ok(gray.fit.radius > 8.8, `gray radius ${gray.fit.radius}`);
    const close = gray.fit.dots.filter(d => d.flags.includes('close')).length;
    assert.ok(close < 0.1 * gray.fit.dots.length, `${close} of ${gray.fit.dots.length} gray dots in close pairs`);
    assert.ok(gray.fit.dots.length >= 270 && gray.fit.dots.length <= 310, `${gray.fit.dots.length} gray dots`);
    // 378 since opaque pairs a JPEG's edges doubled are merged (fit.js MERGE_OPAQUE_RADII): the 5 that went were extras
    assert.ok(red.fit.dots.length >= 370 && red.fit.dots.length <= 430, `${red.fit.dots.length} red dots`);
});

const COMMENT1 = 'comment1-2026-10-22.png';

test('regression: comment1, small round specks beside regular dots are no family', { skip: !existsSync(NATURE + COMMENT1) && 'bench-nature comment1 absent' }, () => {
    // before: the dotted gridlines' black specks (r 2.5, 9 isolated) made a second family beside the orange dots
    // (r 9.2), and fitting it over all the text took minutes
    const { estimates } = estimateFamilies(readImage(NATURE + COMMENT1));
    assert.equal(estimates.length, 1, estimates.map(e => `${e.style?.color.map(Math.round)} r ${e.style?.radius}`).join('; '));
    assert.ok(Math.hypot(...estimates[0].style.color.map((v, ch) => v - [235, 91, 37][ch])) < 6);
    near(estimates[0].style.radius, 9.2, 0.3, 'radius');
});

test('opaqueFamilyEstimate: a family measured with a user-given outline finds its dots and leaves the others\' alone', () => {
    const { image, dots } = strokedImage(7);
    const [light, dark] = STROKED.fills;
    const fit = est => {
        const style = { ...est.style, radius: STROKED.radius };
        const layers = buildMask(image, { ...est, style });
        return fitDots({ image, style, line: est.familyMeasure.line, mask: layers.mask, D: est.D, imgT: est.fitT });
    };
    // the user's outline: the stroke's own color, though no pixel shows it, and its width
    const est = opaqueFamilyEstimate(image, light, { stroke: { color: STROKED.strokeColor, fillRadius: STROKED.radius - STROKED.stroke }, others: [dark], ringsToOthers: true });
    assert.deepEqual(est.style.outline, { color: STROKED.strokeColor, inner: (STROKED.radius - STROKED.stroke) / est.style.radius });
    const res = fit(est);
    const mine = matchDots(dots.filter(d => d.family === 0), res.dots, 2);
    const theirs = matchDots(dots.filter(d => d.family === 1), res.dots, 2);
    assert.ok(mine.pairs.length >= 14 && theirs.pairs.length === 0, `light ${mine.pairs.length} of 15, ${theirs.pairs.length} of the dark`);
    // and the dark family, told of the light fill, gives the light dots' rings to them
    const darkRes = fit(opaqueFamilyEstimate(image, dark, { others: [light], ringsToOthers: true }));
    assert.equal(matchDots(dots.filter(d => d.family === 0), darkRes.dots, 2).pairs.length, 0);
});

test('truth JSON: a dots export converts to truth, families and fills from its labels', () => {
    const csv = 'family,x_px,y_px,se_noise_x_px,se_noise_y_px,visible,depth,flags,fill\n'
        + '1 #323763,197.403,106.808,0.001,0.001,0.92,1,,\n'
        + '2 #bab8f7,253.5,234.0,,,1.00,1,weak,2? #a0c4ff\n';
    assert.equal(isDotsExport(csv), true);
    assert.equal(isDotsExport('# x ticks: 1, 2\nx,y\n1,2\n'), false);
    const t = truthFromExport(csv, 'a.png', 'abc');
    assert.deepEqual(t.families, [{ family: 0, color: [50, 55, 99] }, { family: 1, color: [186, 184, 247] }]);
    assert.deepEqual(t.dots, [{ x: 197.403, y: 106.808, family: 0 }, { x: 253.5, y: 234, family: 1, fill: 1, fillSure: false }]);
});

const HAIR1 = 'hair1-10.1038-s41586-026-11098-y';
const TUMOUR1 = 'tumour1-10.1038-s41586-026-11051-z';

test('regression: hair1, white dots in black outlines on a white bar are a family, without letters or gaps between pink dots', { skip: !existsSync(`${NATURE}${HAIR1}.png`) && 'bench-nature hair1 absent' }, () => {
    // before: no family (the page's color is no cluster); then 26 dots, 15 of them in letters' counters (P, B, D,
    // the title's) and the gaps among the pink dots
    const image = readImage(`${NATURE}${HAIR1}.png`);
    const results = extractAll(image);
    const white = results.filter(r => r.est.familyMeasure.pageFill);
    assert.equal(white.length, 1);
    const dots = white[0].fit.dots;
    assert.equal(dots.length, 11, JSON.stringify(dots.map(d => [Math.round(d.x), Math.round(d.y)])));
    // all in the PBS lane, between the title and the axis labels
    assert.ok(dots.every(d => d.x > 470 && d.x < 570 && d.y > 400 && d.y < 1000), JSON.stringify(dots.map(d => [Math.round(d.x), Math.round(d.y)])));
    // the pink family keeps its dots
    assert.equal(results.find(r => !r.est.familyMeasure.pageFill).fit.dots.length, 16);
    // against the data (lane 1 PBS, 11 dots; lane 2 DT, 18): measured P 1.000, R 0.931
    const px = truthPixelsByTicks(readTruthCsv(`${NATURE}${HAIR1}.csv`), detectAxes(image));
    const centers = laneCentersFromDots(results.flatMap(r => r.fit.dots.map(d => d.x)), 2);
    const r = laneRow(HAIR1, px.dots, centers, results, 0);
    assert.ok(r.precision >= 0.95 && r.recall >= 0.9, `precision ${r.precision}, recall ${r.recall}, found ${r.found}`);
});

test('regression: tumour1, white dots in black outlines over gray and red bars are one family', { skip: !existsSync(`${NATURE}${TUMOUR1}.png`) && 'bench-nature tumour1 absent' }, () => {
    // before: no family; then a light gray cluster near the page's color made a second, duplicate family; then 41 for
    // 44, P 0.976 R 0.909: lane 8's six dots at zero, four of them crescents, read as page (2 found), and the 6 of
    // 0.9964 was a dot. Measured now: 44 for 44, P 1.000 R 1.000.
    const image = readImage(`${NATURE}${TUMOUR1}.png`);
    const results = extractAll(image);
    assert.equal(results.length, 1);
    assert.ok(results[0].est.familyMeasure.pageFill);
    const px = truthPixelsByTicks(readTruthCsv(`${NATURE}${TUMOUR1}.csv`), detectAxes(image));
    const centers = laneCentersFromDots(results[0].fit.dots.map(d => d.x), 8);
    const r = laneRow(TUMOUR1, px.dots, centers, results, 0);
    assert.ok(r.precision >= 0.95 && r.recall >= 0.97, `precision ${r.precision}, recall ${r.recall}, found ${r.found}`);
});

const PLOS = new URL('../bench-plos/', import.meta.url).pathname;

// a PLOS figure's results against its hand truth (truth JSON, px); skipped when the set is absent
function plos(name) {
    const truth = readTruthJson(`${PLOS}truth/${name}.json`);
    const results = extractAll(readImage(PLOS + truth.image));
    const found = results.flatMap(r => r.fit.dots);
    return { results, found, m: matchDots(truth.dots, found, 0.5 * Math.min(...results.map(r => r.fit.radius))) };
}
const plosSkip = name => !existsSync(`${PLOS}truth/${name}.json`) && `bench-plos ${name} absent`;

const MICROPLASTICS1 = 'microplastics1-10.1371-journal.pone.0362569';
const MYELOID1 = 'myeloid1-10.1371-journal.pone.0362747';
const SCORE1 = 'score1-10.1371-journal.pone.0361996';
const BOXDOTS1 = 'boxdots1-10.1371-journal.pone.0361404';

test('regression: microplastics1, a thin gray border bar is no translucent level of the black dots', { skip: plosSkip(MICROPLASTICS1) }, () => {
    // before: "The dot radius is unknown": the 3 px bar at 48 and the black dots read as levels 1 and 3 of one
    // translucent family (α 0.81, no isolated dot), so black was never tried as an opaque family
    const { results, found, m } = plos(MICROPLASTICS1);
    assert.equal(results.length, 1);
    assert.equal(results[0].est.style.alpha, 1);
    assert.equal(found.length, 24);
    assert.equal(m.recall, 1);
});

test('regression: myeloid1, touching teal dots crossed by lines are a family at the orange family\'s radius', { skip: plosSkip(MYELOID1) }, () => {
    // before: 9 of 18, teal no family: its arcs reach 0.72 at most, under the 0.75 of a strong center
    const { results, found, m } = plos(MYELOID1);
    assert.equal(results.length, 2);
    assert.deepEqual(results.map(r => r.fit.dots.length), [9, 9]);
    assert.equal(found.length, 18);
    assert.equal(m.recall, 1);
});

test('a color in two dots of a found family\'s size is a family; at another size, or with a line in it, none', () => {
    // bench-plos synthetic: a group of two points beside groups of seven and five; a dot family needed three strong
    // centers, and the two were left out
    const TEAL = [0, 150, 160];
    const gray = { color: [90, 90, 90], alpha: 1, radius: 8, points: Array.from({ length: 12 }, (_, i) => ({ x: 40 + 44 * (i % 6), y: 40 + 44 * Math.floor(i / 6) })) };
    const scene = (radius, lines = []) => synthGraph({
        width: 320, height: 240, background: [255, 255, 255], blur: 0.5, lines,
        families: [gray, { color: TEAL, alpha: 1, radius, points: [{ x: 70.3, y: 160.6 }, { x: 210.7, y: 172.2 }] }],
    }).image;
    // (with the line, the color's mean is 10 units off the dots')
    const teal = image => estimateStyle(image).candidates.find(c => Math.hypot(...c.seed.map((v, ch) => v - TEAL[ch])) < 20);
    const found = teal(scene(8));
    assert.ok(found, 'two dots at the gray dots\' radius');
    near(found.radius, 8, 0.3, 'radius');
    assert.equal(found.dots, 2);
    // two round marks of another size are not a group of the plot's points
    assert.equal(teal(scene(12)), undefined);
    // … nor two dots with a line in their color, which is most of its pixels
    assert.equal(teal(scene(8, [{ points: [[30, 215], [290, 215]], width: 3, color: TEAL }])), undefined);
});

const SYNTHETIC = 'synthetic-10.1371-journal.pone.0361775';

test('regression: synthetic, a group of two ringed dots is a family at the translucent families\' radius', { skip: plosSkip(SYNTHETIC) }, () => {
    // before: 13 found for 16, teal (2 dots, one touching a salmon dot) no family: one strong center in the reading by its ring,
    // where a family needed three. Olive, two dots 2.5 px apart whose one blob is their overlap's color, is still none.
    const { results, m } = plos(SYNTHETIC);
    assert.equal(results.length, 3);
    const teal = results[2];
    assert.equal(teal.est.style.alpha, 1);
    assert.ok(Math.hypot(...teal.est.style.color.map((v, ch) => v - [59, 202, 206][ch])) < 6, `color ${teal.est.style.color}`);
    near(teal.fit.radius, 11.6, 0.3, 'radius');
    assert.equal(teal.fit.dots.length, 2);
    for (const [x, y] of [[397.15, 116.24], [468.48, 116.23]])
        assert.ok(teal.fit.dots.some(d => Math.hypot(d.x - x, d.y - y) <= 0.5), `teal dot at ${x}, ${y}`);
    // (the truth's 14 dots of the three families found; olive's 2 missed)
    assert.equal(m.missed.length, 2);
});

test('regression: score1, teal dots near a gray line\'s color are not the gray\'s', { skip: plosSkip(SCORE1) }, () => {
    // before: 316 for 168, a gray family (131, dashed line and label leaders) taking the teal dots, 39.9 units away
    const { results, found, m } = plos(SCORE1);
    assert.equal(results.length, 1);
    assert.equal(m.extra.length, 0);
    assert.ok(m.recall >= 0.98, `recall ${m.recall}, found ${found.length}`);
});

test('regression: boxdots1, black dots over a red box and beside gray box outlines', { skip: plosSkip(BOXDOTS1) }, () => {
    // before: 152 for 268 (recall 0.37). A box edge read as the x axis masked the plot above it; then the gray box
    // outlines and median lines (52, read as 81% covered by black) took 64 false dots, and the dots over the red box,
    // read as other ink, lost their edges to the mask (recall 0.74)
    const { results, found, m } = plos(BOXDOTS1);
    assert.equal(results.length, 1);
    assert.ok(m.recall >= 0.95, `recall ${m.recall}, found ${found.length}`);
    assert.ok(m.precision >= 0.92, `precision ${m.precision}, found ${found.length}`);
});

const STEP1 = 'step1-10.1371-journal.pone.0360641';

test('regression: step1, blue dots and hollow red rings under two bands, one over the other', { skip: plosSkip(STEP1) }, () => {
    // before: the blue CI band (in the dots' own hue) was the blue family's level 1, the radius unknown; then, the
    // band read as a region under the dots and the red one too (their overlap voted "under"), 133 blue dots for 100
    // (P 0.55) and 41 hollow rings for 75. The bands are layers, red then blue over the points: divided out in order,
    // the blue family is α 0.50, and the rings under the red band show.
    const { results, found, m } = plos(STEP1);
    assert.equal(results.length, 2);
    const [blue, rings] = results;
    assert.ok(Math.abs(blue.est.style.alpha - 0.5) <= 0.05, `alpha ${blue.est.style.alpha}`);
    assert.ok(Math.abs(blue.est.style.radius - 5.4) <= 0.3, `radius ${blue.est.style.radius}`);
    assert.equal(rings.est.style.layered.fillAlpha, 0, 'hollow');
    assert.ok(Math.abs(rings.est.style.radius - 6.4) <= 0.3, `ring radius ${rings.est.style.radius}`);
    assert.ok(m.precision >= 0.95, `precision ${m.precision}, found ${found.length}`);
    assert.ok(m.recall >= 0.7, `recall ${m.recall}, found ${found.length}`);
});

const LANDING1 = 'landing1-10.1371-journal.pone.0362399';

test('regression: landing1, translucent dots over violins of their own hue, every dot touching a line or an outline', { skip: !existsSync(`${PLOS}${LANDING1}.png`) && 'bench-plos landing1 absent' }, () => {
    // before: no family. Each violin was its family's one level, the dots' colors beyond it; a dot over the page and
    // one over the violin give α (0.75) and the color, the violin is a region under the dots, and with no dot
    // isolated the radius comes from the dots' edges (11.5 and 11.8 px)
    const image = readImage(`${PLOS}${LANDING1}.png`);
    const results = extractAll(image);
    assert.equal(results.length, 2);
    for (const r of results) {
        // (a measure style.js overBand builds itself: it must say what kind it is, as measureFamily's do)
        assert.equal(r.est.familyMeasure.kind, 'translucent');
        assert.ok(r.est.familyMeasure.band, 'read over a band');
        assert.ok(Math.abs(r.est.style.alpha - 0.75) <= 0.03, `alpha ${r.est.style.alpha}`);
        assert.ok(r.est.style.radius >= 10.5 && r.est.style.radius <= 12.5, `radius ${r.est.style.radius}`);
    }
    const px = truthPixelsByTicks(readTruthCsv(`${PLOS}${LANDING1}.csv`), detectAxes(image));
    const centers = px.laneCenters ?? laneCentersFromDots(results.flatMap(r => r.fit.dots.map(d => d.x)), 2);
    const row = laneRow(LANDING1, px.dots, centers, results, 0);
    assert.ok(row.precision >= 0.85 && row.recall >= 0.95, `precision ${row.precision}, recall ${row.recall}, found ${row.found}`);
});

test('regression: dots in a tint of another family\'s color are a family, not that family\'s blend with the page', () => {
    // song-cents: light blue dots, 144, 180, 226, lie 0.69 of the way from the white page to the mid blue's 93, 146,
    // 205, where an antialiased rim of the mid blue lies too, and the cluster was dismissed as that blend. A rim is
    // all edge; this cluster is the insides of dots (interior 0.53).
    const tint = [143, 180, 227];
    const base = () => {
        const rng = createRng(5);
        return {
            width: 520, height: 400, background: [255, 255, 255], blur: 0.5,
            families: [{ color: [93, 146, 205], alpha: 1, radius: 9, points: scatterPoints(rng, { n: 40, box: [40, 40, 480, 360] }) }],
            rng,
        };
    };
    const colors = image => estimateFamilies(image).estimates.map(e => e.style.color.map(Math.round));
    const near = (c, d) => Math.hypot(c[0] - d[0], c[1] - d[1], c[2] - d[2]) <= 3;
    const two = base();
    two.families.push({ color: tint, alpha: 1, radius: 9, points: scatterPoints(two.rng, { n: 25, box: [40, 40, 480, 360] }) });
    const found = colors(synthGraph(two).image);
    assert.equal(found.length, 2, JSON.stringify(found));
    assert.ok(found.some(c => near(c, tint)) && found.some(c => near(c, [93, 146, 205])), JSON.stringify(found));
    // a band in that tint under the dots has insides too, and is no dots: it stays other ink
    const banded = base();
    banded.gridlines = [{ points: [[20, 200], [500, 200]], width: 70, color: tint }];
    const one = colors(synthGraph(banded).image);
    assert.ok(!one.some(c => near(c, tint)), JSON.stringify(one));
    assert.ok(one.some(c => near(c, [93, 146, 205])), JSON.stringify(one));
});

test('regression: a family whose color is another\'s seen through the text\'s gray is a family: text is no layer', () => {
    // song-cents: navy dots at 15, 39, 66 are the labels' gray, 88, plus 0.45 of the mid blue's offset from the
    // page, to 0.3 units, as mid blue dots under a translucent gray band would be. The gray is strokes a few px
    // wide, nothing a dot lies under.
    const rng = createRng(9);
    const spec = {
        width: 520, height: 400, background: [255, 255, 255], blur: 0.5,
        families: [
            { color: [93, 146, 205], alpha: 1, radius: 9, points: scatterPoints(rng, { n: 35, box: [40, 40, 480, 360] }) },
            { color: [15, 39, 66], alpha: 1, radius: 9, points: scatterPoints(rng, { n: 15, box: [40, 40, 480, 360] }) },
        ],
        // strokes in the gray, as text and thin lines have it
        lines: [60, 120, 180, 240, 300, 350].map(y => ({ points: [[30, y], [490, y + 20]], width: 4, color: [88, 88, 88] })),
    };
    const colors = estimateFamilies(synthGraph(spec).image).estimates.map(e => e.style.color.map(Math.round));
    const near = (c, d) => Math.hypot(c[0] - d[0], c[1] - d[1], c[2] - d[2]) <= 4;
    assert.ok(colors.some(c => near(c, [15, 39, 66])), JSON.stringify(colors));
    assert.ok(colors.some(c => near(c, [93, 146, 205])), JSON.stringify(colors));
});

const BLUE = [58, 111, 165];
const ORANGE = [223, 105, 59];

// Opaque dots of 10 px drawn in order on a white page, each over a disc of the page's color `stroke` px larger when
// they have a stroke.
function drawnInOrder(width, height, dots, stroke = 0) {
    let image = solidImage(width, height, [255, 255, 255]);
    for (const d of dots) {
        if (stroke) {
            const ring = createDotModel(image, { color: [255, 255, 255], alpha: 1, radius: 10 + stroke });
            ring.addAll([d]);
            image = ring.image;
        }
        const fill = createDotModel(image, { color: d.color, alpha: 1, radius: 10 });
        fill.addAll([d]);
        image = fill.image;
    }
    return toRgba(image);
}

const familyOf = (list, color) => list.find(f => Math.hypot(...(f.seed ?? f.est.style.color).map((v, ch) => v - color[ch])) < 20);

test('regression: a family drawn first, nearly every dot partly under another family\'s, is a family at that one\'s radius', () => {
    // 14 orange dots, 12 of them a third under a blue one 10.8 px away, and blue dots alone
    const scene = (dx, dy) => {
        const dots = [];
        for (let i = 0; i < 12; i++) {
            const x = 70 + 110 * (i % 4);
            const y = 60 + 90 * Math.floor(i / 4);
            dots.push({ x, y, color: ORANGE }, { x: x + dx, y: y + dy, color: BLUE });
        }
        for (let i = 0; i < 5; i++)
            dots.push({ x: 40 + 90 * i, y: 330, color: [BLUE, ORANGE][i % 2] });
        return dots;
    };
    const dots = scene(10, 4);
    const image = drawnInOrder(520, 370, dots);
    // by the dot test the orange has the two dots alone, a fifth of its pixels in them: a center's disc must be the
    // family's own fill, and a third of each other one is blue. They are dots partly under other ink.
    const orange = measureOpaque(scanOf(image), ORANGE);
    assert.equal(orange.dots, 2);
    assert.ok(orange.coverage < 0.3, `coverage ${orange.coverage}`);
    assert.equal(orange.dotsUnder, 12);
    assert.ok(orange.coverageUnder > 0.9, `coverage with them ${orange.coverageUnder}`);
    // … which count for a color at a found family's radius
    const found = familyOf(estimateStyle(image).candidates, ORANGE);
    assert.ok(found, 'the orange is a family');
    near(found.radius, 10, 0.3, 'radius');
    const results = extractAll(image);
    for (const [color, what] of [[ORANGE, 'orange'], [BLUE, 'blue']]) {
        const truth = dots.filter(d => d.color === color);
        const r = familyOf(results, color);
        assert.equal(matchDots(truth, r.fit.dots, 5).pairs.length, truth.length, `${what}: ${r.fit.dots.length} found for ${truth.length}`);
        assert.equal(r.fit.dots.length, truth.length, what);
    }
    // too little of a dot is none: 8 px from the blue dot over it, most of its edge shows and under half its disc
    const slivers = drawnInOrder(520, 370, scene(8, 0));
    assert.equal(measureOpaque(scanOf(slivers), ORANGE).dotsUnder, 0);
    assert.ok(!familyOf(estimateStyle(slivers).candidates, ORANGE), 'the orange slivers are a family');
    // nor is a dot with a bite of the page out of it one partly under other ink: the page is no ink, and with a
    // line of other ink through what is left, what the line doesn't hold is still not filled
    for (const line of [false, true]) {
        const bitten = [...dots];
        for (let i = 0; i < 4; i++)
            bitten.push({ x: 60 + 110 * i, y: 290, color: ORANGE }, { x: 72 + 110 * i, y: 290, color: [255, 255, 255] });
        const bites = drawnInOrder(520, 370, bitten);
        for (let i = 0; line && i < 4; i++) {
            for (let y = 276; y < 304; y++) {
                for (let x = 55 + 110 * i; x < 57 + 110 * i; x++)
                    bites.data.set([...BLUE, 255], 4 * (y * 520 + x));
            }
        }
        assert.equal(measureOpaque(scanOf(bites), ORANGE).dotsUnder, 12, `with bitten dots${line ? ' and lines' : ''}`);
    }
});

const GREEN = [77, 175, 74];
const TEAL = [8, 176, 175];
const INK = [7, 7, 7];

// `n` dots of one color in a grid, each a third under a dot of another 10.8 px away: [x, y] of the lower ones
function underPairs(n, lower, upper) {
    const dots = [];
    for (let i = 0; i < n; i++) {
        const x = 60 + 110 * (i % 4);
        const y = 50 + 80 * Math.floor(i / 4);
        dots.push({ x, y, color: lower }, { x: x + 10, y: y + 4, color: upper });
    }
    return dots;
}

function fillRect(image, x0, y0, x1, y1, color) {
    for (let y = y0; y < y1; y++) {
        for (let x = x0; x < x1; x++)
            image.data.set([...color, 255], 4 * (y * image.width + x));
    }
}

// the dots of a color among those fitted for it, matched within 5 px
function foundOf(results, dots, color) {
    const truth = dots.filter(d => d.color === color);
    const r = familyOf(results, color);
    return { r, truth: truth.length, matched: r ? matchDots(truth, r.fit.dots, 5).pairs.length : 0 };
}

test('regression: a family with no whole dot keeps the dots\' radius when it is measured again for the fit', () => {
    // 12 green dots, each a third under a blue one, and 4 blue dots alone. before: found as a sibling at 9.9 px, the
    // green was measured again beside the blue at 5.9, the depth of what shows of it, and fitted as 60 dots
    const dots = [...underPairs(12, GREEN, BLUE), ...[0, 1, 2, 3].map(i => ({ x: 60 + 110 * i, y: 290, color: BLUE }))];
    const results = extractAll(drawnInOrder(520, 340, dots));
    const green = foundOf(results, dots, GREEN);
    // the radius of the blue, which has whole dots
    near(green.r.est.familyMeasure.siblingRadius, 10, 0.05, 'radius taken');
    near(green.r.est.style.radius, 10, 0.05, 'radius');
    assert.ok(green.r.est.notes.some(n => /10\.0 px, is that of the image's other families; the dots' own shapes gave 9\.9 px, at which 12 of these dots show/.test(n)), green.r.est.notes.join(' '));
    assert.equal(green.matched, 12);
    assert.equal(green.r.fit.dots.length, 12);
    assert.equal(foundOf(results, dots, BLUE).matched, 16);
    // with no whole blue dot either (those alone touch in pairs), each keeps the radius it was found at
    const touching = [...underPairs(12, GREEN, BLUE), ...[0, 1, 2, 3].flatMap(i => [{ x: 60 + 110 * i, y: 290, color: BLUE }, { x: 79 + 110 * i, y: 290, color: BLUE }])];
    const both = extractAll(drawnInOrder(520, 340, touching));
    for (const [color, what, count] of [[GREEN, 'green', 12], [BLUE, 'blue', 20]]) {
        const f = foundOf(both, touching, color);
        assert.equal(f.r.est.isolated.length, 0, what);
        assert.equal(f.r.est.familyMeasure.siblingRadius, undefined, what);
        near(f.r.est.style.radius, 10, 0.2, `${what}'s radius`);
        assert.equal(f.matched, count, what);
        assert.equal(f.r.fit.dots.length, count, what);
    }
});

test('a color whose shapes give another radius is measured at the found families\' (after aerobic)', () => {
    // 8 green dots, each a third under a blue one, 6 blue dots alone, and a legend with an oblong key of each color:
    // the green's thickest shape is its key, 27 px high, and at 14.5 px it shows no dot
    const dots = [...underPairs(8, GREEN, BLUE), ...[0, 1, 2, 3, 4, 5].map(i => ({ x: 60 + 70 * i, y: 290, color: BLUE }))];
    const image = drawnInOrder(560, 340, dots);
    fillRect(image, 470, 40, 515, 67, GREEN);
    fillRect(image, 470, 90, 515, 117, BLUE);
    const alone = measureOpaque(scanOf(image), GREEN);
    near(alone.radius, 14.5, 0.3, 'radius from its shapes');
    assert.equal(alone.dots + alone.dotsUnder, 0);
    // at the blue dots' 10 px its 8 dots show, while the blue is ink not known to be dots
    const results = extractAll(image);
    const green = foundOf(results, dots, GREEN);
    assert.ok(green.r, 'the green is a family');
    near(green.r.est.style.radius, 10, 0.05, 'radius');
    near(green.r.est.familyMeasure.siblingRadius, 10, 0.05, 'radius taken');
    assert.ok(green.r.est.notes.some(n => /10\.0 px, is that of the image's other families, at which 8 of these dots show/.test(n)), green.r.est.notes.join(' '));
    // (the keys are fitted too: this scene has no legend a mask would find)
    assert.equal(green.matched, 8);
    assert.equal(foundOf(results, dots, BLUE).matched, 14);
});

test('dots in the color of the figure\'s text, at a found family\'s radius with whole dots of their own, are a family (after reward)', () => {
    // 12 dark dots, each a third under a teal one, 2 alone, 16 teal dots, and 12 strokes 4 px thick in the dark color:
    // a tenth of the dark pixels is in the dots
    const scene = whole => {
        const dots = [...underPairs(12, INK, TEAL), ...[0, 1, 2, 3, 4, 5].map(i => ({ x: 60 + 80 * i, y: 290, color: i < whole ? INK : TEAL }))];
        const image = drawnInOrder(520, 470, dots);
        for (let k = 0; k < 12; k++)
            fillRect(image, 40, 330 + 10 * k, 480, 334 + 10 * k, INK);
        return { dots, image };
    };
    const { dots, image } = scene(2);
    const dark = measureOpaque(scanOf(image), INK);
    assert.equal(dark.isolated.length, 2);
    assert.equal(dark.dots + dark.dotsUnder, 14);
    assert.ok(dark.coverageUnder < 0.2, `coverage ${dark.coverageUnder}`);
    assert.ok(!isDotFamily(dark));
    // … but of its pixels in pieces as thick as a dot, all are
    const results = extractAll(image);
    for (const [color, what, count] of [[INK, 'dark', 14], [TEAL, 'teal', 16]]) {
        const f = foundOf(results, dots, color);
        assert.ok(f.r, `${what} is a family`);
        near(f.r.fit.radius, 10, 0.1, `${what}'s radius`);
        assert.equal(f.matched, count, what);
        assert.equal(f.r.fit.dots.length, count, what);
    }
    // with no whole dot of its own the dark color stays text: three centers at a family's radius are in letters too
    assert.ok(!familyOf(extractAll(scene(0).image), INK), 'the dark color is a family with no whole dot');
});

test('regression: a color near a family\'s, which reads that family\'s dots as its own, is no family (after shannon)', () => {
    // 20 slate dots, and a frame and rows of short strokes in a gray 25 units from the slate: measured alone, the gray
    // has the 20 dots too. before: two families, the gray one fitted beside the slate with no dot
    const SLATE = [112, 129, 148];
    const dots = [];
    for (let i = 0; i < 20; i++)
        dots.push({ x: 60 + 90 * (i % 5) + 7 * (i % 3), y: 50 + 60 * Math.floor(i / 5) + 5 * (i % 2) });
    const model = createDotModel(solidImage(520, 400, [255, 255, 255]), { color: SLATE, alpha: 1, radius: 8 });
    model.addAll(dots);
    const image = toRgba(model.image);
    const GRAY = [130, 130, 131];
    for (const box of [[20, 10, 500, 13], [20, 290, 500, 293], [20, 10, 23, 293], [497, 10, 500, 293]])
        fillRect(image, ...box, GRAY);
    for (let k = 0; k < 6; k++) {
        for (let x = 40; x < 470; x += 14)
            fillRect(image, x, 310 + 12 * k, x + 9, 313 + 12 * k, GRAY);
    }
    const gray = measureOpaque(scanOf(image), GRAY);
    assert.equal(gray.dots, 20);
    assert.ok(isDotFamily(gray));
    const results = extractAll(image);
    assert.equal(results.length, 1);
    assert.ok(Math.hypot(...results[0].est.style.color.map((v, ch) => v - SLATE[ch])) < 5, `color ${results[0].est.style.color}`);
    assert.equal(matchDots(dots, results[0].fit.dots, 1).pairs.length, 20);
    assert.equal(results[0].fit.dots.length, 20);
});

const SHANNON = 'shannon-10.3390-nu15092152.png';

test('regression: shannon, four colors of dots over box plots outlined in their colors', { skip: !existsSync(PMC + SHANNON) && 'bench-pmc shannon absent' }, () => {
    // before: 67 found for 111. The whiskers' pale edges were gridlines and the plot ended at the third box's
    // whisker; and a fifth family, the gray of the frame's edges, had the gray-blue dots for its own
    const image = readImage(PMC + SHANNON);
    const axes = detectAxes(image);
    assert.equal(axes.lines.filter(l => !l.dark).length, 0);
    const results = extractAll(image);
    assert.equal(results.length, 4);
    const truth = readTruthJson(PMC + 'truth/' + SHANNON.replace('.png', '.json')).dots;
    const found = results.flatMap(r => r.fit.dots);
    const m = matchDots(truth, found, 0.5 * Math.max(...results.map(r => r.fit.radius)));
    assert.equal(m.recall, 1);
    // (the extras: dots touching a line of their color or each other, fitted twice)
    assert.ok(found.length <= 117, `${found.length} dots`);
});

const AEROBIC = 'aerobic-10.1371-journal.pcbi.1012736.png';
const REWARD = 'reward-10.1523-JNEUROSCI.0020-24.2024.png';

test('regression: aerobic, seven colors of dots and a legend of squares in them', { skip: !existsSync(PMC + AEROBIC) && 'bench-pmc aerobic absent' }, () => {
    // before: 95 found for 222 in 3 families. The legend's 23 px squares passed as whole dots, the only ones of blue,
    // yellow, green and red, whose radius they set to 13.8 px for 8.6; at that no dot showed
    const results = extractAll(readImage(PMC + AEROBIC));
    // (red, nearly every dot under a blue one, shows no three: it is left to be picked by hand)
    assert.equal(results.length, 6);
    for (const r of results)
        near(r.fit.radius, 8.61, 0.1, `radius of ${r.est.style.color.map(Math.round)}`);
    const truth = readTruthJson(PMC + 'truth/' + AEROBIC.replace('.png', '.json')).dots;
    const found = results.flatMap(r => r.fit.dots);
    const m = matchDots(truth, found, 0.5 * 8.61);
    assert.ok(found.length >= 170 && m.precision >= 0.97, `${found.length} found, precision ${m.precision}`);
});

test('regression: reward, black dots at the teal dots\' radius beside large black text', { skip: !existsSync(PMC + REWARD) && 'bench-pmc reward absent' }, () => {
    // before: the teal family alone, 308 found for 342. 0.11 of the black pixels lie in the 15 black centers
    const { estimates } = estimateFamilies(restoreChroma(readImage(PMC + REWARD)), { jpeg: true });
    assert.equal(estimates.length, 2);
    const black = estimates.find(e => e.style.color.every(v => v < 40));
    assert.ok(black, 'a black family');
    near(black.style.radius, 12.3, 0.3, 'its radius');
    assert.equal(black.isolated.length, 2);
});

const TSNE = 'tsne-10.1371-journal.pone.0362839';

test('regression: tsne, the gray where two gray dots lie on each other is no family of its own', { skip: plosSkip(TSNE) }, () => {
    // gray dots of 154 and, where one lies on another, 127: past the family's color on its line from the page. With
    // 5 such stacks whole, the 127 passed as a sibling by its thick pixels, got 3 dots, and cost the gray family 7
    const { estimates } = estimateFamilies(readImage(`${PLOS}${TSNE}.png`));
    assert.equal(estimates.length, 5);
    assert.ok(!estimates.some(e => e.style.color.every(v => Math.abs(v - 127) < 10)), 'a family of 127');
});

test('regression: stroked dots of a color with none alone: the whole dots among the cut ones give its radius', () => {
    // piles of one to four dots with white strokes, 11 px from one to the next, the colors by turns: every dot alone
    // is blue, and most of the orange's pieces are crescents 5 px deep under the strokes of the dots over them. The
    // radius from the pieces' depths was 4.97, the scan of larger radii ended at 7.95, and the scan around the whole
    // dots' 10.5 was left out because it began at 7.88, within the first.
    const rng = createRng(5);
    const dots = [];
    let k = 0;
    for (let gy = 0; gy < 4; gy++) {
        for (let gx = 0; gx < 6; gx++) {
            const size = 1 + (gx + 2 * gy) % 4;
            let x = 50 + 75 * gx + 6 * rng.uniform();
            let y = 50 + 75 * gy + 6 * rng.uniform();
            for (let i = 0; i < size; i++) {
                dots.push({ x, y, color: [BLUE, ORANGE][k++ % 2] });
                const a = 2 * Math.PI * rng.uniform();
                x += 11 * Math.cos(a);
                y += 11 * Math.sin(a);
            }
        }
    }
    const image = drawnInOrder(520, 370, dots, 2.5);
    const found = familyOf(estimateStyle(image).candidates, ORANGE);
    assert.ok(found, 'the orange is a family');
    assert.equal(found.isolated.length, 0);
    near(found.radius, 10, 1.2, 'radius, before the fit refits it');
    const truth = dots.filter(d => d.color === ORANGE);
    const r = familyOf(extractAll(image), ORANGE);
    const hit = matchDots(truth, r.fit.dots, 5).pairs.length;
    assert.ok(hit >= truth.length - 2, `${hit} of ${truth.length} orange dots, ${r.fit.dots.length} found`);
    near(r.fit.radius, 10, 0.8, 'radius by the fit');
});

test('regression: blurred black dots in a gray frame are whole dots at their radius (biomarker)', () => {
    // before: beyond 2 px from the dots' own color their soft edges pass through the frame's gray, which is other ink:
    // every dot had a ring of pixels read as the frame's, none was whole, and the radius came from the distance
    // transform, small (biomarker: 4.5 for 6.0, and 216 dots fitted for 83)
    const rng = createRng(5);
    const dots = [];
    for (let i = 0; i < 24; i++)
        dots.push({ x: 60 + 52 * (i % 6) + 8 * rng.uniform(), y: 60 + 48 * Math.floor(i / 6) + 8 * rng.uniform() });
    // (… and a column of touching ones, as a beeswarm has)
    for (let i = 0; i < 6; i++)
        dots.push({ x: 400 + 11 * (i % 2), y: 60 + 11 * i });
    const model = createDotModel(solidImage(460, 300, [255, 255, 255]), { color: [3, 3, 3], alpha: 1, radius: 6, blur: 1.3 });
    model.addAll(dots);
    const image = toRgba(model.image);
    // the frame, 4 px of gray
    const gray = (x0, y0, x1, y1) => {
        for (let y = y0; y < y1; y++) {
            for (let x = x0; x < x1; x++)
                image.data.set([190, 190, 190, 255], 4 * (y * image.width + x));
        }
    };
    gray(20, 20, 440, 24);
    gray(20, 276, 440, 280);
    gray(20, 20, 24, 280);
    gray(436, 20, 440, 280);
    const { estimates } = estimateFamilies(image);
    const black = familyOf(estimates.map(e => ({ est: e })).filter(f => f.est.style), [3, 3, 3])?.est;
    assert.ok(black, 'the black dots are a family');
    assert.ok(black.isolated.length >= 20, `${black.isolated.length} whole dots`);
    assert.ok(Math.abs(black.style.radius - 6) < 0.25, `radius ${black.style.radius}`);
});

// Opaque dots with a thin dark outline, drawn in order and then blurred, as a small figure enlarged is: discs at 4× and
// averaged down, then a Gaussian of sd `sigma` px. The outline is too thin to show its own color anywhere.
function blurredOutlined(width, height, dots, { fill, outline, rFill, rOut, sigma }) {
    const S = 4;
    const sum = new Float32Array(3 * width * height);
    for (let y = 0; y < height * S; y++) {
        for (let x = 0; x < width * S; x++) {
            let c = [255, 255, 255];
            for (const d of dots) {
                const r = Math.hypot((x + 0.5) / S - d.x, (y + 0.5) / S - d.y);
                if (r <= rOut)
                    c = r <= rFill ? fill : outline;
            }
            const p = 3 * (Math.floor(y / S) * width + Math.floor(x / S));
            for (let ch = 0; ch < 3; ch++)
                sum[p + ch] += c[ch] / (S * S);
        }
    }
    const half = Math.ceil(3 * sigma);
    const kernel = Array.from({ length: 2 * half + 1 }, (_, i) => Math.exp(-((i - half) ** 2) / (2 * sigma * sigma)));
    const norm = kernel.reduce((a, b) => a + b, 0);
    const pass = (src, dx, dy) => {
        const dst = new Float32Array(src.length);
        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
                for (let k = -half; k <= half; k++) {
                    const xx = Math.min(width - 1, Math.max(0, x + k * dx));
                    const yy = Math.min(height - 1, Math.max(0, y + k * dy));
                    for (let ch = 0; ch < 3; ch++)
                        dst[3 * (y * width + x) + ch] += kernel[k + half] / norm * src[3 * (yy * width + xx) + ch];
                }
            }
        }
        return dst;
    };
    const out = pass(pass(sum, 1, 0), 0, 1);
    const data = new Uint8ClampedArray(4 * width * height);
    for (let p = 0; p < width * height; p++)
        data.set([out[3 * p], out[3 * p + 1], out[3 * p + 2], 255], 4 * p);
    return { width, height, data };
}

test('regression: a thin black outline, blurred, is read by its own color, out to the end of its soft edge (stiffness)', () => {
    // before: the outline's color was its darkest pixels', a mix of black and the fill (navy, for a blue fill). The
    // outer side of every ring, black mixed with the page, is gray: off the triangle of page, fill and navy, so
    // unknown, as was everything more than 4 px from the fill, and the mask took the rings whole
    const rng = createRng(11);
    const dots = [];
    for (let i = 0; i < 30; i++)
        dots.push({ x: 50 + 60 * (i % 6) + 14 * rng.uniform(), y: 50 + 56 * Math.floor(i / 6) + 14 * rng.uniform() });
    const fill = [49, 179, 253];
    const image = blurredOutlined(430, 340, dots, { fill, outline: [0, 0, 0], rFill: 6.4, rOut: 8.9, sigma: 1 });
    // the outline's color as the scan has it: the darkest of the rings' inner side, a third fill (stiffness's 16, 57,
    // 83 around this fill)
    const seen = fill.map(v => 0.32 * v);
    const { P, unknown } = familyRings(image, fill, seen, [[255, 255, 255]], []);
    let ring = 0;
    let unread = 0;
    let most = 0;
    for (const d of dots) {
        for (let y = Math.floor(d.y - 12); y <= d.y + 12; y++) {
            for (let x = Math.floor(d.x - 12); x <= d.x + 12; x++) {
                const r = Math.hypot(x + 0.5 - d.x, y + 0.5 - d.y);
                if (r < 6.4 || r > 11)
                    continue;
                ring++;
                unread += unknown[y * image.width + x];
                most = Math.max(most, P[y * image.width + x]);
            }
        }
    }
    assert.ok(unread <= 0.02 * ring, `${unread} of ${ring} ring pixels unknown`);
    // (a full outline reads 0.45: these, never more than 0.8 black, less)
    assert.ok(most > 0.25 && most < 0.4, `the rings' darkest reads ${most}`);
});

const STIFFNESS = 'stiffness-10.1371-journal.pone.0359831';

test('regression: stiffness, thin dark outlines around blue fills are read as rings', { skip: !existsSync(`${PLOS}${STIFFNESS}.png`) && 'bench-plos absent' }, () => {
    // before: 39 dots found of 305. The outline's color as the scan has it, 16, 57, 83, is its darkest pixels', black
    // mixed with the fill; the rings' outer sides are gray, off the triangle of page, fill and that navy, and their
    // soft ends lie more than 4 px from the fill: unknown both, so the mask took the rings whole
    const image = readImage(`${PLOS}${STIFFNESS}.png`);
    const { P, unknown } = familyRings(image, [49.4, 178.7, 252.8], [16, 57, 83], [[255, 255, 255]], []);
    // the dots with no other within three radii: their rings, from the fill's edge to the end of the soft edge
    const dots = readTruthJson(`${PLOS}truth/${STIFFNESS}.json`).dots;
    const alone = dots.filter(d => dots.every(e => e === d || Math.hypot(e.x - d.x, e.y - d.y) > 26));
    assert.ok(alone.length >= 20, `${alone.length} dots alone`);
    let ring = 0;
    let unread = 0;
    for (const d of alone) {
        for (let y = Math.floor(d.y - 12); y <= d.y + 12; y++) {
            for (let x = Math.floor(d.x - 12); x <= d.x + 12; x++) {
                const r = Math.hypot(x + 0.5 - d.x, y + 0.5 - d.y);
                if (r < 6.5 || r > 11)
                    continue;
                ring++;
                unread += unknown[y * image.width + x];
            }
        }
    }
    assert.ok(unread <= 0.05 * ring, `${unread} of ${ring} ring pixels unknown`);
    const found = outlinesInP(P, image.width, image.height, unknown);
    assert.ok(found, 'a ring model');
    assert.ok(Math.abs(found.ring.radius - 8.8) <= 0.4 && found.ring.rms < 0.035, `radius ${found.ring.radius}, rms ${found.ring.rms}`);
    assert.ok(found.isolated.length >= 10, `${found.isolated.length} whole rings`);
});

test('regression: a small family\'s color is a cluster in a large image too (cattle)', () => {
    // before: a cluster needed a thousandth of the image, 4339 px of cattle's 2174 × 1996: its 25 red dots of 7.6 px
    // have 3074 at their color, and red was never tried. The floor stops growing at 1500 px
    const width = 2200;
    const height = 2000;
    const data = new Uint8ClampedArray(4 * width * height).fill(255);
    const blob = (x0, y0, w, h, color) => {
        for (let y = y0; y < y0 + h; y++) {
            for (let x = x0; x < x0 + w; x++)
                data.set(color, 4 * (y * width + x));
        }
    };
    blob(100, 100, 200, 100, [3, 3, 255]); // 20 000 px
    blob(900, 900, 60, 50, [255, 3, 3]); // 3000 px
    blob(1500, 300, 40, 30, [3, 160, 3]); // 1200 px: under the floor
    const clusters = colorClusters({ width, height, data }, [[255, 255, 255]]).map(c => c.color.map(Math.round).join(','));
    assert.deepEqual(clusters, ['3,3,255', '255,3,3']);
});

test('isDotFamily: many whole isolated dots stand in for strong arcs, with a tenth of the color in them', () => {
    // corneal's black: blurred dots, no arc at 0.75 of the edge, 25 whole dots, beside bold labels
    const corneal = { radius: 8.3, isolated: Array.from({ length: 25 }, () => ({ r: 8.3 })), strongDots: 0, coverage: 0.28 };
    assert.equal(isDotFamily(corneal), true);
    assert.equal(isDotFamily({ ...corneal, isolated: corneal.isolated.slice(0, 9) }), false, 'nine whole dots');
    assert.equal(isDotFamily({ ...corneal, coverage: 0.05 }), false, 'a color that is nearly all text');
    assert.equal(isDotFamily({ ...corneal, isolated: [], strongDots: 3, coverage: 0.4 }), true, 'as before: three strong arcs');
});

const CORNEAL = 'corneal-10.1371-journal.pone.0362558';

test('regression: corneal, blurred black dots and thick open circles of the same black, beside bold black labels', { skip: !existsSync(`${PLOS}${CORNEAL}.png`) && 'bench-plos absent' }, () => {
    // before: "No dot family found"; then the dots alone
    const results = extractAll(readImage(`${PLOS}${CORNEAL}.png`));
    assert.equal(results.length, 2);
    assert.ok(results[0].est.style.color.every(v => v < 10), `color ${results[0].est.style.color}`);
    const truth = readTruthJson(`${PLOS}truth/${CORNEAL}.json`);
    const black = truth.dots.filter(d => d.family === truth.families.find(f => f.color[0] < 10).family);
    assert.equal(black.length, 63);
    const m = matchDots(black, results[0].fit.dots, 0.5 * results[0].fit.radius);
    assert.equal(m.pairs.length, 63, `${m.pairs.length} of 63 found`);
    assert.ok(results[0].fit.dots.length <= 67, `${results[0].fit.dots.length} dots in all`);
    // the open circles: every one, and no more than the zeros of a label inside the plot besides
    assert.ok(results[1].est.candidates[0].hollow, 'hollow rings');
    const open = truth.dots.filter(d => !black.includes(d));
    assert.equal(open.length, 58);
    const rings = matchDots(open, results[1].fit.dots, 0.5 * results[1].fit.radius);
    assert.equal(rings.pairs.length, 58, `${rings.pairs.length} of 58 circles found`);
    assert.ok(results[1].fit.dots.length <= 62, `${results[1].fit.dots.length} circles in all`);
});

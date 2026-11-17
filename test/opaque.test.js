import { test } from 'node:test';
import assert from 'node:assert/strict';
import { existsSync } from 'node:fs';
import { createRng } from '../src/core/rng.js';
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
import { extractAll, groupRow, laneRow } from '../tools/bench.mjs';
import { detectAxes } from '../src/core/axes.js';

const near = (a, b, tol, what) => assert.ok(Math.abs(a - b) <= tol, `${what}: ${a} vs ${b} (±${tol})`);

function fitFamily(image, k, candidates) {
    const est = estimateStyle(image, { family: k, candidates });
    const layers = buildMask(image, est);
    const fit = fitDots({ image, style: est.style, line: est.familyMeasure.line, mask: layers.mask, D: est.D, imgT: est.fitT });
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

const TESTGRAPHS = new URL('../testgraphs/', import.meta.url).pathname;
const has = name => existsSync(TESTGRAPHS + name);

test('families of the real images', { skip: !has('cantons-haemocytes-dotplot.png') && 'local test images absent' }, () => {
    const kinds = name => estimateStyle(readPng(TESTGRAPHS + name)).candidates.map(c => c.kind);
    assert.deepEqual(kinds('dog-ages.png'), ['translucent']);
    assert.deepEqual(kinds('economist-canada-unfriendly-chart.png'), ['translucent', 'translucent']);
    const cantons = estimateStyle(readPng(TESTGRAPHS + 'cantons-haemocytes-dotplot.png')).candidates;
    assert.equal(cantons.length, 4);
    assert.ok(cantons.every(c => Math.abs(c.radius - 6.37) < 0.1));
    assert.equal(kinds('Laser-Asyllable-Maturity.png').length, 3);
    // the gray dots share their color with antialiased text, which once made white their background and text their
    // radius; the red dots read as half of pink and once took pink dots as their own
    const nicest = estimateStyle(readPng(TESTGRAPHS + 'nicest language.png')).candidates;
    assert.ok(nicest.length >= 4);
    assert.ok(nicest.some(c => Math.hypot(...c.seed.map((v, ch) => v - [202, 201, 191][ch])) < 6));
    // violin's dots sit on its shading and on white
    assert.equal(kinds('violin-beeswarm.png').length, 1);
});

test('regression: Laser’s black family fits its dots, not its text or bar', { skip: !has('Laser-Asyllable-Maturity.png') && 'local test image absent' }, () => {
    const image = readPng(TESTGRAPHS + 'Laser-Asyllable-Maturity.png');
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
    const { estimates } = estimateFamilies(readPng(new URL('../testgraphs/economist-canada-unfriendly-chart.png', import.meta.url).pathname));
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

test('regression: hair3, pink fills in a black stroke are one family at their full radius; its decimal points are none', { skip: !existsSync(NATURE + HAIR3) && 'bench-nature hair3 absent' }, () => {
    // before: black decimal points (r 4.97, coverage 0.02) were the only family, 722 "dots"; the pink family's
    // isolated dots were outvoted by those specks (black is its stroke), its coverage diluted by all black ink, and
    // once found, its radius cut to 18 by the speck family's color excluded from its stroke
    const { estimates } = estimateFamilies(readImage(NATURE + HAIR3));
    assert.equal(estimates.length, 1, estimates.map(e => `${e.style?.color.map(Math.round)} r ${e.style?.radius}`).join('; '));
    assert.ok(Math.hypot(...estimates[0].style.color.map((v, ch) => v - [255, 175, 179][ch])) < 8);
    near(estimates[0].style.radius, 28.3, 0.5, 'radius');
    assert.ok(estimates[0].style.outline, 'outlined');
});

test('regression: hair2, decimal points alone are no family; its clustered pink dots are one, by their outlines', { skip: !existsSync(NATURE + HAIR2) && 'bench-nature hair2 absent' }, () => {
    // before: black specks (r 4.86, coverage 0.02, 4 isolated) made the only family, 170 "dots". Its nine pink dots
    // sit in two heavy clusters (one strong arc in the stroke model); the outline model finds them by their fills.
    // Its white-filled dots are the page's color: no family yet.
    const { estimates } = estimateFamilies(readImage(NATURE + HAIR2));
    assert.equal(estimates.length, 1, estimates.map(e => `${e.style?.color.map(Math.round)}`).join('; '));
    assert.ok(Math.hypot(...estimates[0].style.color.map((v, ch) => v - [255, 175, 179][ch])) < 8);
    near(estimates[0].style.radius, 30.3, 0.8, 'radius');
});

const PHASE1 = 'phase1-10.1038-s41586-026-11163-6.png';

test('regression: phase1, gray fills in black outlines are families of their own', { skip: !existsSync(NATURE + PHASE1) && 'bench-nature phase1 absent' }, () => {
    // before: no family. White, gray and black are on one line, so the stroke model can't read a gray fill; the
    // outline model counted every black pixel (text, axes, a black family) as dot, was gated on black being thin over
    // the whole image, and the darker grays were taken for rims between the lighter one and its outline, then merged
    // with it as one translucent family over a black "region"; the same-gray curves behind the dots set the radius
    // (6.3 for 19). Still missing: gray 136 (coverage 0.29) and 228, and the black family (text's color).
    const { estimates } = estimateFamilies(readImage(NATURE + PHASE1));
    const grays = estimates.filter(e => e.style?.outline && e.familyMeasure?.kind === 'opaque');
    assert.ok(grays.length >= 2, estimates.map(e => `${e.style?.color.map(Math.round)}`).join('; '));
    for (const e of grays) {
        near(e.style.radius, 19.4, 0.8, `radius of ${e.style.color.map(Math.round)}`);
        assert.ok(Math.max(...e.style.outline.color) < 40, 'a black outline');
    }
});

test('regression: engagement1, five blues in one dark stroke are stroked families, not a scale; text is no family', { skip: !existsSync(`${NATURE}${ENGAGEMENT}.png`) && 'bench-nature engagement1 absent' }, () => {
    // before: read as one family on a color scale fitted by its rings (22 of 49); then, with the families found one
    // at a time preferred, black text's grays as a translucent family at r 3.5 (1362 "dots")
    const { estimates } = estimateFamilies(readImage(`${NATURE}${ENGAGEMENT}.png`));
    assert.ok(!estimates.some(e => e.scale), 'no scale reading');
    assert.ok(estimates.length >= 4, `${estimates.length} families`);
    for (const e of estimates)
        near(e.style.radius, 14.3, 0.6, `radius of ${e.style.color.map(Math.round)}`);
});

const KITE1 = 'kite1-10.1038-s41586-026-11194-z.png';

test('regression: kite1, a translucent family without a measured radius is kept for the user to set it', { skip: !existsSync(NATURE + KITE1) && 'bench-nature kite1 absent' }, () => {
    // the speck filter for translucent families once compared a null radius as 0 and dropped the teal family
    const { estimates } = estimateFamilies(readImage(NATURE + KITE1));
    assert.ok(estimates.some(e => e.familyMeasure?.kind === 'translucent' && !(e.style.radius > 0)), estimates.map(e => `${e.style?.color.map(Math.round)} r ${e.style?.radius}`).join('; '));
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
    assert.ok(row.recall >= 0.83 && row.precision >= 0.95, `P ${row.precision.toFixed(3)} R ${row.recall.toFixed(3)}`);
});

test('regression: beeswarm-boxplot, packed red and gray swarms are two families with their dots, gray not doubled', { skip: !has('beeswarm-boxplot.png') && 'local test image absent' }, () => {
    // before: no family. With exact colors JPEG fringes gave red a dark red "background"; with noisy colors the
    // colormap took gray plus its noise for a scale (one point held 97% of the votes); gray's radius came from the
    // distance transform, 6.2 for 9 (no dot isolated in a packed swarm), and at 8.2–8.7 the fit put two dots in
    // nearly every gray dot
    const results = extractAll(readImage(TESTGRAPHS + 'beeswarm-boxplot.png'));
    assert.equal(results.length, 2, results.map(r => `${r.est.style.color.map(Math.round)}`).join('; '));
    const gray = results.find(r => r.est.style.color[0] < 200);
    const red = results.find(r => r.est.style.color[0] > 200);
    assert.ok(gray && red, 'a gray and a red family');
    assert.ok(gray.fit.radius > 8.8, `gray radius ${gray.fit.radius}`);
    const close = gray.fit.dots.filter(d => d.flags.includes('close')).length;
    assert.ok(close < 0.1 * gray.fit.dots.length, `${close} of ${gray.fit.dots.length} gray dots in close pairs`);
    assert.ok(gray.fit.dots.length >= 270 && gray.fit.dots.length <= 310, `${gray.fit.dots.length} gray dots`);
    assert.ok(red.fit.dots.length >= 380 && red.fit.dots.length <= 430, `${red.fit.dots.length} red dots`);
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
    // before: no family; then a light gray cluster near the page's color made a second, duplicate family
    const image = readImage(`${NATURE}${TUMOUR1}.png`);
    const results = extractAll(image);
    assert.equal(results.length, 1);
    assert.ok(results[0].est.familyMeasure.pageFill);
    const px = truthPixelsByTicks(readTruthCsv(`${NATURE}${TUMOUR1}.csv`), detectAxes(image));
    const centers = laneCentersFromDots(results[0].fit.dots.map(d => d.x), 8);
    const r = laneRow(TUMOUR1, px.dots, centers, results, 0);
    assert.ok(r.precision >= 0.9 && r.recall >= 0.85, `precision ${r.precision}, recall ${r.recall}, found ${r.found}`);
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
const SCORE1 = 'score1-10.1371_journal.pone.0361996';
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
        assert.ok(Math.abs(r.est.style.alpha - 0.75) <= 0.03, `alpha ${r.est.style.alpha}`);
        assert.ok(r.est.style.radius >= 10.5 && r.est.style.radius <= 12.5, `radius ${r.est.style.radius}`);
    }
    const px = truthPixelsByTicks(readTruthCsv(`${PLOS}${LANDING1}.csv`), detectAxes(image));
    const centers = px.laneCenters ?? laneCentersFromDots(results.flatMap(r => r.fit.dots.map(d => d.x)), 2);
    const row = laneRow(LANDING1, px.dots, centers, results, 0);
    assert.ok(row.precision >= 0.85 && row.recall >= 0.95, `precision ${row.precision}, recall ${row.recall}, found ${row.found}`);
});

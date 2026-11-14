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
import { readTruthCsv, truthPixels } from '../tools/truth-csv.mjs';
import { readTruthJson, truthFromExport, isDotsExport } from '../tools/truth-json.mjs';
import { extractAll } from '../tools/bench.mjs';

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
// todo: written for the 2026-11 web capture; the PDF capture that replaced it (2× the pixels, hash 76da23c6) is claimed
// by outlineEstimate as one scale-colored family fitted by its rings (22 of 49 dots), like phase1's outlined grays
test('regression: engagement1, lavender dots in a dark stroke are found, and the navy family no longer takes them', { skip: !(existsSync(`${NATURE}${ENGAGEMENT}.png`) && existsSync(`${NATURE}truth/${ENGAGEMENT}.json`)) && 'bench-nature engagement1 absent', todo: 'the PDF capture is read as outlined dots on a color scale' }, () => {
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

test('regression: hair2, decimal points alone are no family', { skip: !existsSync(NATURE + HAIR2) && 'bench-nature hair2 absent' }, () => {
    // before: black specks (r 4.86, coverage 0.02, 4 isolated) made the only family, 170 "dots". Its pink dots are
    // nine in two heavy clusters (one strong arc), and its white-filled dots are the page's color: no family yet.
    const { estimates } = estimateFamilies(readImage(NATURE + HAIR2));
    assert.ok(estimates.every(e => !e.style || Math.hypot(...e.style.color) > 100), estimates.map(e => `${e.style?.color.map(Math.round)}`).join('; '));
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

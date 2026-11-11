import { test } from 'node:test';
import assert from 'node:assert/strict';
import { existsSync } from 'node:fs';
import { createRng } from '../src/core/rng.js';
import { estimateStyle } from '../src/core/style.js';
import { buildMask } from '../src/core/layers.js';
import { fitDots } from '../src/core/fit.js';
import { synthGraph, SYNTH_PRESETS, scatterPoints } from '../src/core/synth.js';
import { matchDots } from '../src/core/metrics.js';
import { readPng, readImage } from '../tools/png.mjs';
import { createDotModel, solidImage, toRgba } from '../src/core/render.js';
import { estimateFamilies } from '../src/core/families.js';
import { readTruthCsv, truthPixels } from '../tools/truth-csv.mjs';
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
    // 98 at first; 100–101 with splits (one true dot is fitted as two, 4.7 px off)
    const { found, m } = benchRow('bench1', 'solid-opaque-stroke.png');
    assert.ok(found >= 98 && found <= 101, `found ${found}`);
    assert.equal(m.recall, 1);
});

test('regression: bench2 solid-opaque-stroke', { skip: benchAbsent('bench2', 'solid-opaque-stroke.png') }, () => {
    // before: no family. 165 of 204: about 8 dots are under 0.1% visible, and dots in packed clusters are missed
    // because the boundaries between fill and outline inside a cluster aren't used
    const { found, m } = benchRow('bench2', 'solid-opaque-stroke.png');
    assert.ok(found >= 175, `found ${found}`); // 165 before splits
    assert.ok(m.recall >= 0.84, `recall ${m.recall}`);
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

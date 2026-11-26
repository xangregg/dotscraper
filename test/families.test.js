// families.js on the truth sets: shaded regions over and under the dots, and one family over two backgrounds.
// Each set is skipped when absent.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { existsSync } from 'node:fs';
import { readImage } from '../tools/png.mjs';
import { readTruthCsv, truthPixels, truthPixelsByTicks, laneCentersFromDots } from '../tools/truth-csv.mjs';
import { extractAll } from '../src/core/extract.js';
import { groupRow, laneRow, jitterRow } from '../tools/bench.mjs';
import { detectAxes } from '../src/core/axes.js';
import { findColormap } from '../src/core/colormap.js';
import { backgroundColor, flatMask, EXACT_TOLERANCES } from '../src/core/scan.js';
import { estimateFamilies, pickedEstimate } from '../src/core/families.js';
import { fitInput } from '../src/core/extract.js';
import { buildMask } from '../src/core/layers.js';
import { fitDots } from '../src/core/fit.js';
import { createDotModel, solidImage, toRgba } from '../src/core/render.js';
import { matchDots } from '../src/core/metrics.js';
import { readTruthJson } from '../tools/truth-json.mjs';

const dir = set => new URL(`../${set}/`, import.meta.url).pathname;
const absent = set => !existsSync(dir(set) + 'data.csv') && `${set} absent`;

function row(set, name) {
    const px = truthPixels(readTruthCsv(dir(set) + 'data.csv'));
    const results = extractAll(readImage(dir(set) + name));
    return { results, r: px.laneCenters ? laneRow(name, px.dots, px.laneCenters, results, 0) : groupRow(name, px.dots, results, 0) };
}

test('regression: a CI band drawn over the dots is divided out (bench1 smoother-ci)', { skip: absent('bench1') }, () => {
    // before: 80 found, the dots under the band missed; then 197, when the band's edge pixels made a second family
    const { layers } = estimateFamilies(readImage(dir('bench1') + 'smoother-ci.png'));
    assert.deepEqual(layers.regions.map(r => r.mode), ['over']);
    assert.ok(Math.abs(layers.regions[0].s - 0.74) <= 0.02, `s ${layers.regions[0].s}`);
    const { results, r } = row('bench1', 'smoother-ci.png');
    assert.equal(results.length, 1);
    assert.equal(r.found, 100);
    assert.equal(r.recall, 1);
});

test('regression: a CI band drawn under the dots is read as their background there (bench1 smoother-behind-ci)', { skip: absent('bench1') }, () => {
    // before: 83 found
    const { r } = row('bench1', 'smoother-behind-ci.png');
    assert.equal(r.found, 100);
    assert.equal(r.recall, 1);
});

test('regression: one translucent family over two backgrounds is one family (bench1 ellipse)', { skip: absent('bench1') }, () => {
    // before: 194 found, orange on white and orange on lavender as two opaque families each taking every dot
    const { results, r } = row('bench1', 'ellipse.png');
    assert.equal(results.length, 1);
    const s = results[0].est.style;
    assert.ok(Math.abs(s.alpha - 0.5) <= 0.02, `α ${s.alpha}`);
    assert.equal(r.found, 100);
    assert.equal(r.recall, 1);
});

test('regression: opaque dots under a small translucent band (bench3 ANOVA)', { skip: absent('bench3') }, () => {
    // before: 185 found, the 19 under the band missed
    const { r } = row('bench3', 'dots-swarm-anova.png');
    assert.equal(r.found, 204);
    assert.equal(r.recall, 1);
});

test('regression: outlined dots (bench1 stroked-dots): a ring is measured and every dot found', { skip: absent('bench1') }, () => {
    // before: "The dot radius is unknown": the ring reads deeper than one dot, so no dot passed as isolated
    const { results, r } = row('bench1', 'stroked-dots.png');
    const s = results[0].est.style;
    assert.ok(s.ring, 'ring measured');
    // measured: a 2 px ring, 8.3 to 10.3 px, 69% opaque against the fill's 31%
    assert.ok(Math.abs(s.ring.inner * s.radius - 8.3) < 0.3, `ring inner ${s.ring.inner * s.radius}`);
    assert.equal(r.found, 100);
    assert.equal(r.recall, 1);
    // JMP draws stroked dots up-left of the data (fill 0.5 px, ring 0.13 px; plain dots are exact), so the positions
    // carry that offset; around it they scatter as little as plain dots do
    const found = results[0].fit.dots;
    const truth = truthPixels(readTruthCsv(dir('bench1') + 'data.csv')).dots;
    const off = truth.map(t => {
        const f = found.reduce((a, b) => (Math.hypot(b.x - t.x, b.y - t.y) < Math.hypot(a.x - t.x, a.y - t.y) ? b : a));
        return [f.x - t.x, f.y - t.y];
    });
    const mean = off.reduce((a, o) => [a[0] + o[0] / off.length, a[1] + o[1] / off.length], [0, 0]);
    assert.ok(mean.every(v => v > -0.5 && v < -0.35), `offset ${mean}`);
    const scatter = Math.sqrt(off.reduce((sum, o) => sum + (o[0] - mean[0]) ** 2 + (o[1] - mean[1]) ** 2, 0) / off.length);
    assert.ok(scatter < 0.1, `scatter ${scatter}`);
});

// Nature figures: translucent dots over box plots' boxes in their own hue (local only, skipped when absent)
const NATURE = dir('bench-nature');
const protrusion = k => `protrusion${k}-10.1038-s41586-026-11063-9`;
const natureAbsent = name => !existsSync(NATURE + name + '.png') && `bench-nature ${name} absent`;

// scored as the bench scores data-unit truth: through the detected ticks, jittered x matched in a box
function natureRow(name) {
    const image = readImage(NATURE + name + '.png');
    const px = truthPixelsByTicks(readTruthCsv(NATURE + name + '.csv'), detectAxes(image));
    const results = extractAll(image);
    if (px.lanesFromDots)
        px.laneCenters = laneCentersFromDots(results.flatMap(r => r.fit.dots.map(d => d.x)), new Set(px.dots.map(d => d.lane)).size);
    return px.laneCenters ? laneRow(name, px.dots, px.laneCenters, results, 0)
        : px.jitter ? jitterRow(name, px.dots, results, 0) : groupRow(name, px.dots, results, 0);
}

test('regression: protrusion3, the box read as the dots\' level 2 and the dots over it as 1.56 dots', { skip: natureAbsent(protrusion(3)) }, () => {
    // before: 114 found for 50, 89 of them in the box; dots over the box are read against levels measured there
    const r = natureRow(protrusion(3));
    assert.ok(r.precision >= 0.95 && r.recall >= 0.95, `P ${r.precision} R ${r.recall} (${r.found} found)`);
});

test('regression: protrusion2, both boxes read as levels, and orange dots with no whole one', { skip: natureAbsent(protrusion(2)) }, () => {
    // before: radius unknown; the orange family takes the green's radius (its dots all touch a whisker or the box)
    const { estimates, layers } = estimateFamilies(readImage(NATURE + protrusion(2) + '.png'));
    assert.equal(estimates.length, 2);
    assert.ok(estimates.every(e => Math.abs(e.style.radius - 30.1) < 0.5), `radii ${estimates.map(e => e.style.radius)}`);
    assert.deepEqual(layers.regions.map(r => r.mode), ['under', 'under']);
    const r = natureRow(protrusion(2));
    assert.ok(r.precision >= 0.95 && r.recall >= 0.85, `P ${r.precision} R ${r.recall} (${r.found} found)`);
});

test('regression: protrusion1, four hues on whiskers are no color scale, and each has its box', { skip: natureAbsent(protrusion(1)) }, () => {
    // before: a continuous scale (green and lavender chained through their rims' mixes with the gray whiskers) with
    // no radius; then each family's dots read as a band over its box (overBand), and every dot, halved by a whisker,
    // gave half the radius
    const image = readImage(NATURE + protrusion(1) + '.png');
    const B = backgroundColor(image).color;
    assert.ok(!findColormap(image, B, flatMask(image), EXACT_TOLERANCES), 'a color scale found');
    const { estimates } = estimateFamilies(image);
    assert.equal(estimates.length, 4);
    for (const e of estimates) {
        assert.ok(e.familyMeasure.box, `${e.style.color.map(Math.round)} has its box`);
        assert.ok(e.style.radius > 17.5 && e.style.radius < 21, `radius ${e.style.radius}`);
    }
});

test('regression: outlined dots in ties (bench2 stroked-dots): none goes for a ring that doesn\'t show', { skip: absent('bench2') }, () => {
    // the check of each dot's ring (fit.js ringless) took one of three dots within 4 px of each other, two of them
    // 1 px apart: its ring read 0.18 of what the model draws. A dot almost at one whose ring shows stays
    const { results, r } = row('bench2', 'stroked-dots.png');
    assert.ok(results[0].est.style.ring, 'ring measured');
    assert.equal(r.found, 204);
    assert.equal(r.recall, 1);
});

const RICHNESS = new URL('../bench-plos/richness-10.1371-journal.pone.0359333.png', import.meta.url).pathname;

test('regression: richness, a confidence band in the color of one dot is not filled with dots', { skip: !existsSync(RICHNESS) && 'bench-plos absent' }, () => {
    // before: 170 found for 48, the band and the regression line through it (the dots' full color) covered with
    // dots. The dots have rings and the band has none: 113 went for theirs, and 58 were left. Of the 10 too many, 4
    // were a pile on the regression line, which wasn't masked (its color is 25 units past the full color the levels
    // give), and 3 lay under real dots on the band. The last 3 are second dots where the image is two layers deep
    // off the band, which the truth of 48 placed by hand didn't have; it has 51 now
    const image = readImage(RICHNESS);
    const results = extractAll(image);
    assert.equal(results.length, 1);
    const truth = readTruthJson(RICHNESS.replace('bench-plos/', 'bench-plos/truth/').replace('.png', '.json')).dots;
    const m = matchDots(truth, results[0].fit.dots, 0.5 * results[0].fit.radius);
    assert.equal(m.pairs.length, truth.length, `${m.pairs.length} of ${truth.length} found`);
    assert.equal(results[0].fit.dots.length, 51);
    assert.equal(results[0].layers.mask[443 * image.width + 418], 1, 'the regression line is masked');
});

const GRADER1 = new URL('../bench-plos/grader1-10.1371-journal.pone.0361887.png', import.meta.url).pathname;

test('regression: grader1, nearly opaque dots with a ring (α 0.93 and 0.07): none goes for its ring', { skip: !existsSync(GRADER1) && 'bench-plos absent' }, () => {
    // the ring's share read in layers, as the fill's is (fit.js lacking), took 19 of 45 dots: a ring that leaves
    // nothing through is as many layers as one likes. It is read in what the dots let through
    const results = extractAll(readImage(GRADER1));
    const truth = readTruthJson(GRADER1.replace('bench-plos/', 'bench-plos/truth/').replace('.png', '.json')).dots;
    const found = results.flatMap(r => r.fit.dots);
    const m = matchDots(truth, found, 0.5 * results[0].fit.radius);
    assert.equal(m.pairs.length, truth.length, `${m.pairs.length} of ${truth.length} found`);
    assert.ok(!results[0].fit.log.some(l => l.stage === 'rings'), 'no dot removed for its ring');
});

const MPO = new URL('../bench-plos/mpo-10.1371-journal.pone.0359318.png', import.meta.url).pathname;

test('regression: mpo, black dots over gray boxes are an opaque family, and the boxes none', { skip: !existsSync(MPO) && 'bench-plos absent' }, () => {
    // before: "The dot radius is unknown". The page and the boxes' grays, 160 and 127, are the levels 0 to 2 of a gray
    // of 107 at α 0.64: a translucent "family" with no dot. Black lies on its line, 185 units beyond that gray, and was
    // never tried as a family of its own; found, it left the grays a family at its radius, as a sibling's
    const results = extractAll(readImage(MPO));
    assert.equal(results.length, 1, results.map(r => r.est.style.color.map(Math.round).join(',')).join('; '));
    assert.equal(results[0].est.familyMeasure.kind, 'opaque');
    assert.ok(results[0].est.style.color.every(v => v < 10), `color ${results[0].est.style.color}`);
    const truth = readTruthJson(MPO.replace('bench-plos/', 'bench-plos/truth/').replace('.png', '.json')).dots;
    const m = matchDots(truth, results[0].fit.dots, 0.5 * results[0].fit.radius);
    assert.equal(m.pairs.length, 45, `${m.pairs.length} of 45 found`);
    assert.ok(results[0].fit.dots.length <= 50, `${results[0].fit.dots.length} dots in all`);
});

const LUNG = new URL('../bench-plos/lung-tumor-10.1371-journal.pone.0357864.png', import.meta.url).pathname;

test('regression: lung-tumor, small blurred dots with a ring inside the rim, some under a band over the dots', { skip: !existsSync(LUNG) && 'bench-plos absent' }, () => {
    // before: "No dot family found." The fill's color is thin as a whole (more of it in the soft rims than in the
    // fills, 6 px across), so level 1 was left out; no dot was whole, each two layers deep from 0.45 of its radius;
    // the band over 22 of the dots was masked for want of votes (24 flat pixels), and at its waist, 10 px across with
    // the regression line through it, it had no seed and wasn't divided out
    const results = extractAll(readImage(LUNG));
    assert.equal(results.length, 1);
    const { est, fit } = results[0];
    assert.equal(est.familyMeasure.kind, 'translucent');
    // measured: α 0.44, radius 6.9, a ring from 0.51 to 0.76 of it
    assert.ok(Math.abs(est.style.alpha - 0.44) < 0.03 && Math.abs(est.style.radius - 6.9) < 0.3, `α ${est.style.alpha}, radius ${est.style.radius}`);
    assert.ok(est.style.ring && est.style.ring.inner < 0.6 && est.style.ring.outer < 0.85, JSON.stringify(est.style.ring));
    assert.ok(est.notes.some(t => t.startsWith('A translucent layer')), 'the band is divided out');
    // the truth has 501 dots (the figure's title says n = 502): the 473 first placed by hand, and from this fit the
    // 28 lying on others
    const truth = readTruthJson(LUNG.replace('bench-plos/', 'bench-plos/truth/').replace('.png', '.json')).dots;
    const m = matchDots(truth, fit.dots, 0.5 * fit.radius);
    assert.ok(m.pairs.length >= 499, `${m.pairs.length} of ${truth.length} found`);
    assert.ok(fit.dots.length <= 503, `${fit.dots.length} dots in all`);
});

test('a family picked by its color is read against a box under its dots, as the scan\'s families are', () => {
    // translucent dots, half of them over a filled box: read against the page there, they are no level of the family
    const W = 420;
    const H = 260;
    const fam = { color: [200, 60, 40], alpha: 0.5, radius: 10 };
    const bg = solidImage(W, H, [255, 255, 255]);
    for (let y = 30; y < 230; y++) {
        for (let x = 230; x < 400; x++)
            bg.data.set([69, 134, 172], 3 * (y * W + x));
    }
    const dots = [];
    for (let k = 0; k < 6; k++)
        dots.push({ x: 30 + 30 * k, y: 40 }, { x: 30 + 30 * k, y: 110 + 5 * k }, { x: 42 + 30 * k, y: 116 + 5 * k });
    for (let k = 0; k < 5; k++)
        dots.push({ x: 255 + 30 * k, y: 60 }, { x: 255 + 30 * k, y: 130 + 6 * k }, { x: 266 + 30 * k, y: 137 + 6 * k });
    const model = createDotModel(bg, fam);
    model.addAll(dots);
    const image = toRgba(model.image);
    const est = pickedEstimate(image, [228, 158, 148]);
    assert.equal(est.familyMeasure.kind, 'translucent');
    assert.ok(Math.abs(est.style.alpha - 0.5) <= 0.02, `α ${est.style.alpha}`);
    assert.ok(est.notes.some(n => /69, 134, 172 lies under the dots/.test(n)), est.notes.join(' '));
    const fit = fitDots(fitInput(est.image, est, buildMask(est.image, est).mask));
    const m = matchDots(dots, fit.dots, fam.radius / 2);
    assert.equal(fit.dots.length, dots.length);
    assert.equal(m.recall, 1);
    assert.ok(m.rmse < 0.2, `rmse ${m.rmse}`);
});

const BREAD1 = new URL('../bench-pmc/bread1-10.1128-jcm.01578-22.png', import.meta.url).pathname;

test('regression: bread1, grey15 dots on ggplot2\'s gray panel with black dots and gray text on the same line, over two boxes', { skip: !existsSync(BREAD1) && 'bench-pmc absent' }, () => {
    // before: the grays 114, 92 and 85 read as one family's levels, which had no dots; the dots over the red box were
    // an opaque maroon family, 8 found for 24, and the blue box a layer over the dots
    const image = readImage(BREAD1);
    const { layers, estimates } = estimateFamilies(image);
    assert.equal(estimates.length, 1);
    const { style, familyMeasure } = estimates[0];
    assert.equal(familyMeasure.kind, 'translucent');
    assert.ok(Math.abs(style.alpha - 0.6) <= 0.02, `α ${style.alpha}`);
    style.color.forEach(v => assert.ok(Math.abs(v - 39) <= 4, `color ${style.color}`));
    assert.deepEqual(layers.regions.map(r => r.mode), ['under', 'under', 'under']);
    // picked by hand, the same family (before: color 85 at α 0.79, and 509 dots along a box's outline)
    const picked = pickedEstimate(image, [92, 92, 92]);
    assert.ok(Math.abs(picked.style.alpha - 0.6) <= 0.02, `picked α ${picked.style.alpha}`);
    // every dot found; the extras are doubles on the black dots, darker than the family's, and where a dot crosses
    // a box's outline
    const truth = readTruthJson(BREAD1.replace('bench-pmc/', 'bench-pmc/truth/').replace('.png', '.json')).dots;
    const found = extractAll(image).flatMap(r => r.fit.dots);
    const m = matchDots(truth, found, 0.5 * style.radius);
    assert.equal(m.recall, 1);
    assert.ok(found.length <= 40, `${found.length} dots`);
});

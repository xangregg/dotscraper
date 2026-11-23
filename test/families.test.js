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
import { estimateFamilies } from '../src/core/families.js';
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
    assert.equal(findColormap(image, B, flatMask(image), EXACT_TOLERANCES), null);
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
    // dots. The dots have rings and the band has none: 113 go for theirs. Left over: 9, on or beside real dots
    const results = extractAll(readImage(RICHNESS));
    assert.equal(results.length, 1);
    const truth = readTruthJson(RICHNESS.replace('bench-plos/', 'bench-plos/truth/').replace('.png', '.json')).dots;
    const m = matchDots(truth, results[0].fit.dots, 0.5 * results[0].fit.radius);
    assert.equal(m.pairs.length, 48, `${m.pairs.length} of 48 found`);
    assert.ok(results[0].fit.dots.length <= 60, `${results[0].fit.dots.length} dots in all`);
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

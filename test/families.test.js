// families.js on the truth sets: shaded regions over and under the dots, and one family over two backgrounds.
// Each set is skipped when absent.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { existsSync } from 'node:fs';
import { readImage } from '../tools/png.mjs';
import { readTruthCsv, truthPixels } from '../tools/truth-csv.mjs';
import { extractAll, groupRow, laneRow } from '../tools/bench.mjs';
import { estimateFamilies } from '../src/core/families.js';

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

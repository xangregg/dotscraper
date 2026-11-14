import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createRng } from '../src/core/rng.js';
import { houghCenters, fitDots, refitNear, flagClose } from '../src/core/fit.js';
import { estimateStyle } from '../src/core/style.js';
import { buildMask } from '../src/core/layers.js';
import { synthGraph, scatterPoints } from '../src/core/synth.js';
import { matchDots } from '../src/core/metrics.js';

const GREEN = [50, 161, 73];

function prepare(spec) {
    const { image, truth } = synthGraph(spec);
    const est = estimateStyle(image);
    const layers = buildMask(image, est);
    const input = { image, style: est.style, line: est.familyMeasure.line, mask: layers.mask, D: est.D, imgT: est.fitT };
    return { image, truth, est, layers, input };
}

function spec(points, extra = {}) {
    return {
        width: 220, height: 180, background: [255, 255, 255], blur: 0.7,
        families: [{ color: GREEN, alpha: 0.3, radius: 12, points }], ...extra,
    };
}

test('Hough centers of separate dots are within 0.3 px, one per dot', () => {
    const points = [{ x: 40.3, y: 50.7 }, { x: 110.8, y: 60.2 }, { x: 170.1, y: 130.6 }, { x: 70.5, y: 140.4 }];
    const { est, layers, image } = prepare(spec(points));
    const weight = layers.mask.map(m => 1 - m);
    const found = houghCenters(est.D, weight, image.width, image.height, est.style.radius);
    assert.equal(found.length, 4);
    const m = matchDots(points, found, 3);
    assert.equal(m.recall, 1);
    assert.ok(m.rmse < 0.3, `rmse ${m.rmse}`);
    for (const f of found)
        assert.ok(f.arc > 0.8, `arc ${f.arc}`);
});

test('a cluster with a coincident pair and a near-coincident pair is counted exactly', () => {
    const points = [
        { x: 80, y: 80 }, { x: 80, y: 80 }, // coincident: only the doubled color shows there are two
        { x: 95.3, y: 88.1 }, { x: 96.8, y: 87.2 }, // 1.7 px apart
        { x: 70.2, y: 100.4 }, { x: 140.5, y: 70.3 },
    ];
    const { input, truth } = prepare(spec(points));
    const res = fitDots(input);
    const m = matchDots(truth, res.dots, 6);
    assert.equal(res.dots.length, 6);
    assert.equal(m.recall, 1);
    assert.ok(m.rmse < 1, `rmse ${m.rmse}`);
});

test('regression: a dot missing from a stack 6 deep is born (births count in local level steps)', () => {
    // six dots within 3 px: the sixth changes the color only (1 − α)^5 ≈ 0.17 as much as the first; normalizing
    // the birth residual by the first step once left such dots unfound
    const rng = createRng(21);
    const points = Array.from({ length: 6 }, () => ({ x: 110 + 3 * rng.uniform(), y: 90 + 3 * rng.uniform() }));
    points.push({ x: 40, y: 40 }, { x: 50, y: 45 }); // a pair, so the levels are measured
    points.push({ x: 180, y: 140 }); // an isolated dot, so the radius is
    const { input } = prepare(spec(points));
    const res = fitDots(input);
    const inStack = res.dots.filter(d => Math.hypot(d.x - 111.5, d.y - 91.5) < 6);
    assert.equal(inStack.length, 6);
});

test('the radius refit corrects a radius set 0.5 px too large', () => {
    const rng = createRng(22);
    const points = scatterPoints(rng, { n: 12, box: [20, 20, 200, 160] });
    const { input } = prepare(spec(points));
    const res = fitDots({ ...input, style: { ...input.style, radius: input.style.radius + 0.5 } });
    assert.ok(Math.abs(res.radius - 12) < 0.05, `radius ${res.radius}`);
});

test('a dot mostly under an opaque line is flagged hidden; open dots are not flagged', () => {
    // an isolated dot (radius) and an overlapping pair (levels) besides the hidden one
    const points = [{ x: 60, y: 90 }, { x: 150, y: 40 }, { x: 140, y: 140 }, { x: 155, y: 145 }];
    // a 10 px line (masked 15 px with its growth) leaves about a third of the dot at (60, 90) visible; a 16 px line
    // leaves 12 of its 448 px, too little evidence for any dot, so that one is rightly not found
    const { input } = prepare(spec(points, { lines: [{ points: [[0, 90], [220, 90]], width: 10, color: [0, 0, 0] }] }));
    const res = fitDots(input);
    const under = res.dots.find(d => Math.hypot(d.x - 60, d.y - 90) < 3);
    assert.ok(under, 'the hidden dot is found');
    assert.ok(under.flags.includes('hidden'));
    for (const d of res.dots.filter(d => d !== under))
        assert.deepEqual(d.flags, []);
    for (const d of res.dots)
        assert.ok(d.seX > 0 && d.seX < 0.5 && d.seY > 0 && d.seY < 0.5, `se ${d.seX}, ${d.seY}`);
});

// separate dots, enough for a family that never overlaps (an opaque one: at least 4 whole dots)
const SEPARATE = [{ x: 30, y: 30 }, { x: 190, y: 30 }, { x: 30, y: 150 }, { x: 190, y: 150 }, { x: 110, y: 30 }];

test('fitDots refuses to run without a radius', () => {
    const { input } = prepare(spec(SEPARATE));
    assert.throws(() => fitDots({ ...input, style: { ...input.style, radius: null } }), /radius/);
});

test('refitNear: an added dot near another settles where the image has one; a removed one stays removed', () => {
    const points = [{ x: 60, y: 90 }, { x: 140, y: 100 }, ...SEPARATE];
    const { input } = prepare(spec(points));
    // the user clicks 4 px from the true center of a dot the fit "missed"
    const res = refitNear(input, [...points.filter(p => p.x !== 140), { x: 143, y: 97, added: true }], 143, 97);
    const moved = res.dots.find(d => Math.hypot(d.x - 140, d.y - 100) < 1);
    assert.ok(moved, `refined to ${JSON.stringify(res.dots)}`);
    assert.equal(res.dots.length, points.length);
    // the dots come back sorted by x, not in the order given: `added` says which is the new one (Add mode nudges it)
    assert.deepEqual(res.added.map(i => res.dots[i]), [moved]);
    assert.notEqual(res.added[0], res.dots.length - 1);
});

test('regression: a missing dot whose neighbors drifted into its place is born (births re-settle neighbors)', () => {
    // a pentagon of dots 9 px from (110, 90), r 12; the fit starts with one missing and the other four pulled 3 px
    // toward the gap, as a fit does while a dot is missing. Judged alone, the new dot raised the loss.
    const pts = Array.from({ length: 5 }, (_, k) => ({ x: 110 + 9 * Math.cos(2 * Math.PI * k / 5), y: 90 + 9 * Math.sin(2 * Math.PI * k / 5) }));
    const extra = [{ x: 40, y: 40 }, { x: 52, y: 44 }, { x: 190, y: 150 }]; // a pair (levels) and an isolated dot (radius)
    const { input, truth } = prepare(spec([...pts, ...extra]));
    const gap = pts[0];
    const initial = [...pts.slice(1).map(p => {
        const d = Math.hypot(gap.x - p.x, gap.y - p.y);
        return { x: p.x + 3 * (gap.x - p.x) / d, y: p.y + 3 * (gap.y - p.y) / d };
    }), ...extra];
    const res = fitDots(input, { initial, refitRadius: false });
    const m = matchDots(truth, res.dots, 3);
    assert.equal(res.dots.length, truth.length);
    assert.equal(m.recall, 1);
});

test('flagClose marks both dots of a pair closer than 0.35 r, and clears the flag once they part', () => {
    const r = 10; // px
    const dots = [
        { x: 50, y: 50, flags: ['weak'] },
        { x: 50.2, y: 51, flags: [] }, // 1.02 px from the first
        { x: 53.6, y: 50, flags: [] }, // 3.6 px from the first: not close
        { x: 120, y: 50, flags: ['close'] }, // stale flag, alone
    ];
    flagClose(dots, r);
    assert.deepEqual(dots.map(d => d.flags), [['weak', 'close'], ['close'], [], []]);
    dots[1].y = 60;
    flagClose(dots, r);
    assert.deepEqual(dots.map(d => d.flags), [['weak'], [], [], []]);
});

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createRng } from '../src/core/rng.js';
import { houghCenters, fitDots, refitNear, flagClose, createFit, suppressNear } from '../src/core/fit.js';
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
    // the dots come back sorted by x, not in the order given: `from` says which is the new one (Add mode nudges it)
    const added = res.from.indexOf(points.length - 1);
    assert.equal(res.dots[added], moved);
    assert.notEqual(added, res.dots.length - 1);
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

test('births keep away from failed sites, and forget the ones an accepted birth makes stale', () => {
    // one dot of the image is missing from the model; the others are in place
    const missing = { x: 120, y: 80 };
    const others = [{ x: 40, y: 40 }, { x: 52, y: 44 }, { x: 190, y: 150 }]; // a pair (levels) and an isolated dot (radius)
    const { input } = prepare(spec([missing, ...others]));
    const r = input.style.radius;
    const start = () => {
        const fit = createFit(input);
        for (const p of others)
            fit.add(p.x, p.y);
        return fit;
    };
    assert.equal(start().births([]), 1);
    // failed sites 0.4 r apart all around the missing dot: every center a box could propose is within 0.5 r of one
    const sites = [];
    for (let j = -8; j <= 8; j++) {
        for (let i = -8; i <= 8; i++)
            sites.push({ x: missing.x + 0.4 * r * i, y: missing.y + 0.4 * r * j });
    }
    const all = sites.slice();
    assert.equal(start().births(all), 0);
    assert.equal(all.length, sites.length, 'no attempt was made');
    // one failed site beside the dot: the birth comes from a center farther off, and the site, now in a changed
    // neighborhood, is forgotten
    const one = [{ x: missing.x + 0.3 * r, y: missing.y }];
    const fit = start();
    assert.equal(fit.births(one), 1);
    assert.equal(one.length, 0);
    const born = fit.dots().find(d => Math.hypot(d.x - missing.x, d.y - missing.y) < 1);
    assert.ok(born, 'the dot is where the image has it');
});

test('refitNear leaves pinned dots where the user put them and refines the others; from maps each dot back', () => {
    const points = [{ x: 85, y: 90 }, { x: 125, y: 100 }, ...SEPARATE];
    const { input } = prepare(spec(points));
    // both dots 3 px off and within reach of the edit: the pinned one was put there by hand, the other is the fit's
    const given = [{ x: 88, y: 90, pinned: true }, { x: 128, y: 100 }, ...SEPARATE];
    const res = refitNear(input, given, 105, 95);
    const at = j => res.dots[res.from.indexOf(j)];
    assert.deepEqual([at(0).x, at(0).y], [88, 90]);
    assert.ok(Math.hypot(at(1).x - 125, at(1).y - 100) < 1, `unpinned dot at ${at(1).x}, ${at(1).y}`);
    assert.deepEqual([...res.from].sort((a, b) => a - b), [...given.keys()]);
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

test('suppressNear keeps what testing every kept peak keeps: random peaks, and a lattice exactly the distance apart', () => {
    const every = (peaks, nms) => {
        const kept = [];
        for (const pk of peaks) {
            if (kept.every(k => Math.hypot(k.x - pk.x, k.y - pk.y) >= nms))
                kept.push(pk);
        }
        return kept;
    };
    const rng = createRng(5);
    for (const [width, height, n, nms] of [[200, 150, 3000, 2], [200, 150, 3000, 7.3], [64, 900, 2000, 3.1], [50, 40, 500, 60], [300, 300, 1, 2]]) {
        const peaks = Array.from({ length: n }, () => ({ x: width * rng.uniform(), y: height * rng.uniform(), v: rng.uniform() }));
        const kept = suppressNear(peaks, nms, width);
        assert.deepEqual(kept, every(peaks, nms), `${n} peaks in ${width} × ${height}, ${nms} px`);
        assert.ok(kept.length >= 1 && kept.length <= n);
    }
    // neighbors exactly nms apart are both kept (the test is "closer than"); a point between two is dropped
    const lattice = [];
    for (let j = 0; j < 20; j++) {
        for (let i = 0; i < 20; i++)
            lattice.push({ x: 0.5 + 3 * i, y: 0.5 + 3 * j });
    }
    const between = lattice.map(p => ({ x: p.x + 1.5, y: p.y }));
    const all = [...lattice, ...between];
    const kept = suppressNear(all, 3, 62);
    assert.deepEqual(kept, every(all, 3));
    assert.deepEqual(kept, lattice);
    // the order given decides: the same points, the others first, keeps the others
    assert.deepEqual(suppressNear([...between, ...lattice], 3, 62), between);
});

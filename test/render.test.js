import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createRng } from '../src/core/rng.js';
import {
    pixelCoverage, gaussianKernel, blurRect, solidImage, toRgba, fromRgba, createDotModel, createDotStore, unionRect, renderDots,
} from '../src/core/render.js';

function supersampled(cx, cy, r, px, py, n) {
    let inside = 0;
    for (let j = 0; j < n; j++) {
        for (let i = 0; i < n; i++) {
            const x = px + (i + 0.5) / n;
            const y = py + (j + 0.5) / n;
            if ((x - cx) ** 2 + (y - cy) ** 2 < r * r)
                inside++;
        }
    }
    return inside / (n * n);
}

function totalCoverage(cx, cy, r) {
    let sum = 0;
    for (let py = Math.floor(cy - r) - 1; py <= Math.ceil(cy + r) + 1; py++) {
        for (let px = Math.floor(cx - r) - 1; px <= Math.ceil(cx + r) + 1; px++)
            sum += pixelCoverage(cx, cy, r, px, py);
    }
    return sum;
}

test('pixel coverage sums to the disc area (tolerance 1e-9 px²)', () => {
    const rng = createRng(1);
    for (let i = 0; i < 200; i++) {
        const r = 0.1 + 30 * rng.uniform();
        const cx = 50 + rng.uniform();
        const cy = 50 + rng.uniform();
        assert.ok(Math.abs(totalCoverage(cx, cy, r) - Math.PI * r * r) < 1e-9, `r = ${r}`);
    }
});

test('pixel coverage agrees with 400×400 supersampling (tolerance 2e-4)', () => {
    const rng = createRng(2);
    for (let i = 0; i < 60; i++) {
        const r = 0.3 + 10 * rng.uniform();
        const cx = 20 + rng.uniform();
        const cy = 20 + rng.uniform();
        // a pixel on the edge: walk out from the center along a random direction
        const th = 2 * Math.PI * rng.uniform();
        const px = Math.floor(cx + r * Math.cos(th));
        const py = Math.floor(cy + r * Math.sin(th));
        const exact = pixelCoverage(cx, cy, r, px, py);
        assert.ok(Math.abs(exact - supersampled(cx, cy, r, px, py, 400)) < 2e-4, `r ${r} at ${px},${py}: ${exact}`);
    }
});

test('pixel coverage: easy cases and symmetry', () => {
    assert.equal(pixelCoverage(10.5, 10.5, 5, 10, 10), 1);
    assert.equal(pixelCoverage(10.5, 10.5, 5, 20, 10), 0);
    // a disc of radius ½ centered in a pixel covers π/4 of it
    assert.ok(Math.abs(pixelCoverage(3.5, 7.5, 0.5, 3, 7) - Math.PI / 4) < 1e-12);
    // a disc much larger than a pixel, its edge through the pixel's middle: about half
    assert.ok(Math.abs(pixelCoverage(-0.5, 1000.5, 1000, 999, 1000) - 0.5) < 1e-3);
    // mirror symmetry: with 2·cx and 2·cy integers, pixel px mirrors to 2cx − 1 − px (and likewise in y)
    const cx = 10.5;
    const cy = 20;
    const r = 4.7;
    for (let py = 14; py < 27; py++) {
        for (let px = 4; px < 17; px++) {
            const a = pixelCoverage(cx, cy, r, px, py);
            assert.ok(Math.abs(a - pixelCoverage(cx, cy, r, 2 * cx - 1 - px, py)) < 1e-12);
            assert.ok(Math.abs(a - pixelCoverage(cx, cy, r, px, 2 * cy - 1 - py)) < 1e-12);
        }
    }
});

test('compositing: uncovered pixels keep the background; interiors sit at the overlap levels', () => {
    const B = [255, 255, 255];
    const c = [50, 160, 70];
    const alpha = 0.31;
    const bg = solidImage(60, 40, B);
    const im = renderDots(bg, [{ x: 20, y: 20 }, { x: 30, y: 20 }], { color: c, alpha, radius: 8 });
    const at = (x, y) => [0, 1, 2].map(ch => im.data[3 * (y * 60 + x) + ch]);
    assert.deepEqual(at(2, 2), B);
    // (14, 20) is inside the first dot only; (25, 20) inside both
    for (const [x, n] of [[14, 1], [25, 2]]) {
        const v = at(x, 20);
        for (let ch = 0; ch < 3; ch++)
            assert.ok(Math.abs(v[ch] - (c[ch] + (B[ch] - c[ch]) * (1 - alpha) ** n)) < 1e-3);
    }
});

test('Gaussian kernel is normalized and blur keeps a constant image constant', () => {
    for (const s of [0.4, 1.16, 3]) {
        const k = gaussianKernel(s);
        assert.ok(Math.abs(k.reduce((a, b) => a + b, 0) - 1) < 1e-12);
    }
    const src = solidImage(20, 15, [10, 200, 30]);
    const dst = solidImage(20, 15, [0, 0, 0]);
    blurRect(src, dst, gaussianKernel(1.5), [0, 0, 20, 15]);
    for (let i = 0; i < dst.data.length; i++)
        assert.ok(Math.abs(dst.data[i] - src.data[i % 3 === 0 ? 0 : i % 3]) < 1e-3);
});

test('blurred edge of a large dot matches the erf profile measured by the tools (σ within 0.05 px)', () => {
    // a straight-ish edge: radius 200, sample along the horizontal through the center
    const sigma = 1.16;
    const bg = solidImage(460, 9, [255, 255, 255]);
    const im = renderDots(bg, [{ x: 220, y: 4.5 }], { color: [0, 0, 0], alpha: 1, radius: 200, blur: sigma });
    // 1 − value/255 is the coverage; its 16% and 84% points are σ either side of the edge
    const cov = x => 1 - im.data[3 * (4 * 460 + x)] / 255;
    let x16 = 0;
    let x84 = 0;
    for (let x = 410; x < 430; x++) {
        // interpolate the crossing points on the falling edge near x = 420 (pixel centers at x + 0.5)
        const a = cov(x);
        const b = cov(x + 1);
        if (a >= 0.84 && b < 0.84)
            x84 = x + 0.5 + (a - 0.84) / (a - b);
        if (a >= 0.16 && b < 0.16)
            x16 = x + 0.5 + (a - 0.16) / (a - b);
    }
    // the box filter of exact coverage adds variance 1/12 to σ²
    const expected = Math.sqrt(sigma * sigma + 1 / 12);
    assert.ok(Math.abs((x16 - x84) / 2 - expected) < 0.05, `${(x16 - x84) / 2} vs ${expected}`);
});

test('incremental model equals a full rerender after random adds, moves and removes', () => {
    for (const blur of [0, 1.2]) {
        const rng = createRng(3 + blur);
        const bg = solidImage(120, 90, [250, 252, 248]);
        const style = { color: [50, 160, 70], alpha: 0.3, radius: 9.5, blur };
        const model = createDotModel(bg, style);
        const ids = [];
        for (let step = 0; step < 150; step++) {
            const u = rng.uniform();
            if (ids.length < 3 || u < 0.4)
                ids.push(model.add(-5 + 130 * rng.uniform(), -5 + 100 * rng.uniform()));
            else if (u < 0.8) {
                const id = ids[Math.floor(rng.uniform() * ids.length)];
                model.move(id, -5 + 130 * rng.uniform(), -5 + 100 * rng.uniform());
            }
            else {
                const k = Math.floor(rng.uniform() * ids.length);
                model.remove(ids[k]);
                ids.splice(k, 1);
            }
        }
        const full = renderDots(bg, model.dots(), style);
        let maxDiff = 0;
        for (let i = 0; i < full.data.length; i++)
            maxDiff = Math.max(maxDiff, Math.abs(full.data[i] - model.image.data[i]));
        assert.ok(maxDiff < 1e-3, `blur ${blur}: max diff ${maxDiff}`);
    }
});

test('influenceRect bounds every pixel a dot changes', () => {
    const bg = solidImage(80, 80, [255, 255, 255]);
    const model = createDotModel(bg, { color: [0, 0, 0], alpha: 0.5, radius: 6.3, blur: 1 });
    const before = Float32Array.from(model.image.data);
    model.add(40.2, 39.7);
    const [x0, y0, x1, y1] = model.influenceRect(40.2, 39.7);
    for (let y = 0; y < 80; y++) {
        for (let x = 0; x < 80; x++) {
            const changed = model.image.data[3 * (y * 80 + x)] !== before[3 * (y * 80 + x)];
            if (changed)
                assert.ok(x >= x0 && x < x1 && y >= y0 && y < y1, `${x},${y} changed outside`);
        }
    }
});

test('toRgba and fromRgba round-trip integer values', () => {
    const im = solidImage(3, 2, [1, 2, 3]);
    im.data[4] = 254.6;
    const rgba = toRgba(im);
    assert.equal(rgba.data[4 * 1 + 1], 255);
    assert.equal(rgba.data[3], 255);
    const back = fromRgba(rgba);
    assert.equal(back.data[4], 255);
    assert.equal(back.data[0], 1);
});

test('level table: a geometric table reproduces the formula; a measured table sets the plateaus', () => {
    const B = [255, 255, 255];
    const c = [50, 161, 73];
    const alpha = 0.31;
    const geometric = Array.from({ length: 12 }, (_, n) => c.map((v, ch) => v + (B[ch] - v) * (1 - alpha) ** n));
    const points = [{ x: 20.3, y: 20.6 }, { x: 31.1, y: 22.4 }, { x: 25.7, y: 30.2 }, { x: 25.7, y: 30.2 }];
    const bg = solidImage(60, 55, B);
    const plain = renderDots(bg, points, { color: c, alpha, radius: 9.2, blur: 0.7 });
    const tabled = renderDots(bg, points, { color: c, alpha, radius: 9.2, blur: 0.7, levels: geometric });
    let maxDiff = 0;
    for (let i = 0; i < plain.data.length; i++)
        maxDiff = Math.max(maxDiff, Math.abs(plain.data[i] - tabled.data[i]));
    assert.ok(maxDiff < 1e-3, `max diff ${maxDiff}`);
    // perturb level 1 and 3: interiors at those depths take the table's colors exactly
    const measured = geometric.map(v => [...v]);
    measured[1] = [190, 226, 197];
    measured[3] = [118, 193, 133];
    const im = renderDots(bg, points, { color: c, alpha, radius: 9.2, levels: measured });
    const at = (x, y) => [0, 1, 2].map(ch => im.data[3 * (y * 60 + x) + ch]);
    // (13, 20) is inside the first dot only; (25, 24) inside all four (the coincident pair and the other two)
    assert.deepEqual(at(13, 20).map(v => +v.toFixed(3)), measured[1]);
    const four = at(25, 24);
    for (let ch = 0; ch < 3; ch++)
        assert.ok(Math.abs(four[ch] - measured[4][ch]) < 1e-3);
});

test('level table with opaque dots (α = 1) interpolates between B and c at edges', () => {
    const B = [255, 255, 255];
    const c = [0, 0, 0];
    const bg = solidImage(30, 30, B);
    const a = renderDots(bg, [{ x: 15, y: 15 }], { color: c, alpha: 1, radius: 6.4 });
    const b = renderDots(bg, [{ x: 15, y: 15 }], { color: c, alpha: 1, radius: 6.4, levels: [B, c, c] });
    for (let i = 0; i < a.data.length; i++)
        assert.ok(Math.abs(a.data[i] - b.data[i]) < 1e-3);
});

test('createDotStore: records by id, the dots near a rectangle, and snapshots of records and buffers', () => {
    const W = 100;
    const H = 80;
    const plain = new Float32Array(W * H).fill(1);
    const rgb = new Float32Array(3 * W * H).fill(2);
    const blurred = new Float32Array(W * H).fill(3);
    const store = createDotStore(W, H, 5, 2, [
        { data: plain, stride: 1, blurred: false },
        { data: rgb, stride: 3, blurred: false },
        { data: blurred, stride: 1, blurred: true },
    ]);
    const a = store.newId();
    const b = store.newId();
    const c = store.newId();
    assert.deepEqual([a, b, c], [0, 1, 2]);
    assert.equal(store.put(a, { x: 20, y: 20 }), undefined);
    store.put(b, { x: 24, y: 22 });
    store.put(c, { x: 80, y: 60 });
    // near: the two beside the rectangle, not the one across the image
    const rect = [14, 14, 30, 30];
    assert.deepEqual(store.near(rect).map(d => d.x).sort(), [20, 24]);
    assert.deepEqual(store.near([70, 50, 90, 70]).map(d => d.x), [80]);
    assert.deepEqual(unionRect([1, 2, 5, 6], [3, 0, 9, 4]), [1, 0, 9, 6]);
    assert.deepEqual(store.blurredRect([0, 10, 99, 80]), [0, 8, 100, 80]);
    // a trial: one dot moved, one dropped, one added, every buffer scribbled on; restore undoes all of it
    const snap = store.snapshot(rect);
    const moved = { x: 26, y: 18 };
    assert.deepEqual(store.put(a, moved), { x: 20, y: 20 });
    assert.deepEqual(store.drop(b), { x: 24, y: 22 });
    const d = store.newId();
    store.put(d, { x: 22, y: 24 });
    for (let y = 12; y < 32; y++) {
        for (let x = 12; x < 32; x++) {
            const p = y * W + x;
            const inside = x >= 14 && x < 30 && y >= 14 && y < 30;
            if (inside) {
                plain[p] = 9;
                rgb.fill(9, 3 * p, 3 * p + 3);
            }
            blurred[p] = 9; // the blurred plane is saved 2 px farther out
        }
    }
    store.restore(snap);
    assert.deepEqual(store.entries(), [[a, { x: 20, y: 20 }], [b, { x: 24, y: 22 }], [c, { x: 80, y: 60 }]]);
    assert.equal(store.get(d), undefined);
    assert.deepEqual(store.near(rect).map(e => e.x).sort(), [20, 24]);
    assert.ok(plain.every(v => v === 1) && rgb.every(v => v === 2) && blurred.every(v => v === 3), 'buffers restored');
    // an id is never reused
    assert.equal(store.newId(), 4);
});

test('snapshot and restore undo moves, adds and removes exactly', () => {
    const bg = solidImage(90, 70, [255, 255, 255]);
    const model = createDotModel(bg, { color: [50, 160, 70], alpha: 0.3, radius: 8.5, blur: 1 });
    const a = model.add(30, 30);
    const b = model.add(40, 35);
    const before = Float32Array.from(model.image.data);
    const dotsBefore = JSON.stringify(model.dots());
    const rect = (r1, r2) => [Math.min(r1[0], r2[0]), Math.min(r1[1], r2[1]), Math.max(r1[2], r2[2]), Math.max(r1[3], r2[3])];
    let snap = model.snapshot(rect(model.footprint(30, 30), model.footprint(33.4, 28.1)));
    model.move(a, 33.4, 28.1);
    model.restore(snap);
    snap = model.snapshot(model.footprint(40, 35));
    model.remove(b);
    model.restore(snap);
    snap = model.snapshot(model.footprint(55, 50));
    model.add(55, 50);
    model.restore(snap);
    assert.equal(JSON.stringify(model.dots()), dotsBefore);
    for (let i = 0; i < before.length; i++)
        assert.equal(model.image.data[i], before[i]);
    // and later edits still agree with a full render (the grid was restored too)
    model.move(b, 47, 41);
    const full = renderDots(bg, model.dots(), { color: [50, 160, 70], alpha: 0.3, radius: 8.5, blur: 1 });
    for (let i = 0; i < full.data.length; i++)
        assert.ok(Math.abs(full.data[i] - model.image.data[i]) < 1e-3);
});

// snapshot() saves only the dots near its rectangle, so restore() has to put back exactly those and leave the rest
test('snapshot and restore among many dots: rejected trials leave the dots, their order and the image as they were', () => {
    const W = 200;
    const H = 140;
    const COUNT = 60;
    const style = { color: [50, 160, 70], alpha: 0.3, radius: 6.5, blur: 1 };
    const bg = solidImage(W, H, [255, 255, 255]);
    const rng = createRng(11);
    const place = () => ({ x: 10 + (W - 20) * rng.uniform(), y: 10 + (H - 20) * rng.uniform() });
    const union = (a, b) => [Math.min(a[0], b[0]), Math.min(a[1], b[1]), Math.max(a[2], b[2]), Math.max(a[3], b[3])];
    const model = createDotModel(bg, style);
    model.addAll(Array.from({ length: COUNT }, place));
    for (let t = 0; t < 45; t++) {
        const image = Float32Array.from(model.image.data);
        const dots = JSON.stringify(model.dots());
        const d = model.dots()[Math.floor(COUNT * rng.uniform())];
        const to = { x: d.x + 8 * (rng.uniform() - 0.5), y: d.y + 8 * (rng.uniform() - 0.5) };
        const extra = { x: d.x + 10 * (rng.uniform() - 0.5), y: d.y + 10 * (rng.uniform() - 0.5) };
        const snap = model.snapshot(union(union(model.footprint(d.x, d.y), model.footprint(to.x, to.y)), model.footprint(extra.x, extra.y)));
        // a move, a removal (its id must come back in its place, not at the end), or a birth with a move
        if (t % 3 === 0)
            model.move(d.id, to.x, to.y);
        else if (t % 3 === 1)
            model.remove(d.id);
        else {
            model.add(extra.x, extra.y);
            model.move(d.id, to.x, to.y);
        }
        model.restore(snap);
        assert.equal(JSON.stringify(model.dots()), dots, `dots after trial ${t}`);
        assert.deepEqual(model.image.data, image, `image after trial ${t}`);
        // an accepted move between the trials: the next snapshot is of a changed model
        if (t % 5 === 4)
            model.move(d.id, to.x, to.y);
    }
    // the grid still knows where every dot is: moving each one agrees with a full render
    for (const d of model.dots())
        model.move(d.id, d.x + 1.3, d.y - 0.7);
    const full = renderDots(bg, model.dots(), style);
    for (let i = 0; i < full.data.length; i++)
        assert.ok(Math.abs(full.data[i] - model.image.data[i]) < 1e-3);
});

test('projected rendering equals the projection of the RGB rendering', () => {
    const B = [255, 255, 255];
    const c = [50, 161, 73];
    const d = B.map((v, i) => v - c[i]);
    const len = Math.hypot(...d);
    const line = { B, u: d.map(v => v / len) };
    const levels = Array.from({ length: 10 }, (_, n) => c.map((v, ch) => v + (B[ch] - v) * 0.69 ** n + (n === 1 ? 1.5 : 0)));
    const points = [{ x: 20.3, y: 20.6 }, { x: 31.1, y: 22.4 }, { x: 25.7, y: 30.2 }];
    const bg = solidImage(60, 55, B);
    for (const style of [{ color: c, alpha: 0.31, radius: 9.2, blur: 0.9 }, { color: c, alpha: 0.31, radius: 9.2, blur: 0.9, levels }]) {
        const rgb = renderDots(bg, points, style);
        const t = renderDots(bg, points, { ...style, project: line });
        for (let p = 0; p < 60 * 55; p++) {
            const proj = [0, 1, 2].reduce((s, ch) => s + (B[ch] - rgb.data[3 * p + ch]) * line.u[ch], 0);
            assert.ok(Math.abs(proj - t.data[p]) < 1e-3, `pixel ${p}: ${proj} vs ${t.data[p]}`);
        }
    }
});

test('an outlined dot: α inside, α + ring α in the annulus, exact coverage at the ring’s inner edge', () => {
    const style = { color: [0, 0, 0], alpha: 0.3, radius: 10, ring: { inner: 0.8, alpha: 0.4 } };
    const im = renderDots(solidImage(40, 40, [255, 255, 255]), [{ x: 20, y: 20 }], style);
    const at = (x, y) => im.data[3 * (y * 40 + x)];
    // center, and a pixel [28, 29) × [19, 20), 8.0–9.0 px out along the row: inside the annulus 8–10
    assert.ok(Math.abs(at(20, 20) - 255 * 0.7) < 1e-3);
    assert.ok(Math.abs(at(28, 19) - 255 * 0.3) < 0.2, `ring ${at(28, 19)}`);
    // a pixel straddling the inner circle: 1 − α·m − ring α·(m − m_inner)
    const m = pixelCoverage(20, 20, 10, 27, 23);
    const mi = pixelCoverage(20, 20, 8, 27, 23);
    assert.ok(mi > 0.05 && mi < 0.95, `straddles: ${mi}`);
    assert.ok(Math.abs(at(27, 23) - 255 * (1 - 0.3 * m - 0.4 * (m - mi))) < 1e-3);
    assert.equal(at(5, 5), 255);
});

test('an outline inside the rim: past outer × radius the dot is at its own α again', () => {
    const style = { color: [0, 0, 0], alpha: 0.3, radius: 10, ring: { inner: 0.5, outer: 0.8, alpha: 0.4 } };
    const im = renderDots(solidImage(40, 40, [255, 255, 255]), [{ x: 20, y: 20 }], style);
    const at = (x, y) => im.data[3 * (y * 40 + x)];
    // the center; a pixel 6.0–7.1 px out, inside the annulus 5–8; one 8.0–9.1 px out, past it and inside the rim
    assert.ok(Math.abs(at(20, 20) - 255 * 0.7) < 1e-3);
    assert.ok(Math.abs(at(26, 19) - 255 * 0.3) < 1e-3, `ring ${at(26, 19)}`);
    assert.ok(Math.abs(at(28, 19) - 255 * 0.7) < 1e-3, `past the ring ${at(28, 19)}`);
    // a pixel straddling the outer circle: 1 − α − ring α·m_outer
    const mo = pixelCoverage(20, 20, 8, 27, 23);
    assert.ok(mo > 0.05 && mo < 0.95, `straddles: ${mo}`);
    assert.ok(Math.abs(at(27, 23) - 255 * (1 - 0.3 - 0.4 * mo)) < 1e-3);
});

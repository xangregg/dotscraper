import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createRng } from '../src/core/rng.js';
import {
    pixelCoverage, gaussianKernel, blurRect, solidImage, toRgba, fromRgba, createDotModel, renderDots,
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

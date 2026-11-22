import { test } from 'node:test';
import assert from 'node:assert/strict';
import { components, isolatedDots, discShare } from '../src/core/shape.js';
import { pixelCoverage } from '../src/core/render.js';

const W = 80;
const H = 60;

// coverage of discs, exact per pixel, summed (so a doubled disc reads 2)
function discs(list) {
    const D = new Float32Array(W * H);
    for (const { x, y, r } of list) {
        for (let py = 0; py < H; py++) {
            for (let px = 0; px < W; px++)
                D[py * W + px] += pixelCoverage(x, y, r, px, py);
        }
    }
    return D;
}

test('components: 4-connected pieces above 0.5, diagonal neighbors apart', () => {
    const D = new Float32Array(W * H);
    for (const [x, y] of [[3, 3], [4, 3], [4, 4], [5, 5], [20, 10]])
        D[y * W + x] = 1;
    D[30 * W + 30] = 0.4;
    const comps = components(D, W, H).map(c => c.length).sort((a, b) => b - a);
    assert.deepEqual(comps, [3, 1, 1]);
});

test('isolatedDots: whole single dots with their centers and radius; touching and doubled dots are not isolated', () => {
    const lone = [{ x: 15.3, y: 14.6, r: 7 }, { x: 60.5, y: 40.2, r: 7 }];
    const D = discs([...lone, { x: 30, y: 44, r: 7 }, { x: 40, y: 44, r: 7 }, { x: 62, y: 14, r: 7 }, { x: 62, y: 14, r: 7 }]);
    const found = isolatedDots(D, W, H);
    assert.equal(found.length, 2);
    for (const d of lone) {
        const f = found.find(g => Math.hypot(g.cx - d.x, g.cy - d.y) < 0.1);
        assert.ok(f, `the dot at ${d.x}, ${d.y}`);
        assert.ok(Math.abs(f.r - d.r) < 0.1, `radius ${f.r}`);
    }
});

test('discShare', () => {
    const D = discs([{ x: 20, y: 20, r: 8 }]);
    assert.ok(discShare(D, W, H, 20, 20, 6) > 0.99);
    assert.ok(Math.abs(discShare(D, W, H, 28, 20, 6) - 0.5) < 0.1, `half on the edge: ${discShare(D, W, H, 28, 20, 6)}`);
    assert.equal(discShare(D, W, H, 60, 40, 6), 0);
});

// a disc's coverage with an edge `ramp` px wide (linear across it), as resampling leaves it
function softDisc(D, x, y, r, ramp) {
    for (let py = 0; py < H; py++) {
        for (let px = 0; px < W; px++)
            D[py * W + px] += Math.min(1, Math.max(0, 0.5 + (r - Math.hypot(px + 0.5 - x, py + 0.5 - y)) / ramp));
    }
}

test('regression: a blurred dot is an isolated dot where blurred ones are asked for; one in a faint halo is not', () => {
    // before: a soft rim spreads a dot's second moments and not its area (here r(moment) / r(area) is 1.07 against
    // ISOLATED_MAX_MOMENT_RATIO 1.03), so blurred dots were never isolated (biomarker: none of its dots, and the
    // radius was read from the distance transform, 4.5 for 6.0)
    const D = new Float32Array(W * H);
    softDisc(D, 20.3, 20.6, 6, 5);
    softDisc(D, 58.5, 38.2, 6, 5);
    // a halo at 0.3 around the second dot, out to 10 px: round too, and no part of the dot
    for (let py = 0; py < H; py++) {
        for (let px = 0; px < W; px++) {
            if (Math.hypot(px + 0.5 - 58.5, py + 0.5 - 38.2) <= 10)
                D[py * W + px] = Math.max(D[py * W + px], 0.3);
        }
    }
    assert.equal(isolatedDots(D, W, H).length, 0, 'none as crisp dots');
    const found = isolatedDots(D, W, H, undefined, true);
    assert.equal(found.length, 1, `${found.length} isolated`);
    assert.ok(Math.hypot(found[0].cx - 20.3, found[0].cy - 20.6) < 0.1, `center ${found[0].cx}, ${found[0].cy}`);
    // (the radius of the soft area: a linear edge 5 px wide adds 5² / 12 to r²)
    assert.ok(Math.abs(found[0].r - Math.hypot(6, 5 / Math.sqrt(12))) < 0.05, `radius ${found[0].r}`);
});

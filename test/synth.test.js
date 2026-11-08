import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createRng } from '../src/core/rng.js';
import { pixelHash } from '../src/core/image.js';
import { scatterPoints, synthGraph, SYNTH_PRESETS } from '../src/core/synth.js';

// coverage-weighted centroid of the darkening (B − pixel, summed over channels) in a window
function inkCentroid(image, x0, y0, x1, y1) {
    let s = 0;
    let sx = 0;
    let sy = 0;
    for (let y = y0; y < y1; y++) {
        for (let x = x0; x < x1; x++) {
            const p = 4 * (y * image.width + x);
            const ink = 3 * 255 - image.data[p] - image.data[p + 1] - image.data[p + 2];
            s += ink;
            sx += ink * (x + 0.5);
            sy += ink * (y + 0.5);
        }
    }
    return { x: sx / s, y: sy / s, ink: s };
}

test('scatterPoints: deterministic, inside the box, ties duplicate points', () => {
    const box = [10, 20, 110, 220];
    for (const layout of ['uniform', 'curve', 'clusters']) {
        const a = scatterPoints(createRng(7), { n: 200, box, layout, ties: 0.1 });
        const b = scatterPoints(createRng(7), { n: 200, box, layout, ties: 0.1 });
        assert.deepEqual(a, b);
        assert.equal(a.length, 200);
        for (const p of a)
            assert.ok(p.x >= 10 && p.x <= 110 && p.y >= 20 && p.y <= 220);
    }
    const tied = scatterPoints(createRng(8), { n: 100, box, ties: 0.2 });
    const keys = new Set(tied.map(p => `${p.x},${p.y}`));
    assert.ok(keys.size < 100 && keys.size >= 80, `${keys.size} distinct`);
});

test('synthGraph is deterministic and returns the truth', () => {
    for (const name of Object.keys(SYNTH_PRESETS)) {
        const a = synthGraph(SYNTH_PRESETS[name](3));
        const b = synthGraph(SYNTH_PRESETS[name](3));
        assert.equal(pixelHash(a.image), pixelHash(b.image), name);
        assert.equal(a.truth.length, SYNTH_PRESETS[name](3).families.reduce((s, f) => s + f.points.length, 0));
    }
});

test('an isolated dot is centered on its truth position, at native and upsampled resolution (0.02 px)', () => {
    for (const scale of [1, 2, 1.7]) {
        const spec = {
            width: 200, height: 160, background: [255, 255, 255], scale, blur: scale > 1 ? 0.6 : 0,
            families: [{ color: [50, 160, 70], alpha: 0.3, radius: 14, points: [{ x: 83.37, y: 71.81 }] }],
        };
        const { image } = synthGraph(spec);
        const c = inkCentroid(image, 50, 40, 120, 110);
        assert.ok(Math.hypot(c.x - 83.37, c.y - 71.81) < 0.02, `scale ${scale}: ${c.x}, ${c.y}`);
    }
});

test('upsampling keeps the dot area: the ink of a dot scales with its final-pixel area (0.5%)', () => {
    const inkAt = scale => {
        const { image } = synthGraph({
            width: 200, height: 200, background: [255, 255, 255], scale,
            families: [{ color: [0, 0, 0], alpha: 1, radius: 20, points: [{ x: 100, y: 100 }] }],
        });
        return inkCentroid(image, 60, 60, 140, 140).ink;
    };
    const native = inkAt(1);
    assert.ok(Math.abs(native / (3 * 255 * Math.PI * 400) - 1) < 0.005);
    for (const scale of [2, 2.5])
        assert.ok(Math.abs(inkAt(scale) / native - 1) < 0.005, `scale ${scale}`);
});

test('lines are drawn over the dots, with gaps where dashed', () => {
    const { image } = synthGraph({
        width: 120, height: 40, background: [255, 255, 255],
        families: [{ color: [50, 160, 70], alpha: 0.3, radius: 15, points: [{ x: 30, y: 20 }] }],
        lines: [{ points: [[0, 20], [120, 20]], width: 4, color: [0, 0, 0], dash: [20, 10] }],
    });
    const at = x => [...image.data.slice(4 * (20 * 120 + x), 4 * (20 * 120 + x) + 3)];
    assert.deepEqual(at(30), [0, 0, 0]); // on a dash, over the dot
    assert.deepEqual(at(25), [194, 227, 200]); // in a gap (arc 20–30), the dot shows: 255 + (c − 255)·0.3, rounded
    assert.deepEqual(at(85), [255, 255, 255]); // in a gap (arc 80–90), the background
});

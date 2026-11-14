import { test } from 'node:test';
import assert from 'node:assert/strict';
import { existsSync } from 'node:fs';
import { readdirSync } from 'node:fs';
import jpegJs from 'jpeg-js';
import { restoreChroma, jpegNoise } from '../src/core/jpeg.js';
import { synthGraph, SYNTH_PRESETS } from '../src/core/synth.js';
import { readImage } from '../tools/png.mjs';
import { readTruthCsv, truthPixels } from '../tools/truth-csv.mjs';
import { extractAll, groupRow } from '../tools/bench.mjs';

const BENCH = new URL('../bench1/', import.meta.url).pathname;
const MEDIUM_R = 6.72; // px, bench1 medium-dots
const EDGE_BAND = 1.5; // px either side of the rim

// rms RGB difference between two images over the band [lo, hi] px from the centers of dots no other dot comes near
function bandError(a, b, dots, r, lo, hi) {
    const alone = dots.filter(d => !dots.some(e => e !== d && Math.hypot(e.x - d.x, e.y - d.y) < 2 * r + 4));
    let sum = 0;
    let count = 0;
    for (const d of alone) {
        for (let y = Math.floor(d.y - hi - 1); y <= d.y + hi + 1; y++) {
            for (let x = Math.floor(d.x - hi - 1); x <= d.x + hi + 1; x++) {
                const q = Math.hypot(x + 0.5 - d.x, y + 0.5 - d.y);
                if (q < lo || q > hi)
                    continue;
                const p = 4 * (y * a.width + x);
                for (let ch = 0; ch < 3; ch++)
                    sum += (a.data[p + ch] - b.data[p + ch]) ** 2;
                count += 3;
            }
        }
    }
    return Math.sqrt(sum / count);
}

test('restoreChroma on a real JPEG: dot rims and interiors move toward the PNG of the same chart', { skip: !existsSync(BENCH + 'medium-dots.jpg') && 'bench1 absent' }, () => {
    // measured: rims 12.0 → 6.2, interiors 5.5 → 4.1 (rms, 0–255 units)
    const png = readImage(BENCH + 'medium-dots.png');
    const raw = readImage(BENCH + 'medium-dots.jpg');
    const restored = restoreChroma(raw);
    const dots = truthPixels(readTruthCsv(BENCH + 'data.csv')).dots;
    const rim = [MEDIUM_R - EDGE_BAND, MEDIUM_R + EDGE_BAND];
    const inside = [0, MEDIUM_R - EDGE_BAND];
    const rimBefore = bandError(raw, png, dots, MEDIUM_R, ...rim);
    const rimAfter = bandError(restored, png, dots, MEDIUM_R, ...rim);
    const inBefore = bandError(raw, png, dots, MEDIUM_R, ...inside);
    const inAfter = bandError(restored, png, dots, MEDIUM_R, ...inside);
    assert.ok(rimAfter < 0.6 * rimBefore, `rims ${rimBefore.toFixed(2)} → ${rimAfter.toFixed(2)}`);
    assert.ok(inAfter < 0.85 * inBefore, `interiors ${inBefore.toFixed(2)} → ${inAfter.toFixed(2)}`);
});

test('restoreChroma leaves a flat image and a two-color edge as they are (±1 for rounding)', () => {
    const width = 40;
    const height = 20;
    const data = new Uint8ClampedArray(4 * width * height);
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const c = x < 20 ? [255, 255, 255] : [242, 185, 130];
            data.set([...c, 255], 4 * (y * width + x));
        }
    }
    const image = { width, height, data };
    const out = restoreChroma(image);
    for (let i = 0; i < data.length; i++)
        assert.ok(Math.abs(out.data[i] - data[i]) <= 1, `value ${i}: ${out.data[i]} vs ${data[i]}`);
});

const BENCH2 = new URL('../bench2/', import.meta.url).pathname;

// The red family's two-dot lenses are too narrow for many flat pixels to survive JPEG noise: its level 2 was spread
// over 7 histogram bins with none reaching a peak, the next level was read as level 2, the fit failed, and the family
// was taken for opaque pink (131 of 204 found, recall 0.637). Smoothing the level histogram finds level 2; level 3
// is still missing and is inferred.
test('regression: bench2 group-smoother.jpg reads its red family as translucent at α 0.5', { skip: !existsSync(BENCH2 + 'group-smoother.jpg') && 'bench2 absent' }, () => {
    const results = extractAll(readImage(BENCH2 + 'group-smoother.jpg'), true);
    assert.equal(results.length, 2);
    for (const res of results) {
        assert.ok(Math.abs(res.est.style.alpha - 0.5) <= 0.02, `α ${res.est.style.alpha} for ${res.est.style.color.map(Math.round)}`);
    }
    const r = groupRow('group-smoother.jpg', truthPixels(readTruthCsv(BENCH2 + 'data.csv')).dots, results, 0);
    assert.ok(r.precision >= 0.98, `precision ${r.precision}`);
    assert.ok(r.recall >= 0.95, `recall ${r.recall}`);
});

const JPEG_QUALITY = 85; // a typical web figure's

test('jpegNoise: a clean render has none; the same graph through JPEG has it', () => {
    const { image } = synthGraph(SYNTH_PRESETS.sparse(1));
    const clean = jpegNoise(image);
    assert.equal(clean.fraction, 0);
    assert.equal(clean.likely, false);
    const enc = jpegJs.encode({ width: image.width, height: image.height, data: Buffer.from(image.data) }, JPEG_QUALITY);
    const dec = jpegJs.decode(enc.data, { useTArray: true, formatAsRGBA: true });
    const lossy = jpegNoise({ width: dec.width, height: dec.height, data: new Uint8ClampedArray(dec.data) });
    assert.ok(lossy.likely && lossy.fraction > 0.2, `fraction ${lossy.fraction}`);
});

test('jpegNoise: the bench renders have none and their JPEGs have it', { skip: !existsSync(BENCH) && 'bench1 absent' }, () => {
    for (const f of readdirSync(BENCH).filter(f => /\.(png|jpg)$/.test(f))) {
        const r = jpegNoise(readImage(BENCH + f));
        assert.equal(r.likely, f.endsWith('.jpg'), `${f}: ${(100 * r.fraction).toFixed(3)}%`);
    }
});

const NATURE = new URL('../bench-nature/', import.meta.url).pathname;
const PRIMATES1 = 'primates1-10.1038-s41586-026-11005-5.png';

// Written for the web capture of primates1, a PNG of a JPEG. The PDF capture that replaced it (2026-11-14) is clean
// and needs no fallback, and a JPEG made here doesn't reproduce the failure (jpeg-js keeps full chroma), so the test
// waits for a JPEG-derived image under that name.
const primates1Skip = !existsSync(NATURE + PRIMATES1) ? 'bench-nature absent'
    : !jpegNoise(readImage(NATURE + PRIMATES1)).likely && 'primates1 is no longer a PNG of a JPEG';

test('regression: a PNG made from a JPEG is analyzed as a JPEG when no family shows with exact colors', { skip: primates1Skip }, () => {
    // before: "No dot family found" (the blue dots' edges leave the white–blue line, so no isolated dot passed)
    const results = extractAll(readImage(NATURE + PRIMATES1), false);
    assert.equal(results.length, 1);
    assert.equal(results[0].jpegAuto, true);
    assert.ok(Math.abs(results[0].fit.radius - 13) < 1, `radius ${results[0].fit.radius}`);
    const n = results[0].fit.dots.length;
    assert.ok(n >= 26 && n <= 28, `${n} dots`); // 27, every dot in the image (checked on an overlay)
});

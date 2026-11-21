import { test } from 'node:test';
import assert from 'node:assert/strict';
import { existsSync } from 'node:fs';
import { readdirSync } from 'node:fs';
import jpegJs from 'jpeg-js';
import { restoreChroma, jpegNoise } from '../src/core/jpeg.js';
import { synthGraph, SYNTH_PRESETS } from '../src/core/synth.js';
import { readImage } from '../tools/png.mjs';
import { readTruthCsv, truthPixels } from '../tools/truth-csv.mjs';
import { readTruthJson } from '../tools/truth-json.mjs';
import { extractAll, scanFamilies } from '../src/core/extract.js';
import { groupRow } from '../tools/bench.mjs';

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

const PLOS = new URL('../bench-plos/', import.meta.url).pathname;
const STEP1 = 'step1-10.1371-journal.pone.0360641';

// On step1, a PNG with JPEG noise (first written for the web capture of primates1, a PNG of a JPEG, which a clean
// PDF capture replaced; a JPEG made here doesn't reproduce it, as jpeg-js keeps full chroma). Skipped if step1 is
// ever replaced by a clean capture too.
const step1Skip = !existsSync(`${PLOS}${STEP1}.png`) ? 'bench-plos absent'
    : !jpegNoise(readImage(`${PLOS}${STEP1}.png`)).likely && 'step1 no longer has JPEG noise';

test('regression: a PNG with JPEG noise is analyzed as a JPEG when no family with a radius shows with exact colors', { skip: step1Skip }, () => {
    // before (primates1): "No dot family found" (the blue dots' edges leave the white–blue line, so no isolated dot
    // passed). step1 with exact colors: the blue dots alone, their radius unknown, so nothing to fit, and the red
    // rings not at all; with noisy matching both, at 5.4 and 6.4 px.
    const image = readImage(`${PLOS}${STEP1}.png`);
    const exact = scanFamilies(image).prepared.estimates;
    assert.equal(exact.filter(e => e.style?.radius).length, 0, `with exact colors: ${exact.map(e => e.style?.radius).join(', ')}`);
    const scan = scanFamilies(image, { tryNoisy: true });
    assert.equal(scan.switched, true);
    assert.equal(scan.exact, 0);
    assert.equal(scan.found, 2);
    assert.notEqual(scan.work, image, 'analyzed with its chroma restored');
    const results = extractAll(image, false);
    assert.equal(results.length, 2);
    assert.ok(results.every(r => r.jpegAuto));
    const blue = results.find(r => r.est.style.color[2] > r.est.style.color[0]);
    const red = results.find(r => r.est.style.color[0] > r.est.style.color[2]);
    assert.ok(Math.abs(blue.fit.radius - 5.45) < 0.5, `blue radius ${blue.fit.radius}`);
    assert.ok(Math.abs(red.fit.radius - 6.3) < 0.6, `ring radius ${red.fit.radius}`);
    // measured: 73 blue dots and 55 rings, 128 for 175 by hand (those under the curve and the dashed line are missed)
    const truth = readTruthJson(`${PLOS}truth/${STEP1}.json`);
    const r = groupRow('step1', truth.dots, results, 0);
    assert.ok(r.precision >= 0.95 && r.recall >= 0.68, `precision ${r.precision}, recall ${r.recall}, found ${r.found}`);
});

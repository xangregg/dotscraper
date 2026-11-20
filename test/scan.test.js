import { test } from 'node:test';
import assert from 'node:assert/strict';
import { backgroundColor, flatMask, scanOf, solidRegion, EXACT_TOLERANCES, JPEG_TOLERANCES } from '../src/core/scan.js';
import { estimateStyle } from '../src/core/style.js';
import { synthGraph, SYNTH_PRESETS } from '../src/core/synth.js';

// a 60 × 40 white page with a 20 × 12 block of one color and a column of single pixels of another
function blockImage() {
    const width = 60;
    const height = 40;
    const data = new Uint8ClampedArray(4 * width * height).fill(255);
    const set = (x, y, c) => data.set(c, 4 * (y * width + x));
    for (let y = 10; y < 22; y++) {
        for (let x = 5; x < 25; x++)
            set(x, y, [30, 90, 200]);
    }
    for (let y = 4; y < 36; y += 4)
        set(45, y, [200, 40, 40]);
    return { width, height, data };
}

test('backgroundColor: the most common color and its share', () => {
    const { color, fraction } = backgroundColor(blockImage());
    assert.deepEqual(color, [255, 255, 255]);
    assert.ok(Math.abs(fraction - (60 * 40 - 20 * 12 - 8) / (60 * 40)) < 1e-12, `fraction ${fraction}`);
});

test('flatMask: inside a block and on the open page, not at an edge or beside a speck', () => {
    const image = blockImage();
    const flat = flatMask(image);
    const at = (x, y) => flat[y * image.width + x];
    assert.equal(at(15, 16), 1, 'inside the block');
    assert.equal(at(35, 30), 1, 'open page');
    assert.equal(at(5, 16), 0, 'the block\'s edge');
    assert.equal(at(44, 8), 0, 'beside a speck');
    // a 5-unit ripple is flat for a JPEG, not for an exact render
    const noisy = { ...image, data: Uint8ClampedArray.from(image.data) };
    noisy.data[4 * (16 * image.width + 15)] = 35;
    assert.equal(flatMask(noisy, EXACT_TOLERANCES)[16 * image.width + 15], 0);
    assert.equal(flatMask(noisy, JPEG_TOLERANCES)[16 * image.width + 15], 1);
});

test('scanOf and solidRegion: a block is a solid region, specks are not; the answer is remembered per color and radius', () => {
    const image = blockImage();
    const scan = scanOf(image);
    assert.deepEqual(scan.B, [255, 255, 255]);
    assert.equal(scan.image, image);
    assert.equal(scan.tol, EXACT_TOLERANCES);
    assert.equal(scanOf(image, { jpeg: true }).tol, JPEG_TOLERANCES);
    assert.equal(solidRegion(scan, [30, 90, 200], 3), true);
    assert.equal(solidRegion(scan, [200, 40, 40], 3), false);
    const remembered = scan.memo.size;
    assert.equal(remembered, 2);
    assert.equal(solidRegion(scan, [30, 90, 200], 3), true);
    assert.equal(scan.memo.size, remembered);
    // a copy with another flat mask shares the memo; a bare scan computes without one
    assert.equal({ ...scan, flat: new Uint8Array(scan.flat.length) }.memo, scan.memo);
    assert.equal(solidRegion({ image }, [30, 90, 200], 3), true);
});

test('estimateStyle with a scan made earlier gives the same estimate, and a scan of another image is not used', () => {
    const { image } = synthGraph(SYNTH_PRESETS.sparse(3));
    const other = synthGraph(SYNTH_PRESETS.dense(4)).image;
    const plain = estimateStyle(image);
    const scan = scanOf(image);
    assert.deepEqual(scan.B, plain.background.color);
    assert.equal(scan.backgrounds, scan.pageBackgrounds);
    for (const opts of [{ scan }, { scan }, { scan: scanOf(other) }]) {
        const est = estimateStyle(image, opts);
        assert.deepEqual(est.style, plain.style);
        assert.equal(est.candidates.length, plain.candidates.length);
        assert.deepEqual(est.D, plain.D);
    }
    // regions extend the backgrounds for that call only
    estimateStyle(image, { scan, regions: [[200, 220, 240]] });
    assert.equal(scan.backgrounds, scan.pageBackgrounds);
});


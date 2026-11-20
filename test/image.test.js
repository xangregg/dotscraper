import { test } from 'node:test';
import assert from 'node:assert/strict';
import { pixelHash, pixelAt, nearColor, nearAnyColor, colorMask, colorDistance, sub, dot, unit } from '../src/core/image.js';

function solid(width, height, rgba) {
    const data = new Uint8ClampedArray(4 * width * height);
    for (let p = 0; p < width * height; p++)
        data.set(rgba, 4 * p);
    return { width, height, data };
}

test('pixelHash is the FNV-1a hash of the bytes', () => {
    // FNV-1a 32-bit of the empty input is the offset basis; of the single byte 0x61 ('a') it is e40c292c
    assert.equal(pixelHash({ width: 0, height: 0, data: new Uint8ClampedArray(0) }), '811c9dc5');
    assert.equal(pixelHash({ width: 1, height: 1, data: new Uint8ClampedArray([0x61]) }), 'e40c292c');
});

test('pixelHash changes when one channel of one pixel changes', () => {
    const a = solid(5, 4, [10, 20, 30, 255]);
    const b = solid(5, 4, [10, 20, 30, 255]);
    assert.equal(pixelHash(a), pixelHash(b));
    b.data[4 * 7 + 2] = 31;
    assert.notEqual(pixelHash(a), pixelHash(b));
});

test('pixelAt indexes row-major from the top left', () => {
    const im = solid(3, 2, [0, 0, 0, 255]);
    im.data.set([1, 2, 3, 4], 4 * (1 * 3 + 2));
    assert.deepEqual(pixelAt(im, 2, 1), [1, 2, 3, 4]);
    assert.deepEqual(pixelAt(im, 0, 0), [0, 0, 0, 255]);
});

// every color within 12 units per channel of (100, 100, 100): 25³ pixels, so every integer distance up to 20 occurs
function colorCube() {
    const side = 25;
    const data = new Uint8ClampedArray(4 * side ** 3);
    let p = 0;
    for (let r = 88; r < 88 + side; r++) {
        for (let g = 88; g < 88 + side; g++) {
            for (let b = 88; b < 88 + side; b++)
                data.set([r, g, b, 255], 4 * p++);
        }
    }
    return { width: side * side, height: side, data };
}

test('nearColor is Math.hypot(...) <= tol for every pixel, also at exact distances and for fractional colors', () => {
    // (for tolerances whose square is exact, as the app's are: whole and half units)
    const { data } = colorCube();
    // 3, 7 and 9 are the lengths of integer triples such as (1, 2, 2), (2, 3, 6) and (1, 4, 8): pixels exactly at
    // the tolerance, where a squared distance and a rounded Math.hypot could part
    for (const color of [[100, 100, 100], [99.5, 101.25, 100.1], [93.4, 104.7, 98.2]]) {
        for (const tol of [0, 1, 3, 5, 7, 8, 9, 15, 2.5, 7.5, -1, NaN]) {
            const near = nearColor(data, color, tol);
            let inside = 0;
            for (let p = 0; p < data.length / 4; p++) {
                const want = Math.hypot(data[4 * p] - color[0], data[4 * p + 1] - color[1], data[4 * p + 2] - color[2]) <= tol;
                assert.equal(near(p), want, `pixel ${p}, color ${color}, tol ${tol}`);
                inside += want;
            }
            if (tol >= 3)
                assert.ok(inside > 0 && inside < data.length / 4, `tol ${tol} splits the cube`);
        }
    }
});

test('colorMask and nearAnyColor mark the same pixels as nearColor', () => {
    const image = colorCube();
    const colors = [[92, 92, 92], [108, 100, 96]];
    const tests = colors.map(c => nearColor(image.data, c, 5));
    const any = nearAnyColor(image.data, colors, 5);
    const mask = colorMask(image, colors[0], 5);
    let count = 0;
    for (let p = 0; p < mask.length; p++) {
        assert.equal(mask[p], tests[0](p) ? 1 : 0);
        assert.equal(any(p), tests[0](p) || tests[1](p));
        count += mask[p];
    }
    assert.ok(count > 100, `${count} pixels within 5 units`);
    assert.equal(nearAnyColor(image.data, [], 5)(0), false);
});

test('colorDistance is Math.hypot of the channel differences, however a call site used to spell it', () => {
    const a = [12.5, 200, 31];
    const b = [250, 3, 77.25];
    const d = Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
    assert.equal(colorDistance(a, b), d);
    assert.equal(colorDistance(a, b), Math.hypot(...a.map((v, ch) => v - b[ch])));
    assert.equal(colorDistance(b, a), d);
    assert.equal(colorDistance([3, 4, 0], [0, 0, 0]), 5);
    assert.equal(colorDistance(a, a), 0);
    // only the color channels count: a pixel's α doesn't
    assert.equal(colorDistance([3, 4, 0, 255], [0, 0, 0]), 5);
});

test('sub, dot and unit: a color\'s offset from another, the inner product, and a direction', () => {
    assert.deepEqual(sub([10, 20, 30], [1, 2, 3]), [9, 18, 27]);
    assert.equal(dot([1, 2, 3], [4, 5, 6]), 32);
    assert.deepEqual(unit([0, 3, 4]), [0, 0.6, 0.8]);
    const u = unit(sub([255, 255, 255], [31, 119, 180]));
    assert.ok(Math.abs(dot(u, u) - 1) < 1e-12);
});

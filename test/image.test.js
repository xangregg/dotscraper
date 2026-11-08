import { test } from 'node:test';
import assert from 'node:assert/strict';
import { pixelHash, pixelAt } from '../src/core/image.js';

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

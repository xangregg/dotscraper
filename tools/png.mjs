// Read and write PNG files, and read JPEG files, in Node (tools and tests only; the app decodes images with the
// browser). Returns the shape src/core expects: { width, height, data } with data an RGBA Uint8ClampedArray.
//
// JPEG decoding isn't bit-exact across decoders (IDCT precision, chroma upsampling): on bench1/smoother.jpg,
// jpeg-js and macOS agree on 97% of channel values, but differ by up to 31 levels at color edges, where the
// half-resolution chroma is upsampled differently. Fine for benchmarks; don't pin exact pixel values of a JPEG.

import { readFileSync, writeFileSync } from 'node:fs';
import { PNG } from 'pngjs';
import jpeg from 'jpeg-js';

const JPEG_MAX_MEGAPIXELS = 100; // jpeg-js's own default is 100 too; stated so the limit is visible

/**
 * @param {string} path
 * @returns {{width: number, height: number, data: Uint8ClampedArray}}
 */
export function readPng(path) {
    const png = PNG.sync.read(readFileSync(path));
    return { width: png.width, height: png.height, data: new Uint8ClampedArray(png.data.buffer, png.data.byteOffset, png.data.length) };
}

/**
 * A PNG or JPEG file, by extension.
 *
 * @param {string} path
 * @returns {{width: number, height: number, data: Uint8ClampedArray}}  RGBA, 0–255, sRGB as stored
 */
export function readImage(path) {
    if (!/\.jpe?g$/i.test(path))
        return readPng(path);
    const jpg = jpeg.decode(readFileSync(path), { useTArray: true, formatAsRGBA: true, maxResolutionInMP: JPEG_MAX_MEGAPIXELS });
    return { width: jpg.width, height: jpg.height, data: new Uint8ClampedArray(jpg.data.buffer, jpg.data.byteOffset, jpg.data.length) };
}

/**
 * @param {string} path
 * @param {{width: number, height: number, data: Uint8ClampedArray}} image
 */
export function writePng(path, image) {
    const png = new PNG({ width: image.width, height: image.height });
    png.data = Buffer.from(image.data.buffer, image.data.byteOffset, image.data.length);
    writeFileSync(path, PNG.sync.write(png));
}

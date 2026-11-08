// Read and write PNG files in Node (tools and tests only; the app decodes images with the browser). Returns the
// shape src/core expects: { width, height, data } with data an RGBA Uint8ClampedArray.

import { readFileSync, writeFileSync } from 'node:fs';
import { PNG } from 'pngjs';

/**
 * @param {string} path
 * @returns {{width: number, height: number, data: Uint8ClampedArray}}
 */
export function readPng(path) {
    const png = PNG.sync.read(readFileSync(path));
    return { width: png.width, height: png.height, data: new Uint8ClampedArray(png.data.buffer, png.data.byteOffset, png.data.length) };
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

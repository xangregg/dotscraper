// Image helpers shared by the app, the worker, tools and tests. An image is { width, height, data } with data an
// RGBA Uint8ClampedArray, row-major, the shape of the browser's ImageData.

/**
 * @typedef {object} RgbaImage
 * @property {number} width   px
 * @property {number} height  px
 * @property {Uint8ClampedArray} data  RGBA, 0–255, row-major
 */

/**
 * FNV-1a hash of the RGBA bytes, as 8 hex digits. Used to check that the browser decodes an image to the same
 * values as Node (no color-profile conversion), and to key cached results.
 *
 * @param {RgbaImage} image
 * @returns {string}
 */
export function pixelHash(image) {
    let h = 0x811c9dc5;
    const d = image.data;
    for (let i = 0; i < d.length; i++) {
        h ^= d[i];
        h = Math.imul(h, 0x01000193);
    }
    return (h >>> 0).toString(16).padStart(8, '0');
}

/**
 * The RGBA values of one pixel.
 *
 * @param {RgbaImage} image
 * @param {number} x  column, 0-based
 * @param {number} y  row, 0-based
 * @returns {number[]}  [r, g, b, a], 0–255
 */
export function pixelAt(image, x, y) {
    const i = 4 * (y * image.width + x);
    return [image.data[i], image.data[i + 1], image.data[i + 2], image.data[i + 3]];
}

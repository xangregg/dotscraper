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

/**
 * The distance between two colors, the app's one measure of how alike they are: Euclidean, in RGB. Also between two
 * offsets or two directions in that space.
 *
 * @param {number[]} a  0–255
 * @param {number[]} b  0–255
 * @returns {number}  0–255 units
 */
export function colorDistance(a, b) {
    return Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
}

// Colors as vectors: a color's offset from another (a − b), the inner product of two offsets, and an offset's
// direction (a unit vector; not for a zero offset).

/**
 * @param {number[]} a
 * @param {number[]} b
 * @returns {number[]}
 */
export function sub(a, b) {
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

/**
 * @param {number[]} a
 * @param {number[]} b
 * @returns {number}
 */
export function dot(a, b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

/**
 * @param {number[]} v  not zero
 * @returns {number[]}
 */
export function unit(v) {
    const n = Math.hypot(...v);
    return v.map(x => x / n);
}

/**
 * A test of whether a pixel is within `tol` of a color (colorDistance). The scans make it tens of times per pixel, so it compares squared distances: Math.hypot is 7–9
 * times slower there. The two agree on every pixel, also on those
 * exactly `tol` away, when tol² is exact, as it is for the whole and half units used here (test/image.test.js).
 *
 * @param {Uint8ClampedArray} data  RGBA, 0–255
 * @param {number[]} color  0–255
 * @param {number} tol  0–255 units
 * @returns {(p: number) => boolean}  for pixel index p (row-major, not the byte offset)
 */
export function nearColor(data, color, tol) {
    const [c0, c1, c2] = color;
    // (a negative or NaN tolerance admits nothing)
    const limit = tol >= 0 ? tol * tol : -1;
    return p => {
        const d0 = data[4 * p] - c0;
        const d1 = data[4 * p + 1] - c1;
        const d2 = data[4 * p + 2] - c2;
        return d0 * d0 + d1 * d1 + d2 * d2 <= limit;
    };
}

/**
 * As nearColor, for any of several colors.
 *
 * @param {Uint8ClampedArray} data  RGBA, 0–255
 * @param {number[][]} colors  0–255
 * @param {number} tol  0–255 units
 * @returns {(p: number) => boolean}
 */
export function nearAnyColor(data, colors, tol) {
    const tests = colors.map(c => nearColor(data, c, tol));
    return p => {
        for (let i = 0; i < tests.length; i++) {
            if (tests[i](p))
                return true;
        }
        return false;
    };
}

/**
 * The pixels within `tol` of a color (nearColor), 1 or 0 per pixel.
 *
 * @param {RgbaImage} image
 * @param {number[]} color  0–255
 * @param {number} tol  0–255 units
 * @returns {Uint8Array}
 */
export function colorMask(image, color, tol) {
    const near = nearColor(image.data, color, tol);
    const mask = new Uint8Array(image.width * image.height);
    for (let p = 0; p < mask.length; p++)
        mask[p] = near(p) ? 1 : 0;
    return mask;
}

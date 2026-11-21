// What a scan knows about its image: the page and the other background colors, the flat pixels, the tolerances
// (exact renders or JPEGs), and the Scan that carries them through the family search (style.js) with the answers
// about single colors that many rules ask for (isThin, solidRegion, straightShare).

import { opening } from './layers.js';
import { backgroundColors } from './axes.js';
import { colorMask } from './image.js';

const FLAT_TOL = 3; // 0–255 units: max channel difference in a 3×3 neighborhood for a pixel to count as flat
// (in the count image, levels.js: perp is a pixel's distance off a family's color line, t its distance along it)
const NEAR_B_ABS_TOL = 2; // 0–255 units: perp always accepted below this, whatever t is
export const LEVEL_SEPARATION = 10; // 0–255 units between levels
export const CLUSTER_TOL = 40; // 0–255 units: max distance from a cluster's color for a pixel to belong to it

// A region is solid when much of its color survives an opening by a dot's radius; a "region" of dots doesn't
// (solidRegion). Measured: boxes, bands and the page 0.71–0.99; step1's blue band 0.52, every blue dot's antialiased
// rim passing through its color; regions made of dots or outlines 0.00–0.06 (engagement1's cyan, delta1's black,
// phase1's outlines).
const REGION_INK_TOL = 15; // 0–255 units (RGB distance) from a region's color (JPEG noise)
const REGION_MIN_SOLID = 0.3; // share of the region's color an opening by the dots' radius keeps
// (straightShare: a box under the dots is told by its straight edges, style.js boxBand)
export const BOX_COLOR_TOL = 3; // 0–255 units (RGB distance) from the box's color

// A fill is thick and an outline is a thin ring: an opening by a disc wider than the ring leaves little of its color
// (JMP's 1.7 px outlines keep 0.5–4.8%), so a thin cluster is no fill to try the outline model on (a synthetic ring's
// color read as a "fill" with the real fill for its outline)
const OUTLINE_OPENING = 1.5; // px, radius of the disc that must remove an outline's color …
const OUTLINE_MAX_THICK = 0.25; // … all but this share of it

/**
 * Tolerances that assume exact renders, and their JPEG replacements (for an image whose chroma jpeg.js has
 * restored). JPEG's block noise leaves dot interiors a few units from their color, so few 3×3 neighborhoods are
 * flat; and it rings the background next to edges with near-white pixels slightly off every color line ((252, 252,
 * 248) around orange dots), which made a false first level just above B, made every isolated dot look partly hidden,
 * and gave dots a faint halo that failed the disc test. Measured on the 12 bench JPEGs (PLAN.md phase 5): flat 5–8,
 * minLevelT 15–20, nearB 5–12 and isolatedFloor 0.1–0.2 work alike; flat 3, minLevelT 10, nearB 2 or isolatedFloor
 * 0.05 each lose images. The distance off the line for level votes stays PERP_TOL (6 lost an image).
 *
 * @typedef {object} Tolerances
 * @property {number} flat           0–255 units: max channel difference in a 3×3 neighborhood for a flat pixel
 * @property {number} minLevelT      0–255 units: flat pixels this close to B along the line don't vote for levels
 * @property {number} nearB          0–255 units: off-line distance always accepted in the count image
 * @property {number} isolatedFloor  counts below this weigh 0 in an isolated dot's area and moments
 * @property {number} levelSmooth    bins either side summed before the level histogram's peaks are found: JPEG noise
 *                                   spreads a small level over several bins (bench2 group-smoother.jpg's red level 2:
 *                                   29 flat pixels over 7 bins, none reaching PEAK_MIN_PIXELS)
 */
/** @type {Tolerances} */
export const EXACT_TOLERANCES = { flat: FLAT_TOL, minLevelT: LEVEL_SEPARATION / 2, nearB: NEAR_B_ABS_TOL, isolatedFloor: 0, levelSmooth: 0 };
/** @type {Tolerances} */
export const JPEG_TOLERANCES = { flat: 6, minLevelT: 15, nearB: 8, isolatedFloor: 0.1, levelSmooth: 3 };

/**
 * The most common color and its share of the pixels.
 *
 * @param {import('./image.js').RgbaImage} image
 * @returns {{color: number[], fraction: number}}
 */
export function backgroundColor(image) {
    const counts = new Map();
    const n = image.width * image.height;
    let best = 0;
    let bestKey = 0;
    for (let p = 0; p < n; p++) {
        const key = (image.data[4 * p] << 16) | (image.data[4 * p + 1] << 8) | image.data[4 * p + 2];
        const c = (counts.get(key) ?? 0) + 1;
        counts.set(key, c);
        if (c > best) {
            best = c;
            bestKey = key;
        }
    }
    return { color: [bestKey >> 16, (bestKey >> 8) & 255, bestKey & 255], fraction: best / n };
}

/**
 * 1 where every channel of the pixel's 3×3 neighborhood is within tol.flat of the pixel (borders are 0).
 *
 * @param {import('./image.js').RgbaImage} image
 * @param {Tolerances} [tol]
 * @returns {Uint8Array}
 */
export function flatMask(image, tol = EXACT_TOLERANCES) {
    const { width, height, data } = image;
    const flat = new Uint8Array(width * height);
    for (let y = 1; y < height - 1; y++) {
        for (let x = 1; x < width - 1; x++) {
            const p = y * width + x;
            let ok = 1;
            for (let dy = -1; dy <= 1 && ok; dy++) {
                for (let dx = -1; dx <= 1 && ok; dx++) {
                    const q = p + dy * width + dx;
                    for (let ch = 0; ch < 3; ch++) {
                        if (Math.abs(data[4 * q + ch] - data[4 * p + ch]) > tol.flat)
                            ok = 0;
                    }
                }
            }
            flat[p] = ok;
        }
    }
    return flat;
}

/**
 * Whether a color's pixels are thin as a whole: an opening by OUTLINE_OPENING keeps under OUTLINE_MAX_THICK of them.
 *
 * @param {Scan} scan
 * @param {number[]} color  0–255; its pixels are those within CLUSTER_TOL of it
 * @returns {boolean}
 */
export function isThin(scan, color) {
    return remembered(scan, 'thin', color, () => {
        const { image } = scan;
        const mask = colorMask(image, color, CLUSTER_TOL);
        let total = 0;
        for (let p = 0; p < mask.length; p++)
            total += mask[p];
        let kept = 0;
        for (const v of opening(mask, image.width, image.height, OUTLINE_OPENING))
            kept += v;
        return total > 0 && kept < OUTLINE_MAX_THICK * total;
    });
}

/**
 * What a scan knows about its image, made once and handed to everything below estimateStyle instead of the image,
 * the page, the flat mask, the backgrounds and the tolerances one by one. A rule that reads the image differently
 * passes a copy with that part replaced ({ ...scan, flat: rest }); the copy shares `memo`, which holds answers about
 * the image's colors that many rules ask for (remembered).
 *
 * @typedef {object} Scan
 * @property {import('./image.js').RgbaImage} image
 * @property {Tolerances} tol
 * @property {{color: number[], fraction: number}} background  the page (backgroundColor)
 * @property {number[]} B  the page's color, 0–255
 * @property {Uint8Array} flat  flatMask(image, tol)
 * @property {number[][]} pageBackgrounds  the page and the other large flat colors (axes.js backgroundColors)
 * @property {number[][]} backgrounds  pageBackgrounds, then the regions under the dots when there are any
 * @property {Map<string, any>} memo
 */

/**
 * @param {import('./image.js').RgbaImage} image
 * @param {object} [opts]
 * @param {boolean} [opts.jpeg]  noise-tolerant thresholds (JPEG_TOLERANCES)
 * @returns {Scan}
 */
export function scanOf(image, opts = {}) {
    const tol = opts.jpeg ? JPEG_TOLERANCES : EXACT_TOLERANCES;
    const background = backgroundColor(image);
    const pageBackgrounds = backgroundColors(image);
    return { image, tol, background, B: background.color, flat: flatMask(image, tol), pageBackgrounds, backgrounds: pageBackgrounds, memo: new Map() };
}

// An answer about one color of the scan's image that depends on nothing else, computed once per scan: the same
// colors come up in rule after rule (engagement1 asked isThin 10 times about 3 colors). A scan without a memo (a
// bare { image, ... }) computes every time.
function remembered(scan, what, color, compute) {
    if (!scan.memo)
        return compute();
    const key = `${what}|${color.join(',')}`;
    if (!scan.memo.has(key))
        scan.memo.set(key, compute());
    return scan.memo.get(key);
}

/**
 * Whether pixels of `color` form a solid region rather than dots: REGION_MIN_SOLID of them survive an opening by the
 * dots' radius. O(pixels · radius).
 *
 * @param {Scan} scan  its image; the answer is remembered per color and radius
 * @param {number[]} color  0–255
 * @param {number} radius  px, the dots'
 * @returns {boolean}
 */
export function solidRegion(scan, color, radius) {
    return remembered(scan, `solid ${radius}`, color, () => solidShare(scan.image, color, radius));
}

function solidShare(image, color, radius) {
    const { width, height } = image;
    const mask = colorMask(image, color, REGION_INK_TOL);
    let all = 0;
    for (let p = 0; p < mask.length; p++)
        all += mask[p];
    let kept = 0;
    for (const v of opening(mask, width, height, radius))
        kept += v;
    return all > 0 && kept >= REGION_MIN_SOLID * all;
}

/**
 * The share of the boundary of a color's pixels (within BOX_COLOR_TOL) in straight horizontal or vertical runs of at
 * least minRun px. O(pixels).
 *
 * @param {Scan} scan
 * @param {number[]} color  0–255
 * @param {number} minRun  px
 * @returns {number}  0–1
 */
export function straightShare(scan, color, minRun) {
    return remembered(scan, `straight ${minRun}`, color, () => straightEdgeShare(scan.image, color, minRun));
}

function straightEdgeShare(image, color, minRun) {
    const { width, height } = image;
    const m = colorMask(image, color, BOX_COLOR_TOL);
    let edge = 0;
    let straight = 0;
    // boundary pixels facing up, down, left and right, in runs along rows (up, down) or columns (left, right)
    for (const [dx, dy] of [[0, -1], [0, 1], [-1, 0], [1, 0]]) {
        const rows = dy !== 0;
        const outer = rows ? height : width;
        const inner = rows ? width : height;
        for (let a = 0; a < outer; a++) {
            let run = 0;
            for (let b = 0; b <= inner; b++) {
                let on = false;
                if (b < inner) {
                    const x = rows ? b : a;
                    const y = rows ? a : b;
                    const X = x + dx;
                    const Y = y + dy;
                    on = m[y * width + x] === 1 && (X < 0 || Y < 0 || X >= width || Y >= height || !m[Y * width + X]);
                }
                if (on) {
                    run++;
                    edge++;
                    continue;
                }
                if (run >= minRun)
                    straight += run;
                run = 0;
            }
        }
    }
    return edge ? straight / edge : 0;
}

// Dot style estimation (PLAN.md §2.2, stage 4): the background, the translucent dot families, each family's
// overlap levels, alpha, full color, radius and edge blur. Phase 0 (tools/measure-dots.mjs) established the
// approach on real images; this is its library form.
//
// A translucent family's pixels lie on the line from the background B toward its full color c, at plateaus
// L_n for n overlapping dots. Levels are found as histogram peaks of *flat* pixels projected on that line (edge
// pixels would make false peaks around small dots). Renderers composite in sRGB with constant alpha, but 8-bit
// rounding drifts with depth and some images have an odd first step, so the level colors used by the forward
// model are the measured ones; a geometric fit gives α and c and extrapolates beyond the deepest measured level.

import { houghCenters } from './fit.js';
import { distanceTransform, dilate, nearestTransform, opening } from './layers.js';
import { backgroundColors } from './axes.js';
import { pixelCoverage } from './render.js';

const BACKGROUND_TOL = 3; // 0–255 units (Euclidean RGB): a pixel this close to B is background
const FLAT_TOL = 3; // 0–255 units: max channel difference in a 3×3 neighborhood for a pixel to count as flat
const PERP_TOL = 4; // 0–255 units: max distance off the color line for a level-voting pixel
// Near B every color is close to the line in absolute terms (antialiased gray text one step from white is within
// PERP_TOL of any line), so the distance off the line must also be small relative to the distance along it.
const PERP_REL_TOL = 0.12; // max perp / t
// For the count image, faint edge pixels near B must still count, or dots lose ink at their rims and read small,
// and a rim that is off the line makes a dot look partly hidden. Rims in dog-ages sit at perp/t ≈ 0.14 (perp
// 2.2–2.8 at t 13–20, from resampling); one-step-from-white gray text sits at 0.31.
const NEAR_B_ABS_TOL = 2; // 0–255 units: perp always accepted below this, whatever t is
const COUNT_PERP_REL_TOL = 0.2; // max perp / t for a pixel to get an overlap count
const COUNT_PERP_TOL = 12; // 0–255 units: max perp for a pixel to get an overlap count
const DIRECTION_MIN_T = 15; // 0–255 units: flat pixels at least this far from B vote for family directions
const DIRECTION_BIN = 2; // degrees, family direction histogram
const MAX_CANDIDATES = 6; // family directions examined
const LEVEL_BIN = 1; // 0–255 units along the line, level histogram
const LEVEL_MAX_T = 450; // 0–255 units; > the RGB cube diagonal
const PEAK_MIN_FRACTION = 0.004; // a level's peak bin must hold this fraction of the family's flat pixels
const PEAK_MIN_PIXELS = 20; // and at least this many
const LEVEL_SEPARATION = 10; // 0–255 units between levels
const PEAK_HALF_WIDTH = 3; // bins either side summed into a level's color
const MAX_STEP_RATIO = 0.95; // a level reached by a step larger than this × the previous step ends the sequence
// Sparse translucent dots overlap rarely, so the second level can be a handful of pixels (bench1: the one coincident
// pair gives 2 flat pixels at r 2.6 and 12 at r 3.8). With only one level found, a weaker peak is the second level
// if it sits where a second level can: t₂ / t₁ = 2 − α, between these ratios.
const WEAK_LEVEL_MIN_PIXELS = 2;
const WEAK_LEVEL_RATIO = [1.05, 1.95];
// A translucent family must fit the geometric model with a full color c a renderer could use. Gray text with two
// shades can mimic a level sequence, but its fitted c falls far outside the RGB cube.
const FIT_MAX_RMS = 2; // 0–255 units
const FIT_C_TOL = 10; // 0–255 units c may fall outside [0, 255] (fitting noise)
const EXTRA_LEVELS = 12; // geometric levels appended beyond the deepest measured one
const MAX_MISSING_LEVELS = 1; // overlap counts that may lack a level between measured ones (JPEG lenses)
const ISOLATED_MAX_LEVEL = 1.25; // components never above this count are single dots
const ISOLATED_MIN_AREA = 12; // px² (JMP's tiny dots: r 2.6, area 21)
const ISOLATED_MAX_ASPECT = 1.05; // ratio of principal second-moment radii
const ISOLATED_MAX_MOMENT_RATIO = 1.03; // r(moment) / r(area); 1 for a uniform disc
// Outlined dots (bench1 stroked-dots: a light gray fill, a darker ring 2 px wide at the rim) read deeper than one dot
// at the ring, so none passed as isolated. A component above ISOLATED_MAX_LEVEL is still one dot if every such pixel
// lies in its outer part, beyond this fraction of the radius from the center.
const RING_MIN_RADIUS_FRACTION = 0.6;
const RING_INNER_RANGE = [0.5, 0.97]; // the ring's inner radius searched, × the radius
const RING_INNER_STEP = 0.01;
const RING_MIN_ALPHA = 0.05; // extra opacity below this is no ring
const RINGED_MIN_SHARE = 0.5; // of the isolated dots, for the family to be outlined
const EDGE_RING = 4; // px outside a dot sampled for its edge
const PROFILE_BIN = 0.25; // px, radial edge profile bins
// Opaque families (PLAN.md phase 5) are color clusters, not directions: a beeswarm's dots and the violin shading
// under them can share a hue, and compressed images (violin) have dot interiors too noisy to be flat.
const CLUSTER_BIN = 8; // 0–255 units per channel, color histogram bins
const CLUSTER_MIN_FRACTION = 0.001; // a cluster holds at least this share of the image …
const CLUSTER_MIN_PIXELS = 100; // … and at least this many pixels
const CLUSTER_TOL = 40; // 0–255 units: max distance from a cluster's color for a pixel to belong to it
const CLUSTER_ON_LINE_TOL = 8; // 0–255 units: a cluster this close to a translucent family's line is one of its levels
const INK_TOL = 15; // 0–255 units from every background color for a pixel to be ink
const RING = 3; // px: the ring around a family's pixels whose most common color is its background
// A dot family (measured on every test image, PLAN.md phase 5): radius ≥ DOT_MIN_RADIUS (text and thin lines give
// 2.5–2.74), at least DOT_MIN_STRONG filled centers with DOT_STRONG_ARC of their edge seen (nicest language's red:
// 3 of 9), and either discs that explain DOT_MIN_COVERAGE of the cluster's pixels (dots: 0.44–1.00; text, dashes,
// gridlines: ≤ 0.23) or
// DOT_MIN_ISOLATED whole isolated dots (black and gray dots share their color with text: coverage 0.18–0.23, but
// 9 and 17 isolated dots).
const DOT_MIN_RADIUS = 3; // px
// Smaller dots count when isolated whole dots show them round (bench1 tiny-dots: r 2.6, 58 isolated; text clusters
// measure r 2.5–2.74 but never have an isolated round component)
const SMALL_DOT_MIN_RADIUS = 2; // px
const SMALL_DOT_MIN_ISOLATED = 5;
export const SPECK_RADIUS_RATIO = 0.4; // × the largest family's radius: below it, a mostly-text color's dots are specks
const DOT_MIN_ARC = 0.5; // fraction of the circumference for a Hough center to count
const DOT_STRONG_ARC = 0.75;
const DOT_MIN_STRONG = 3;
const DOT_MIN_COVERAGE = 0.35;
// … and with isolated dots instead, at least this much: a color 98% text with a few round specks (hair2–3: decimal
// points and a colon's dots, coverage 0.02) is no family, even alone; real ones sharing text's color measured 0.18
const DOT_MIN_COVERAGE_ANY = 0.1;
const DOT_MIN_ISOLATED = 3;
// Text looks like dots to a Hough vote (round glyphs, and every gray of its antialiasing is a color cluster), but
// it is strokes: mostly edge pixels, and the inside of an "o" is empty.
const DOT_MIN_INTERIOR = 0.1; // share of a cluster's pixels with all 8 neighbors in the cluster
const BLEND_OUTLINE_MAX_INTERIOR = 0.3; // a cluster between a fill and its outline with more interior is a family
const DOT_FILL_RADII = 0.75; // × r: the disc that must be filled around a Hough center …
const DOT_MIN_FILL = 0.85; // … to this fraction
// distance-transform peaks underestimate the radius where dots touch or overlap; the upper quartile of the peaks,
// plus the half pixel the transform loses between pixel centers, matched synthetic swarms within a few tenths
const DT_RADIUS_QUANTILE = 0.75;
const DT_RADIUS_OFFSET = 0.5; // px
// A dot color can also be text's (nicest language: its gray dots' color is the gray of antialiased black text on the
// white page, which made white their "background" and text strokes their radius). Text is thin: the family's
// background comes from its thick parts (an opening), and distance peaks below DOT_MIN_RADIUS are ignored.
const THICK_OPENING = 1.5; // px
const BOUNDARY_PROFILE_REACH = 4; // px either side of a boundary for the edge-blur fit
// Opaque coverage is measured against the background where each pixel is (shading, white, a neighboring dot of
// another color), estimated from nearby pixels that are not the family's: violin's dots sit on shading and on
// white, and against a single background color the edges on the other one were unreadable.
const NEAR_FAMILY = 2; // px: pixels this close to the family's color don't inform its background
const MIN_CONTRAST = 30; // 0–255 units between a dot's color and the local background for its coverage to be read
// A pixel is a mix of the dot and its background only if it lies close to the line between them, within OFF_MIX_ABS
// or OFF_MIX_REL × its distance along the line. An absolute 25 units let another family's antialiased rims through
// (synthetic swarm: blue–white rims read as gray coverage 0.89 at 20.5 units off, about 13° off the gray mix, and
// births put phantom gray dots on the blue ones); a real rim pixel is off only by noise.
const OFF_MIX_ABS = 10; // 0–255 units, at least …
const OFF_MIX_NOISE = 2.5; // … this many times the family's own color noise (compressed images: violin, nicest)
const OFF_MIX_REL = 0.12;
// … and so is one well past the dot's color along the mix (black text against light gray dots: on the same gray
// axis, it read as coverage 4, clipped to 1, and text strokes became the family's radius)
const MAX_MIX = 1.3;
// Some renderers draw every overlap of opaque dots in one dark color, as dark for three as for two (fanc2's PDF:
// gray 86 dots, overlaps 19, the text's black). A region beyond the family's color on its line, enclosed by the
// family and not another family's color, is such an overlap: covered, not unknown. Text and lines border mostly on
// the page; a darker family's dot packed among this family's in a swarm also borders gaps and other dots (at 0.6
// the synthetic swarm's maroon filled in).
const ISOLATED_FILL_RADII = 0.5; // × r: the middle of an isolated dot that must be the family's fill
const OVERLAP_BORDER_SHARE = 0.9; // of a region's border pixels that must be the family's
// … and a lens has a middle: a line drawn over a swarm in the same dark ink is enclosed by dots too (the synthetic
// swarm's black median line over its light gray dots, 2 px wide), but has no square this deep inside it
const OVERLAP_MIN_DEPTH = 2; // px, half the side of a square that must fit inside the region
// An overlap is darker than one dot by at most a second dot's contrast (fanc2: 19 at 1.4 of the way from the page to
// 86). Unbounded, a pale family's line reached black text (economist: a light gray "family" at 226 filled its text).
const OVERLAP_MAX_MIX = 2; // × the family's contrast with its background
const OUTLINE_SLACK = 0.1; // a pixel this far beyond either end of the fill–outline segment still lies on it
// An outline is the family's only next to its fill: farther out, the outline's color is other ink (phase1: black
// outlines around gray fills, and black text, axes and a black family, which counted as dots everywhere)
const OUTLINE_REACH = 4; // px from the family's fill
// A family's ring image (familyRings) reads a full outline as this, below outline.js's MASK_INK (0.8), which masks
// near-black ink as text: the reach already masks text here, and black outlines are exactly text's color
const FAMILY_RING_P = 0.45;
const COLLINEAR_SIN2 = 0.01; // sin² of the angle between page→fill and page→outline below which they're one line
// A fill is thick and an outline is a thin ring: an opening by a disc wider than the ring leaves little of its color
// (JMP's 1.7 px outlines keep 0.5–4.8%), so a thin cluster is no fill to try the outline model on (a synthetic ring's
// color read as a "fill" with the real fill for its outline)
const OUTLINE_OPENING = 1.5; // px, radius of the disc that must remove an outline's color …
const OUTLINE_MAX_THICK = 0.25; // … all but this share of it
// Stroked fills (PLAN.md open question 10): a fill inside a stroke so thin and blurred that it never shows its own
// color (bench-nature engagement1: on lavender 188, 186, 234 its darkest pixel is about 80, 80, 105). Its edge
// pixels mix three colors, background, stroke and fill, where the opaque test accepts two, so the edge read as
// unknown and no dot was found. The stroke is read from the rings of isolated fills, as the mean of the ring pixels
// farthest off the background–fill line: not its true color, which no pixel shows, but the far corner of the
// colors seen, so every ring pixel lies inside the triangle background–fill–stroke.
const STROKE_RING = 4; // px outside a fill read as its ring (engagement1: dark out to 3–4 px past the fill's color, background from 5)
const STROKE_CLEAR = 7; // px: beyond the ring, out to this, a fill must have background around it to count as isolated …
const STROKE_CLEAR_SHARE = 0.9; // … for this share of those pixels
const STROKE_COMPONENT = [12, 5000]; // px, sizes of the fill components whose rings are read
const STROKE_MIN_RINGS = 3; // isolated fills needed
const STROKE_TOP = 0.8; // quantile of the ring pixels' distance off the line: those beyond are the stroke
const STROKE_MIN_OFF = 30; // 0–255 units: the stroke must lie this far off the background–fill line
const STROKE_SLACK = 0.15; // barycentric weights may go this far below 0
// A ring belongs to the fill it surrounds: in each family's coverage, a ring pixel nearer another family's fill (by
// more than STROKE_TIE) is that dot's, and reads 0, so a disc of this family can't be placed to explain it.
const STROKE_TIE = 0.5; // px

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
 * @typedef {object} ColorLine
 * @property {number[]} B  background color
 * @property {number[]} u  unit vector from B toward the family's colors
 */

/**
 * @typedef {object} Level
 * @property {number} t      0–255 units along the line
 * @property {number[]} color  mean color of the flat pixels at the level
 * @property {number} count  flat pixels at the level
 */

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
 * The line from B toward a seed color.
 *
 * @param {number[]} B
 * @param {number[]} seed
 * @returns {ColorLine}
 */
export function colorLine(B, seed) {
    const d = [B[0] - seed[0], B[1] - seed[1], B[2] - seed[2]];
    const len = Math.hypot(...d);
    return { B, u: d.map(v => v / len) };
}

/**
 * Distance along the line (t) and off it (perp) for every pixel.
 *
 * @param {import('./image.js').RgbaImage} image
 * @param {ColorLine} line
 * @returns {{t: Float32Array, perp: Float32Array}}
 */
export function projectOnLine(image, line) {
    const { B, u } = line;
    const n = image.width * image.height;
    const t = new Float32Array(n);
    const perp = new Float32Array(n);
    for (let p = 0; p < n; p++) {
        const e0 = B[0] - image.data[4 * p];
        const e1 = B[1] - image.data[4 * p + 1];
        const e2 = B[2] - image.data[4 * p + 2];
        const tp = e0 * u[0] + e1 * u[1] + e2 * u[2];
        t[p] = tp;
        perp[p] = Math.hypot(e0 - tp * u[0], e1 - tp * u[1], e2 - tp * u[2]);
    }
    return { t, perp };
}

function votesForLevel(t, perp) {
    return perp <= PERP_TOL && perp <= PERP_REL_TOL * t;
}

/**
 * Overlap levels along a color line: peaks of the histogram of t over flat on-line pixels. The sequence ends at
 * the first level whose step exceeds MAX_STEP_RATIO times the previous one (overlap steps shrink by 1 − α);
 * later peaks are returned as `beyond`, typically opaque marks in the family's full color.
 *
 * @param {import('./image.js').RgbaImage} image
 * @param {{t: Float32Array, perp: Float32Array}} proj
 * @param {Uint8Array} flat
 * @param {Tolerances} [tol]
 * @returns {{levels: Level[], beyond: Level[]}}
 */
export function findLevels(image, proj, flat, tol = EXACT_TOLERANCES) {
    const { t, perp } = proj;
    const nb = Math.ceil(LEVEL_MAX_T / LEVEL_BIN);
    const raw = new Float64Array(nb + 1);
    let total = 0;
    for (let p = 0; p < t.length; p++) {
        if (flat[p] && t[p] > tol.minLevelT && votesForLevel(t[p], perp[p])) {
            raw[Math.min(nb, Math.round(t[p] / LEVEL_BIN))]++;
            total++;
        }
    }
    const hist = smoothed(raw, tol.levelSmooth);
    const minPeak = Math.max(PEAK_MIN_PIXELS, PEAK_MIN_FRACTION * total);
    const peaks = [];
    for (let b = 1; b < nb; b++) {
        if (hist[b] < minPeak || hist[b] < hist[b - 1] || hist[b] < hist[b + 1])
            continue;
        const last = peaks[peaks.length - 1];
        if (last !== undefined && (b - last) * LEVEL_BIN < LEVEL_SEPARATION) {
            if (hist[b] > hist[last])
                peaks[peaks.length - 1] = b;
            continue;
        }
        peaks.push(b);
    }
    if (peaks.length === 1) {
        // the strongest bin in the second-level window, its neighbors summed (a plateau of a few pixels can straddle
        // two bins)
        let best = -1;
        let bestCount = 0;
        // and as far from the first level as any two levels (shoulders of the first peak in compressed images gave
        // ratios 1.05–1.07: beeswarm-boxplot, fibers, nicest language, violin)
        const lo = Math.ceil(Math.max(WEAK_LEVEL_RATIO[0] * peaks[0], peaks[0] + LEVEL_SEPARATION / LEVEL_BIN));
        const hi = Math.min(nb - 1, Math.floor(WEAK_LEVEL_RATIO[1] * peaks[0]));
        // raw counts: smoothing would carry the first peak's shoulder past LEVEL_SEPARATION (bench3 dots-packed.jpg)
        for (let b = lo; b <= hi; b++) {
            const c = raw[b - 1] + raw[b] + raw[b + 1];
            if (raw[b] > 0 && c > bestCount) {
                best = b;
                bestCount = c;
            }
        }
        if (bestCount >= WEAK_LEVEL_MIN_PIXELS)
            peaks.push(best);
    }
    const found = peaks.map(b => {
        const sum = [0, 0, 0];
        let count = 0;
        for (let p = 0; p < t.length; p++) {
            if (flat[p] && votesForLevel(t[p], perp[p]) && Math.abs(t[p] / LEVEL_BIN - b) <= PEAK_HALF_WIDTH) {
                sum[0] += image.data[4 * p];
                sum[1] += image.data[4 * p + 1];
                sum[2] += image.data[4 * p + 2];
                count++;
            }
        }
        return { t: b * LEVEL_BIN, color: sum.map(s => s / count), count };
    });
    let nSeq = found.length;
    for (let i = 1; i < found.length; i++) {
        const prevStep = found[i - 1].t - (i > 1 ? found[i - 2].t : 0);
        if ((found[i].t - found[i - 1].t) / prevStep > MAX_STEP_RATIO) {
            nSeq = i;
            break;
        }
    }
    return { levels: found.slice(0, nSeq), beyond: found.slice(nSeq) };
}

// box sum over `half` bins either side
function smoothed(hist, half) {
    if (!half)
        return hist;
    const out = new Float64Array(hist.length);
    for (let b = 0; b < hist.length; b++) {
        for (let k = Math.max(0, b - half); k <= Math.min(hist.length - 1, b + half); k++)
            out[b] += hist[k];
    }
    return out;
}

/**
 * Least-squares fit of L_n = c + (B − c) q^n per channel with one q = 1 − α shared by the channels, to colors
 * L_0 = B, L_1, …: golden-section search over q, c in closed form per channel for each q.
 *
 * @param {number[][]} colors  B and the measured levels
 * @param {number[]} [ns]  each color's overlap count, increasing from 0 (default 0, 1, 2, …: no level missing)
 * @returns {{alpha: number, c: number[], rms: number, maxAbs: number}}
 */
export function fitGeometric(colors, ns = colors.map((_, n) => n)) {
    const evaluate = q => {
        const c = [];
        let sse = 0;
        let maxAbs = 0;
        for (let ch = 0; ch < 3; ch++) {
            const B = colors[0][ch];
            // L_n − B q^n = c (1 − q^n): regression through the origin on w_n = 1 − q^n
            let sw = 0;
            let swy = 0;
            for (let i = 0; i < colors.length; i++) {
                const w = 1 - q ** ns[i];
                sw += w * w;
                swy += w * (colors[i][ch] - B * q ** ns[i]);
            }
            c.push(swy / sw);
            for (let i = 0; i < colors.length; i++) {
                const r = colors[i][ch] - (c[ch] + (B - c[ch]) * q ** ns[i]);
                sse += r * r;
                maxAbs = Math.max(maxAbs, Math.abs(r));
            }
        }
        return { q, c, sse, maxAbs };
    };
    if (colors.length < 3) {
        // B and one level: α and c are not separately determined; report the level as c at α = 1
        return { alpha: 1, c: [...colors[colors.length - 1]], rms: 0, maxAbs: 0, underdetermined: true };
    }
    let lo = 0.02;
    let hi = 0.98;
    const g = (Math.sqrt(5) - 1) / 2;
    let a = hi - g * (hi - lo);
    let b = lo + g * (hi - lo);
    let fa = evaluate(a);
    let fb = evaluate(b);
    for (let i = 0; i < 80; i++) {
        if (fa.sse < fb.sse) {
            hi = b;
            b = a;
            fb = fa;
            a = hi - g * (hi - lo);
            fa = evaluate(a);
        }
        else {
            lo = a;
            a = b;
            fa = fb;
            b = lo + g * (hi - lo);
            fb = evaluate(b);
        }
    }
    const best = fa.sse < fb.sse ? fa : fb;
    return { alpha: 1 - best.q, c: best.c, rms: Math.sqrt(best.sse / (3 * colors.length)), maxAbs: best.maxAbs };
}

/**
 * Level colors for n = 0 … K + EXTRA_LEVELS: B, the measured levels, then geometric continuation from the deepest
 * measured level with the fitted α and c (none when α is not measured).
 *
 * @param {number[]} B
 * @param {Level[]} levels
 * @param {{alpha: number, c: number[]}} fit
 * @returns {number[][]}
 */
export function levelTable(B, levels, fit) {
    const table = [B, ...levels.map(L => L.color)];
    // one level: α isn't measured, so there is nothing to extrapolate with (copies of the level would make every
    // plateau pixel read as the deepest count)
    if (fit.underdetermined)
        return table;
    const last = table[table.length - 1];
    const q = 1 - fit.alpha;
    for (let k = 1; k <= EXTRA_LEVELS; k++)
        table.push(last.map((v, ch) => fit.c[ch] + (v - fit.c[ch]) * q ** k));
    return table;
}

/**
 * Fractional overlap count per pixel: t interpolated linearly between the table levels' positions along the line.
 * Linear in t means linear in coverage within a level (an edge half over one more dot reads n + ½). Pixels off the
 * line get NaN; pixels at or inside B get 0.
 *
 * @param {{t: Float32Array, perp: Float32Array}} proj
 * @param {ColorLine} line
 * @param {number[][]} table
 * @param {Tolerances} [tol]
 * @returns {Float32Array}
 */
export function countImage(proj, line, table, tol = EXACT_TOLERANCES) {
    const { B, u } = line;
    const ts = table.map(c => (B[0] - c[0]) * u[0] + (B[1] - c[1]) * u[1] + (B[2] - c[2]) * u[2]);
    const K = ts.length - 1;
    const { t, perp } = proj;
    const D = new Float32Array(t.length);
    for (let p = 0; p < t.length; p++) {
        const tp = t[p];
        if (perp[p] > COUNT_PERP_TOL || (perp[p] > tol.nearB && perp[p] > COUNT_PERP_REL_TOL * tp)) {
            D[p] = NaN;
            continue;
        }
        if (tp <= 0)
            continue;
        if (tp >= ts[K]) {
            D[p] = K;
            continue;
        }
        let k = 0;
        while (tp > ts[k + 1])
            k++;
        D[p] = k + (tp - ts[k]) / (ts[k + 1] - ts[k]);
    }
    return D;
}

/**
 * Family candidates: directions from B, in 2° bins, of flat pixels at least DIRECTION_MIN_T from B. For each of
 * the most popular directions, the most common color in its bin seeds a color line whose levels are measured.
 * Translucent families (two or more levels) come first, by pixel count.
 *
 * @param {import('./image.js').RgbaImage} image
 * @param {number[]} B
 * @param {Uint8Array} flat
 * @param {Tolerances} [tol]
 * @returns {object[]}  candidates: { seed, line, levels, beyond, fit, pixels, translucent }
 */
export function familyCandidates(image, B, flat, tol = EXACT_TOLERANCES) {
    const n = image.width * image.height;
    const nTheta = Math.ceil(360 / DIRECTION_BIN);
    const nPhi = Math.ceil(180 / DIRECTION_BIN);
    const bins = new Map(); // bin → { count, colors: Map }
    for (let p = 0; p < n; p++) {
        if (!flat[p])
            continue;
        const e0 = B[0] - image.data[4 * p];
        const e1 = B[1] - image.data[4 * p + 1];
        const e2 = B[2] - image.data[4 * p + 2];
        const len = Math.hypot(e0, e1, e2);
        if (len < DIRECTION_MIN_T)
            continue;
        const theta = (Math.atan2(e1, e0) * 180 / Math.PI + 360) % 360;
        const phi = Math.acos(Math.max(-1, Math.min(1, e2 / len))) * 180 / Math.PI;
        const bin = Math.floor(phi / DIRECTION_BIN) * nTheta + Math.floor(theta / DIRECTION_BIN);
        let b = bins.get(bin);
        if (!b) {
            b = { count: 0, colors: new Map() };
            bins.set(bin, b);
        }
        b.count++;
        const key = (image.data[4 * p] << 16) | (image.data[4 * p + 1] << 8) | image.data[4 * p + 2];
        b.colors.set(key, (b.colors.get(key) ?? 0) + 1);
    }
    // local maxima over the 8 neighboring bins (theta wraps)
    const peaks = [];
    for (const [bin, b] of bins) {
        const pi = Math.floor(bin / nTheta);
        const ti = bin % nTheta;
        let isMax = true;
        for (let dp = -1; dp <= 1 && isMax; dp++) {
            for (let dt = -1; dt <= 1 && isMax; dt++) {
                if (dp === 0 && dt === 0)
                    continue;
                const q = bins.get((pi + dp) * nTheta + (ti + dt + nTheta) % nTheta);
                if (q && q.count > b.count)
                    isMax = false;
            }
        }
        if (isMax)
            peaks.push(b);
    }
    peaks.sort((a, b) => b.count - a.count);
    const candidates = [];
    for (const peak of peaks.slice(0, MAX_CANDIDATES)) {
        let seedKey = 0;
        let best = 0;
        for (const [k, c] of peak.colors) {
            if (c > best) {
                best = c;
                seedKey = k;
            }
        }
        const seed = [seedKey >> 16, (seedKey >> 8) & 255, seedKey & 255];
        candidates.push(measureFamily(image, B, seed, flat, tol));
    }
    // two peaks can describe the same family (levels at slightly different hues): keep the first of each
    const distinct = [];
    for (const c of candidates) {
        if (!c.levels.length)
            continue;
        const dup = distinct.some(d => d.line.u.reduce((s, v, i) => s + v * c.line.u[i], 0) > Math.cos(4 * DIRECTION_BIN * Math.PI / 180));
        if (!dup)
            distinct.push(c);
    }
    distinct.sort((a, b) => (b.translucent - a.translucent) || (b.pixels - a.pixels));
    return distinct;
}

/**
 * Levels and the geometric fit along the line from B toward a seed color.
 *
 * @param {import('./image.js').RgbaImage} image
 * @param {number[]} B
 * @param {number[]} seed
 * @param {Uint8Array} flat
 * @param {Tolerances} [tol]
 */
export function measureFamily(image, B, seed, flat, tol = EXACT_TOLERANCES) {
    const line = colorLine(B, seed);
    const proj = projectOnLine(image, line);
    const found = findLevels(image, proj, flat, tol);
    let { levels } = found;
    let fit = levels.length ? fitGeometric([B, ...levels.map(L => L.color)]) : null;
    let plausible = isPlausible(fit);
    if (levels.length >= 2 && !plausible) {
        const gapped = withMissingLevels(B, line, levels);
        if (gapped) {
            ({ levels, fit } = gapped);
            plausible = true;
        }
    }
    return {
        seed, line, proj, levels, beyond: found.beyond, fit,
        pixels: levels.reduce((s, L) => s + L.count, 0),
        translucent: levels.length >= 2 && plausible,
    };
}

function isPlausible(fit) {
    return fit !== null && fit.rms <= FIT_MAX_RMS && fit.c.every(v => v >= -FIT_C_TOL && v <= 255 + FIT_C_TOL);
}

/**
 * Levels whose geometric fit fails as consecutive counts may skip one: in a JPEG, the lens-shaped two-dot overlaps
 * are too narrow for many flat pixels to survive the block noise (bench2 group-smoother.jpg's red: 4 flat pixels at
 * level 2 against 14 at level 3), so the next level was read as level 2 and the fit put c far outside the cube.
 * Counts after the first level may skip up to MAX_MISSING_LEVELS; the fewest skips with a plausible fit win, then
 * the lowest rms. The missing levels get the fitted colors, with count 0 and `inferred` set.
 *
 * @returns {{levels: Level[], fit: object}|null}
 */
function withMissingLevels(B, line, levels) {
    const colors = [B, ...levels.map(L => L.color)];
    let best = null;
    for (let missing = 1; missing <= MAX_MISSING_LEVELS && !best; missing++) {
        for (const ns of countAssignments(levels.length, missing)) {
            const fit = fitGeometric(colors, [0, ...ns]);
            if (isPlausible(fit) && (!best || fit.rms < best.fit.rms))
                best = { ns, fit };
        }
    }
    if (!best)
        return null;
    const { ns, fit } = best;
    const q = 1 - fit.alpha;
    const out = [];
    for (let n = 1; n <= ns[ns.length - 1]; n++) {
        const i = ns.indexOf(n);
        if (i >= 0) {
            out.push(levels[i]);
            continue;
        }
        const color = B.map((v, ch) => fit.c[ch] + (v - fit.c[ch]) * q ** n);
        const t = (line.B[0] - color[0]) * line.u[0] + (line.B[1] - color[1]) * line.u[1] + (line.B[2] - color[2]) * line.u[2];
        out.push({ t, color, count: 0, inferred: true });
    }
    return { levels: out, fit };
}

// increasing counts for k levels, the first at 1, with exactly `missing` counts skipped below the last
function countAssignments(k, missing) {
    const out = [];
    const extend = (ns, left) => {
        if (ns.length === k) {
            if (left === 0)
                out.push(ns);
            return;
        }
        for (let skip = 0; skip <= left; skip++)
            extend([...ns, ns[ns.length - 1] + 1 + skip], left - skip);
    };
    extend([1], missing);
    return out;
}

/** Connected components (4-neighbor) of pixels with D > 0.5. */
function components(D, width, height) {
    const label = new Int32Array(D.length).fill(-1);
    const comps = [];
    const stack = [];
    for (let s = 0; s < D.length; s++) {
        if (!(D[s] > 0.5) || label[s] >= 0)
            continue;
        const comp = [];
        label[s] = comps.length;
        stack.push(s);
        while (stack.length) {
            const p = stack.pop();
            comp.push(p);
            const x = p % width;
            const y = (p - x) / width;
            if (x > 0 && D[p - 1] > 0.5 && label[p - 1] < 0) {
                label[p - 1] = comps.length;
                stack.push(p - 1);
            }
            if (x < width - 1 && D[p + 1] > 0.5 && label[p + 1] < 0) {
                label[p + 1] = comps.length;
                stack.push(p + 1);
            }
            if (y > 0 && D[p - width] > 0.5 && label[p - width] < 0) {
                label[p - width] = comps.length;
                stack.push(p - width);
            }
            if (y < height - 1 && D[p + width] > 0.5 && label[p + width] < 0) {
                label[p + width] = comps.length;
                stack.push(p + width);
            }
        }
        comps.push(comp);
    }
    return comps;
}

/**
 * Isolated whole dots: components that never exceed ISOLATED_MAX_LEVEL, away from the image border, with no
 * off-line pixel nearby (a dot touching a line is partly hidden), and disc-shaped by their second moments.
 * Centroid and area weight pixels by min(D, 1), so the soft rim counts fractionally.
 *
 * @param {Float32Array} D
 * @param {number} width
 * @param {number} height
 * @param {Tolerances} [tol]
 * @returns {{cx: number, cy: number, r: number, aspect: number, box: number[], ringed: boolean}[]}  ringed: deeper
 *   than one dot only in its outer part (an outline)
 */
export function isolatedDots(D, width, height, tol = EXACT_TOLERANCES) {
    const dots = [];
    const m = EDGE_RING;
    for (const comp of components(D, width, height)) {
        if (comp.length < ISOLATED_MIN_AREA)
            continue;
        let maxD = 0;
        let x0 = width;
        let x1 = 0;
        let y0 = height;
        let y1 = 0;
        for (const p of comp) {
            maxD = Math.max(maxD, D[p]);
            const x = p % width;
            const y = (p - x) / width;
            x0 = Math.min(x0, x);
            x1 = Math.max(x1, x);
            y0 = Math.min(y0, y);
            y1 = Math.max(y1, y);
        }
        if (x0 - m < 0 || y0 - m < 0 || x1 + m >= width || y1 + m >= height)
            continue;
        let a = 0;
        let sx = 0;
        let sy = 0;
        let offLine = false;
        for (let y = y0 - m; y <= y1 + m; y++) {
            for (let x = x0 - m; x <= x1 + m; x++) {
                const d = D[y * width + x];
                if (Number.isNaN(d)) {
                    offLine = true;
                    continue;
                }
                const w = d < tol.isolatedFloor ? 0 : Math.min(1, d);
                a += w;
                sx += w * (x + 0.5);
                sy += w * (y + 0.5);
            }
        }
        if (offLine)
            continue;
        const cx = sx / a;
        const cy = sy / a;
        let mxx = 0;
        let myy = 0;
        let mxy = 0;
        for (let y = y0 - m; y <= y1 + m; y++) {
            for (let x = x0 - m; x <= x1 + m; x++) {
                const d = D[y * width + x];
                const w = d < tol.isolatedFloor ? 0 : Math.min(1, d);
                const dx = x + 0.5 - cx;
                const dy = y + 0.5 - cy;
                mxx += w * dx * dx;
                myy += w * dy * dy;
                mxy += w * dx * dy;
            }
        }
        mxx /= a;
        myy /= a;
        mxy /= a;
        const tr = mxx + myy;
        const disc = Math.sqrt(Math.max(0, tr * tr / 4 - (mxx * myy - mxy * mxy)));
        const aspect = Math.sqrt((tr / 2 + disc) / (tr / 2 - disc));
        const r = Math.sqrt(a / Math.PI);
        if (aspect > ISOLATED_MAX_ASPECT || Math.sqrt(2 * tr) / r > ISOLATED_MAX_MOMENT_RATIO)
            continue;
        let ringed = false;
        if (maxD > ISOLATED_MAX_LEVEL) {
            // deeper than one dot: an outline if all of that is in the outer part
            ringed = comp.every(p => {
                if (!(D[p] > ISOLATED_MAX_LEVEL))
                    return true;
                const x = p % width;
                return Math.hypot(x + 0.5 - cx, (p - x) / width + 0.5 - cy) >= RING_MIN_RADIUS_FRACTION * r;
            });
            if (!ringed)
                continue;
        }
        dots.push({ cx, cy, r, aspect, box: [x0, y0, x1, y1], ringed });
    }
    return dots;
}

/**
 * An outline's inner radius (× radius) and extra opacity, fitted to isolated dots' pixels along the color line: a
 * dot is tC·(α·m + a·m_ring), with m its coverage and m_ring the annulus's (render.js style.ring). For each inner
 * radius the opacity a is linear least squares; blur is left out (outlined renderings so far are crisp).
 *
 * @param {Float32Array} t  0–255 units along the line, per pixel
 * @param {number} width
 * @param {{cx: number, cy: number}[]} dots  isolated dots
 * @param {number} radius  px
 * @param {number} alpha  the fill's
 * @param {number} tC  0–255 units: t of the full color
 * @returns {{inner: number, alpha: number}|null}  null when there is no ring
 */
export function measureRing(t, width, dots, radius, alpha, tC) {
    const pixels = [];
    for (const d of dots) {
        for (let y = Math.floor(d.cy - radius - 1); y <= d.cy + radius + 1; y++) {
            for (let x = Math.floor(d.cx - radius - 1); x <= d.cx + radius + 1; x++) {
                const m = pixelCoverage(d.cx, d.cy, radius, x, y);
                pixels.push({ d, x, y, m, t: t[y * width + x] });
            }
        }
    }
    let best = null;
    for (let inner = RING_INNER_RANGE[0]; inner <= RING_INNER_RANGE[1] + 1e-9; inner += RING_INNER_STEP) {
        let num = 0;
        let den = 0;
        const rings = pixels.map(q => (q.m > 0 ? q.m - pixelCoverage(q.d.cx, q.d.cy, inner * radius, q.x, q.y) : 0));
        pixels.forEach((q, i) => {
            const g = tC * rings[i];
            num += (q.t - tC * alpha * q.m) * g;
            den += g * g;
        });
        const a = den > 0 ? Math.min(1 - alpha, Math.max(0, num / den)) : 0;
        let sse = 0;
        pixels.forEach((q, i) => {
            sse += (q.t - tC * (alpha * q.m + a * rings[i])) ** 2;
        });
        if (!best || sse < best.sse)
            best = { inner, alpha: a, sse };
    }
    return best && best.alpha >= RING_MIN_ALPHA ? { inner: best.inner, alpha: best.alpha } : null;
}

function erfc(x) {
    // Abramowitz & Stegun 7.1.26, |error| < 1.5e-7
    const z = Math.abs(x);
    const t = 1 / (1 + 0.3275911 * z);
    const y = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429)))) * Math.exp(-z * z);
    return x >= 0 ? y : 2 - y;
}

/**
 * Edge blur from the isolated dots: D against distance from each dot's centroid, relative to its area radius,
 * pooled and binned, then fitted with ½ erfc((ρ − ρ₀) / (σ√2)), a disc edge blurred by a Gaussian of sd σ. The
 * fitted σ includes the pixel box (variance 1/12).
 *
 * @param {Float32Array} D
 * @param {number} width
 * @param {object[]} dots  from isolatedDots
 * @returns {{sigma: number, offset: number, profile: {rel: number, d: number}[]}}
 */
export function edgeBlur(D, width, dots) {
    const nb = Math.ceil(2 * EDGE_RING / PROFILE_BIN) + 1;
    const sum = new Float64Array(nb);
    const cnt = new Float64Array(nb);
    for (const dot of dots) {
        const [x0, y0, x1, y1] = dot.box;
        for (let y = y0 - EDGE_RING; y <= y1 + EDGE_RING; y++) {
            for (let x = x0 - EDGE_RING; x <= x1 + EDGE_RING; x++) {
                const rel = Math.hypot(x + 0.5 - dot.cx, y + 0.5 - dot.cy) - dot.r;
                const b = Math.round((rel + EDGE_RING) / PROFILE_BIN);
                if (b < 0 || b >= nb)
                    continue;
                sum[b] += Math.min(1, D[y * width + x]);
                cnt[b]++;
            }
        }
    }
    const profile = [];
    for (let b = 0; b < nb; b++) {
        if (cnt[b] > 0)
            profile.push({ rel: b * PROFILE_BIN - EDGE_RING, d: sum[b] / cnt[b] });
    }
    let best = null;
    for (let off = -1.5; off <= 1.5; off += 0.01) {
        for (let s = 0.2; s <= 3; s += 0.01) {
            let sse = 0;
            for (const { rel, d } of profile)
                sse += (d - 0.5 * erfc((rel - off) / (s * Math.SQRT2))) ** 2;
            if (best === null || sse < best.sse)
                best = { offset: off, sigma: s, sse };
        }
    }
    return { sigma: best.sigma, offset: best.offset, profile };
}

function quantile(a, q) {
    const s = [...a].sort((x, y) => x - y);
    return s[Math.min(s.length - 1, Math.max(0, Math.round(q * (s.length - 1))))];
}

/**
 * Color clusters: peaks of a 3-D color histogram (CLUSTER_BIN bins, smoothed over 3×3×3 neighbors) of the ink
 * pixels, each with the mean color of the pixels in its neighborhood of bins.
 *
 * @param {import('./image.js').RgbaImage} image
 * @param {number[][]} backgrounds
 * @returns {{color: number[], count: number}[]}  largest first
 */
export function colorClusters(image, backgrounds) {
    const nb = 256 / CLUSTER_BIN;
    const n = image.width * image.height;
    const count = new Float64Array(nb ** 3);
    const sum = new Float64Array(3 * nb ** 3);
    const d = image.data;
    for (let p = 0; p < n; p++) {
        const r = d[4 * p];
        const g = d[4 * p + 1];
        const b = d[4 * p + 2];
        if (backgrounds.some(B => Math.hypot(r - B[0], g - B[1], b - B[2]) <= INK_TOL))
            continue;
        const k = ((r / CLUSTER_BIN | 0) * nb + (g / CLUSTER_BIN | 0)) * nb + (b / CLUSTER_BIN | 0);
        count[k]++;
        sum[3 * k] += r;
        sum[3 * k + 1] += g;
        sum[3 * k + 2] += b;
    }
    const at = (i, j, k) => (i < 0 || j < 0 || k < 0 || i >= nb || j >= nb || k >= nb ? -1 : (i * nb + j) * nb + k);
    const smooth = new Float64Array(nb ** 3);
    for (let i = 0; i < nb; i++) {
        for (let j = 0; j < nb; j++) {
            for (let k = 0; k < nb; k++) {
                let t = 0;
                for (let a = -1; a <= 1; a++) {
                    for (let b = -1; b <= 1; b++) {
                        for (let c = -1; c <= 1; c++) {
                            const q = at(i + a, j + b, k + c);
                            if (q >= 0)
                                t += count[q];
                        }
                    }
                }
                smooth[at(i, j, k)] = t;
            }
        }
    }
    const min = Math.max(CLUSTER_MIN_PIXELS, CLUSTER_MIN_FRACTION * n);
    const out = [];
    for (let i = 0; i < nb; i++) {
        for (let j = 0; j < nb; j++) {
            for (let k = 0; k < nb; k++) {
                const v = smooth[at(i, j, k)];
                if (v < min)
                    continue;
                let peak = true;
                const rgb = [0, 0, 0];
                for (let a = -1; a <= 1 && peak; a++) {
                    for (let b = -1; b <= 1 && peak; b++) {
                        for (let c = -1; c <= 1; c++) {
                            const q = at(i + a, j + b, k + c);
                            if (q < 0)
                                continue;
                            if (smooth[q] > v || (smooth[q] === v && q > at(i, j, k))) {
                                peak = false;
                                break;
                            }
                            rgb[0] += sum[3 * q];
                            rgb[1] += sum[3 * q + 1];
                            rgb[2] += sum[3 * q + 2];
                        }
                    }
                }
                if (peak)
                    out.push({ color: rgb.map(x => x / v), count: v });
            }
        }
    }
    return out.sort((a, b) => b.count - a.count);
}

/** The most common color in a ring RING px wide around a mask, and how many ring pixels have it. */
function ringBackground(image, mask) {
    const { width, height, data } = image;
    const grown = dilate(mask, width, height, RING);
    const counts = new Map();
    for (let p = 0; p < mask.length; p++) {
        if (!grown[p] || mask[p])
            continue;
        const key = (data[4 * p] << 16) | (data[4 * p + 1] << 8) | data[4 * p + 2];
        counts.set(key, (counts.get(key) ?? 0) + 1);
    }
    let best = null;
    for (const [k, c] of counts) {
        if (!best || c > best.c)
            best = { k, c };
    }
    return best ? [best.k >> 16, (best.k >> 8) & 255, best.k & 255] : null;
}

/**
 * Radius from the distance transform of a family's pixels: its local maxima sit at dot centers, at about r where a
 * dot is whole and less where neighbors overlap it; the upper quartile, plus the half pixel lost between pixel
 * centers.
 */
function radiusFromDistance(D, width, height) {
    const inside = new Uint8Array(D.length);
    for (let p = 0; p < D.length; p++)
        inside[p] = D[p] > 0.5 ? 0 : 1; // distance to the nearest pixel outside the family
    const d2 = distanceTransform(inside, width, height);
    const peaks = [];
    for (let y = 1; y < height - 1; y++) {
        for (let x = 1; x < width - 1; x++) {
            const p = y * width + x;
            const v = d2[p];
            if (v < DOT_MIN_RADIUS * DOT_MIN_RADIUS || v === Infinity)
                continue;
            let isMax = true;
            for (let dy = -1; dy <= 1 && isMax; dy++) {
                for (let dx = -1; dx <= 1; dx++) {
                    if ((dx || dy) && d2[p + dy * width + dx] > v) {
                        isMax = false;
                        break;
                    }
                }
            }
            if (isMax)
                peaks.push(Math.sqrt(v));
        }
    }
    return peaks.length ? quantile(peaks, DT_RADIUS_QUANTILE) + DT_RADIUS_OFFSET : null;
}

/**
 * Edge blur from a family's boundaries, for families without isolated dots: D against the signed distance to the
 * D = ½ boundary, pooled within BOUNDARY_PROFILE_REACH px, fitted with the same erf as edgeBlur.
 */
function boundaryBlur(D, width, height) {
    const inMask = new Uint8Array(D.length);
    const outMask = new Uint8Array(D.length);
    for (let p = 0; p < D.length; p++) {
        inMask[p] = D[p] > 0.5 ? 1 : 0;
        outMask[p] = D[p] > 0.5 ? 0 : 1;
    }
    const toOut = distanceTransform(outMask, width, height);
    const toIn = distanceTransform(inMask, width, height);
    const nb = Math.ceil(2 * BOUNDARY_PROFILE_REACH / PROFILE_BIN) + 1;
    const sum = new Float64Array(nb);
    const cnt = new Float64Array(nb);
    for (let p = 0; p < D.length; p++) {
        if (Number.isNaN(D[p]))
            continue;
        // signed distance, positive inside; pixel-center distances, so the boundary sits half a pixel in
        const sd = inMask[p] ? Math.sqrt(toOut[p]) - 0.5 : 0.5 - Math.sqrt(toIn[p]);
        if (Math.abs(sd) > BOUNDARY_PROFILE_REACH)
            continue;
        const b = Math.round((BOUNDARY_PROFILE_REACH - sd) / PROFILE_BIN);
        sum[b] += Math.min(1, D[p]);
        cnt[b]++;
    }
    const profile = [];
    for (let b = 0; b < nb; b++) {
        if (cnt[b] > 0)
            profile.push({ rel: b * PROFILE_BIN - BOUNDARY_PROFILE_REACH, d: sum[b] / cnt[b] });
    }
    let best = null;
    for (let off = -1.5; off <= 1.5; off += 0.01) {
        for (let s = 0.2; s <= 3; s += 0.01) {
            let sse = 0;
            for (const { rel, d } of profile)
                sse += (d - 0.5 * erfc((rel - off) / (s * Math.SQRT2))) ** 2;
            if (best === null || sse < best.sse)
                best = { offset: off, sigma: s, sse };
        }
    }
    return { sigma: best.sigma, offset: best.offset, profile };
}

/**
 * The background under and around a family, per pixel: the color of the nearest pixel that is not close to the
 * family's color. For an edge pixel that is the pixel just outside the dot, which is what the dot's edge was
 * blended with (a window mean mixed in other marks: the synthetic swarms' median lines).
 */
function localBackground(image, near, colors, others) {
    const { width, height, data } = image;
    const n = width * height;
    // Only background colors and other families' colors can be a dot's background. Other ink can't: a median line
    // across a cantons dot took the line beyond the dot as its "background" and read as no dot there.
    const known = new Uint8Array(n);
    for (let p = 0; p < n; p++) {
        if (near[p])
            continue;
        const r = data[4 * p];
        const g = data[4 * p + 1];
        const b = data[4 * p + 2];
        known[p] = colors.some(P => Math.hypot(r - P[0], g - P[1], b - P[2]) <= INK_TOL)
            || others.some(O => Math.hypot(r - O[0], g - O[1], b - O[2]) <= CLUSTER_TOL) ? 1 : 0;
    }
    const nearest = nearestTransform(known, width, height);
    const bg = new Float32Array(3 * n);
    for (let p = 0; p < n; p++) {
        const q = nearest[p] >= 0 ? nearest[p] : p;
        bg[3 * p] = data[4 * q];
        bg[3 * p + 1] = data[4 * q + 1];
        bg[3 * p + 2] = data[4 * q + 2];
    }
    return bg;
}

// 1 where a pixel is another family's ring: within `reach` of that family's fill and nearer it (by STROKE_TIE) than to
// this family's fill (`own`, its thick parts); null without rings. O(families · pixels).
function ringOwners(image, own, rings) {
    if (!rings?.length)
        return null;
    const { width, height } = image;
    const n = width * height;
    const dist = idx => {
        const out = new Float32Array(n);
        for (let p = 0; p < n; p++) {
            const q = idx[p];
            out[p] = q < 0 ? Infinity : Math.hypot(p % width - q % width, Math.floor(p / width) - Math.floor(q / width));
        }
        return out;
    };
    const dOwn = dist(nearestTransform(own, width, height));
    const out = new Uint8Array(n);
    for (const { fill, reach } of rings) {
        const theirs = thickMembers(image, [fill]);
        for (let p = 0; p < n; p++)
            theirs[p] = own[p] ? 0 : theirs[p];
        const d = dist(nearestTransform(theirs, width, height));
        for (let p = 0; p < n; p++) {
            if (!own[p] && !theirs[p] && d[p] <= reach && d[p] < dOwn[p] - STROKE_TIE)
                out[p] = 1;
        }
    }
    return out;
}

/**
 * One outlined or stroked family's ring image, for fitting it by its rings in drawing order (outline.js): each pixel's
 * weight of the outline in the triangle page–fill–outline, scaled so a full outline reads FAMILY_RING_P. The fill and
 * the page read 0, so an outline crossing a lower dot of the family is the upper dot's edge, which the opaque union
 * can't see (degradation1: two mint dots 6–10 px apart read as one solid blob, fill and outline alike "dot").
 * Unknown: pixels off the triangle, other families' fills and the rings nearer them, and outline-colored ink beyond
 * OUTLINE_REACH of this family's thick fill (text, axes).
 *
 * @param {import('./image.js').RgbaImage} image
 * @param {number[]} fill  0–255
 * @param {number[]} outline  0–255
 * @param {number[][]} backgrounds  0–255, the page's first
 * @param {number[][]} others  0–255, the other families' fill colors
 * @returns {{P: Float32Array, unknown: Uint8Array}}  P 0–FAMILY_RING_P
 */
export function familyRings(image, fill, outline, backgrounds, others) {
    const { width, height, data } = image;
    const n = width * height;
    const own = new Uint8Array(n);
    for (let p = 0; p < n; p++)
        own[p] = Math.hypot(data[4 * p] - fill[0], data[4 * p + 1] - fill[1], data[4 * p + 2] - fill[2]) <= CLUSTER_TOL ? 1 : 0;
    const reach = dilate(opening(own, width, height, THICK_OPENING), width, height, OUTLINE_REACH);
    const theirs = ringOwners(image, own, others.map(c => ({ fill: c, reach: STROKE_RING })));
    const otherFill = others.length ? thickMembers(image, others) : null;
    const B = backgrounds[0];
    const a1 = fill.map((v, ch) => v - B[ch]);
    const a2 = outline.map((v, ch) => v - B[ch]);
    const m11 = a1[0] * a1[0] + a1[1] * a1[1] + a1[2] * a1[2];
    const m12 = a1[0] * a2[0] + a1[1] * a2[1] + a1[2] * a2[2];
    const m22 = a2[0] * a2[0] + a2[1] * a2[1] + a2[2] * a2[2];
    const det = m11 * m22 - m12 * m12;
    const P = new Float32Array(n);
    const unknown = new Uint8Array(n);
    for (let p = 0; p < n; p++) {
        const r = data[4 * p];
        const g = data[4 * p + 1];
        const b = data[4 * p + 2];
        if (backgrounds.some(Q => Math.hypot(r - Q[0], g - Q[1], b - Q[2]) <= INK_TOL))
            continue;
        if ((otherFill && !own[p] && otherFill[p]) || (theirs && theirs[p]) || !reach[p]) {
            unknown[p] = 1;
            continue;
        }
        const v = [r - B[0], g - B[1], b - B[2]];
        let wf;
        let ws;
        if (det > COLLINEAR_SIN2 * m11 * m22) {
            const r1 = v[0] * a1[0] + v[1] * a1[1] + v[2] * a1[2];
            const r2 = v[0] * a2[0] + v[1] * a2[1] + v[2] * a2[2];
            wf = (m22 * r1 - m12 * r2) / det;
            ws = (m11 * r2 - m12 * r1) / det;
        }
        else {
            // page, fill and outline on one line (gray fills in black outlines): the position on the outline–fill
            // segment, or beyond the fill on the page–outline one (an outline's rim on the page passes through the
            // fill's color too, and reads as fill: unavoidable on one line)
            const e = outline.map((c, ch) => [r, g, b][ch] - c);
            const fo = fill.map((c, ch) => c - outline[ch]);
            const k = (e[0] * fo[0] + e[1] * fo[1] + e[2] * fo[2]) / (fo[0] * fo[0] + fo[1] * fo[1] + fo[2] * fo[2]);
            if (k <= 1) {
                wf = Math.max(0, k);
                ws = 1 - Math.max(0, k);
            }
            else {
                wf = 0;
                ws = (v[0] * a2[0] + v[1] * a2[1] + v[2] * a2[2]) / m22;
            }
        }
        const off = Math.hypot(v[0] - wf * a1[0] - ws * a2[0], v[1] - wf * a1[1] - ws * a2[1], v[2] - wf * a1[2] - ws * a2[2]);
        const span = Math.hypot(wf * a1[0] + ws * a2[0], wf * a1[1] + ws * a2[1], wf * a1[2] + ws * a2[2]);
        if (off > Math.max(OFF_MIX_ABS, OFF_MIX_REL * span) || wf < -STROKE_SLACK || ws < -STROKE_SLACK || wf + ws > MAX_MIX) {
            unknown[p] = 1;
            continue;
        }
        P[p] = FAMILY_RING_P * Math.min(1, Math.max(0, ws));
    }
    return { P, unknown };
}

// 1 on the thick parts (an opening) of the pixels within CLUSTER_TOL of any of `colors`
function thickMembers(image, colors) {
    const { width, height, data } = image;
    const m = new Uint8Array(width * height);
    for (let p = 0; p < m.length; p++)
        m[p] = colors.some(c => Math.hypot(data[4 * p] - c[0], data[4 * p + 1] - c[1], data[4 * p + 2] - c[2]) <= CLUSTER_TOL) ? 1 : 0;
    return opening(m, width, height, THICK_OPENING);
}

/**
 * The stroke around a fill color, read from the rings of its isolated components, or null when they show none.
 *
 * @param {import('./image.js').RgbaImage} image
 * @param {number[]} color  the fill, 0–255
 * @param {number[]} B  the page background, 0–255
 * @returns {{color: number[], off: number, rings: number}|null}  color 0–255; off: its distance from the
 *   background–fill line, 0–255 units; rings: isolated fills read
 */
export function strokeColor(image, color, B) {
    const { width, height, data } = image;
    const n = width * height;
    const own = new Float32Array(n);
    for (let p = 0; p < n; p++)
        own[p] = Math.hypot(data[4 * p] - color[0], data[4 * p + 1] - color[1], data[4 * p + 2] - color[2]) <= CLUSTER_TOL ? 1 : 0;
    const comps = components(own, width, height);
    const label = new Int32Array(n).fill(-1);
    comps.forEach((c, k) => {
        for (const p of c)
            label[p] = k;
    });
    const nearest = nearestTransform(Uint8Array.from(own), width, height);
    const clear = new Float64Array(comps.length);
    const clearBg = new Float64Array(comps.length);
    const ring = comps.map(() => []);
    for (let p = 0; p < n; p++) {
        const q = nearest[p];
        if (own[p] || q < 0)
            continue;
        const d = Math.hypot(p % width - q % width, Math.floor(p / width) - Math.floor(q / width));
        const k = label[q];
        if (d <= STROKE_RING) {
            ring[k].push(p);
        }
        else if (d <= STROKE_CLEAR) {
            clear[k]++;
            if (Math.hypot(data[4 * p] - B[0], data[4 * p + 1] - B[1], data[4 * p + 2] - B[2]) <= INK_TOL)
                clearBg[k]++;
        }
    }
    const u = color.map((v, ch) => v - B[ch]);
    const uu = u[0] * u[0] + u[1] * u[1] + u[2] * u[2];
    const pixels = [];
    let rings = 0;
    comps.forEach((c, k) => {
        if (c.length < STROKE_COMPONENT[0] || c.length > STROKE_COMPONENT[1] || !clear[k] || clearBg[k] < STROKE_CLEAR_SHARE * clear[k])
            return;
        rings++;
        for (const p of ring[k]) {
            const v = [0, 1, 2].map(ch => data[4 * p + ch] - B[ch]);
            if (Math.hypot(...v) <= INK_TOL)
                continue;
            const m = (v[0] * u[0] + v[1] * u[1] + v[2] * u[2]) / uu;
            pixels.push({ p, off: Math.hypot(v[0] - m * u[0], v[1] - m * u[1], v[2] - m * u[2]) });
        }
    });
    if (rings < STROKE_MIN_RINGS || !pixels.length)
        return null;
    pixels.sort((a, b) => a.off - b.off);
    const top = pixels.slice(Math.floor(STROKE_TOP * (pixels.length - 1)));
    const S = [0, 1, 2].map(ch => top.reduce((t, q) => t + data[4 * q.p + ch], 0) / top.length);
    const v = S.map((x, ch) => x - B[ch]);
    const m = (v[0] * u[0] + v[1] * u[1] + v[2] * u[2]) / uu;
    const off = Math.hypot(v[0] - m * u[0], v[1] - m * u[1], v[2] - m * u[2]);
    return off >= STROKE_MIN_OFF ? { color: S, off, rings } : null;
}

/**
 * An opaque family from a color: the coverage image D (0 its local background … 1 the color; NaN where a pixel is
 * neither, or contrast is too low to tell), its typical background (the most common color around it), the radius
 * (isolated dots, else the distance transform), and how many filled, dot-shaped edges a Hough vote finds.
 *
 * With an outline color, a dot is its fill inside an opaque ring of the outline (JMP's outlined markers: an outline
 * disc 1.7 px larger in radius under the fill): every pixel between the two colors is covered, and the outer rim is
 * read as the outline's coverage of the background. The boundary between fill and outline isn't used.
 *
 * With a stroke (strokeColor), a pixel inside the triangle background–fill–stroke is dot by the weight it takes off
 * the background, so the ring counts as dot; the fill's own weight decides whether a center is this family's (its
 * inner disc must be the fill: a darker family's fill can lie inside the triangle, near the stroke). With `rings`,
 * the other families' fills, a ring pixel nearer one of them than to this family's fill reads 0.
 *
 * @param {object} [opts]
 * @param {number[]|null} [opts.stroke]  0–255
 * @param {number} [opts.fillRadius]  px, the fill inside the stroke
 * @param {{fill: number[], reach: number}[]|null} [opts.rings]  other families' fills, and how far (px) their rings
 *   reach beyond them
 */
export function measureOpaque(image, color, backgrounds, others = [], tol = EXACT_TOLERANCES, outline = null, opts = {}) {
    const stroke = opts.stroke ?? null;
    const { width, height, data } = image;
    const n = width * height;
    const member = (p, c) => Math.hypot(data[4 * p] - c[0], data[4 * p + 1] - c[1], data[4 * p + 2] - c[2]) <= CLUSTER_TOL;
    const own = new Uint8Array(n);
    for (let p = 0; p < n; p++)
        own[p] = member(p, color) ? 1 : 0;
    // the reach is from the fill's thick parts: an antialiased edge between the outline's color and a lighter one
    // passes through the fill's color (phase1: every black edge, text's and other grays' outlines, had 82-gray
    // pixels, and the reach covered them all)
    const thickOwn = outline ? opening(own, width, height, THICK_OPENING) : null;
    const ownReach = outline ? dilate(thickOwn, width, height, OUTLINE_REACH) : null;
    const mask = new Uint8Array(n);
    for (let p = 0; p < n; p++)
        mask[p] = own[p] || (outline && ownReach[p] && member(p, outline)) ? 1 : 0;
    const thick = opening(mask, width, height, THICK_OPENING);
    // a stroked fill's surroundings are its ring: its background lies beyond
    const body = thick.some(v => v) ? thick : mask;
    const B = ringBackground(image, stroke ? dilate(body, width, height, STROKE_RING) : body) ?? backgrounds[0];
    const near = dilate(mask, width, height, NEAR_FAMILY + (stroke ? STROKE_RING : 0));
    // the outer rim mixes the background with the outline, if there is one
    const edge = outline ?? color;
    // the family's color noise: RMS distance from its color over interior pixels (all 8 neighbors in the family)
    let noiseSum = 0;
    let noiseN = 0;
    for (let y = 1; y < height - 1; y++) {
        for (let x = 1; x < width - 1; x++) {
            const p = y * width + x;
            if (!(thickOwn ?? thick)[p] || !own[p - 1] || !own[p + 1] || !own[p - width] || !own[p + width])
                continue;
            noiseSum += (data[4 * p] - color[0]) ** 2 + (data[4 * p + 1] - color[1]) ** 2 + (data[4 * p + 2] - color[2]) ** 2;
            noiseN++;
        }
    }
    const offAbs = Math.max(OFF_MIX_ABS, OFF_MIX_NOISE * (noiseN ? Math.sqrt(noiseSum / noiseN) : 0));
    const pageColorsForBg = [...backgrounds, B].filter(P => Math.hypot(...P.map((v, ch) => v - color[ch])) > CLUSTER_TOL
        && (!outline || Math.hypot(...P.map((v, ch) => v - outline[ch])) > CLUSTER_TOL)
        && (!stroke || Math.hypot(...P.map((v, ch) => v - stroke[ch])) > CLUSTER_TOL));
    const bg = localBackground(image, near, pageColorsForBg, others);
    // background colors, but not this family's own (a dot color can be common enough to count as a background)
    const pageColors = pageColorsForBg;
    const fo = outline ? color.map((v, ch) => v - outline[ch]) : null;
    const foo = fo ? fo[0] * fo[0] + fo[1] * fo[1] + fo[2] * fo[2] : 0;
    // from the family's thick parts: a stroke near its color (engagement1's navy) is thin, and joins its color mask
    const theirs = ringOwners(image, thick.some(v => v) ? thick : own, opts.rings ?? null);
    // other families' fills, thick parts only: a stroke near another family's color is still this dot's ring
    const otherFill = stroke ? thickMembers(image, others) : null;
    const D = new Float32Array(n);
    const fillW = stroke ? new Float32Array(n) : null;
    // with an outline, how far each pixel is from the outline toward the fill: the fill's antialiased coverage
    const fillK = outline ? new Float32Array(n) : null;
    for (let p = 0; p < n; p++) {
        const r = data[4 * p];
        const g = data[4 * p + 1];
        const b = data[4 * p + 2];
        if (!near[p] && pageColors.some(P => Math.hypot(r - P[0], g - P[1], b - P[2]) <= INK_TOL)) {
            D[p] = 0; // away from the family on a background color: no dot here
            continue;
        }
        if (theirs && theirs[p]) {
            D[p] = 0; // another family's dot's ring
            continue;
        }
        if (stroke) {
            // another family's fill is that dot, never this one's partial coverage
            if (!own[p] && otherFill[p]) {
                D[p] = NaN;
                continue;
            }
            const w = near[p] ? [bg[3 * p], bg[3 * p + 1], bg[3 * p + 2]] : B;
            const a1 = color.map((v, ch) => v - w[ch]);
            const a2 = stroke.map((v, ch) => v - w[ch]);
            const v = [r - w[0], g - w[1], b - w[2]];
            const m11 = a1[0] * a1[0] + a1[1] * a1[1] + a1[2] * a1[2];
            const m12 = a1[0] * a2[0] + a1[1] * a2[1] + a1[2] * a2[2];
            const m22 = a2[0] * a2[0] + a2[1] * a2[1] + a2[2] * a2[2];
            const det = m11 * m22 - m12 * m12;
            if (m11 < MIN_CONTRAST * MIN_CONTRAST || det <= 0) {
                D[p] = NaN;
                continue;
            }
            const r1 = v[0] * a1[0] + v[1] * a1[1] + v[2] * a1[2];
            const r2 = v[0] * a2[0] + v[1] * a2[1] + v[2] * a2[2];
            const wf = (m22 * r1 - m12 * r2) / det;
            const ws = (m11 * r2 - m12 * r1) / det;
            const off = Math.hypot(v[0] - wf * a1[0] - ws * a2[0], v[1] - wf * a1[1] - ws * a2[1], v[2] - wf * a1[2] - ws * a2[2]);
            const reach = Math.hypot(wf * a1[0] + ws * a2[0], wf * a1[1] + ws * a2[1], wf * a1[2] + ws * a2[2]);
            if (off > Math.max(offAbs, OFF_MIX_REL * reach) || wf < -STROKE_SLACK || ws < -STROKE_SLACK || wf + ws > MAX_MIX) {
                D[p] = NaN;
                continue;
            }
            D[p] = Math.min(1, Math.max(0, wf + ws));
            fillW[p] = Math.max(0, wf);
            continue;
        }
        if (fo) {
            // ink beyond the outline's reach of any fill isn't this family's
            if (!ownReach[p]) {
                D[p] = NaN;
                continue;
            }
            // between the outline and the fill: inside a dot
            const e0 = r - outline[0];
            const e1 = g - outline[1];
            const e2 = b - outline[2];
            const k = (e0 * fo[0] + e1 * fo[1] + e2 * fo[2]) / foo;
            if (k >= -OUTLINE_SLACK && k <= 1 + OUTLINE_SLACK && Math.hypot(e0 - k * fo[0], e1 - k * fo[1], e2 - k * fo[2]) <= offAbs) {
                D[p] = 1;
                fillK[p] = Math.min(1, Math.max(0, k));
                continue;
            }
        }
        // away from the family, another family's color is that family's pixel, not partial coverage of this one
        // (near it, pixels are read against their actual neighbor, which keeps the edges where dots touch)
        const dOwn = Math.min(Math.hypot(r - color[0], g - color[1], b - color[2]), outline ? Math.hypot(r - outline[0], g - outline[1], b - outline[2]) : Infinity);
        if (!near[p] && others.some(O => { const d = Math.hypot(r - O[0], g - O[1], b - O[2]); return d <= CLUSTER_TOL && d < dOwn; })) {
            D[p] = NaN;
            continue;
        }
        // near the family, against the nearest other color; away from it, against the family's usual background
        // (faint antialiased rims just beyond the near zone were left unknown and broke the dots' edges)
        const b0 = near[p] ? bg[3 * p] : B[0];
        const b1 = near[p] ? bg[3 * p + 1] : B[1];
        const b2 = near[p] ? bg[3 * p + 2] : B[2];
        const w0 = edge[0] - b0;
        const w1 = edge[1] - b1;
        const w2 = edge[2] - b2;
        const ww = w0 * w0 + w1 * w1 + w2 * w2;
        if (ww < MIN_CONTRAST * MIN_CONTRAST) {
            D[p] = NaN;
            continue;
        }
        const v0 = r - b0;
        const v1 = g - b1;
        const v2 = b - b2;
        const m = (v0 * w0 + v1 * w1 + v2 * w2) / ww;
        const off = Math.hypot(v0 - m * w0, v1 - m * w1, v2 - m * w2);
        const tol = Math.max(offAbs, OFF_MIX_REL * Math.abs(m) * Math.sqrt(ww));
        D[p] = off > tol || m > MAX_MIX ? NaN : Math.min(1, Math.max(0, m));
    }
    if (!stroke)
        fillSaturatedOverlaps(image, D, B, color, others);
    const line = colorLine(B, color);
    const table = [B, color];
    // an isolated dot of this family has its fill at its middle: specks of another ink that D reads as covered
    // (hair2: the stroke's black, so the decimal points and a colon's dots, r 4.9, outvoted the one whole dot, r 30.3)
    // are not its dots
    const fill = stroke ? fillW : own;
    const isolated = isolatedDots(D, width, height, tol).filter(d => discShare(fill, width, height, d.cx, d.cy, ISOLATED_FILL_RADII * d.r) >= DOT_MIN_FILL);
    // outlined or stroked dots without an isolated whole one: their fills give the radius, and the center search runs
    // on the fills, whose edges are each dot's own wherever it's on top (D's edge is the union's, joined to lines of
    // the fill's color and to touching dots; cells1's magenta, chained to its neighbors by gray lines that read as
    // half the stroke, got 4.5 for 20 from the distance transform)
    const fillCover = outline ? fillK : stroke ? fillW : null;
    const byFill = !isolated.length && fillCover ? outlinedRadius(fillCover, own, D, width, height) : null;
    const radius = isolated.length ? median(isolated.map(d => d.r)) : byFill?.radius ?? radiusFromDistance(D, width, height);
    let dots = 0;
    let strongDots = 0;
    let coverage = 0;
    if (radiusPlausible(radius, isolated.length)) {
        const weight = new Uint8Array(n);
        for (let p = 0; p < n; p++)
            weight[p] = Number.isNaN(D[p]) ? 0 : 1;
        // a center counts if most of its edge is seen and its disc is filled with the family's own color (text:
        // round glyphs are hollow; fanc2: a rim gray between dot and overlap read black text as beyond its color,
        // so D was full there, and every glyph was a dot); a stroked family's inner disc must be its fill
        const rf = DOT_FILL_RADII * (byFill?.fillRadius ?? (stroke ? opts.fillRadius ?? radius * DOT_FILL_RADII : radius));
        const inside = stroke ? fillW : own;
        const filled = c => {
            let count = 0;
            let on = 0;
            for (let y = Math.max(0, Math.floor(c.y - rf)); y < Math.min(height, Math.ceil(c.y + rf)); y++) {
                for (let x = Math.max(0, Math.floor(c.x - rf)); x < Math.min(width, Math.ceil(c.x + rf)); x++) {
                    if ((x + 0.5 - c.x) ** 2 + (y + 0.5 - c.y) ** 2 > rf * rf)
                        continue;
                    count++;
                    if (inside[y * width + x] > 0.5)
                        on++;
                }
            }
            return count > 0 && on >= DOT_MIN_FILL * count;
        };
        const centers = (byFill ? houghCenters(fillCover, weight, width, height, byFill.fillRadius) : houghCenters(D, weight, width, height, radius))
            .filter(c => c.arc >= DOT_MIN_ARC && filled(c));
        dots = centers.length;
        strongDots = centers.filter(c => c.arc >= DOT_STRONG_ARC).length;
        // share of the family's pixels inside the found discs
        const inDisc = new Uint8Array(n);
        for (const c of centers) {
            for (let y = Math.max(0, Math.floor(c.y - radius)); y < Math.min(height, Math.ceil(c.y + radius)); y++) {
                for (let x = Math.max(0, Math.floor(c.x - radius)); x < Math.min(width, Math.ceil(c.x + radius)); x++) {
                    if ((x + 0.5 - c.x) ** 2 + (y + 0.5 - c.y) ** 2 <= radius * radius)
                        inDisc[y * width + x] = 1;
                }
            }
        }
        // a stroked family's coverage is its fill's: D also covers the stroke's color, which is text and axes too
        // (hair3: the black ink diluted it to 0.26); so is one measured by its fills
        const ofFamily = stroke ? fillW : byFill ? own : D;
        let fam = 0;
        let covered = 0;
        for (let p = 0; p < n; p++) {
            if (ofFamily[p] > 0.5) {
                fam++;
                covered += inDisc[p];
            }
        }
        coverage = fam ? covered / fam : 0;
    }
    const pixels = mask.reduce((t, v) => t + v, 0);
    return {
        kind: 'opaque', seed: color, outline, stroke, line, background: B, table, D, isolated, radius, dots, strongDots, coverage, pixels,
        ...(byFill ? { fillRadius: byFill.fillRadius } : {}),
        levels: [{ t: Math.hypot(...B.map((v, ch) => v - color[ch])), color, count: pixels }],
        beyond: [], fit: { alpha: 1, c: [...color], rms: 0, maxAbs: 0, underdetermined: true }, translucent: false,
    };
}

/**
 * Mark saturated overlaps covered (D = 1), in place: 8-connected regions of unknown pixels whose colors lie beyond
 * the family's color on its line from B (up to OVERLAP_MAX_MIX, and within OFF_MIX_ABS of it, not the family's
 * noise allowance: an overlap is one flat color, and on the noisy economist the allowance reached its pink and blue
 * dots) and aren't another family's, with at least OVERLAP_BORDER_SHARE of their border covered (D > 0.5) and a
 * (2·OVERLAP_MIN_DEPTH + 1)² square inside. O(pixels · OVERLAP_MIN_DEPTH²).
 *
 * @param {import('./image.js').RgbaImage} image
 * @param {Float32Array} D  coverage, NaN unknown
 * @param {number[]} B  0–255, the family's background
 * @param {number[]} color  0–255
 * @param {number[][]} others  0–255, the other families' colors
 */
function fillSaturatedOverlaps(image, D, B, color, others) {
    const { width, height, data } = image;
    const w = color.map((v, ch) => v - B[ch]);
    const ww = w[0] * w[0] + w[1] * w[1] + w[2] * w[2];
    if (ww < MIN_CONTRAST * MIN_CONTRAST)
        return;
    const beyond = p => {
        if (!Number.isNaN(D[p]))
            return false;
        if (others.some(O => Math.hypot(data[4 * p] - O[0], data[4 * p + 1] - O[1], data[4 * p + 2] - O[2]) <= CLUSTER_TOL))
            return false;
        const v = [data[4 * p] - B[0], data[4 * p + 1] - B[1], data[4 * p + 2] - B[2]];
        const m = (v[0] * w[0] + v[1] * w[1] + v[2] * w[2]) / ww;
        return m > 1 && m <= OVERLAP_MAX_MIX && Math.hypot(v[0] - m * w[0], v[1] - m * w[1], v[2] - m * w[2]) <= Math.max(OFF_MIX_ABS, OFF_MIX_REL * m * Math.sqrt(ww));
    };
    const seen = new Uint8Array(width * height);
    for (let start = 0; start < seen.length; start++) {
        if (seen[start] || !beyond(start))
            continue;
        const comp = [start];
        seen[start] = 1;
        let border = 0;
        let covered = 0;
        for (let k = 0; k < comp.length; k++) {
            const p = comp[k];
            const x = p % width;
            const y = (p - x) / width;
            for (let dy = -1; dy <= 1; dy++) {
                for (let dx = -1; dx <= 1; dx++) {
                    if ((dx || dy) && x + dx >= 0 && x + dx < width && y + dy >= 0 && y + dy < height) {
                        const q = p + dy * width + dx;
                        if (seen[q])
                            continue;
                        if (beyond(q)) {
                            seen[q] = 1;
                            comp.push(q);
                        }
                        else {
                            border++;
                            if (D[q] > 0.5)
                                covered++;
                        }
                    }
                }
            }
        }
        if (border && covered >= OVERLAP_BORDER_SHARE * border && hasSquare(comp, width, height, OVERLAP_MIN_DEPTH)) {
            for (const p of comp)
                D[p] = 1;
        }
    }
}

/**
 * Radius of outlined dots without an isolated whole one: each dot's fill is bounded by its own outline, so the
 * topmost dots' fills are whole discs even where the dots touch others or a line of their own color drawn behind
 * them (phase1: most dots sit on their fitted curve, whose distance-transform ridge outvoted them, r 4–6 against
 * 18). The fill radius is the median of the round fill components; the outline's width comes from the covered
 * pixels in a ring just outside each, as the area they add to the disc. Null without a round fill. fillK is the fill's
 * antialiased coverage (0 at the outline's color, 1 at the fill's), or a stroked family's fill weight.
 *
 * @returns {{radius: number, fillRadius: number}|null}  px
 */
function outlinedRadius(fillK, own, D, width, height) {
    const discs = roundFills(fillK, width, height).filter(d => d.r >= DOT_MIN_RADIUS);
    if (!discs.length)
        return null;
    const rf = median(discs.map(d => d.r));
    const widths = discs.map(d => {
        let added = 0;
        const outer = d.r + OUTLINE_REACH;
        for (let y = Math.max(0, Math.floor(d.cy - outer)); y < Math.min(height, Math.ceil(d.cy + outer)); y++) {
            for (let x = Math.max(0, Math.floor(d.cx - outer)); x < Math.min(width, Math.ceil(d.cx + outer)); x++) {
                const q = Math.hypot(x + 0.5 - d.cx, y + 0.5 - d.cy);
                const p = y * width + x;
                if (q > d.r && q <= outer && !own[p] && D[p] > 0)
                    added += Math.min(1, D[p]);
            }
        }
        return Math.sqrt(d.r * d.r + added / Math.PI) - d.r;
    });
    return { radius: rf + median(widths), fillRadius: rf };
}

/**
 * Round fill components: those of fill > 0.5, away from the image border, round by their second moments over their
 * own pixels and their antialiased rim (fill 0–0.5 next to them), weighted by min(fill, 1). Unlike isolatedDots, a
 * window around the component isn't read: fills in a cluster sit 2 px apart, across their strokes (cells1's magenta:
 * every fill failed when its neighbors' fills fell in the window). O(pixels).
 *
 * @returns {{cx: number, cy: number, r: number}[]}  px
 */
function roundFills(fill, width, height) {
    const out = [];
    const taken = new Uint8Array(fill.length);
    for (const comp of components(fill, width, height)) {
        for (const p of comp)
            taken[p] = 1;
    }
    for (const comp of components(fill, width, height)) {
        const pixels = new Set(comp);
        for (const p of comp) {
            const x = p % width;
            for (const q of [p - 1, p + 1, p - width, p + width, p - width - 1, p - width + 1, p + width - 1, p + width + 1]) {
                const qx = q % width;
                if (q >= 0 && q < fill.length && Math.abs(qx - x) <= 1 && !taken[q] && fill[q] > 0)
                    pixels.add(q);
            }
        }
        let a = 0;
        let sx = 0;
        let sy = 0;
        let edge = false;
        for (const p of pixels) {
            const x = p % width;
            const y = (p - x) / width;
            edge ||= x === 0 || y === 0 || x === width - 1 || y === height - 1;
            const w = Math.min(1, fill[p]);
            a += w;
            sx += w * (x + 0.5);
            sy += w * (y + 0.5);
        }
        if (edge || a < ISOLATED_MIN_AREA)
            continue;
        const cx = sx / a;
        const cy = sy / a;
        let mxx = 0;
        let myy = 0;
        let mxy = 0;
        for (const p of pixels) {
            const x = p % width;
            const y = (p - x) / width;
            const w = Math.min(1, fill[p]);
            mxx += w * (x + 0.5 - cx) ** 2;
            myy += w * (y + 0.5 - cy) ** 2;
            mxy += w * (x + 0.5 - cx) * (y + 0.5 - cy);
        }
        mxx /= a;
        myy /= a;
        mxy /= a;
        const tr = mxx + myy;
        const disc = Math.sqrt(Math.max(0, tr * tr / 4 - (mxx * myy - mxy * mxy)));
        const r = Math.sqrt(a / Math.PI);
        if (Math.sqrt((tr / 2 + disc) / (tr / 2 - disc)) <= ISOLATED_MAX_ASPECT && Math.sqrt(2 * tr) / r <= ISOLATED_MAX_MOMENT_RATIO)
            out.push({ cx, cy, r });
    }
    return out;
}

// share of the pixels whose centers lie within r of (cx, cy) where v > 0.5
function discShare(v, width, height, cx, cy, r) {
    let count = 0;
    let on = 0;
    for (let y = Math.max(0, Math.floor(cy - r)); y < Math.min(height, Math.ceil(cy + r)); y++) {
        for (let x = Math.max(0, Math.floor(cx - r)); x < Math.min(width, Math.ceil(cx + r)); x++) {
            if ((x + 0.5 - cx) ** 2 + (y + 0.5 - cy) ** 2 > r * r)
                continue;
            count++;
            if (v[y * width + x] > 0.5)
                on++;
        }
    }
    return count ? on / count : 0;
}

// whether a (2·depth + 1)² square of pixels fits inside a region (pixel indices)
function hasSquare(comp, width, height, depth) {
    const inside = new Set(comp);
    return comp.some(p => {
        const x = p % width;
        const y = (p - x) / width;
        if (x < depth || y < depth || x + depth >= width || y + depth >= height)
            return false;
        for (let dy = -depth; dy <= depth; dy++) {
            for (let dx = -depth; dx <= depth; dx++) {
                if (!inside.has(p + dy * width + dx))
                    return false;
            }
        }
        return true;
    });
}

// relative luminance proxy, 0–255 (Rec. 709 weights on the sRGB values): only compared, never displayed
function luminance(c) {
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
}

// whether a color's pixels are thin as a whole: an opening by OUTLINE_OPENING keeps under OUTLINE_MAX_THICK of them
function isThin(image, color) {
    const { width, height, data } = image;
    const mask = new Uint8Array(width * height);
    let total = 0;
    for (let p = 0; p < mask.length; p++) {
        mask[p] = Math.hypot(data[4 * p] - color[0], data[4 * p + 1] - color[1], data[4 * p + 2] - color[2]) <= CLUSTER_TOL ? 1 : 0;
        total += mask[p];
    }
    let kept = 0;
    for (const v of opening(mask, width, height, OUTLINE_OPENING))
        kept += v;
    return total > 0 && kept < OUTLINE_MAX_THICK * total;
}

function radiusPlausible(radius, isolatedCount) {
    return radius >= DOT_MIN_RADIUS || (radius >= SMALL_DOT_MIN_RADIUS && isolatedCount >= SMALL_DOT_MIN_ISOLATED);
}

function isDotFamily(f) {
    return radiusPlausible(f.radius, f.isolated.length) && f.strongDots >= DOT_MIN_STRONG && f.coverage >= DOT_MIN_COVERAGE_ANY
        && (f.coverage >= DOT_MIN_COVERAGE || f.isolated.length >= DOT_MIN_ISOLATED);
}

/**
 * Opaque dot families: color clusters that aren't a background, aren't levels of a translucent family, and show
 * at least DOT_MIN_COUNT dot-shaped edges at a radius of at least DOT_MIN_RADIUS.
 */
export function opaqueFamilies(image, backgrounds, translucent, tol = EXACT_TOLERANCES) {
    const onTranslucent = c => translucent.some(f => {
        const { B, u } = f.line;
        const e = B.map((v, ch) => v - c[ch]);
        const t = e[0] * u[0] + e[1] * u[1] + e[2] * u[2];
        return Math.hypot(e[0] - t * u[0], e[1] - t * u[1], e[2] - t * u[2]) <= CLUSTER_ON_LINE_TOL;
    });
    const { width, height, data } = image;
    let out = [];
    // a cluster between a family found already and that family's background, its outline or a page color is an
    // antialiased rim, not a family (synthetic swarm: maroon's blend with white made a "family"). Between a fill and
    // its outline, only if it is mostly edge, as a rim is: other fills of an outlined scale sit there too (phase1:
    // grays 136 and 82 between gray 187 and the black outline, 62% interior; a rim is 1–2 px wide, unless the fill's
    // own tolerance takes in fill pixels: a synthetic rim at 149, 6, 220 was 39 units from its fill)
    const isOutline = c => out.some(f => f.outline && Math.hypot(...f.outline.map((v, ch) => v - c[ch])) <= CLUSTER_TOL);
    const near = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]) <= CLUSTER_TOL;
    const rimOf = (f, c, interiorShare) => interiorShare < BLEND_OUTLINE_MAX_INTERIOR || near(c, f.seed) || near(c, f.outline);
    const isBlend = (c, interiorShare) => out.some(f => [f.background, ...(f.outline && rimOf(f, c, interiorShare) ? [f.outline] : []), ...backgrounds].some(P => {
        const w = f.seed.map((v, ch) => v - P[ch]);
        const ww = w[0] * w[0] + w[1] * w[1] + w[2] * w[2];
        const v = c.map((x, ch) => x - P[ch]);
        const m = (v[0] * w[0] + v[1] * w[1] + v[2] * w[2]) / ww;
        return m > 0.05 && m < 0.95 && Math.hypot(v[0] - m * w[0], v[1] - m * w[1], v[2] - m * w[2]) <= CLUSTER_ON_LINE_TOL;
    }));
    // Only the dominant color is excluded up front: a dot color can cover 5% of an image (sparse #3: 5.9%) and so
    // count as a background color; the dot test tells it from a panel.
    for (const cl of colorClusters(image, backgrounds.slice(0, 1))) {
        if (onTranslucent(cl.color) || isOutline(cl.color))
            continue;
        // cheap test first: a dot family's pixels are mostly interior; text's are mostly edges
        const c = cl.color;
        const member = p => Math.hypot(data[4 * p] - c[0], data[4 * p + 1] - c[1], data[4 * p + 2] - c[2]) <= CLUSTER_TOL;
        let count = 0;
        let interior = 0;
        for (let y = 1; y < height - 1; y++) {
            for (let x = 1; x < width - 1; x++) {
                const p = y * width + x;
                if (!member(p))
                    continue;
                count++;
                let all = true;
                for (let dy = -1; dy <= 1 && all; dy++) {
                    for (let dx = -1; dx <= 1; dx++) {
                        if (!member(p + dy * width + dx)) {
                            all = false;
                            break;
                        }
                    }
                }
                if (all)
                    interior++;
            }
        }
        if (isBlend(cl.color, count ? interior / count : 0))
            continue;
        if (!count || interior / count < DOT_MIN_INTERIOR)
            continue;
        let fam = measureOpaque(image, cl.color, backgrounds, [], tol);
        // Outlined or stroked fills, in order: (1) a ring whose color is thin over the whole image is an outline that
        // shows its own color (JMP's 1.7 px outlines); (2) else a fill inside a stroke too thin to show its color, off
        // the line from the page to the fill (engagement1, cells1: it places colored fills better than the outline
        // model, rmse 0.28 px against 0.95); (3) else an outline whose color is also other ink, counted only next to
        // the fill (measureOpaque's OUTLINE_REACH), for a fill that is thick as an outline is not (phase1: black
        // outlines around grays, which the stroke model can't read, white, gray and black being on one line, with
        // text, axes and a black colorbar block in the outline's color; hair2's pink, one strong arc as stroked).
        // Not around a translucent family's level (economist: a red–blue lens inside red dots made a phantom
        // "outlined" family). In (3) the outline is darker than the fill, as ink shared with text is: a lighter ring is
        // a band or region the dots lie under (bench3 ANOVA: pink under a pale blue band read as fills "outlined" in
        // the band's color, a second family of 31). That guard and the thick fill stop economist's lens, so (3) may
        // take a ring on a translucent family's line: gray outlines and a gray line can read as one (degradation1:
        // outline 136 and line 166 as two levels of a "family" with no dots).
        const plain = fam;
        const ring = plain.background;
        const notPage = !backgrounds.some(P => Math.hypot(...P.map((v, ch) => v - ring[ch])) <= CLUSTER_TOL);
        const ringable = !onTranslucent(ring) && notPage;
        const outlinedFamily = () => {
            const outlined = measureOpaque(image, cl.color, backgrounds, [], tol, ring);
            return isDotFamily(outlined) ? { ...outlined, fillRadius: outlined.fillRadius ?? plain.radius } : null;
        };
        if (!isDotFamily(fam) && ringable && isThin(image, ring))
            fam = outlinedFamily() ?? fam;
        if (!isDotFamily(fam))
            fam = strokedFamily(image, cl.color, backgrounds, tol, plain) ?? fam;
        if (!isDotFamily(fam) && notPage && !isThin(image, cl.color) && luminance(ring) < luminance(cl.color))
            fam = outlinedFamily() ?? fam;
        fam.interior = interior / count;
        if (isDotFamily(fam))
            out.push(fam);
    }
    // Specks go before the others are measured again: a speck family's color can be a real family's stroke (hair3:
    // the black of the decimal points is the pink dots' outline, which excluded left their radius 18 for 28)
    out = withoutSpecks(out, translucent.length > 0);
    // Measure again with the other families' colors excluded from each, now that they are known, for the fit. The
    // first measurement decides what is a family: a small one surrounded by others (nicest language's red and dark
    // blue, about 10 dots each) can fail the test once its neighbors' colors are excluded.
    const colors = out.map(f => f.seed);
    // with stroked families, every family's coverage gives the other fills' rings to their dots (strokes are usually
    // drawn on every dot, also those whose fill passed as plain: engagement1's navy took the others' rings)
    // (the reach is from the fill's color mask, as strokeColor reads the ring)
    const stroked = out.some(f => f.stroke);
    const final = out.map(f => {
        const rings = stroked ? out.filter(g => g !== f).map(g => ({ fill: g.seed, reach: STROKE_RING })) : null;
        const g = measureOpaque(image, f.seed, backgrounds, colors.filter(c => c !== f.seed), tol, f.outline, { stroke: f.stroke, fillRadius: f.fillRadius, rings });
        return { ...g, radius: g.radius ?? f.radius, fillRadius: f.fillRadius, dots: f.dots, strongDots: f.strongDots, coverage: f.coverage };
    });
    return final.sort((a, b) => b.dots - a.dots);
}

// The family of a fill color inside a stroke (strokeColor), or null when no stroke shows or it isn't a dot family.
// `plain` is the color measured without one: its radius is the fill's.
function strokedFamily(image, color, backgrounds, tol, plain) {
    const stroke = strokeColor(image, color, backgrounds[0]);
    if (!stroke)
        return null;
    const fam = measureOpaque(image, color, backgrounds, [], tol, null, { stroke: stroke.color, fillRadius: plain.radius });
    return isDotFamily(fam) ? { ...fam, fillRadius: plain.radius, strokeOff: stroke.off } : null;
}

/**
 * An opaque family measured as the user describes it (the app's outline edits): its fill color, its outline or
 * none, and the other families' fills. With rings to others, a ring pixel nearer another fill is that dot's, as in
 * opaqueFamilies' final pass, so every family can be measured again when one's outline changes.
 *
 * @param {import('./image.js').RgbaImage} image
 * @param {number[]} seed  the fill, 0–255
 * @param {object} [opts]
 * @param {{color: number[], fillRadius: number}|null} [opts.stroke]  0–255, px
 * @param {number[]|null} [opts.outline]  a JMP-style outline color (measureOpaque), kept from the estimate
 * @param {number[][]} [opts.others]  the other families' fills, 0–255
 * @param {boolean} [opts.ringsToOthers]
 * @param {boolean} [opts.jpeg]
 * @returns {object}  an estimate, as estimateStyle's; style.radius is null when no dot shape shows
 */
export function opaqueFamilyEstimate(image, seed, opts = {}) {
    const tol = opts.jpeg ? JPEG_TOLERANCES : EXACT_TOLERANCES;
    const backgrounds = backgroundColors(image);
    const others = opts.others ?? [];
    const stroke = opts.stroke ?? null;
    const rings = opts.ringsToOthers && others.length ? others.map(fill => ({ fill, reach: STROKE_RING })) : null;
    const fam = measureOpaque(image, seed, backgrounds, others, tol, stroke ? null : opts.outline ?? null,
        { stroke: stroke?.color ?? null, fillRadius: stroke?.fillRadius, rings });
    const measured = { ...fam, fillRadius: stroke?.fillRadius ?? fam.fillRadius };
    return opaqueEstimate(image, measured, [measured], 0, flatMask(image, tol), backgroundColor(image));
}

function median(a) {
    const s = [...a].sort((x, y) => x - y);
    return s.length % 2 ? s[(s.length - 1) / 2] : (s[s.length / 2 - 1] + s[s.length / 2]) / 2;
}

/**
 * Estimate the style of one translucent dot family.
 *
 * @param {import('./image.js').RgbaImage} image
 * @param {object} [opts]
 * @param {number[]} [opts.seed]    a color of the family (e.g. picked by the user); default: the best candidate
 * @param {number} [opts.family]    index into the candidates, instead of a seed
 * @param {boolean} [opts.jpeg]     the image is JPEG-compressed: noise-tolerant thresholds (JPEG_TOLERANCES); candidates
 *   passed in must have been found with the same setting
 * @returns {object}  { background, candidates, family, style, D, isolated, edge, notes }
 *   style: { color, alpha, levels, radius, blur } ready for render.js; radius and blur are null when no isolated
 *   dot was found. notes: human-readable caveats for display.
 */
export function estimateStyle(image, opts = {}) {
    const { width, height } = image;
    const background = backgroundColor(image);
    const B = background.color;
    const tol = opts.jpeg ? JPEG_TOLERANCES : EXACT_TOLERANCES;
    const flat = flatMask(image, tol);
    const backgrounds = backgroundColors(image);
    // candidates are the slow part (seconds on large images): reuse them when estimating another family
    const candidates = opts.candidates ?? findCandidates(image, B, flat, backgrounds, tol);
    let fam;
    let familyIndex = opts.family ?? 0;
    if (opts.seed) {
        fam = measureFamily(image, B, opts.seed, flat, tol);
        fam.kind = 'translucent';
        if (!fam.translucent) {
            const plain = measureOpaque(image, opts.seed, backgrounds, [], tol);
            fam = isDotFamily(plain) ? plain : strokedFamily(image, opts.seed, backgrounds, tol, plain) ?? plain;
        }
        familyIndex = -1;
    }
    else {
        fam = candidates[familyIndex];
    }
    const notes = [];
    if (!fam || !fam.levels.length)
        return { background, candidates, family: familyIndex, style: null, D: null, isolated: [], edge: null, notes: ['No dot family found.'] };
    if (fam.kind === 'opaque')
        return opaqueEstimate(image, fam, candidates, familyIndex, flat, background);
    const table = levelTable(B, fam.levels, fam.fit);
    const D = countImage(fam.proj, fam.line, table, tol);
    const isolated = isolatedDots(D, width, height, tol);
    let radius = null;
    let blur = null;
    let edge = null;
    let ring = null;
    if (isolated.length) {
        radius = median(isolated.map(d => d.r));
        edge = edgeBlur(D, width, isolated);
        // the profile's σ includes the pixel box, which the renderer's exact coverage already supplies
        blur = Math.sqrt(Math.max(0, edge.sigma ** 2 - 1 / 12));
        const ringed = isolated.filter(d => d.ringed);
        if (ringed.length >= RINGED_MIN_SHARE * isolated.length) {
            const { B: lb, u } = fam.line;
            const tC = (lb[0] - fam.fit.c[0]) * u[0] + (lb[1] - fam.fit.c[1]) * u[1] + (lb[2] - fam.fit.c[2]) * u[2];
            ring = measureRing(fam.proj.t, width, ringed, radius, fam.fit.alpha, tC);
            if (ring)
                notes.push(`Outlined dots: a ring from ${(ring.inner * radius).toFixed(1)} to ${radius.toFixed(1)} px, ${Math.round(100 * (fam.fit.alpha + ring.alpha))}% opaque against ${Math.round(100 * fam.fit.alpha)}% inside, drawn in the fill's color line (an approximation where outlines cross other dots).`);
        }
    }
    else {
        notes.push('No isolated whole dot found, so the radius and edge blur must be set by hand.');
    }
    const inferredLevels = fam.levels.flatMap((L, i) => L.inferred ? [i + 1] : []);
    if (inferredLevels.length)
        notes.push(`Too few flat pixels show ${inferredLevels.length > 1 ? 'levels' : 'level'} ${inferredLevels.join(', ')} (overlap count), so its color was inferred from the levels around it.`);
    return {
        background, candidates, family: familyIndex, flat,
        familyMeasure: fam,
        style: { color: fam.fit.c, alpha: fam.fit.alpha, levels: table, measuredLevels: fam.levels.length, inferredLevels, radius, blur, ...(ring ? { ring } : {}) },
        D, isolated, edge, notes,
    };
}

/**
 * All dot families: translucent ones (levels along a color direction) first, then opaque ones (color clusters).
 */
export function findCandidates(image, B, flat, backgrounds, tol = EXACT_TOLERANCES) {
    const translucent = familyCandidates(image, B, flat, tol).filter(c => c.translucent);
    for (const c of translucent)
        c.kind = 'translucent';
    const opaque = opaqueFamilies(image, backgrounds, translucent, tol);
    // a translucent "family" with a level in an opaque family's outline color is those outlines (degradation1: the
    // gray outlines and the gray line as two levels, no isolated dot, no radius)
    const rims = opaque.flatMap(f => [f.outline, f.stroke].filter(Boolean));
    const real = translucent.filter(t => !t.levels.some(L => rims.some(c => Math.hypot(...c.map((v, ch) => v - L.color[ch])) <= CLUSTER_TOL)));
    return [...real, ...opaque];
}

// Small dots count only when nothing else does: beside regular dots, a few round specks are a dotted gridline, the
// dots of i's or decimal points (comment1: 9 black specks at r 2.5 beside orange dots at r 9.2; wave1: 27 decimal
// points at r 3.23, coverage 0.19, beside dots at r 14.9). Small is under DOT_MIN_RADIUS, or under SPECK_RADIUS_RATIO
// of the largest family's radius with most of the color not dot-shaped (text). A translucent family's radius isn't
// measured yet, but its levels already show real dots.
function withoutSpecks(opaque, anyTranslucent) {
    const largest = Math.max(...opaque.map(c => c.radius ?? 0), 0);
    const small = c => c.radius < DOT_MIN_RADIUS || (c.radius < SPECK_RADIUS_RATIO * largest && c.coverage < DOT_MIN_COVERAGE);
    const kept = opaque.filter(c => !small(c));
    return anyTranslucent || kept.length ? kept : opaque;
}

// the estimate for an opaque family: its own background, radius from isolated dots or the distance transform, blur
// from isolated dots or the family's boundaries
function opaqueEstimate(image, fam, candidates, familyIndex, flat, pageBackground) {
    const { width, height } = image;
    const notes = ['Opaque dots: overlapping dots hide each other, so a dot entirely behind others can’t be found.'];
    if (fam.stroke && fam.fillRadius)
        notes.push(`Outlined: a fill of radius ${fam.fillRadius.toFixed(1)} px inside a stroke too thin to show its own color; `
            + `its darkest mixes average ${fam.stroke.map(Math.round).join(', ')}. The ring counts as part of the dot, and a ring nearer another family's fill is that dot's.`);
    let edge;
    if (fam.isolated.length) {
        edge = edgeBlur(fam.D, width, fam.isolated);
    }
    else {
        edge = boundaryBlur(fam.D, width, height);
        notes.push('No isolated whole dot: the radius comes from the dots’ shapes and is refit by Find dots.');
    }
    const blur = Math.sqrt(Math.max(0, edge.sigma ** 2 - 1 / 12));
    const count = fam.background.every((v, ch) => v === pageBackground.color[ch]) ? pageBackground.fraction : null;
    // what the fit compares: coverage scaled to color units along the line from the family's background to its
    // color, which is what the renderer produces for the table [B, c] at α = 1 (NaN pixels are masked)
    const span = Math.hypot(...fam.background.map((v, ch) => v - fam.seed[ch]));
    const fitT = new Float32Array(fam.D.length);
    for (let p = 0; p < fitT.length; p++)
        fitT[p] = Number.isNaN(fam.D[p]) ? 0 : span * fam.D[p];
    return {
        background: { color: fam.background, fraction: count },
        candidates, family: familyIndex, flat,
        familyMeasure: fam,
        style: {
            color: [...fam.seed], alpha: 1, levels: fam.table, measuredLevels: 1, radius: fam.radius, blur,
            // for display only: the fit sees an outlined dot as one disc (measureOpaque); a stroke is drawn in the
            // darkest mix seen, not its true color, which no pixel shows
            ...((fam.outline ?? fam.stroke) && fam.fillRadius ? { outline: { color: [...(fam.outline ?? fam.stroke)], inner: fam.fillRadius / fam.radius } } : {}),
        },
        D: fam.D, fitT, isolated: fam.isolated, edge, notes,
    };
}

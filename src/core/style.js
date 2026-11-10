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
const ISOLATED_MAX_LEVEL = 1.25; // components never above this count are single dots
const ISOLATED_MIN_AREA = 12; // px² (JMP's tiny dots: r 2.6, area 21)
const ISOLATED_MAX_ASPECT = 1.05; // ratio of principal second-moment radii
const ISOLATED_MAX_MOMENT_RATIO = 1.03; // r(moment) / r(area); 1 for a uniform disc
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
const DOT_MIN_ARC = 0.5; // fraction of the circumference for a Hough center to count
const DOT_STRONG_ARC = 0.75;
const DOT_MIN_STRONG = 3;
const DOT_MIN_COVERAGE = 0.35;
const DOT_MIN_ISOLATED = 3;
// Text looks like dots to a Hough vote (round glyphs, and every gray of its antialiasing is a color cluster), but
// it is strokes: mostly edge pixels, and the inside of an "o" is empty.
const DOT_MIN_INTERIOR = 0.1; // share of a cluster's pixels with all 8 neighbors in the cluster
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
 * 1 where every channel of the pixel's 3×3 neighborhood is within FLAT_TOL of the pixel (borders are 0).
 *
 * @param {import('./image.js').RgbaImage} image
 * @returns {Uint8Array}
 */
export function flatMask(image) {
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
                        if (Math.abs(data[4 * q + ch] - data[4 * p + ch]) > FLAT_TOL)
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
 * @returns {{levels: Level[], beyond: Level[]}}
 */
export function findLevels(image, proj, flat) {
    const { t, perp } = proj;
    const nb = Math.ceil(LEVEL_MAX_T / LEVEL_BIN);
    const hist = new Float64Array(nb + 1);
    let total = 0;
    for (let p = 0; p < t.length; p++) {
        if (flat[p] && t[p] > LEVEL_SEPARATION / 2 && votesForLevel(t[p], perp[p])) {
            hist[Math.min(nb, Math.round(t[p] / LEVEL_BIN))]++;
            total++;
        }
    }
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
        for (let b = lo; b <= hi; b++) {
            const c = hist[b - 1] + hist[b] + hist[b + 1];
            if (hist[b] > 0 && c > bestCount) {
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

/**
 * Least-squares fit of L_n = c + (B − c) q^n per channel with one q = 1 − α shared by the channels, to colors
 * L_0 = B, L_1, …: golden-section search over q, c in closed form per channel for each q.
 *
 * @param {number[][]} colors  L_0 … L_K, K ≥ 1
 * @returns {{alpha: number, c: number[], rms: number, maxAbs: number}}
 */
export function fitGeometric(colors) {
    const evaluate = q => {
        const c = [];
        let sse = 0;
        let maxAbs = 0;
        for (let ch = 0; ch < 3; ch++) {
            const B = colors[0][ch];
            // L_n − B q^n = c (1 − q^n): regression through the origin on w_n = 1 − q^n
            let sw = 0;
            let swy = 0;
            for (let n = 0; n < colors.length; n++) {
                const w = 1 - q ** n;
                sw += w * w;
                swy += w * (colors[n][ch] - B * q ** n);
            }
            c.push(swy / sw);
            for (let n = 0; n < colors.length; n++) {
                const r = colors[n][ch] - (c[ch] + (B - c[ch]) * q ** n);
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
 * @returns {Float32Array}
 */
export function countImage(proj, line, table) {
    const { B, u } = line;
    const ts = table.map(c => (B[0] - c[0]) * u[0] + (B[1] - c[1]) * u[1] + (B[2] - c[2]) * u[2]);
    const K = ts.length - 1;
    const { t, perp } = proj;
    const D = new Float32Array(t.length);
    for (let p = 0; p < t.length; p++) {
        const tp = t[p];
        if (perp[p] > COUNT_PERP_TOL || (perp[p] > NEAR_B_ABS_TOL && perp[p] > COUNT_PERP_REL_TOL * tp)) {
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
 * @returns {object[]}  candidates: { seed, line, levels, beyond, fit, pixels, translucent }
 */
export function familyCandidates(image, B, flat) {
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
        candidates.push(measureFamily(image, B, seed, flat));
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
 */
export function measureFamily(image, B, seed, flat) {
    const line = colorLine(B, seed);
    const proj = projectOnLine(image, line);
    const { levels, beyond } = findLevels(image, proj, flat);
    const fit = levels.length ? fitGeometric([B, ...levels.map(L => L.color)]) : null;
    const plausible = fit !== null && fit.rms <= FIT_MAX_RMS && fit.c.every(v => v >= -FIT_C_TOL && v <= 255 + FIT_C_TOL);
    return {
        seed, line, proj, levels, beyond, fit,
        pixels: levels.reduce((s, L) => s + L.count, 0),
        translucent: levels.length >= 2 && plausible,
    };
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
 * @returns {{cx: number, cy: number, r: number, aspect: number, box: number[]}[]}
 */
export function isolatedDots(D, width, height) {
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
        if (maxD > ISOLATED_MAX_LEVEL || x0 - m < 0 || y0 - m < 0 || x1 + m >= width || y1 + m >= height)
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
                const w = Math.min(1, d);
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
                const w = Math.min(1, D[y * width + x]);
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
        dots.push({ cx, cy, r, aspect, box: [x0, y0, x1, y1] });
    }
    return dots;
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

/**
 * An opaque family from a color: the coverage image D (0 its local background … 1 the color; NaN where a pixel is
 * neither, or contrast is too low to tell), its typical background (the most common color around it), the radius
 * (isolated dots, else the distance transform), and how many filled, dot-shaped edges a Hough vote finds.
 */
export function measureOpaque(image, color, backgrounds, others = []) {
    const { width, height, data } = image;
    const n = width * height;
    const mask = new Uint8Array(n);
    for (let p = 0; p < n; p++)
        mask[p] = Math.hypot(data[4 * p] - color[0], data[4 * p + 1] - color[1], data[4 * p + 2] - color[2]) <= CLUSTER_TOL ? 1 : 0;
    const thick = opening(mask, width, height, THICK_OPENING);
    const B = ringBackground(image, thick.some(v => v) ? thick : mask) ?? backgrounds[0];
    const near = dilate(mask, width, height, NEAR_FAMILY);
    // the family's color noise: RMS distance from its color over interior pixels (all 8 neighbors in the family)
    let noiseSum = 0;
    let noiseN = 0;
    for (let y = 1; y < height - 1; y++) {
        for (let x = 1; x < width - 1; x++) {
            const p = y * width + x;
            if (!thick[p] || !mask[p - 1] || !mask[p + 1] || !mask[p - width] || !mask[p + width])
                continue;
            noiseSum += (data[4 * p] - color[0]) ** 2 + (data[4 * p + 1] - color[1]) ** 2 + (data[4 * p + 2] - color[2]) ** 2;
            noiseN++;
        }
    }
    const offAbs = Math.max(OFF_MIX_ABS, OFF_MIX_NOISE * (noiseN ? Math.sqrt(noiseSum / noiseN) : 0));
    const pageColorsForBg = [...backgrounds, B].filter(P => Math.hypot(...P.map((v, ch) => v - color[ch])) > CLUSTER_TOL);
    const bg = localBackground(image, near, pageColorsForBg, others);
    // background colors, but not this family's own (a dot color can be common enough to count as a background)
    const pageColors = [...backgrounds, B].filter(P => Math.hypot(...P.map((v, ch) => v - color[ch])) > CLUSTER_TOL);
    const D = new Float32Array(n);
    for (let p = 0; p < n; p++) {
        const r = data[4 * p];
        const g = data[4 * p + 1];
        const b = data[4 * p + 2];
        if (!near[p] && pageColors.some(P => Math.hypot(r - P[0], g - P[1], b - P[2]) <= INK_TOL)) {
            D[p] = 0; // away from the family on a background color: no dot here
            continue;
        }
        // away from the family, another family's color is that family's pixel, not partial coverage of this one
        // (near it, pixels are read against their actual neighbor, which keeps the edges where dots touch)
        const own = Math.hypot(r - color[0], g - color[1], b - color[2]);
        if (!near[p] && others.some(O => { const d = Math.hypot(r - O[0], g - O[1], b - O[2]); return d <= CLUSTER_TOL && d < own; })) {
            D[p] = NaN;
            continue;
        }
        // near the family, against the nearest other color; away from it, against the family's usual background
        // (faint antialiased rims just beyond the near zone were left unknown and broke the dots' edges)
        const b0 = near[p] ? bg[3 * p] : B[0];
        const b1 = near[p] ? bg[3 * p + 1] : B[1];
        const b2 = near[p] ? bg[3 * p + 2] : B[2];
        const w0 = color[0] - b0;
        const w1 = color[1] - b1;
        const w2 = color[2] - b2;
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
    const line = colorLine(B, color);
    const table = [B, color];
    const isolated = isolatedDots(D, width, height);
    const radius = isolated.length ? median(isolated.map(d => d.r)) : radiusFromDistance(D, width, height);
    let dots = 0;
    let strongDots = 0;
    let coverage = 0;
    if (radiusPlausible(radius, isolated.length)) {
        const weight = new Uint8Array(n);
        for (let p = 0; p < n; p++)
            weight[p] = Number.isNaN(D[p]) ? 0 : 1;
        // a center counts if most of its edge is seen and its disc is filled (text: round glyphs are hollow)
        const rf = DOT_FILL_RADII * radius;
        const filled = c => {
            let inside = 0;
            let on = 0;
            for (let y = Math.max(0, Math.floor(c.y - rf)); y < Math.min(height, Math.ceil(c.y + rf)); y++) {
                for (let x = Math.max(0, Math.floor(c.x - rf)); x < Math.min(width, Math.ceil(c.x + rf)); x++) {
                    if ((x + 0.5 - c.x) ** 2 + (y + 0.5 - c.y) ** 2 > rf * rf)
                        continue;
                    inside++;
                    if (D[y * width + x] > 0.5)
                        on++;
                }
            }
            return inside > 0 && on >= DOT_MIN_FILL * inside;
        };
        const centers = houghCenters(D, weight, width, height, radius).filter(c => c.arc >= DOT_MIN_ARC && filled(c));
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
        let fam = 0;
        let covered = 0;
        for (let p = 0; p < n; p++) {
            if (D[p] > 0.5) {
                fam++;
                covered += inDisc[p];
            }
        }
        coverage = fam ? covered / fam : 0;
    }
    const pixels = mask.reduce((t, v) => t + v, 0);
    return {
        kind: 'opaque', seed: color, line, background: B, table, D, isolated, radius, dots, strongDots, coverage, pixels,
        levels: [{ t: Math.hypot(...B.map((v, ch) => v - color[ch])), color, count: pixels }],
        beyond: [], fit: { alpha: 1, c: [...color], rms: 0, maxAbs: 0, underdetermined: true }, translucent: false,
    };
}

function radiusPlausible(radius, isolatedCount) {
    return radius >= DOT_MIN_RADIUS || (radius >= SMALL_DOT_MIN_RADIUS && isolatedCount >= SMALL_DOT_MIN_ISOLATED);
}

function isDotFamily(f) {
    return radiusPlausible(f.radius, f.isolated.length) && f.strongDots >= DOT_MIN_STRONG
        && (f.coverage >= DOT_MIN_COVERAGE || f.isolated.length >= DOT_MIN_ISOLATED);
}

/**
 * Opaque dot families: color clusters that aren't a background, aren't levels of a translucent family, and show
 * at least DOT_MIN_COUNT dot-shaped edges at a radius of at least DOT_MIN_RADIUS.
 */
export function opaqueFamilies(image, backgrounds, translucent) {
    const onTranslucent = c => translucent.some(f => {
        const { B, u } = f.line;
        const e = B.map((v, ch) => v - c[ch]);
        const t = e[0] * u[0] + e[1] * u[1] + e[2] * u[2];
        return Math.hypot(e[0] - t * u[0], e[1] - t * u[1], e[2] - t * u[2]) <= CLUSTER_ON_LINE_TOL;
    });
    const { width, height, data } = image;
    const out = [];
    // a cluster between a family found already and that family's background (or a page color) is its antialiased
    // rim, not a family (synthetic swarm: maroon's blend with white made a "family")
    const isBlend = c => out.some(f => [f.background, ...backgrounds].some(P => {
        const w = f.seed.map((v, ch) => v - P[ch]);
        const ww = w[0] * w[0] + w[1] * w[1] + w[2] * w[2];
        const v = c.map((x, ch) => x - P[ch]);
        const m = (v[0] * w[0] + v[1] * w[1] + v[2] * w[2]) / ww;
        return m > 0.05 && m < 0.95 && Math.hypot(v[0] - m * w[0], v[1] - m * w[1], v[2] - m * w[2]) <= CLUSTER_ON_LINE_TOL;
    }));
    // Only the dominant color is excluded up front: a dot color can cover 5% of an image (sparse #3: 5.9%) and so
    // count as a background color; the dot test tells it from a panel.
    for (const cl of colorClusters(image, backgrounds.slice(0, 1))) {
        if (onTranslucent(cl.color) || isBlend(cl.color))
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
        if (!count || interior / count < DOT_MIN_INTERIOR)
            continue;
        const fam = measureOpaque(image, cl.color, backgrounds);
        fam.interior = interior / count;
        if (isDotFamily(fam))
            out.push(fam);
    }
    // Measure again with the other families' colors excluded from each, now that they are known, for the fit. The
    // first measurement decides what is a family: a small one surrounded by others (nicest language's red and dark
    // blue, about 10 dots each) can fail the test once its neighbors' colors are excluded.
    const colors = out.map(f => f.seed);
    const final = out.map(f => {
        const g = measureOpaque(image, f.seed, backgrounds, colors.filter(c => c !== f.seed));
        return { ...g, radius: g.radius ?? f.radius, dots: f.dots, strongDots: f.strongDots, coverage: f.coverage };
    });
    return final.sort((a, b) => b.dots - a.dots);
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
 * @returns {object}  { background, candidates, family, style, D, isolated, edge, notes }
 *   style: { color, alpha, levels, radius, blur } ready for render.js; radius and blur are null when no isolated
 *   dot was found. notes: human-readable caveats for display.
 */
export function estimateStyle(image, opts = {}) {
    const { width, height } = image;
    const background = backgroundColor(image);
    const B = background.color;
    const flat = flatMask(image);
    const backgrounds = backgroundColors(image);
    // candidates are the slow part (seconds on large images): reuse them when estimating another family
    const candidates = opts.candidates ?? findCandidates(image, B, flat, backgrounds);
    let fam;
    let familyIndex = opts.family ?? 0;
    if (opts.seed) {
        fam = measureFamily(image, B, opts.seed, flat);
        fam.kind = 'translucent';
        if (!fam.translucent)
            fam = measureOpaque(image, opts.seed, backgrounds);
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
    const D = countImage(fam.proj, fam.line, table);
    const isolated = isolatedDots(D, width, height);
    let radius = null;
    let blur = null;
    let edge = null;
    if (isolated.length) {
        radius = median(isolated.map(d => d.r));
        edge = edgeBlur(D, width, isolated);
        // the profile's σ includes the pixel box, which the renderer's exact coverage already supplies
        blur = Math.sqrt(Math.max(0, edge.sigma ** 2 - 1 / 12));
    }
    else {
        notes.push('No isolated whole dot found, so the radius and edge blur must be set by hand.');
    }
    return {
        background, candidates, family: familyIndex, flat,
        familyMeasure: fam,
        style: { color: fam.fit.c, alpha: fam.fit.alpha, levels: table, measuredLevels: fam.levels.length, radius, blur },
        D, isolated, edge, notes,
    };
}

/**
 * All dot families: translucent ones (levels along a color direction) first, then opaque ones (color clusters).
 */
export function findCandidates(image, B, flat, backgrounds) {
    const translucent = familyCandidates(image, B, flat).filter(c => c.translucent);
    for (const c of translucent)
        c.kind = 'translucent';
    return [...translucent, ...opaqueFamilies(image, backgrounds, translucent)];
}

// the estimate for an opaque family: its own background, radius from isolated dots or the distance transform, blur
// from isolated dots or the family's boundaries
function opaqueEstimate(image, fam, candidates, familyIndex, flat, pageBackground) {
    const { width, height } = image;
    const notes = ['Opaque dots: overlapping dots hide each other, so a dot entirely behind others can’t be found.'];
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
        style: { color: [...fam.seed], alpha: 1, levels: fam.table, measuredLevels: 1, radius: fam.radius, blur },
        D: fam.D, fitT, isolated: fam.isolated, edge, notes,
    };
}

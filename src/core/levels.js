// Translucent dot families (PLAN.md §2.2, stage 4): each family's overlap levels, alpha and full color. Phase 0
// (tools/measure-dots.mjs) established the approach on real images; this is its library form.
//
// A translucent family's pixels lie on the line from the background B toward its full color c, at plateaus
// L_n for n overlapping dots. Levels are found as histogram peaks of *flat* pixels projected on that line (edge
// pixels would make false peaks around small dots). Renderers composite in sRGB with constant alpha, but 8-bit
// rounding drifts with depth and some images have an odd first step, so the level colors used by the forward
// model are the measured ones; a geometric fit gives α and c and extrapolates beyond the deepest measured level.

import { sub, unit } from './image.js';
import { LEVEL_SEPARATION, EXACT_TOLERANCES, isThin } from './scan.js';

/** @typedef {import('./scan.js').Scan} Scan */
/** @typedef {import('./scan.js').Tolerances} Tolerances */

const PERP_TOL = 4; // 0–255 units: max distance off the color line for a level-voting pixel
// Near B every color is close to the line in absolute terms (antialiased gray text one step from white is within
// PERP_TOL of any line), so the distance off the line must also be small relative to the distance along it.
const PERP_REL_TOL = 0.12; // max perp / t
// For the count image, faint edge pixels near B must still count, or dots lose ink at their rims and read small,
// and a rim that is off the line makes a dot look partly hidden. Rims in dog-ages sit at perp/t ≈ 0.14 (perp
// 2.2–2.8 at t 13–20, from resampling); one-step-from-white gray text sits at 0.31.
const COUNT_PERP_REL_TOL = 0.2; // max perp / t for a pixel to get an overlap count
const COUNT_PERP_TOL = 12; // 0–255 units: max perp for a pixel to get an overlap count
const DIRECTION_MIN_T = 15; // 0–255 units: flat pixels at least this far from B vote for family directions
const DIRECTION_BIN = 2; // degrees, family direction histogram
const MAX_CANDIDATES = 6; // family directions examined
const LEVEL_BIN = 1; // 0–255 units along the line, level histogram
const LEVEL_MAX_T = 450; // 0–255 units; > the RGB cube diagonal
const PEAK_MIN_FRACTION = 0.004; // a level's peak bin must hold this fraction of the family's flat pixels
const PEAK_MIN_PIXELS = 20; // and at least this many
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
export const FIT_C_TOL = 10; // 0–255 units c may fall outside [0, 255] (fitting noise)
const EXTRA_LEVELS = 12; // geometric levels appended beyond the deepest measured one
const MAX_MISSING_LEVELS = 1; // overlap counts that may lack a level between measured ones (JPEG lenses)

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
 * @property {boolean} [inferred]  no flat pixel shows it: its color is the fit's (withMissingLevels), its count 0
 */

/**
 * A family measured as levels along a color line (measureFamily).
 *
 * @typedef {object} LevelMeasure
 * @property {'translucent'} kind  how it was measured (opaque.js measureOpaque's are 'opaque'), not whether it is
 *   a family: see `translucent`
 * @property {number[]} seed  0–255, the color the line was aimed at
 * @property {ColorLine} line
 * @property {{t: Float32Array, perp: Float32Array}} proj  every pixel's distance along the line and off it
 * @property {Level[]} levels  the overlap levels, nearest B first
 * @property {Level[]} beyond  plateaus past the last level (an opaque mark in the family's hue)
 * @property {{alpha: number, c: number[], rms: number, maxAbs: number}|null} fit  α and the full color c the levels
 *   give (fitGeometric); null without a level
 * @property {number} pixels  flat pixels at the levels
 * @property {boolean} translucent  two levels or more, and a fit a translucent family can have
 * @property {number[]} [band]  0–255: a band or box in the family's own hue that was read as a level, and set apart
 *   (style.js withoutBand, overBand, boxBand)
 * @property {boolean} [box]  the band is a box under the dots (style.js boxBand)
 */

/**
 * Where a vector lies against another: `m`, the multiple of w nearest v, and `off`, v's distance from that point.
 * With v and w two colors measured from a third, it is the test for a mix: on the line (off small) and between the
 * two (m from 0 to 1). The rules that ask it each have their own range for m.
 *
 * @param {number[]} v  0–255 units
 * @param {number[]} w  0–255 units, not zero
 * @returns {{m: number, off: number}}  off in 0–255 units
 */
export function alongLine(v, w) {
    const ww = w[0] * w[0] + w[1] * w[1] + w[2] * w[2];
    const m = (v[0] * w[0] + v[1] * w[1] + v[2] * w[2]) / ww;
    return { m, off: Math.hypot(v[0] - m * w[0], v[1] - m * w[1], v[2] - m * w[2]) };
}

/**
 * The line from B toward a seed color.
 *
 * @param {number[]} B  0–255
 * @param {number[]} seed  0–255
 * @returns {ColorLine}
 */
export function colorLine(B, seed) {
    return { B, u: unit(sub(B, seed)) };
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
 * @param {Scan|{image: object, flat: Uint8Array, tol?: Tolerances}} scan  its image, flat mask and tolerances
 *   (default EXACT_TOLERANCES)
 * @param {{t: Float32Array, perp: Float32Array}} proj
 * @returns {{levels: Level[], beyond: Level[]}}
 */
export function findLevels(scan, proj) {
    const { image, flat, tol = EXACT_TOLERANCES } = scan;
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
    // A first level fills single dots' interiors, so it is thick; one that is only lines is the rims of the next
    // level's dots or other thin ink in their hue (step1 with its bands divided out: their edges, left as they were,
    // and the dots' rims at 206, 222, 246, 94 flat pixels, ahead of the dots' own 149, 185, 233, 374, which the step
    // ratio then put beyond the sequence)
    while (found.length > 1 && isThin(scan, found[0].color))
        found.shift();
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
 * @param {Scan} scan  its image, page color, flat mask and tolerances
 * @returns {LevelMeasure[]}  the candidates
 */
export function familyCandidates(scan) {
    const { image, B, flat } = scan;
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
        candidates.push(measureFamily(scan, seed));
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
 * @param {Scan|{image: object, B: number[], flat: Uint8Array, tol?: Tolerances}} scan  its image, the color the line
 *   starts from (B), the flat mask and the tolerances; a rule measuring without some pixels passes
 *   { ...scan, flat: rest }
 * @param {number[]} seed  0–255
 * @returns {LevelMeasure}
 */
export function measureFamily(scan, seed) {
    const { image, B } = scan;
    const line = colorLine(B, seed);
    const proj = projectOnLine(image, line);
    const found = findLevels(scan, proj);
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
        kind: 'translucent', seed, line, proj, levels, beyond: found.beyond, fit,
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
 * @returns {{levels: Level[], fit: {alpha: number, c: number[], rms: number, maxAbs: number}}|null}  fit: fitGeometric's
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

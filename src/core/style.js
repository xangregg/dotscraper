// Dot style estimation (PLAN.md §2.2, stage 4): the background, the translucent dot families, each family's
// overlap levels, alpha, full color, radius and edge blur. Phase 0 (tools/measure-dots.mjs) established the
// approach on real images; this is its library form.
//
// A translucent family's pixels lie on the line from the background B toward its full color c, at plateaus
// L_n for n overlapping dots. Levels are found as histogram peaks of *flat* pixels projected on that line (edge
// pixels would make false peaks around small dots). Renderers composite in sRGB with constant alpha, but 8-bit
// rounding drifts with depth and some images have an odd first step, so the level colors used by the forward
// model are the measured ones; a geometric fit gives α and c and extrapolates beyond the deepest measured level.

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
// A translucent family must fit the geometric model with a full color c a renderer could use. Gray text with two
// shades can mimic a level sequence, but its fitted c falls far outside the RGB cube.
const FIT_MAX_RMS = 2; // 0–255 units
const FIT_C_TOL = 10; // 0–255 units c may fall outside [0, 255] (fitting noise)
const EXTRA_LEVELS = 12; // geometric levels appended beyond the deepest measured one
const ISOLATED_MAX_LEVEL = 1.25; // components never above this count are single dots
const ISOLATED_MIN_AREA = 30; // px²
const ISOLATED_MAX_ASPECT = 1.05; // ratio of principal second-moment radii
const ISOLATED_MAX_MOMENT_RATIO = 1.03; // r(moment) / r(area); 1 for a uniform disc
const EDGE_RING = 4; // px outside a dot sampled for its edge
const PROFILE_BIN = 0.25; // px, radial edge profile bins

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
    const candidates = familyCandidates(image, B, flat);
    let fam;
    let familyIndex = opts.family ?? 0;
    if (opts.seed) {
        fam = measureFamily(image, B, opts.seed, flat);
        familyIndex = -1;
    }
    else {
        fam = candidates[familyIndex];
    }
    const notes = [];
    if (!fam || !fam.levels.length)
        return { background, candidates, family: familyIndex, style: null, D: null, isolated: [], edge: null, notes: ['No dot family found.'] };
    if (!fam.translucent)
        notes.push('Only one level found: the dots look opaque, so overlaps can’t be counted from color.');
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

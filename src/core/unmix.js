// Several translucent families over one background (PLAN.md phase 5): separate each pixel into one transmittance
// per family, so each family can be fitted alone on an image of itself.
//
// Renderers composite dot by dot in drawing order, and that order interleaves the families (JMP draws rows in
// order: on bench1 and bench2's group charts the later row is on top at every one of 44 red–blue overlaps). With
// one dot of A and one of B over a pixel, the color is
//
//     U − B = w_A (c_A − B) + w_B (c_B − B),
//
// with weights (w_A, w_B) = (α_A·m_A, (1 − α_A·m_A)·α_B·m_B) if B is on top, and the mirror image if A is. So the
// weights solve a 3 × 2 least-squares problem from the pixel alone (the colors are not collinear with B: red and blue
// over white), and either order turns them into transmittances: B on top gives T_B = 1 − w_B, T_A = 1 − w_A / T_B.
// Only the true order makes the transmittances whole powers of 1 − α inside a lens (one dot each: 0.5 and 0.5
// against 0.75 and 0.33 at α 0.5), so each pixel takes the order closer to whole dot counts; at rims, where the
// orders differ less, either is close.
//
// Pixels on one family's color line belong to it alone (the others' transmittance is 1); pixels no single family
// or pair explains (text, lines, other marks) are left as they are, and each family's mask covers them as before.
// Stacks of three or more dots in both colors (red, blue, red) fit no single order. bench2 group's last 5 misses
// were such stacks within 4 px; they were found once flat pixels had to be whole counts (FLAT_MAX_FRACTION), which
// masks the stacks' fractional plateaus. Why that let the fit place them wasn't examined.

import { colorDistance, dot } from './image.js';
import { flatMask, EXACT_TOLERANCES } from './scan.js';
import { isolatedDots, edgeBlur } from './shape.js';
import { median } from './stats.js';
import { countImage, projectOnLine, colorLine, findLevels } from './levels.js';
import { shapeRadius } from './opaque.js';
import { NO_ISOLATED_NOTE } from './style.js';

const PAIR_MAX_RESIDUAL = 6; // 0–255 units (RGB distance): a pair explains a pixel this well or not at all
const ONE_FAMILY_PERP_REL = 0.2; // a pixel this close to a family's line (perp / t) …
const ONE_FAMILY_PERP_ABS = 4; // … or within this many units of it, belongs to that family alone
const WEIGHT_SLACK = 0.03; // weights may fall this far outside [0, 1] (rounding and blur)
// A blue dot over a light steel-blue CI band lies close enough to the blue line to read as 1.5 blue dots, and those
// counts made 20 phantom blue dots (bench2 group-smoother-ci). Where a family's dots overlap, the counts are whole;
// only at rims, where the color changes from pixel to pixel, are they fractional. So a flat pixel given to one
// family must be this close to a whole count, or it is left to the masks. (The same test on pixels a pair explains
// changed nothing on the bench.)
const FLAT_MAX_FRACTION = 0.3; // dots
const MIN_T = 1e-3;
// A family's dots over a box under them are read against levels measured there (regionLevels): renderers that
// don't composite in sRGB put a dot over a box in its own hue far from c + (B₂ − c)(1 − α) (protrusion3: 116, 163,
// 122 against 128, 167, 127 predicted, which read as 1.56 dots and was masked as fractional). The first level is the
// commonest flat color in the region within this many predicted steps of the prediction, with at least this many pixels.
const REGION_SEED_STEPS = 0.5;
const REGION_SEED_MIN_PIXELS = 50;
const REGION_MIN_LEVEL_SHARE = 0.5; // of the first level's distance from B₂: nearer colors aren't read as levels
const MAX_EXTRA_COUNT = 20; // dots beyond the deepest measured level, where the steps run out

// distance of a transmittance from a whole number of dots: |n − round(n)| with n = ln T / ln q
function fractional(T, q) {
    const n = Math.log(Math.max(MIN_T, T)) / Math.log(q);
    return Math.abs(n - Math.round(n));
}

/**
 * Per-family transmittances.
 *
 * @param {import('./image.js').RgbaImage} image
 * @param {number[]} B  the common background color, 0–255
 * @param {{color: number[], alpha: number}[]} families  translucent families: full color c (0–255), α (0–1)
 * @param {Uint8Array} [flat]  1 where the pixel's neighborhood is flat (style.flatMask); a flat pixel on one family's
 *   line must be a whole number of its dots
 * @param {{colors: number[][], index: Uint8Array, levels?: ({unit: number[], ts: number[]}|null)[][]}} [under]  a
 *   background that varies under the dots: pixel p's is colors[index[p]] (background.js, regions under the dots);
 *   default B everywhere. levels[k][j]: family j's levels measured over colors[k] (regionLevels), read instead of
 *   the geometric model there
 * @returns {{T: Float32Array[], mixed: Uint8Array}}  T per family (1 where it has no dot); mixed = 1 where a pair
 *   explained the pixel, 2 where nothing did (the pixel is left to each family's mask)
 */
export function unmixFamilies(image, B, families, flat = null, under = null) {
    const { width, height, data } = image;
    const n = width * height;
    const F = families.length;
    const T = families.map(() => new Float32Array(n).fill(1));
    const mixed = new Uint8Array(n);
    const q = families.map(f => 1 - f.alpha);
    // each family's offset from each background, and the pair normal equations, precomputed per background
    const geometry = (Bk, i) => {
        const dirs = families.map(f => f.color.map((v, ch) => v - Bk[ch])); // c − B_k
        const span = dirs.map(d => Math.hypot(...d));
        const measured = families.map((f, k) => under?.levels?.[i]?.[k] ?? null);
        const unit = dirs.map((d, k) => measured[k]?.unit ?? d.map(v => v / span[k]));
        const pairs = [];
        for (let a = 0; a < F; a++) {
            for (let b = a + 1; b < F; b++) {
                const aa = dot(dirs[a], dirs[a]);
                const bb = dot(dirs[b], dirs[b]);
                const ab = dot(dirs[a], dirs[b]);
                const det = aa * bb - ab * ab;
                if (Math.abs(det) > 1e-6 * aa * bb)
                    pairs.push({ a, b, aa, bb, ab, det });
            }
        }
        return { Bk, dirs, span, unit, measured, pairs };
    };
    const geometries = (under?.colors ?? [B]).map(geometry);
    const e = [0, 0, 0];
    for (let p = 0; p < n; p++) {
        const { Bk, dirs, span, unit, measured, pairs } = geometries[under ? under.index[p] : 0];
        e[0] = data[4 * p] - Bk[0];
        e[1] = data[4 * p + 1] - Bk[1];
        e[2] = data[4 * p + 2] - Bk[2];
        const len2 = dot(e, e);
        if (len2 < 1)
            continue;
        // one family: the nearest line the pixel is on
        let best = -1;
        let bestPerp = Infinity;
        for (let k = 0; k < F; k++) {
            const t = dot(e, unit[k]);
            if (t <= 0)
                continue;
            const perp = Math.sqrt(Math.max(0, len2 - t * t));
            if ((perp <= ONE_FAMILY_PERP_ABS || perp <= ONE_FAMILY_PERP_REL * t) && perp < bestPerp) {
                best = k;
                bestPerp = perp;
            }
        }
        if (best >= 0) {
            const m = measured[best];
            const Tb = m ? q[best] ** countAt(dot(e, unit[best]), m.ts, q[best]) : Math.max(0, 1 - dot(e, unit[best]) / span[best]);
            if (flat && flat[p] && fractional(Tb, q[best]) > FLAT_MAX_FRACTION) {
                mixed[p] = 2;
                continue;
            }
            T[best][p] = Tb;
            continue;
        }
        // two families
        let pick = null;
        for (const pr of pairs) {
            const ea = dot(e, dirs[pr.a]);
            const eb = dot(e, dirs[pr.b]);
            const wa = (pr.bb * ea - pr.ab * eb) / pr.det;
            const wb = (pr.aa * eb - pr.ab * ea) / pr.det;
            if (wa < -WEIGHT_SLACK || wb < -WEIGHT_SLACK || wa + wb > 1 + WEIGHT_SLACK)
                continue;
            let r2 = 0;
            for (let ch = 0; ch < 3; ch++)
                r2 += (e[ch] - wa * dirs[pr.a][ch] - wb * dirs[pr.b][ch]) ** 2;
            if (r2 <= PAIR_MAX_RESIDUAL * PAIR_MAX_RESIDUAL && (!pick || r2 < pick.r2))
                pick = { pr, wa: Math.max(0, wa), wb: Math.max(0, wb), r2 };
        }
        if (!pick) {
            mixed[p] = 2;
            continue;
        }
        const { pr, wa, wb } = pick;
        // b on top: T_b = 1 − w_b, T_a = 1 − w_a / T_b; a on top: the mirror
        const tbTop = Math.max(MIN_T, 1 - wb);
        const taUnder = Math.min(1, Math.max(0, 1 - wa / tbTop));
        const taTop = Math.max(MIN_T, 1 - wa);
        const tbUnder = Math.min(1, Math.max(0, 1 - wb / taTop));
        const bTopScore = fractional(tbTop, q[pr.b]) + fractional(taUnder, q[pr.a]);
        const aTopScore = fractional(taTop, q[pr.a]) + fractional(tbUnder, q[pr.b]);
        if (bTopScore <= aTopScore) {
            T[pr.a][p] = taUnder;
            T[pr.b][p] = tbTop;
        }
        else {
            T[pr.a][p] = taTop;
            T[pr.b][p] = tbUnder;
        }
        mixed[p] = 1;
    }
    return { T, mixed };
}

/**
 * A family's levels over a region under the dots, measured: the commonest flat color of the region near where one dot
 * over it should be (REGION_SEED_STEPS), then levels along the line from B₂ through it. Null when none shows.
 * O(pixels).
 *
 * @param {import('./image.js').RgbaImage} image
 * @param {Uint8Array} flat  style.flatMask
 * @param {Uint8Array} index  each pixel's background index (unmixFamilies' under.index)
 * @param {number} k  the region's index
 * @param {number[]} B2  its color, 0–255
 * @param {{color: number[], alpha: number}} family  full color (0–255) and α (0–1)
 * @returns {{unit: number[], ts: number[]}|null}  the direction from B₂ and each level's distance along it (0–255
 *   units)
 */
export function regionLevels(image, flat, index, k, B2, family) {
    const { data } = image;
    const pred = family.color.map((v, ch) => v + (B2[ch] - v) * (1 - family.alpha));
    const step = colorDistance(pred, B2);
    const own = new Uint8Array(flat.length);
    const counts = new Map();
    for (let p = 0; p < flat.length; p++) {
        if (!flat[p] || index[p] !== k)
            continue;
        own[p] = 1;
        if (Math.hypot(data[4 * p] - pred[0], data[4 * p + 1] - pred[1], data[4 * p + 2] - pred[2]) > REGION_SEED_STEPS * step)
            continue;
        const key = (data[4 * p] << 16) | (data[4 * p + 1] << 8) | data[4 * p + 2];
        counts.set(key, (counts.get(key) ?? 0) + 1);
    }
    const best = [...counts].sort((a, b) => b[1] - a[1])[0];
    if (!best || best[1] < REGION_SEED_MIN_PIXELS)
        return null;
    const seed = [best[0] >> 16, (best[0] >> 8) & 255, best[0] & 255];
    const line = colorLine(B2, seed);
    const proj = projectOnLine(image, line);
    // other colors near B₂ are no dot over it (protrusion3: two dots on the page, 142, 178, 142, 12 units from the box,
    // in the region where dots cross its edge, read as level 1 at a quarter of the seed's distance)
    const seedT = colorDistance(seed, B2);
    for (let p = 0; p < own.length; p++) {
        if (proj.t[p] < REGION_MIN_LEVEL_SHARE * seedT)
            own[p] = 0;
    }
    const { levels } = findLevels({ image, flat: own }, proj);
    if (!levels.length)
        return null;
    return {
        unit: line.u.map(v => -v),
        ts: levels.map(L => L.color.reduce((t, v, ch) => t + (B2[ch] - v) * line.u[ch], 0)),
    };
}

// overlap count at distance t from B₂ along measured levels ts: linear between levels, then steps shrinking by q
function countAt(t, ts, q) {
    if (t <= ts[0])
        return Math.max(0, t / ts[0]);
    for (let i = 1; i < ts.length; i++) {
        if (t <= ts[i])
            return i + (t - ts[i - 1]) / (ts[i] - ts[i - 1]);
    }
    const m = ts.length;
    const last = m > 1 ? ts[m - 1] - ts[m - 2] : ts[0];
    const left = 1 - (t - ts[m - 1]) * (1 - q) / (last * q);
    return m + (left > MIN_T ? Math.log(left) / Math.log(q) : MAX_EXTRA_COUNT);
}

/**
 * The image as if only family k were drawn: its transmittance composited over B where the pixel was explained, the
 * pixel unchanged where it wasn't (so the family's own off-line test still masks it).
 *
 * @param {import('./image.js').RgbaImage} image
 * @param {number[]} B
 * @param {{color: number[]}} family
 * @param {Float32Array} T  this family's transmittance (unmixFamilies)
 * @param {Uint8Array} mixed
 * @returns {import('./image.js').RgbaImage}
 */
export function familyImage(image, B, family, T, mixed) {
    const { width, height, data } = image;
    const out = new Uint8ClampedArray(data.length);
    const c = family.color;
    for (let p = 0; p < width * height; p++) {
        if (mixed[p] === 2) {
            out.set(data.subarray(4 * p, 4 * p + 4), 4 * p);
            continue;
        }
        for (let ch = 0; ch < 3; ch++)
            out[4 * p + ch] = Math.round(c[ch] + (B[ch] - c[ch]) * T[p]);
        out[4 * p + 3] = data[4 * p + 3];
    }
    return { width, height, data: out };
}

/**
 * Each family's estimate with the image it should be fitted on. Translucent families get their own image
 * (familyImage) and the overlap count recomputed on it when there are two or more of them (pixels where they
 * overlap count instead of being masked) or when part of the background under the dots is another color (a dot
 * there is read against that color). Opaque families, which read coverage against their local background already,
 * keep the image as it is.
 *
 * @param {import('./image.js').RgbaImage} image
 * @param {import('./style.js').Estimate[]} estimates  one for each family (one without a style is passed through)
 * @param {import('./scan.js').Tolerances} [tol]  as the estimates used
 * @param {{regions: {color: number[], mode: string}[], label: Uint8Array}|null} [layers]  background.js
 * @returns {import('./style.js').Estimate[]}  the estimates with `image` added (and D and notes replaced where
 *   separated)
 */
export function unmixEstimates(image, estimates, tol = EXACT_TOLERANCES, layers = null) {
    const translucent = estimates.map((e, k) => (e.style && e.familyMeasure?.kind !== 'opaque' && e.style.alpha < 1 ? k : -1)).filter(k => k >= 0);
    const underRegions = (layers?.regions ?? []).map((r, i) => (r.mode === 'under' ? i : -1)).filter(i => i >= 0);
    if (translucent.length === 0 || (translucent.length < 2 && !underRegions.length))
        return estimates.map(e => ({ ...e, image }));
    const B = estimates[translucent[0]].familyMeasure.line.B;
    let under = null;
    if (underRegions.length) {
        const index = new Uint8Array(layers.label.length);
        for (let p = 0; p < index.length; p++) {
            const r = layers.label[p] - 1;
            index[p] = r >= 0 && layers.regions[r].mode === 'under' ? r + 1 : 0;
        }
        under = { colors: [B, ...layers.regions.map(r => r.color)], index };
    }
    const flat = estimates[translucent[0]].flat ?? flatMask(image, tol);
    if (under) {
        // (boxes only: over landing1's violins, which the sRGB model reads, measured levels gave 87 found for 70, not 75)
        under.levels = under.colors.map((Bk, k) => (layers.regions[k - 1]?.mode === 'under' && layers.regions[k - 1].box
            ? translucent.map(j => regionLevels(image, flat, under.index, k, Bk, estimates[j].style))
            : null));
    }
    const { T, mixed } = unmixFamilies(image, B, translucent.map(k => estimates[k].style), flat, under);
    const others = translucent.length - 1;
    const notes = [];
    if (others)
        notes.push(`Overlaps with the other ${others === 1 ? 'translucent family are' : `${others} translucent families are`} separated pixel by pixel; the counts shown are this family’s alone.`);
    for (const i of underRegions) {
        const c = layers.regions[i].color.map(Math.round).join(', ');
        notes.push(layers.regions[i].box
            ? `A box of color ${c} lies under the dots; dots over it are read against the levels they show there.`
            : `A region of color ${c} lies under the dots; dots there are read against it.`);
    }
    return estimates.map((e, k) => {
        const j = translucent.indexOf(k);
        if (j < 0)
            return { ...e, image };
        const own = familyImage(image, B, e.style, T[j], mixed);
        const line = e.familyMeasure.line;
        const D = countImage(projectOnLine(own, line), line, e.style.levels, tol);
        // with what is under them read per pixel, dots over a region can be whole ones: the radius, if it wasn't
        // known (landing1: nearly every dot over a violin of its own hue; there no dot is whole even so, and the
        // radius is the one at which the dots' edges show best)
        // (no ring is measured here, so a piece deeper in its outer part is no whole dot: style.js estimateStyle)
        if (!e.style.radius) {
            const isolated = isolatedDots(D, image.width, image.height, tol).filter(d => !d.ringed);
            if (isolated.length) {
                const edge = edgeBlur(D, image.width, isolated);
                const radius = median(isolated.map(d => d.r));
                const style = { ...e.style, radius, blur: Math.sqrt(Math.max(0, edge.sigma ** 2 - 1 / 12)) };
                return { ...e, image: own, D, style, isolated, edge, notes: [...e.notes.filter(n => n !== NO_ISOLATED_NOTE), ...notes] };
            }
            // only for dots over a band of their own hue, where nothing else gives a radius (elsewhere it took a
            // family the user must measure for one: protrusion3, 120 found for 50)
            const shape = e.familyMeasure.band ? shapeRadius(D, image.width, image.height) : null;
            if (shape) {
                const note = `No isolated whole dot: the radius, ${shape.radius.toFixed(1)} px, is the one at which most dots' edges show as circles (${shape.centers} dots, half their edge or more), and the blur is read from the dots' boundaries. Check it against a few dots.`;
                return { ...e, image: own, D, style: { ...e.style, radius: shape.radius, blur: shape.blur }, notes: [...e.notes.filter(n => n !== NO_ISOLATED_NOTE), note, ...notes] };
            }
        }
        return { ...e, image: own, D, notes: [...e.notes, ...notes] };
    });
}

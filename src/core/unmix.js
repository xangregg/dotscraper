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

import { countImage, flatMask, projectOnLine, EXACT_TOLERANCES } from './style.js';

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

function dot(a, b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

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
 * @returns {{T: Float32Array[], mixed: Uint8Array}}  T per family (1 where it has no dot); mixed = 1 where a pair
 *   explained the pixel, 2 where nothing did (the pixel is left to each family's mask)
 */
export function unmixFamilies(image, B, families, flat = null) {
    const { width, height, data } = image;
    const n = width * height;
    const F = families.length;
    const T = families.map(() => new Float32Array(n).fill(1));
    const mixed = new Uint8Array(n);
    const dirs = families.map(f => f.color.map((v, ch) => v - B[ch])); // c − B
    const span = dirs.map(d => Math.hypot(...d));
    const unit = dirs.map((d, k) => d.map(v => v / span[k]));
    const q = families.map(f => 1 - f.alpha);
    // pair normal equations, precomputed
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
    const e = [0, 0, 0];
    for (let p = 0; p < n; p++) {
        e[0] = data[4 * p] - B[0];
        e[1] = data[4 * p + 1] - B[1];
        e[2] = data[4 * p + 2] - B[2];
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
            const Tb = Math.max(0, 1 - dot(e, unit[best]) / span[best]);
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
 * Each family's estimate with the image it should be fitted on. With two or more translucent families, each of them
 * gets its unmixed image (familyImage) and the overlap count recomputed on it, so pixels where it overlaps another
 * family count instead of being masked; opaque families, and every family when fewer than two are translucent, keep
 * the image as it is.
 *
 * @param {import('./image.js').RgbaImage} image
 * @param {object[]} estimates  style.estimateStyle for each family (one without a style is passed through)
 * @param {import('./style.js').Tolerances} [tol]  as the estimates used
 * @returns {object[]}  the estimates with `image` added (and D and notes replaced where unmixed)
 */
export function unmixEstimates(image, estimates, tol = EXACT_TOLERANCES) {
    const translucent = estimates.map((e, k) => (e.style && e.familyMeasure?.kind !== 'opaque' && e.style.alpha < 1 ? k : -1)).filter(k => k >= 0);
    if (translucent.length < 2)
        return estimates.map(e => ({ ...e, image }));
    const B = estimates[translucent[0]].familyMeasure.line.B;
    const { T, mixed } = unmixFamilies(image, B, translucent.map(k => estimates[k].style), estimates[translucent[0]].flat ?? flatMask(image, tol));
    return estimates.map((e, k) => {
        const j = translucent.indexOf(k);
        if (j < 0)
            return { ...e, image };
        const own = familyImage(image, B, e.style, T[j], mixed);
        const line = e.familyMeasure.line;
        return {
            ...e, image: own,
            D: countImage(projectOnLine(own, line), line, e.style.levels, tol),
            notes: [...e.notes, `Overlaps with the other ${translucent.length - 1 === 1 ? 'translucent family are' : `${translucent.length - 1} translucent families are`} separated pixel by pixel; the counts shown are this family’s alone.`],
        };
    });
}

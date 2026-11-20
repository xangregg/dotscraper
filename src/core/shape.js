// Dots as shapes in a coverage or count image D: connected components, isolated whole dots and their rings, and the
// blur of edges. Shared by translucent families (levels.js) and opaque ones (opaque.js).

import { distanceTransform } from './layers.js';
import { pixelCoverage } from './render.js';
import { EXACT_TOLERANCES } from './scan.js';

/** @typedef {import('./scan.js').Tolerances} Tolerances */

const ISOLATED_MAX_LEVEL = 1.25; // components never above this count are single dots
export const ISOLATED_MIN_AREA = 12; // px² (JMP's tiny dots: r 2.6, area 21)
export const ISOLATED_MAX_ASPECT = 1.05; // ratio of principal second-moment radii
export const ISOLATED_MAX_MOMENT_RATIO = 1.03; // r(moment) / r(area); 1 for a uniform disc
// Outlined dots (bench1 stroked-dots: a light gray fill, a darker ring 2 px wide at the rim) read deeper than one dot
// at the ring, so none passed as isolated. A component above ISOLATED_MAX_LEVEL is still one dot if every such pixel
// lies in its outer part, beyond this fraction of the radius from the center.
const RING_MIN_RADIUS_FRACTION = 0.6;
const RING_INNER_RANGE = [0.5, 0.97]; // the ring's inner radius searched, × the radius
const RING_INNER_STEP = 0.01;
const RING_MIN_ALPHA = 0.05; // extra opacity below this is no ring
const EDGE_RING = 4; // px outside a dot sampled for its edge
const PROFILE_BIN = 0.25; // px, radial edge profile bins

const BOUNDARY_PROFILE_REACH = 4; // px either side of a boundary for the edge-blur fit

/**
 * Connected components (4-neighbor) of pixels with D > 0.5.
 *
 * @param {ArrayLike<number>} D
 * @param {number} width
 * @param {number} height
 * @returns {number[][]}  each component's pixel indices
 */
export function components(D, width, height) {
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

/**
 * Edge blur from a family's boundaries, for families without isolated dots: D against the signed distance to the
 * D = ½ boundary, pooled within BOUNDARY_PROFILE_REACH px, fitted with the same erf as edgeBlur.
 */
export function boundaryBlur(D, width, height) {
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

// share of the pixels whose centers lie within r of (cx, cy) where v > 0.5
export function discShare(v, width, height, cx, cy, r) {
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

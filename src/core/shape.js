// Dots as shapes in a coverage or count image D: connected components, isolated whole dots and their rings, and the
// blur of edges. Shared by translucent families (levels.js) and opaque ones (opaque.js).

import { distanceTransform } from './layers.js';
import { pixelCoverage, gaussianKernel, blurRect } from './render.js';
import { EXACT_TOLERANCES } from './scan.js';

/** @typedef {import('./scan.js').Tolerances} Tolerances */

const ISOLATED_MAX_LEVEL = 1.25; // components never above this count are single dots
export const ISOLATED_MIN_AREA = 12; // px² (JMP's tiny dots: r 2.6, area 21)
export const ISOLATED_MAX_ASPECT = 1.05; // ratio of principal second-moment radii
export const ISOLATED_MAX_MOMENT_RATIO = 1.03; // r(moment) / r(area); 1 for a uniform disc
// A blurred disc's soft rim spreads its moments and not its area: biomarker's dots, r 6.0 with an edge 4 px wide, have
// a ratio of 1.07 and none passed, so the radius came from the distance transform, 4.5, and the fit put two or three
// dots in each. Cut at half coverage, a blurred disc is the disc again (ratio 1.00) of the same area (r 5.9), which
// a dot on something faint (a halo, a lighter disc under it) is not: its soft area is the larger. Asked for by the caller, where no
// dot passes as a crisp one: counted always, round specks of text passed too and outvoted the dots (adults1: no
// family), and four more images' counts moved.
const ISOLATED_HALF_LEVEL = 0.5; // counts
// A square's second moments are a disc's within 2.3%, under ISOLATED_MAX_MOMENT_RATIO, but its corners reach 1.25 r
// from its middle (r the radius of a disc of its area): a component with pixels farther out than ISOLATED_MAX_REACH
// toward each of the four corners of its box is no dot. aerobic: a legend of squares 23 px wide in the dots' colors;
// four of its seven colors had no other isolated shape, took 13.8 px for their radius (the dots: 8.6), and were no
// families. All four: far pixels to one side or two are a bump on a dot, which the moments judge. A square under
// 12 px or so can't be told from a disc this way.
const ISOLATED_MAX_REACH = [1.05, 0.75]; // × r, plus px
const ISOLATED_BLUR_RADIUS_TOL = 0.05; // × r, between the radius of the soft area and of the area at half coverage
// Outlined dots (bench1 stroked-dots: a light gray fill, a darker ring 2 px wide at the rim) read deeper than one dot
// at the ring, so none passed as isolated. A component above ISOLATED_MAX_LEVEL is still one dot if every such pixel
// lies in its outer part, beyond this fraction of the radius from the center. The ring need not be at the rim:
// matplotlib strokes a translucent dot's edge in the fill's color at the fill's alpha, half of the stroke over the
// fill and half beyond it, so the dot is two layers deep in a ring inside its rim (lung-tumor: a radius of 6.8 px,
// deeper from 0.42 to 0.50 of it in 43 dots; at 0.6 none was whole and the family had no radius).
const RING_MIN_RADIUS_FRACTION = 1 / 3;
const RING_RANGE = [0.3, 1]; // the ring's inner and outer radius searched, × the radius
const RING_STEP = 0.01;
const RING_MIN_WIDTH = 0.03; // × the radius
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
 * off-line pixel nearby (a dot touching a line is partly hidden), and disc-shaped by their second moments, or, with
 * `blurred`, by those of their pixels at half coverage or more. Centroid and area weight pixels by min(D, 1), so
 * the soft rim counts fractionally.
 *
 * @param {Float32Array} D
 * @param {number} width
 * @param {number} height
 * @param {Tolerances} [tol]
 * @param {boolean} [blurred]  blurred discs count too (ISOLATED_HALF_LEVEL)
 * @returns {{cx: number, cy: number, r: number, aspect: number, box: number[], ringed: boolean}[]}  ringed: deeper
 *   than one dot only in its outer part (an outline)
 */
export function isolatedDots(D, width, height, tol = EXACT_TOLERANCES, blurred = false) {
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
        let halfArea = 0; // px, at half coverage or more, and their summed squared distance from the centroid
        let halfTr = 0;
        for (let y = y0 - m; y <= y1 + m; y++) {
            for (let x = x0 - m; x <= x1 + m; x++) {
                const d = D[y * width + x];
                const w = d < tol.isolatedFloor ? 0 : Math.min(1, d);
                const dx = x + 0.5 - cx;
                const dy = y + 0.5 - cy;
                mxx += w * dx * dx;
                myy += w * dy * dy;
                mxy += w * dx * dy;
                if (d >= ISOLATED_HALF_LEVEL) {
                    halfArea++;
                    halfTr += dx * dx + dy * dy;
                }
            }
        }
        mxx /= a;
        myy /= a;
        mxy /= a;
        const tr = mxx + myy;
        const disc = Math.sqrt(Math.max(0, tr * tr / 4 - (mxx * myy - mxy * mxy)));
        const aspect = Math.sqrt((tr / 2 + disc) / (tr / 2 - disc));
        const r = Math.sqrt(a / Math.PI);
        const halfR = Math.sqrt(halfArea / Math.PI);
        // the component's own pixels: the box's margin can hold a neighbor's rim (curve preset: dots a pixel apart)
        const reach = ISOLATED_MAX_REACH[0] * r + ISOLATED_MAX_REACH[1];
        const corners = [false, false, false, false];
        for (const p of comp) {
            const x = p % width;
            const dx = x + 0.5 - cx;
            const dy = (p - x) / width + 0.5 - cy;
            if (Math.hypot(dx, dy) > reach)
                corners[(dx > 0 ? 1 : 0) + (dy > 0 ? 2 : 0)] = true;
        }
        const blurredDisc = blurred && Math.sqrt(2 * halfTr / halfArea) / halfR <= ISOLATED_MAX_MOMENT_RATIO && Math.abs(r - halfR) <= ISOLATED_BLUR_RADIUS_TOL * r;
        if (aspect > ISOLATED_MAX_ASPECT || (Math.sqrt(2 * tr) / r > ISOLATED_MAX_MOMENT_RATIO && !blurredDisc) || corners.every(Boolean))
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
 * An outline's inner and outer radius (× radius) and extra opacity, fitted to isolated dots' pixels along the color
 * line: a dot is tC·(α·m + a·m_ring), blurred, with m its coverage and m_ring the annulus's (render.js style.ring).
 * For each pair of radii the opacity a is linear least squares, and the pair with the least squared error wins.
 *
 * Every candidate radius's disc is rendered once per dot and blurred (the blur is linear, so a ring is the difference
 * of two discs), and the sums a pair needs come from the discs' products with the pixels and with each other:
 * O(radii · pixels) coverages and O(radii² · pixels) multiplications, about 2400 pairs.
 *
 * @param {Float32Array} t  0–255 units along the line, per pixel
 * @param {number} width
 * @param {number} height
 * @param {{cx: number, cy: number}[]} dots  isolated dots
 * @param {number} radius  px
 * @param {number} alpha  the fill's
 * @param {number} tC  0–255 units: t of the full color
 * @param {number} [blur]  px, sd of the image's blur (the renderer's)
 * @returns {{inner: number, outer: number, alpha: number}|null}  null when there is no ring
 */
export function measureRing(t, width, height, dots, radius, alpha, tC, blur = 0) {
    const kernel = blur > 0 ? gaussianKernel(blur) : null;
    const half = Math.ceil(radius) + 1 + (kernel ? (kernel.length - 1) / 2 : 0);
    const side = 2 * half + 1;
    const fractions = [];
    for (let f = RING_RANGE[0]; f < RING_RANGE[1] - 1e-9; f += RING_STEP)
        fractions.push(f);
    fractions.push(RING_RANGE[1]);
    const nf = fractions.length;
    const whole = nf - 1; // the dot's own disc
    // the pixels compared: each dot's square, where it lies in the image
    const at = [];
    for (const d of dots) {
        const x0 = Math.floor(d.cx) - half;
        const y0 = Math.floor(d.cy) - half;
        for (let j = 0; j < side; j++) {
            for (let i = 0; i < side; i++)
                at.push(x0 + i >= 0 && x0 + i < width && y0 + j >= 0 && y0 + j < height ? (y0 + j) * width + x0 + i : -1);
        }
    }
    const n = at.length;
    const discs = fractions.map(f => {
        const all = new Float32Array(n);
        const crisp = { width: side, height: side, data: new Float32Array(side * side) };
        const soft = kernel ? { width: side, height: side, data: new Float32Array(side * side) } : crisp;
        dots.forEach((d, k) => {
            const x0 = Math.floor(d.cx) - half;
            const y0 = Math.floor(d.cy) - half;
            for (let j = 0; j < side; j++) {
                for (let i = 0; i < side; i++)
                    crisp.data[j * side + i] = pixelCoverage(d.cx, d.cy, f * radius, x0 + i, y0 + j);
            }
            if (kernel)
                blurRect(crisp, soft, kernel, [0, 0, side, side]);
            all.set(soft.data, k * side * side);
        });
        return all;
    });
    // y: what the fill leaves of each pixel; its product with each disc, and the discs' with each other
    const y = new Float32Array(n);
    let syy = 0;
    for (let p = 0; p < n; p++) {
        if (at[p] < 0)
            continue;
        y[p] = t[at[p]] - tC * alpha * discs[whole][p];
        syy += y[p] * y[p];
    }
    const sy = new Float64Array(nf);
    const gram = new Float64Array(nf * nf);
    for (let a = 0; a < nf; a++) {
        const da = discs[a];
        for (let p = 0; p < n; p++) {
            if (at[p] >= 0)
                sy[a] += y[p] * da[p];
        }
        for (let b = 0; b <= a; b++) {
            const db = discs[b];
            let s = 0;
            for (let p = 0; p < n; p++) {
                if (at[p] >= 0)
                    s += da[p] * db[p];
            }
            gram[a * nf + b] = s;
            gram[b * nf + a] = s;
        }
    }
    let best = null;
    for (let i = 0; i < nf; i++) {
        for (let o = i + 1; o < nf; o++) {
            if (fractions[o] - fractions[i] < RING_MIN_WIDTH - 1e-9)
                continue;
            const num = tC * (sy[o] - sy[i]);
            const den = tC * tC * (gram[o * nf + o] - 2 * gram[o * nf + i] + gram[i * nf + i]);
            const a = den > 0 ? Math.min(1 - alpha, Math.max(0, num / den)) : 0;
            const sse = syy - 2 * a * num + a * a * den;
            if (!best || sse < best.sse)
                best = { inner: fractions[i], outer: fractions[o], alpha: a, sse };
        }
    }
    return best && best.alpha >= RING_MIN_ALPHA ? { inner: best.inner, outer: best.outer, alpha: best.alpha } : null;
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
 * @param {{cx: number, cy: number, r: number}[]} dots  from isolatedDots: centroids and area radii, px
 * @returns {{sigma: number, offset: number, profile: {rel: number, d: number}[]}}  sigma px; offset px: where the
 *   edge lies from the area radius; profile: the binned D against the distance from the area radius, px
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
 *
 * @param {Float32Array} D
 * @param {number} width
 * @param {number} height
 * @returns {{sigma: number, offset: number, profile: {rel: number, d: number}[]}}  sigma and offset px; profile: the
 *   binned D against the signed distance, px, positive inside
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

/**
 * The share of the pixels whose centers lie within r of (cx, cy) where v > 0.5.
 *
 * @param {ArrayLike<number>} v  width × height
 * @param {number} width
 * @param {number} height
 * @param {number} cx  px
 * @param {number} cy  px
 * @param {number} r  px
 * @returns {number}  0–1; 0 when no pixel center lies in the disc
 */
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

// Dots drawn as a translucent fill under a dark outline ring (ggplot's shape 21, JMP's outlined markers with a gray
// edge), where the fill varies by dot (PLAN.md phase 5, milestone 2: pathogen).
//
// Fills of a color scale over a background B lie on a curve, and every composite of them (a fill over a fill, over
// B) stays in the span of B and that curve. When the curve lies in a plane through B, its normal n sees none of it:
// P = n·(X − B) / n·(K − B), with K black, is 0 for every fill and every stack of fills (pathogen: 11 106 flat fill
// pixels within 2.8 units of the plane). A ring of color K and opacity αk composites in P exactly as a translucent
// family does, P ← (1 − αk)·P + αk, and a fill drawn later multiplies P by (1 − αf), dimming the rings and
// gridlines under it. So in P the picture is rings and their occlusion, in drawing order, and the fill colors drop
// out. Gridlines, a band's level and thick lines are a per-pixel base under the dots. A scale's colors are a curve,
// not exactly a plane, so each fill keeps a small value of its own, f (pathogen: P from −0.04 to +0.08 inside
// isolated fills, against 0.345 for a ring), which the fit sets per dot.

import { pixelCoverage, gaussianKernel, blurRect, createDotModel, solidImage } from './render.js';
import { ringCenters } from './fit.js';

const RING_EDGE_MARGIN = 1; // px around a dot's rectangle for antialiasing
const K = [0, 0, 0]; // the ring color P is measured toward
// The fill plane: flat pixels (3 × 3 within PLANE_FLAT_TOL; pathogen's fills carry ±5–8 units of noise) at least
// PLANE_MIN_SATURATION from gray. Through B, the smallest principal axis is n. It is a scale's plane when the fills
// spread in two directions (the middle axis), lie close to it (the smallest), and K is well off it.
const PLANE_FLAT_TOL = 6; // 0–255 units
const PLANE_MIN_SATURATION = 20; // 0–255 units, max − min channel
const PLANE_MIN_PIXELS = 1000; // (pathogen: 11 106; the 360 × 240 synthetic test: 1631)
const PLANE_MAX_RMS = 4; // 0–255 units off the plane (pathogen: 2.8)
const PLANE_MIN_SPREAD = 10; // 0–255 units rms along the middle axis (pathogen: 30.7; one family's line: ~1)
const PLANE_MIN_K = 0.2; // |n·(K − B)| / |K − B| (pathogen: 0.32)
// Ring ink A = (P − P_bg) / (1 − P_bg): P_bg, what is under the rings, is the largest of linear openings of P in
// OPEN_DIRECTIONS directions with a segment longer than a ring is across, which keeps lines and areas and drops
// rings. Where P_bg is above MASK_BG (the black line, text) nothing can be read.
const OPEN_DIRECTIONS = 12;
const OPEN_EXTRA = 2; // px beyond the ring's diameter
const MASK_BG = 0.5;
// Ink nearly black (text, ticks) is masked with what lies within MASK_INK_REACH of it: thin strokes don't survive
// the openings, and letters like "o" and "e" look like rings (bench1 solid-graystroke: 81 phantom dots piled on
// the title). Rings stay well below: one is about 0.34 in P, three crossing about 0.7.
const MASK_INK = 0.8;
const MASK_INK_REACH = 2; // px
// Isolated rings, for the radius and style: connected components of A above ISO_LEVEL, square within ISO_SQUARE,
// hollow (under ISO_FILL of their box), ISO_SIZE across
const ISO_LEVEL = 0.12;
const ISO_SQUARE = 2; // px
const ISO_FILL = 0.5;
const ISO_SIZE = [6, 80]; // px
const ISO_MIN = 3; // isolated rings needed
const ISO_MAX = 12; // isolated rings used for the style
const STYLE_HALF = 4; // px beyond the radius in each style patch
const MIN_RING_ALPHA = 0.1; // a ring fainter than this in P isn't an outline
const MAX_RING_RMS = 0.06; // P units: the isolated rings must fit the annulus this well (pathogen: 0.024)
// αf from pairs of strong candidates (ringCenters at PAIR_MIN_ARC) 0.8–1.5 r apart: a ring's ink inside the other's
// fill over its ink outside is 1 where the other is under it and 1 − αf where on top (pathogen: modes near 1 and
// 0.2); the median of the ratios under PAIR_LOW_MODE is 1 − αf
const PAIR_MIN_ARC = 0.5;
const PAIR_DIST = [0.8, 1.5]; // × r
const PAIR_LOW_MODE = 0.6;
const PAIR_MIN = 5;
const FILL_ALPHA_DEFAULT = 0.5; // when too few pairs show it

/**
 * An incrementally updatable rendering of outlined dots in P (one value per pixel, scaled to t units), composited
 * in drawing order over a per-pixel base: each dot first lays its fill over what is under it,
 * P ← (1 − αf·m_f)·P + αf·m_f·f, with m_f its coverage of the disc at the ring's mid-radius and f the fill's own
 * value (0 unless set), then adds its ring, P ← (1 − αk·m_r)·P + αk·m_r·K, with
 * m_r the coverage of the annulus between inner × radius and radius. The order is each dot's z (later is higher).
 * The same interface as render.createDotModel, plus z.
 *
 * Complexity per change: O((2r + 2)² · k) for k dots overlapping the patch, plus the blur.
 *
 * @param {Float32Array} base  t per pixel under the dots (width × height)
 * @param {number} width
 * @param {number} height
 * @param {object} style
 * @param {number} style.radius  px, the ring's outer radius
 * @param {{inner: number, alpha: number}} style.ring  inner radius over radius; αk, 0–1
 * @param {number} style.fillAlpha  αf, 0–1
 * @param {number} style.ringT  t of the ring color K
 * @param {number} [style.blur]  px, sd of the Gaussian applied after compositing
 */
export function createOutlineModel(base, width, height, style) {
    const { radius, ring, fillAlpha, ringT } = style;
    const innerRadius = ring.inner * radius;
    const fillRadius = radius * (1 + ring.inner) / 2;
    const blur = style.blur ?? 0;
    const kernel = blur > 0 ? gaussianKernel(blur) : null;
    const kHalf = kernel ? (kernel.length - 1) / 2 : 0;
    const n = width * height;
    const T = new Float32Array(n).fill(1);
    const composite = { width, height, data: Float32Array.from(base) };
    const output = kernel ? { width, height, data: Float32Array.from(base) } : composite;
    const cell = Math.max(1, 2 * radius);
    const gw = Math.ceil(width / cell) + 1;
    const gh = Math.ceil(height / cell) + 1;
    const grid = Array.from({ length: gw * gh }, () => new Set());
    const dots = new Map(); // id → { x, y, z, rect, fill, ring }
    let nextId = 0;
    let topZ = 0;

    const cellOf = (x, y) => Math.min(gh - 1, Math.max(0, Math.floor(y / cell))) * gw + Math.min(gw - 1, Math.max(0, Math.floor(x / cell)));

    function dotRect(x, y) {
        return [
            Math.max(0, Math.floor(x - radius) - RING_EDGE_MARGIN + 1),
            Math.max(0, Math.floor(y - radius) - RING_EDGE_MARGIN + 1),
            Math.min(width, Math.floor(x + radius) + RING_EDGE_MARGIN),
            Math.min(height, Math.floor(y + radius) + RING_EDGE_MARGIN),
        ];
    }

    function makeRecord(x, y, z, f = 0) {
        const rect = dotRect(x, y);
        const [x0, y0, x1, y1] = rect;
        const w = Math.max(0, x1 - x0);
        const size = w * Math.max(0, y1 - y0);
        const fill = new Float32Array(size);
        const rg = new Float32Array(size);
        for (let yy = y0; yy < y1; yy++) {
            for (let xx = x0; xx < x1; xx++) {
                const m = pixelCoverage(x, y, radius, xx, yy);
                const i = (yy - y0) * w + xx - x0;
                fill[i] = m > 0 ? pixelCoverage(x, y, fillRadius, xx, yy) : 0;
                rg[i] = m > 0 ? m - pixelCoverage(x, y, innerRadius, xx, yy) : 0;
            }
        }
        return { x, y, z, f, rect, fill, ring: rg };
    }

    function dotsNear(rect) {
        const [x0, y0, x1, y1] = rect;
        const gx0 = Math.max(0, Math.floor((x0 - radius) / cell));
        const gy0 = Math.max(0, Math.floor((y0 - radius) / cell));
        const gx1 = Math.min(gw - 1, Math.floor((x1 + radius) / cell));
        const gy1 = Math.min(gh - 1, Math.floor((y1 + radius) / cell));
        const out = [];
        for (let gy = gy0; gy <= gy1; gy++) {
            for (let gx = gx0; gx <= gx1; gx++) {
                for (const id of grid[gy * gw + gx])
                    out.push(dots.get(id));
            }
        }
        return out;
    }

    const blurredRect = ([x0, y0, x1, y1]) => [Math.max(0, x0 - kHalf), Math.max(0, y0 - kHalf), Math.min(width, x1 + kHalf), Math.min(height, y1 + kHalf)];
    const union = (a, b) => [Math.min(a[0], b[0]), Math.min(a[1], b[1]), Math.max(a[2], b[2]), Math.max(a[3], b[3])];

    // Dots are applied in z order, each to all of its pixels in the rectangle: per pixel that is the composite in
    // drawing order.
    function recompute(rect) {
        const [x0, y0, x1, y1] = rect;
        if (x0 >= x1 || y0 >= y1)
            return;
        const v = composite.data;
        for (let y = y0; y < y1; y++) {
            v.set(base.subarray(y * width + x0, y * width + x1), y * width + x0);
            T.fill(1, y * width + x0, y * width + x1);
        }
        const near = dotsNear(rect).sort((a, b) => a.z - b.z);
        const aK = ring.alpha;
        for (const d of near) {
            const [dx0, dy0, dx1, dy1] = d.rect;
            const dw = dx1 - dx0;
            const ya = Math.max(y0, dy0);
            const yb = Math.min(y1, dy1);
            const xa = Math.max(x0, dx0);
            const xb = Math.min(x1, dx1);
            for (let y = ya; y < yb; y++) {
                const row = (y - dy0) * dw - dx0;
                for (let x = xa; x < xb; x++) {
                    const p = y * width + x;
                    const mf = fillAlpha * d.fill[row + x];
                    const mr = aK * d.ring[row + x];
                    v[p] = (1 - mr) * ((1 - mf) * v[p] + mf * d.f) + mr * ringT;
                    T[p] *= (1 - mf) * (1 - mr);
                }
            }
        }
        if (kernel)
            blurRect(composite, output, kernel, blurredRect(rect));
    }

    function copyRect(src, rect) {
        const [x0, y0, x1, y1] = rect;
        const w = Math.max(0, x1 - x0);
        const out = new Float32Array(w * Math.max(0, y1 - y0));
        for (let y = y0; y < y1; y++)
            out.set(src.subarray(y * width + x0, y * width + x1), (y - y0) * w);
        return out;
    }

    function pasteRect(dst, rect, saved) {
        const [x0, y0, x1, y1] = rect;
        const w = Math.max(0, x1 - x0);
        for (let y = y0; y < y1; y++)
            dst.set(saved.subarray((y - y0) * w, (y - y0 + 1) * w), y * width + x0);
    }

    function place(id, rec) {
        const old = dots.get(id);
        if (old)
            grid[cellOf(old.x, old.y)].delete(id);
        dots.set(id, rec);
        grid[cellOf(rec.x, rec.y)].add(id);
        topZ = Math.max(topZ, rec.z);
        return old;
    }

    return {
        /** t of a pixel of transmittance Tp under one ring over the background (for birth thresholds). */
        valueOf(Tp) {
            return ringT * (1 - Tp);
        },

        /** The rendered t, one value per pixel. Live. */
        image: output,
        /** Product of what each dot lets through, before blur. Live. */
        transmittance: T,

        /**
         * @param {number} x  px
         * @param {number} y  px
         * @param {number} [z]  drawing order (default: over every dot so far)
         * @returns {number} the new dot's id
         */
        add(x, y, z = topZ + 1) {
            const id = nextId++;
            const rec = makeRecord(x, y, z);
            place(id, rec);
            recompute(rec.rect);
            return id;
        },

        /** Add many dots ({x, y, z?}) and recompute once. */
        addAll(points) {
            const ids = [];
            for (const p of points) {
                const id = nextId++;
                place(id, makeRecord(p.x, p.y, p.z ?? topZ + 1, p.f ?? 0));
                ids.push(id);
            }
            recompute([0, 0, width, height]);
            return ids;
        },

        remove(id) {
            const d = dots.get(id);
            grid[cellOf(d.x, d.y)].delete(id);
            dots.delete(id);
            recompute(d.rect);
        },

        move(id, x, y) {
            const old = dots.get(id);
            const rec = makeRecord(x, y, old.z, old.f);
            place(id, rec);
            recompute(union(old.rect, rec.rect));
        },

        zOf(id) {
            return dots.get(id).z;
        },

        /** Set a dot's own fill value f, in t units. */
        setFill(id, f) {
            const old = dots.get(id);
            place(id, { ...old, f });
            recompute(old.rect);
        },

        fillOf(id) {
            return dots.get(id).f;
        },

        /** The fill's radius (the ring's mid-radius), px. */
        fillRadius,

        snapshot(rect) {
            const outRect = blurredRect(rect);
            return {
                rect, outRect, topZ,
                records: new Map(dots),
                T: copyRect(T, rect),
                comp: copyRect(composite.data, rect),
                out: kernel ? copyRect(output.data, outRect) : null,
            };
        },

        restore(snap) {
            for (const [id, d] of dots) {
                if (snap.records.get(id) !== d)
                    grid[cellOf(d.x, d.y)].delete(id);
            }
            for (const [id, d] of snap.records) {
                if (dots.get(id) !== d)
                    grid[cellOf(d.x, d.y)].add(id);
            }
            dots.clear();
            for (const [id, d] of snap.records)
                dots.set(id, d);
            topZ = snap.topZ;
            pasteRect(T, snap.rect, snap.T);
            pasteRect(composite.data, snap.rect, snap.comp);
            if (kernel)
                pasteRect(output.data, snap.outRect, snap.out);
        },

        /** @returns {{id: number, x: number, y: number, z: number, f: number}[]} */
        dots() {
            return [...dots].map(([id, d]) => ({ id, x: d.x, y: d.y, z: d.z, f: d.f }));
        },

        footprint: dotRect,

        influenceRect(x, y) {
            return blurredRect(dotRect(x, y));
        },
    };
}

// smallest principal axis of the offsets o (3 per point) through the origin: Jacobi on Σ o oᵀ
function principalAxes(sums, count) {
    let A = sums.map(r => r.slice());
    let V = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
    for (let it = 0; it < 50; it++) {
        let p = 0;
        let q = 1;
        for (let i = 0; i < 3; i++) {
            for (let j = i + 1; j < 3; j++) {
                if (Math.abs(A[i][j]) > Math.abs(A[p][q])) {
                    p = i;
                    q = j;
                }
            }
        }
        if (Math.abs(A[p][q]) < 1e-9)
            break;
        const th = 0.5 * Math.atan2(2 * A[p][q], A[q][q] - A[p][p]);
        const c = Math.cos(th);
        const sn = Math.sin(th);
        for (let k = 0; k < 3; k++) {
            const akp = A[k][p];
            const akq = A[k][q];
            A[k][p] = c * akp - sn * akq;
            A[k][q] = sn * akp + c * akq;
        }
        for (let k = 0; k < 3; k++) {
            const apk = A[p][k];
            const aqk = A[q][k];
            A[p][k] = c * apk - sn * aqk;
            A[q][k] = sn * apk + c * aqk;
        }
        for (let k = 0; k < 3; k++) {
            const vkp = V[k][p];
            const vkq = V[k][q];
            V[k][p] = c * vkp - sn * vkq;
            V[k][q] = sn * vkp + c * vkq;
        }
    }
    const order = [0, 1, 2].sort((a, b) => A[a][a] - A[b][b]);
    return order.map(i => ({ rms: Math.sqrt(Math.max(0, A[i][i]) / count), axis: [V[0][i], V[1][i], V[2][i]] }));
}

/**
 * The plane of a color scale's fills through B, if the image has one.
 *
 * @param {import('./image.js').RgbaImage} image
 * @param {number[]} B  0–255
 * @returns {{n: number[], rms: number, spread: number, pixels: number, kOff: number, oneColor: boolean}|null}
 *   n: unit normal; rms and spread in 0–255 units; kOff = |n·(K − B)| / |K − B|; oneColor: the fills lie on one
 *   line through B
 */
export function fillPlane(image, B) {
    const { width, height, data } = image;
    const sums = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
    let count = 0;
    for (let y = 1; y < height - 1; y++) {
        for (let x = 1; x < width - 1; x++) {
            const p = 4 * (y * width + x);
            const r = data[p];
            const g = data[p + 1];
            const b = data[p + 2];
            if (Math.max(r, g, b) - Math.min(r, g, b) < PLANE_MIN_SATURATION)
                continue;
            let flat = true;
            for (let dy = -1; dy <= 1 && flat; dy++) {
                for (let dx = -1; dx <= 1 && flat; dx++) {
                    const q = p + 4 * (dy * width + dx);
                    if (Math.abs(data[q] - r) > PLANE_FLAT_TOL || Math.abs(data[q + 1] - g) > PLANE_FLAT_TOL || Math.abs(data[q + 2] - b) > PLANE_FLAT_TOL)
                        flat = false;
                }
            }
            if (!flat)
                continue;
            const o = [r - B[0], g - B[1], b - B[2]];
            for (let i = 0; i < 3; i++) {
                for (let j = 0; j < 3; j++)
                    sums[i][j] += o[i] * o[j];
            }
            count++;
        }
    }
    if (count < PLANE_MIN_PIXELS)
        return null;
    const [low, mid, high] = principalAxes(sums, count);
    const kb = K.map((v, ch) => v - B[ch]);
    const kl = Math.hypot(...kb);
    let n = low.axis;
    let rms = low.rms;
    const oneColor = mid.rms < PLANE_MIN_SPREAD;
    // One fill color: its pixels lie on a line through B and any plane holding the line will do. The one that
    // sees K best has the normal along K's direction made orthogonal to the line.
    if (oneColor) {
        const u = high.axis;
        const g = kb.map(v => v / kl);
        const gu = g[0] * u[0] + g[1] * u[1] + g[2] * u[2];
        const m = g.map((v, ch) => v - gu * u[ch]);
        const ml = Math.hypot(...m);
        if (ml === 0)
            return null;
        n = m.map(v => v / ml);
        rms = Math.hypot(low.rms, mid.rms);
    }
    const kOff = Math.abs(n[0] * kb[0] + n[1] * kb[1] + n[2] * kb[2]) / kl;
    if (rms > PLANE_MAX_RMS || kOff < PLANE_MIN_K)
        return null;
    return { n, rms, spread: mid.rms, pixels: count, kOff, oneColor };
}

/** P per pixel: n·(X − B) / n·(K − B), 0 for the fills, 1 for K. */
export function planeValue(image, B, n) {
    const { width, height, data } = image;
    const sK = n[0] * (K[0] - B[0]) + n[1] * (K[1] - B[1]) + n[2] * (K[2] - B[2]);
    const P = new Float32Array(width * height);
    for (let p = 0; p < P.length; p++)
        P[p] = (n[0] * (data[4 * p] - B[0]) + n[1] * (data[4 * p + 1] - B[1]) + n[2] * (data[4 * p + 2] - B[2])) / sK;
    return P;
}

// Linear opening (min, then max, over a segment of `len` px in direction ang). O(pixels · len).
function lineOpening(src, width, height, ang, len) {
    const h = (len - 1) / 2;
    const offs = [];
    for (let k = -h; k <= h; k++)
        offs.push(Math.round(k * Math.sin(ang)) * width + Math.round(k * Math.cos(ang)));
    const dxs = [];
    for (let k = -h; k <= h; k++)
        dxs.push(Math.round(k * Math.cos(ang)));
    const pass = (a, sign, pick) => {
        const out = new Float32Array(a.length);
        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
                const p = y * width + x;
                let m = pick === 'min' ? Infinity : -Infinity;
                for (let i = 0; i < offs.length; i++) {
                    const xx = x + sign * dxs[i];
                    const q = p + sign * offs[i];
                    if (xx < 0 || xx >= width || q < 0 || q >= a.length)
                        continue;
                    m = pick === 'min' ? Math.min(m, a[q]) : Math.max(m, a[q]);
                }
                out[p] = m;
            }
        }
        return out;
    };
    return pass(pass(src, 1, 'min'), -1, 'max');
}

/**
 * What is under the rings in P: the largest of linear openings in OPEN_DIRECTIONS directions, each with a segment
 * longer than a ring is across. O(pixels · OPEN_DIRECTIONS · (2r + OPEN_EXTRA)).
 */
export function underRings(P, width, height, radius) {
    const len = 2 * Math.ceil(radius + OPEN_EXTRA / 2) + 1;
    const out = new Float32Array(P.length).fill(-Infinity);
    for (let k = 0; k < OPEN_DIRECTIONS; k++) {
        const o = lineOpening(P, width, height, Math.PI * k / OPEN_DIRECTIONS, len);
        for (let p = 0; p < out.length; p++)
            out[p] = Math.max(out[p], o[p]);
    }
    return out;
}

// connected components (8-neighbors) of v ≥ level: boxes and pixel counts
function components(v, width, height, level) {
    const seen = new Uint8Array(v.length);
    const out = [];
    for (let s = 0; s < v.length; s++) {
        if (seen[s] || !(v[s] >= level))
            continue;
        const comp = { x0: Infinity, y0: Infinity, x1: -Infinity, y1: -Infinity, count: 0 };
        const stack = [s];
        seen[s] = 1;
        while (stack.length) {
            const p = stack.pop();
            const x = p % width;
            const y = (p - x) / width;
            comp.count++;
            comp.x0 = Math.min(comp.x0, x);
            comp.x1 = Math.max(comp.x1, x);
            comp.y0 = Math.min(comp.y0, y);
            comp.y1 = Math.max(comp.y1, y);
            for (let dy = -1; dy <= 1; dy++) {
                for (let dx = -1; dx <= 1; dx++) {
                    const xx = x + dx;
                    const yy = y + dy;
                    if (xx < 0 || yy < 0 || xx >= width || yy >= height)
                        continue;
                    const q = yy * width + xx;
                    if (!seen[q] && v[q] >= level) {
                        seen[q] = 1;
                        stack.push(q);
                    }
                }
            }
        }
        out.push(comp);
    }
    return out;
}

// Isolated rings: square, hollow components of P of a plausible size, their box centers
function isolatedRings(P, width, height) {
    return components(P, width, height, ISO_LEVEL).filter(c => {
        const w = c.x1 - c.x0 + 1;
        const h = c.y1 - c.y0 + 1;
        if (Math.abs(w - h) > ISO_SQUARE || w < ISO_SIZE[0] || w > ISO_SIZE[1] || c.count > ISO_FILL * w * h)
            return false;
        const mid = Math.floor((c.y0 + c.y1) / 2) * width + Math.floor((c.x0 + c.x1) / 2);
        return P[mid] < ISO_LEVEL;
    }).map(c => ({ x: (c.x0 + c.x1 + 1) / 2, y: (c.y0 + c.y1 + 1) / 2, r: (c.x1 - c.x0 + c.y1 - c.y0 + 2) / 4 }));
}

// A for one ring at (cx, cy) in a patch: an annulus of opacity 1, blurred; least-squares α against the image's A
function ringPatchError(A, width, height, cx, cy, st) {
    const half = Math.ceil(st.r + STYLE_HALF);
    const x0 = Math.round(cx) - half;
    const y0 = Math.round(cy) - half;
    const S = 2 * half + 1;
    if (x0 < 0 || y0 < 0 || x0 + S > width || y0 + S > height)
        return null;
    const m = createDotModel(solidImage(S, S, [255, 255, 255]), { color: [0, 0, 0], alpha: 0, radius: st.r, ring: { inner: st.inner, alpha: 1 }, blur: st.blur });
    m.add(cx - x0, cy - y0);
    let num = 0;
    let den = 0;
    let saa = 0;
    let count = 0;
    for (let j = 0; j < S; j++) {
        for (let i = 0; i < S; i++) {
            const a = A[(y0 + j) * width + x0 + i];
            if (Number.isNaN(a))
                continue;
            const mm = 1 - m.image.data[3 * (j * S + i)] / 255;
            num += a * mm;
            den += mm * mm;
            saa += a * a;
            count++;
        }
    }
    const alpha = den > 0 ? num / den : 0;
    return { sse: saa - alpha * num, alpha, count };
}

/**
 * The ring's style from isolated rings in A: radius, inner over radius, blur and αk, by coordinate descent on the
 * squared error over their patches, each center refined too.
 *
 * @returns {{radius: number, inner: number, blur: number, alpha: number, rms: number, rings: number}}
 */
function measureRing(A, width, height, rings) {
    const r0 = rings.map(c => c.r).sort((a, b) => a - b)[rings.length >> 1];
    let st = { r: r0, inner: Math.max(0.5, (r0 - 1.5) / r0), blur: 0.5 };
    let centers = rings.map(c => ({ x: c.x, y: c.y }));
    const total = () => {
        let sse = 0;
        let count = 0;
        const alphas = [];
        for (const c of centers) {
            const e = ringPatchError(A, width, height, c.x, c.y, st);
            if (!e)
                continue;
            sse += e.sse;
            count += e.count;
            alphas.push(e.alpha);
        }
        return { sse, count, alphas };
    };
    const refineCenters = () => {
        centers = centers.map(c => {
            let { x, y } = c;
            for (const step of [0.5, 0.25, 0.1]) {
                let improved = true;
                while (improved) {
                    improved = false;
                    const e0 = ringPatchError(A, width, height, x, y, st);
                    if (!e0)
                        return { x, y };
                    for (const [dx, dy] of [[step, 0], [-step, 0], [0, step], [0, -step]]) {
                        const e = ringPatchError(A, width, height, x + dx, y + dy, st);
                        if (e && e.sse < e0.sse - 1e-9) {
                            x += dx;
                            y += dy;
                            improved = true;
                            break;
                        }
                    }
                }
            }
            return { x, y };
        });
    };
    const steps = { r: [0.5, 0.25, 0.125], width: [0.5, 0.25, 0.125], blur: [0.2, 0.1] };
    for (let it = 0; it < 3; it++) {
        refineCenters();
        for (const k of ['r', 'width', 'blur']) {
            for (const step of steps[k]) {
                let best = total().sse;
                let improved = true;
                while (improved) {
                    improved = false;
                    for (const sgn of [1, -1]) {
                        const w0 = st.r * (1 - st.inner);
                        const next = { ...st };
                        if (k === 'r') {
                            next.r = st.r + sgn * step;
                            next.inner = (next.r - w0) / next.r;
                        }
                        else if (k === 'width') {
                            next.inner = (st.r - Math.max(0.5, w0 + sgn * step)) / st.r;
                        }
                        else {
                            next.blur = Math.max(0, st.blur + sgn * step);
                        }
                        const saved = st;
                        st = next;
                        const e = total().sse;
                        if (e < best - 1e-9) {
                            best = e;
                            improved = true;
                            break;
                        }
                        st = saved;
                    }
                }
            }
        }
    }
    const { sse, count, alphas } = total();
    alphas.sort((a, b) => a - b);
    return { radius: st.r, inner: st.inner, blur: st.blur, alpha: alphas[alphas.length >> 1], rms: Math.sqrt(sse / Math.max(1, count)), rings: alphas.length };
}

/**
 * αf from pairs of strong ring candidates: along one ring, its ink inside the other's fill over its ink well
 * outside every other candidate; the low mode is 1 − αf.
 *
 * @returns {{fillAlpha: number, pairs: number}}
 */
function measureFillAlpha(A, width, height, cands, ring) {
    const { radius: R, inner } = ring;
    const rm = R * (1 + inner) / 2;
    const rf = rm; // the fill reaches the ring's middle
    const bil = (x, y) => {
        const fx = x - 0.5;
        const fy = y - 0.5;
        const x0 = Math.floor(fx);
        const y0 = Math.floor(fy);
        if (x0 < 0 || y0 < 0 || x0 + 1 >= width || y0 + 1 >= height)
            return NaN;
        const tx = fx - x0;
        const ty = fy - y0;
        const g = (i, j) => A[(y0 + j) * width + x0 + i];
        return (1 - tx) * (1 - ty) * g(0, 0) + tx * (1 - ty) * g(1, 0) + (1 - tx) * ty * g(0, 1) + tx * ty * g(1, 1);
    };
    const low = [];
    let pairs = 0;
    for (const a of cands) {
        for (const b of cands) {
            const d = Math.hypot(a.x - b.x, a.y - b.y);
            if (a === b || d < PAIR_DIST[0] * R || d > PAIR_DIST[1] * R)
                continue;
            const inside = [];
            const outside = [];
            for (let k = 0; k < 4 * RING_SAMPLES_PAIR; k++) {
                const t = 2 * Math.PI * k / (4 * RING_SAMPLES_PAIR);
                const x = a.x + rm * Math.cos(t);
                const y = a.y + rm * Math.sin(t);
                if (cands.some(o => o !== a && o !== b && Math.hypot(x - o.x, y - o.y) < R + PAIR_MARGIN))
                    continue;
                const v = bil(x, y);
                if (Number.isNaN(v))
                    continue;
                const db = Math.hypot(x - b.x, y - b.y);
                if (db < rf - PAIR_MARGIN)
                    inside.push(v);
                else if (db > R + PAIR_MARGIN)
                    outside.push(v);
            }
            if (inside.length < PAIR_MIN_INSIDE || outside.length < PAIR_MIN_OUTSIDE)
                continue;
            const med = s => s.sort((u, v) => u - v)[s.length >> 1];
            const ratio = med(inside) / med(outside);
            pairs++;
            if (ratio < PAIR_LOW_MODE)
                low.push(ratio);
        }
    }
    // rings drawn over every fill (JMP plots the outlines as a second layer of open circles): none is dimmed
    if (low.length < PAIR_MIN && pairs - low.length >= PAIR_MIN)
        return { fillAlpha: 0, pairs, measured: true };
    if (low.length < PAIR_MIN)
        return { fillAlpha: FILL_ALPHA_DEFAULT, pairs, measured: false };
    low.sort((u, v) => u - v);
    return { fillAlpha: Math.min(1, Math.max(0, 1 - low[low.length >> 1])), pairs, measured: true };
}
const RING_SAMPLES_PAIR = 45; // × 4 samples along a ring
const PAIR_MARGIN = 1.5; // px clear of the other dots' rims
const PAIR_MIN_INSIDE = 8;
const PAIR_MIN_OUTSIDE = 40;

/**
 * Outlined dots whose fills vary by dot, if the image has them: the fill plane, P and what is under the rings, the
 * ring's style and αf. Null when there is no fill plane or no consistent ring in it.
 *
 * @param {import('./image.js').RgbaImage} image
 * @param {number[]} B  0–255
 * @returns {{plane: object, P: Float32Array, under: Float32Array, A: Float32Array, ring: object,
 *            fillAlpha: number, fillAlphaMeasured: boolean, pairs: number, isolated: object[]}|null}
 */
export function findOutlines(image, B) {
    const { width, height } = image;
    const plane = fillPlane(image, B);
    if (!plane)
        return null;
    const P = planeValue(image, B, plane.n);
    const rings = isolatedRings(P, width, height);
    if (rings.length < ISO_MIN)
        return null;
    const r0 = rings.map(c => c.r).sort((a, b) => a - b)[rings.length >> 1];
    const under = underRings(P, width, height, r0);
    const ink = new Uint8Array(P.length);
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            if (P[y * width + x] < MASK_INK)
                continue;
            for (let yy = Math.max(0, y - MASK_INK_REACH); yy <= Math.min(height - 1, y + MASK_INK_REACH); yy++) {
                for (let xx = Math.max(0, x - MASK_INK_REACH); xx <= Math.min(width - 1, x + MASK_INK_REACH); xx++)
                    ink[yy * width + xx] = 1;
            }
        }
    }
    const A = new Float32Array(P.length);
    for (let p = 0; p < P.length; p++)
        A[p] = under[p] > MASK_BG || ink[p] ? NaN : (P[p] - under[p]) / (1 - under[p]);
    // the most ring-like of the isolated components
    const used = rings.slice().sort((a, b) => Math.abs(a.r - r0) - Math.abs(b.r - r0)).slice(0, ISO_MAX);
    const ring = measureRing(A, width, height, used);
    if (!(ring.alpha >= MIN_RING_ALPHA) || ring.rms > MAX_RING_RMS || ring.rings < ISO_MIN)
        return null;
    const D = new Float32Array(P.length);
    const weight = new Uint8Array(P.length);
    for (let p = 0; p < P.length; p++) {
        weight[p] = Number.isNaN(A[p]) ? 0 : 1;
        D[p] = weight[p] ? Math.max(0, A[p]) / ring.alpha : NaN;
    }
    const cands = ringCenters(D, weight, width, height, ring.radius, ring.inner, PAIR_MIN_ARC);
    const fa = measureFillAlpha(A, width, height, cands, ring);
    return { plane, P, under, A, D, ring, fillAlpha: fa.fillAlpha, fillAlphaMeasured: fa.measured, pairs: fa.pairs, isolated: used };
}

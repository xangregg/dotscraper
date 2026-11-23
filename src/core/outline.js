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
// isolated fills, against 0.345 for a ring), which the fit sets per dot. Outlines drawn in a darker shade of each
// dot's own fill (findHueOutlines) are off any plane by an amount that depends on the fill, so there P is read
// against the local fill instead.

import { pixelCoverage, gaussianKernel, blurRect, createDotModel, createDotStore, unionRect, solidImage } from './render.js';
import { ringCenters } from './fit.js';
import { colorDistance } from './image.js';
import { components8, dilate, opening } from './layers.js';
import { upperMedian } from './stats.js';

const RING_EDGE_MARGIN = 1; // px around a dot's rectangle for antialiasing
const K = [0, 0, 0]; // the ring color P is measured toward
// The fill plane: flat pixels (3 × 3 within PLANE_FLAT_TOL; pathogen's fills carry ±5–8 units of noise) at least
// PLANE_MIN_SATURATION from gray. Through B, the smallest principal axis is n. It is a fill plane when the fills lie
// close to it and K is off it by enough that P isn't noise: the fills' scatter off the plane, with the pixels'
// rounding, over K's distance from it, is at most PLANE_MAX_P_NOISE in P. K may be close to the plane: two nearly
// complementary fills (bench1–2 group-graystroke, blue and orange) span a plane that almost holds gray, K only 0.05
// of |K − B| off it, but their pixels are exact, so P's noise is 0.022 (pathogen: K 0.32 off, noise 0.020).
const PLANE_FLAT_TOL = 6; // 0–255 units
const PLANE_MIN_SATURATION = 20; // 0–255 units, max − min channel
const PLANE_MIN_PIXELS = 1000; // (pathogen: 11 106; the 360 × 240 synthetic test: 1631)
const PLANE_MAX_RMS = 4; // 0–255 units off the plane (pathogen: 2.8)
const PLANE_MIN_SPREAD = 10; // 0–255 units rms along the middle axis (pathogen: 30.7; one family's line: ~1)
const PLANE_ROUNDING = 0.5; // 0–255 units, a pixel's rounding
// Discrete fill families (fillFamilies, dotFamilies)
const FAMILY_BIN = 8; // 0–255 units per channel
const FAMILY_MIN_SHARE = 0.05; // of the flat colored pixels, for a bin to be a family
const FAMILY_MIN_TOTAL = 0.75; // the families together
const FAMILY_MAX = 6;
const FAMILY_INSET = 1.5; // px inside a fill, and beyond another dot's rim
const FAMILY_MIN_PIXELS = 6;
const PLANE_MAX_P_NOISE = 0.03; // P units
// Ring ink A = (P − P_bg) / (1 − P_bg): P_bg, what is under the rings, is the largest of linear openings of P in
// OPEN_DIRECTIONS directions with a segment longer than a ring is across, which keeps lines and areas and drops
// rings. Where P_bg is above MASK_BG (the black line, text) nothing can be read.
const OPEN_DIRECTIONS = 12;
const OPEN_EXTRA = 2; // px beyond the ring's diameter
// relativeDarkness: a closing's segment, longer than a ring and its antialiasing are across (bench2
// gradient-opaque-stroke: 1.5 px rings, 3 px with their edges), short enough to keep narrow slivers of a darker fill
const CLOSE_LEN = 5; // px
const WHITE_LENGTH = 255 * Math.sqrt(3); // 0–255 units, white's distance from black
const MASK_BG = 0.5;
// Ink nearly black (text, ticks) is masked with what lies within MASK_INK_REACH of it: thin strokes don't survive
// the openings, and letters like "o" and "e" look like rings (bench1 solid-graystroke: 81 phantom dots piled on
// the title). Rings stay well below: one is about 0.34 in P, three crossing about 0.7.
// Opaque black rings are as dark as text, and the mask takes them all (droplet1: 2 px black outlines, 0.93 in P at
// their cores, αk measured 0.028). For fills that vary, when no ring shows with the mask and the isolated rings'
// cores reach MASK_INK, they are read without it, text left to the plot bounds and the ring fit. Measured at
// RING_PEAK_QUANTILE on the fill plane: translucent and gray rings 0.33–0.53, droplet1 0.97; Laser, primates2 and
// tumour1 0.92–0.93, from glyphs that fit no ring.
const MASK_INK = 0.8;
const MASK_INK_REACH = 2; // px
const RING_PEAK_QUANTILE = 0.9;
// A black ring's fit error grows with its contrast: droplet1's rms is 0.052 at αk 1 on its clean rings, about 0.09
// with two crowded ones, against pathogen's 0.024 at αk 0.39 (0.06 of it)
const BLACK_RING_MAX_RMS = 0.1; // P units, at αk 1
// Isolated rings, for the radius and style: connected components of A above ISO_LEVEL, square within ISO_SQUARE,
// hollow (under ISO_FILL of their box), ISO_SIZE across
const ISO_LEVEL = 0.12;
const ISO_SQUARE = 2; // px
const ISO_FILL = 0.5;
const ISO_SIZE = [6, 80]; // px
const ISO_MIN = 3; // isolated rings needed
const ISO_MAX = 12; // isolated rings used for the style
const ISO_RADIUS_TOL = 0.2; // × the median radius, for the rings used for the style
const STYLE_HALF = 4; // px beyond the radius in each style patch
const MIN_RING_ALPHA = 0.1; // a ring fainter than this in P isn't an outline
const MAX_RING_RMS = 0.06; // P units: the isolated rings must fit the annulus this well (pathogen: 0.024)
// Thick-walled rings (outlinesInP's `thick`: open circles drawn with a heavy stroke, corneal's 5 px of 9.5): their ink
// is up to this share of their box (a ring from 0.47 r to r: 0.61; a disc: 0.79), their darkness is fitted with
// nothing masked (a thick dark ring's core is as dark as text, and soft: 0.71 of black at its peak), and they fit
// the annulus to this × their darkness (corneal: 0.148 at αk 0.66, its rings ragged from a small figure enlarged)
const THICK_ISO_FILL = 0.7;
const THICK_RING_MAX_RMS = 0.25; // × αk
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
 * @returns {object}  the model: image and transmittance (live planes); valueOf; add, addAll, remove, move; zOf,
 *   setFill, fillOf and fillRadius (a dot's drawing order and its fill); footprint and influenceRect (the pixels a
 *   dot touches, before and after the blur); snapshot and restore; dots. Each is documented where it is defined.
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
    // a record is { x, y, z, f, rect, fill, ring }
    const store = createDotStore(width, height, radius, kHalf, [
        { data: T, stride: 1, blurred: false },
        { data: composite.data, stride: 1, blurred: false },
        ...(kernel ? [{ data: output.data, stride: 1, blurred: true }] : []),
    ]);
    let topZ = 0;

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
        const near = store.near(rect).sort((a, b) => a.z - b.z);
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
            blurRect(composite, output, kernel, store.blurredRect(rect));
    }

    function place(id, rec) {
        topZ = Math.max(topZ, rec.z);
        return store.put(id, rec);
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
            const id = store.newId();
            const rec = makeRecord(x, y, z);
            place(id, rec);
            recompute(rec.rect);
            return id;
        },

        /** Add many dots ({x, y, z?}) and recompute once. */
        addAll(points) {
            const ids = [];
            for (const p of points) {
                const id = store.newId();
                place(id, makeRecord(p.x, p.y, p.z ?? topZ + 1, p.f ?? 0));
                ids.push(id);
            }
            recompute([0, 0, width, height]);
            return ids;
        },

        remove(id) {
            recompute(store.drop(id).rect);
        },

        move(id, x, y) {
            const old = store.get(id);
            const rec = makeRecord(x, y, old.z, old.f);
            place(id, rec);
            recompute(unionRect(old.rect, rec.rect));
        },

        zOf(id) {
            return store.get(id).z;
        },

        /** Set a dot's own fill value f, in t units. */
        setFill(id, f) {
            const old = store.get(id);
            place(id, { ...old, f });
            recompute(old.rect);
        },

        fillOf(id) {
            return store.get(id).f;
        },

        /** The fill's radius (the ring's mid-radius), px. */
        fillRadius,

        // createDotStore's snapshot, with the top of the drawing order
        snapshot(rect) {
            return { ...store.snapshot(rect), topZ };
        },

        restore(snap) {
            store.restore(snap);
            topZ = snap.topZ;
        },

        /** @returns {{id: number, x: number, y: number, z: number, f: number}[]} in id order */
        dots() {
            return store.entries().map(([id, d]) => ({ id, x: d.x, y: d.y, z: d.z, f: d.f }));
        },

        footprint: dotRect,

        influenceRect(x, y) {
            return store.blurredRect(dotRect(x, y));
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
    if (rms > PLANE_MAX_RMS || !(kOff > 0) || Math.hypot(rms, PLANE_ROUNDING) / (kOff * kl) > PLANE_MAX_P_NOISE)
        return null;
    return { n, rms, spread: mid.rms, pixels: count, kOff, oneColor };
}

/**
 * The fill colors, when they are a few discrete families rather than a scale: flat colored pixels binned by
 * FAMILY_BIN; the largest bins, each holding at least FAMILY_MIN_SHARE of them, are families if together they hold
 * FAMILY_MIN_TOTAL (bench1–2 group-graystroke: two colors hold 0.87 of the flat pixels, their overlaps the rest;
 * pathogen's scale: no bin above 0.02). Each family's color is the mean of its bin and the bins beside it.
 *
 * @param {import('./image.js').RgbaImage} image
 * @returns {number[][]|null}  colors 0–255, largest first, or null for a scale
 */
export function fillFamilies(image) {
    const { width, height, data } = image;
    const bins = new Map();
    let total = 0;
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
            const key = [r, g, b].map(v => Math.floor(v / FAMILY_BIN)).join(',');
            let e = bins.get(key);
            if (!e) {
                e = { n: 0, sum: [0, 0, 0], key: [r, g, b].map(v => Math.floor(v / FAMILY_BIN)) };
                bins.set(key, e);
            }
            e.n++;
            e.sum[0] += r;
            e.sum[1] += g;
            e.sum[2] += b;
            total++;
        }
    }
    if (!total)
        return null;
    const sorted = [...bins.values()].sort((a, b) => b.n - a.n);
    const families = [];
    let held = 0;
    for (const e of sorted) {
        if (e.n < FAMILY_MIN_SHARE * total || families.length >= FAMILY_MAX)
            break;
        // a bin beside one already taken is the same family split by the binning
        const near = families.find(f => f.key.every((v, i) => Math.abs(v - e.key[i]) <= 1));
        if (near) {
            near.n += e.n;
            near.sum = near.sum.map((v, i) => v + e.sum[i]);
        }
        else {
            families.push({ ...e, sum: [...e.sum] });
        }
        held += e.n;
    }
    if (held < FAMILY_MIN_TOTAL * total || families.length < 2)
        return null;
    return families.map(f => f.sum.map(v => v / f.n));
}

/**
 * Each dot's fill family: the median color of its fill where no other dot reaches (FAMILY_INSET inside its fill,
 * clear of the others' rims), the nearest family color. A dot with fewer than FAMILY_MIN_PIXELS such pixels is read
 * where the fit draws it over every dot covering it, and marked unsure. Its color may simply not be in the image:
 * bench2's near-coincident pairs hide the lower dot's fill (bench2 group-graystroke: 128 sure dots all right, 52 of
 * 70 unsure; group-solid-graystroke: 137, 41 of 50). Reading the order from the rings instead of the fit's z did no
 * better (50 and 37).
 *
 * @param {import('./image.js').RgbaImage} image  the original RGB
 * @param {{x: number, y: number, z: number}[]} dots  px; z the fit's drawing order
 * @param {number[][]} families  fill colors
 * @param {number} radius  px, the ring's outer radius
 * @param {number} fillRadius  px
 * @returns {{family: number, sure: boolean}[]}
 */
export function dotFamilies(image, dots, families, radius, fillRadius) {
    const { width, height, data } = image;
    const rIn = fillRadius - FAMILY_INSET;
    return dots.map(d => {
        const others = dots.filter(o => o !== d && Math.hypot(o.x - d.x, o.y - d.y) < radius + fillRadius);
        const own = [];
        const top = [];
        for (let y = Math.max(0, Math.floor(d.y - rIn)); y < Math.min(height, Math.ceil(d.y + rIn)); y++) {
            for (let x = Math.max(0, Math.floor(d.x - rIn)); x < Math.min(width, Math.ceil(d.x + rIn)); x++) {
                const cx = x + 0.5;
                const cy = y + 0.5;
                if ((cx - d.x) ** 2 + (cy - d.y) ** 2 > rIn * rIn)
                    continue;
                const p = 4 * (y * width + x);
                const c = [data[p], data[p + 1], data[p + 2]];
                const covering = others.filter(o => Math.hypot(cx - o.x, cy - o.y) < radius + FAMILY_INSET);
                if (!covering.length)
                    own.push(c);
                else if (covering.every(o => o.z < d.z))
                    top.push(c);
            }
        }
        const sure = own.length >= FAMILY_MIN_PIXELS;
        const use = sure ? own : top.length ? top : own;
        if (!use.length)
            return { family: 0, sure: false };
        const med = [0, 1, 2].map(ch => upperMedian(use.map(c => c[ch])));
        let best = 0;
        families.forEach((f, k) => {
            if (colorDistance(f, med) < colorDistance(families[best], med))
                best = k;
        });
        return { family: best, sure };
    });
}

/**
 * P per pixel: n·(X − B) / n·(K − B), 0 for the fills, 1 for K.
 *
 * @param {import('./image.js').RgbaImage} image
 * @param {number[]} B  the page, 0–255
 * @param {number[]} n  the fill plane's normal (fillPlane)
 * @returns {Float32Array}  width × height
 */
export function planeValue(image, B, n) {
    const { width, height, data } = image;
    const sK = n[0] * (K[0] - B[0]) + n[1] * (K[1] - B[1]) + n[2] * (K[2] - B[2]);
    const P = new Float32Array(width * height);
    for (let p = 0; p < P.length; p++)
        P[p] = (n[0] * (data[4 * p] - B[0]) + n[1] * (data[4 * p + 1] - B[1]) + n[2] * (data[4 * p + 2] - B[2])) / sK;
    return P;
}

/**
 * Linear opening: the minimum, then the maximum, over a segment of `len` px through each pixel in direction `ang`
 * (the maximum over the segment reflected, as an opening asks). The segment is the pixels at
 * (round(k cos ang), round(k sin ang)) for k from −(len − 1)/2 to (len − 1)/2; what falls outside the image is left
 * out. O(pixels · len).
 *
 * @param {Float32Array} src  width × height
 * @param {number} width
 * @param {number} height
 * @param {number} ang  radians
 * @param {number} len  px, odd
 * @returns {Float32Array}
 */
export function lineOpening(src, width, height, ang, len) {
    const h = (len - 1) / 2;
    // each pixel of the segment once: rounding gives some twice
    const dx = [];
    const dy = [];
    const seen = new Set();
    for (let k = -h; k <= h; k++) {
        const x = Math.round(k * Math.cos(ang));
        const y = Math.round(k * Math.sin(ang));
        if (seen.has(`${x},${y}`))
            continue;
        seen.add(`${x},${y}`);
        dx.push(x);
        dy.push(y);
    }
    return overSegment(overSegment(src, width, height, dx, dy, 1, true), width, height, dx, dy, -1, false);
}

// The minimum (or maximum) of `a` over the pixels at sign · (dx, dy) from each pixel, those outside the image left
// out. Away from the image's edges every one is inside, so the bounds are tested only in the border band; with each
// pixel of the segment taken once, that halved the openings' time (11% of the scan and fit over the test images
// before), for the same values to the bit. Math.min and Math.max, not comparisons: they order −0 and 0 and carry NaN.
function overSegment(a, width, height, dx, dy, sign, isMin) {
    const n = dx.length;
    const off = new Int32Array(n);
    let mx = 0;
    let my = 0;
    for (let i = 0; i < n; i++) {
        off[i] = sign * (dy[i] * width + dx[i]);
        mx = Math.max(mx, Math.abs(dx[i]));
        my = Math.max(my, Math.abs(dy[i]));
    }
    const out = new Float32Array(a.length);
    const nearEdge = (x, y) => {
        const p = y * width + x;
        let m = isMin ? Infinity : -Infinity;
        for (let i = 0; i < n; i++) {
            const xx = x + sign * dx[i];
            const yy = y + sign * dy[i];
            if (xx < 0 || xx >= width || yy < 0 || yy >= height)
                continue;
            m = isMin ? Math.min(m, a[p + off[i]]) : Math.max(m, a[p + off[i]]);
        }
        out[p] = m;
    };
    for (let y = 0; y < height; y++) {
        if (y < my || y >= height - my || width <= 2 * mx) {
            for (let x = 0; x < width; x++)
                nearEdge(x, y);
            continue;
        }
        for (let x = 0; x < mx; x++)
            nearEdge(x, y);
        for (let x = width - mx; x < width; x++)
            nearEdge(x, y);
        const row = y * width;
        // (one loop for each, so that the choice isn't made per tap)
        if (isMin) {
            for (let p = row + mx; p < row + width - mx; p++) {
                let m = Infinity;
                for (let i = 0; i < n; i++)
                    m = Math.min(m, a[p + off[i]]);
                out[p] = m;
            }
        }
        else {
            for (let p = row + mx; p < row + width - mx; p++) {
                let m = -Infinity;
                for (let i = 0; i < n; i++)
                    m = Math.max(m, a[p + off[i]]);
                out[p] = m;
            }
        }
    }
    return out;
}

/**
 * What is under the rings in P: the largest of linear openings in OPEN_DIRECTIONS directions, each with a segment
 * longer than a ring is across. O(pixels · OPEN_DIRECTIONS · (2r + OPEN_EXTRA)).
 *
 * @param {Float32Array} P  width × height
 * @param {number} width
 * @param {number} height
 * @param {number} radius  px, the rings'
 * @returns {Float32Array}  width × height, in P's units
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

// Isolated rings: square, hollow components of P of a plausible size (their ink at most `maxFill` of their box),
// their box centers
function isolatedRings(P, width, height, maxFill = ISO_FILL) {
    return components8(P.map(v => (v >= ISO_LEVEL ? 1 : 0)), width, height).filter(c => {
        const w = c.x1 - c.x0 + 1;
        const h = c.y1 - c.y0 + 1;
        if (Math.abs(w - h) > ISO_SQUARE || w < ISO_SIZE[0] || w > ISO_SIZE[1] || c.pixels.length > maxFill * w * h)
            return false;
        const mid = Math.floor((c.y0 + c.y1) / 2) * width + Math.floor((c.x0 + c.x1) / 2);
        return P[mid] < ISO_LEVEL;
    }).map(c => ({ x: (c.x0 + c.x1 + 1) / 2, y: (c.y0 + c.y1 + 1) / 2, r: (c.x1 - c.x0 + c.y1 - c.y0 + 2) / 4 }));
}

// how dark the isolated rings' cores are: the RING_PEAK_QUANTILE of P over their ink (pixels above ISO_LEVEL in
// their boxes; a ring's antialiased sides are most of it)
function ringPeak(P, width, height, rings) {
    const v = [];
    for (const c of rings) {
        for (let y = Math.max(0, Math.floor(c.y - c.r)); y < Math.min(height, Math.ceil(c.y + c.r)); y++) {
            for (let x = Math.max(0, Math.floor(c.x - c.r)); x < Math.min(width, Math.ceil(c.x + c.r)); x++) {
                if (P[y * width + x] >= ISO_LEVEL)
                    v.push(P[y * width + x]);
            }
        }
    }
    v.sort((a, b) => a - b);
    return v.length ? v[Math.floor(RING_PEAK_QUANTILE * (v.length - 1))] : 0;
}

// A for one ring at (cx, cy) in a patch: an annulus of opacity 1, blurred; least-squares α against the image's A, or
// the error at a given α
function ringPatchError(A, width, height, cx, cy, st, fixedAlpha = null) {
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
    if (fixedAlpha !== null)
        return { sse: saa - 2 * fixedAlpha * num + fixedAlpha * fixedAlpha * den, alpha: fixedAlpha, count };
    const alpha = den > 0 ? num / den : 0;
    return { sse: saa - alpha * num, alpha, count };
}

/**
 * The ring's style from isolated rings in A: radius, inner over radius, blur and αk, by coordinate descent on the
 * squared error over their patches, each center refined too. With `fixedAlpha`, αk is that and the rest is fitted.
 *
 * @returns {{radius: number, inner: number, blur: number, alpha: number, rms: number, rings: number}}
 */
function measureRing(A, width, height, rings, fixedAlpha = null) {
    const r0 = upperMedian(rings.map(c => c.r));
    let st = { r: r0, inner: Math.max(0.5, (r0 - 1.5) / r0), blur: 0.5 };
    let centers = rings.map(c => ({ x: c.x, y: c.y }));
    const total = () => {
        let sse = 0;
        let count = 0;
        const alphas = [];
        for (const c of centers) {
            const e = ringPatchError(A, width, height, c.x, c.y, st, fixedAlpha);
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
                    const e0 = ringPatchError(A, width, height, x, y, st, fixedAlpha);
                    if (!e0)
                        return { x, y };
                    for (const [dx, dy] of [[step, 0], [-step, 0], [0, step], [0, -step]]) {
                        const e = ringPatchError(A, width, height, x + dx, y + dy, st, fixedAlpha);
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
    return { radius: st.r, inner: st.inner, blur: st.blur, alpha: upperMedian(alphas), rms: Math.sqrt(sse / Math.max(1, count)), rings: alphas.length };
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
            const ratio = upperMedian(inside) / upperMedian(outside);
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
    return { fillAlpha: Math.min(1, Math.max(0, 1 - upperMedian(low))), pairs, measured: true };
}
// Hollow rings (ggplot's shape 1, R's pch 1): a stroke of one color with nothing inside, drawn over bands and other
// dots (step1: red rings r 6.4, 1.75 px wide, about 0.31 of white's distance at their cores, some under a band that
// lets 60% through). The fill around each pixel (localFill) minus the pixel is a ring's darkening, along one
// direction from any background; thin ink of other colors (text, lines, other dots' rims) points elsewhere and is
// unknown. The directions with HOLLOW_MIN_PIXELS of thin ink are tried, each by outlinesInP.
const HOLLOW_MIN_INK = 40; // 0–255 units below the local fill, for the direction histogram
const HOLLOW_DIR_BIN = 6; // degrees
const HOLLOW_MIN_PIXELS = 300; // thin ink pixels in one direction (step1's rings: about 3000)
const HOLLOW_MAX_DIRS = 4;
const HOLLOW_SAME_DIR = Math.cos(3 * HOLLOW_DIR_BIN * Math.PI / 180); // directions closer than this are one
// A direction that a found family's color has is that family's when its thin ink is the family's own: the rim of its
// fill, its outline, its stroke (hair1's black outlines, step1's blue rims), which lie within HOLLOW_OWN_REACH px of
// pixels of that color (within HOLLOW_OWN_TOL of it). With less than HOLLOW_OWN_SHARE of the ink there, the
// direction holds other marks and is tried (song-cents: gray rings, 119, 119, 106, 9° from the direction of its
// navy dots, 15, 39, 66, and nowhere near them; leaving the direction out for the navy lost the 46 rings).
const HOLLOW_OWN_REACH = 3; // px
const HOLLOW_OWN_TOL = 40; // 0–255 units, a color cluster's (scan.js CLUSTER_TOL)
const HOLLOW_OWN_SHARE = 0.5;
const HOLLOW_OFF_ABS = 10; // 0–255 units off the ring's direction, or …
const HOLLOW_OFF_REL = 0.3; // … this × the darkening along it, and the pixel is other ink
// A hollow ring's inside is what is outside it (a filled outlined dot's isn't): the median over the isolated rings
// of the distance between their insides' and surroundings' mean colors (step1: page and band alike)
const HOLLOW_INSIDE_TOL = 12; // 0–255 units
const HOLLOW_INSIDE_MARGIN = 1.5; // px clear of the ring, inside and out
const HOLLOW_OUTSIDE_WIDTH = 2; // px of surroundings
// thickRings: a solid family's ink is at least this far from the page toward its color, and solid where an opening by
// this × its dots' radius keeps it; unknown out to this many px from there
const THICK_SOLID_LEVEL = 0.5;
const THICK_SOLID_RADII = 0.45;
const THICK_SOLID_REACH = 3; // px
// … and the rings it finds are thick-walled: thinner ones are the search by thin ink's, which this reading finds
// again along a solid family's direction (song-cents: its 46 gray rings, inner 0.79, as a second family of 42; the
// o's of viability's red legend text, 0.73)
const THICK_MAX_INNER = 0.7; // × the radius

const RING_SAMPLES_PAIR = 45; // × 4 samples along a ring
const PAIR_MARGIN = 1.5; // px clear of the other dots' rims
const PAIR_MIN_INSIDE = 8;
const PAIR_MIN_OUTSIDE = 40;

/**
 * The rings read in a ring image P (outlinesInP).
 *
 * @typedef {object} RingReading
 * @property {Float32Array} P  the ring image: fills near 0, rings toward 1
 * @property {Float32Array} under  what is under the rings, in P's units (underRings)
 * @property {Float32Array} A  the rings' own ink, 0–1: P over what is under it; NaN where it can't be read
 * @property {Float32Array} D  A in whole rings (A / the ring's α); NaN where A is
 * @property {{radius: number, inner: number, blur: number, alpha: number, rms: number, rings: number}} ring  the
 *   ring's style from the isolated rings: outer radius px, inner radius over it, edge blur px, αk 0–1, the rms of
 *   the annulus fitted to them (P units), and how many were used
 * @property {number} fillAlpha  αf, 0–1: how far a fill dims the rings under it
 * @property {boolean} fillAlphaMeasured  from overlapping pairs; else assumed
 * @property {number} pairs  the overlapping pairs it was measured on
 * @property {{x: number, y: number, r: number}[]} isolated  px, the isolated rings used
 */

/**
 * Outlined dots whose fills vary by dot, if the image has them: the fill plane, P and what is under the rings, the
 * ring's style and αf. Null when there is no fill plane or no consistent ring in it.
 *
 * @param {import('./image.js').RgbaImage} image
 * @param {number[]} B  0–255
 * @returns {(RingReading & {plane: {n: number[], rms: number, oneColor: boolean}})|null}  plane: fillPlane's
 *   result, the fills' plane
 */
export function findOutlines(image, B) {
    const plane = fillPlane(image, B);
    if (!plane)
        return null;
    // black rings around one fill color are the opaque families' (hair1: pink and page-filled dots in black outlines)
    const found = outlinesInP(planeValue(image, B, plane.n), image.width, image.height, null, { blackRings: !plane.oneColor });
    return found && { plane, ...found };
}

/**
 * Families of hollow rings, one per direction of thin ink that shows rings of one style whose insides match their
 * surroundings. Each is as findOutlines returns, with fillAlpha 0 and `hollow`: the direction of its darkening and
 * the color its rings show at αk on the page. O(pixels · (OPEN_DIRECTIONS · CLOSE_LEN + directions · ring work)).
 *
 * @param {import('./image.js').RgbaImage} image
 * @param {number[]} B  the page, 0–255
 * @param {number[][]} known  0–255, colors already some family's (fills, outlines, strokes): a direction of
 *   darkening one of them has is left out when the thin ink in it is that family's own (HOLLOW_OWN_SHARE), or, for
 *   one of `opts.solids`, read for thick-walled rings of that color (thickRings)
 * @param {object} [opts]
 * @param {{color: number[], radius: number}[]} [opts.solids]  the plain opaque families: color 0–255, radius px
 * @param {function({x: number, y: number}): boolean} [opts.inside]  whether a point is inside the plot
 * @returns {(RingReading & {plane: null, hollow: {dir: number[], color: number[], thick?: boolean}})[]}  one per ring
 *   color; hollow: the unit direction of its darkening from the page, the color its rings show, 0–255, and whether
 *   they are thick-walled rings in a solid family's color
 */
export function findHollowRings(image, B, known = [], opts = {}) {
    const { width, height, data } = image;
    const n = width * height;
    const F = localFill(image);
    const v = new Float32Array(3 * n);
    const bins = new Map();
    // per known color that darkens the page: its direction, and the pixels within reach of pixels of that color
    const families = known.map(k => ({ k, d: B.map((x, ch) => x - k[ch]) })).filter(({ d }) => Math.hypot(...d) > HOLLOW_OWN_TOL).map(({ k, d }) => {
        const mask = new Uint8Array(n);
        for (let p = 0; p < n; p++)
            mask[p] = Math.hypot(data[4 * p] - k[0], data[4 * p + 1] - k[1], data[4 * p + 2] - k[2]) <= HOLLOW_OWN_TOL ? 1 : 0;
        const l = Math.hypot(...d);
        const solid = opts.solids?.find(s => Math.hypot(...s.color.map((x, ch) => x - k[ch])) <= HOLLOW_OWN_TOL) ?? null;
        return { dir: d.map(x => x / l), near: dilate(mask, width, height, HOLLOW_OWN_REACH), solid };
    });
    for (let p = 0; p < n; p++) {
        const a = F[0][p] - data[4 * p];
        const b = F[1][p] - data[4 * p + 1];
        const c = F[2][p] - data[4 * p + 2];
        v[3 * p] = a;
        v[3 * p + 1] = b;
        v[3 * p + 2] = c;
        const m = Math.hypot(a, b, c);
        if (m < HOLLOW_MIN_INK)
            continue;
        const key = Math.floor(Math.acos(c / m) * 180 / Math.PI / HOLLOW_DIR_BIN) * 1000 + Math.floor((Math.atan2(b, a) * 180 / Math.PI + 360) % 360 / HOLLOW_DIR_BIN);
        const bin = bins.get(key) ?? { count: 0, sum: [0, 0, 0], own: families.map(() => 0) };
        bin.count++;
        families.forEach((f, i) => {
            bin.own[i] += f.near[p];
        });
        bin.sum[0] += a / m;
        bin.sum[1] += b / m;
        bin.sum[2] += c / m;
        bins.set(key, bin);
    }
    const dirs = [];
    const thick = []; // directions that are a solid family's own, with that family
    const cos = (u, w) => u[0] * w[0] + u[1] * w[1] + u[2] * w[2];
    for (const bin of [...bins.values()].sort((x, y) => y.count - x.count)) {
        if (bin.count < HOLLOW_MIN_PIXELS || dirs.length >= HOLLOW_MAX_DIRS)
            break;
        const l = Math.hypot(...bin.sum);
        const u = bin.sum.map(x => x / l);
        const owner = families.find((f, i) => cos(u, f.dir) > HOLLOW_SAME_DIR && bin.own[i] >= HOLLOW_OWN_SHARE * bin.count);
        if (owner?.solid && !thick.some(t => t.solid === owner.solid))
            thick.push({ u, solid: owner.solid });
        if (!owner && !dirs.some(w => cos(u, w) > HOLLOW_SAME_DIR))
            dirs.push(u);
    }
    const out = [];
    for (const u of dirs) {
        const P = new Float32Array(n);
        const unknown = new Uint8Array(n);
        for (let p = 0; p < n; p++) {
            const a = v[3 * p];
            const b = v[3 * p + 1];
            const c = v[3 * p + 2];
            const t = a * u[0] + b * u[1] + c * u[2];
            if (Math.hypot(a - t * u[0], b - t * u[1], c - t * u[2]) > Math.max(HOLLOW_OFF_ABS, HOLLOW_OFF_REL * Math.abs(t)))
                unknown[p] = 1;
            else
                P[p] = Math.max(0, t) / WHITE_LENGTH;
        }
        const found = outlinesInP(P, width, height, unknown, { hollow: (isolated, ring) => hollowInside(image, isolated, ring) });
        if (!found)
            continue;
        const color = B.map((x, ch) => Math.round(Math.min(255, Math.max(0, x - u[ch] * found.ring.alpha * WHITE_LENGTH))));
        out.push({ ...found, plane: null, fillAlpha: 0, fillAlphaMeasured: false, hollow: { dir: u, color } });
    }
    for (const { u, solid } of thick) {
        const found = thickRings(image, B, u, solid, opts.inside);
        if (!found)
            continue;
        const color = B.map((x, ch) => Math.round(Math.min(255, Math.max(0, x - u[ch] * found.ring.alpha * WHITE_LENGTH))));
        out.push({ ...found, plane: null, fillAlpha: 0, fillAlphaMeasured: false, hollow: { dir: u, color, thick: true } });
    }
    return out;
}

/**
 * Thick-walled hollow rings in a solid family's own color: open circles beside filled ones, drawn with a heavy stroke
 * (corneal: 58 rings and 63 dots in one black, the rings' walls 5 px of their 9.5). The search by thin ink leaves a
 * found family's color alone, since the rims of its dots read as rings, and a wall this thick is no thin ink: against
 * the fill around it (localFill's 5 px) it reads unevenly, and half of it is as dark as text. Here each pixel is read
 * against the page, along the family's direction of darkening; the family's solid ink (what an opening by
 * THICK_SOLID_RADII of its radius keeps: its dots, not a wall or a stroke) is unknown with THICK_SOLID_REACH px around
 * it, where its dots' soft edges are; and the rings are measured as thick ones (outlinesInP's `thick`).
 *
 * @returns {RingReading|null}
 */
function thickRings(image, B, u, solid, inside) {
    const { width, height, data } = image;
    const n = width * height;
    const P = new Float32Array(n);
    const unknown = new Uint8Array(n);
    const dark = new Uint8Array(n);
    const full = Math.hypot(...B.map((x, ch) => x - solid.color[ch]));
    for (let p = 0; p < n; p++) {
        const e = [B[0] - data[4 * p], B[1] - data[4 * p + 1], B[2] - data[4 * p + 2]];
        const t = e[0] * u[0] + e[1] * u[1] + e[2] * u[2];
        if (Math.hypot(e[0] - t * u[0], e[1] - t * u[1], e[2] - t * u[2]) > Math.max(HOLLOW_OFF_ABS, HOLLOW_OFF_REL * Math.abs(t)))
            unknown[p] = 1;
        else
            P[p] = Math.max(0, t) / WHITE_LENGTH;
        dark[p] = t >= THICK_SOLID_LEVEL * full ? 1 : 0;
    }
    const dots = dilate(opening(dark, width, height, THICK_SOLID_RADII * solid.radius), width, height, THICK_SOLID_REACH);
    for (let p = 0; p < n; p++)
        unknown[p] ||= dots[p];
    const found = outlinesInP(P, width, height, unknown, { hollow: (isolated, ring) => hollowInside(image, isolated, ring), thick: true, inside });
    return found && found.ring.inner <= THICK_MAX_INNER ? found : null;
}

// whether the isolated rings' insides match their surroundings (HOLLOW_INSIDE_TOL)
function hollowInside(image, rings, ring) {
    const { width, height, data } = image;
    const rIn = ring.inner * ring.radius - HOLLOW_INSIDE_MARGIN;
    const r0 = ring.radius + HOLLOW_INSIDE_MARGIN;
    const r1 = r0 + HOLLOW_OUTSIDE_WIDTH;
    if (rIn <= 0)
        return false;
    const d = [];
    for (const c of rings) {
        const inside = [0, 0, 0, 0];
        const outside = [0, 0, 0, 0];
        for (let y = Math.max(0, Math.floor(c.y - r1)); y < Math.min(height, Math.ceil(c.y + r1)); y++) {
            for (let x = Math.max(0, Math.floor(c.x - r1)); x < Math.min(width, Math.ceil(c.x + r1)); x++) {
                const e = Math.hypot(x + 0.5 - c.x, y + 0.5 - c.y);
                const acc = e <= rIn ? inside : e >= r0 && e <= r1 ? outside : null;
                if (!acc)
                    continue;
                const p = 4 * (y * width + x);
                acc[0] += data[p];
                acc[1] += data[p + 1];
                acc[2] += data[p + 2];
                acc[3]++;
            }
        }
        if (inside[3] && outside[3])
            d.push(Math.hypot(...[0, 1, 2].map(ch => inside[ch] / inside[3] - outside[ch] / outside[3])));
    }
    return d.length > 0 && upperMedian(d) <= HOLLOW_INSIDE_TOL;
}

/**
 * Outlined dots whose outlines are a darker shade of each dot's own fill (JMP's outlined markers on a color scale:
 * fill 238, 218, 145 → ring 229, 199, 89; 189, 90, 78 → 146, 68, 58), if the image has them. Their fills needn't
 * lie in a plane, and a ring's distance off any plane depends on its fill, so P is read against the local fill
 * instead (relativeDarkness). Every fill, and every edge between two fills, is then 0, and a ring about 0.21. Null when no consistent ring shows.
 *
 * @param {import('./image.js').RgbaImage} image
 * @returns {(RingReading & {plane: null})|null}
 */
export function findHueOutlines(image) {
    const found = outlinesInP(relativeDarkness(image), image.width, image.height);
    return found && { plane: null, ...found };
}

/**
 * How much darker each pixel is than the fill around it: |F − X| over |white|, the RGB distance from the fill
 * (localFill). On bench1–2 gradient-opaque-stroke's isolated dots a ring is 85–101 units from its fill, from dark red
 * to pale yellow, so the distance reads every ring alike, about 0.21; relative to the fill (|F − X| / |F|) it read
 * 0.27–0.61, and the darkest fills' rings, stronger than the fitted αk, drew second dots stacked on the first.
 *
 * @param {import('./image.js').RgbaImage} image
 * @returns {Float32Array}  0 for fills, 1 for black on white
 */
export function relativeDarkness(image) {
    const { data } = image;
    const F = localFill(image);
    const out = new Float32Array(F[0].length);
    for (let p = 0; p < out.length; p++) {
        const d = Math.hypot(...[0, 1, 2].map(ch => Math.max(0, F[ch][p] - data[4 * p + ch])));
        out[p] = Math.min(1, d / WHITE_LENGTH);
    }
    return out;
}

/**
 * The fill around each pixel, per channel: the largest of linear closings in OPEN_DIRECTIONS directions with a
 * CLOSE_LEN segment. A closing fills dark structures narrower than its segment (rings, thin lines) and keeps edges
 * between areas, whatever their colors. O(pixels · OPEN_DIRECTIONS · CLOSE_LEN).
 *
 * @param {import('./image.js').RgbaImage} image
 * @returns {Float32Array[]}  three planes, 0–255
 */
export function localFill(image) {
    const { width, height, data } = image;
    const n = width * height;
    return [0, 1, 2].map(ch => {
        // closing(X) = −opening(−X)
        const neg = new Float32Array(n);
        for (let p = 0; p < n; p++)
            neg[p] = -data[4 * p + ch];
        const fill = new Float32Array(n).fill(-Infinity);
        for (let k = 0; k < OPEN_DIRECTIONS; k++) {
            const o = lineOpening(neg, width, height, Math.PI * k / OPEN_DIRECTIONS, CLOSE_LEN);
            for (let p = 0; p < n; p++)
                fill[p] = Math.max(fill[p], -o[p]);
        }
        return fill;
    });
}

/**
 * Rings from P (fills near 0, rings toward 1): the isolated rings, what is under the rings, ring ink A, the ring's
 * style and αf. Null when no consistent ring shows.
 *
 * @param {Float32Array} P  0–1 per pixel
 * @param {number} width
 * @param {number} height
 * @param {Uint8Array} [unknown]  1 where P can't be read (one family's rings: other families' fills and their rings,
 *   ink beyond the outline's reach); masked in A and D
 * @param {object} [opts]
 * @param {boolean} [opts.blackRings]  when no ring shows with dark ink masked, try opaque black rings (MASK_INK)
 * @param {function(object[], object): boolean} [opts.hollow]  hollow rings: a test of the isolated rings and the
 *   ring's style that must pass, made before the work that follows; no αf is measured (fillAlpha 0, pairs 0)
 * @param {boolean} [opts.thick]  thick-walled rings (THICK_ISO_FILL): read with nothing masked, their darkness fitted
 * @param {function({x: number, y: number}): boolean} [opts.inside]  the isolated rings that count: those inside the
 *   plot (a label's 0 or O is a ring too)
 * @returns {RingReading|null}
 */
export function outlinesInP(P, width, height, unknown = null, opts = {}) {
    const rings = isolatedRings(P, width, height, opts.thick ? THICK_ISO_FILL : ISO_FILL).filter(c => !opts.inside || opts.inside(c));
    if (rings.length < ISO_MIN)
        return null;
    const r0 = upperMedian(rings.map(c => c.r));
    const under = underRings(P, width, height, r0);
    // the most ring-like of the isolated components, of about one size (droplet1: a glyph at r 15 beside rings of
    // 9.75–10.5)
    const used = rings.slice().sort((a, b) => Math.abs(a.r - r0) - Math.abs(b.r - r0)).slice(0, ISO_MAX)
        .filter(c => Math.abs(c.r - r0) <= ISO_RADIUS_TOL * r0);
    const read = (maskInk, darkness = maskInk ? null : 1) => {
        const ink = new Uint8Array(P.length);
        for (let y = 0; y < height && maskInk; y++) {
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
            A[p] = under[p] > MASK_BG || ink[p] || unknown?.[p] ? NaN : (P[p] - under[p]) / (1 - under[p]);
        // (unmasked, the rings are opaque black: αk fitted freely went to 1.26, a thin blurred annulus standing in
        // for a sharp one, and the fit stacked dots to make up the difference)
        const ring = measureRing(A, width, height, used, darkness);
        const fits = ring.rms <= (opts.thick ? THICK_RING_MAX_RMS * ring.alpha : maskInk ? MAX_RING_RMS : BLACK_RING_MAX_RMS);
        return ring.alpha >= MIN_RING_ALPHA && fits && ring.rings >= ISO_MIN ? { A, ring } : null;
    };
    const found = opts.thick ? read(false, null) : read(true) ?? (opts.blackRings && ringPeak(P, width, height, used) >= MASK_INK ? read(false) : null);
    if (!found)
        return null;
    const { A, ring } = found;
    if (opts.hollow && !opts.hollow(used, ring))
        return null;
    const D = new Float32Array(P.length);
    const weight = new Uint8Array(P.length);
    for (let p = 0; p < P.length; p++) {
        weight[p] = Number.isNaN(A[p]) ? 0 : 1;
        D[p] = weight[p] ? Math.max(0, A[p]) / ring.alpha : NaN;
    }
    // (the pairs come from ring candidates, a ring score per pixel, and a hollow ring has no fill to measure)
    if (opts.hollow)
        return { P, under, A, D, ring, fillAlpha: 0, fillAlphaMeasured: false, pairs: 0, isolated: used };
    const cands = ringCenters(D, weight, width, height, ring.radius, ring.inner, PAIR_MIN_ARC);
    const fa = measureFillAlpha(A, width, height, cands, ring);
    return { P, under, A, D, ring, fillAlpha: fa.fillAlpha, fillAlphaMeasured: fa.measured, pairs: fa.pairs, isolated: used };
}

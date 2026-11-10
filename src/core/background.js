// Shaded regions (PLAN.md §2.3, phase 5 milestone 3): CI bands, ellipses, violin shading. A region is a large
// flat area of one color B₂ that isn't a dot family's. It lies either under the dots (drawn first: a dot over it is
// c + (B₂ − c)·T, on the line from B₂ toward c) or over them (drawn last as a translucent layer: what's under it,
// U, shows as B₂ + s·(U − B), with s = 1 − the layer's α, so a dot's offset from B keeps its direction and shrinks
// by s). The two predict different directions for a dot's offset from B₂ (c − B₂ against c − B), which the flat
// pixels inside the region tell apart.
//
// An overlay is removed before anything else: U = B + (I − B₂) / s inside it. A region under the dots is left to
// the per-pixel background of unmix.js. Each pixel belongs to the region of its nearest background-colored pixel,
// so a dot inside a band is in the band, and a dot across its edge is split along it.

import { nearestTransform, opening } from './layers.js';

const REGION_MIN_PIXELS = 1500; // flat pixels of one color for a region (bench3's ANOVA band: two boxes of about 2600)
const REGION_COLOR_TOL = 3; // 0–255 units (RGB distance): a pixel this close to a region's color is that color
const ON_LINE_REL = 0.2; // a color within this × its distance along a family's line is that family's
const ON_LINE_ABS = 4; // 0–255 units
const MODE_MIN_T = 15; // 0–255 units: region pixels at least this far from B₂ show a dot's offset
const MODE_MIN_SHARE = 0.6; // of the region's dot pixels that one mode must explain better, or the region is masked
const MODE_MIN_VOTES = 50; // flat dot pixels in the region needed to decide (fibers' band: 3)
// A region is an area, not a stroke: its pixels must hold discs this many dot radii across. dog-ages' black curve
// and text (52 000 flat pixels after upscaling) are strokes.
const REGION_OPEN_RADII = 0.75;
const S_BIN = 0.02; // histogram bin for the overlay's s
const S_MAX = 0.95; // an "overlay" that keeps more than this of what's under it isn't one (nicest language's white
                    // page read as one with s = 1)

function sub(a, b) {
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

function dot(a, b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

// distance of the offset e from the line through 0 with unit direction u, and the position along it
function offLine(e, u) {
    const t = dot(e, u);
    return { t, perp: Math.sqrt(Math.max(0, dot(e, e) - t * t)) };
}

function unit(v) {
    const n = Math.hypot(...v);
    return v.map(x => x / n);
}

/**
 * Colors of large flat areas that aren't the background or a dot family's.
 *
 * @param {import('./image.js').RgbaImage} image
 * @param {number[]} B
 * @param {{color: number[], alpha: number}[]} families  full color and α (1 for opaque)
 * @param {Uint8Array} flat  style.flatMask
 * @param {number} radius  px, the dots' radius
 * @returns {number[][]}
 */
export function regionColors(image, B, families, flat, radius) {
    const counts = new Map();
    const { width, height, data } = image;
    for (let p = 0; p < flat.length; p++) {
        if (!flat[p])
            continue;
        const key = (data[4 * p] << 16) | (data[4 * p + 1] << 8) | data[4 * p + 2];
        counts.set(key, (counts.get(key) ?? 0) + 1);
    }
    const dirs = families.map(f => unit(sub(f.color, B)));
    const isFamily = c => families.some((f, k) => {
        if (Math.hypot(...sub(c, f.color)) <= REGION_COLOR_TOL * 3)
            return true;
        if (f.alpha >= 1)
            return false;
        const { t, perp } = offLine(sub(c, B), dirs[k]);
        return t > 0 && (perp <= ON_LINE_ABS || perp <= ON_LINE_REL * t);
    });
    // a dot over or under an accepted region (orange dots under a blue band are a flat color of their own)
    const isDotOn = (c, B2) => families.some(f => [sub(f.color, B2), sub(f.color, B)].some(d => {
        const { t, perp } = offLine(sub(c, B2), unit(d));
        return t > 0 && (perp <= ON_LINE_ABS || perp <= ON_LINE_REL * t);
    }));
    const out = [];
    for (const [key, count] of [...counts].sort((a, b) => b[1] - a[1])) {
        if (count < REGION_MIN_PIXELS)
            break;
        const c = [key >> 16, (key >> 8) & 255, key & 255];
        if (Math.hypot(...sub(c, B)) <= REGION_COLOR_TOL || isFamily(c) || out.some(o => Math.hypot(...sub(c, o)) <= REGION_COLOR_TOL || isDotOn(c, o)))
            continue;
        const mask = new Uint8Array(width * height);
        for (let p = 0; p < mask.length; p++)
            mask[p] = Math.hypot(data[4 * p] - c[0], data[4 * p + 1] - c[1], data[4 * p + 2] - c[2]) <= REGION_COLOR_TOL ? 1 : 0;
        const wide = opening(mask, width, height, REGION_OPEN_RADII * radius);
        let area = 0;
        for (const v of wide)
            area += v;
        if (area >= REGION_MIN_PIXELS)
            out.push(c);
    }
    return out;
}

/**
 * The shaded regions and which pixels belong to them.
 *
 * @param {import('./image.js').RgbaImage} image
 * @param {number[]} B
 * @param {{color: number[], alpha: number}[]} families
 * @param {Uint8Array} flat
 * @param {number} radius  px, the dots' radius
 * @returns {{regions: {color: number[], mode: 'over'|'under'|'unknown', s: number|null, pixels: number}[],
 *            label: Uint8Array}}  label: 0 the main background, k the region k − 1
 */
export function backgroundRegions(image, B, families, flat, radius) {
    const { width, height, data } = image;
    const n = width * height;
    const colors = regionColors(image, B, families, flat, radius);
    const label = new Uint8Array(n);
    if (!colors.length)
        return { regions: [], label };
    // seeds: pixels of the background or a region color; every pixel takes its nearest seed's region
    const all = [B, ...colors];
    const seedOf = new Int8Array(n).fill(-1);
    const seeds = new Uint8Array(n);
    for (let p = 0; p < n; p++) {
        const c = [data[4 * p], data[4 * p + 1], data[4 * p + 2]];
        const k = all.findIndex(b => Math.hypot(...sub(c, b)) <= REGION_COLOR_TOL);
        if (k >= 0) {
            seedOf[p] = k;
            seeds[p] = 1;
        }
    }
    const nearest = nearestTransform(seeds, width, height);
    for (let p = 0; p < n; p++)
        label[p] = nearest[p] >= 0 ? seedOf[nearest[p]] : 0;
    const regions = colors.map((c, i) => regionMode(image, B, c, families, flat, label, i + 1));
    return { regions, label };
}

// over or under, from the flat pixels inside the region that a dot shows in: which direction their offsets from B₂
// take, c − B₂ (under) or c − B (over); and for an overlay, s from the commonest offset against one dot's
function regionMode(image, B, B2, families, flat, label, k) {
    const { data } = image;
    let under = 0;
    let over = 0;
    let pixels = 0;
    const sHist = new Map();
    const dUnder = families.map(f => unit(sub(f.color, B2)));
    const dOver = families.map(f => unit(sub(f.color, B)));
    // one dot's offset from B along c − B: the first level of a translucent family, the color itself if opaque
    const t1 = families.map(f => Math.hypot(...sub(f.color, B)) * Math.min(1, f.alpha));
    for (let p = 0; p < label.length; p++) {
        if (label[p] !== k)
            continue;
        pixels++;
        if (!flat[p])
            continue;
        const e = sub([data[4 * p], data[4 * p + 1], data[4 * p + 2]], B2);
        if (Math.hypot(...e) < MODE_MIN_T)
            continue;
        let bestU = Infinity;
        let bestO = Infinity;
        let bestF = -1;
        families.forEach((f, j) => {
            const u = offLine(e, dUnder[j]);
            const o = offLine(e, dOver[j]);
            if (u.t > 0)
                bestU = Math.min(bestU, u.perp / u.t);
            if (o.t > 0 && o.perp / o.t < bestO) {
                bestO = o.perp / o.t;
                bestF = j;
            }
        });
        if (Math.min(bestU, bestO) > ON_LINE_REL)
            continue;
        if (bestU <= bestO) {
            under++;
        }
        else {
            over++;
            const s = offLine(e, dOver[bestF]).t / t1[bestF];
            const bin = Math.round(s / S_BIN);
            sHist.set(bin, (sHist.get(bin) ?? 0) + 1);
        }
    }
    const shown = under + over;
    let mode = 'unknown';
    if (shown >= MODE_MIN_VOTES && under >= MODE_MIN_SHARE * shown)
        mode = 'under';
    else if (shown >= MODE_MIN_VOTES && over >= MODE_MIN_SHARE * shown)
        mode = 'over';
    let s = null;
    if (mode === 'over') {
        // the commonest scaled offset is one dot under the layer (deeper stacks are rarer)
        const [bin] = [...sHist].sort((a, b) => b[1] - a[1])[0];
        s = bin * S_BIN;
        if (s > S_MAX) {
            mode = 'unknown';
            s = null;
        }
    }
    return { color: B2, mode, s, pixels, votes: { under, over } };
}

/**
 * The image with overlays removed: inside a region drawn over the dots, U = B + (I − B₂) / s. Pixels on the
 * region's edge are only partly under the layer, and inverting them as if wholly under it overshoots (bench1
 * smoother-ci: slivers where the band's edge crosses a dot became (255, 214, 118), an "opaque family" that took
 * every orange dot a second time), so they are left as they are, for the masks.
 *
 * @param {import('./image.js').RgbaImage} image
 * @param {number[]} B
 * @param {ReturnType<typeof backgroundRegions>} layers
 * @returns {import('./image.js').RgbaImage}
 */
export function removeOverlays(image, B, layers) {
    const { width, height, data } = image;
    const out = Uint8ClampedArray.from(data);
    const { label } = layers;
    for (let p = 0; p < width * height; p++) {
        const r = label[p] ? layers.regions[label[p] - 1] : null;
        if (!r || r.mode !== 'over')
            continue;
        const x = p % width;
        const edge = (x > 0 && label[p - 1] !== label[p]) || (x < width - 1 && label[p + 1] !== label[p])
            || (p >= width && label[p - width] !== label[p]) || (p + width < label.length && label[p + width] !== label[p]);
        if (edge)
            continue;
        for (let ch = 0; ch < 3; ch++)
            out[4 * p + ch] = Math.round(B[ch] + (data[4 * p + ch] - r.color[ch]) / r.s);
    }
    return { width, height, data: out };
}

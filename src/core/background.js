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
import { components } from './shape.js';
import { colorMask, colorDistance, sub, dot, unit } from './image.js';

const REGION_MIN_PIXELS = 1500; // flat pixels of one color for a region (bench3's ANOVA band: two boxes of about 2600)
const REGION_COLOR_TOL = 3; // 0–255 units (RGB distance): a pixel this close to a region's color is that color
const BAND_COLOR_TOL = 6; // 0–255 units (RGB distance) from a family's band (its first level's color as measured)
const ON_LINE_REL = 0.2; // a color within this × its distance along a family's line is that family's
const ON_LINE_ABS = 4; // 0–255 units
const MODE_MIN_T = 15; // 0–255 units: region pixels at least this far from B₂ show a dot's offset
const MODE_MIN_SHARE = 0.6; // of the region's dot pixels that one mode must explain better, or the region is masked
// flat dot pixels in the region needed to decide (fibers' band: 3). Small blurred dots have few flat pixels each:
// lung-tumor's 22 dots under its band, 6 px across inside their rings, give 24, every one for a layer over them (at
// 50 the band was masked with its dots).
const MODE_MIN_VOTES = 20;
// An opaque dot over a region shows its own color, all the way along the line from the region's color B₂ to it; a
// flat pixel short of that, on the line, is a dot seen through a layer. Where the page, the layer and the dots are
// one gray, both lines are one and only this tells (bench-r2 gg-loess: black dots, 54 under a band of 192 with
// s = 0.6, 0.73 of the way from it to black; its 'over' had come from the dots' color measured at 2, 2, 2). Only
// when the regions are told from the page: told from violin-beeswarm's violin, the page is a region whose pixels
// include the violins' second shade, and that shade, short of the dots on the line from white, made the page a
// layer over the dots (16784 votes to 1138, s 0.66; 0 found right).
const OPAQUE_UNDER_SHARE = 0.9; // of the way from B₂ to an opaque family's color
// A pixel of a dot under a layer lies on the line from the layer's color B₂ along c − B, and one beside the layer on
// the line from B: where it is on one (ON_LINE_ABS) and this far off the other, its own color says which side it is
// on, whatever seed is nearest. Only within LABEL_REACH dot radii of a pixel of the layer's own color.
const LABEL_OFF_LINE = 8; // 0–255 units
const LABEL_REACH = 2; // × the dots' radius
// A region is an area, not a stroke: its pixels must hold discs this many dot radii across. dog-ages' black curve
// and text (52 000 flat pixels after upscaling) are strokes.
const REGION_OPEN_RADII = 0.75;
// A flat color on the line from a region along a family's offset is that family's dots on the region only where
// the region is beside it: a dot under a layer, or over a box, is surrounded by it, so most of its flat pixels lie
// within DOT_ON_REACH dot radii of the region's own color. In a gray theme every gray is on the line from the white
// page margin toward black dots, and bench-r2 gg-loess's gray band over its dots (192, 9592 flat pixels) was read
// as such dots: 0.03 of it lies that near the margin. On the bench's images, the colors this keeps are 0.09 near or
// less, and the dots it still rejects 0.86 or more. Only when the regions are told from the page, where a layer
// over the dots is found (families.js shadedRegions): told from the dots' own background, a second shade of
// violin-beeswarm's violins (0.2 near the page) became a region under the dots, with 18 extras along its error bars.
const DOT_ON_REACH = 1.5; // × the dots' radius
const DOT_ON_SHARE = 0.5; // of the color's flat pixels
// Two bands drawn one over the other overlap in a color of their own: the upper band's B₂ plus s times the lower's
// offset from B (step1: the blue band, 181, 207, 240, over the red, 237, 190, 179: 169, 165, 191 = blue + 0.65·(red −
// B) in each channel, 2482 flat pixels; the other order fits no channel). The upper band is then a layer over
// everything, with that s, whatever the dots' votes say (a band in the dots' own hue can't be told by them), and the
// overlap is a region of its own, so its pixels don't vote for the lower band (step1's red band: 1388 "under" to 248,
// most from the overlap).
const REGION_RIM_OPENING = 1; // px: thinner runs of a region's color are rims, not the region (bench3 ANOVA: its
                              // band shows between packed dots in slivers that 2 px removed)
const REGION_SEED_TOL = 10; // 0–255 units from a region's color, for its seeds (JPEG noise: step1's band 4–8 off)
const STACK_TOL = 4; // 0–255 units off the line from the upper band along the lower's offset
const STACK_MIN_PIXELS = 1000; // flat pixels at one s
const STACK_S_RANGE = [0.05, 0.95];
const STACK_MIN_SEPARATION = 15; // 0–255 units from B and every region color
// A box under the dots (style.js boxBand) holds its rectangle whatever seed is nearer: a dot as wide as the box is
// tall split it, and pixels in the dot nearer the page above or below than the box's pieces beside it were read
// against the page (protrusion1: boxes 37 px tall, dots 39 px across). Outside it, nearest seeds decide as before
// (a box can be hidden in part: protrusion2's lower half, under a stack). Pieces of its color that share their extent across one axis
// (within RECT_TOL) and lie within RECT_MAX_GAP times that extent of each other along the other are one box; one
// inside another's rectangle is part of it.
const RECT_TOL = 2; // px
const RECT_MAX_GAP = 3; // × the shared extent
const RECT_MIN_PIXELS = 10; // pieces smaller than this are ignored
const RECT_COLOR_TOL = 3; // 0–255 units (RGB distance) from the box's color
const S_BIN = 0.02; // histogram bin for the overlay's s
const S_MAX = 0.95; // an "overlay" that keeps more than this of what's under it isn't one (nicest language's white
                    // page read as one with s = 1)
const S_MIN = 0.1; // … nor one that keeps less than this: nothing shows through it to vote (crimson: 28 pixels of
                   // black ink in a dark gray region of 35, 35, 35 read as dots under a layer with s = 0.04)

// distance of the offset e from the line through 0 with unit direction u, and the position along it
function offLine(e, u) {
    const t = dot(e, u);
    return { t, perp: Math.sqrt(Math.max(0, dot(e, e) - t * t)) };
}

/**
 * Colors of large flat areas that aren't the background or a dot family's.
 *
 * @param {import('./image.js').RgbaImage} image
 * @param {number[]} B
 * @param {{color: number[], alpha: number, band?: number[], box?: boolean}[]} families  full color and α (1 for opaque);
 *   band: a region color in a translucent family's own hue; box: the band is a box under the dots (style.js boxBand)
 * @param {Uint8Array} flat  style.flatMask
 * @param {number} radius  px, the dots' radius
 * @param {boolean} [page]  B is the page's color: a color off the regions found isn't their dots (DOT_ON_REACH)
 * @returns {number[][]}
 */
export function regionColors(image, B, families, flat, radius, page = false) {
    const counts = new Map();
    const { width, height, data } = image;
    for (let p = 0; p < flat.length; p++) {
        if (!flat[p])
            continue;
        const key = (data[4 * p] << 16) | (data[4 * p + 1] << 8) | data[4 * p + 2];
        counts.set(key, (counts.get(key) ?? 0) + 1);
    }
    const dirs = families.map(f => unit(sub(f.color, B)));
    // (a band in a translucent family's own hue lies on its line but is no level of it: style.js withoutBand)
    const isFamily = c => families.some((f, k) => {
        if (colorDistance(c, f.color) <= REGION_COLOR_TOL * 3)
            return true;
        if (f.band && colorDistance(c, f.band) <= BAND_COLOR_TOL)
            return false;
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
    const beside = (mask, o) => {
        const nearest = nearestTransform(colorMask(image, o, REGION_COLOR_TOL), width, height);
        const reach = DOT_ON_REACH * radius;
        let near = 0;
        let all = 0;
        for (let p = 0; p < mask.length; p++) {
            if (!mask[p] || !flat[p])
                continue;
            all++;
            const q = nearest[p];
            if (q < 0)
                continue;
            const dx = (q % width) - (p % width);
            const dy = Math.floor(q / width) - Math.floor(p / width);
            if (dx * dx + dy * dy <= reach * reach)
                near++;
        }
        return near >= DOT_ON_SHARE * all;
    };
    const out = [];
    for (const [key, count] of [...counts].sort((a, b) => b[1] - a[1])) {
        if (count < REGION_MIN_PIXELS)
            break;
        const c = [key >> 16, (key >> 8) & 255, key & 255];
        if (colorDistance(c, B) <= REGION_COLOR_TOL || isFamily(c) || out.some(o => colorDistance(c, o) <= REGION_COLOR_TOL))
            continue;
        const mask = colorMask(image, c, REGION_COLOR_TOL);
        if (out.some(o => isDotOn(c, o) && (!page || beside(mask, o))))
            continue;
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
 * @param {{page?: boolean}} [opts]  page: B is the page's color (DOT_ON_REACH, OPAQUE_UNDER_SHARE)
 * @returns {{regions: {color: number[], mode: 'over'|'under'|'unknown', s: number|null, pixels: number,
 *            base?: number[], stack?: number[], stackedOver?: number, box?: boolean}[], label: Uint8Array,
 *            sure: Uint8Array}}  label: 0 the main
 *   background, k the region k − 1; an overlap of two bands (stack: the upper's and lower's indices) comes after the
 *   region colors, with the base its pixels are undone to; sure: 1 where a pixel's own color says it lies under the
 *   layer it is labeled with (labelByColor)
 */
export function backgroundRegions(image, B, families, flat, radius, opts = {}) {
    const { width, height, data } = image;
    const n = width * height;
    const colors = regionColors(image, B, families, flat, radius, !!opts.page);
    const label = new Uint8Array(n);
    const sure = new Uint8Array(n);
    if (!colors.length)
        return { regions: [], label, sure };
    const stacks = stackedRegions(image, B, colors, flat);
    // seeds: pixels of the background or a region color; every pixel takes its nearest seed's region. Region seeds
    // are the regions' bodies: a dot's antialiased rim passes through a band's color, and as seeds its rim pixels
    // took the dot's middle for the band (step1: a dot beside the blue band, 149, 185, 234, divided out to 178,
    // 178, 204, off its family's line and masked); JPEG noise puts a band's pixels 4–8 units off its color.
    const all = [B, ...colors, ...stacks.map(st => st.color)];
    const seedOf = new Int8Array(n).fill(-1);
    const seeds = new Uint8Array(n);
    const near = new Int8Array(n).fill(-1);
    const inRegion = new Uint8Array(n);
    for (let p = 0; p < n; p++) {
        const c = [data[4 * p], data[4 * p + 1], data[4 * p + 2]];
        if (colorDistance(c, B) <= REGION_COLOR_TOL) {
            seedOf[p] = 0;
            seeds[p] = 1;
            continue;
        }
        let best = -1;
        let bestD = REGION_SEED_TOL;
        for (let k = 1; k < all.length; k++) {
            const d = colorDistance(c, all[k]);
            if (d <= bestD) {
                best = k;
                bestD = d;
            }
        }
        near[p] = best;
        inRegion[p] = best > 0 ? 1 : 0;
    }
    const body = opening(inRegion, width, height, REGION_RIM_OPENING);
    // a box's rectangle is the box's, whatever seed is nearer
    const isBox = colors.map(c => families.some(f => f.box && colorDistance(f.band, c) <= BAND_COLOR_TOL));
    const rects = colors.map((c, i) => (isBox[i] ? boxRects(image, c, width, height) : []));
    for (let p = 0; p < n; p++) {
        if (body[p]) {
            seedOf[p] = near[p];
            seeds[p] = 1;
        }
    }
    const nearest = nearestTransform(seeds, width, height);
    for (let p = 0; p < n; p++)
        label[p] = nearest[p] >= 0 ? seedOf[nearest[p]] : 0;
    rects.forEach((rs, i) => {
        for (const r of rs) {
            for (let y = r.y0; y <= r.y1; y++) {
                for (let x = r.x0; x <= r.x1; x++)
                    label[y * width + x] = i + 1;
            }
        }
    });
    const regions = colors.map((c, i) => ({ ...regionMode(image, B, c, families, flat, label, i + 1, !!opts.page), ...(isBox[i] ? { box: true } : {}) }));
    if (!stacks.length) {
        regions.forEach((r, i) => {
            if (r.mode === 'over')
                labelByColor(image, B, r, families, seeds, near, label, sure, i + 1, LABEL_REACH * radius);
        });
    }
    const ownBand = c => families.some(f => f.band && !f.box && colorDistance(f.band, c) <= BAND_COLOR_TOL);
    for (const st of stacks) {
        // Two families' own bands: both are regions under the dots, and the overlap a third, in the color it shows. A
        // family's dots compose with a band of their own color in either order, so their votes on it are noise
        // (bread2's blue band: under by 1887 to 129 as Node decodes the JPEG, not so as Chrome does, a unit apart in a
        // sixth of the pixels). Dividing the upper band out needs its s, which the overlap's color gives poorly when
        // the lower band is faint (0.78 from an overlap 2 units lighter than the bands' colors predict, where 0.85
        // lets through the 15% the bands are drawn at; undone with it, the dots under the band came out a tenth too
        // deep, and the scan of the undone image found one family of the three, at α 0.81 for 0.51).
        if (ownBand(colors[st.top]) && ownBand(colors[st.bottom])) {
            for (const k of [st.top, st.bottom])
                regions[k] = { ...regions[k], mode: 'under', s: null };
            regions.push({ color: st.color, mode: 'under', s: null, pixels: 0 });
            continue;
        }
        regions[st.top] = { ...regions[st.top], mode: 'over', s: st.s, stackedOver: st.bottom };
        const low = regions[st.bottom];
        // under both: what is under the lower band too if it is a layer, else the lower band with the dots over it
        regions.push(low.mode === 'over'
            ? { color: st.color, mode: 'over', s: st.s * low.s, base: B, pixels: 0, stack: [st.top, st.bottom] }
            : { color: st.color, mode: 'over', s: st.s, base: low.color, pixels: 0, stack: [st.top, st.bottom] });
    }
    for (let p = 0; p < n; p++) {
        if (label[p] > colors.length)
            regions[label[p] - 1].pixels++;
    }
    return { regions, label, sure };
}

// Where a layer over the dots is narrower than a dot, or shows only in strips beside a line drawn along it, it has no
// seed near the dots under it and they were read as dots on the page: left as they were, off their family's line,
// and masked (lung-tumor: a confidence band 10 px across at its waist with the regression line through it, 8 dots of
// 6.8 px radius under it). And a dot beside a band, nearer its seeds than the page's, was divided out with it. The
// pixels of region k and of the page within `reach` px of a pixel of the region's color (`near`: its strips count,
// which are no seeds) are labeled again by their colors (LABEL_OFF_LINE); the rest keep their nearest seed's region.
// `sure` is set where the color says under the layer: such a pixel is wholly under it, even at the region's edge
// (removeOverlays).
function labelByColor(image, B, region, families, seeds, near, label, sure, k, reach) {
    const { width, height, data } = image;
    const own = new Uint8Array(label.length);
    for (let p = 0; p < label.length; p++)
        own[p] = near[p] === k ? 1 : 0;
    // a family's two lines are parallel, apart by the layer's color's distance from the family's line from B: where
    // they are too close the colors can't tell (mpo: black dots, gray boxes, every color on one line: the boxes'
    // lighter edges went to the page and the dots across them were fitted twice, 56 found for 45)
    const dirs = families.map(f => unit(sub(f.color, B))).filter(d => offLine(sub(region.color, B), d).perp > LABEL_OFF_LINE);
    if (!dirs.length)
        return;
    const nearest = nearestTransform(own, width, height);
    // the least distance of an offset from the families' lines, among those it lies along by MODE_MIN_T or more
    const off = e => Math.min(...dirs.map(d => {
        const { t, perp } = offLine(e, d);
        return t >= MODE_MIN_T ? perp : Infinity;
    }));
    for (let p = 0; p < label.length; p++) {
        if (seeds[p] || (label[p] !== 0 && label[p] !== k) || nearest[p] < 0)
            continue;
        const q = nearest[p];
        if (Math.hypot(p % width - q % width, Math.floor(p / width) - Math.floor(q / width)) > reach)
            continue;
        const c = [data[4 * p], data[4 * p + 1], data[4 * p + 2]];
        const page = off(sub(c, B));
        const under = off(sub(c, region.color));
        if (under <= ON_LINE_ABS && page > LABEL_OFF_LINE) {
            label[p] = k;
            sure[p] = 1;
        }
        else if (page <= ON_LINE_ABS && under > LABEL_OFF_LINE)
            label[p] = 0;
    }
}

// The rectangles of the pieces of a box's color (within RECT_COLOR_TOL: two dots' overlap 6 units off made pieces
// outside protrusion2's green box), merged as RECT_TOL and RECT_MAX_GAP say. Inclusive pixel bounds.
function boxRects(image, color, width, height) {
    const mask = colorMask(image, color, RECT_COLOR_TOL);
    let rects = components(mask, width, height).filter(c => c.length >= RECT_MIN_PIXELS).map(c => {
        const r = { x0: width, y0: height, x1: -1, y1: -1 };
        for (const p of c) {
            const x = p % width;
            const y = (p - x) / width;
            r.x0 = Math.min(r.x0, x);
            r.x1 = Math.max(r.x1, x);
            r.y0 = Math.min(r.y0, y);
            r.y1 = Math.max(r.y1, y);
        }
        return r;
    });
    const near1 = (a, b) => Math.abs(a - b) <= RECT_TOL;
    const gap = (a0, a1, b0, b1) => Math.max(0, Math.max(a0, b0) - Math.min(a1, b1) - 1);
    const joins = (a, b) => {
        const inside = (u, v) => u.x0 >= v.x0 && u.x1 <= v.x1 && u.y0 >= v.y0 && u.y1 <= v.y1;
        if (inside(a, b) || inside(b, a))
            return true;
        if (near1(a.y0, b.y0) && near1(a.y1, b.y1))
            return gap(a.x0, a.x1, b.x0, b.x1) <= RECT_MAX_GAP * (a.y1 - a.y0 + 1);
        if (near1(a.x0, b.x0) && near1(a.x1, b.x1))
            return gap(a.y0, a.y1, b.y0, b.y1) <= RECT_MAX_GAP * (a.x1 - a.x0 + 1);
        return false;
    };
    for (let merged = true; merged;) {
        merged = false;
        for (let i = 0; i < rects.length && !merged; i++) {
            for (let j = i + 1; j < rects.length; j++) {
                if (!joins(rects[i], rects[j]))
                    continue;
                const a = rects[i];
                const b = rects[j];
                rects[i] = { x0: Math.min(a.x0, b.x0), y0: Math.min(a.y0, b.y0), x1: Math.max(a.x1, b.x1), y1: Math.max(a.y1, b.y1) };
                rects = rects.filter((_, m) => m !== j);
                merged = true;
                break;
            }
        }
    }
    return rects;
}

// Pairs of region colors where one band lies over the other: flat pixels on B₂ᵗᵒᵖ + s·(B₂ᵇᵒᵗᵗᵒᵐ − B), a
// STACK_MIN_PIXELS peak in s. O(regions² · pixels).
function stackedRegions(image, B, colors, flat) {
    const { data } = image;
    const out = [];
    colors.forEach((top, i) => colors.forEach((bottom, j) => {
        if (i === j || out.some(st => st.top === j && st.bottom === i))
            return;
        const d = sub(bottom, B);
        const dd = dot(d, d);
        const hist = new Map();
        for (let p = 0; p < flat.length; p++) {
            if (!flat[p])
                continue;
            const e = sub([data[4 * p], data[4 * p + 1], data[4 * p + 2]], top);
            const sp = dot(e, d) / dd;
            if (sp <= STACK_S_RANGE[0] || sp >= STACK_S_RANGE[1] || Math.hypot(e[0] - sp * d[0], e[1] - sp * d[1], e[2] - sp * d[2]) > STACK_TOL)
                continue;
            const bin = Math.round(sp / S_BIN);
            const h = hist.get(bin) ?? { count: 0, sum: [0, 0, 0] };
            h.count++;
            h.sum[0] += data[4 * p];
            h.sum[1] += data[4 * p + 1];
            h.sum[2] += data[4 * p + 2];
            hist.set(bin, h);
        }
        // the peak, with its neighbors (a noisy s straddles bins)
        let best = null;
        for (const [bin, h] of hist) {
            const count = h.count + (hist.get(bin - 1)?.count ?? 0) + (hist.get(bin + 1)?.count ?? 0);
            if (!best || count > best.count)
                best = { bin, count };
        }
        if (!best || best.count < STACK_MIN_PIXELS)
            return;
        const h = hist.get(best.bin);
        const color = h.sum.map(v => v / h.count);
        // an overlap is a color of its own (kite1: its page and a white region made a "stack" whose overlap was the
        // page again, at s 0.13)
        if ([B, ...colors].some(c => colorDistance(c, color) < STACK_MIN_SEPARATION))
            return;
        out.push({ top: i, bottom: j, color, s: dot(sub(color, top), d) / dd });
    }));
    return out;
}

// over or under, from the flat pixels inside the region that a dot shows in: which direction their offsets from B₂
// take, c − B₂ (under) or c − B (over); and for an overlay, s from the commonest offset against one dot's
function regionMode(image, B, B2, families, flat, label, k, page) {
    const { data } = image;
    let under = 0;
    let over = 0;
    let pixels = 0;
    const sHist = new Map();
    const dUnder = families.map(f => unit(sub(f.color, B2)));
    const dOver = families.map(f => unit(sub(f.color, B)));
    // one dot's offset from B along c − B: the first level of a translucent family, the color itself if opaque
    const t1 = families.map(f => colorDistance(f.color, B) * Math.min(1, f.alpha));
    const whole = families.map(f => (page && f.alpha >= 1 ? OPAQUE_UNDER_SHARE * colorDistance(f.color, B2) : 0));
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
            if (u.t > 0 && u.t >= whole[j])
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
    // A box plot's box in its dots' own hue (style.js boxBand) is under them: c − B₂ and c − B nearly agree, so the
    // dots' colors can't tell (protrusion3: 7.7° and 6.8°, over by a hair), but its outline can, lightened where a dot
    // crosses it (19, 20, 19 to 87, 94, 87)
    if (families.some(f => f.box && colorDistance(f.band, B2) <= BAND_COLOR_TOL))
        mode = 'under';
    else if (shown >= MODE_MIN_VOTES && under >= MODE_MIN_SHARE * shown)
        mode = 'under';
    else if (shown >= MODE_MIN_VOTES && over >= MODE_MIN_SHARE * shown)
        mode = 'over';
    let s = null;
    if (mode === 'over') {
        // the commonest scaled offset is one dot under the layer (deeper stacks are rarer)
        const [bin] = [...sHist].sort((a, b) => b[1] - a[1])[0];
        s = bin * S_BIN;
        if (s > S_MAX || s < S_MIN) {
            mode = 'unknown';
            s = null;
        }
    }
    return { color: B2, mode, s, pixels, votes: { under, over } };
}

/**
 * The image with overlays removed: inside a region drawn over the dots, U = B + (I − B₂) / s (an overlap of two bands:
 * its base, B or the lower band's color, for B). Pixels on the
 * region's edge are only partly under the layer, and inverting them as if wholly under it overshoots (bench1
 * smoother-ci: slivers where the band's edge crosses a dot became (255, 214, 118), an "opaque family" that took
 * every orange dot a second time), so they are left as they are, for the masks, but for those whose own colors say
 * they are under the layer (layers.sure).
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
        if (edge && !layers.sure[p])
            continue;
        const base = r.base ?? B;
        for (let ch = 0; ch < 3; ch++)
            out[4 * p + ch] = Math.round(base[ch] + (data[4 * p + ch] - r.color[ch]) / r.s);
    }
    return { width, height, data: out };
}

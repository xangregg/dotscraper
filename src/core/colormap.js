// Dots colored by a continuous color scale (PLAN.md phase 5, milestone 2): every dot has its own fill color, so
// there is no color line to project on. But a translucent dot keeps its offset's direction from the background:
// one dot of color c over B is B + α(c − B), a stack of n is B + (1 − (1 − α)^n)(c − B), and a dot's rim mixes it
// with B along the same ray. So a pixel's direction from B names its dot's color, and its distance from B, divided
// by one dot's distance in that direction (m₁), counts the dots: 1, 2 − α, … (bench1 and bench2 gradient-dots:
// medians 1.49–1.50 at two dots, 1.72–1.77 at three, against 1.5 and 1.75 at α 0.5, from the truth). Stacks of
// unlike colors point between their colors, within 0.02–0.04 of the one-dot curve, close enough to read.
//
// The one-dot distances form a curve over directions: flat pixels near the background are the outer parts of
// dots, one deep (lenses lie inside), so per direction cell the median distance of those is m₁. A coincident pair
// also borders the background, and a cell made only of one reads 2 − α times its neighbors (bench2: 123 and 122
// against 85); such cells are dropped. Each pixel's m₁ is a kernel regression over the cells, so it varies smoothly
// with direction.
//
// Directions are read from the sum of the offsets over a small window, not from the pixel alone: inside a dot the
// direction is constant and a rim doesn't change it, so the window only averages away noise, and JPEG chroma noise
// otherwise moved pixels along the curve to the wrong m₁ (bench2 gradient-dots.jpg: the one-dot level 64–96).
//
// The image is then rewritten as one virtual family: every pixel near the curve becomes B + r·(P − B), with r its
// count ratio and P one real one-dot color, so the single-family pipeline (levels, isolated dots, radius, Hough,
// fit) runs on it unchanged. Pixels off the curve (text, axes, lines) are left as they are, for the masks.
//
// Opaque dots of a scale (no second level) are rewritten as one virtual opaque family instead (opaqueImage).
//
// Color scales are rare, so this stays a side path: it assumes no other family shares the plot, doesn't report
// each dot's color, and fits JMP's outlined markers (an outline-colored disc under a smaller fill) as one disc at
// the outline's radius. A categorical palette (many colors, its votes in peaks) is fitted the same way, and its dots'
// colors are read from their own pixels after the fit (paletteColors).

import { colorDistance } from './image.js';
import { upperMedian } from './stats.js';

const WINDOW = 2; // px: half side of the window whose summed offsets give a pixel's direction
const MIN_OFFSET = 15; // 0–255 units from B: flat pixels at least this far vote for the one-dot curve
const CURVE_CELL = 0.02; // direction cell (unit-vector components)
// Flat pixels this close to a background pixel are the outer, one-dot part of a dot. An image enlarged from a small
// raster has edges wider than that and no such pixel anywhere, of dots or of text (shap: edges about 6 px wide, no
// curve point at 4 px, 77 at 8); the wider reaches are tried only then.
const RIM_REACHES = [4, 8, 12]; // px
const CELL_MIN_VOTES = 5; // pixels for a cell to be a curve point
const NEIGHBOR_DIST = 0.08; // between unit directions: cells this close are neighbors along the curve
const MAX_NEIGHBOR_RATIO = 1.2; // a cell farther than this × its neighbors' median m₁ is a deeper stack
const KERNEL_WIDTH = 0.025; // sd of the direction kernel for m₁
const OFF_CURVE = 0.06; // a pixel farther than this from every curve point is not a dot of the scale
const MIN_PIXEL_OFFSET = 1; // 0–255 units: closer to B than this is B
// Curve points link into a chain within LINK_DIST of each other and with one-dot distances within LINK_MAX_RATIO:
// a scale's extremes have few dots, and gaps (bench2's one darkest dot is 0.116 from the next color), while other
// inks differ in distance or direction (black text on white: 2.7 times the darkest dot's m₁; gray: 0.29 off).
const LINK_DIST = 0.15;
const LINK_MAX_RATIO = 1.35;
// A color scale, not a few families: the largest chain spans at least this (bench gradients: about 0.6; one
// translucent family: under 0.05 in a PNG).
const MIN_SPREAD = 0.2;
const MIN_POINTS = 8;
// … and no point holds most of its votes: a scale's colors are many (bench1–2 gradients: at most 11% in one point),
// while one family's color with JPEG noise around it is one point and a fringe (beeswarm-boxplot: gray, 97% of the
// votes, with 15 noise points of 5–27 votes reaching a spread of 0.23)
const MAX_POINT_SHARE = 0.5;
// … nor do a few: discrete families in a few hues chain through the mixes of their rims with lines crossing them
// (protrusion1: green and lavender dots on gray whiskers, 0.92 of the votes in 3 points, linked by points of 5–89
// votes; scales: 0.23–0.36, bench2 group-graystroke 0.48)
const FEW_POINTS = 3;
const MAX_FEW_SHARE = 0.7;
// Many in peaks is a categorical palette, not a scale: its colors chain through the mixes of overlapping dots of
// unlike colors, but its votes pile up at its colors. Peaks are points with the most votes within NEIGHBOR_DIST; a
// palette has PALETTE_MIN_PEAKS or more and its top PALETTE_TOP_POINTS of the points hold over PALETTE_TOP_SHARE of
// the votes (bench-local wages2025, 22 colors: 26 peaks, 0.87; scales: 3–6 peaks, 0.32–0.41 for bench1–2's
// gradients, 0.61 for score2's viridis). It is fitted as a scale is, one virtual family, which reads overlaps of
// unlike colors; split one at a time its 22 colors gave 9 families and took 6 minutes against 1. Each fitted dot's
// color is then its own pixels' (paletteColors).
const PALETTE_MIN_PEAKS = 10;
const PALETTE_TOP_POINTS = 0.1;
const PALETTE_TOP_SHARE = 0.75;
// a palette's color outside the chain counts when this far from B, × the nearest of the chain's peaks
const PALETTE_MIN_M_RATIO = 0.5;
// A palette's colors from the fitted dots (paletteColors): their own pixels this far inside them and beyond the others'
// rims, the white rims (wages2025: about 2 px) and blur left out …
const PALETTE_INSET = 3; // px
const PALETTE_MIN_OWN = 20; // px of them for a dot's color to count
// … if one color: this share of them within PALETTE_COLOR_TOL of their median
const PALETTE_UNIFORM = 0.9;
// … one color when this close (wages2025: each color's dots within 0–11 of its median, the two nearest colors 17.6
// apart, both blues), and shared by this many dots at least
const PALETTE_COLOR_TOL = 8; // 0–255 units
const PALETTE_MIN_DOTS = 2;
// α from the ratios of flat pixels: a histogram peak at 1 (one dot, by construction) and one at 2 − α
const RATIO_BIN = 0.01;
const RATIO_SMOOTH = 1.5; // bins, sd
const RATIO_MAX = 3;
const SECOND_RANGE = [1.05, 1.95]; // where 2 − α can be (α 0.05–0.95)
const FIRST_FALL = 0.1; // the first peak ends where the histogram falls below this fraction of it
const SECOND_MIN_FRACTION = 0.01; // of the first peak's pixels, for the second level to count
const SECOND_MIN_PIXELS = 10;
const PEAK_CENTROID = 3; // bins either side of the second peak averaged for its position
// Opaque dots of a scale show no second level (alpha null). With no level to confirm them, the chain must span
// more: the bench gradients span 0.54–0.60, while bench1 smoother-ci.jpg's band edges chain to 0.215.
const MIN_OPAQUE_SPREAD = 0.4;
// An opaque scale with few dots has gaps in its colors, each dot a point of its own, and they don't all chain
// (potato: 86 dots on viridis, a chain of 10 points from purple to teal and the greens and yellow in four chains of
// one). A point as far from B as the chain's (LINK_MAX_RATIO beyond their range either way) is one of its colors,
// unless it is gray: text and lines (black is this far from B as well, and within GRAY_DIST of the gray direction;
// viridis's darkest purple is 0.17 from it).
const GRAY_DIST = 0.1; // between unit directions
const OPAQUE_WINDOW = 2; // px: half side of the window searched for a rim pixel's full color
// A pixel is partial only next to one this much lighter than its full color: the background, or the middle of a
// narrow gap between dots. A fill beside its own outline is not (the lightest JMP fill is 0.61 of its outline's
// offset, bench2 gradient-opaque-stroke).
const LIGHT = 0.45;
// A rim pixel and its full color are this close in direction, plus the rounding of the rim pixel's offset of m
// units, about ROUNDING / m (bench2 gradient-opaque: (253, 247, 245) is 0.045 from its fill (218, 131, 89)).
const SAME_DIRECTION = 0.03; // between unit directions
const ROUNDING = 1; // 0–255 units
// Outline pixels off the curve: within this of a curve pixel, and nearer the curve than this
const OUTLINE_REACH = 3; // px
const OUTLINE_OFF_CURVE = 0.25; // between unit directions

/**
 * Per pixel, the summed offsets from B over a (2·WINDOW + 1)² window, as unit directions (0 where they cancel).
 * O(pixels) with box sums.
 */
function directionField(image, B) {
    const { width, height, data } = image;
    const n = width * height;
    const rows = new Float32Array(3 * n);
    const out = new Float32Array(3 * n);
    for (let y = 0; y < height; y++) {
        for (let ch = 0; ch < 3; ch++) {
            let s = 0;
            for (let x = -WINDOW; x < width + WINDOW; x++) {
                const add = x + WINDOW;
                if (add < width)
                    s += data[4 * (y * width + add) + ch] - B[ch];
                const drop = x - WINDOW - 1;
                if (drop >= 0)
                    s -= data[4 * (y * width + drop) + ch] - B[ch];
                if (x >= 0 && x < width)
                    rows[3 * (y * width + x) + ch] = s;
            }
        }
    }
    for (let x = 0; x < width; x++) {
        for (let ch = 0; ch < 3; ch++) {
            let s = 0;
            for (let y = -WINDOW; y < height + WINDOW; y++) {
                const add = y + WINDOW;
                if (add < height)
                    s += rows[3 * (add * width + x) + ch];
                const drop = y - WINDOW - 1;
                if (drop >= 0)
                    s -= rows[3 * (drop * width + x) + ch];
                if (y >= 0 && y < height)
                    out[3 * (y * width + x) + ch] = s;
            }
        }
    }
    for (let p = 0; p < n; p++) {
        const l = Math.hypot(out[3 * p], out[3 * p + 1], out[3 * p + 2]);
        if (l > 0) {
            out[3 * p] /= l;
            out[3 * p + 1] /= l;
            out[3 * p + 2] /= l;
        }
    }
    return out;
}

// m₁ at direction u by kernel regression over the curve points, and the distance to the nearest point
function oneDot(points, u0, u1, u2) {
    let nearest = Infinity;
    let sw = 0;
    let sm = 0;
    for (const q of points) {
        const d2 = (q.u[0] - u0) ** 2 + (q.u[1] - u1) ** 2 + (q.u[2] - u2) ** 2;
        if (d2 < nearest)
            nearest = d2;
        const w = q.votes * Math.exp(-d2 / (2 * KERNEL_WIDTH * KERNEL_WIDTH));
        sw += w;
        sm += w * q.m;
    }
    return { m1: sw > 0 ? sm / sw : NaN, dist: Math.sqrt(nearest) };
}

// The chain of points that is a scale's, and how far apart its ends are; null when there is none. Points chain by
// LINK_DIST and LINK_MAX_RATIO, and other inks form chains of their own (the title's black strokes). The chains are
// taken by their votes. One that is a single color (fewer than MIN_POINTS points, a span under MIN_SPREAD, or
// MAX_POINT_SHARE of its votes in one point) is other ink and passed over: light gridlines hold more votes than a
// scale's dots (potato: the gray of 246, 4823 votes in one point, against 559 in the 10 points of its dots'
// colors). The first chain of many colors decides: a scale, unless FEW_POINTS of its points hold MAX_FEW_SHARE of
// its votes, which is a few families (protrusion1, where a smaller chain of their rims' mixes then read as a scale).
function scaleChain(points) {
    const label = new Int32Array(points.length).fill(-1);
    const chains = [];
    for (let s = 0; s < points.length; s++) {
        if (label[s] >= 0)
            continue;
        const members = [s];
        label[s] = chains.length;
        for (let k = 0; k < members.length; k++) {
            const i = members[k];
            for (let j = 0; j < points.length; j++) {
                const a = points[i];
                const b = points[j];
                if (label[j] >= 0 || colorDistance(a.u, b.u) > LINK_DIST
                    || Math.max(a.m, b.m) > LINK_MAX_RATIO * Math.min(a.m, b.m))
                    continue;
                label[j] = chains.length;
                members.push(j);
            }
        }
        chains.push(members.map(i => points[i]));
    }
    const votes = c => c.reduce((sum, q) => sum + q.votes, 0);
    const spreadOf = c => Math.max(0, ...c.flatMap(a => c.map(b => colorDistance(a.u, b.u))));
    const oneColor = c => c.length < MIN_POINTS || spreadOf(c) < MIN_SPREAD || Math.max(...c.map(q => q.votes)) > MAX_POINT_SHARE * votes(c);
    const best = chains.filter(c => !oneColor(c)).sort((a, b) => votes(b) - votes(a))[0];
    if (!best)
        return null;
    const few = best.map(q => q.votes).sort((a, b) => b - a).slice(0, FEW_POINTS).reduce((t, v) => t + v, 0);
    return few <= MAX_FEW_SHARE * votes(best) ? { points: best, spread: spreadOf(best) } : null;
}

/**
 * A palette's colors, from fitted dots: each dot's own color is the median of its pixels PALETTE_INSET inside it and
 * as far beyond every other dot's rim (one dot over the background: its one-dot color, exact in a PNG), taken from
 * dots with PALETTE_MIN_OWN such pixels, PALETTE_UNIFORM of them that color: a fitted dot partly over a missed one,
 * or on text or a line, isn't one color (wages2025: 38 colors for 22, among them two-dot stacks, gridline grays and
 * the median line's purple; testing the colors against the one-dot curve instead lost both grays, whose direction
 * the subtitle's slate text holds). Colors within PALETTE_COLOR_TOL of the commonest are one, greedily,
 * while PALETTE_MIN_DOTS share one. Each dot's family is then paletteFamilies'. O(dots² + dots · r²) per color.
 *
 * @param {import('./image.js').RgbaImage} image  the original RGB
 * @param {{x: number, y: number}[]} dots  px
 * @param {number} radius  px
 * @param {number[]} B  the background, 0–255
 * @returns {number[][]}  0–255, the commonest first
 */
export function paletteColors(image, dots, radius, B) {
    const { width, height, data } = image;
    const rIn = radius - PALETTE_INSET;
    const colors = [];
    for (const d of dots) {
        const others = dots.filter(o => o !== d && Math.hypot(o.x - d.x, o.y - d.y) < 2 * radius);
        const own = [[], [], []];
        for (let y = Math.max(0, Math.floor(d.y - rIn)); y < Math.min(height, Math.ceil(d.y + rIn)); y++) {
            for (let x = Math.max(0, Math.floor(d.x - rIn)); x < Math.min(width, Math.ceil(d.x + rIn)); x++) {
                const cx = x + 0.5;
                const cy = y + 0.5;
                if ((cx - d.x) ** 2 + (cy - d.y) ** 2 > rIn * rIn || others.some(o => Math.hypot(cx - o.x, cy - o.y) < radius + PALETTE_INSET))
                    continue;
                const p = 4 * (y * width + x);
                for (let ch = 0; ch < 3; ch++)
                    own[ch].push(data[p + ch]);
            }
        }
        const n = own[0].length;
        if (n < PALETTE_MIN_OWN)
            continue;
        const c = own.map(upperMedian);
        let same = 0;
        for (let i = 0; i < n; i++) {
            if (Math.hypot(own[0][i] - c[0], own[1][i] - c[1], own[2][i] - c[2]) <= PALETTE_COLOR_TOL)
                same++;
        }
        if (same >= PALETTE_UNIFORM * n && colorDistance(c, B) >= MIN_OFFSET)
            colors.push(c);
    }
    const close = (a, b) => colorDistance(a, b) <= PALETTE_COLOR_TOL;
    const out = [];
    let left = colors;
    while (left.length) {
        let best = null;
        for (const c of left) {
            const n = left.filter(o => close(o, c)).length;
            if (!best || n > best.n)
                best = { c, n };
        }
        if (best.n < PALETTE_MIN_DOTS)
            break;
        const members = left.filter(o => close(o, best.c));
        out.push([0, 1, 2].map(ch => upperMedian(members.map(c => c[ch]))));
        left = left.filter(o => !close(o, best.c));
    }
    return out;
}

/**
 * Each fitted dot's palette color: its pixels PALETTE_INSET inside it and beyond the other fitted dots' rims vote
 * for the palette color they are within PALETTE_COLOR_TOL of. Missed dots cover some of those pixels, so their
 * median is often a two-dot stack, whose nearest color is another family's (wages2025: 41 of 652 matched dots in the
 * wrong family that way); where the dot shows alone its color is exact. Sure with PALETTE_MIN_OWN votes; else the
 * color with the most, marked unsure (none: the nearest to the pixels' median, unsure). O(dots · (neighbors + r² ·
 * colors)).
 *
 * @param {import('./image.js').RgbaImage} image  the original RGB
 * @param {{x: number, y: number}[]} dots  px
 * @param {number} radius  px
 * @param {number[][]} colors  from paletteColors
 * @returns {{family: number, sure: boolean}[]}
 */
export function paletteFamilies(image, dots, radius, colors) {
    const { width, height, data } = image;
    const rIn = radius - PALETTE_INSET;
    return dots.map(d => {
        const others = dots.filter(o => o !== d && Math.hypot(o.x - d.x, o.y - d.y) < 2 * radius);
        const votes = new Array(colors.length).fill(0);
        const seen = [[], [], []];
        for (let y = Math.max(0, Math.floor(d.y - rIn)); y < Math.min(height, Math.ceil(d.y + rIn)); y++) {
            for (let x = Math.max(0, Math.floor(d.x - rIn)); x < Math.min(width, Math.ceil(d.x + rIn)); x++) {
                const cx = x + 0.5;
                const cy = y + 0.5;
                if ((cx - d.x) ** 2 + (cy - d.y) ** 2 > rIn * rIn || others.some(o => Math.hypot(cx - o.x, cy - o.y) < radius + PALETTE_INSET))
                    continue;
                const p = 4 * (y * width + x);
                for (let ch = 0; ch < 3; ch++)
                    seen[ch].push(data[p + ch]);
                const k = colors.findIndex(c => Math.hypot(data[p] - c[0], data[p + 1] - c[1], data[p + 2] - c[2]) <= PALETTE_COLOR_TOL);
                if (k >= 0)
                    votes[k]++;
            }
        }
        const most = Math.max(...votes);
        if (most > 0)
            return { family: votes.indexOf(most), sure: most >= PALETTE_MIN_OWN };
        // no pixel shows a palette color: the nearest to their median, unsure
        if (!seen[0].length)
            return { family: 0, sure: false };
        const med = seen.map(upperMedian);
        const dist = c => colorDistance(c, med);
        return { family: colors.reduce((best, c, k) => (dist(c) < dist(colors[best]) ? k : best), 0), sure: false };
    });
}

// points with the most votes within NEIGHBOR_DIST of them. O(points²).
function peaksOf(points) {
    const dist = (a, b) => colorDistance(a.u, b.u);
    return points.filter(q => points.every(o => o === q || o.votes < q.votes || dist(o, q) >= NEIGHBOR_DIST));
}

// Whether a chain's votes are a categorical palette's (see PALETTE_MIN_PEAKS): many peaks, most votes in few points.
// O(points²).
function isPalette(points, votes) {
    const peaks = peaksOf(points).length;
    const sorted = points.map(q => q.votes).sort((a, b) => b - a);
    const top = sorted.slice(0, Math.ceil(PALETTE_TOP_POINTS * sorted.length)).reduce((t, v) => t + v, 0);
    return peaks >= PALETTE_MIN_PEAKS && top > PALETTE_TOP_SHARE * votes;
}

/**
 * α from the count ratios of flat pixels on the curve: the second histogram peak sits at 2 − α.
 *
 * @param {number[]} ratios  each flat pixel's distance from B over its direction's one-dot distance
 * @returns {number|null}
 */
function ratioAlpha(ratios) {
    const nb = Math.ceil(RATIO_MAX / RATIO_BIN);
    const hist = new Float64Array(nb + 1);
    for (const r of ratios) {
        if (r < RATIO_MAX)
            hist[Math.round(r / RATIO_BIN)]++;
    }
    const half = Math.ceil(3 * RATIO_SMOOTH);
    const smooth = new Float64Array(nb + 1);
    for (let b = 0; b <= nb; b++) {
        for (let k = -half; k <= half; k++) {
            if (b + k >= 0 && b + k <= nb)
                smooth[b] += hist[b + k] * Math.exp(-k * k / (2 * RATIO_SMOOTH * RATIO_SMOOTH));
        }
    }
    const one = Math.round(1 / RATIO_BIN);
    let first = one;
    for (let b = one - 10; b <= one + 10; b++) {
        if (smooth[b] > smooth[first])
            first = b;
    }
    let end = first;
    while (end < nb && smooth[end] >= FIRST_FALL * smooth[first])
        end++;
    let firstCount = 0;
    for (let b = 2 * first - end; b <= end; b++)
        firstCount += hist[b];
    const lo = Math.max(end, Math.round(SECOND_RANGE[0] / RATIO_BIN));
    const hi = Math.round(SECOND_RANGE[1] / RATIO_BIN);
    let second = -1;
    for (let b = lo; b <= hi; b++) {
        if (smooth[b] >= smooth[b - 1] && smooth[b] >= smooth[b + 1] && (second < 0 || smooth[b] > smooth[second]))
            second = b;
    }
    if (second < 0)
        return null;
    let count = 0;
    let sum = 0;
    for (let b = second - PEAK_CENTROID; b <= second + PEAK_CENTROID; b++) {
        count += hist[b];
        sum += hist[b] * b * RATIO_BIN;
    }
    if (count < Math.max(SECOND_MIN_PIXELS, SECOND_MIN_FRACTION * firstCount))
        return null;
    const r1 = first * RATIO_BIN;
    return 2 - sum / count / r1;
}

/**
 * A family colored by a continuous scale, if the image has one: translucent with its α, or opaque (alpha null) when
 * the dots show no second level.
 *
 * @param {import('./image.js').RgbaImage} image
 * @param {number[]} B  the background, 0–255
 * @param {Uint8Array} flat  style.flatMask
 * @param {import('./scan.js').Tolerances} tol
 * @returns {{points: {u: number[], m: number, votes: number}[], alpha: number|null, spread: number,
 *            dir: Float32Array}|null}  points: the one-dot curve (unit direction from B, distance m₁ in 0–255
 *            units); dir: each pixel's unit direction (3 per pixel)
 */
export function findColormap(image, B, flat, tol) {
    const { width, height, data } = image;
    const n = width * height;
    const nearB = Math.max(tol.nearB, 3);
    const background = new Uint8Array(n);
    for (let p = 0; p < n; p++)
        background[p] = Math.hypot(data[4 * p] - B[0], data[4 * p + 1] - B[1], data[4 * p + 2] - B[2]) <= nearB ? 1 : 0;
    const dir = directionField(image, B);
    let all = [];
    for (let i = 0; i < RIM_REACHES.length && !all.length; i++)
        all = curvePoints(image, B, flat, background, nearReach(background, width, height, RIM_REACHES[i]), dir);
    const scale = scaleChain(all);
    if (!scale)
        return null;
    let points = scale.points;
    const spread = scale.spread;
    const votes = points.reduce((t, q) => t + q.votes, 0);
    const palette = isPalette(points, votes);
    // a palette's colors needn't chain: one far from the others in direction, with few dots, links to none (wages2025:
    // pink, 8 dots, 0.38 from the nearest color); it is as far from B as they are, a light line not (the gridlines'
    // gray, at 0.28 of the nearest peak's distance; pink 0.77)
    if (palette) {
        const minM = Math.min(...peaksOf(points).map(q => q.m));
        points = [...points, ...all.filter(q => !points.includes(q) && q.m >= PALETTE_MIN_M_RATIO * minM)];
    }
    const ratios = [];
    for (let p = 0; p < n; p++) {
        if (!flat[p] || background[p])
            continue;
        const m = Math.hypot(data[4 * p] - B[0], data[4 * p + 1] - B[1], data[4 * p + 2] - B[2]);
        if (m < MIN_OFFSET)
            continue;
        const { m1, dist } = oneDot(points, dir[3 * p], dir[3 * p + 1], dir[3 * p + 2]);
        if (dist <= OFF_CURVE)
            ratios.push(m / m1);
    }
    const alpha = ratioAlpha(ratios);
    if (alpha === null && spread < MIN_OPAQUE_SPREAD)
        return null;
    if (alpha === null && !palette) {
        const ms = points.map(q => q.m);
        const gray = -1 / Math.sqrt(3);
        const far = q => q.m >= Math.min(...ms) / LINK_MAX_RATIO && q.m <= LINK_MAX_RATIO * Math.max(...ms);
        const colored = q => colorDistance(q.u, [gray, gray, gray]) >= GRAY_DIST;
        points = [...points, ...all.filter(q => !points.includes(q) && far(q) && colored(q))];
    }
    return { points, alpha, spread, dir, palette };
}

// The one-dot curve's points: per direction cell with CELL_MIN_VOTES flat pixels within reach of the background, its
// mean direction and the median of their distances from B; cells of deeper stacks dropped (MAX_NEIGHBOR_RATIO).
function curvePoints(image, B, flat, background, reach, dir) {
    const { data } = image;
    const cells = new Map();
    for (let p = 0; p < flat.length; p++) {
        if (!flat[p] || background[p] || !reach[p])
            continue;
        const m = Math.hypot(data[4 * p] - B[0], data[4 * p + 1] - B[1], data[4 * p + 2] - B[2]);
        if (m < MIN_OFFSET)
            continue;
        const u = [dir[3 * p], dir[3 * p + 1], dir[3 * p + 2]];
        const key = u.map(v => Math.round(v / CURVE_CELL)).join(',');
        let c = cells.get(key);
        if (!c) {
            c = { sum: [0, 0, 0], ms: [] };
            cells.set(key, c);
        }
        for (let ch = 0; ch < 3; ch++)
            c.sum[ch] += u[ch];
        c.ms.push(m);
    }
    const points = [];
    for (const c of cells.values()) {
        if (c.ms.length < CELL_MIN_VOTES)
            continue;
        const l = Math.hypot(...c.sum);
        points.push({ u: c.sum.map(v => v / l), m: upperMedian(c.ms), votes: c.ms.length });
    }
    return points.filter((q, i) => {
        const near = points.filter((o, j) => j !== i && colorDistance(o.u, q.u) < NEIGHBOR_DIST).map(o => o.m);
        return !near.length || q.m <= MAX_NEIGHBOR_RATIO * upperMedian(near);
    });
}

// 1 where a background pixel lies within `reach` px (a square neighborhood; cheap and close enough)
function nearReach(background, width, height, reach) {
    const out = new Uint8Array(background.length);
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            if (!background[y * width + x])
                continue;
            for (let yy = Math.max(0, y - reach); yy <= Math.min(height - 1, y + reach); yy++) {
                for (let xx = Math.max(0, x - reach); xx <= Math.min(width - 1, x + reach); xx++)
                    out[yy * width + xx] = 1;
            }
        }
    }
    return out;
}

/**
 * The image as one virtual translucent family: each pixel near the curve becomes B + r·(P − B), with r its distance
 * from B over m₁ in its direction and P the most common one-dot color; other pixels are left as they are.
 *
 * @param {import('./image.js').RgbaImage} image
 * @param {number[]} B
 * @param {ReturnType<typeof findColormap>} cm
 * @returns {{image: import('./image.js').RgbaImage, P: number[]}}  P: the virtual family's one-dot color
 */
export function rayImage(image, B, cm) {
    const { width, height, data } = image;
    const out = Uint8ClampedArray.from(data);
    const ref = cm.points.reduce((a, b) => (b.votes > a.votes ? b : a));
    const refE = ref.u.map(v => v * ref.m);
    const { dir } = cm;
    for (let p = 0; p < width * height; p++) {
        const m = Math.hypot(data[4 * p] - B[0], data[4 * p + 1] - B[1], data[4 * p + 2] - B[2]);
        if (m < MIN_PIXEL_OFFSET)
            continue;
        const { m1, dist } = oneDot(cm.points, dir[3 * p], dir[3 * p + 1], dir[3 * p + 2]);
        if (dist > OFF_CURVE)
            continue;
        const r = m / m1;
        for (let ch = 0; ch < 3; ch++)
            out[4 * p + ch] = Math.round(B[ch] + r * refE[ch]);
    }
    return { image: { width, height, data: out }, P: B.map((v, ch) => v + refE[ch]) };
}

/**
 * Opaque dots of a scale as one virtual opaque family: each dot pixel becomes B + a·(P − B), with a its coverage.
 * A ray from B can't give the coverage here: JMP's outline of a light fill points the way the fill does but lies
 * farther out (bench2 gradient-opaque-stroke: yellow fill 155 from white, its outline 255, 0.01 apart in direction),
 * so a full fill and a partial outline look alike. Position settles it instead: a pixel is partial only next to a
 * much lighter one (the background, or the middle of a narrow gap between dots), and then its full color is the
 * farthest pixel nearby in its direction; or where it is itself much lighter than the scale's color in its direction
 * (LIGHT of it: beyond what an outline's or a fill's own color can be). Every other dot pixel, a fill, an outline, or
 * the boundary between two dots, is covered. The result is a one-color rendering of the dots' union.
 *
 * @param {import('./image.js').RgbaImage} image
 * @param {number[]} B
 * @param {ReturnType<typeof findColormap>} cm  with alpha null
 * @returns {{image: import('./image.js').RgbaImage, P: number[]}}  P: the virtual family's color
 */
export function opaqueImage(image, B, cm) {
    const { width, height, data } = image;
    const n = width * height;
    const out = Uint8ClampedArray.from(data);
    const ref = cm.points.reduce((a, b) => (b.votes > a.votes ? b : a));
    const P = B.map((v, ch) => v + ref.u[ch] * ref.m);
    const off = new Float32Array(3 * n);
    const m = new Float32Array(n);
    for (let p = 0; p < n; p++) {
        for (let ch = 0; ch < 3; ch++)
            off[3 * p + ch] = data[4 * p + ch] - B[ch];
        m[p] = Math.hypot(off[3 * p], off[3 * p + 1], off[3 * p + 2]);
    }
    const { dots, m1 } = dotPixels(m, cm, width, height);
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const p = y * width + x;
            if (!dots[p])
                continue;
            const tolDir = SAME_DIRECTION + ROUNDING / m[p];
            let full = m[p];
            for (let yy = Math.max(0, y - OPAQUE_WINDOW); yy <= Math.min(height - 1, y + OPAQUE_WINDOW); yy++) {
                for (let xx = Math.max(0, x - OPAQUE_WINDOW); xx <= Math.min(width - 1, x + OPAQUE_WINDOW); xx++) {
                    const q = yy * width + xx;
                    if (m[q] <= full)
                        continue;
                    let d2 = 0;
                    for (let ch = 0; ch < 3; ch++)
                        d2 += (off[3 * q + ch] / m[q] - off[3 * p + ch] / m[p]) ** 2;
                    if (d2 <= tolDir * tolDir)
                        full = m[q];
                }
            }
            // (a light gridline beside a dot takes the dot's direction from the window, and is the fullest pixel of
            // its own direction there: potato, whiskers of full color along the gridlines at a dot's rim, and lone
            // dots fitted as two and three)
            const a = m[p] <= LIGHT * m1[p] ? m[p] / Math.max(full, m1[p])
                : lightest(m, width, height, x, y) <= LIGHT * full ? m[p] / full : 1;
            for (let ch = 0; ch < 3; ch++)
                out[4 * p + ch] = Math.round(B[ch] + a * (P[ch] - B[ch]));
        }
    }
    return { image: { width, height, data: out }, P };
}

// `dots`: 1 for the pixels of opaque dots, those on the curve, and those near it within OUTLINE_REACH of one. Outlines
// of dark fills lie as far from the curve as black text does (bench2 gradient-opaque-stroke: a dark red outline
// (109, 47, 45) is 0.15 off, and 0.16 from black), but only outlines touch the fills. `m1`: the scale's distance from
// B in each pixel's direction. O(pixels · reach²).
function dotPixels(m, cm, width, height) {
    const { dir } = cm;
    const n = width * height;
    const dist = new Float32Array(n).fill(Infinity);
    const m1 = new Float32Array(n);
    const on = new Uint8Array(n);
    for (let p = 0; p < n; p++) {
        if (m[p] < MIN_PIXEL_OFFSET)
            continue;
        const curve = oneDot(cm.points, dir[3 * p], dir[3 * p + 1], dir[3 * p + 2]);
        dist[p] = curve.dist;
        m1[p] = curve.m1;
        on[p] = dist[p] <= OFF_CURVE ? 1 : 0;
    }
    const out = Uint8Array.from(on);
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const p = y * width + x;
            if (on[p] || dist[p] > OUTLINE_OFF_CURVE)
                continue;
            search: for (let yy = Math.max(0, y - OUTLINE_REACH); yy <= Math.min(height - 1, y + OUTLINE_REACH); yy++) {
                for (let xx = Math.max(0, x - OUTLINE_REACH); xx <= Math.min(width - 1, x + OUTLINE_REACH); xx++) {
                    if (on[yy * width + xx]) {
                        out[p] = 1;
                        break search;
                    }
                }
            }
        }
    }
    return { dots: out, m1 };
}

// the smallest offset from B among (x, y) and its 8 neighbors
function lightest(m, width, height, x, y) {
    let low = Infinity;
    for (let yy = Math.max(0, y - 1); yy <= Math.min(height - 1, y + 1); yy++) {
        for (let xx = Math.max(0, x - 1); xx <= Math.min(width - 1, x + 1); xx++)
            low = Math.min(low, m[yy * width + xx]);
    }
    return low;
}

/**
 * The virtual family as a candidate for style.estimateStyle: its line toward P, with the levels the colormap's α
 * implies (the level finder reads a compressed image's broad first level as two).
 *
 * @param {import('./levels.js').LevelMeasure} measured  levels.js measureFamily on the virtual image, seeded with P
 * @param {number[]} B  the background, 0–255
 * @param {number[]} P  the virtual family's one-dot color, 0–255
 * @param {number} alpha  0–1, the colormap's
 * @returns {import('./levels.js').LevelMeasure}  as measured, with two levels at 1 and 2 − α dots and the fit they
 *   imply
 */
export function colormapCandidate(measured, B, P, alpha) {
    const level = r => ({ t: r * colorDistance(P, B), color: B.map((v, ch) => v + r * (P[ch] - v)), count: 0 });
    const c = B.map((v, ch) => v + (P[ch] - v) / alpha);
    return {
        ...measured,
        levels: [level(1), level(2 - alpha)],
        fit: { alpha, c, rms: 0, maxAbs: 0, underdetermined: false },
        kind: 'translucent', translucent: true,
    };
}

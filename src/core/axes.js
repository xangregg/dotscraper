// Graph scales (PLAN.md §2.5, stage 1): axis lines, frames and gridlines, tick marks and their label boxes, and
// the mapping from pixels to data once tick values are known.
//
// Structural lines are gray: black or dark gray for axes, frames and ticks, light gray for gridlines, low chroma
// in every test image, while dots, CI bands and violin shading are colored. So a line is a long, thin, straight run
// of gray pixels that differ from the background; dark ones are axis candidates, light ones gridlines. Thinness
// rejects panels and bands of dots; grayness rejects colored fills.
//
// Coordinates follow the rest of the project: pixel (i, j) spans [i, i + 1) × [j, j + 1), so a one-pixel line in row
// j has its center at j + 0.5. Positions of lines and ticks are darkness-weighted centers.

import { components8 } from './layers.js';
import { median } from './stats.js';

const GRAY_CHROMA = 24; // 0–255: max(r, g, b) − min(r, g, b) for a gray pixel
// … or, for a dark pixel, this: dark navy text and ticks are ink as black is (wages2025: 24, 37, 50, chroma 26)
const DARK_GRAY_CHROMA = 40;
const DARK_GRAY_LUMINANCE = 64; // 0–255
const INK_TOL = 15; // 0–255 units (Euclidean RGB) from the background for a pixel to be ink
// Gray ink darker than this is "dark" (axes, ticks); lighter is gridline gray. Measured: axes and frames 0–125 in
// the published figures but 164–165 in JMP's (bench1), gridlines 194–244.
const DARK_LUMINANCE = 180; // 0–255
// A tick axis needs labels: a gray significance bracket on beeswarm-boxplot (luminance 160, 400 px long, its ends
// and middle stroke three "ticks") is otherwise an x axis.
const AXIS_MIN_LABELED_TICKS = 2;
const LINE_MIN_FRACTION = 0.3; // a line's longest run spans at least this fraction of the image
const LINE_GAP = 3; // px: gaps up to this long don't break a run (antialiasing, dots over gridlines)
const LINE_MAX_THICKNESS_FRACTION = 0.02; // of the smaller image dimension …
const LINE_MIN_MAX_THICKNESS = 4; // … but at least this many px
const BAND_EXTENT_TOL = 0.1; // rows of one band have runs agreeing in start and end within this fraction of length
const TICK_MIN_LENGTH = 3; // px
const TICK_MAX_LENGTH_FRACTION = 0.06; // of the image dimension along the tick
const TICK_MAX_WIDTH_THICKNESSES = 3; // a tick is at most this many axis-line thicknesses wide (and ≥ 1 px)
const TICK_LINE_CLEARANCE = 2; // px: a tick within this of a perpendicular line may be that line (an axis corner)
const TICK_MINORITY_RATIO = 0.5; // ticks on the side with fewer than this × the other side's count are dropped
const CORNER_SPACING_TOL = 0.03; // × spacing: a line at the axis's end this close to the ticks' progression is a tick
// A tick is a straight stroke, about as long at its sides as at its middle; a dot touching a line is rounded, longest
// in the middle (boxdots1: dots along a box's edge, 4 of 14 and 11 of 22 px, made it an x axis whose "labels" were
// dots). Measured, the median over an axis's labeled ticks: 0.8–1.0 on every real axis (step1's 5 px ticks 0.8).
const TICK_MIN_STRAIGHTNESS = 0.6; // shortest ÷ longest stroke across a tick's width
// A tick label lies just beyond its tick in plain view of the axis; "labels" beyond other ink are marks inside the plot
// (boxdots1: dots above a filled box read as labels of its lower edge, 0.62 and 0.74 of the way to them inked by the
// fill). Measured, the median over an axis's labels: 0.00 on every real axis, single labels up to 0.84.
const LABEL_PATH_MAX_INK = 0.5; // share of pixels between the axis and a label, across the label's extent
// How far beyond a tick to look for its label, as a fraction of the image dimension across the axis. y labels are
// horizontal text that grows with its digits: at 8% of the width dog-ages-actual-size's "100" lost its "1".
const LABEL_SEARCH_FRACTION = { h: 0.08, v: 0.15 };
const LABEL_MIN_PIXELS = 3; // ink components smaller than this are noise
const LABEL_COLUMN_MIN_TOL = 4; // px, least allowance for a label's gap beyond the nearest label's
const WORD_GAP_HEIGHTS = 0.6; // glyphs closer than this × their height (along the axis) are one word …
// … if most of their heights overlap: glyphs of a word share a baseline, while chaining through tall glyphs merged
// pathogen's tick labels with its axis title, and cantons' "200" with its rotated title
const WORD_MIN_OVERLAP = 0.5; // × the shorter glyph's height
const WORD_MAX_HEIGHT_RATIO = 2.5; // glyphs of one word differ in height by less than this (a tall rule doesn't join)
const LABEL_ON_TICK_SPACINGS = 0.3; // a word belongs to a tick only if centered within this × spacing of it …
const LABEL_BEYOND_TOL = 2; // px: … and not wholly beyond the axis line's ends by more than this
// Labels on the side away from the ticks must label at least this many (beeswarm-boxplot: the "***" above a gray
// significance bracket labeled 2 of its 3 down-pointing ends and made it an x axis) …
const OTHER_SIDE_MIN_LABELS = 3;
// … with words no wider along the axis than this many tick spacings: data beside a sparsely labeled axis can
// outnumber its labels (economist: dots "labeled" 3 ticks above its baseline, one of them a chain of dots 566 px wide
// at a 74 px spacing, against "2007" and "2023" below; the plot was bounded above the baseline and every dot masked).
// No ratio of the two sides' counts separates that from dots beside 4 of an inward frame's 6 ticks (6 to 4).
const OTHER_SIDE_MAX_WIDTH = 1.5; // × the tick spacing
const BACKGROUND_MIN_FRACTION = 0.05; // any flat color covering this share of the image is a background (panels)
const GRID_BASELINE_TOL = 0.03; // × spacing: a dark baseline this close to the gridlines' progression is a tick
const GRID_LABEL_END_FRACTION = 0.2; // gridline labels sit within this share of the line's length of either end

/**
 * @typedef {object} Line
 * @property {'h'|'v'} dir
 * @property {number} pos        px, center across the line (y for 'h', x for 'v')
 * @property {number} thickness  px
 * @property {number} from       px, start along the line
 * @property {number} to         px, end along the line
 * @property {number} luminance  0–255, mean over the line
 * @property {boolean} dark
 */

/**
 * @typedef {object} Tick
 * @property {number} pos     px, center along the axis
 * @property {number} length  px, the longest stroke across the tick's width
 * @property {number} [shortest]  px, the shortest (a straight tick's is about its length)
 * @property {'before'|'after'} side  'before' = above (x axis) or left (y axis)
 * @property {number[]|null} label  [x0, y0, x1, y1) box of the label's ink, or null
 * @property {number} [labelPath]  0–1, share of ink between the axis and the label
 * @property {'tick'|'grid'} source
 */

function luminance(d, i) {
    return 0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2];
}

/**
 * Background colors: every color covering at least BACKGROUND_MIN_FRACTION of the image (the page and any panel),
 * most common first.
 *
 * @param {import('./image.js').RgbaImage} image
 * @returns {number[][]}
 */
export function backgroundColors(image) {
    const counts = new Map();
    const n = image.width * image.height;
    for (let p = 0; p < n; p++) {
        const key = (image.data[4 * p] << 16) | (image.data[4 * p + 1] << 8) | image.data[4 * p + 2];
        counts.set(key, (counts.get(key) ?? 0) + 1);
    }
    return [...counts].filter(([, c]) => c >= BACKGROUND_MIN_FRACTION * n).sort((a, b) => b[1] - a[1])
        .map(([k]) => [k >> 16, (k >> 8) & 255, k & 255]);
}

function isInk(data, i, backgrounds) {
    return backgrounds.every(B => Math.hypot(data[i] - B[0], data[i + 1] - B[1], data[i + 2] - B[2]) > INK_TOL);
}

/**
 * Per pixel: 0 not structural ink, 1 light gray ink, 2 dark gray ink.
 *
 * @param {import('./image.js').RgbaImage} image
 * @param {number[][]} backgrounds
 * @returns {Uint8Array}
 */
export function grayInk(image, backgrounds) {
    const { data } = image;
    const n = image.width * image.height;
    const out = new Uint8Array(n);
    for (let p = 0; p < n; p++) {
        const i = 4 * p;
        const r = data[i];
        const g = data[i + 1];
        const b = data[i + 2];
        const chroma = Math.max(r, g, b) - Math.min(r, g, b);
        if ((chroma > GRAY_CHROMA && !(chroma <= DARK_GRAY_CHROMA && luminance(data, i) < DARK_GRAY_LUMINANCE)) || !isInk(data, i, backgrounds))
            continue;
        out[p] = luminance(data, i) < DARK_LUMINANCE ? 2 : 1;
    }
    return out;
}

// longest run of ink along one row or column, gaps up to LINE_GAP bridged; returns [start, end) or null
function longestRun(ink, n, at) {
    let best = null;
    let start = -1;
    let lastOn = -1;
    for (let j = 0; j <= n; j++) {
        const on = j < n && ink[at(j)] > 0;
        if (on) {
            if (start < 0 || j - lastOn - 1 > LINE_GAP)
                start = j;
            lastOn = j;
            if (best === null || lastOn + 1 - start > best[1] - best[0])
                best = [start, lastOn + 1];
        }
    }
    return best;
}

/**
 * Long thin straight gray lines, horizontal and vertical.
 *
 * @param {import('./image.js').RgbaImage} image
 * @param {Uint8Array} ink  from grayInk
 * @returns {Line[]}
 */
export function findLines(image, ink) {
    const { width, height, data } = image;
    const maxThick = Math.max(LINE_MIN_MAX_THICKNESS, LINE_MAX_THICKNESS_FRACTION * Math.min(width, height));
    const lines = [];
    for (const dir of ['h', 'v']) {
        const across = dir === 'h' ? height : width;
        const along = dir === 'h' ? width : height;
        const idx = dir === 'h' ? (i, j) => i * width + j : (i, j) => j * width + i;
        const runs = [];
        for (let i = 0; i < across; i++) {
            const run = longestRun(ink, along, j => idx(i, j));
            runs.push(run && run[1] - run[0] >= LINE_MIN_FRACTION * along ? run : null);
        }
        // bands of consecutive qualifying rows whose runs agree
        let i = 0;
        while (i < across) {
            if (!runs[i]) {
                i++;
                continue;
            }
            let j = i + 1;
            while (j < across && runs[j]) {
                const tol = BAND_EXTENT_TOL * (runs[i][1] - runs[i][0]);
                if (Math.abs(runs[j][0] - runs[i][0]) > tol || Math.abs(runs[j][1] - runs[i][1]) > tol)
                    break;
                j++;
            }
            const thickness = j - i;
            if (thickness <= maxThick) {
                const from = Math.min(...runs.slice(i, j).map(r => r[0]));
                const to = Math.max(...runs.slice(i, j).map(r => r[1]));
                // darkness-weighted center across the line; its luminance is that of its darkest row (the rows at
                // the edges of a thin antialiased line are much lighter than its core)
                let w = 0;
                let wpos = 0;
                let lum = 255;
                for (let k = i; k < j; k++) {
                    let rowSum = 0;
                    let rowN = 0;
                    for (let a = from; a < to; a++) {
                        const p = idx(k, a);
                        if (!ink[p])
                            continue;
                        const L = luminance(data, 4 * p);
                        const dark = 255 - L;
                        w += dark;
                        wpos += dark * (k + 0.5);
                        rowSum += L;
                        rowN++;
                    }
                    if (rowN)
                        lum = Math.min(lum, rowSum / rowN);
                }
                lines.push({ dir, pos: wpos / w, thickness, from, to, luminance: lum, dark: lum < DARK_LUMINANCE });
            }
            i = j;
        }
    }
    return lines;
}

/**
 * Tick marks along a dark axis line: short gray strokes perpendicular to it with a dark core, on either side,
 * grouped into ticks. A tick that coincides with a perpendicular line is marked `onLine`: it may be an axis corner
 * (dropped later unless it has a label) or a zero line crossing the axis (Laser: it is the 0 tick).
 *
 * @param {import('./image.js').RgbaImage} image
 * @param {Uint8Array} ink
 * @param {Line} axis
 * @param {Line[]} lines  all lines, to skip perpendicular ones
 * @returns {Tick[]}
 */
export function findTicks(image, ink, axis, lines) {
    const { width, height, data } = image;
    const h = axis.dir === 'h';
    const alongMax = h ? width : height;
    const acrossMax = h ? height : width;
    const at = (a, c) => (h ? c * width + a : a * width + c); // a along the axis, c across it
    const maxLen = TICK_MAX_LENGTH_FRACTION * acrossMax;
    const maxWidth = Math.max(1, TICK_MAX_WIDTH_THICKNESSES * axis.thickness);
    const near = (axis.pos - axis.thickness / 2);
    const far = (axis.pos + axis.thickness / 2);
    const perpendicular = lines.filter(l => l.dir !== axis.dir && l.dark);
    const ticks = [];
    for (const side of ['before', 'after']) {
        const start = side === 'before' ? Math.floor(near) - 1 : Math.ceil(far);
        const step = side === 'before' ? -1 : 1;
        // length of the dark stroke leaving the axis at each position along it
        const len = new Float64Array(alongMax);
        for (let a = Math.max(0, Math.floor(axis.from) - 2); a < Math.min(alongMax, Math.ceil(axis.to) + 2); a++) {
            let c = start;
            let n = 0;
            while (c >= 0 && c < acrossMax && ink[at(a, c)] > 0 && n <= maxLen) {
                n++;
                c += step;
            }
            len[a] = n >= TICK_MIN_LENGTH && n <= maxLen ? n : 0;
        }
        let a = 0;
        while (a < alongMax) {
            if (!len[a]) {
                a++;
                continue;
            }
            let b = a;
            while (b < alongMax && len[b])
                b++;
            if (b - a <= maxWidth) {
                let w = 0;
                let wpos = 0;
                let core = 255;
                for (let k = a; k < b; k++) {
                    // weight by darkness of the stroke's first pixel, to center antialiased edges
                    const L = luminance(data, 4 * at(k, start));
                    core = Math.min(core, L);
                    w += (255 - L) * len[k];
                    wpos += (255 - L) * len[k] * (k + 0.5);
                }
                // a light stroke is a gridline stub (pathogen), not a tick
                if (core < DARK_LUMINANCE) {
                    const pos = wpos / w;
                    const onLine = perpendicular.some(l => Math.abs(l.pos - pos) <= l.thickness / 2 + TICK_LINE_CLEARANCE);
                    ticks.push({ pos, length: Math.max(...len.slice(a, b)), shortest: Math.min(...len.slice(a, b)), side, label: null, source: 'tick', onLine });
                }
            }
            a = b;
        }
    }
    // Ticks drawn across the axis appear on both sides at one position: keep one. Otherwise keep the side with
    // most ticks (dashed reference lines touching an axis from inside made extra "ticks" on beeswarm-boxplot).
    const count = sd => ticks.filter(t => t.side === sd).length;
    const major = count('after') >= count('before') ? 'after' : 'before';
    const minor = major === 'after' ? 'before' : 'after';
    const kept = ticks.filter(t => t.side === major);
    if (count(minor) >= TICK_MINORITY_RATIO * count(major)) {
        for (const t of ticks.filter(t => t.side === minor)) {
            if (!kept.some(k => Math.abs(k.pos - t.pos) <= maxWidth))
                kept.push(t);
        }
    }
    kept.sort((p, q) => p.pos - q.pos);
    const corner = cornerTick(axis, kept, perpendicular, major);
    if (corner)
        kept.push(corner);
    return kept.sort((p, q) => p.pos - q.pos);
}

/**
 * A perpendicular line that starts at the axis, at one end of it, hides the tick there (cells1: the bars' baseline
 * runs from the y axis's bottom end, its "0" tick, to the right). It is that end's tick when it continues the ticks'
 * regular spacing; marked `onLine`, so it is kept only if it gets a label.
 */
function cornerTick(axis, ticks, perpendicular, side) {
    if (ticks.length < 2)
        return null;
    const spacing = (ticks[ticks.length - 1].pos - ticks[0].pos) / (ticks.length - 1);
    const reach = axis.thickness / 2 + TICK_LINE_CLEARANCE;
    for (const l of perpendicular) {
        const touches = Math.abs(l.from - axis.pos) <= reach + l.thickness || Math.abs(l.to - axis.pos) <= reach + l.thickness;
        const atEnd = Math.abs(l.pos - axis.from) <= reach + l.thickness || Math.abs(l.pos - axis.to) <= reach + l.thickness;
        if (!touches || !atEnd || ticks.some(t => Math.abs(t.pos - l.pos) <= reach + l.thickness))
            continue;
        const gap = Math.min(Math.abs(l.pos - ticks[0].pos), Math.abs(l.pos - ticks[ticks.length - 1].pos));
        if (Math.abs(gap - spacing) <= Math.max(1, CORNER_SPACING_TOL * spacing))
            return { pos: l.pos, length: 0, side, label: null, source: 'tick', onLine: true };
    }
    return null;
}

/**
 * Connected components (8-neighbor) of any ink (not just gray: labels can be colored) inside a rectangle.
 */
function inkComponents(image, backgrounds, rect, grayOnly = false) {
    const { width, data } = image;
    const [x0, y0, x1, y1] = rect.map(Math.round);
    const w = x1 - x0;
    const h = y1 - y0;
    if (w <= 0 || h <= 0)
        return [];
    const ink = (x, y) => {
        const i = 4 * (y * width + x);
        if (grayOnly && Math.max(data[i], data[i + 1], data[i + 2]) - Math.min(data[i], data[i + 1], data[i + 2]) > GRAY_CHROMA)
            return false;
        return isInk(data, i, backgrounds);
    };
    const mask = new Uint8Array(w * h);
    for (let y = y0; y < y1; y++) {
        for (let x = x0; x < x1; x++)
            mask[(y - y0) * w + x - x0] = ink(x, y) ? 1 : 0;
    }
    const comps = [];
    for (const c of components8(mask, w, h)) {
        const box = [c.x0 + x0, c.y0 + y0, c.x1 + x0 + 1, c.y1 + y0 + 1];
        // ink touching the image border is a frame or a cut-off mark, never a label (pathogen: a 1 px border
        // at the right edge chained every x label and the axis title into one word)
        const touches = box[0] === 0 || box[1] === 0 || box[2] === image.width || box[3] === image.height;
        if (c.pixels.length >= LABEL_MIN_PIXELS && !touches)
            comps.push({ box, n: c.pixels.length });
    }
    return comps;
}

// glyph boxes merged into words: boxes that overlap by most of their height and are closer than WORD_GAP_HEIGHTS ×
// their height along the reading direction
function mergeWords(boxes) {
    const words = boxes.map(b => [...b]);
    let merged = true;
    while (merged) {
        merged = false;
        for (let i = 0; i < words.length && !merged; i++) {
            for (let j = i + 1; j < words.length && !merged; j++) {
                const a = words[i];
                const b = words[j];
                const height = Math.max(a[3] - a[1], b[3] - b[1]);
                const shorter = Math.min(a[3] - a[1], b[3] - b[1]);
                const overlapY = Math.min(a[3], b[3]) - Math.max(a[1], b[1]);
                const gapX = Math.max(a[0], b[0]) - Math.min(a[2], b[2]);
                if (overlapY >= WORD_MIN_OVERLAP * shorter && gapX <= WORD_GAP_HEIGHTS * height && height <= WORD_MAX_HEIGHT_RATIO * shorter) {
                    words[i] = [Math.min(a[0], b[0]), Math.min(a[1], b[1]), Math.max(a[2], b[2]), Math.max(a[3], b[3])];
                    words.splice(j, 1);
                    merged = true;
                }
            }
        }
    }
    return words;
}

function union(a, b) {
    return a ? [Math.min(a[0], b[0]), Math.min(a[1], b[1]), Math.max(a[2], b[2]), Math.max(a[3], b[3])] : [...b];
}

/**
 * Label boxes for the ticks of an axis: ink beyond the ticks merged into words; a word belongs to the tick it is
 * centered on (within LABEL_ON_TICK_SPACINGS of the spacing). Words between ticks (labels of intervals, as on the
 * economist chart) belong to none. Labels are usually on the ticks' side, but ticks drawn into the plot (Nature's
 * frames) have them on the other side, and their own side holds data (crystal1: dots "labeled" 2 of 5 y ticks and
 * masked the plot as beyond the axes). Both sides are searched and the one labeling more ticks wins; on a tie, the
 * ticks' side. The other side must label at least OTHER_SIDE_MIN_LABELS ticks, with words no wider than
 * OTHER_SIDE_MAX_WIDTH spacings.
 *
 * @param {import('./image.js').RgbaImage} image
 * @param {number[][]} backgrounds
 * @param {Line} axis
 * @param {Tick[]} ticks  modified: each gets `label`
 */
export function findLabels(image, backgrounds, axis, ticks) {
    if (!ticks.length)
        return;
    const tickSide = ticks.filter(t => t.side === 'after').length >= ticks.filter(t => t.side === 'before').length ? 'after' : 'before';
    const tickLen = Math.max(...ticks.map(t => t.length));
    const own = labelsOnSide(image, backgrounds, axis, ticks, tickSide, tickLen);
    const other = labelsOnSide(image, backgrounds, axis, ticks, tickSide === 'after' ? 'before' : 'after', 0, OTHER_SIDE_MAX_WIDTH);
    const far = other.size >= OTHER_SIDE_MIN_LABELS && other.size > own.size;
    for (const [t, { box, path }] of far ? other : own) {
        t.label = box;
        t.labelPath = path;
    }
}

// tick → { box, path } for the words on one side of the axis, beyond `reachFrom` px of tick (0 for the side without
// them), and no wider along the axis than `maxWidth` spacings: the label's box and the share of ink between the axis
// and it
function labelsOnSide(image, backgrounds, axis, ticks, side, reachFrom, maxWidth = Infinity) {
    const { width, height } = image;
    const h = axis.dir === 'h';
    const labels = new Map();
    const reach = LABEL_SEARCH_FRACTION[axis.dir] * (h ? height : width);
    const edge = side === 'after' ? axis.pos + axis.thickness / 2 + reachFrom + 1 : axis.pos - axis.thickness / 2 - reachFrom - 1;
    const c0 = Math.max(0, Math.floor(side === 'after' ? edge : edge - reach));
    const c1 = Math.min(h ? height : width, Math.ceil(side === 'after' ? edge + reach : edge));
    if (c1 <= c0)
        return labels;
    const spacing = ticks.length > 1 ? (ticks[ticks.length - 1].pos - ticks[0].pos) / (ticks.length - 1) : reach;
    const a0 = Math.max(0, Math.floor(ticks[0].pos - spacing / 2));
    const a1 = Math.min(h ? width : height, Math.ceil(ticks[ticks.length - 1].pos + spacing / 2));
    const rect = h ? [a0, c0, a1, c1] : [c0, a0, c1, a1];
    // Tick labels form a column (y) or row (x) at a consistent gap from the ticks; axis titles and other text are
    // farther out. Keep words whose gap is within half a label height of the smallest gap.
    const gapOf = b => (h ? (side === 'after' ? b[1] - edge : edge - b[3]) : (side === 'after' ? b[0] - edge : edge - b[2]));
    // share of ink between the axis (beyond reachFrom) and a word, across the word's extent along the axis
    const pathInk = b => {
        const [a0w, a1w] = h ? [b[0], b[2]] : [b[1], b[3]];
        const [n0, n1] = side === 'after' ? [Math.ceil(edge), h ? b[1] : b[0]] : [h ? b[3] : b[2], Math.floor(edge)];
        let all = 0;
        let ink = 0;
        for (let a = a0w; a < a1w; a++) {
            for (let c = n0; c < n1; c++) {
                all++;
                if (isInk(image.data, 4 * (h ? c * width + a : a * width + c), backgrounds))
                    ink++;
            }
        }
        return all ? ink / all : 0;
    };
    const assigned = [];
    for (const b of mergeWords(inkComponents(image, backgrounds, rect).map(c => c.box))) {
        const mid = h ? (b[0] + b[2]) / 2 : (b[1] + b[3]) / 2;
        if ((h ? b[2] - b[0] : b[3] - b[1]) > maxWidth * spacing)
            continue;
        // a word wholly beyond the axis line's ends is no label of its ticks: the other axis's, at the corner
        // (wages2025: the x axis's "100", under the corner, "labeled" the y axis's corner tick at gap 0, and the y
        // labels, 19 px out, fell outside the column it set). Requiring a word to span its tick lost labels that run
        // off the image (bench1–2 medium-dots-grid2: the last x label's "1" alone, and the gridlines above the axis
        // outnumbered the labels)
        const [lo, hi] = h ? [b[0], b[2]] : [b[1], b[3]];
        if (hi < axis.from - LABEL_BEYOND_TOL || lo > axis.to + LABEL_BEYOND_TOL)
            continue;
        let best = null;
        for (const t of ticks) {
            if (Math.abs(t.pos - mid) <= LABEL_ON_TICK_SPACINGS * spacing && (best === null || Math.abs(t.pos - mid) < Math.abs(best.pos - mid)))
                best = t;
        }
        if (best)
            assigned.push({ b, tick: best, gap: gapOf(b), path: pathInk(b) });
    }
    if (!assigned.length)
        return labels;
    const minGap = Math.min(...assigned.map(a => a.gap));
    const heights = assigned.map(a => a.b[3] - a.b[1]).sort((p, q) => p - q);
    const tol = Math.max(LABEL_COLUMN_MIN_TOL, 0.5 * heights[Math.floor(heights.length / 2)]);
    for (const a of assigned) {
        if (a.gap <= minGap + tol) {
            const was = labels.get(a.tick);
            labels.set(a.tick, { box: union(was?.box, a.b), path: Math.max(was?.path ?? 0, a.path) });
        }
    }
    return labels;
}

/**
 * Labels of gridlines: a word just above or below the line, near either end of it or beyond (right- or
 * left-aligned axis labels, as on the economist chart). Labels of one axis sit at the same end, so each line takes
 * the nearest word at the end most lines' nearest words are at (economist: a dot rim at the left end of the zero
 * line was nearer than its "0" at the right).
 */
function gridLabels(image, backgrounds, ticks, gridlines, dir) {
    const h = dir === 'h';
    const spacing = ticks.length > 1 ? (ticks[ticks.length - 1].pos - ticks[0].pos) / (ticks.length - 1) : 0;
    const options = ticks.map(t => {
        const line = gridlines.find(l => Math.abs(l.pos - t.pos) < 0.5) ?? null;
        const from = line ? line.from : Math.min(...gridlines.map(l => l.from));
        const to = line ? line.to : Math.max(...gridlines.map(l => l.to));
        const ext = GRID_LABEL_END_FRACTION * (to - from);
        const across = [Math.max(0, t.pos - spacing / 3), Math.min(h ? image.height : image.width, t.pos + spacing / 3)];
        const along = [Math.max(0, from - ext), Math.min(h ? image.width : image.height, to + ext)];
        const rect = h ? [along[0], across[0], along[1], across[1]] : [across[0], along[0], across[1], along[1]];
        // gray ink only: the nearest colored ink to a gridline is usually data (economist: the red trend line)
        const words = [];
        for (const b of mergeWords(inkComponents(image, backgrounds, rect, true).map(c => c.box))) {
            const lo = h ? b[0] : b[1];
            const hi = h ? b[2] : b[3];
            const end = hi <= from + ext ? 'start' : (lo >= to - ext ? 'end' : null);
            const clear = h ? (b[3] <= t.pos - 1 || b[1] >= t.pos + 1) : (b[2] <= t.pos - 1 || b[0] >= t.pos + 1);
            if (end && clear) {
                const d = h ? Math.min(Math.abs(b[3] - t.pos), Math.abs(b[1] - t.pos)) : Math.min(Math.abs(b[2] - t.pos), Math.abs(b[0] - t.pos));
                words.push({ b, end, d });
            }
        }
        return words.sort((p, q) => p.d - q.d);
    });
    const votes = { start: 0, end: 0 };
    for (const words of options) {
        if (words.length)
            votes[words[0].end]++;
    }
    const side = votes.end >= votes.start ? 'end' : 'start';
    ticks.forEach((t, i) => {
        t.label = options[i].find(w => w.end === side)?.b ?? null;
    });
}

/**
 * Detect the axes: the x axis is the dark horizontal line with the most ticks (ties: the lowest), the y axis the
 * dark vertical line with the most ticks (ties: the leftmost). Without a dark line with ticks on an axis, light
 * gridlines across the other direction stand in as its ticks (economist: no y axis line, horizontal gridlines),
 * with a dark baseline that continues their spacing (the economist's zero line).
 *
 * @param {import('./image.js').RgbaImage} image
 * @returns {{lines: Line[], backgrounds: number[][], x: object|null, y: object|null}}
 *   each axis: { line, ticks, spacing, regular }
 */
export function detectAxes(image) {
    const backgrounds = backgroundColors(image);
    const ink = grayInk(image, backgrounds);
    const lines = findLines(image, ink);
    // labeled ticks drawn as strokes (a corner tick, on a line, has none) are straight, and their labels are in
    // plain view of the axis
    const axisLike = ticks => {
        const strokes = ticks.filter(t => t.length > 0);
        return (!strokes.length || median(strokes.map(t => t.shortest / t.length)) >= TICK_MIN_STRAIGHTNESS)
            && median(ticks.map(t => t.labelPath)) <= LABEL_PATH_MAX_INK;
    };
    const pick = dir => {
        const candidates = lines.filter(l => l.dir === dir && l.dark).map(line => ({ line, ticks: findTicks(image, ink, line, lines) }));
        candidates.sort((a, b) => (b.ticks.length - a.ticks.length) || (dir === 'h' ? b.line.pos - a.line.pos : a.line.pos - b.line.pos));
        let best = null;
        for (const c of candidates) {
            findLabels(image, backgrounds, c.line, c.ticks);
            const labeled = c.ticks.filter(t => t.label);
            if (labeled.length >= AXIS_MIN_LABELED_TICKS && axisLike(labeled)) {
                best = { ...c, ticks: c.ticks.filter(t => !t.onLine || t.label) };
                break;
            }
        }
        if (!best) {
            // gridlines perpendicular to this axis give its tick positions
            const other = dir === 'h' ? 'v' : 'h';
            const grid = lines.filter(l => l.dir === other && !l.dark).sort((a, b) => a.pos - b.pos);
            if (grid.length >= 2) {
                const ticks = grid.map(l => ({ pos: l.pos, length: 0, side: 'after', label: null, source: 'grid' }));
                const step = (ticks[ticks.length - 1].pos - ticks[0].pos) / (ticks.length - 1);
                for (const base of lines.filter(l => l.dir === other && l.dark)) {
                    const k = (base.pos - ticks[0].pos) / step;
                    if (Math.abs(k - Math.round(k)) * step <= GRID_BASELINE_TOL * step && !ticks.some(t => Math.abs(t.pos - base.pos) < 1))
                        ticks.push({ pos: base.pos, length: 0, side: 'after', label: null, source: 'grid' });
                }
                ticks.sort((a, b) => a.pos - b.pos);
                gridLabels(image, backgrounds, ticks, lines.filter(l => l.dir === other), other);
                best = { line: null, ticks };
            }
        }
        return best;
    };
    // An unlabeled "tick" beyond the other axis, away from the plot, is that axis's label crossing this line (JMP:
    // the y axis's bottom label "30" straddles the x axis line left of the corner and its strokes made three ticks).
    const beyondOther = (axis, other) => {
        if (!axis || !other?.line)
            return axis;
        const at = other.line.pos;
        const plotAfter = axis.ticks.filter(t => t.pos > at).length >= axis.ticks.filter(t => t.pos < at).length;
        const outside = t => (plotAfter ? t.pos < at - TICK_LINE_CLEARANCE : t.pos > at + TICK_LINE_CLEARANCE);
        return { ...axis, ticks: axis.ticks.filter(t => t.label || !outside(t)) };
    };
    const finish = axis => {
        if (!axis)
            return null;
        const gaps = axis.ticks.slice(1).map((t, i) => t.pos - axis.ticks[i].pos);
        const spacing = gaps.length ? gaps.reduce((sum, g) => sum + g, 0) / gaps.length : NaN;
        const regular = gaps.length > 0 && gaps.every(g => Math.abs(g - spacing) <= Math.max(1, 0.03 * spacing));
        return { ...axis, spacing, regular };
    };
    const x = pick('h');
    const y = pick('v');
    return { lines, backgrounds, x: finish(beyondOther(x, y)), y: finish(beyondOther(y, x)) };
}

/**
 * The plot area's sides that the axes show: an axis line with tick marks whose labels all lie on one side of it
 * bounds the data on the other side (x labels below the line: the plot is above it). Gridline axes and axes with
 * labels on both sides bound nothing (the economist's zero line has data on both sides; a baseline through the
 * middle of a beeswarm would too).
 *
 * @param {{x: object|null, y: object|null}} axes  from detectAxes
 * @returns {{left: number|null, right: number|null, top: number|null, bottom: number|null}}  px, the axis lines'
 *   centers; null where unbounded
 */
export function plotBounds(axes) {
    const bounds = { left: null, right: null, top: null, bottom: null };
    const side = axis => {
        if (!axis?.line)
            return 0;
        const labeled = axis.ticks.filter(t => t.source === 'tick' && t.label);
        if (labeled.length < AXIS_MIN_LABELED_TICKS)
            return 0;
        const across = axis.line.dir === 'h' ? b => (b[1] + b[3] + 1) / 2 : b => (b[0] + b[2] + 1) / 2;
        const signs = labeled.map(t => Math.sign(across(t.label) - axis.line.pos));
        return signs.every(v => v === signs[0]) ? signs[0] : 0;
    };
    const sx = side(axes.x);
    if (sx > 0)
        bounds.bottom = axes.x.line.pos;
    else if (sx < 0)
        bounds.top = axes.x.line.pos;
    const sy = side(axes.y);
    if (sy < 0)
        bounds.left = axes.y.line.pos;
    else if (sy > 0)
        bounds.right = axes.y.line.pos;
    return bounds;
}

/**
 * Lines where a plot may clip its dots: each bounding axis line (plotBounds' sides) and the far end of each such line
 * (autocor1: dots cut straight where the x axis line ends, at the plot's right, with no line there). Whether dots are
 * cut there is the mask's to read (layers.js buildMask), from a family's coverage.
 *
 * @param {{x: object|null, y: object|null}} axes  from detectAxes
 * @returns {{dir: 'h'|'v', pos: number, out: number, thickness: number, from: number|null, to: number|null}[]}
 *   dir: the cut's direction (a vertical cut runs down a column); pos px, its position; out: +1 when beyond is
 *   right or down, −1 when left or up; thickness: the line's there, px (0 at an end); from, to: its extent along,
 *   px (the other axis line's, when known)
 */
export function clipEdges(axes) {
    const bounds = plotBounds(axes);
    const edges = [];
    const x = axes.x?.line;
    const y = axes.y?.line;
    if (x && (bounds.bottom !== null || bounds.top !== null)) {
        const out = bounds.bottom !== null ? 1 : -1;
        edges.push({ dir: 'h', pos: x.pos, out, thickness: x.thickness, from: x.from, to: x.to });
        // the far end: away from the y axis
        if (y) {
            const right = Math.abs(x.to - y.pos) > Math.abs(x.from - y.pos);
            edges.push({ dir: 'v', pos: right ? x.to : x.from, out: right ? 1 : -1, thickness: 0, from: y.from, to: y.to });
        }
    }
    if (y && (bounds.left !== null || bounds.right !== null)) {
        const out = bounds.left !== null ? -1 : 1;
        edges.push({ dir: 'v', pos: y.pos, out, thickness: y.thickness, from: y.from, to: y.to });
        if (x) {
            const down = Math.abs(y.to - x.pos) > Math.abs(y.from - x.pos);
            edges.push({ dir: 'h', pos: down ? y.to : y.from, out: down ? 1 : -1, thickness: 0, from: x.from, to: x.to });
        }
    }
    return edges;
}

/**
 * Fit a scale to ticks with known values.
 *
 * Linear: value = a + b·px. Log: log10(value) = a + b·px (values must be positive). Least squares; with more than two
 * values the residuals, in px, say how well the ticks agree with the typed values.
 *
 * @param {{pos: number, value: number}[]} known  px and data value
 * @param {'linear'|'log'} type
 * @returns {{type: string, a: number, b: number, residualPx: number[], maxResidualPx: number}|null}
 *   null when fewer than two usable values or all at one position
 */
export function fitScale(known, type) {
    const pts = known.filter(k => Number.isFinite(k.value) && (type !== 'log' || k.value > 0));
    if (pts.length < 2)
        return null;
    const f = v => (type === 'log' ? Math.log10(v) : v);
    const n = pts.length;
    const mx = pts.reduce((s, k) => s + k.pos, 0) / n;
    const my = pts.reduce((s, k) => s + f(k.value), 0) / n;
    let sxx = 0;
    let sxy = 0;
    for (const k of pts) {
        sxx += (k.pos - mx) ** 2;
        sxy += (k.pos - mx) * (f(k.value) - my);
    }
    if (sxx === 0)
        return null;
    const b = sxy / sxx;
    const a = my - b * mx;
    // residuals as pixel offsets: where each typed value falls on the fitted scale, minus where its tick is
    const residualPx = pts.map(k => (f(k.value) - a) / b - k.pos);
    return { type, a, b, residualPx, maxResidualPx: Math.max(...residualPx.map(Math.abs)) };
}

/**
 * Data value at a pixel position (continuous; pixel i spans [i, i + 1)).
 *
 * @param {{type: string, a: number, b: number}} scale
 * @param {number} px
 * @returns {number}
 */
export function toData(scale, px) {
    const v = scale.a + scale.b * px;
    return scale.type === 'log' ? 10 ** v : v;
}

/**
 * Pixel position of a data value.
 */
export function toPixel(scale, value) {
    const v = scale.type === 'log' ? Math.log10(value) : value;
    return (v - scale.a) / scale.b;
}

/**
 * How much one pixel is in data units at a position: |dv/dpx| (constant for linear scales).
 */
export function resolution(scale, px) {
    return scale.type === 'log' ? Math.abs(toData(scale, px) * Math.LN10 * scale.b) : Math.abs(scale.b);
}

/**
 * Guess values for ticks without one from those with: on the fitted scale, rounded to the precision the user typed
 * (decimals for linear, three significant figures for log).
 *
 * @param {{pos: number, value: number|null, text?: string}[]} ticks
 * @param {'linear'|'log'} type
 * @returns {(number|null)[]}  a guess per tick (null where a value is known or no scale can be fitted)
 */
export function guessValues(ticks, type) {
    const known = ticks.filter(t => t.value !== null && Number.isFinite(t.value));
    const scale = fitScale(known, type);
    if (!scale)
        return ticks.map(() => null);
    const decimals = Math.max(0, ...known.map(t => {
        const s = t.text ?? String(t.value);
        const dot = s.indexOf('.');
        return dot < 0 ? 0 : s.length - dot - 1;
    }));
    return ticks.map(t => {
        if (t.value !== null && Number.isFinite(t.value))
            return null;
        const v = toData(scale, t.pos);
        return type === 'log' ? Number(v.toPrecision(3)) : Number(v.toFixed(decimals));
    });
}

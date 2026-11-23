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
// A gridline lies under the dots, and enough of them leave no run of LINE_MIN_FRACTION (song-cents: seven gridlines,
// the longest runs of two of them 0.15 and 0.25 of the width under 325 dots). Its light ink is still most of the
// row's: pieces of BROKEN_MIN_SEGMENT of the image or more that add up to LINE_MIN_FRACTION of it and fill
// BROKEN_MIN_COVER of the span from the first to the last (those rows: 0.67–0.92 of the width in light ink, and no
// other row of the image 0.2). Text is strokes a few px across; an axis line is dark, and found by its run.
const BROKEN_MIN_SEGMENT = 0.02; // of the image dimension along the line
const BROKEN_MIN_COVER = 0.5;
// … and a piece is flat: the light ink through BROKEN_MIN_FLAT of its pixels, across the line, starts and ends at
// one distance from them (to a pixel), no farther than a line is thick. A row through a swarm of light gray dots is
// pieces too, chords of the dots, whose ink across is as much as the dot is tall there (nicest language: three such
// "lines" through its gray dots, and a y axis of them); so is a row of a gray band with names on it, as thick as
// the band (beeswarm-boxplot: three more "lines" in its band). The pieces of one line have the same edges.
const BROKEN_MIN_FLAT = 0.8;
// The line runs from its first flat piece to its last, shorter ones too if they are this long and as thick as the
// long ones, edge for edge: a dot near a line's end leaves a stub beyond it (song-cents: 21 px of the 10 line before
// its first dot, which read as the line's label). The edge of a label's stroke is flat too, and no thicker than
// itself (grader1: a column of a "3" below its gridline's end).
const BROKEN_MIN_END = 8; // px
const DARK_EDGE_REACH = 2; // px: rows this near a dark line's run are its antialiased edge, not a light line
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
// An exponent is of its number's word, raised and small as it is (a minus sign's height is a tenth of the digits'):
// a shorter box right after a taller one, starting and ending in its upper part. Without it primates1's "10" was
// boxed without its raised 2, and the box's center was nearer the minor tick before its own. All × the taller's
// height:
const EXPONENT_MAX_GAP = 0.3; // after the number (measured 0.07–0.18; the next label's minus sign is farther)
const EXPONENT_MAX_HEIGHT = 0.8;
const EXPONENT_TOP = [-0.6, 0.25]; // where it starts, from the number's top, down (a minus sign on the line: 0.5)
// where it ends, at most; and below the number's top (pathogen: a y label beside a part of the rotated title and
// wholly above it joined the title)
const EXPONENT_MAX_BOTTOM = 0.6;
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
// A category axis often has no tick marks: its labels alone mark the positions (labelAxis). It is the line that meets
// an end of the other axis's line within AXIS_CORNER_TOL px (beyond the two lines' half thicknesses), with at least
// CATEGORY_MIN_LABELS words in a row beside it: one word under a line is an axis title.
const AXIS_CORNER_TOL = 3; // px
const END_LINE_MIN_OVERLAP = 0.8; // of a line's length: another at the plot's end runs beside it this far
const CATEGORY_MIN_LABELS = 2;
// A frame is one color, and its sides' means can fall either side of DARK_LUMINANCE (biomarker's gray frame: 175 at the
// left, the y axis, and 181 at the bottom): a line this near the other axis's luminance is as dark as it.
const FRAME_LUMINANCE_TOL = 16; // 0–255
const BACKGROUND_MIN_FRACTION = 0.05; // any flat color covering this share of the image is a background (panels)
const PANEL_MAX_CONTRAST = 128; // 0–255, luminance: a background of the axes differs from the page by less (detectAxes)
// A panel is a rectangle filled with a background color that is not the page's (ggplot2's default theme: gray, with
// white gridlines and no axis lines; the tick marks leave its edges). Its first and last rows and columns are of
// its color at least this share of its width and height: gridlines cross them, a few percent in all, while the
// first row of a band, an ellipse or a crowd of dots of one color is a sliver of it (panelEdges).
const PANEL_EDGE_MIN_FILL = 0.8;
// A tick at a panel's edge may begin a few pixels out from it: ggplot2 at 300 dpi, 7 in wide, leaves 2 px of page
// between the panel and its y ticks, and the edge's own pixel there is white too, a gridline's end. As a share of
// the image dimension across the edge, and at least TICK_MIN_LENGTH px; labels begin a tick's length farther out.
const EDGE_TICK_GAP_FRACTION = 0.003;
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
 * @property {boolean} [edge]    a panel's edge standing in for an axis line (panelEdges), or the line through the ends
 *   of an axis's gridlines (gridEndLine): no ink of its own, thickness 0, `pos` the boundary between the panel's last
 *   row or column and the page's first, or where the gridlines end
 */

/**
 * @typedef {object} Tick
 * @property {number} pos     px, center along the axis
 * @property {number} length  px, the longest stroke across the tick's width
 * @property {number} [shortest]  px, the shortest (a straight tick's is about its length)
 * @property {'before'|'after'} side  'before' = above (x axis) or left (y axis)
 * @property {number[]|null} label  [x0, y0, x1, y1) box of the label's ink, or null
 * @property {number} [labelPath]  0–1, share of ink between the axis and the label
 * @property {'tick'|'grid'|'label'} source  a tick mark; a gridline standing in for one; or, on an axis without tick
 *   marks, a label, whose center is the position
 */

/**
 * @typedef {object} Axis
 * @property {Line} line  the axis line (for an axis read from gridlines, a line standing in for it)
 * @property {Tick[]} ticks
 * @property {number|null} spacing  px between ticks, when they are evenly spaced
 * @property {boolean} regular  whether they are
 */

/**
 * @typedef {object} Axes  detectAxes' result
 * @property {Line[]} lines  every long straight gray line found
 * @property {number[][]} backgrounds  0–255, the page's color first
 * @property {Axis|null} x
 * @property {Axis|null} y
 */

/**
 * A line where a plot may clip its dots (clipEdges).
 *
 * @typedef {object} ClipEdge
 * @property {'h'|'v'} dir  the cut's direction: a vertical cut runs down a column
 * @property {number} pos  px, its position across
 * @property {number} out  +1 when beyond is right or down, −1 when left or up
 * @property {number} thickness  px, the axis line's there; 0 at a line's end
 * @property {number|null} from  px, where it starts along (the other axis line's extent, when known)
 * @property {number|null} to  px
 * @property {number} [cut]  px, once dots are found cut there (layers.js plotClips): the first pixel row or column
 *   beyond
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

/**
 * The edges of filled panels, as lines: where a plot is drawn on a panel of its own color with no axis lines, its
 * axes are the panel's edges, and the tick marks leave them into the page (ggplot2's default theme). A panel is the
 * extent of a background color other than the page's (the one at the image's border), kept when it is a rectangle:
 * its first and last rows and columns are of that color PANEL_EDGE_MIN_FILL of the way across.
 *
 * @param {import('./image.js').RgbaImage} image
 * @param {number[][]} backgrounds  from backgroundColors
 * @returns {Line[]}  four per panel, thickness 0, `edge` true
 */
export function panelEdges(image, backgrounds) {
    const { width, height, data } = image;
    const like = (p, B) => Math.hypot(data[4 * p] - B[0], data[4 * p + 1] - B[1], data[4 * p + 2] - B[2]) <= INK_TOL;
    const border = [];
    for (let x = 0; x < width; x++)
        border.push(x, (height - 1) * width + x);
    for (let y = 0; y < height; y++)
        border.push(y * width, y * width + width - 1);
    const onBorder = backgrounds.map(B => border.filter(p => like(p, B)).length);
    const page = backgrounds[onBorder.indexOf(Math.max(...onBorder))];
    const edges = [];
    for (const B of backgrounds.filter(b => b !== page)) {
        const rows = new Int32Array(height);
        const cols = new Int32Array(width);
        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
                if (like(y * width + x, B)) {
                    rows[y]++;
                    cols[x]++;
                }
            }
        }
        // [first, last + 1) of the rows or columns that the color runs along, as a line must
        const extent = (counts, least) => {
            const first = counts.findIndex(n => n >= least);
            return first < 0 ? null : [first, counts.findLastIndex(n => n >= least) + 1];
        };
        const ys = extent(rows, LINE_MIN_FRACTION * width);
        const xs = extent(cols, LINE_MIN_FRACTION * height);
        if (!ys || !xs)
            continue;
        const full = (counts, [a, b], length) => counts[a] >= PANEL_EDGE_MIN_FILL * length && counts[b - 1] >= PANEL_EDGE_MIN_FILL * length;
        if (!full(rows, ys, xs[1] - xs[0]) || !full(cols, xs, ys[1] - ys[0]))
            continue;
        const L = 0.299 * B[0] + 0.587 * B[1] + 0.114 * B[2];
        for (const pos of ys)
            edges.push({ dir: 'h', pos, thickness: 0, from: xs[0], to: xs[1], luminance: L, dark: false, edge: true });
        for (const pos of xs)
            edges.push({ dir: 'v', pos, thickness: 0, from: ys[0], to: ys[1], luminance: L, dark: false, edge: true });
    }
    return edges;
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

// The span [start, end) of a light line in pieces along one row or column (BROKEN_MIN_SEGMENT, BROKEN_MIN_COVER), or
// null. A piece is a run of light ink with gaps up to LINE_GAP of anything, flat across (BROKEN_MIN_FLAT). `beside(j,
// d)` is the index of the pixel d rows or columns to the side of position j, or -1 beyond the image; `maxThick` the
// thickest a line may be, px.
function brokenLine(ink, n, at, beside, maxThick) {
    // how far the light ink runs to one side of position j, up to one more than a line's thickness
    const reach = (j, step) => {
        let d = 0;
        while (d <= maxThick) {
            const p = beside(j, step * (d + 1));
            if (p < 0 || ink[p] !== 1)
                break;
            d++;
        }
        return d;
    };
    const flat = (a, b) => {
        const sides = [];
        for (let j = a; j < b; j++) {
            if (ink[at(j)] === 1)
                sides.push([reach(j, -1), reach(j, 1)]);
        }
        // the commonest pair is the line's two edges
        const counts = new Map();
        for (const [u, d] of sides)
            counts.set(u * 1000 + d, (counts.get(u * 1000 + d) ?? 0) + 1);
        const mode = [...counts].sort((p, q) => q[1] - p[1])[0][0];
        const [mu, md] = [Math.floor(mode / 1000), mode % 1000];
        if (mu + md + 1 > maxThick)
            return null;
        return sides.filter(([u, d]) => Math.abs(u - mu) <= 1 && Math.abs(d - md) <= 1).length >= BROKEN_MIN_FLAT * sides.length ? mode : null;
    };
    // (cheap first: the light ink there is at all)
    let light = 0;
    for (let j = 0; j < n; j++)
        light += ink[at(j)] === 1 ? 1 : 0;
    if (light < LINE_MIN_FRACTION * n)
        return null;
    const least = BROKEN_MIN_SEGMENT * n;
    // the flat pieces: where each starts and ends, and its edges
    const flats = [];
    let start = -1;
    let last = -1;
    for (let j = 0; j <= n; j++) {
        if (j < n && ink[at(j)] === 1) {
            if (start < 0)
                start = j;
            last = j;
        }
        else if (start >= 0 && (j === n || j - last > LINE_GAP)) {
            const sides = last + 1 - start >= Math.min(least, BROKEN_MIN_END) ? flat(start, last + 1) : null;
            if (sides !== null)
                flats.push({ start, end: last + 1, sides });
            start = -1;
        }
    }
    // the line's edges are those most of the long pieces' length has; the pieces with other edges are other ink
    const lengths = new Map();
    for (const f of flats.filter(f => f.end - f.start >= least))
        lengths.set(f.sides, (lengths.get(f.sides) ?? 0) + f.end - f.start);
    if (!lengths.size)
        return null;
    const edges = [...lengths].sort((p, q) => q[1] - p[1])[0][0];
    const own = flats.filter(f => f.sides === edges);
    const long = own.filter(f => f.end - f.start >= least);
    const total = long.reduce((sum, f) => sum + f.end - f.start, 0);
    if (total < LINE_MIN_FRACTION * n || total < BROKEN_MIN_COVER * (long[long.length - 1].end - long[0].start))
        return null;
    return [own[0].start, own[own.length - 1].end];
}

/**
 * Long thin straight gray lines, horizontal and vertical. A line is a long run of gray ink in its row or column, or,
 * for a light one under other marks, pieces that fill most of their span (brokenLine).
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
        // a light line in pieces, where no dark line's run is within its antialiased edge of the row (a dark line's
        // edge is light ink too, and its run is the line's)
        const darkRun = runs.map((run, i) => {
            if (!run)
                return false;
            let dark = 0;
            for (let j = run[0]; j < run[1]; j++)
                dark += ink[idx(i, j)] === 2 ? 1 : 0;
            return dark >= (run[1] - run[0]) / 2;
        });
        const pieces = runs.map((run, i) => {
            for (let k = Math.max(0, i - DARK_EDGE_REACH); k <= Math.min(across - 1, i + DARK_EDGE_REACH); k++) {
                if (darkRun[k])
                    return null;
            }
            return brokenLine(ink, along, j => idx(i, j), (j, d) => (i + d < 0 || i + d >= across ? -1 : idx(i + d, j)), maxThick);
        });
        for (let i = 0; i < across; i++) {
            const span = pieces[i];
            if (span && (!runs[i] || span[1] - span[0] > runs[i][1] - runs[i][0]))
                runs[i] = span;
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
 * grouped into ticks. Along a panel's edge, which has no thickness to measure a tick's width by, a tick is no
 * wider than it is long, and may begin a few pixels out (EDGE_TICK_GAP_FRACTION). A tick that coincides with a
 * perpendicular line is marked `onLine`: it may be an axis corner (dropped later unless it has a label) or a zero
 * line crossing the axis (Laser: it is the 0 tick).
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
    const maxGap = axis.edge ? Math.max(TICK_MIN_LENGTH, Math.round(EDGE_TICK_GAP_FRACTION * acrossMax)) : 0;
    const near = (axis.pos - axis.thickness / 2);
    const far = (axis.pos + axis.thickness / 2);
    const perpendicular = lines.filter(l => l.dir !== axis.dir && l.dark);
    const ticks = [];
    for (const side of ['before', 'after']) {
        const start = side === 'before' ? Math.floor(near) - 1 : Math.ceil(far);
        const step = side === 'before' ? -1 : 1;
        // length of the dark stroke leaving the axis at each position along it, and where it begins
        const len = new Float64Array(alongMax);
        const begin = new Int32Array(alongMax);
        // (… looked for a little beyond the axis's ends: 2 px, and at a panel's edge as far again as its ticks may
        // stand off it and are wide. The tick at its corner is then seen whole, and from the other edge whole too,
        // wider than long: with 2 px it was that edge's last tick, 2 px across)
        const beyond = 2 + 2 * maxGap;
        for (let a = Math.max(0, Math.floor(axis.from) - beyond); a < Math.min(alongMax, Math.ceil(axis.to) + beyond); a++) {
            let c = start;
            for (let gap = 0; gap < maxGap && c >= 0 && c < acrossMax && !ink[at(a, c)]; gap++)
                c += step;
            begin[a] = c;
            let n = 0;
            while (c >= 0 && c < acrossMax && ink[at(a, c)] > 0 && n <= maxLen) {
                n++;
                c += step;
            }
            // (… from the axis: with the pixels between, at a panel's edge)
            len[a] = n >= TICK_MIN_LENGTH && n <= maxLen ? n + Math.abs(begin[a] - start) : 0;
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
            if (b - a <= (axis.edge ? Math.max(...len.slice(a, b)) : maxWidth)) {
                let w = 0;
                let wpos = 0;
                let core = 255;
                for (let k = a; k < b; k++) {
                    // weight by darkness of the stroke's first pixel, to center antialiased edges (at a panel's
                    // edge its second: the first is the edge's own, blended with the page's and the stroke's)
                    const L = luminance(data, 4 * at(k, axis.edge ? begin[k] + step : start));
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

// glyph boxes merged into words: boxes that overlap by most of their height, or of which one is the other's
// exponent, and are closer than WORD_GAP_HEIGHTS × their height along the reading direction
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
                const [tall, small] = a[3] - a[1] >= b[3] - b[1] ? [a, b] : [b, a];
                const exponent = small[0] > tall[0] && small[2] > tall[2] && gapX <= EXPONENT_MAX_GAP * height
                    && shorter <= EXPONENT_MAX_HEIGHT * height && small[3] > tall[1] && small[3] <= tall[1] + EXPONENT_MAX_BOTTOM * height
                    && small[1] >= tall[1] + EXPONENT_TOP[0] * height && small[1] <= tall[1] + EXPONENT_TOP[1] * height;
                const sameLine = overlapY >= WORD_MIN_OVERLAP * shorter && height <= WORD_MAX_HEIGHT_RATIO * shorter;
                if ((sameLine || exponent) && gapX <= WORD_GAP_HEIGHTS * height) {
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
    // (a panel's edge has its labels where its ticks are, out on the page: inside the panel are the gridlines, whose
    // rims, between the panel's color and their own, made a "label" beside every tick)
    const other = axis.edge ? new Map() : labelsOnSide(image, backgrounds, axis, ticks, tickSide === 'after' ? 'before' : 'after', 0, OTHER_SIDE_MAX_WIDTH);
    const far = other.size >= OTHER_SIDE_MIN_LABELS && other.size > own.size;
    for (const [t, { box, path }] of far ? other : own) {
        t.label = box;
        t.labelPath = path;
    }
}

// how far a word's box is from an axis's edge, px, on the side it lies
function wordGap(b, h, side, edge) {
    return h ? (side === 'after' ? b[1] - edge : edge - b[3]) : (side === 'after' ? b[0] - edge : edge - b[2]);
}

// share of ink between an axis's edge and a word, across the word's extent along the axis
function inkBetween(image, backgrounds, b, h, side, edge) {
    const [a0, a1] = h ? [b[0], b[2]] : [b[1], b[3]];
    const [n0, n1] = side === 'after' ? [Math.ceil(edge), h ? b[1] : b[0]] : [h ? b[3] : b[2], Math.floor(edge)];
    let all = 0;
    let ink = 0;
    for (let a = a0; a < a1; a++) {
        for (let c = n0; c < n1; c++) {
            all++;
            if (isInk(image.data, 4 * (h ? c * image.width + a : a * image.width + c), backgrounds))
                ink++;
        }
    }
    return all ? ink / all : 0;
}

/**
 * An axis whose labels alone mark its positions, with no tick marks: a category axis (bench3: "1" and "2" under the
 * two swarms; cells1: the bars' names under their baseline). It is a dark line (or one as dark as the other axis's:
 * a frame's side) meeting an end of the other axis's line, the plot's corner, and the words in the row nearest it on the side away from the plot, between the other
 * axis and the line's far end; each word's center is a position. Words farther out are the axis's title. Null
 * without such a line or with fewer than CATEGORY_MIN_LABELS words.
 *
 * Not for a line elsewhere in the plot (a bracket, a box's edge), which has dots and text beside it, and not for the
 * other axis's own labels, which lie beyond its line (JMP's bottom y label straddles the x axis line there).
 *
 * @param {import('./image.js').RgbaImage} image
 * @param {number[][]} backgrounds
 * @param {Line[]} lines
 * @param {'h'|'v'} dir  the direction of the axis looked for
 * @param {Line} other  the other axis's line
 * @returns {{line: Line, ticks: Tick[]}|null}
 */
function labelAxis(image, backgrounds, lines, dir, other) {
    const h = dir === 'h';
    // (a line standing in for the other axis, where its gridlines end, is as light as they are: no frame's side)
    const asDark = l => l.dark || (!other.edge && l.luminance <= other.luminance + FRAME_LUMINANCE_TOL);
    for (const line of lines.filter(l => l.dir === dir && asDark(l))) {
        const corner = Math.abs(other.from - line.pos) <= Math.abs(other.to - line.pos) ? other.from : other.to;
        const tol = AXIS_CORNER_TOL + (line.thickness + other.thickness) / 2;
        if (Math.abs(corner - line.pos) > tol || other.pos < line.from - tol || other.pos > line.to + tol)
            continue;
        // the plot lies where the other axis runs to, the labels on the line's other side
        const far = corner === other.from ? other.to : other.from;
        const side = far > line.pos ? 'before' : 'after';
        const reach = LABEL_SEARCH_FRACTION[dir] * (h ? image.height : image.width);
        const edge = side === 'after' ? line.pos + line.thickness / 2 + 1 : line.pos - line.thickness / 2 - 1;
        const c0 = Math.max(0, Math.floor(side === 'after' ? edge : edge - reach));
        const c1 = Math.min(h ? image.height : image.width, Math.ceil(side === 'after' ? edge + reach : edge));
        const toEnd = Math.abs(line.to - other.pos) >= Math.abs(line.from - other.pos);
        const a0 = toEnd ? other.pos + other.thickness / 2 : line.from;
        const a1 = toEnd ? line.to : other.pos - other.thickness / 2;
        if (c1 <= c0 || a1 <= a0)
            continue;
        const words = mergeWords(inkComponents(image, backgrounds, h ? [a0, c0, a1, c1] : [c0, a0, c1, a1]).map(c => c.box));
        if (words.length < CATEGORY_MIN_LABELS)
            continue;
        // the labels are the row at the smallest gap, as an axis's tick labels are (labelsOnSide)
        const minGap = Math.min(...words.map(b => wordGap(b, h, side, edge)));
        const tolGap = Math.max(LABEL_COLUMN_MIN_TOL, 0.5 * median(words.map(b => b[3] - b[1])));
        const row = words.filter(b => wordGap(b, h, side, edge) <= minGap + tolGap);
        if (row.length < CATEGORY_MIN_LABELS)
            continue;
        const ticks = row.map(b => ({
            pos: h ? (b[0] + b[2]) / 2 : (b[1] + b[3]) / 2, length: 0, side, label: b,
            labelPath: inkBetween(image, backgrounds, b, h, side, edge), source: 'label',
        })).sort((p, q) => p.pos - q.pos);
        return { line, ticks };
    }
    return null;
}

/**
 * An axis of labels alone under the last gridline of the other axis: where the axes are gridlines with their labels
 * and no lines (song-cents: years in a row under the zero line, no tick marks and no vertical gridlines), the row of
 * words nearest below that gridline, each centered within its extent, gives the positions, as labelAxis takes them
 * at an axis line. Null with fewer than CATEGORY_MIN_LABELS words.
 *
 * @param {import('./image.js').RgbaImage} image
 * @param {number[][]} backgrounds
 * @param {Line} line  the gridline (horizontal)
 * @returns {{line: Line, ticks: Tick[]}|null}
 */
function labelsUnder(image, backgrounds, line) {
    const reach = LABEL_SEARCH_FRACTION.h * image.height;
    const edge = line.pos + line.thickness / 2 + 1;
    const c0 = Math.floor(edge);
    const c1 = Math.min(image.height, Math.ceil(edge + reach));
    if (c1 <= c0)
        return null;
    const middle = b => (b[0] + b[2]) / 2;
    const words = mergeWords(inkComponents(image, backgrounds, [0, c0, image.width, c1]).map(c => c.box))
        .filter(b => middle(b) >= line.from - LABEL_BEYOND_TOL && middle(b) <= line.to + LABEL_BEYOND_TOL);
    if (words.length < CATEGORY_MIN_LABELS)
        return null;
    const minGap = Math.min(...words.map(b => wordGap(b, true, 'after', edge)));
    const tolGap = Math.max(LABEL_COLUMN_MIN_TOL, 0.5 * median(words.map(b => b[3] - b[1])));
    const row = words.filter(b => wordGap(b, true, 'after', edge) <= minGap + tolGap);
    if (row.length < CATEGORY_MIN_LABELS)
        return null;
    const ticks = row.map(b => ({
        pos: middle(b), length: 0, side: 'after', label: b, labelPath: inkBetween(image, backgrounds, b, true, 'after', edge), source: 'label',
    })).sort((p, q) => p.pos - q.pos);
    return { line, ticks };
}

/**
 * The line through the ends of an axis's gridlines, where every label lies off that end of its gridline: the
 * gridlines stop there, and so does the plot (ggplot2's theme_minimal, song-cents). It stands in for the axis line,
 * as a panel's edge does. Null when the labels lie along the lines (economist: above them, at their right ends), or
 * fewer than AXIS_MIN_LABELED_TICKS have one.
 *
 * @param {'h'|'v'} dir  the axis's direction: its gridlines run the other way
 * @param {Tick[]} ticks  the gridlines' positions, with their labels
 * @param {Line[]} gridlines
 * @returns {Line|null}  thickness 0, `edge` true
 */
function gridEndLine(dir, ticks, gridlines) {
    const ends = ticks.filter(t => t.label).map(t => {
        const line = gridlines.find(l => Math.abs(l.pos - t.pos) < 0.5);
        const [lo, hi] = dir === 'v' ? [t.label[0], t.label[2]] : [t.label[1], t.label[3]];
        return { line, start: hi <= line.from + LABEL_BEYOND_TOL, end: lo >= line.to - LABEL_BEYOND_TOL };
    });
    if (ends.length < AXIS_MIN_LABELED_TICKS || !(ends.every(e => e.start) || ends.every(e => e.end)))
        return null;
    const pos = ends[0].start ? median(ends.map(e => e.line.from)) : median(ends.map(e => e.line.to));
    return { dir, pos, thickness: 0, from: ticks[0].pos, to: ticks[ticks.length - 1].pos, luminance: median(ends.map(e => e.line.luminance)), dark: false, edge: true };
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
    const gapOf = b => wordGap(b, h, side, edge);
    const pathInk = b => inkBetween(image, backgrounds, b, h, side, edge);
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
 * left-aligned axis labels, as on the economist chart), or a word off the line's end and level with it, which the
 * line would run through (ggplot2's theme_minimal, song-cents: the labels where an axis's would be, and no axis).
 * Labels of one axis sit at the same end, so each line takes the nearest word at the end most lines' nearest words
 * are at (economist: a dot rim at the left end of the zero line was nearer than its "0" at the right). A word level
 * with the line is as near as its middle is to it.
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
        // (without the line itself and what is joined to it: a ring on the line made one blob of both, as tall as
        // the search, which took the label off the line's end for a part of its word)
        const onLine = b => (h ? b[1] <= t.pos && b[3] > t.pos && b[2] > from && b[0] < to : b[0] <= t.pos && b[2] > t.pos && b[3] > from && b[1] < to);
        for (const b of mergeWords(inkComponents(image, backgrounds, rect, true).map(c => c.box).filter(b => !onLine(b)))) {
            const lo = h ? b[0] : b[1];
            const hi = h ? b[2] : b[3];
            const end = hi <= from + ext ? 'start' : (lo >= to - ext ? 'end' : null);
            const clear = h ? (b[3] <= t.pos - 1 || b[1] >= t.pos + 1) : (b[2] <= t.pos - 1 || b[0] >= t.pos + 1);
            const off = hi <= from + LABEL_BEYOND_TOL || lo >= to - LABEL_BEYOND_TOL;
            if (end && clear) {
                const d = h ? Math.min(Math.abs(b[3] - t.pos), Math.abs(b[1] - t.pos)) : Math.min(Math.abs(b[2] - t.pos), Math.abs(b[0] - t.pos));
                words.push({ b, end, d });
            }
            else if (end && off && (h ? b[1] <= t.pos - LABEL_BEYOND_TOL && b[3] >= t.pos + LABEL_BEYOND_TOL : b[0] <= t.pos - LABEL_BEYOND_TOL && b[2] >= t.pos + LABEL_BEYOND_TOL)) {
                // (… with some of it on either side: two "gridlines" 2 px apart, columns of light dots in a
                // synthetic swarm, each took a sliver of the dot beyond its end, and the plot ended there)
                words.push({ b, end, d: Math.abs((h ? (b[1] + b[3]) / 2 : (b[0] + b[2]) / 2) - t.pos) });
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
 * dark vertical line with the most ticks (ties: the leftmost). Without a dark line with ticks on an axis, the edge
 * of a filled panel with ticks is the axis (ggplot2's default theme: panelEdges). Without that, light
 * gridlines across the other direction stand in as its ticks (economist: no y axis line, horizontal gridlines),
 * with a dark baseline that continues their spacing (the economist's zero line); where their labels lie off their
 * ends, the ends are the axis line (gridEndLine), and the row of words under the last gridline of such a y axis is
 * the x axis (labelsUnder: song-cents). Without those either, a line at the other axis's end with a row of labels
 * and no tick marks is the axis, its labels the positions (labelAxis).
 *
 * @param {import('./image.js').RgbaImage} image
 * @returns {Axes}
 */
export function detectAxes(image) {
    // A panel is a tint of the page. A color as far from it as PANEL_MAX_CONTRAST is ink, however much of it there is:
    // mpo's black (dots of 13 px, axis lines and whiskers 7 px thick, bold labels) is 5.6% of the image, and as a
    // background its axis lines were no ink: the y axis had one tick and the top bracket, with two stars for
    // labels, was the x axis.
    const all = backgroundColors(image);
    const backgrounds = all.filter((c, i) => i === 0 || Math.abs(luminance(c, 0) - luminance(all[0], 0)) < PANEL_MAX_CONTRAST);
    const ink = grayInk(image, backgrounds);
    const lines = findLines(image, ink);
    // labeled ticks drawn as strokes (a corner tick, on a line, has none) are straight, and their labels are in
    // plain view of the axis
    const axisLike = ticks => {
        const strokes = ticks.filter(t => t.length > 0);
        return (!strokes.length || median(strokes.map(t => t.shortest / t.length)) >= TICK_MIN_STRAIGHTNESS)
            && median(ticks.map(t => t.labelPath)) <= LABEL_PATH_MAX_INK;
    };
    const edges = panelEdges(image, backgrounds);
    // A line through the middle of the plot (the other axis has labeled ticks on both sides of it), when another line
    // of its direction lies at an end of the other axis, is a reference line, and that other one the plot's edge.
    // Biomarker: the zero line of two beeswarms in a frame; the dots on it are strokes off it as ticks are, 10 of them,
    // with the dots above them for labels, and the plot above it was taken for the axis's side and masked. Axes that
    // cross in the middle of a plot with nothing at its edges stay axes.
    const reference = (line, other) => {
        const labeled = other?.line ? other.ticks.filter(t => t.label) : [];
        const reach = line.thickness / 2 + AXIS_CORNER_TOL;
        if (!labeled.some(t => t.pos < line.pos - reach) || !labeled.some(t => t.pos > line.pos + reach))
            return false;
        return lines.some(l => l !== line && l.dir === line.dir
            && Math.min(Math.abs(l.pos - other.line.from), Math.abs(l.pos - other.line.to)) <= l.thickness / 2 + AXIS_CORNER_TOL
            && Math.min(l.to, line.to) - Math.max(l.from, line.from) >= END_LINE_MIN_OVERLAP * (line.to - line.from));
    };
    // the line with the most ticks, of those given, that has labeled ticks as an axis has them (and isn't a
    // reference line across the other axis's plot)
    const among = (dir, given, other) => {
        const candidates = given.filter(l => l.dir === dir).map(line => ({ line, ticks: findTicks(image, ink, line, lines) }));
        candidates.sort((a, b) => (b.ticks.length - a.ticks.length) || (dir === 'h' ? b.line.pos - a.line.pos : a.line.pos - b.line.pos));
        for (const c of candidates) {
            if (reference(c.line, other))
                continue;
            findLabels(image, backgrounds, c.line, c.ticks);
            const labeled = c.ticks.filter(t => t.label);
            if (labeled.length >= AXIS_MIN_LABELED_TICKS && axisLike(labeled))
                return { ...c, ticks: c.ticks.filter(t => !t.onLine || t.label) };
        }
        return null;
    };
    const pick = (dir, other = null) => {
        let best = among(dir, lines.filter(l => l.dark), other) ?? among(dir, edges, other);
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
                best = { line: gridEndLine(dir, ticks, lines.filter(l => l.dir === other)), ticks };
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
    // (each against the other: the y axis as first found tells the x axis from a reference line, and that x axis
    // the y axis)
    let y = pick('v');
    let x = pick('h', y);
    if (x?.line)
        y = pick('v', x);
    // an axis with no tick marks and none standing in: its labels, under the last gridline of a y axis of gridlines
    // that have their labels off their ends (gridEndLine; a slanted light curve is "gridlines" too, a pixel or two
    // apart, with no labels, and the dots under it are no x axis), or at the plot's corner with the other axis
    const last = y?.ticks[y.ticks.length - 1];
    if (!x && last?.source === 'grid' && y.line?.edge)
        x = labelsUnder(image, backgrounds, lines.find(l => l.dir === 'h' && Math.abs(l.pos - last.pos) < 0.5));
    else if (!x && y?.line)
        x = labelAxis(image, backgrounds, lines, 'h', y.line);
    else if (!y && x?.line)
        y = labelAxis(image, backgrounds, lines, 'v', x.line);
    return { lines, backgrounds, x: finish(beyondOther(x, y)), y: finish(beyondOther(y, x)) };
}

/**
 * The plot area's sides that the axes show: an axis line with tick marks (or, without them, labels) whose labels
 * all lie on one side of it bounds the data on the other side (x labels below the line: the plot is above it). Gridline axes and axes with
 * labels on both sides bound nothing (the economist's zero line has data on both sides; a baseline through the
 * middle of a beeswarm would too).
 *
 * @param {Axes} axes  from detectAxes
 * @returns {{left: number|null, right: number|null, top: number|null, bottom: number|null}}  px, the axis lines'
 *   centers; null where unbounded
 */
export function plotBounds(axes) {
    const bounds = { left: null, right: null, top: null, bottom: null };
    const side = axis => {
        if (!axis?.line)
            return 0;
        // (gridlines bound nothing across themselves; where their labels lie off their ends, the ends do: gridEndLine)
        const labeled = axis.ticks.filter(t => (t.source !== 'grid' || axis.line.edge) && t.label);
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
 * @param {Axes} axes  from detectAxes
 * @returns {ClipEdge[]}
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
 *
 * @param {{type: string, a: number, b: number}} scale  from fitScale
 * @param {number} value  data units
 * @returns {number}  px
 */
export function toPixel(scale, value) {
    const v = scale.type === 'log' ? Math.log10(value) : value;
    return (v - scale.a) / scale.b;
}

/**
 * How much one pixel is in data units at a position: |dv/dpx| (constant for linear scales).
 *
 * @param {{type: string, a: number, b: number}} scale  from fitScale
 * @param {number} px  the position
 * @returns {number}  data units per px
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

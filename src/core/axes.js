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

const GRAY_CHROMA = 24; // 0–255: max(r, g, b) − min(r, g, b) for a gray pixel
const INK_TOL = 15; // 0–255 units (Euclidean RGB) from the background for a pixel to be ink
const DARK_LUMINANCE = 160; // 0–255: gray ink darker than this is "dark" (axes, ticks); lighter is gridline gray
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
const LABEL_ON_TICK_SPACINGS = 0.3; // a word belongs to a tick only if centered within this × spacing of it
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
 * @property {number} length  px
 * @property {'before'|'after'} side  'before' = above (x axis) or left (y axis)
 * @property {number[]|null} label  [x0, y0, x1, y1) box of the label's ink, or null
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
        if (Math.max(r, g, b) - Math.min(r, g, b) > GRAY_CHROMA || !isInk(data, i, backgrounds))
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
                    ticks.push({ pos, length: Math.max(...len.slice(a, b)), side, label: null, source: 'tick', onLine });
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
    return kept.sort((p, q) => p.pos - q.pos);
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
    const seen = new Uint8Array(w * h);
    const comps = [];
    for (let y = y0; y < y1; y++) {
        for (let x = x0; x < x1; x++) {
            if (seen[(y - y0) * w + x - x0] || !ink(x, y))
                continue;
            const stack = [[x, y]];
            seen[(y - y0) * w + x - x0] = 1;
            const box = [x, y, x + 1, y + 1];
            let n = 0;
            while (stack.length) {
                const [cx, cy] = stack.pop();
                n++;
                box[0] = Math.min(box[0], cx);
                box[1] = Math.min(box[1], cy);
                box[2] = Math.max(box[2], cx + 1);
                box[3] = Math.max(box[3], cy + 1);
                for (let dy = -1; dy <= 1; dy++) {
                    for (let dx = -1; dx <= 1; dx++) {
                        const nx = cx + dx;
                        const ny = cy + dy;
                        if (nx < x0 || ny < y0 || nx >= x1 || ny >= y1 || seen[(ny - y0) * w + nx - x0])
                            continue;
                        if (ink(nx, ny)) {
                            seen[(ny - y0) * w + nx - x0] = 1;
                            stack.push([nx, ny]);
                        }
                    }
                }
            }
            // ink touching the image border is a frame or a cut-off mark, never a label (pathogen: a 1 px border
            // at the right edge chained every x label and the axis title into one word)
            const touches = box[0] === 0 || box[1] === 0 || box[2] === image.width || box[3] === image.height;
            if (n >= LABEL_MIN_PIXELS && !touches)
                comps.push({ box, n });
        }
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
 * Label boxes for the ticks of an axis: ink beyond the ticks (on their side) merged into words; a word belongs to
 * the tick it is centered on (within LABEL_ON_TICK_SPACINGS of the spacing). Words between ticks (labels of
 * intervals, as on the economist chart) belong to none.
 *
 * @param {import('./image.js').RgbaImage} image
 * @param {number[][]} backgrounds
 * @param {Line} axis
 * @param {Tick[]} ticks  modified: each gets `label`
 */
export function findLabels(image, backgrounds, axis, ticks) {
    if (!ticks.length)
        return;
    const { width, height } = image;
    const h = axis.dir === 'h';
    const side = ticks.filter(t => t.side === 'after').length >= ticks.filter(t => t.side === 'before').length ? 'after' : 'before';
    const tickLen = Math.max(...ticks.map(t => t.length));
    const reach = LABEL_SEARCH_FRACTION[axis.dir] * (h ? height : width);
    const edge = side === 'after' ? axis.pos + axis.thickness / 2 + tickLen + 1 : axis.pos - axis.thickness / 2 - tickLen - 1;
    const c0 = Math.max(0, Math.floor(side === 'after' ? edge : edge - reach));
    const c1 = Math.min(h ? height : width, Math.ceil(side === 'after' ? edge + reach : edge));
    const spacing = ticks.length > 1 ? (ticks[ticks.length - 1].pos - ticks[0].pos) / (ticks.length - 1) : reach;
    const a0 = Math.max(0, Math.floor(ticks[0].pos - spacing / 2));
    const a1 = Math.min(h ? width : height, Math.ceil(ticks[ticks.length - 1].pos + spacing / 2));
    const rect = h ? [a0, c0, a1, c1] : [c0, a0, c1, a1];
    // Tick labels form a column (y) or row (x) at a consistent gap from the ticks; axis titles and other text are
    // farther out. Keep words whose gap is within half a label height of the smallest gap.
    const gapOf = b => (h ? (side === 'after' ? b[1] - edge : edge - b[3]) : (side === 'after' ? b[0] - edge : edge - b[2]));
    const assigned = [];
    for (const b of mergeWords(inkComponents(image, backgrounds, rect).map(c => c.box))) {
        const mid = h ? (b[0] + b[2]) / 2 : (b[1] + b[3]) / 2;
        let best = null;
        for (const t of ticks) {
            if (Math.abs(t.pos - mid) <= LABEL_ON_TICK_SPACINGS * spacing && (best === null || Math.abs(t.pos - mid) < Math.abs(best.pos - mid)))
                best = t;
        }
        if (best)
            assigned.push({ b, tick: best, gap: gapOf(b) });
    }
    if (!assigned.length)
        return;
    const minGap = Math.min(...assigned.map(a => a.gap));
    const heights = assigned.map(a => a.b[3] - a.b[1]).sort((p, q) => p - q);
    const tol = Math.max(LABEL_COLUMN_MIN_TOL, 0.5 * heights[Math.floor(heights.length / 2)]);
    for (const a of assigned) {
        if (a.gap <= minGap + tol)
            a.tick.label = union(a.tick.label, a.b);
    }
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
    const pick = dir => {
        const candidates = lines.filter(l => l.dir === dir && l.dark).map(line => ({ line, ticks: findTicks(image, ink, line, lines) }));
        candidates.sort((a, b) => (b.ticks.length - a.ticks.length) || (dir === 'h' ? b.line.pos - a.line.pos : a.line.pos - b.line.pos));
        let best = candidates.find(c => c.ticks.length >= 2) ?? null;
        if (best) {
            findLabels(image, backgrounds, best.line, best.ticks);
            best = { ...best, ticks: best.ticks.filter(t => !t.onLine || t.label) };
        }
        else {
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
        if (!best)
            return null;
        const gaps = best.ticks.slice(1).map((t, i) => t.pos - best.ticks[i].pos);
        const spacing = gaps.length ? gaps.reduce((sum, g) => sum + g, 0) / gaps.length : NaN;
        const regular = gaps.length > 0 && gaps.every(g => Math.abs(g - spacing) <= Math.max(1, 0.03 * spacing));
        return { ...best, spacing, regular };
    };
    return { lines, backgrounds, x: pick('h'), y: pick('v') };
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

// Text and legends, to mask (PLAN.md: testgraphs wages2025). A graph's titles, notes, labels and legend sit on the
// page beside or among the dots; the fit put dots on them wherever their ink reads as a family's (wages2025: the
// title, subtitle, the median line's label and the legend's markers, 105 extras of 767).
//
// Text is lines of glyphs: pieces of ink thin for their height (a stroke's half width under TEXT_MAX_STROKE of the
// glyph's height; a dot's inscribed radius is half its height) and thinner than a dot (TEXT_MAX_STROKE_RADII of the
// dots' radius), side by side along a row. A chain of touching dots is thin for its height too, and as black as text
// (bench-plos adults1: a cluster's pieces, strokes 13.5–20 px beside dots of 13, made a "line" and lost 18 dots), so
// the dots' radius is the test that holds; bold text as thick as a dot stays unmasked (wages2025's title: 15–17 px
// beside dots of 17.6). A legend is a column of markers, each just before a line of text; its markers are the dots'
// size and shape (wages2025: radius 17.1), so only their place tells them from data.
//
// Images enter as { width, height, data } RGBA; boxes are [x0, y0, x1, y1) px.

import { distanceTransform } from './layers.js';

const INK_TOL = 15; // 0–255 units from the page for a pixel to be ink
const GLYPH_MIN_HEIGHT = 6; // px: smaller pieces are specks or the dots of i and j, which join no line by themselves
const GLYPH_MIN_PIXELS = 10;
// a glyph's widest stroke: its inscribed radius at most this × its height (wages2025: title 0.18, subtitle and legend
// text 0.1–0.15, the median label 0.18–0.2; a disc 0.5)
const TEXT_MAX_STROKE = 0.3;
const TEXT_MAX_STROKE_RADII = 0.6; // × the dots' radius (adults1's tick labels 6–8 px beside dots of 13; wages2025's subtitle and legend 4–6 beside 17.6)
// … filling this share of its box at least (a plot's frame is thin for its height: testgraphs violin-beeswarm's, 0.02,
// made a "line" of the whole plot; letters fill 0.15 or more), and no wider than GLYPH_MAX_ASPECT × its height
const GLYPH_MIN_FILL = 0.08;
const GLYPH_MAX_ASPECT = 4;
// A line of text sits on a baseline: TEXT_BASELINE_SHARE of its glyphs end within TEXT_BASELINE_TOL of its median
// bottom (descenders aside). Pieces of overlapping dots don't: white rims cut them into crescents, thin for their
// height as glyphs are, which chained into "lines" in wages2025's dense middle.
const TEXT_BASELINE_SHARE = 0.6;
const TEXT_BASELINE_TOL = 0.08; // × the line's median glyph height, at least 2 px
// … and its glyphs vary: a row of hollow ring markers is one piece repeated (wages2025's lowercase letters are the
// dots' size, so size can't tell a letter o from a ring); a line with TEXT_MAX_SAME_SHARE of its glyphs within
// TEXT_SAME_TOL of its median glyph's width and height and TEXT_SAME_PIXELS of its pixel count is no text (o, n
// and e share a box, not a count: "one" was no line by boxes alone)
const TEXT_MAX_SAME_SHARE = 0.8;
const TEXT_SAME_TOL = 2; // px
const TEXT_SAME_PIXELS = 0.05;
// a legend marker is round within this aspect
const MARKER_ASPECT = 1.25;
// glyphs of one line: overlapping by this × the shorter one's height, apart by at most GLYPH_GAP_HEIGHTS × the taller
// one's, and heights within GLYPH_HEIGHT_RATIO (as axes.js mergeWords)
const GLYPH_OVERLAP = 0.5;
const GLYPH_GAP_HEIGHTS = 0.6;
const GLYPH_HEIGHT_RATIO = 2.5;
// … and one ink: their strokes' colors this close (a dot's piece beside a label joined it, and its box covered dots:
// wages2025's "Police officers")
const GLYPH_COLOR_TOL = 60; // 0–255 units
const TEXT_MIN_GLYPHS = 3; // glyphs for a line of text
// A legend marker: a round, filled piece (inscribed radius at least MARKER_MIN_FILL × its height, aspect within
// MARKER_ASPECT) before a line of text, within LEGEND_GAP_HEIGHTS × the line's height of its start and centered on it
// within LEGEND_CENTER_TOL × that height; LEGEND_MIN_ENTRIES of them in a column, centers within LEGEND_COLUMN_TOL
// (a data dot beside an annotation stands alone)
const MARKER_MIN_FILL = 0.35;
const LEGEND_GAP_HEIGHTS = 2;
const LEGEND_CENTER_TOL = 0.35;
const LEGEND_MIN_ENTRIES = 3;
const LEGEND_COLUMN_TOL = 3; // px
const BOX_PAD = 2; // px around each box, the antialiasing

/**
 * Boxes of text lines and of legend markers, to mask. O(pixels) for the ink's pieces and their widths, then
 * O(glyphs · glyphs within reach) for the lines.
 *
 * @param {import('./image.js').RgbaImage} image  the image as analyzed (not a family's virtual image)
 * @param {number[]} page  0–255, the page's color (axes.backgroundColors' first): other common colors, a region's or
 *   a dot family's, are ink (bench3 dots-swarm-violin: with the dots' gray as a background, their rims were "text")
 * @param {number} radius  px, the dots' radius: strokes as thick are no text
 * @returns {{text: number[][], legend: number[][]}}  boxes [x0, y0, x1, y1) px, padded
 */
export function annotationBoxes(image, page, radius) {
    const pieces = inkPieces(image, page);
    const glyphs = pieces.filter(c => {
        const h = c.y1 - c.y0;
        const w = c.x1 - c.x0;
        return h >= GLYPH_MIN_HEIGHT && w <= GLYPH_MAX_ASPECT * h && c.n >= GLYPH_MIN_PIXELS && c.n >= GLYPH_MIN_FILL * w * h
            && c.r <= TEXT_MAX_STROKE * h && c.r <= TEXT_MAX_STROKE_RADII * radius;
    });
    const lines = textLines(glyphs);
    const pad = b => [b[0] - BOX_PAD, b[1] - BOX_PAD, b[2] + BOX_PAD, b[3] + BOX_PAD];
    // markers before lines
    const markers = [];
    for (const c of pieces) {
        const w = c.x1 - c.x0;
        const h = c.y1 - c.y0;
        if (c.n < GLYPH_MIN_PIXELS || Math.max(w, h) > MARKER_ASPECT * Math.min(w, h) || c.r < MARKER_MIN_FILL * h)
            continue;
        const cy = (c.y0 + c.y1) / 2;
        const line = lines.find(l => {
            const lh = l[3] - l[1];
            return h <= LEGEND_GAP_HEIGHTS * lh && l[0] >= c.x1 && l[0] - c.x1 <= LEGEND_GAP_HEIGHTS * lh
                && Math.abs((l[1] + l[3]) / 2 - cy) <= LEGEND_CENTER_TOL * lh;
        });
        if (line)
            markers.push({ box: [c.x0, c.y0, c.x1, c.y1], cx: (c.x0 + c.x1) / 2 });
    }
    const legend = markers.filter(m => markers.filter(o => Math.abs(o.cx - m.cx) <= LEGEND_COLUMN_TOL).length >= LEGEND_MIN_ENTRIES);
    return { text: lines.map(pad), legend: legend.map(m => pad(m.box)) };
}

// 8-connected pieces of ink: bounding box, pixel count, inscribed radius (the largest distance from one of its pixels
// to the nearest background pixel), px, and the color there (a stroke's middle: its ink, not the antialiasing), 0–255
function inkPieces(image, page) {
    const { width, height, data } = image;
    const n = width * height;
    const ink = new Uint8Array(n);
    const onPage = new Uint8Array(n);
    for (let p = 0; p < n; p++) {
        const r = data[4 * p];
        const g = data[4 * p + 1];
        const b = data[4 * p + 2];
        const isPage = Math.hypot(r - page[0], g - page[1], b - page[2]) <= INK_TOL;
        ink[p] = isPage ? 0 : 1;
        onPage[p] = isPage ? 1 : 0;
    }
    const d2 = distanceTransform(onPage, width, height);
    const seen = new Uint8Array(n);
    const stack = new Int32Array(n);
    const pieces = [];
    for (let s = 0; s < n; s++) {
        if (!ink[s] || seen[s])
            continue;
        const c = { x0: width, y0: height, x1: 0, y1: 0, n: 0, r: 0, color: null };
        let maxD = -1;
        let at = s;
        let top = 0;
        stack[top++] = s;
        seen[s] = 1;
        while (top) {
            const p = stack[--top];
            const x = p % width;
            const y = (p - x) / width;
            c.n++;
            c.x0 = Math.min(c.x0, x);
            c.x1 = Math.max(c.x1, x + 1);
            c.y0 = Math.min(c.y0, y);
            c.y1 = Math.max(c.y1, y + 1);
            if (d2[p] > maxD) {
                maxD = d2[p];
                at = p;
            }
            for (let dy = -1; dy <= 1; dy++) {
                for (let dx = -1; dx <= 1; dx++) {
                    const xx = x + dx;
                    const yy = y + dy;
                    if (xx < 0 || yy < 0 || xx >= width || yy >= height)
                        continue;
                    const q = yy * width + xx;
                    if (ink[q] && !seen[q]) {
                        seen[q] = 1;
                        stack[top++] = q;
                    }
                }
            }
        }
        c.r = Math.sqrt(maxD);
        c.color = [data[4 * at], data[4 * at + 1], data[4 * at + 2]];
        pieces.push(c);
    }
    return pieces;
}

// glyphs joined into lines (union-find over neighbors along a row); lines of TEXT_MIN_GLYPHS or more, as boxes
function textLines(glyphs) {
    const order = glyphs.map((_, i) => i).sort((a, b) => glyphs[a].x0 - glyphs[b].x0);
    const parent = glyphs.map((_, i) => i);
    const find = i => {
        while (parent[i] !== i) {
            parent[i] = parent[parent[i]];
            i = parent[i];
        }
        return i;
    };
    const maxH = Math.max(0, ...glyphs.map(g => g.y1 - g.y0));
    for (let a = 0; a < order.length; a++) {
        const g = glyphs[order[a]];
        for (let b = a + 1; b < order.length; b++) {
            const o = glyphs[order[b]];
            if (o.x0 - g.x1 > GLYPH_GAP_HEIGHTS * maxH)
                break;
            const hg = g.y1 - g.y0;
            const ho = o.y1 - o.y0;
            const taller = Math.max(hg, ho);
            const shorter = Math.min(hg, ho);
            const overlap = Math.min(g.y1, o.y1) - Math.max(g.y0, o.y0);
            const gap = Math.max(g.x0, o.x0) - Math.min(g.x1, o.x1);
            if (overlap >= GLYPH_OVERLAP * shorter && gap <= GLYPH_GAP_HEIGHTS * taller && taller <= GLYPH_HEIGHT_RATIO * shorter
                && Math.hypot(g.color[0] - o.color[0], g.color[1] - o.color[1], g.color[2] - o.color[2]) <= GLYPH_COLOR_TOL)
                parent[find(order[a])] = find(order[b]);
        }
    }
    const groups = new Map();
    glyphs.forEach((g, i) => {
        const k = find(i);
        if (!groups.has(k))
            groups.set(k, []);
        groups.get(k).push(g);
    });
    const median = a => [...a].sort((u, v) => u - v)[a.length >> 1];
    const lines = [];
    for (const gs of groups.values()) {
        if (gs.length < TEXT_MIN_GLYPHS)
            continue;
        const h = median(gs.map(g => g.y1 - g.y0));
        const w = median(gs.map(g => g.x1 - g.x0));
        const bottom = median(gs.map(g => g.y1));
        const tol = Math.max(2, TEXT_BASELINE_TOL * h);
        const onBase = gs.filter(g => Math.abs(g.y1 - bottom) <= tol).length;
        const n = median(gs.map(g => g.n));
        const same = gs.filter(g => Math.abs(g.y1 - g.y0 - h) <= TEXT_SAME_TOL && Math.abs(g.x1 - g.x0 - w) <= TEXT_SAME_TOL
            && Math.abs(g.n - n) <= TEXT_SAME_PIXELS * n).length;
        if (onBase < TEXT_BASELINE_SHARE * gs.length || same >= TEXT_MAX_SAME_SHARE * gs.length)
            continue;
        lines.push([Math.min(...gs.map(g => g.x0)), Math.min(...gs.map(g => g.y0)), Math.max(...gs.map(g => g.x1)), Math.max(...gs.map(g => g.y1))]);
    }
    return lines;
}

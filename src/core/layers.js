// Layers and masks (PLAN.md §2.3, stages 2–3). Each pixel is background, explained by the dot family (on its
// color line, at any overlap count), or other. The fit ignores masked pixels: the "other" pixels, plus opaque thin
// marks in the family's own full color (the economist trend lines are the dots' c), dilated so antialiased fringes
// don't count either.
//
// Morphology uses the exact Euclidean distance transform, so dilation and opening by a disc cost O(pixels)
// whatever the radius.

import { colorDistance, sub, dot } from './image.js';

export const CLASS_BACKGROUND = 0;
export const CLASS_FAMILY = 1;
export const CLASS_OTHER = 2;

// Opaque marks in the family's full color (the economist trend lines) are saturated: within a few units of c. For
// an opaque family every pixel of its color is a candidate, and the opening and length tests alone decide.
// Translucent dots get that close only in deep stacks (α 0.3: about 10 dots), and such piles are compact, while a
// line is long and narrower than a dot. A "deep" pixel is saturated and beyond the measured levels; deep pixels
// that an opening removes, in components long enough to be lines, are thin marks.
const SATURATION_TOL = 8; // 0–255 units from the saturated color
const PAST_FULL = 1.5; // 0–255 units, over α²: how far past the full color on the line a pixel is still saturated
const THIN_EDGE_REACH = 2; // px from a thin mark: ink farther past the full color there is the mark's edge
// The fitted c extrapolates from the levels (economist red: 2 levels, c off by 4.5 units in a synthetic copy); a
// plateau measured beyond the levels near c is the opaque mark's own color and a better anchor.
const ANCHOR_MAX_OFFSET = 20; // 0–255 units: a beyond-level this close to c anchors the saturation test
const DEEP_EXTRA_LEVELS = 2; // deep pixels are also at least this many levels beyond the deepest measured one
const OPEN_RADIUS_FRACTION = 0.75; // × dot radius: the opening removes marks narrower than 1.5 radii (economist
                                   // trend lines are ~9 px wide, r 8.4)
const THIN_MIN_LENGTH_RADII = 3; // × dot radius, the larger side of a thin component's bounding box
// The opening also shaves one-pixel slivers off jagged rims, and where opaque dots touch in a chain the slivers
// join into long components (synthetic swarm: 8 of 84 dots masked). A second, 1 px opening of the candidates
// removes the slivers, and with them lines 2 px wide.
const SLIVER_OPENING = 1; // px
// A line 2 px wide is thin too where it runs straight along a row or a column for the length a thin mark needs:
// frames, axes, error bars. A sliver along a rim is an arc, flat for a few pixels. What is 2 px wide is what an
// opening by a 2 × 2 block keeps. funnel: the plot's frame, 2 px in the dots' stroke color, stayed a dot's worth of
// coverage all along, and the fit strung 45 dots on it, hidden beyond the plot's edge. (Every 2 px mark taken as
// thin moved 24 rows of the benchmark: 12 fewer extras, 4 more, and positions worse where rims were masked:
// adults1's rmse 0.06 → 0.13 px.)
// Dots sitting on a bar of their own color cut it into pieces shorter than the length test (Laser's brackets: 5–9 px
// between dots 11 px across, 10 dots fitted on the bars). Straight pieces in one row (or column), one's rows within
// the other's, filling their boxes as a bar does and a crescent doesn't, with gaps no wider than a dot, are one line,
// as is such a piece in line with a thin mark (barPieces).
const BAR_FILL = 0.7; // share of a piece's bounding box (Laser's corner piece, rounded: 0.72)
// a piece is longer than the bar is thick (Laser's: 1.4–2.4 times); a dotted line's round specks are not, and masked
// as bars they cut into the dots beside them (adults1: a dot added to a cluster)
const BAR_MIN_ELONGATION = 1.3; // a piece's length over its thickness
const BAR_EDGE_TOL = 1; // px, between pieces' rows (columns) and in a thin mark's cover of them
const BAR_GAP_SLACK = 1; // px beyond a dot's diameter between pieces
// A dashed line in the dots' color, at any angle, is cut into pieces by its own gaps as well as by the dots
// (prot-cath-chris: ggplot2's fitted curves 10 px wide in the families' colors, dashes 15–45 px long, dots r 9.6;
// the dashes shorter than the length test were explained by 63 extra dots, most beside a real one). The pieces are
// measured along their principal axis instead of a row or column (dashPieces); a curve turns a little between dashes.
const DASH_ANGLE_TOL = 0.35; // rad between two pieces' directions
const DASH_ACROSS_TOL = 1.5; // px beyond half a piece's thickness: how far off its axis the next piece's center lies
// Between two dashes the line may pass under a dot as well as through its own gap: the dot's pixels are the
// family's (deep), and kept whole by the opening. A chain reaches across this many gaps (a dot's diameter each) as
// long as no stretch of its axis without deep pixels is longer than one.
const DASH_REACH_GAPS = 3;
// A dash keeps its width to its ends, where a piece of a dot tapers: half a disc (a dot clipped at the plot's edge)
// fills π/4 of its box, as a dash's antialiased pixels may, but only 0.61 of the box's outer quarters. A chain
// starts only at a piece filling its ends this much (prot-cath-chris: 8 extras without the test, 5 at 0.65, 11 at
// 0.7, 32 at 0.8, where its own dashes stop starting chains)
const DASH_END_FILL = 0.65;
// Between two long pieces (a solid line through a dot of its color) only the stubs are bridged, this far into the gap
// from each end: the line's pixels the opening kept beside the dot, which a second dot explained (wave1: a purple dot
// doubled 0.9 px off where its line leaves it; none at 0.2, 0.35 or 0.5; bridged all the way, shannon's dots on a
// median line 6 px wide, r 6.2, were masked whole and lost)
const DASH_STUB_RADII = 0.2;
// A pixel at count ≥ 1.5 is inside at least one dot, so a disc of nearly the dot's radius around it lies in the
// family's pixels (or under other marks). Text in a translucent family's own color fails that: JMP's 50% black dots
// share their line with black text, whose antialiased strokes read as counts 1–8, too short for the length test,
// and the fit put 15 dots on the tick labels and title.
const SOLID_COUNT = 1.5; // overlap count
const SOLID_OPEN_FRACTION = 0.9; // × dot radius: the disc that must fit
const MASK_DILATE_BASE = 1; // px
const OVER_INK_TOL = 15; // 0–255 units (RGB distance): an edge ink this close to a family drawn over this one is its
const MASK_DILATE_BLURS = 2; // px per px of edge blur
// Beyond the axes is no data, only labels, whose round glyphs ("0") read as small dots (bench2 tiny-dots: three dots
// in the "100" below the x axis). Pixels farther out than this are masked; a dot centered on the axis line (data at
// the scale's end) keeps every pixel.
const OUTSIDE_MARGIN_RADII = 1; // × dot radius beyond the axis line
// An outlined dot's fill never meets the page: its outline lies between. A patch of the fill's color that does, at
// BARE_MIN_PIXELS of its pixels and BARE_MIN_SHARE of its border, is a line or curve in that color, not part of a
// dot (bareFill). "Meets": the page within BARE_REACH px, one antialiased pixel between; so only for outlines at least
// BARE_MIN_OUTLINE px wide, which keep a fill farther from the page than that.
const BARE_REACH = 2; // px
const BARE_MIN_PIXELS = 6;
const BARE_MIN_SHARE = 0.1;
const BARE_MIN_OUTLINE = 2; // px
// Some plots clip their dots at the plot area (autocor1: at the y axis and where the x axis line ends, a straight cut
// through dots). Beyond such a cut nothing can be seen, so it is masked with no margin, and a cut dot is fitted whole,
// half of it unseen; read as white, the cut was its edge, and the fit moved the dots in and doubled them. A cut is
// the family's coverage full (≥ CLIP_FULL) just inside and empty (≤ CLIP_EMPTY) just beyond, CLIP_GAP px out past
// the line (or the boundary pixel at a line's end), in CLIP_MIN_SHARE of the rows full inside and in CLIP_MIN_RADII ×
// the radius of them at least: a dot's round edge meets a line in a few rows, a cut in a dot's height.
const CLIP_FULL = 0.9;
const CLIP_EMPTY = 0.1;
const CLIP_GAP = 1.5; // px
const CLIP_SEARCH = 3; // px either side of a line's end for the cut's column
const CLIP_MIN_SHARE = 0.7;
const CLIP_MIN_RADII = 2;
// At a line's end the panel may have a border across it that reads as the family's ink (bench-r2 gg-loess-groups:
// theme_bw's gray border, 2 px, is what translucent red and teal over each other make, and unmixing gave each family
// 0.6 to 2 dots of it). Its columns then pass as the cut dots' last full ones, the cut fell just beyond it, and dots
// centered past the cut explained the border, 51 of them in a column. Columns just inside a cut with the family's ink
// in this share of the edge's rows are a line, not dots, and the cut moves in over them, CLIP_SEARCH px at most.
const CLIP_BORDER_SHARE = 0.9;
// Stand-in for an infinite distance in the transform: finite so the parabola intersections stay exact (q² ≤ ~10⁷
// px² is far below the precision of FAR), and far beyond any real squared distance.
const FAR = 1e12; // px²

/**
 * Squared Euclidean distance from each pixel to the nearest pixel where mask is 1 (Infinity if there is none).
 * Felzenszwalb & Huttenlocher (2012), the lower envelope of parabolas, by columns then rows: O(pixels).
 *
 * @param {Uint8Array} mask
 * @param {number} width
 * @param {number} height
 * @returns {Float32Array}
 */
export function distanceTransform(mask, width, height) {
    const n = Math.max(width, height);
    const f = new Float64Array(n);
    const d = new Float64Array(n);
    const v = new Int32Array(n);
    const z = new Float64Array(n + 1);
    const out = new Float32Array(width * height);

    // squared distance along one line: d[q] = min_p (q − p)² + f[p]
    function dt1(len) {
        let k = 0;
        v[0] = 0;
        z[0] = -Infinity;
        z[1] = Infinity;
        for (let q = 1; q < len; q++) {
            let s = ((f[q] + q * q) - (f[v[k]] + v[k] * v[k])) / (2 * q - 2 * v[k]);
            while (s <= z[k]) {
                k--;
                s = ((f[q] + q * q) - (f[v[k]] + v[k] * v[k])) / (2 * q - 2 * v[k]);
            }
            k++;
            v[k] = q;
            z[k] = s;
            z[k + 1] = Infinity;
        }
        k = 0;
        for (let q = 0; q < len; q++) {
            while (z[k + 1] < q)
                k++;
            d[q] = (q - v[k]) ** 2 + f[v[k]];
        }
    }

    for (let x = 0; x < width; x++) {
        for (let y = 0; y < height; y++)
            f[y] = mask[y * width + x] ? 0 : FAR;
        dt1(height);
        for (let y = 0; y < height; y++)
            out[y * width + x] = d[y];
    }
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++)
            f[x] = out[y * width + x];
        dt1(width);
        for (let x = 0; x < width; x++)
            out[y * width + x] = d[x] >= FAR / 2 ? Infinity : d[x];
    }
    return out;
}

/**
 * Index of the nearest pixel where mask is 1 (Euclidean; −1 if there is none): the distance transform's parabola
 * envelope, keeping which parabola wins. O(pixels).
 *
 * @param {Uint8Array} mask
 * @param {number} width
 * @param {number} height
 * @returns {Int32Array}
 */
export function nearestTransform(mask, width, height) {
    const n = Math.max(width, height);
    const f = new Float64Array(n);
    const v = new Int32Array(n);
    const z = new Float64Array(n + 1);
    const colD = new Float64Array(width * height);
    const colRow = new Int32Array(width * height).fill(-1);
    const out = new Int32Array(width * height).fill(-1);
    const arg = new Int32Array(n);

    // per line: for each q, the p minimizing (q − p)² + f[p]; writes arg[q] and returns nothing (f FAR = none)
    function nearest1(len) {
        let k = 0;
        v[0] = 0;
        z[0] = -Infinity;
        z[1] = Infinity;
        for (let q = 1; q < len; q++) {
            let s = ((f[q] + q * q) - (f[v[k]] + v[k] * v[k])) / (2 * q - 2 * v[k]);
            while (s <= z[k]) {
                k--;
                s = ((f[q] + q * q) - (f[v[k]] + v[k] * v[k])) / (2 * q - 2 * v[k]);
            }
            k++;
            v[k] = q;
            z[k] = s;
            z[k + 1] = Infinity;
        }
        k = 0;
        for (let q = 0; q < len; q++) {
            while (z[k + 1] < q)
                k++;
            arg[q] = v[k];
        }
    }

    for (let x = 0; x < width; x++) {
        for (let y = 0; y < height; y++)
            f[y] = mask[y * width + x] ? 0 : FAR;
        nearest1(height);
        for (let y = 0; y < height; y++) {
            const r = arg[y];
            if (mask[r * width + x]) {
                colRow[y * width + x] = r;
                colD[y * width + x] = (y - r) ** 2;
            }
            else {
                colD[y * width + x] = FAR;
            }
        }
    }
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++)
            f[x] = colD[y * width + x];
        nearest1(width);
        for (let x = 0; x < width; x++) {
            const c = arg[x];
            const r = colRow[y * width + c];
            if (r >= 0)
                out[y * width + x] = r * width + c;
        }
    }
    return out;
}

/**
 * Pixels within distance r of the mask (a disc dilation).
 *
 * @param {Uint8Array} mask
 * @param {number} width
 * @param {number} height
 * @param {number} r  px
 * @returns {Uint8Array}
 */
export function dilate(mask, width, height, r) {
    const d2 = distanceTransform(mask, width, height);
    const out = new Uint8Array(mask.length);
    for (let p = 0; p < mask.length; p++)
        out[p] = d2[p] <= r * r ? 1 : 0;
    return out;
}

/**
 * Pixels of the mask farther than r from every non-mask pixel (a disc erosion; the image border is not treated
 * as outside).
 *
 * @param {ArrayLike<number>} mask  nonzero where set, width × height
 * @param {number} width
 * @param {number} height
 * @param {number} r  px
 * @returns {Uint8Array}  1 or 0 per pixel
 */
export function erode(mask, width, height, r) {
    const inv = new Uint8Array(mask.length);
    for (let p = 0; p < mask.length; p++)
        inv[p] = mask[p] ? 0 : 1;
    const d2 = distanceTransform(inv, width, height);
    const out = new Uint8Array(mask.length);
    for (let p = 0; p < mask.length; p++)
        out[p] = mask[p] && d2[p] > r * r ? 1 : 0;
    return out;
}

/**
 * Opening by a disc of radius r: the union of all discs of radius r that fit inside the mask. Removes parts
 * thinner than 2r and keeps blobs that contain such a disc.
 *
 * @param {ArrayLike<number>} mask  nonzero where set, width × height
 * @param {number} width
 * @param {number} height
 * @param {number} r  px
 * @returns {Uint8Array}  1 or 0 per pixel
 */
export function opening(mask, width, height, r) {
    return dilate(erode(mask, width, height, r), width, height, r);
}

/**
 * Opening by a 2 × 2 block: the pixels that lie in some 2 × 2 block of the mask. Removes what is 1 px wide and keeps
 * lines of 2.
 *
 * @param {ArrayLike<number>} mask  nonzero where set, width × height
 * @param {number} width
 * @param {number} height
 * @returns {Uint8Array}  1 or 0 per pixel
 */
export function blockOpening(mask, width, height) {
    const out = new Uint8Array(mask.length);
    for (let y = 0; y < height - 1; y++) {
        for (let x = 0; x < width - 1; x++) {
            const p = y * width + x;
            if (mask[p] && mask[p + 1] && mask[p + width] && mask[p + width + 1]) {
                out[p] = 1;
                out[p + 1] = 1;
                out[p + width] = 1;
                out[p + width + 1] = 1;
            }
        }
    }
    return out;
}

// Marks as thin the pixels of `mask` in a run along a row or a column at least `length` px long. O(pixels).
function markRuns(mask, thin, width, height, length) {
    const along = (count, step, start) => {
        let run = 0;
        for (let i = 0; i <= count; i++) {
            if (i < count && mask[start + i * step]) {
                run++;
                continue;
            }
            for (let k = i - run; run >= length && k < i; k++)
                thin[start + k * step] = 1;
            run = 0;
        }
    };
    for (let y = 0; y < height; y++)
        along(width, 1, y * width);
    for (let x = 0; x < width; x++)
        along(height, width, x);
}

/**
 * Marks as thin the pieces of bars cut by dots: short components (components8's), longer along the bar than across
 * it (BAR_MIN_ELONGATION), no thicker than the opening removes and filling their boxes (BAR_FILL), that line up in a row (or a column):
 * with each other, across gaps up to a dot's diameter, together reaching the length test; or with a thin mark across
 * such a gap (a bar's rest, long enough alone). A piece lines up with another when its rows (columns) lie within the
 * other's, BAR_EDGE_TOL either side. The bar goes on under the dots in the gaps: its deep pixels there, in the rows
 * (columns) the two sides share, are marked too (else the strip of bar below a dot's edge was a second dot's). In
 * place. O(pieces² + pieces · radius · thickness).
 *
 * @param {{pixels: number[], x0: number, y0: number, x1: number, y1: number}[]} pieces
 * @param {Uint8Array} deep
 * @param {Uint8Array} thin
 * @param {number} width
 * @param {number} height
 * @param {number} radius  px, the dots'
 */
function barPieces(pieces, deep, thin, width, height, radius) {
    const thick = 2 * OPEN_RADIUS_FRACTION * radius;
    const gap = 2 * radius + BAR_GAP_SLACK;
    const done = new Set();
    const mark = c => {
        done.add(c);
        for (const p of c.pixels)
            thin[p] = 1;
    };
    // bars in rows, then in columns: the box keys along the bar and across it, and a pixel at (along, across)
    const axes = [['x0', 'x1', 'y0', 'y1', (a, b) => b * width + a, width], ['y0', 'y1', 'x0', 'x1', (a, b) => a * width + b, height]];
    for (const [lo, hi, across0, across1, at, extent] of axes) {
        const bars = pieces.filter(c => {
            const along = c[hi] - c[lo] + 1;
            const across = c[across1] - c[across0] + 1;
            return along >= BAR_MIN_ELONGATION * across && across <= thick && c.pixels.length >= BAR_FILL * along * across;
        }).sort((a, b) => a[lo] - b[lo]);
        // the deep pixels between along-positions a0 and a1 (exclusive), across b0..b1
        const bridge = (a0, a1, b0, b1) => {
            for (let a = Math.min(a0, a1) + 1; a < Math.max(a0, a1); a++) {
                for (let b = b0; b <= b1; b++) {
                    const p = at(a, b);
                    if (deep[p])
                        thin[p] = 1;
                }
            }
        };
        const within = (a, b) => a[across0] >= b[across0] - BAR_EDGE_TOL && a[across1] <= b[across1] + BAR_EDGE_TOL;
        const inLine = (a, b) => within(a, b) || within(b, a);
        const used = new Set();
        for (const start of bars) {
            if (used.has(start))
                continue;
            const group = [start];
            let end = start[hi];
            for (const c of bars) {
                if (!used.has(c) && !group.includes(c) && c[lo] >= start[lo] && inLine(c, start) && c[lo] - end - 1 <= gap) {
                    group.push(c);
                    end = Math.max(end, c[hi]);
                }
            }
            for (const c of group)
                used.add(c);
            if (group.length >= 2 && end - start[lo] + 1 >= THIN_MIN_LENGTH_RADII * radius) {
                group.forEach(mark);
                for (let i = 1; i < group.length; i++) {
                    const [a, b] = [group[i - 1], group[i]];
                    bridge(a[hi], b[lo], Math.max(a[across0], b[across0]), Math.min(a[across1], b[across1]));
                }
            }
        }
        // a thin mark on either side, covering the piece's rows (columns) BAR_EDGE_TOL short at most, within the gap;
        // repeated, as a piece marked thin may be the next one's neighbor
        const meets = c => {
            const need = c[across1] - c[across0] + 1 - BAR_EDGE_TOL;
            for (const [from, step] of [[c[lo] - 1, -1], [c[hi] + 1, 1]]) {
                for (let k = 0, a = from; k < gap && a >= 0 && a < extent; k++, a += step) {
                    let n = 0;
                    for (let b = c[across0]; b <= c[across1]; b++)
                        n += thin[at(a, b)];
                    if (n >= need)
                        return a;
                }
            }
            return null;
        };
        let changed = true;
        while (changed) {
            changed = false;
            for (const c of bars) {
                const a = done.has(c) ? null : meets(c);
                if (a !== null) {
                    mark(c);
                    bridge(a < c[lo] ? a : c[hi], a < c[lo] ? c[lo] : a, c[across0], c[across1]);
                    changed = true;
                }
            }
        }
    }
}

// a piece's principal axis: its center (px), unit direction (ux, uy), extent along it (a0..a1) and across (b0..b1),
// both measured on pixel centers and widened by a pixel, as a bounding box counts them; and `ends`, the share of the
// box's outer quarters along it that the piece fills
function pieceFrame(c, width) {
    let sx = 0;
    let sy = 0;
    for (const p of c.pixels) {
        sx += p % width;
        sy += Math.floor(p / width);
    }
    const n = c.pixels.length;
    const cx = sx / n;
    const cy = sy / n;
    let xx = 0;
    let yy = 0;
    let xy = 0;
    for (const p of c.pixels) {
        const dx = p % width - cx;
        const dy = Math.floor(p / width) - cy;
        xx += dx * dx;
        yy += dy * dy;
        xy += dx * dy;
    }
    const angle = 0.5 * Math.atan2(2 * xy, xx - yy);
    const ux = Math.cos(angle);
    const uy = Math.sin(angle);
    let a0 = Infinity;
    let a1 = -Infinity;
    let b0 = Infinity;
    let b1 = -Infinity;
    for (const p of c.pixels) {
        const dx = p % width - cx;
        const dy = Math.floor(p / width) - cy;
        const a = dx * ux + dy * uy;
        const b = -dx * uy + dy * ux;
        a0 = Math.min(a0, a);
        a1 = Math.max(a1, a);
        b0 = Math.min(b0, b);
        b1 = Math.max(b1, b);
    }
    // the share of the box's outer quarters (along) that the piece fills
    const quarter = (a1 - a0 + 1) / 4;
    let outer = 0;
    for (const p of c.pixels) {
        const a = (p % width - cx) * ux + (Math.floor(p / width) - cy) * uy;
        if (a < a0 - 0.5 + quarter || a > a1 + 0.5 - quarter)
            outer++;
    }
    const ends = outer / (2 * quarter * (b1 - b0 + 1));
    return { c, cx, cy, ux, uy, angle, a0: a0 - 0.5, a1: a1 + 0.5, b0: b0 - 0.5, b1: b1 + 0.5, n, ends };
}

/**
 * Marks as thin the pieces of dashed lines at any angle. A chain starts at a dash: a component, short or long enough
 * to be a thin mark already, left unmarked by barPieces, longer along its principal axis than across it (if short),
 * no thicker than the opening removes, filling its oriented box and the box's ends. It grows along the dash's strip
 * (its thickness, DASH_ACROSS_TOL either side) across gaps up to a dot's diameter (DASH_REACH_GAPS of them where the
 * axis runs through deep pixels, under dots), taking in any piece lying wholly in the strip (the rest of a dash cut
 * by a dot is ragged where the dot was, and fills no box), and following each dash it takes in that turns no more
 * than DASH_ANGLE_TOL. A chain of two pieces or more that reaches the length test is a line: its pieces are marked,
 * and so are its deep pixels in the strip between them and on to a thin mark just past its ends (the line goes on
 * under the dots), except between two long pieces, a solid line through a dot of its color: there only the first
 * DASH_STUB_RADII of the gap from each end. In place. O(pieces² + pieces · radius · thickness).
 *
 * @param {{pixels: number[], x0: number, y0: number, x1: number, y1: number}[]} pieces  short components
 * @param {{pixels: number[], x0: number, y0: number, x1: number, y1: number}[]} long  the components long enough to be
 *   thin marks alone
 * @param {{pixels: number[], x0: number, y0: number, x1: number, y1: number}[]} leaning  pieces of the family's ink
 *   that the opening kept only with other ink beside them: a chain may take them in, never start at one
 * @param {Uint8Array} deep
 * @param {Uint8Array} thin
 * @param {number} width
 * @param {number} height
 * @param {number} radius  px, the dots'
 */
function dashPieces(pieces, long, leaning, deep, thin, width, height, radius) {
    const thick = 2 * OPEN_RADIUS_FRACTION * radius;
    const gap = 2 * radius + BAR_GAP_SLACK;
    const length = THIN_MIN_LENGTH_RADII * radius;
    const dashLike = (f, short) => {
        const along = f.a1 - f.a0;
        const across = f.b1 - f.b0;
        return (!short || along >= BAR_MIN_ELONGATION * across) && across <= thick && f.n >= BAR_FILL * along * across && f.ends >= DASH_END_FILL;
    };
    const frames = pieces.filter(c => !thin[c.pixels[0]]).map(c => pieceFrame(c, width));
    for (const f of frames)
        f.dash = dashLike(f, true);
    for (const c of leaning)
        frames.push({ ...pieceFrame(c, width), dash: false });
    for (const c of long) {
        const f = pieceFrame(c, width);
        if (dashLike(f, false))
            frames.push({ ...f, dash: true, long: true });
    }
    const turn = (p, q) => {
        const d = Math.abs(p.angle - q.angle) % Math.PI;
        return Math.min(d, Math.PI - d);
    };
    // a piece's pixels along and across the axis of frame f, through f's center
    const extent = (f, q) => {
        const half = (f.b1 - f.b0) / 2 + DASH_ACROSS_TOL;
        let a0 = Infinity;
        let a1 = -Infinity;
        let inside = 0;
        for (const p of q.c.pixels) {
            const dx = p % width - f.cx;
            const dy = Math.floor(p / width) - f.cy;
            const a = dx * f.ux + dy * f.uy;
            a0 = Math.min(a0, a - 0.5);
            a1 = Math.max(a1, a + 0.5);
            if (Math.abs(-dx * f.uy + dy * f.ux) <= half)
                inside++;
        }
        return { a0, a1, inStrip: inside === q.n };
    };
    const point = (f, a) => [f.cx + 0.5 + a * f.ux, f.cy + 0.5 + a * f.uy];
    const at = ([x, y]) => (x >= 0 && y >= 0 && x < width && y < height ? Math.floor(y) * width + Math.floor(x) : -1);
    // along f's axis from a for d px (side: +1 or −1), no stretch without deep pixels longer than a gap
    const crossable = (f, a, side, d) => {
        let run = 0;
        for (let s = 1; s <= d; s++) {
            const p = at(point(f, a + side * s));
            run = p >= 0 && deep[p] ? 0 : run + 1;
            if (run > gap)
                return false;
        }
        return true;
    };
    // the deep pixels in the strip from point (x0, y0) to (x1, y1), half as wide as h
    const bridge = ([x0, y0], [x1, y1], h) => {
        const len = Math.hypot(x1 - x0, y1 - y0);
        if (len === 0)
            return;
        const ux = (x1 - x0) / len;
        const uy = (y1 - y0) / len;
        const reach = Math.ceil(h) + 1;
        for (let y = Math.max(0, Math.floor(Math.min(y0, y1)) - reach); y <= Math.min(height - 1, Math.ceil(Math.max(y0, y1)) + reach); y++) {
            for (let x = Math.max(0, Math.floor(Math.min(x0, x1)) - reach); x <= Math.min(width - 1, Math.ceil(Math.max(x0, x1)) + reach); x++) {
                const a = (x + 0.5 - x0) * ux + (y + 0.5 - y0) * uy;
                const b = -(x + 0.5 - x0) * uy + (y + 0.5 - y0) * ux;
                const p = y * width + x;
                if (a >= 0 && a <= len && Math.abs(b) <= h && deep[p])
                    thin[p] = 1;
            }
        }
    };
    // a thin mark on f's axis past the point at a (side: +1 or −1), within reach as a piece is, across the middle half
    // of f's thickness; the point reached, or null
    const meets = (f, a, side) => {
        const h = (f.b1 - f.b0) / 4;
        let run = 0;
        for (let s = 1; s <= DASH_REACH_GAPS * gap && run <= gap; s++) {
            const [x, y] = point(f, a + side * s);
            const p = at([x, y]);
            run = p >= 0 && deep[p] ? 0 : run + 1;
            let hits = 0;
            for (const t of [-h, 0, h])
                hits += thin[at([x - t * f.uy, y + t * f.ux])] ?? 0;
            if (hits === 3)
                return [x, y];
        }
        return null;
    };
    const used = new Set();
    for (const start of frames.filter(f => f.dash).sort((p, q) => q.n - p.n)) {
        if (used.has(start))
            continue;
        // each side's chain, outward: the pieces and, between each and the one before, the strip to bridge
        const chain = [start];
        const bridges = [];
        const ends = [];
        for (const side of [1, -1]) {
            let f = start;
            let prev = start;
            let edge = side > 0 ? f.a1 : f.a0;
            for (;;) {
                let best = null;
                for (const q of frames) {
                    if (used.has(q) || chain.includes(q))
                        continue;
                    if (Math.hypot(q.cx - f.cx, q.cy - f.cy) > Math.abs(edge) + DASH_REACH_GAPS * gap + Math.max(q.c.x1 - q.c.x0, q.c.y1 - q.c.y0) + 2)
                        continue;
                    const e = extent(f, q);
                    const near = side > 0 ? e.a0 - edge : edge - e.a1;
                    if (!e.inStrip || near > DASH_REACH_GAPS * gap || near < -1 || (side > 0 ? e.a1 <= edge : e.a0 >= edge))
                        continue;
                    if (near > gap && !crossable(f, edge, side, near))
                        continue;
                    if (!best || near < best.near)
                        best = { q, e, near };
                }
                if (!best) {
                    // the line goes on under a dot to a thin mark beyond it: bridged, unless the chain ends in a long
                    // piece (a thin mark itself: two with a dot between)
                    const hit = prev.long ? null : meets(f, edge, side);
                    if (hit)
                        bridges.push([point(f, edge), hit, (f.b1 - f.b0) / 2]);
                    ends.push(point(f, edge));
                    break;
                }
                const { q, e } = best;
                chain.push(q);
                bridges.push([point(f, edge), point(f, side > 0 ? e.a0 : e.a1), (f.b1 - f.b0) / 2, q.long && prev.long]);
                prev = q;
                const farEdge = side > 0 ? e.a1 : e.a0;
                if (q.dash && turn(f, q) <= DASH_ANGLE_TOL) {
                    // follow the new dash, its axis pointing the way the chain grows: the chain's edge in its frame
                    const [x, y] = point(f, farEdge);
                    const a = (x - 0.5 - q.cx) * q.ux + (y - 0.5 - q.cy) * q.uy;
                    const flip = f.ux * q.ux + f.uy * q.uy < 0;
                    f = flip ? { ...q, ux: -q.ux, uy: -q.uy, angle: q.angle + Math.PI, a0: -q.a1, a1: -q.a0, b0: -q.b1, b1: -q.b0 } : q;
                    edge = flip ? -a : a;
                }
                else {
                    edge = farEdge;
                }
            }
        }
        if (chain.length < 2 || Math.hypot(ends[0][0] - ends[1][0], ends[0][1] - ends[1][1]) < length)
            continue;
        for (const f of chain) {
            used.add(f);
            for (const p of f.c.pixels)
                thin[p] = 1;
        }
        for (const [a, b, h, solid] of bridges) {
            if (!solid) {
                bridge(a, b, h);
                continue;
            }
            const len = Math.hypot(b[0] - a[0], b[1] - a[1]);
            if (len === 0)
                continue;
            const reach = Math.min(DASH_STUB_RADII * radius, len / 2);
            const ux = (b[0] - a[0]) / len;
            const uy = (b[1] - a[1]) / len;
            bridge(a, [a[0] + reach * ux, a[1] + reach * uy], h);
            bridge([b[0] - reach * ux, b[1] - reach * uy], b, h);
        }
    }
}

/**
 * Connected components (8-neighbor) of a binary mask, in the order of their first pixels. shape.js components is
 * the 4-neighbor one, for the pieces of a coverage image.
 *
 * @param {ArrayLike<number>} mask  nonzero where set
 * @param {number} width
 * @param {number} height
 * @returns {{pixels: number[], x0: number, y0: number, x1: number, y1: number}[]}  each component's pixel indices
 *   and its bounding box, px, the last column and row included
 */
export function components8(mask, width, height) {
    const seen = new Uint8Array(mask.length);
    const comps = [];
    const stack = [];
    for (let s = 0; s < mask.length; s++) {
        if (!mask[s] || seen[s])
            continue;
        const comp = { pixels: [], x0: width, y0: height, x1: 0, y1: 0 };
        seen[s] = 1;
        stack.push(s);
        while (stack.length) {
            const p = stack.pop();
            comp.pixels.push(p);
            const x = p % width;
            const y = (p - x) / width;
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
                    if (mask[q] && !seen[q]) {
                        seen[q] = 1;
                        stack.push(q);
                    }
                }
            }
        }
        comps.push(comp);
    }
    return comps;
}

/**
 * Marks as thin the patches of an outlined family's fill color that meet the page bare, with no outline between (see
 * BARE_REACH): pieces of a line or curve drawn in the fill's color. The long ones are thin marks anyway; this takes
 * the stubs left between dots and where the curve leaves a dot, which read as dot and are hidden on all other sides
 * by other families' dots, so that the fit placed a dot under those to explain each (phase1: the dark gray curve
 * along its top row, 8 px of it showing between two dots of another gray). A patch is a connected piece of fill
 * pixels: a dot's own fill is one, closed off by its outline.
 *
 * Where the plot clips the dots, a cut dot's fill meets the page along the cut; the page beyond a cut doesn't count.
 *
 * @param {import('./image.js').RgbaImage} image
 * @param {Uint8Array} classes
 * @param {ArrayLike<number>} D  the family's coverage
 * @param {number[]} fill  0–255
 * @param {number[]} outline  0–255
 * @param {import('./axes.js').ClipEdge[]} cuts  the edges the dots are cut at, each with `cut`
 * @param {Uint8Array} thin  marked in place
 */
function bareFill(image, classes, D, fill, outline, cuts, thin) {
    const { width, height, data } = image;
    const n = width * height;
    const beyondCut = (x, y) => cuts.some(e => {
        const c = e.dir === 'v' ? x : y;
        return e.out > 0 ? c >= e.cut : c <= e.cut;
    });
    // the family's pixels nearer the fill's color than the outline's, along the line between the two
    const fo = sub(fill, outline);
    const half = dot(fo, fo) / 2;
    const isFill = new Uint8Array(n);
    const page = new Uint8Array(n);
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const p = y * width + x;
            if (classes[p] === CLASS_BACKGROUND)
                page[p] = beyondCut(x, y) ? 0 : 1;
            else if (classes[p] === CLASS_FAMILY && D[p] > 0.5)
                isFill[p] = dot([data[4 * p] - outline[0], data[4 * p + 1] - outline[1], data[4 * p + 2] - outline[2]], fo) >= half ? 1 : 0;
        }
    }
    const nearPage = dilate(page, width, height, BARE_REACH);
    for (const comp of components8(isFill, width, height)) {
        let bare = 0;
        let border = 0;
        for (const p of comp.pixels) {
            const x = p % width;
            if ((x > 0 && !isFill[p - 1]) || (x < width - 1 && !isFill[p + 1]) || (p >= width && !isFill[p - width]) || (p + width < n && !isFill[p + width]))
                border++;
            bare += nearPage[p];
        }
        if (bare < BARE_MIN_PIXELS || bare < BARE_MIN_SHARE * border)
            continue;
        for (const p of comp.pixels)
            thin[p] = 1;
    }
}

// Outlined dots (families.js outlineEstimate) bring their own mask: what can't be read in P, plus what lies beyond
// the axes. Pixels are classed as family where a ring may be (D > 0), other where masked.
// Where a family's coverage is cut at a clip edge (see CLIP_FULL): the cut's position (px, the first pixel row or
// column beyond it), or null when the dots aren't cut there. At a line the cut is the line; at a line's end, the
// column (or row) within CLIP_SEARCH of it with the most cut rows. O(extent · CLIP_SEARCH).
function clipAt(D, width, height, edge, radius) {
    const v = edge.dir === 'v';
    const alongMax = v ? height : width;
    const acrossMax = v ? width : height;
    const a0 = Math.max(0, Math.floor(edge.from ?? 0));
    const a1 = Math.min(alongMax, Math.ceil(edge.to ?? alongMax));
    const at = (a, c) => (v ? a * width + c : c * width + a);
    const count = (inside, beyond) => {
        if (inside < 0 || beyond < 0 || inside >= acrossMax || beyond >= acrossMax)
            return { full: 0, cut: 0 };
        let full = 0;
        let cut = 0;
        for (let a = a0; a < a1; a++) {
            if (!(D[at(a, inside)] >= CLIP_FULL))
                continue;
            full++;
            if (D[at(a, beyond)] <= CLIP_EMPTY)
                cut++;
        }
        return { full, cut };
    };
    let best = null;
    if (edge.thickness > 0) {
        const half = edge.thickness / 2 + CLIP_GAP;
        const inside = Math.floor(edge.pos - edge.out * half);
        const beyond = Math.floor(edge.pos + edge.out * half);
        // the line's pixels and beyond: from its near side outward
        const at = edge.out > 0 ? Math.round(edge.pos - edge.thickness / 2) : Math.round(edge.pos + edge.thickness / 2) - 1;
        best = { ...count(inside, beyond), at };
    }
    else {
        for (let c = Math.round(edge.pos) - CLIP_SEARCH; c <= Math.round(edge.pos) + CLIP_SEARCH; c++) {
            // the boundary pixel c may be antialiased: inside is the one before it, beyond the one after
            // (ties go outward: the dot's last full pixel and the first empty one both bound a cut)
            const k = count(c - edge.out, c + edge.out);
            if (!best || k.cut > best.cut || (k.cut === best.cut && (c - best.at) * edge.out > 0))
                best = { ...k, at: c };
        }
        const inked = c => {
            if (c < 0 || c >= acrossMax)
                return false;
            let rows = 0;
            for (let a = a0; a < a1; a++) {
                if (D[at(a, c)] > CLIP_EMPTY)
                    rows++;
            }
            return rows >= CLIP_BORDER_SHARE * (a1 - a0);
        };
        for (let k = 0; k < CLIP_SEARCH && inked(best.at - edge.out); k++)
            best.at -= edge.out;
    }
    return best.cut >= CLIP_MIN_RADII * radius && best.cut >= CLIP_MIN_SHARE * best.full ? best.at : null;
}

/**
 * Where the plot clips each family's dots: at an edge where the family's own coverage is cut (clipAt), or where the
 * coverage of all the families together is. A plot clips every series alike, and one family seldom has the two radii
 * of cut rows that clipAt asks of it: phase1's five families each have one dot cut in half where the x axis line
 * ends, 1.9 to 2.0 r of rows, so only one passed, and the other four were fitted on the colorbar beyond (27 false
 * dots, whose close pairs then had the fit raise their radii). Together they show 7.6 r, and 5 r at the y axis
 * line, where no family passed. The union, not a sum over families: ink in a color the families share is counted
 * once (engagement1: the top of a line of text in the stroke color read as 1.9 r of cut in each of four families).
 *
 * @param {{D: ArrayLike<number>|undefined, radius: number|undefined}[]} families  each one's coverage or count
 *   image and radius, px
 * @param {number} width
 * @param {number} height
 * @param {import('./axes.js').ClipEdge[]} edges  where a plot may clip (axes.clipEdges)
 * @returns {import('./axes.js').ClipEdge[][]}  per family, the edges its dots are clipped at, each with `cut`: the
 *   first pixel row or column beyond, px. None for a family without a radius.
 */
export function plotClips(families, width, height, edges) {
    const measured = families.filter(f => f.D && f.radius > 0);
    const union = new Float32Array(width * height);
    for (const f of measured) {
        for (let p = 0; p < union.length; p++) {
            if (f.D[p] > union[p])
                union[p] = f.D[p];
        }
    }
    const largest = Math.max(0, ...measured.map(f => f.radius));
    const together = edges.map(edge => (measured.length > 1 ? clipAt(union, width, height, edge, largest) : null));
    return families.map(f => {
        if (!measured.includes(f))
            return [];
        return edges.map((edge, i) => ({ ...edge, cut: clipAt(f.D, width, height, edge, f.radius) ?? together[i] })).filter(e => e.cut !== null);
    });
}

// mask every pixel at or beyond a cut (the first row or column beyond, from clipAt), in place; returns the count
function maskBeyond(mask, width, height, edge, cut) {
    let n = 0;
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const c = edge.dir === 'v' ? x : y;
            if (edge.out > 0 ? c >= cut : c <= cut) {
                n += 1 - mask[y * width + x];
                mask[y * width + x] = 1;
            }
        }
    }
    return n;
}

// set the pixels of each box [x0, y0, x1, y1) to 1, in place
function fillBoxes(mask, width, height, boxes) {
    for (const [x0, y0, x1, y1] of boxes ?? []) {
        for (let y = Math.max(0, Math.floor(y0)); y < Math.min(height, Math.ceil(y1)); y++) {
            for (let x = Math.max(0, Math.floor(x0)); x < Math.min(width, Math.ceil(x1)); x++)
                mask[y * width + x] = 1;
        }
    }
}

// Masks the pixels whose centers lie more than m px beyond the plot area's sides (a side that is null bounds
// nothing); returns how many.
function maskOutside(mask, width, height, b, m) {
    let outside = 0;
    for (let y = 0; y < height; y++) {
        const cy = y + 0.5;
        const rowOut = (b.top !== null && cy < b.top - m) || (b.bottom !== null && cy > b.bottom + m);
        for (let x = 0; x < width; x++) {
            const cx = x + 0.5;
            if (rowOut || (b.left !== null && cx < b.left - m) || (b.right !== null && cx > b.right + m)) {
                mask[y * width + x] = 1;
                outside++;
            }
        }
    }
    return outside;
}

function outlineLayers(estimate, width, height, opts) {
    const n = width * height;
    const { D, style } = estimate;
    const mask = Uint8Array.from(estimate.outlineMask);
    fillBoxes(mask, width, height, opts.exclude);
    const classes = new Uint8Array(n);
    for (let p = 0; p < n; p++)
        classes[p] = mask[p] ? CLASS_OTHER : (D[p] > 0 ? CLASS_FAMILY : CLASS_BACKGROUND);
    const outside = opts.bounds ? maskOutside(mask, width, height, opts.bounds, OUTSIDE_MARGIN_RADII * style.radius) : 0;
    const stats = { background: 0, family: 0, other: 0, thin: 0, masked: 0, maskedFamily: 0, outside };
    for (let p = 0; p < n; p++) {
        if (classes[p] === CLASS_BACKGROUND)
            stats.background++;
        else if (classes[p] === CLASS_FAMILY)
            stats.family++;
        else
            stats.other++;
        stats.masked += mask[p];
    }
    return { classes, thin: new Uint8Array(n), mask, dilate: 0, stats };
}

// Which of an opaque family's edgeInk values (layers.js buildMask) are inks of families drawn over it: a lookup by value,
// or null when there are none
function overInk(fm) {
    if (!fm?.edgeInk || !fm.under?.length)
        return null;
    const over = new Uint8Array(256);
    fm.inkColors.forEach((c, k) => {
        if (fm.under.some(u => colorDistance(u, c) <= OVER_INK_TOL))
            over[k + 1] = 1;
    });
    return over;
}

/**
 * Classify pixels and build the mask of pixels the fit should ignore.
 *
 * @param {import('./image.js').RgbaImage} image
 * @param {import('./style.js').Estimate} estimate  uses D, style and familyMeasure (and outlineMask, for a family
 *   fitted by its rings)
 * @param {object} [opts]
 * @param {number} [opts.dilate]  px, mask dilation (default MASK_DILATE_BASE + MASK_DILATE_BLURS × blur)
 * @param {{left: number|null, right: number|null, top: number|null, bottom: number|null}} [opts.bounds]  px, the
 *   plot area's sides (axes.plotBounds); pixels more than OUTSIDE_MARGIN_RADII × radius beyond them are masked
 * @param {number[][]} [opts.exclude]  boxes [x0, y0, x1, y1) px masked as other ink: text and legend markers
 *   (annotations.annotationBoxes)
 * @param {import('./axes.js').ClipEdge[]} [opts.clips]  lines where the plot may clip its dots (axes.clipEdges), or, with `cut`, lines
 *   where it does (plotClips, which also asks the other families); beyond one that cuts this
 *   family's coverage, everything is masked
 * @param {Uint8Array|null} [opts.strokes]  1 on the pixels of a stroke in the page's color around the dots, where it
 *   cuts the dots under them (stroke.js pageStroke): masked, with no margin
 * @returns {{classes: Uint8Array, thin: Uint8Array, mask: Uint8Array, dilate: number,
 *            stats: {background: number, family: number, other: number, thin: number, masked: number,
 *                    maskedFamily: number, outside: number, clipped?: import('./axes.js').ClipEdge[]}}}
 *   classes: CLASS_BACKGROUND, CLASS_FAMILY or CLASS_OTHER per pixel; thin: 1 on thin marks in the family's color;
 *   mask: 1 where the fit should not look; counts in pixels: maskedFamily is the family pixels the mask removes
 *   from the fit, outside those beyond the plot; clipped: the edges the dots are cut at, each with its cut
 */
export function buildMask(image, estimate, opts = {}) {
    const { width, height } = image;
    const n = width * height;
    const { D, style } = estimate;
    if (estimate.outlineMask)
        return outlineLayers(estimate, width, height, opts);
    const classes = new Uint8Array(n);
    for (let p = 0; p < n; p++) {
        const d = D[p];
        classes[p] = Number.isNaN(d) ? CLASS_OTHER : (d > 0 ? CLASS_FAMILY : CLASS_BACKGROUND);
    }
    const thin = new Uint8Array(n);
    const r = opts.dilate ?? MASK_DILATE_BASE + MASK_DILATE_BLURS * (style.blur ?? 0);
    // where the plot clips this family's dots: given (plotClips), or read from its own coverage
    const cuts = (opts.clips ?? []).map(edge => ({ ...edge, cut: edge.cut ?? (style.radius ? clipAt(D, width, height, edge, style.radius) : null) }))
        .filter(edge => edge.cut !== null);
    if (style.radius) {
        const deep = new Uint8Array(n);
        const limit = style.measuredLevels + DEEP_EXTRA_LEVELS;
        const beyond = (estimate.familyMeasure?.beyond ?? []).filter(L => colorDistance(L.color, style.color) <= ANCHOR_MAX_OFFSET);
        const anchor = beyond.length ? beyond.reduce((a, b) => (colorDistance(a.color, style.color) <= colorDistance(b.color, style.color) ? a : b)).color : style.color;
        const d = image.data;
        const opaque = style.alpha >= 1;
        // (… or a little past the full color on the family's line, where no stack of translucent dots gets: c is
        // extrapolated from the levels, and an error in them grows as 1 / α² on the way. richness: c, from two levels
        // at α 0.22, is 25 units short of the regression line's color, and nothing of the line was deep. Farther
        // past, it is another ink: bench2 gradient-dots-stroke's darker strokes around its dots, at α 0.74)
        const line = estimate.familyMeasure?.line;
        const along = p => (line.B[0] - d[4 * p]) * line.u[0] + (line.B[1] - d[4 * p + 1]) * line.u[1] + (line.B[2] - d[4 * p + 2]) * line.u[2];
        const past = Math.max(SATURATION_TOL, PAST_FULL / style.alpha ** 2);
        const full = line ? (line.B[0] - anchor[0]) * line.u[0] + (line.B[1] - anchor[1]) * line.u[1] + (line.B[2] - anchor[2]) * line.u[2] : Infinity;
        for (let p = 0; p < n; p++) {
            // opaque families: every pixel of the family's color is "deep" (Laser's bar outlines are drawn in the
            // dot colors, about 6 px wide against dots 11 px across)
            if (opaque) {
                deep[p] = classes[p] === CLASS_FAMILY && D[p] > 0.5 ? 1 : 0;
                continue;
            }
            if (classes[p] !== CLASS_FAMILY || D[p] < limit)
                continue;
            const e = Math.hypot(d[4 * p] - anchor[0], d[4 * p + 1] - anchor[1], d[4 * p + 2] - anchor[2]);
            deep[p] = e <= SATURATION_TOL || (line && along(p) >= full && along(p) <= full + past) ? 1 : 0;
        }
        // Other marks may hide the rest of a dot: open the family's pixels together with them, so a dot cut in two by
        // a line over it stays whole (synthetic swarm: half-discs along the median lines joined into "thin" marks).
        // Other ink next to the dots is read as their background (opaque.js measureOpaque's edgeInk), and counts as
        // other here: else its one-pixel seam cut a dot from the one over it, and the rest of the dot, joined to a
        // line of its color, was masked as thin (delta1: purple dots under black ones, and under black error bars)
        const ink = estimate.familyMeasure?.edgeInk;
        const withOther = new Uint8Array(n);
        for (let p = 0; p < n; p++)
            withOther[p] = deep[p] || classes[p] === CLASS_OTHER || ink?.[p] ? 1 : 0;
        const kept = opening(withOther, width, height, OPEN_RADIUS_FRACTION * style.radius);
        const raw = new Uint8Array(n);
        for (let p = 0; p < n; p++)
            raw[p] = deep[p] && !kept[p] ? 1 : 0;
        const candidate = opening(raw, width, height, SLIVER_OPENING);
        const short = [];
        const long = [];
        for (const comp of components8(candidate, width, height)) {
            if (Math.max(comp.x1 - comp.x0, comp.y1 - comp.y0) + 1 < THIN_MIN_LENGTH_RADII * style.radius) {
                short.push(comp);
                continue;
            }
            long.push(comp);
            for (const p of comp.pixels)
                thin[p] = 1;
        }
        // (2 px lines: not what the opening above kept, whose pieces end in corners it shaved. adults1: two pixels
        // more at each end of a masked line's pieces, where dots cross it, and two more extras)
        const narrow = blockOpening(raw, width, height);
        for (let p = 0; p < n; p++) {
            if (candidate[p])
                narrow[p] = 0;
        }
        markRuns(narrow, thin, width, height, THIN_MIN_LENGTH_RADII * style.radius);
        barPieces(short, deep, thin, width, height, style.radius);
        // the family's ink that the opening kept only with other ink beside it: a dash touching a dot of another
        // color (prot-cath-chris: the red curve between green and blue dots), a piece a dashed line may take in
        const own = opening(deep, width, height, OPEN_RADIUS_FRACTION * style.radius);
        const leaning = new Uint8Array(n);
        for (let p = 0; p < n; p++)
            leaning[p] = deep[p] && kept[p] && !own[p] ? 1 : 0;
        dashPieces(short, long, components8(opening(leaning, width, height, SLIVER_OPENING), width, height), deep, thin, width, height, style.radius);
        const fam = estimate.familyMeasure;
        if (opaque && fam?.outline && fam.fillRadius && fam.radius - fam.fillRadius >= BARE_MIN_OUTLINE)
            bareFill(image, classes, D, fam.seed, fam.outline, cuts, thin);
        if (!opaque && line) {
            // darker still beside a thin mark is its edge (richness: the regression line's first two rows, resampled,
            // are 10 and 22 units past its own color)
            const beside = dilate(thin, width, height, THIN_EDGE_REACH);
            for (let p = 0; p < n; p++) {
                if (beside[p] && classes[p] === CLASS_FAMILY && along(p) > full + past)
                    thin[p] = 1;
            }
        }
        if (!opaque) {
            // pixels masked anyway are unknown, not empty: the pale halos of dog-ages' curve and dashes broke the
            // dots beside them
            const hidden = new Uint8Array(n);
            for (let p = 0; p < n; p++)
                hidden[p] = classes[p] === CLASS_OTHER || thin[p] ? 1 : 0;
            const support = dilate(hidden, width, height, r);
            for (let p = 0; p < n; p++) {
                if (D[p] >= 0.5)
                    support[p] = 1;
            }
            const body = opening(support, width, height, SOLID_OPEN_FRACTION * style.radius);
            for (let p = 0; p < n; p++) {
                if (D[p] >= SOLID_COUNT && !body[p])
                    thin[p] = 1;
            }
        }
    }
    // Other ink next to the dots of a family drawn over this one may hide this one's dots: unknown, not background (a
    // dot partly under it paid for every pixel of that ink it covered, and the fit dropped it: delta1's purple under
    // black, 18 found for 24). Against a family drawn under this one it is this one's edge, and stays.
    const over = overInk(estimate.familyMeasure);
    const seedMask = new Uint8Array(n);
    for (let p = 0; p < n; p++)
        seedMask[p] = classes[p] === CLASS_OTHER || thin[p] || (over && over[estimate.familyMeasure.edgeInk[p]]) ? 1 : 0;
    fillBoxes(seedMask, width, height, opts.exclude);
    const mask = dilate(seedMask, width, height, r);
    // a stroke's pixels are the page's color where a dot may be: unknown, and only they (the ink on either side is
    // what shows of the dots)
    if (opts.strokes) {
        for (let p = 0; p < n; p++)
            mask[p] ||= opts.strokes[p];
    }
    let outside = opts.bounds && style.radius ? maskOutside(mask, width, height, opts.bounds, OUTSIDE_MARGIN_RADII * style.radius) : 0;
    for (const edge of cuts)
        outside += maskBeyond(mask, width, height, edge, edge.cut);
    const stats = { background: 0, family: 0, other: 0, thin: 0, masked: 0, maskedFamily: 0, outside, clipped: cuts };
    for (let p = 0; p < n; p++) {
        if (classes[p] === CLASS_BACKGROUND)
            stats.background++;
        else if (classes[p] === CLASS_FAMILY)
            stats.family++;
        else
            stats.other++;
        stats.thin += thin[p];
        stats.masked += mask[p];
        if (mask[p] && classes[p] === CLASS_FAMILY && !thin[p])
            stats.maskedFamily++;
    }
    return { classes, thin, mask, dilate: r, stats };
}

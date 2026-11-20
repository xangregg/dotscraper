// Layers and masks (PLAN.md §2.3, stages 2–3). Each pixel is background, explained by the dot family (on its
// color line, at any overlap count), or other. The fit ignores masked pixels: the "other" pixels, plus opaque thin
// marks in the family's own full color (the economist trend lines are the dots' c), dilated so antialiased fringes
// don't count either.
//
// Morphology uses the exact Euclidean distance transform, so dilation and opening by a disc cost O(pixels)
// whatever the radius.

export const CLASS_BACKGROUND = 0;
export const CLASS_FAMILY = 1;
export const CLASS_OTHER = 2;

// Opaque marks in the family's full color (the economist trend lines) are saturated: within a few units of c. For
// an opaque family every pixel of its color is a candidate, and the opening and length tests alone decide.
// Translucent dots get that close only in deep stacks (α 0.3: about 10 dots), and such piles are compact, while a
// line is long and narrower than a dot. A "deep" pixel is saturated and beyond the measured levels; deep pixels
// that an opening removes, in components long enough to be lines, are thin marks.
const SATURATION_TOL = 8; // 0–255 units from the saturated color
// The fitted c extrapolates from the levels (economist red: 2 levels, c off by 4.5 units in a synthetic copy); a
// plateau measured beyond the levels near c is the opaque mark's own color and a better anchor.
const ANCHOR_MAX_OFFSET = 20; // 0–255 units: a beyond-level this close to c anchors the saturation test
const DEEP_EXTRA_LEVELS = 2; // deep pixels are also at least this many levels beyond the deepest measured one
const OPEN_RADIUS_FRACTION = 0.75; // × dot radius: the opening removes marks narrower than 1.5 radii (economist
                                   // trend lines are ~9 px wide, r 8.4)
const THIN_MIN_LENGTH_RADII = 3; // × dot radius, the larger side of a thin component's bounding box
// The opening also shaves one-pixel slivers off jagged rims, and where opaque dots touch in a chain the slivers
// join into long components (synthetic swarm: 8 of 84 dots masked). Lines are at least 2 px wide: a second, 1 px
// opening of the candidates removes the slivers.
const SLIVER_OPENING = 1; // px
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
 */
export function opening(mask, width, height, r) {
    return dilate(erode(mask, width, height, r), width, height, r);
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

/** Connected components (8-neighbor) of a binary mask, with their bounding boxes. */
function components8(mask, width, height) {
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
    }
    return best.cut >= CLIP_MIN_RADII * radius && best.cut >= CLIP_MIN_SHARE * best.full ? best.at : null;
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

function outlineLayers(estimate, width, height, opts) {
    const n = width * height;
    const { D, style } = estimate;
    const mask = Uint8Array.from(estimate.outlineMask);
    fillBoxes(mask, width, height, opts.exclude);
    const classes = new Uint8Array(n);
    for (let p = 0; p < n; p++)
        classes[p] = mask[p] ? CLASS_OTHER : (D[p] > 0 ? CLASS_FAMILY : CLASS_BACKGROUND);
    let outside = 0;
    const b = opts.bounds;
    if (b) {
        const m = OUTSIDE_MARGIN_RADII * style.radius;
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
    }
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
        if (fm.under.some(u => Math.hypot(u[0] - c[0], u[1] - c[1], u[2] - c[2]) <= OVER_INK_TOL))
            over[k + 1] = 1;
    });
    return over;
}

/**
 * Classify pixels and build the mask of pixels the fit should ignore.
 *
 * @param {import('./image.js').RgbaImage} image
 * @param {object} estimate  from style.estimateStyle (uses D, style, background)
 * @param {object} [opts]
 * @param {number} [opts.dilate]  px, mask dilation (default MASK_DILATE_BASE + MASK_DILATE_BLURS × blur)
 * @param {{left: number|null, right: number|null, top: number|null, bottom: number|null}} [opts.bounds]  px, the
 *   plot area's sides (axes.plotBounds); pixels more than OUTSIDE_MARGIN_RADII × radius beyond them are masked
 * @param {number[][]} [opts.exclude]  boxes [x0, y0, x1, y1) px masked as other ink: text and legend markers
 *   (annotations.annotationBoxes)
 * @param {object[]} [opts.clips]  lines where the plot may clip its dots (axes.clipEdges); beyond one that cuts this
 *   family's coverage, everything is masked
 * @returns {{classes: Uint8Array, thin: Uint8Array, mask: Uint8Array, dilate: number,
 *            stats: {background: number, family: number, other: number, thin: number, masked: number,
 *                    maskedFamily: number, outside: number}}}
 *   counts in pixels; maskedFamily is the family pixels the mask removes from the fit
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
    if (style.radius) {
        const deep = new Uint8Array(n);
        const limit = style.measuredLevels + DEEP_EXTRA_LEVELS;
        const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
        const beyond = (estimate.familyMeasure?.beyond ?? []).filter(L => dist(L.color, style.color) <= ANCHOR_MAX_OFFSET);
        const anchor = beyond.length ? beyond.reduce((a, b) => (dist(a.color, style.color) <= dist(b.color, style.color) ? a : b)).color : style.color;
        const d = image.data;
        const opaque = style.alpha >= 1;
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
            deep[p] = e <= SATURATION_TOL ? 1 : 0;
        }
        // Other marks may hide the rest of a dot: open the family's pixels together with them, so a dot cut in two by
        // a line over it stays whole (synthetic swarm: half-discs along the median lines joined into "thin" marks).
        // Other ink next to the dots is read as their background (style.js measureOpaque's edgeInk), and counts as
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
        for (const comp of components8(candidate, width, height)) {
            if (Math.max(comp.x1 - comp.x0, comp.y1 - comp.y0) + 1 < THIN_MIN_LENGTH_RADII * style.radius) {
                short.push(comp);
                continue;
            }
            for (const p of comp.pixels)
                thin[p] = 1;
        }
        barPieces(short, deep, thin, width, height, style.radius);
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
    let outside = 0;
    const b = opts.bounds;
    if (b && style.radius) {
        const m = OUTSIDE_MARGIN_RADII * style.radius;
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
    }
    const clipped = [];
    for (const edge of opts.clips ?? []) {
        const cut = style.radius ? clipAt(D, width, height, edge, style.radius) : null;
        if (cut === null)
            continue;
        clipped.push({ ...edge, cut });
        outside += maskBeyond(mask, width, height, edge, cut);
    }
    const stats = { background: 0, family: 0, other: 0, thin: 0, masked: 0, maskedFamily: 0, outside, clipped };
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

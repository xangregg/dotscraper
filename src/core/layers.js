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
// A pixel at count ≥ 1.5 is inside at least one dot, so a disc of nearly the dot's radius around it lies in the
// family's pixels (or under other marks). Text in a translucent family's own color fails that: JMP's 50% black dots
// share their line with black text, whose antialiased strokes read as counts 1–8, too short for the length test,
// and the fit put 15 dots on the tick labels and title.
const SOLID_COUNT = 1.5; // overlap count
const SOLID_OPEN_FRACTION = 0.9; // × dot radius: the disc that must fit
const MASK_DILATE_BASE = 1; // px
const MASK_DILATE_BLURS = 2; // px per px of edge blur
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

/**
 * Classify pixels and build the mask of pixels the fit should ignore.
 *
 * @param {import('./image.js').RgbaImage} image
 * @param {object} estimate  from style.estimateStyle (uses D, style, background)
 * @param {object} [opts]
 * @param {number} [opts.dilate]  px, mask dilation (default MASK_DILATE_BASE + MASK_DILATE_BLURS × blur)
 * @returns {{classes: Uint8Array, thin: Uint8Array, mask: Uint8Array, dilate: number,
 *            stats: {background: number, family: number, other: number, thin: number, masked: number,
 *                    maskedFamily: number}}}
 *   counts in pixels; maskedFamily is the family pixels the mask removes from the fit
 */
export function buildMask(image, estimate, opts = {}) {
    const { width, height } = image;
    const n = width * height;
    const { D, style } = estimate;
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
        // a line over it stays whole (synthetic swarm: half-discs along the median lines joined into "thin" marks)
        const withOther = new Uint8Array(n);
        for (let p = 0; p < n; p++)
            withOther[p] = deep[p] || classes[p] === CLASS_OTHER ? 1 : 0;
        const kept = opening(withOther, width, height, OPEN_RADIUS_FRACTION * style.radius);
        const raw = new Uint8Array(n);
        for (let p = 0; p < n; p++)
            raw[p] = deep[p] && !kept[p] ? 1 : 0;
        const candidate = opening(raw, width, height, SLIVER_OPENING);
        for (const comp of components8(candidate, width, height)) {
            if (Math.max(comp.x1 - comp.x0, comp.y1 - comp.y0) + 1 < THIN_MIN_LENGTH_RADII * style.radius)
                continue;
            for (const p of comp.pixels)
                thin[p] = 1;
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
    const seedMask = new Uint8Array(n);
    for (let p = 0; p < n; p++)
        seedMask[p] = classes[p] === CLASS_OTHER || thin[p] ? 1 : 0;
    const mask = dilate(seedMask, width, height, r);
    const stats = { background: 0, family: 0, other: 0, thin: 0, masked: 0, maskedFamily: 0 };
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

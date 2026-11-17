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
// the outline's radius.

const WINDOW = 2; // px: half side of the window whose summed offsets give a pixel's direction
const MIN_OFFSET = 15; // 0–255 units from B: flat pixels at least this far vote for the one-dot curve
const CURVE_CELL = 0.02; // direction cell (unit-vector components)
const RIM_REACH = 4; // px: flat pixels this close to a background pixel are the outer, one-dot part of a dot
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

// The points' largest chain by votes (LINK_DIST, LINK_MAX_RATIO), and how far apart its ends are. Points of other
// inks (the title's black strokes) form chains of their own.
function largestChain(points) {
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
                if (label[j] >= 0 || Math.hypot(a.u[0] - b.u[0], a.u[1] - b.u[1], a.u[2] - b.u[2]) > LINK_DIST
                    || Math.max(a.m, b.m) > LINK_MAX_RATIO * Math.min(a.m, b.m))
                    continue;
                label[j] = chains.length;
                members.push(j);
            }
        }
        chains.push(members.map(i => points[i]));
    }
    const votes = c => c.reduce((sum, q) => sum + q.votes, 0);
    const best = chains.reduce((a, b) => (votes(b) > votes(a) ? b : a));
    let spread = 0;
    for (const a of best) {
        for (const b of best)
            spread = Math.max(spread, Math.hypot(a.u[0] - b.u[0], a.u[1] - b.u[1], a.u[2] - b.u[2]));
    }
    return { points: best, spread };
}

/**
 * α from the count ratios of flat pixels on the curve: the second histogram peak sits at 2 − α.
 *
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
 * @param {import('./style.js').Tolerances} tol
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
    const reach = nearReach(background, width, height);
    const dir = directionField(image, B);
    const cells = new Map();
    for (let p = 0; p < n; p++) {
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
    let points = [];
    for (const c of cells.values()) {
        if (c.ms.length < CELL_MIN_VOTES)
            continue;
        c.ms.sort((a, b) => a - b);
        const l = Math.hypot(...c.sum);
        points.push({ u: c.sum.map(v => v / l), m: c.ms[c.ms.length >> 1], votes: c.ms.length });
    }
    points = points.filter((q, i) => {
        const near = points.filter((o, j) => j !== i && Math.hypot(o.u[0] - q.u[0], o.u[1] - q.u[1], o.u[2] - q.u[2]) < NEIGHBOR_DIST)
            .map(o => o.m).sort((a, b) => a - b);
        return !near.length || q.m <= MAX_NEIGHBOR_RATIO * near[near.length >> 1];
    });
    if (!points.length)
        return null;
    const scale = largestChain(points);
    points = scale.points;
    const spread = scale.spread;
    const votes = points.reduce((t, q) => t + q.votes, 0);
    const top = points.map(q => q.votes).sort((a, b) => b - a).slice(0, FEW_POINTS).reduce((t, v) => t + v, 0);
    if (points.length < MIN_POINTS || spread < MIN_SPREAD || Math.max(...points.map(q => q.votes)) > MAX_POINT_SHARE * votes
        || top > MAX_FEW_SHARE * votes)
        return null;
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
    return { points, alpha, spread, dir };
}

// 1 where a background pixel lies within RIM_REACH (a square neighborhood; cheap and close enough)
function nearReach(background, width, height) {
    const out = new Uint8Array(background.length);
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            if (!background[y * width + x])
                continue;
            for (let yy = Math.max(0, y - RIM_REACH); yy <= Math.min(height - 1, y + RIM_REACH); yy++) {
                for (let xx = Math.max(0, x - RIM_REACH); xx <= Math.min(width - 1, x + RIM_REACH); xx++)
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
 * farthest pixel nearby in its direction. Every other dot pixel, a fill, an outline, or the boundary between two
 * dots, is covered. The result is a one-color rendering of the dots' union.
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
    const dots = dotPixels(m, cm, width, height);
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
            const a = lightest(m, width, height, x, y) <= LIGHT * full ? m[p] / full : 1;
            for (let ch = 0; ch < 3; ch++)
                out[4 * p + ch] = Math.round(B[ch] + a * (P[ch] - B[ch]));
        }
    }
    return { image: { width, height, data: out }, P };
}

// 1 for the pixels of opaque dots: those on the curve, and those near it within OUTLINE_REACH of one. Outlines of
// dark fills lie as far from the curve as black text does (bench2 gradient-opaque-stroke: a dark red outline
// (109, 47, 45) is 0.15 off, and 0.16 from black), but only outlines touch the fills. O(pixels · reach²).
function dotPixels(m, cm, width, height) {
    const { dir } = cm;
    const n = width * height;
    const dist = new Float32Array(n).fill(Infinity);
    const on = new Uint8Array(n);
    for (let p = 0; p < n; p++) {
        if (m[p] < MIN_PIXEL_OFFSET)
            continue;
        dist[p] = oneDot(cm.points, dir[3 * p], dir[3 * p + 1], dir[3 * p + 2]).dist;
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
    return out;
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
 * @param {object} measured  style.measureFamily on the virtual image, seeded with P
 * @param {number[]} B
 * @param {number[]} P
 * @param {number} alpha
 * @returns {object}
 */
export function colormapCandidate(measured, B, P, alpha) {
    const level = r => ({ t: r * Math.hypot(...P.map((v, ch) => v - B[ch])), color: B.map((v, ch) => v + r * (P[ch] - v)), count: 0 });
    const c = B.map((v, ch) => v + (P[ch] - v) / alpha);
    return {
        ...measured,
        levels: [level(1), level(2 - alpha)],
        fit: { alpha, c, rms: 0, maxAbs: 0, underdetermined: false },
        kind: 'translucent', translucent: true,
    };
}

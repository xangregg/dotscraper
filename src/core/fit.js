// The fit (PLAN.md §2.4, stage 5): find the set of dot centers whose rendering best reproduces the image.
//
// Objective: Σ_p w_p ρ(t_p − t̂_p) + λ·N. t is a pixel's distance from the background along the family's color line
// (all a translucent family's information lies on it; off-line pixels are masked), t̂ the same for the rendered
// model, w_p = 0 on the mask, ρ the Huber loss (rims of resampled and canvas-drawn dots misfit by tens of units in
// single pixels whatever the position, so they must not dominate), and λ the cost of a dot: DOT_COST_PIXELS pixels
// each DOT_COST_STEP units wrong. λ is absolute, not a fraction of the first level step: in a stack n deep one
// more dot changes the color only (1 − α)^n as much as the first does (α 0.3, n 6: 0.12), and such a dot must
// still be able to pay for itself.
//
// Steps: candidates from a Hough vote with the known radius (each visible edge arc votes for its own center, which
// finds dots inside clusters where blob detection fails); then rounds of compass-search refinement of each center,
// births where the residual says a level is missing (coincident dots), deaths and merges where they lower the
// objective, and a global radius refit. Everything is incremental: a change re-renders and re-scores only the
// patch it touches.

import { createDotModel, renderDots, solidImage, gaussianKernel } from './render.js';
import { createOutlineModel } from './outline.js';

const HUBER_DELTA = 3; // t units (0–255 scale): residuals beyond this count linearly
const DOT_COST_PIXELS = 20; // λ = this many pixels …
const DOT_COST_STEP = 10; // … each this many t units wrong
const COMPASS_STEPS = [2, 1, 0.5, 0.25, 0.125, 0.0625]; // px
const FIRST_SWEEP_STEP = 1; // index into COMPASS_STEPS where the first sweep starts (1 px: Hough centers are
                            // within about a pixel); births start at 0, from a box center
const RESWEEP_FIRST_STEP = 2; // index into COMPASS_STEPS where later sweeps start (0.5 px)
const NEIGHBOR_RADII = 2.2; // × r: a dot this close to one that moved is refined again
const MAX_SWEEPS = 8; // refinement sweeps per round
const CONVERGED_MOVE = 0.05; // px: a sweep moving no dot farther than this ends the round's refinement
export const MAX_ROUNDS = 6; // rounds of refine, birth, death, merge
const HOUGH_GRAD_MIN = 0.08; // levels per px: weaker gradients don't vote
const HOUGH_SMOOTH = 1; // px, Gaussian smoothing of the accumulator before peak finding
const HOUGH_PEAK_RADIUS = 2.5; // px: votes within this of a peak are its mass
const HOUGH_MIN_ARC = 0.2; // fraction of the circumference a peak's mass must reach
const HOUGH_NMS_RADII = 0.3; // × r: of two peaks closer than this, the smaller is dropped
// Outline-only dots (ringCenters): the ring's mid-radius circle is sampled at this many points, and a center needs
// this mean count on it: a ring a quarter seen for births, on the residual; the first candidates need more, since
// in a dense stack the arcs of neighbors add up to spurious centers (pathogen at 0.25: 1125 candidates for 583
// dots, 668 of them later removed, and 80 s of a 106 s fit spent on them)
const RING_SAMPLES = 48;
const RING_MIN_ARC = 0.15;
const RING_MIN_ARC_FIRST = 0.5;
// An outlined dot's own fill value (fitFill) is read this far inside its fill, from at least this many pixels it is
// on top of, and changed only by more than this
const FILL_FIT_INSET = 1.5; // px
const FILL_FIT_MIN_PIXELS = 8;
const FILL_FIT_TOL = 1; // t units
const BIRTH_MIN_LEVELS = 0.2; // mean missing dots over a dot-sized box to try a birth there
// Missing dots are counted per pixel in units of what one more dot would change there; where that is below this
// (very deep stacks, where the colors no longer separate) the pixel can't show a missing dot.
const BIRTH_MIN_STEP = 3; // t units
const BIRTH_BOX_RADII = 0.7; // × r, half side of the box (inscribed in the disc)
const BIRTH_MIN_SUPPORT = 0.3; // fraction of the box that must be unmasked
// A dot whose middle is under a line shows only its rims, outside the inscribed box: a second pass uses a box the
// size of the dot's bounding square, with less support required.
const BIRTH_WIDE_BOX_RADII = 1;
const BIRTH_WIDE_MIN_SUPPORT = 0.15;
const BIRTH_TABU_RADII = 0.5; // × r: failed birth sites are not retried within this
// While a dot is missing its neighbors drift to cover part of its area, so a birth is judged after they re-settle
// (dog-ages: adding a missing dot alone raised the loss by 13 000; with its neighbors refined it fell by 44 000).
// Their moves are limited so a snapshot of the neighborhood can undo a rejected birth.
const BIRTH_NEIGHBOR_MOVE_RADII = 0.5; // × r, max distance a neighbor may move while a birth is judged
const MAX_BIRTHS_PER_ROUND = 200;
// An opaque dot less visible than this is supported by too little of itself to report: faint antialiased rims of
// another family's dots made 2–12% visible phantoms (synthetic swarm). Not applied to translucent families, whose
// rims are read along their own color line: dots under the curve preset's black line, under 15% visible but
// correctly found from their rims, were lost when it was.
const MIN_VISIBLE_OPAQUE = 0.15;
// An outlined dot needs this much of itself unmasked: a ring under a mask can't be told from no ring, and dots
// beside pathogen's y-axis ticks, 3–10% unmasked, explained the ticks' gray fringes
const MIN_VISIBLE_OUTLINE = 0.3;
// An opaque dot's visible interior must be filled: text in a dot family's color (Laser: black labels, "0", "O",
// "S") explained enough of a glyph's ring for a dot to pay for itself, but glyphs are hollow.
const FILL_RADII = 0.75; // × r, the interior checked
const MIN_FILL_OPAQUE = 0.85; // share of its visible pixels at least FILL_LEVEL of the dot's color
// a dot's interior reads near 1; another family's dot that is a blend of this one's color with the background
// reads part way (nicest language: pink is red at 0.47 on beige, and red dots were fitted on pink ones)
const FILL_LEVEL = 0.75;
// The check needs this many visible interior pixels; a share of the interior was too lenient (nicest language:
// phantom gray dots on its antialiased title were 22–34% visible and skipped the check)
const FILL_MIN_SEEN = 8; // px
const MERGE_RADII = 0.35; // × r: dots closer than this are tried as one
// A dot is tried as two when the loss over its patch is at least this many λ and this many times the median dot's:
// kept splits were 4.4–12 times the median (bench2 solid-opaque: 3.1–15.5 λ against 0.7; violin, a JPEG whose every
// patch carries noise: 59–107 λ against 13), and trying every patch over λ made violin's fit 5 times slower
const SPLIT_MIN_LOSS_LAMBDAS = 1;
const SPLIT_MIN_LOSS_MEDIANS = 4;
const SPLIT_TABU_RADII = 0.25; // × r: a dot that failed to split isn't tried again until it moves this far
const SPLIT_OFFSET_RADII = 0.4; // × r: each half starts this far from the dot
const SPLIT_DIRECTIONS = 4; // directions tried, over half a turn
const RADIUS_SEARCH = 1; // px either side of the estimated radius
const RADIUS_TOL = 0.01; // px
const UNCERTAINTY_STEP = 0.25; // px, finite-difference step for the curvature
const FLAG_MARGIN = 3; // flag a dot whose removal costs less than this × λ
const FLAG_VISIBLE = 0.5; // flag a dot less than this fraction visible
const PROGRESS_INTERVAL = 150; // ms between progress reports

function huber(r) {
    const a = Math.abs(r);
    return a <= HUBER_DELTA ? 0.5 * r * r : HUBER_DELTA * (a - 0.5 * HUBER_DELTA);
}

/**
 * Hough vote for dot centers with a known radius on the overlap count image. Each unmasked pixel with a gradient
 * votes, with weight |∇D|, at the point r along the gradient (D rises into a dot, so the gradient points to its
 * center). Across one dot edge |∇D| integrates to about 1 per unit of arc, so a fully visible dot collects about
 * 2πr; a peak's mass divided by that is the fraction of its circumference seen.
 *
 * @param {Float32Array} D  overlap count, NaN off the color line
 * @param {Uint8Array} weight  1 where the fit may look
 * @param {number} width
 * @param {number} height
 * @param {number} r  px
 * @returns {{x: number, y: number, arc: number}[]}  centers in px, strongest first
 */
export function houghCenters(D, weight, width, height, r) {
    const n = width * height;
    const acc = new Float32Array(n);
    for (let y = 1; y < height - 1; y++) {
        for (let x = 1; x < width - 1; x++) {
            const p = y * width + x;
            let ok = true;
            for (let dy = -1; dy <= 1 && ok; dy++) {
                for (let dx = -1; dx <= 1 && ok; dx++) {
                    const q = p + dy * width + dx;
                    if (!weight[q] || Number.isNaN(D[q]))
                        ok = false;
                }
            }
            if (!ok)
                continue;
            // Sobel, scaled to levels per px
            const gx = (D[p - width + 1] + 2 * D[p + 1] + D[p + width + 1] - D[p - width - 1] - 2 * D[p - 1] - D[p + width - 1]) / 8;
            const gy = (D[p + width - 1] + 2 * D[p + width] + D[p + width + 1] - D[p - width - 1] - 2 * D[p - width] - D[p - width + 1]) / 8;
            const g = Math.hypot(gx, gy);
            if (g < HOUGH_GRAD_MIN)
                continue;
            // vote in pixel-index coordinates (center of pixel i is i + 0.5), bilinear
            const fx = x + r * gx / g;
            const fy = y + r * gy / g;
            const ix = Math.floor(fx);
            const iy = Math.floor(fy);
            const tx = fx - ix;
            const ty = fy - iy;
            for (const [ox, oy, w] of [[0, 0, (1 - tx) * (1 - ty)], [1, 0, tx * (1 - ty)], [0, 1, (1 - tx) * ty], [1, 1, tx * ty]]) {
                const xx = ix + ox;
                const yy = iy + oy;
                if (xx >= 0 && yy >= 0 && xx < width && yy < height)
                    acc[yy * width + xx] += g * w;
            }
        }
    }
    const smooth = blurPlane(acc, width, height, HOUGH_SMOOTH);
    const nms = Math.max(2, HOUGH_NMS_RADII * r);
    const pr = Math.ceil(HOUGH_PEAK_RADIUS);
    const minMass = HOUGH_MIN_ARC * 2 * Math.PI * r;
    const peaks = [];
    for (let y = 1; y < height - 1; y++) {
        for (let x = 1; x < width - 1; x++) {
            const p = y * width + x;
            const v = smooth[p];
            if (v <= 0 || v < smooth[p - 1] || v < smooth[p + 1] || v < smooth[p - width] || v < smooth[p + width])
                continue;
            let mass = 0;
            let sx = 0;
            let sy = 0;
            for (let dy = -pr; dy <= pr; dy++) {
                for (let dx = -pr; dx <= pr; dx++) {
                    const xx = x + dx;
                    const yy = y + dy;
                    if (xx < 0 || yy < 0 || xx >= width || yy >= height || dx * dx + dy * dy > HOUGH_PEAK_RADIUS ** 2)
                        continue;
                    const a = acc[yy * width + xx];
                    mass += a;
                    sx += a * (xx + 0.5);
                    sy += a * (yy + 0.5);
                }
            }
            if (mass >= minMass)
                peaks.push({ x: sx / mass, y: sy / mass, mass, v });
        }
    }
    peaks.sort((a, b) => b.v - a.v);
    const kept = [];
    for (const pk of peaks) {
        if (kept.every(k => Math.hypot(k.x - pk.x, k.y - pk.y) >= nms))
            kept.push(pk);
    }
    return kept.map(k => ({ x: k.x, y: k.y, arc: k.mass / (2 * Math.PI * r) }));
}

/**
 * Candidate centers for outline-only dots: a matched filter on the overlap count. A ring is too thin for the
 * gradient vote (the Sobel kernel spans both of its edges, which cancel), so each center is scored by the mean
 * count on the circle at the ring's mid-radius, sampled at RING_SAMPLES points: about 1 where a whole ring is seen,
 * the fraction seen where part of it is hidden. Masked samples count as 0. O(pixels · RING_SAMPLES).
 *
 * @param {Float32Array} D  ring count (NaN where unknown)
 * @param {Uint8Array} weight  1 where the fit may look
 * @param {number} width
 * @param {number} height
 * @param {number} r  px, the ring's outer radius
 * @param {number} inner  the ring's inner radius over r
 * @param {number} [minArc]  the mean count a center needs
 * @returns {{x: number, y: number, arc: number}[]}  centers in px, strongest first; arc: the mean count
 */
export function ringCenters(D, weight, width, height, r, inner, minArc = RING_MIN_ARC) {
    const rm = r * (1 + inner) / 2;
    const offs = [];
    for (let k = 0; k < RING_SAMPLES; k++) {
        const a = 2 * Math.PI * k / RING_SAMPLES;
        offs.push([rm * Math.cos(a), rm * Math.sin(a)]);
    }
    // bilinear sample of D at (x, y) in px, 0 off the image or where masked
    const at = (x, y) => {
        const fx = x - 0.5;
        const fy = y - 0.5;
        const ix = Math.floor(fx);
        const iy = Math.floor(fy);
        const tx = fx - ix;
        const ty = fy - iy;
        let s = 0;
        for (const [ox, oy, w] of [[0, 0, (1 - tx) * (1 - ty)], [1, 0, tx * (1 - ty)], [0, 1, (1 - tx) * ty], [1, 1, tx * ty]]) {
            const xx = ix + ox;
            const yy = iy + oy;
            if (xx < 0 || yy < 0 || xx >= width || yy >= height)
                continue;
            const q = yy * width + xx;
            if (weight[q] && !Number.isNaN(D[q]))
                s += w * D[q];
        }
        return s;
    };
    const score = new Float32Array(width * height);
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            let s = 0;
            for (const [ox, oy] of offs)
                s += at(x + 0.5 + ox, y + 0.5 + oy);
            score[y * width + x] = s / RING_SAMPLES;
        }
    }
    const nms = Math.max(2, HOUGH_NMS_RADII * r);
    const peaks = [];
    for (let y = 1; y < height - 1; y++) {
        for (let x = 1; x < width - 1; x++) {
            const p = y * width + x;
            const v = score[p];
            if (v < minArc || v < score[p - 1] || v < score[p + 1] || v < score[p - width] || v < score[p + width])
                continue;
            // subpixel: score-weighted centroid of the 3 × 3 neighborhood above its smallest value
            let low = Infinity;
            for (let dy = -1; dy <= 1; dy++) {
                for (let dx = -1; dx <= 1; dx++)
                    low = Math.min(low, score[p + dy * width + dx]);
            }
            let sw = 0;
            let sx = 0;
            let sy = 0;
            for (let dy = -1; dy <= 1; dy++) {
                for (let dx = -1; dx <= 1; dx++) {
                    const w = score[p + dy * width + dx] - low;
                    sw += w;
                    sx += w * (x + dx + 0.5);
                    sy += w * (y + dy + 0.5);
                }
            }
            peaks.push(sw > 0 ? { x: sx / sw, y: sy / sw, v } : { x: x + 0.5, y: y + 0.5, v });
        }
    }
    peaks.sort((a, b) => b.v - a.v);
    const kept = [];
    for (const pk of peaks) {
        if (kept.every(k => Math.hypot(k.x - pk.x, k.y - pk.y) >= nms))
            kept.push(pk);
    }
    return kept.map(k => ({ x: k.x, y: k.y, arc: k.v }));
}

function blurPlane(src, width, height, sigma) {
    const k = gaussianKernel(sigma);
    const h = (k.length - 1) / 2;
    const tmp = new Float32Array(src.length);
    const out = new Float32Array(src.length);
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            let s = 0;
            for (let i = -h; i <= h; i++)
                s += k[i + h] * src[y * width + Math.min(width - 1, Math.max(0, x + i))];
            tmp[y * width + x] = s;
        }
    }
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            let s = 0;
            for (let i = -h; i <= h; i++)
                s += k[i + h] * tmp[Math.min(height - 1, Math.max(0, y + i)) * width + x];
            out[y * width + x] = s;
        }
    }
    return out;
}

/**
 * A fit in progress: the image along the color line, the weights, the incremental model, and the dots.
 *
 * @param {object} input
 * @param {import('./image.js').RgbaImage} input.image
 * @param {object} input.style   render style with levels (from style.estimateStyle, overrides applied)
 * @param {{B: number[], u: number[]}} input.line  the family's color line
 * @param {Uint8Array} input.mask  1 = ignore
 * @param {Float32Array} [input.base]  t per pixel under the dots, for outlined dots (style.layered: outline.js)
 * @param {Float32Array} [input.imgT]  the image's values to compare, if not its projection on the line (opaque
 *                                     families: coverage against the local background, in color units)
 */
export function createFit(input) {
    const { image, line, mask } = input;
    const { width, height } = image;
    const n = width * height;
    const { B, u } = line;
    const imgT = input.imgT ? Float32Array.from(input.imgT) : new Float32Array(n);
    const weight = new Uint8Array(n);
    for (let p = 0; p < n; p++) {
        if (!input.imgT)
            imgT[p] = (B[0] - image.data[4 * p]) * u[0] + (B[1] - image.data[4 * p + 1]) * u[1] + (B[2] - image.data[4 * p + 2]) * u[2];
        weight[p] = mask[p] ? 0 : 1;
    }
    const background = solidImage(width, height, B);
    let style = input.style;
    const lambda = DOT_COST_PIXELS * huber(DOT_COST_STEP);
    // the fit renders along the color line only (one value per pixel); modelImage() renders RGB for display
    const lineStyle = st => ({ ...st, project: line });
    // outlined dots (style.layered): an ordered model over the per-pixel base input.base, in t units
    const makeModel = st => (st.layered
        ? createOutlineModel(input.base, width, height, { radius: st.radius, ring: st.ring, fillAlpha: st.layered.fillAlpha, ringT: st.layered.ringT, blur: st.blur })
        : createDotModel(background, lineStyle(st)));
    let model = makeModel(style);
    const opaque = style.alpha >= 1;
    const minVisible = opaque ? MIN_VISIBLE_OPAQUE : (style.layered ? MIN_VISIBLE_OUTLINE : 0);
    const fillT = FILL_LEVEL * Math.hypot(B[0] - style.color[0], B[1] - style.color[1], B[2] - style.color[2]);
    const pos = new Map(); // id → { x, y }

    function modelT(p) {
        return model.image.data[p];
    }

    function loss(rect) {
        const [x0, y0, x1, y1] = rect;
        let s = 0;
        for (let y = y0; y < y1; y++) {
            for (let x = x0; x < x1; x++) {
                const p = y * width + x;
                if (weight[p])
                    s += huber(imgT[p] - modelT(p));
            }
        }
        return s;
    }

    const union = (a, b) => [Math.min(a[0], b[0]), Math.min(a[1], b[1]), Math.max(a[2], b[2]), Math.max(a[3], b[3])];
    const clampX = x => Math.min(width + style.radius / 2, Math.max(-style.radius / 2, x));
    const clampY = y => Math.min(height + style.radius / 2, Math.max(-style.radius / 2, y));

    // z: drawing order, for outlined dots only (default: on top)
    function add(x, y, z) {
        const id = model.add(x, y, z);
        pos.set(id, { x, y });
        return id;
    }

    function remove(id) {
        model.remove(id);
        pos.delete(id);
    }

    // move if it lowers the loss; returns whether it moved
    function tryMove(id, x, y) {
        const p = pos.get(id);
        x = clampX(x);
        y = clampY(y);
        const foot = union(model.footprint(p.x, p.y), model.footprint(x, y));
        const rect = union(model.influenceRect(p.x, p.y), model.influenceRect(x, y));
        const before = loss(rect);
        const snap = model.snapshot(foot);
        model.move(id, x, y);
        if (loss(rect) < before - 1e-9) {
            pos.set(id, { x, y });
            return true;
        }
        model.restore(snap);
        return false;
    }

    // loss over a dot's patch without it, the dot left in place (a snapshot undoes the removal)
    function lossWithout(id) {
        const p = pos.get(id);
        const rect = model.influenceRect(p.x, p.y);
        const snap = model.snapshot(model.footprint(p.x, p.y));
        model.remove(id);
        const without = loss(rect);
        model.restore(snap);
        return { without, withDot: loss(rect) };
    }

    // compass search on one center, from COMPASS_STEPS[first], never farther than maxMove from where it started;
    // returns the distance moved
    function refineDot(id, first = 0, maxMove = Infinity) {
        const start = { ...pos.get(id) };
        for (const step of COMPASS_STEPS.slice(first)) {
            let improved = true;
            while (improved) {
                improved = false;
                const { x, y } = pos.get(id);
                for (const [dx, dy] of [[step, 0], [-step, 0], [0, step], [0, -step]]) {
                    if (Math.hypot(x + dx - start.x, y + dy - start.y) > maxMove)
                        continue;
                    if (tryMove(id, x + dx, y + dy)) {
                        improved = true;
                        break;
                    }
                }
            }
        }
        const end = pos.get(id);
        return Math.hypot(end.x - start.x, end.y - start.y);
    }

    // Sweeps over the active dots: all at first, then only those that moved or have a neighbor that moved
    // (a dot's best position depends only on the dots overlapping it). Later sweeps start at a smaller step.
    function refineAll(ids = [...pos.keys()]) {
        let active = new Set(ids);
        for (let sweep = 0; sweep < MAX_SWEEPS; sweep++) {
            const moved = [];
            for (const id of active) {
                if (pos.has(id) && refineDot(id, sweep === 0 ? FIRST_SWEEP_STEP : RESWEEP_FIRST_STEP) >= CONVERGED_MOVE)
                    moved.push(pos.get(id));
            }
            if (!moved.length)
                return sweep + 1;
            active = new Set();
            for (const [id, p] of pos) {
                if (moved.some(m => Math.hypot(m.x - p.x, m.y - p.y) <= NEIGHBOR_RADII * style.radius))
                    active.add(id);
            }
        }
        return MAX_SWEEPS;
    }

    // share of the dot's disc the fit can see (unmasked)
    function visibleFraction(x, y) {
        const r = style.radius;
        let inDisc = 0;
        let seen = 0;
        for (let yy = Math.max(0, Math.floor(y - r)); yy < Math.min(height, Math.ceil(y + r)); yy++) {
            for (let xx = Math.max(0, Math.floor(x - r)); xx < Math.min(width, Math.ceil(x + r)); xx++) {
                if ((xx + 0.5 - x) ** 2 + (yy + 0.5 - y) ** 2 <= r * r) {
                    inDisc++;
                    seen += weight[yy * width + xx];
                }
            }
        }
        return inDisc ? seen / inDisc : 0;
    }

    // opaque dots only: is the visible interior mostly the dot's color? (true when too little of it is visible)
    function filled(x, y) {
        if (!opaque)
            return true;
        const rf = FILL_RADII * style.radius;
        let seen = 0;
        let on = 0;
        for (let yy = Math.max(0, Math.floor(y - rf)); yy < Math.min(height, Math.ceil(y + rf)); yy++) {
            for (let xx = Math.max(0, Math.floor(x - rf)); xx < Math.min(width, Math.ceil(x + rf)); xx++) {
                if ((xx + 0.5 - x) ** 2 + (yy + 0.5 - y) ** 2 > rf * rf)
                    continue;
                const p = yy * width + xx;
                if (!weight[p])
                    continue;
                seen++;
                if (imgT[p] >= fillT)
                    on++;
            }
        }
        return seen < FILL_MIN_SEEN || on >= MIN_FILL_OPAQUE * seen;
    }

    function near(x, y, d) {
        return [...pos].filter(([, p]) => Math.hypot(p.x - x, p.y - y) <= d).map(([id]) => id);
    }

    // ΔL of removing a dot (positive: the dot explains that much)
    function removalGain(id) {
        const { without, withDot } = lossWithout(id);
        return without - withDot;
    }

    // dots whose removal costs less than λ go, and dots too hidden to report
    function deaths() {
        let removed = 0;
        for (const id of [...pos.keys()]) {
            const p = pos.get(id);
            if (visibleFraction(p.x, p.y) < minVisible || !filled(p.x, p.y) || removalGain(id) < lambda) {
                remove(id);
                removed++;
            }
        }
        return removed;
    }

    // pairs closer than MERGE_RADII·r tried as one dot
    function merges() {
        let merged = 0;
        const ids = [...pos.keys()];
        for (let i = 0; i < ids.length; i++) {
            for (let j = i + 1; j < ids.length; j++) {
                const a = pos.get(ids[i]);
                const b = pos.get(ids[j]);
                if (!a || !b || Math.hypot(a.x - b.x, a.y - b.y) >= MERGE_RADII * style.radius)
                    continue;
                const mid = { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
                // the merged dot can wander a few px while refining: cover that in the snapshot and the score
                const grow = r => [Math.max(0, r[0] - 3), Math.max(0, r[1] - 3), Math.min(width, r[2] + 3), Math.min(height, r[3] + 3)];
                const foot = grow(union(model.footprint(a.x, a.y), model.footprint(b.x, b.y)));
                const rect = grow(union(model.influenceRect(a.x, a.y), model.influenceRect(b.x, b.y)));
                const before = loss(rect) + 2 * lambda;
                const snap = model.snapshot(foot);
                const posA = pos.get(ids[i]);
                const posB = pos.get(ids[j]);
                model.remove(ids[i]);
                model.remove(ids[j]);
                pos.delete(ids[i]);
                pos.delete(ids[j]);
                const m = add(mid.x, mid.y);
                refineDot(m);
                const pm = pos.get(m);
                const inside = pm.x - style.radius >= foot[0] && pm.y - style.radius >= foot[1] && pm.x + style.radius < foot[2] && pm.y + style.radius < foot[3];
                if (inside && loss(rect) + lambda < before) {
                    merged++;
                    continue;
                }
                model.restore(snap);
                pos.delete(m);
                pos.set(ids[i], posA);
                pos.set(ids[j], posB);
            }
        }
        return merged;
    }

    // Opaque dots of one color hide the edges between them, so a cluster can be covered by fewer dots than it has, each
    // stretched over two (bench2 solid-opaque: 180 dots with data loss 89 700, where the truth's 204 give 8 800). Their
    // union then matches the image to a few pixels, and births, which look for missing ink, find no site. A split
    // replaces a poorly fitting dot by two, SPLIT_OFFSET_RADII either side of it in one of SPLIT_DIRECTIONS, lets the
    // neighbors re-settle as a birth does, and keeps the best variant if the neighborhood's loss falls by more than λ.
    // Translucent families don't need it: a dot standing in for two leaves a missing level, which births see.
    function splits(tabu) {
        if (!opaque)
            return 0;
        const r = style.radius;
        const slack = BIRTH_NEIGHBOR_MOVE_RADII * r;
        const grow = ([x0, y0, x1, y1], d) => [Math.max(0, x0 - d), Math.max(0, y0 - d), Math.min(width, x1 + d), Math.min(height, y1 + d)];
        let made = 0;
        const patches = [...pos.keys()].map(id => ({ id, l: loss(model.influenceRect(pos.get(id).x, pos.get(id).y)) }));
        const ls = patches.map(c => c.l).sort((a, b) => a - b);
        const minLoss = Math.max(SPLIT_MIN_LOSS_LAMBDAS * lambda, SPLIT_MIN_LOSS_MEDIANS * (ls[ls.length >> 1] ?? 0));
        const candidates = patches.filter(c => c.l > minLoss).sort((a, b) => b.l - a.l);
        for (const { id } of candidates) {
            const p0 = pos.get(id);
            if (!p0 || tabu.some(t => Math.hypot(t.x - p0.x, t.y - p0.y) < SPLIT_TABU_RADII * r))
                continue;
            const neighbors = near(p0.x, p0.y, NEIGHBOR_RADII * r).filter(nid => nid !== id);
            let foot = model.footprint(p0.x, p0.y);
            for (const nid of neighbors) {
                const q0 = pos.get(nid);
                foot = union(foot, model.footprint(q0.x, q0.y));
            }
            foot = grow(foot, Math.ceil(SPLIT_OFFSET_RADII * r + slack + COMPASS_STEPS[0] + 1));
            const scoreRect = grow(foot, Math.ceil(style.blur ? 3 * style.blur + 1 : 0));
            const before = loss(scoreRect);
            const snap = model.snapshot(foot);
            const saved = [[id, p0], ...neighbors.map(nid => [nid, pos.get(nid)])];
            const undo = () => {
                model.restore(snap);
                for (const [nid] of saved)
                    pos.delete(nid);
                for (const k of [...pos.keys()]) {
                    if (k > lastId)
                        pos.delete(k);
                }
                for (const [nid, q] of saved)
                    pos.set(nid, q);
            };
            const lastId = Math.max(...pos.keys());
            let best = null;
            for (let k = 0; k < SPLIT_DIRECTIONS; k++) {
                const a = Math.PI * k / SPLIT_DIRECTIONS;
                const dx = SPLIT_OFFSET_RADII * r * Math.cos(a);
                const dy = SPLIT_OFFSET_RADII * r * Math.sin(a);
                remove(id);
                const ids = [add(p0.x + dx, p0.y + dy), add(p0.x - dx, p0.y - dy)];
                for (const nid of ids)
                    refineDot(nid, RESWEEP_FIRST_STEP, slack);
                for (const nid of neighbors)
                    refineDot(nid, RESWEEP_FIRST_STEP, slack);
                const gain = before - loss(scoreRect);
                // each half must pay for itself, as deaths require: otherwise the "split" only moved the dot (Laser: a
                // dot half under a bar split into one dot and one on the bar, and a later death took the real one)
                const ok = gain > lambda && ids.every(nid => visibleFraction(pos.get(nid).x, pos.get(nid).y) >= minVisible
                    && filled(pos.get(nid).x, pos.get(nid).y) && removalGain(nid) > lambda);
                if (ok && (!best || gain > best.gain))
                    best = { k, gain };
                undo();
            }
            if (!best) {
                tabu.push({ x: p0.x, y: p0.y });
                continue;
            }
            // redo the best variant (the trials were undone so that each started from the same neighborhood)
            const a = Math.PI * best.k / SPLIT_DIRECTIONS;
            const dx = SPLIT_OFFSET_RADII * r * Math.cos(a);
            const dy = SPLIT_OFFSET_RADII * r * Math.sin(a);
            remove(id);
            const ids = [add(p0.x + dx, p0.y + dy), add(p0.x - dx, p0.y - dy)];
            for (const nid of ids)
                refineDot(nid, RESWEEP_FIRST_STEP, slack);
            for (const nid of neighbors)
                refineDot(nid, RESWEEP_FIRST_STEP, slack);
            made++;
        }
        return made;
    }

    const grow = ([x0, y0, x1, y1], d) => [Math.max(0, x0 - d), Math.max(0, y0 - d), Math.min(width, x1 + d), Math.min(height, y1 + d)];

    // Add a dot at (x, y), refine it, let its neighbors re-settle, and keep it if the neighborhood's loss falls by
    // more than λ; otherwise restore the neighborhood. Returns the rectangle scored when kept, else null.
    function tryBirth(x, y, z) {
        const r = style.radius;
        const slack = BIRTH_NEIGHBOR_MOVE_RADII * r;
        const neighbors = near(x, y, NEIGHBOR_RADII * r);
        let foot = model.footprint(x, y);
        for (const nid of neighbors) {
            const q0 = pos.get(nid);
            foot = union(foot, model.footprint(q0.x, q0.y));
        }
        foot = grow(foot, Math.ceil(slack + COMPASS_STEPS[0] + 1));
        const scoreRect = grow(foot, Math.ceil(style.blur ? 3 * style.blur + 1 : 0));
        const before = loss(scoreRect);
        const snap = model.snapshot(foot);
        const saved = neighbors.map(nid => [nid, pos.get(nid)]);
        const id = add(x, y, z);
        refineDot(id, 0, slack + COMPASS_STEPS[0]);
        for (const nid of neighbors)
            refineDot(nid, RESWEEP_FIRST_STEP, slack);
        refineDot(id, RESWEEP_FIRST_STEP, slack);
        if (style.layered && fitFill(id))
            refineDot(id, RESWEEP_FIRST_STEP, slack);
        const born = pos.get(id);
        if (before - loss(scoreRect) > lambda && visibleFraction(born.x, born.y) >= minVisible && filled(born.x, born.y))
            return scoreRect;
        model.restore(snap);
        pos.delete(id);
        for (const [nid, p0] of saved)
            pos.set(nid, p0);
        birthFailures++;
        return null;
    }

    // the neighborhood changed: earlier failures in it may succeed now
    function forgetTabu(tabu, rect) {
        for (let i = tabu.length - 1; i >= 0; i--) {
            if (rect[0] <= tabu[i].x && tabu[i].x < rect[2] && rect[1] <= tabu[i].y && tabu[i].y < rect[3])
                tabu.splice(i, 1);
        }
    }

    // The drawing order for a new outlined dot at (x, y): each gap in its neighbors' order (below all, between two,
    // above all) is scored with the dot added as it stands, and the best is returned. O(k) renders of its patch for
    // k neighbors.
    function bestZ(x, y) {
        const zs = near(x, y, 2 * style.radius).map(id => model.zOf(id)).sort((a, b) => a - b);
        if (!zs.length)
            return undefined;
        const gaps = [zs[0] - 1, ...zs.slice(1).map((z, i) => (zs[i] + z) / 2), zs[zs.length - 1] + 1];
        const rect = model.influenceRect(x, y);
        const foot = model.footprint(x, y);
        let best = null;
        for (const z of gaps) {
            const snap = model.snapshot(foot);
            model.add(x, y, z);
            const L = loss(rect);
            model.restore(snap);
            if (!best || L < best.L)
                best = { z, L };
        }
        return best.z;
    }

    // An outlined dot's own fill value from the part of its interior it is on top of, where the image is
    // (1 − αf)·(what is under it) + αf·f: f = f0 + (t − t̂) / αf, the median over those pixels. Interior:
    // FILL_FIT_INSET px inside the fill, clear of every ring. Returns whether f changed.
    function fitFill(id) {
        const p = pos.get(id);
        const z = model.zOf(id);
        const aF = style.layered.fillAlpha;
        const f0 = model.fillOf(id);
        const rIn = model.fillRadius - FILL_FIT_INSET;
        const above = near(p.x, p.y, 2 * style.radius).filter(o => o !== id && model.zOf(o) > z).map(o => pos.get(o));
        const vals = [];
        for (let y = Math.max(0, Math.floor(p.y - rIn)); y < Math.min(height, Math.ceil(p.y + rIn)); y++) {
            for (let x = Math.max(0, Math.floor(p.x - rIn)); x < Math.min(width, Math.ceil(p.x + rIn)); x++) {
                const cx = x + 0.5;
                const cy = y + 0.5;
                const q = y * width + x;
                if (!weight[q] || (cx - p.x) ** 2 + (cy - p.y) ** 2 > rIn * rIn)
                    continue;
                if (above.some(o => Math.hypot(cx - o.x, cy - o.y) < style.radius + FILL_FIT_INSET))
                    continue;
                // the model's value here is (1 − αf)·under + αf·f0, so f moves it by αf·(f − f0); no division by
                // 1 − αf, which is 0 for opaque fills
                vals.push(f0 + (imgT[q] - modelT(q)) / aF);
            }
        }
        if (vals.length < FILL_FIT_MIN_PIXELS)
            return false;
        vals.sort((a, b) => a - b);
        const f = vals[vals.length >> 1];
        if (Math.abs(f - f0) < FILL_FIT_TOL)
            return false;
        model.setFill(id, f);
        return true;
    }

    function fitFills() {
        let changed = 0;
        for (const id of [...pos.keys()])
            changed += fitFill(id) ? 1 : 0;
        return changed;
    }

    // Births for outline-only dots (style.alpha 0 with a ring): a missing ring leaves nothing at its center for a
    // box to find, only its own arc in the residual, so candidates come from ringCenters on the residual counted
    // in rings, strongest first. O(pixels · RING_SAMPLES) per round.
    function ringBirths(tabu) {
        const one = model.valueOf(1 - style.ring.alpha);
        const Dres = new Float32Array(n);
        for (let p = 0; p < n; p++)
            Dres[p] = weight[p] ? Math.max(0, imgT[p] - modelT(p)) / one : NaN;
        let added = 0;
        for (const c of ringCenters(Dres, weight, width, height, style.radius, style.ring.inner).slice(0, MAX_BIRTHS_PER_ROUND)) {
            if (tabu.some(t => Math.hypot(t.x - c.x, t.y - c.y) < BIRTH_TABU_RADII * style.radius))
                continue;
            // outlined dots: a missing ring may lie anywhere in its neighbors' order
            const scoreRect = tryBirth(c.x, c.y, style.layered ? bestZ(c.x, c.y) : undefined);
            if (scoreRect) {
                added++;
                forgetTabu(tabu, scoreRect);
            }
            else {
                tabu.push({ x: c.x, y: c.y });
            }
        }
        return added;
    }

    // Missing dots, (t − t̂)/Δt on unmasked pixels with Δt what one more dot would add at the pixel's current depth,
    // averaged over boxes inscribed in a dot; the best box not near a failed site is tried as a new dot, kept if it
    // pays for itself. The residual and box-mean planes are updated only where an accepted birth changed the model.
    function births(tabu) {
        if (style.alpha === 0 && style.ring)
            return ringBirths(tabu);
        return birthsWithBox(tabu, BIRTH_BOX_RADII, BIRTH_MIN_SUPPORT) + birthsWithBox(tabu, BIRTH_WIDE_BOX_RADII, BIRTH_WIDE_MIN_SUPPORT);
    }

    function birthsWithBox(tabu, boxRadii, minSupport) {
        const h = Math.max(1, Math.round(boxRadii * style.radius));
        const q = 1 - style.alpha;
        const area = (2 * h + 1) ** 2;
        const R = new Float32Array(n);
        const S = new Uint8Array(n);
        const M = new Float32Array(n).fill(-Infinity);

        function fillResidual([x0, y0, x1, y1]) {
            for (let y = y0; y < y1; y++) {
                for (let x = x0; x < x1; x++) {
                    const p = y * width + x;
                    R[p] = 0;
                    S[p] = 0;
                    if (!weight[p])
                        continue;
                    const Tp = model.transmittance[p];
                    const dt = model.valueOf(Tp * q) - model.valueOf(Tp);
                    if (dt >= BIRTH_MIN_STEP) {
                        R[p] = (imgT[p] - modelT(p)) / dt;
                        S[p] = 1;
                    }
                }
            }
        }

        // box means for centers in the rectangle, from a local integral image
        function fillMeans([x0, y0, x1, y1]) {
            const cx0 = Math.max(h, x0);
            const cy0 = Math.max(h, y0);
            const cx1 = Math.min(width - h, x1);
            const cy1 = Math.min(height - h, y1);
            if (cx0 >= cx1 || cy0 >= cy1)
                return;
            const ix0 = cx0 - h;
            const iy0 = cy0 - h;
            const iw = cx1 + h - ix0 + 1;
            const ih = cy1 + h - iy0 + 1;
            const sumR = new Float64Array(iw * ih);
            const sumS = new Float64Array(iw * ih);
            for (let y = 1; y < ih; y++) {
                let rowR = 0;
                let rowS = 0;
                for (let x = 1; x < iw; x++) {
                    const p = (iy0 + y - 1) * width + ix0 + x - 1;
                    rowR += R[p];
                    rowS += S[p];
                    sumR[y * iw + x] = sumR[(y - 1) * iw + x] + rowR;
                    sumS[y * iw + x] = sumS[(y - 1) * iw + x] + rowS;
                }
            }
            const box = (A, x, y) => {
                const a = x - h - ix0;
                const b = y - h - iy0;
                const c = a + 2 * h + 1;
                const d = b + 2 * h + 1;
                return A[d * iw + c] - A[b * iw + c] - A[d * iw + a] + A[b * iw + a];
            };
            for (let y = cy0; y < cy1; y++) {
                for (let x = cx0; x < cx1; x++) {
                    const w = box(sumS, x, y);
                    M[y * width + x] = w >= minSupport * area ? box(sumR, x, y) / w : -Infinity;
                }
            }
        }

        fillResidual([0, 0, width, height]);
        fillMeans([0, 0, width, height]);
        let added = 0;
        for (let attempt = 0; attempt < MAX_BIRTHS_PER_ROUND; attempt++) {
            let best = null;
            for (let p = 0; p < n; p++) {
                const v = M[p];
                if (v < BIRTH_MIN_LEVELS || (best !== null && v <= best.mean))
                    continue;
                const cx = p % width + 0.5;
                const cy = Math.floor(p / width) + 0.5;
                if (tabu.some(t => Math.hypot(t.x - cx, t.y - cy) < BIRTH_TABU_RADII * style.radius))
                    continue;
                best = { x: cx, y: cy, mean: v };
            }
            if (!best)
                break;
            const scoreRect = tryBirth(best.x, best.y);
            if (scoreRect) {
                added++;
                fillResidual(scoreRect);
                fillMeans(grow(scoreRect, h));
                forgetTabu(tabu, scoreRect);
            }
            else {
                tabu.push({ x: best.x, y: best.y });
            }
        }
        return added;
    }
    let birthFailures = 0;

    // golden-section search of the radius on the whole loss, dots fixed
    function refitRadius() {
        const r0 = style.radius;
        const dots = style.layered ? model.dots() : [...pos.values()];
        const evalR = r => {
            const m = makeModel({ ...style, radius: r });
            m.addAll(dots);
            const saved = model;
            model = m;
            const L = loss([0, 0, width, height]);
            model = saved;
            return L;
        };
        let lo = Math.max(0.5, r0 - RADIUS_SEARCH);
        let hi = r0 + RADIUS_SEARCH;
        const g = (Math.sqrt(5) - 1) / 2;
        let a = hi - g * (hi - lo);
        let b = lo + g * (hi - lo);
        let fa = evalR(a);
        let fb = evalR(b);
        while (hi - lo > RADIUS_TOL) {
            if (fa < fb) {
                hi = b;
                b = a;
                fb = fa;
                a = hi - g * (hi - lo);
                fa = evalR(a);
            }
            else {
                lo = a;
                a = b;
                fa = fb;
                b = lo + g * (hi - lo);
                fb = evalR(b);
            }
        }
        const r = (lo + hi) / 2;
        setStyle({ ...style, radius: r });
        return r;
    }

    function setStyle(s) {
        const dots = style.layered ? model.dots() : [...pos.values()];
        style = s;
        model = makeModel(style);
        pos.clear();
        for (const d of dots) {
            const id = add(d.x, d.y, d.z);
            if (style.layered)
                model.setFill(id, d.f);
        }
    }

    function totalLoss() {
        return loss([0, 0, width, height]);
    }

    // robust residual sd over unmasked pixels the model covers (inside dots)
    function noiseSigma() {
        const res = [];
        const T = model.transmittance;
        for (let p = 0; p < n; p++) {
            if (weight[p] && T[p] < 0.999)
                res.push(Math.abs(imgT[p] - modelT(p)));
        }
        if (!res.length)
            return NaN;
        res.sort((a, b) => a - b);
        return 1.4826 * res[Math.floor(res.length / 2)];
    }

    /**
     * Per-dot uncertainty and flags: position standard errors from the loss curvature (σ / √H per axis), the
     * removal margin in units of λ, the visible fraction of the disc, and the model's overlap count at the center.
     */
    function assess() {
        const sigma = noiseSigma();
        const q = 1 - style.alpha;
        const out = [];
        for (const id of [...pos.keys()]) {
            const p = pos.get(id);
            const rect = union(model.influenceRect(p.x - UNCERTAINTY_STEP, p.y - UNCERTAINTY_STEP), model.influenceRect(p.x + UNCERTAINTY_STEP, p.y + UNCERTAINTY_STEP));
            const L0 = loss(rect);
            const curv = [];
            const foot = union(model.footprint(p.x - UNCERTAINTY_STEP, p.y - UNCERTAINTY_STEP), model.footprint(p.x + UNCERTAINTY_STEP, p.y + UNCERTAINTY_STEP));
            for (const [dx, dy] of [[1, 0], [0, 1]]) {
                const snap = model.snapshot(foot);
                model.move(id, p.x + dx * UNCERTAINTY_STEP, p.y + dy * UNCERTAINTY_STEP);
                const Lp = loss(rect);
                model.move(id, p.x - dx * UNCERTAINTY_STEP, p.y - dy * UNCERTAINTY_STEP);
                const Lm = loss(rect);
                model.restore(snap);
                curv.push((Lp + Lm - 2 * L0) / UNCERTAINTY_STEP ** 2);
            }
            const visible = visibleFraction(p.x, p.y);
            const cp = Math.min(height - 1, Math.max(0, Math.floor(p.y))) * width + Math.min(width - 1, Math.max(0, Math.floor(p.x)));
            const T = model.transmittance[cp];
            const depth = q > 0 && T > 0 ? Math.log(T) / Math.log(q) : 1;
            const margin = removalGain(id) / lambda;
            const flags = [];
            if (margin < FLAG_MARGIN)
                flags.push('weak');
            if (visible < FLAG_VISIBLE)
                flags.push('hidden');
            if (depth > style.measuredLevels + 0.5)
                flags.push('deep');
            out.push({
                x: p.x, y: p.y,
                seX: curv[0] > 0 ? sigma / Math.sqrt(curv[0]) : Infinity,
                seY: curv[1] > 0 ? sigma / Math.sqrt(curv[1]) : Infinity,
                margin, visible, depth: Math.round(depth * 10) / 10, flags,
                ...(style.layered ? { z: model.zOf(id) } : {}),
            });
        }
        out.sort((a, b) => a.x - b.x || a.y - b.y);
        return { dots: out, sigma };
    }

    return {
        get style() {
            return style;
        },
        lambda, weight, imgT,
        add, remove, refineAll, refineDot, births, splits, deaths, merges, fitFills, refitRadius, totalLoss, assess, bestZ,
        get birthFailures() {
            return birthFailures;
        },
        modelImage: () => renderDots(background, [...pos.values()], style),
        dots: () => [...pos.values()].map(p => ({ x: p.x, y: p.y })),
        ids: () => [...pos.keys()],
        position: id => pos.get(id),
        /** ids of dots within distance d of (x, y) */
        near,
    };
}

/**
 * Run the whole fit.
 *
 * @param {object} input  as createFit, plus D (overlap count image)
 * @param {object} [opts]
 * @param {boolean} [opts.refitRadius]  default true
 * @param {{x: number, y: number}[]} [opts.initial]  start from these centers instead of the Hough candidates
 * @param {function(object): void} [onProgress]  { stage, round, dots, loss }
 * @returns {object}  { dots (assessed), radius, sigma, lambda, loss, rounds, log, ms }
 */
export function fitDots(input, opts = {}, onProgress = () => {}) {
    if (!(input.style.radius > 0))
        throw new Error('The dot radius is unknown: set it before fitting.');
    const t0 = Date.now();
    const fit = createFit(input);
    const { width, height } = input.image;
    const log = [];
    let lastReport = 0;
    const report = (stage, round, force = false) => {
        const now = Date.now();
        if (!force && now - lastReport < PROGRESS_INTERVAL)
            return;
        lastReport = now;
        onProgress({ stage, round, dots: fit.dots(), loss: fit.totalLoss() + fit.lambda * fit.ids().length });
    };
    const st = input.style;
    const ringOnly = st.alpha === 0 && st.ring;
    const initial = opts.initial ?? (ringOnly ? ringCenters(input.D, fit.weight, width, height, st.radius, st.ring.inner, RING_MIN_ARC_FIRST)
        : houghCenters(input.D, fit.weight, width, height, st.radius));
    for (const c of initial)
        fit.add(c.x, c.y);
    log.push({ stage: 'candidates', dots: initial.length });
    report('candidates', 0, true);
    let tabu = [];
    const splitTabu = []; // kept across rounds: a split that failed fails again unless the dot moves
    let rounds = 0;
    for (let round = 1; round <= MAX_ROUNDS; round++) {
        rounds = round;
        const tr = Date.now();
        let sweeps = fit.refineAll();
        // outlined dots: each fill's own value, then the positions it frees
        if (st.layered && fit.fitFills())
            sweeps += fit.refineAll();
        const msRefine = Date.now() - tr;
        report('refine', round);
        // a site that failed may succeed once its neighbors have moved: forget failures each round
        tabu = [];
        const tb = Date.now();
        const born = fit.births(tabu);
        const split = fit.splits(splitTabu);
        if (born || split)
            fit.refineAll();
        const msBirths = Date.now() - tb;
        report('births', round);
        const td = Date.now();
        const died = fit.deaths();
        const merged = fit.merges();
        log.push({ stage: 'round', round, sweeps, born, split, failed: fit.birthFailures, died, merged, dots: fit.ids().length, msRefine, msBirths, msDeathsMerges: Date.now() - td });
        report('round', round, true);
        if (!born && !split && !died && !merged)
            break;
    }
    if (opts.refitRadius ?? true) {
        const tr = Date.now();
        const r = fit.refitRadius();
        fit.refineAll();
        tabu = [];
        const born = fit.births(tabu);
        const died = fit.deaths();
        if (born || died)
            fit.refineAll();
        log.push({ stage: 'radius', radius: r, born, died, dots: fit.ids().length, ms: Date.now() - tr });
        report('radius', rounds, true);
    }
    const ta = Date.now();
    const { dots, sigma } = fit.assess();
    log.push({ stage: 'assess', ms: Date.now() - ta });
    return {
        dots, radius: fit.style.radius, sigma, lambda: fit.lambda,
        loss: fit.totalLoss(), rounds, log, ms: Date.now() - t0,
    };
}

/**
 * Re-refine the dots near (x, y) after a manual edit, leaving the rest where they are, and assess them all.
 * The dot set itself is kept as given: no births or deaths, since the user just decided what is there.
 *
 * Outlined dots keep their drawing order (z); one marked `added` is placed where in its neighbors' order it fits
 * best (bestZ), after the others are in.
 *
 * @param {object} input  as createFit
 * @param {{x: number, y: number, z?: number, added?: boolean}[]} dots  px
 * @param {number} x  px, the edit
 * @param {number} y  px
 * @returns {{dots: object[], sigma: number, lambda: number}}
 */
export function refitNear(input, dots, x, y) {
    const fit = createFit(input);
    for (const d of dots.filter(d => !d.added))
        fit.add(d.x, d.y, d.z);
    for (const d of dots.filter(d => d.added))
        fit.add(d.x, d.y, input.style.layered ? fit.bestZ(d.x, d.y) : undefined);
    fit.refineAll(fit.near(x, y, NEIGHBOR_RADII * input.style.radius));
    const { dots: assessed, sigma } = fit.assess();
    return { dots: assessed, sigma, lambda: fit.lambda };
}

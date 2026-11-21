// Synthetic graphs with known dots (PLAN.md §2.6): ground truth at any difficulty for tests and the benchmark.
// Dots use the forward model in render.js, so a synthetic image can't by itself reveal a model error; tools/
// render-check compares the renderer with Chrome's canvas for that.
//
// A graph may be rendered at a lower native resolution and upsampled (bilinear) to its final size, which is how
// dog-ages looks (a figure enlarged by a non-integer factor). All coordinates in a spec and in the truth are final
// pixels.

import { createRng } from './rng.js';
import { createDotModel, solidImage, toRgba, gaussianKernel, blurRect } from './render.js';

/**
 * @typedef {object} FamilySpec
 * @property {number[]} color   full color c, [r, g, b]
 * @property {number} alpha
 * @property {number} radius    px
 * @property {{x: number, y: number}[]} points
 */

/**
 * @typedef {object} LineSpec
 * @property {number[][]} points  polyline vertices [[x, y], …], px
 * @property {number} width       px
 * @property {number[]} color
 * @property {number} [alpha]     default 1
 * @property {number[]} [dash]    [on, off] lengths along the line, px
 */

/**
 * @typedef {object} AxisSpec
 * @property {number} pos        px: y of an x axis, x of a y axis (line center)
 * @property {number} from       px, start of the line along the axis
 * @property {number} to         px, end
 * @property {number[]} ticks    px positions along the axis
 */

/**
 * @typedef {object} AxesSpec
 * @property {AxisSpec} [x]
 * @property {AxisSpec} [y]
 * @property {number} [width]       px, line and tick stroke width (default 2)
 * @property {number} [tickLength]  px, outward (below an x axis, left of a y axis; default 8)
 * @property {boolean} [inward]     ticks drawn into the plot instead, labels still outside (Nature's frames)
 * @property {number[]} [color]     default [0, 0, 0]
 * @property {number[]} [label]     [w, h] px of a solid block standing in for each tick's label (default [14, 10]);
 *                                  centered on the tick, LABEL_GAP px beyond its end (beyond the line if inward)
 */

/**
 * @typedef {object} GraphSpec
 * @property {number} width
 * @property {number} height
 * @property {number[]} background
 * @property {FamilySpec[]} families  composited in order, each over the result of the previous ones
 * @property {LineSpec[]} [lines]     drawn over all dots
 * @property {LineSpec[]} [gridlines] drawn under the dots
 * @property {AxesSpec} [axes]        axis lines, ticks and label blocks, drawn over everything
 * @property {number} [scale]         native resolution is 1/scale of the final size (default 1)
 * @property {number} [blur]          px, Gaussian sd applied after upsampling (default 0)
 * @property {number} [noise]         sd of Gaussian noise added before rounding, 0–255 units (default 0)
 * @property {number} [seed]          for the noise (default 1)
 */

/**
 * Points for a scatterplot, in final pixels.
 *
 * Layouts: 'uniform' over the box; 'curve', x uniform and y = f(u) of the box height plus Gaussian scatter, like a
 * regression scatterplot; 'clusters', Gaussian clusters at uniform centers. A fraction `ties` of the points is then
 * replaced by copies of other points, displaced by up to `tieJitter` px, for coincident and near-coincident dots.
 *
 * @param {import('./rng.js').Rng} rng
 * @param {object} opts
 * @param {number} opts.n
 * @param {number[]} opts.box               [x0, y0, x1, y1], px
 * @param {string} [opts.layout]            'uniform' (default), 'curve', 'clusters' or 'beeswarm'
 * @param {function(number): number} [opts.curve]  u in [0, 1] → height fraction in [0, 1], 1 at the top
 * @param {number} [opts.curveSd]           scatter about the curve, fraction of the box height
 * @param {number} [opts.clusters]          number of clusters
 * @param {number} [opts.clusterSd]         px
 * @param {number} [opts.radius]            beeswarm: dot radius, px
 * @param {number} [opts.gap]               beeswarm: distance between centers as a multiple of 2r (default 1.05)
 * @param {number} [opts.ties]              fraction 0–1
 * @param {number} [opts.tieJitter]         px
 * @returns {{x: number, y: number}[]}
 */
export function scatterPoints(rng, opts) {
    const { n, box } = opts;
    const [x0, y0, x1, y1] = box;
    const w = x1 - x0;
    const h = y1 - y0;
    const clampPt = (x, y) => ({ x: Math.min(x1, Math.max(x0, x)), y: Math.min(y1, Math.max(y0, y)) });
    const pts = [];
    const layout = opts.layout ?? 'uniform';
    if (layout === 'uniform') {
        for (let i = 0; i < n; i++)
            pts.push({ x: x0 + w * rng.uniform(), y: y0 + h * rng.uniform() });
    }
    else if (layout === 'curve') {
        const f = opts.curve ?? (u => u);
        const sd = opts.curveSd ?? 0.1;
        for (let i = 0; i < n; i++) {
            const u = rng.uniform();
            pts.push(clampPt(x0 + w * u, y1 - h * (f(u) + sd * rng.normal())));
        }
    }
    else if (layout === 'beeswarm') {
        // values (y) normal about the box's middle; each dot, in order of value, goes as close to the column's center
        // as it can without coming nearer than gap·2r to a placed dot (the usual swarm plot construction)
        const r = opts.radius ?? 8;
        const d = 2 * r * (opts.gap ?? 1.05);
        const cx = (x0 + x1) / 2;
        const ys = Array.from({ length: n }, () => Math.min(y1, Math.max(y0, (y0 + y1) / 2 + 0.22 * h * rng.normal()))).sort((a, b) => a - b);
        for (const y of ys) {
            const near = pts.filter(p => Math.abs(p.y - y) < d);
            let best = null;
            for (let k = 0; k < 4 * n && best === null; k++) {
                // candidate offsets 0, +s, −s, +2s, … where a dot just clears each neighbor
                const cands = [0];
                for (const p of near) {
                    const dx = Math.sqrt(Math.max(0, d * d - (p.y - y) ** 2));
                    cands.push(p.x - cx + dx, p.x - cx - dx);
                }
                cands.sort((a, b) => Math.abs(a) - Math.abs(b));
                best = cands.find(c => near.every(p => Math.hypot(p.x - (cx + c), p.y - y) >= d - 1e-6)) ?? null;
            }
            pts.push({ x: cx + best, y });
        }
    }
    else if (layout === 'clusters') {
        const k = opts.clusters ?? 4;
        const sd = opts.clusterSd ?? 0.05 * Math.min(w, h);
        const centers = Array.from({ length: k }, () => [x0 + w * rng.uniform(), y0 + h * rng.uniform()]);
        for (let i = 0; i < n; i++) {
            const [cx, cy] = centers[Math.floor(rng.uniform() * k)];
            pts.push(clampPt(cx + sd * rng.normal(), cy + sd * rng.normal()));
        }
    }
    else {
        throw new Error(`unknown layout ${layout}`);
    }
    const nTies = Math.round((opts.ties ?? 0) * n);
    const jitter = opts.tieJitter ?? 0;
    for (let i = 0; i < nTies; i++) {
        const src = pts[Math.floor(rng.uniform() * n)];
        const dst = Math.floor(rng.uniform() * n);
        const th = 2 * Math.PI * rng.uniform();
        const d = jitter * rng.uniform();
        pts[dst] = clampPt(src.x + d * Math.cos(th), src.y + d * Math.sin(th));
    }
    return pts;
}

// Distance from (x, y) to the polyline, and the arc length along it of the nearest point.
function polylineDistance(x, y, pts) {
    let best = Infinity;
    let bestArc = 0;
    let arc = 0;
    for (let i = 0; i + 1 < pts.length; i++) {
        const [ax, ay] = pts[i];
        const [bx, by] = pts[i + 1];
        const dx = bx - ax;
        const dy = by - ay;
        const len2 = dx * dx + dy * dy;
        const len = Math.sqrt(len2);
        const t = len2 > 0 ? Math.min(1, Math.max(0, ((x - ax) * dx + (y - ay) * dy) / len2)) : 0;
        const d = Math.hypot(x - (ax + t * dx), y - (ay + t * dy));
        if (d < best) {
            best = d;
            bestArc = arc + t * len;
        }
        arc += len;
    }
    return { d: best, arc: bestArc };
}

/**
 * Draw a polyline over an RgbImage in place. Coverage is approximate, clamp(w/2 + ½ − d, 0, 1) for distance d from
 * the pixel center, which is enough for masks to have something realistic to remove; dashes are cut by the arc
 * length of the nearest point (butt ends).
 */
function drawLine(image, line) {
    const { width, height, data } = image;
    const { points, color } = line;
    const alpha = line.alpha ?? 1;
    const half = line.width / 2;
    const xs = points.map(p => p[0]);
    const ys = points.map(p => p[1]);
    const pad = half + 1;
    const bx0 = Math.max(0, Math.floor(Math.min(...xs) - pad));
    const by0 = Math.max(0, Math.floor(Math.min(...ys) - pad));
    const bx1 = Math.min(width, Math.ceil(Math.max(...xs) + pad));
    const by1 = Math.min(height, Math.ceil(Math.max(...ys) + pad));
    for (let y = by0; y < by1; y++) {
        for (let x = bx0; x < bx1; x++) {
            const { d, arc } = polylineDistance(x + 0.5, y + 0.5, points);
            const m = Math.min(1, Math.max(0, half + 0.5 - d));
            if (m === 0)
                continue;
            if (line.dash && arc % (line.dash[0] + line.dash[1]) >= line.dash[0])
                continue;
            const p = 3 * (y * width + x);
            for (let ch = 0; ch < 3; ch++)
                data[p + ch] += (color[ch] - data[p + ch]) * alpha * m;
        }
    }
}

// axis lines and tick strokes as polylines (final px)
function axisLines(axes) {
    if (!axes)
        return [];
    const width = axes.width ?? 2;
    const len = axes.tickLength ?? 8;
    const color = axes.color ?? [0, 0, 0];
    const out = [];
    const dir = axes.inward ? -1 : 1; // +1: away from the plot (down for x, left for y)
    if (axes.x) {
        const { pos, from, to, ticks } = axes.x;
        out.push({ points: [[from, pos], [to, pos]], width, color });
        for (const t of ticks)
            out.push({ points: [[t, pos], [t, pos + dir * (width / 2 + len)]], width, color });
    }
    if (axes.y) {
        const { pos, from, to, ticks } = axes.y;
        out.push({ points: [[pos, from], [pos, to]], width, color });
        for (const t of ticks)
            out.push({ points: [[pos, t], [pos - dir * (width / 2 + len), t]], width, color });
    }
    return out;
}

// px between a tick's end and its label block: real labels leave a gap (at 4 px the round cap and blur joined the
// tick to its block)
const LABEL_GAP = 6;

/**
 * Label blocks in final px: centered on each tick, beyond its end.
 *
 * @param {AxesSpec} [axes]
 * @returns {number[][]}  [x0, y0, x1, y1) each
 */
export function labelBlocks(axes) {
    if (!axes)
        return [];
    const width = axes.width ?? 2;
    const len = axes.inward ? 0 : axes.tickLength ?? 8;
    const [w, h] = axes.label ?? [14, 10];
    const out = [];
    if (axes.x) {
        for (const t of axes.x.ticks)
            out.push([t - w / 2, axes.x.pos + width / 2 + len + LABEL_GAP, t + w / 2, axes.x.pos + width / 2 + len + LABEL_GAP + h]);
    }
    if (axes.y) {
        for (const t of axes.y.ticks)
            out.push([axes.y.pos - width / 2 - len - LABEL_GAP - w, t - h / 2, axes.y.pos - width / 2 - len - LABEL_GAP, t + h / 2]);
    }
    return out;
}

// solid rectangle with exact area coverage at its edges
function fillRect(image, [x0, y0, x1, y1], color) {
    for (let y = Math.max(0, Math.floor(y0)); y < Math.min(image.height, Math.ceil(y1)); y++) {
        const cy = Math.min(y + 1, y1) - Math.max(y, y0);
        for (let x = Math.max(0, Math.floor(x0)); x < Math.min(image.width, Math.ceil(x1)); x++) {
            const m = cy * (Math.min(x + 1, x1) - Math.max(x, x0));
            const p = 3 * (y * image.width + x);
            for (let ch = 0; ch < 3; ch++)
                image.data[p + ch] += (color[ch] - image.data[p + ch]) * m;
        }
    }
}

// Bilinear resampling to a new size, pixel centers aligned: destination center (i + ½) samples the source at
// (i + ½)/scale − ½ in source pixel indices, as browsers and most image tools do.
function resample(src, width, height) {
    const sx = src.width / width;
    const sy = src.height / height;
    const out = solidImage(width, height, [0, 0, 0]);
    for (let j = 0; j < height; j++) {
        const fy = Math.min(src.height - 1, Math.max(0, (j + 0.5) * sy - 0.5));
        const y0 = Math.floor(fy);
        const y1 = Math.min(src.height - 1, y0 + 1);
        const ty = fy - y0;
        for (let i = 0; i < width; i++) {
            const fx = Math.min(src.width - 1, Math.max(0, (i + 0.5) * sx - 0.5));
            const x0 = Math.floor(fx);
            const x1 = Math.min(src.width - 1, x0 + 1);
            const tx = fx - x0;
            for (let ch = 0; ch < 3; ch++) {
                const a = src.data[3 * (y0 * src.width + x0) + ch];
                const b = src.data[3 * (y0 * src.width + x1) + ch];
                const c = src.data[3 * (y1 * src.width + x0) + ch];
                const d = src.data[3 * (y1 * src.width + x1) + ch];
                out.data[3 * (j * width + i) + ch] = (a * (1 - tx) + b * tx) * (1 - ty) + (c * (1 - tx) + d * tx) * ty;
            }
        }
    }
    return out;
}

/**
 * Render a synthetic graph.
 *
 * @param {GraphSpec} spec
 * @returns {{image: import('./image.js').RgbaImage, truth: {x: number, y: number, family: number}[]}}
 */
export function synthGraph(spec) {
    const scale = spec.scale ?? 1;
    const nw = Math.round(spec.width / scale);
    const nh = Math.round(spec.height / scale);
    // native coordinates: final pixel coordinates divided by the scale (both measured from the top-left corner)
    const sx = nw / spec.width;
    const sy = nh / spec.height;
    let im = solidImage(nw, nh, spec.background);
    const native = line => ({ ...line, points: line.points.map(([x, y]) => [x * sx, y * sy]), width: line.width * sx, dash: line.dash?.map(v => v * sx) });
    for (const line of spec.gridlines ?? [])
        drawLine(im, native(line));
    const truth = [];
    spec.families.forEach((f, k) => {
        const model = createDotModel(im, { color: f.color, alpha: f.alpha, radius: f.radius * sx });
        model.addAll(f.points.map(p => ({ x: p.x * sx, y: p.y * sy })));
        im = model.image;
        for (const p of f.points)
            truth.push({ x: p.x, y: p.y, family: k });
    });
    for (const line of spec.lines ?? [])
        drawLine(im, native(line));
    for (const line of axisLines(spec.axes))
        drawLine(im, native(line));
    for (const [x0, y0, x1, y1] of labelBlocks(spec.axes))
        fillRect(im, [x0 * sx, y0 * sy, x1 * sx, y1 * sy], spec.axes.color ?? [0, 0, 0]);
    if (nw !== spec.width || nh !== spec.height)
        im = resample(im, spec.width, spec.height);
    if (spec.blur > 0) {
        const out = solidImage(spec.width, spec.height, [0, 0, 0]);
        blurRect(im, out, gaussianKernel(spec.blur), [0, 0, spec.width, spec.height]);
        im = out;
    }
    if (spec.noise > 0) {
        const rng = createRng(spec.seed ?? 1);
        for (let i = 0; i < im.data.length; i++)
            im.data[i] += spec.noise * rng.normal();
    }
    return { image: toRgba(im), truth };
}

/**
 * Named graph specs for tests, the benchmark and the app's examples. Each takes a seed and returns a GraphSpec.
 * `curve` imitates dog-ages: 832 × 1078, r = 24 px, α = 0.31, rendered at half size and upsampled, a fit curve and
 * dashed band lines over the dots, and a stack of near-coincident dots at the left.
 */
export const SYNTH_PRESETS = {
    sparse(seed) {
        const rng = createRng(seed);
        return {
            width: 400, height: 400, background: [255, 255, 255], seed,
            families: [{ color: [40, 110, 200], alpha: 0.35, radius: 10, points: scatterPoints(rng, { n: 30, box: [20, 20, 380, 380] }) }],
        };
    },
    dense(seed) {
        const rng = createRng(seed);
        return {
            width: 500, height: 400, background: [255, 255, 255], seed, blur: 0.8, noise: 0.5,
            families: [{
                color: [50, 160, 70], alpha: 0.3, radius: 11,
                points: scatterPoints(rng, { n: 150, box: [30, 30, 470, 370], layout: 'clusters', clusters: 5, clusterSd: 35, ties: 0.08, tieJitter: 2 }),
            }],
        };
    },
    // opaque swarm columns in three colors (cantons-like), with a black median line over each; dots touch
    beeswarm(seed) {
        const rng = createRng(seed);
        const r = 8;
        const colors = [[197, 193, 194], [122, 148, 179], [131, 22, 60]];
        const families = colors.map((color, k) => ({
            color, alpha: 1, radius: r,
            points: scatterPoints(rng, { n: 28, box: [60 + 140 * k, 40, 160 + 140 * k, 300], layout: 'beeswarm', radius: r, gap: 1.02 }),
        }));
        const lines = families.map(f => {
            const ys = f.points.map(p => p.y).sort((a, b) => a - b);
            const med = ys[Math.floor(ys.length / 2)];
            const cx = f.points.reduce((t, p) => t + p.x, 0) / f.points.length;
            return { points: [[cx - 40, med], [cx + 40, med]], width: 2, color: [0, 0, 0] };
        });
        return { width: 480, height: 340, background: [255, 255, 255], blur: 0.6, noise: 0.4, seed, families, lines };
    },
    curve(seed) {
        const rng = createRng(seed);
        const box = [285, 230, 745, 845]; // plot area of dog-ages, px
        const f = u => 0.17 * Math.log(Math.max(1e-3, u * 15)) + 0.33;
        const points = scatterPoints(rng, { n: 95, box, layout: 'curve', curve: f, curveSd: 0.12, ties: 0.06, tieJitter: 3 });
        // the stack near x = 0: dogs under a year old, many at nearly the same place
        for (let i = 0; i < 9; i++)
            points.push({ x: 288 + 6 * rng.uniform(), y: 830 - 30 * rng.uniform() });
        const curvePts = [];
        for (let u = 0.004; u <= 1.0001; u += 0.01)
            curvePts.push([box[0] + (box[2] - box[0]) * u, box[3] - (box[3] - box[1]) * f(u)]);
        const offset = d => curvePts.map(([x, y]) => [x, y + d]);
        return {
            width: 832, height: 1078, background: [255, 255, 255], scale: 2, blur: 0.6, noise: 0.4, seed,
            families: [{ color: [50, 161, 73], alpha: 0.31, radius: 24, points }],
            lines: [
                { points: curvePts, width: 12, color: [0, 0, 0] },
                { points: offset(-40), width: 9, color: [147, 149, 152], dash: [30, 14] },
                { points: offset(40), width: 9, color: [147, 149, 152], dash: [30, 14] },
            ],
        };
    },
};

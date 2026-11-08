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
 * @typedef {object} GraphSpec
 * @property {number} width
 * @property {number} height
 * @property {number[]} background
 * @property {FamilySpec[]} families  composited in order, each over the result of the previous ones
 * @property {LineSpec[]} [lines]     drawn over all dots
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
 * @param {string} [opts.layout]            'uniform' (default), 'curve' or 'clusters'
 * @param {function(number): number} [opts.curve]  u in [0, 1] → height fraction in [0, 1], 1 at the top
 * @param {number} [opts.curveSd]           scatter about the curve, fraction of the box height
 * @param {number} [opts.clusters]          number of clusters
 * @param {number} [opts.clusterSd]         px
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
    const truth = [];
    spec.families.forEach((f, k) => {
        const model = createDotModel(im, { color: f.color, alpha: f.alpha, radius: f.radius * sx });
        model.addAll(f.points.map(p => ({ x: p.x * sx, y: p.y * sy })));
        im = model.image;
        for (const p of f.points)
            truth.push({ x: p.x, y: p.y, family: k });
    });
    for (const line of spec.lines ?? [])
        drawLine(im, { ...line, points: line.points.map(([x, y]) => [x * sx, y * sy]), width: line.width * sx, dash: line.dash?.map(v => v * sx) });
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

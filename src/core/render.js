// Forward model: render dots of one family as a plotting library would, then blur as resampling would (PLAN.md
// §2.1). Each dot covers pixel p with fraction m(p), the exact area of the pixel square inside the disc. Dots are
// composited source-over in sRGB values (measured in phase 0) onto a background:
//
//     U(p) = c + (B(p) − c) · T(p),   T(p) = Π_i (1 − α · m_i(p))
//
// T is the family's transmittance; within one family the product commutes, so drawing order doesn't matter. The
// image is then blurred with a Gaussian of sd σ, since blur from resampling acts on the composited picture.
//
// Images here are RgbImage: { width, height, data } with data a Float32Array of 3 values per pixel (sRGB, 0–255,
// unrounded), row-major. Pixel (i, j) covers [i, i + 1) × [j, j + 1).

/**
 * @typedef {object} RgbImage
 * @property {number} width   px
 * @property {number} height  px
 * @property {Float32Array} data  RGB, sRGB values 0–255 (unrounded), 3 per pixel, row-major
 */

/**
 * @typedef {object} DotStyle
 * @property {number[]} color   [r, g, b] the full color c, 0–255
 * @property {number} alpha     0–1
 * @property {number} radius    px
 * @property {{inner: number, alpha: number}} [ring]  an outline: between inner × radius and radius the dot is more
 *                              opaque by alpha (its opacity there is alpha + the dot's α, at most 1), in the same
 *                              color c. Exact for a gray outline on a gray fill over white up to compositing order;
 *                              otherwise an approximation (PLAN.md phase 5, milestone 2)
 * @property {number} [blur]    px, sd of the Gaussian applied to the composited image (default 0)
 * @property {{B: number[], u: number[]}} [project]  render one value per pixel, the distance from B along the
 *                              unit vector u (a family's color line), instead of RGB: what the fit compares, at a
 *                              third of the cost (blur is linear, so projecting commutes with it). Assumes a
 *                              uniform background B.
 * @property {number[][]} [levels]  measured level colors L_0 = B, L_1, …: when given, a pixel's color is
 *                              interpolated in this table by its transmittance instead of c + (B − c)·T, and
 *                              the background image is ignored (the table assumes a uniform background)
 */

const BLUR_EXTENT = 3; // kernel half-width in units of σ
const GRID_CELL_RADII = 2; // spatial grid cell size, in dot radii

// ∫_{-r}^{x} √(r² − u²) du for x in [−r, r]
function chordIntegral(x, r) {
    return 0.5 * (x * Math.sqrt(Math.max(0, r * r - x * x)) + r * r * Math.asin(x / r)) + Math.PI * r * r / 4;
}

// Area of the disc of radius r at the origin intersected with the quarter plane {u ≤ x, v ≤ y}. With
// h(u) = √(r² − u²), the disc's column at u below y has length h + clamp(y, −h, h), which equals 2h where the
// column lies wholly below y, 0 where wholly above, and h + y between: integrate over u ≤ x.
function quarterArea(x, y, r) {
    if (x <= -r || y <= -r)
        return 0;
    const xc = Math.min(x, r);
    const H = chordIntegral(xc, r);
    if (y >= r)
        return 2 * H;
    // columns with |u| ≤ w have h ≥ |y|, so clamp(y, −h, h) = y there; elsewhere it is sign(y) · h
    const w = Math.sqrt(r * r - y * y);
    let inner = 0; // ∫ h over [−w, w] ∩ [−r, xc]
    let innerLen = 0;
    if (xc > -w) {
        const hi = Math.min(xc, w);
        inner = chordIntegral(hi, r) - chordIntegral(-w, r);
        innerLen = hi + w;
    }
    return H + y * innerLen + Math.sign(y) * (H - inner);
}

/**
 * Exact area of pixel (px, py), the square [px, px + 1) × [py, py + 1), inside the disc of radius r centered at
 * (cx, cy): the coverage fraction a renderer with exact area antialiasing would use.
 *
 * @param {number} cx  px
 * @param {number} cy  px
 * @param {number} r   px
 * @param {number} px  pixel column
 * @param {number} py  pixel row
 * @returns {number}   0–1
 */
export function pixelCoverage(cx, cy, r, px, py) {
    const x0 = px - cx;
    const x1 = x0 + 1;
    const y0 = py - cy;
    const y1 = y0 + 1;
    // nearest and farthest points of the square from the center decide the easy cases
    const nx = x0 > 0 ? x0 : (x1 < 0 ? x1 : 0);
    const ny = y0 > 0 ? y0 : (y1 < 0 ? y1 : 0);
    if (nx * nx + ny * ny >= r * r)
        return 0;
    const fx = Math.max(-x0, x1);
    const fy = Math.max(-y0, y1);
    if (fx * fx + fy * fy <= r * r)
        return 1;
    const a = quarterArea(x1, y1, r) - quarterArea(x0, y1, r) - quarterArea(x1, y0, r) + quarterArea(x0, y0, r);
    return Math.min(1, Math.max(0, a));
}

/**
 * Normalized Gaussian kernel of sd sigma, half-width ⌈BLUR_EXTENT·σ⌉.
 *
 * @param {number} sigma  px
 * @returns {Float64Array}
 */
export function gaussianKernel(sigma) {
    const k = Math.ceil(BLUR_EXTENT * sigma);
    const w = new Float64Array(2 * k + 1);
    let sum = 0;
    for (let i = -k; i <= k; i++) {
        w[i + k] = Math.exp(-0.5 * (i / sigma) ** 2);
        sum += w[i + k];
    }
    for (let i = 0; i < w.length; i++)
        w[i] /= sum;
    return w;
}

/**
 * Blur the rectangle [x0, x1) × [y0, y1) of src into dst with a separable Gaussian, clamping at the image edges.
 * Reads src up to the kernel half-width outside the rectangle.
 *
 * @param {RgbImage} src
 * @param {RgbImage} dst  same size as src; only the rectangle is written
 * @param {Float64Array} kernel
 * @param {number[]} rect  [x0, y0, x1, y1]
 */
export function blurRect(src, dst, kernel, rect) {
    const { width, height } = src;
    const ch = src.data.length / (width * height); // 3 for RgbImage, 1 for a plane
    if (ch === 1) {
        blurPlaneRect(src.data, dst.data, width, height, kernel, rect);
        return;
    }
    const k = (kernel.length - 1) / 2;
    const [x0, y0, x1, y1] = rect;
    const ty0 = Math.max(0, y0 - k);
    const ty1 = Math.min(height, y1 + k);
    const tw = x1 - x0;
    const tmp = new Float32Array(3 * tw * (ty1 - ty0)); // horizontal pass over the rows the vertical pass reads
    for (let y = ty0; y < ty1; y++) {
        for (let x = x0; x < x1; x++) {
            let r = 0;
            let g = 0;
            let b = 0;
            for (let i = -k; i <= k; i++) {
                const xs = Math.min(width - 1, Math.max(0, x + i));
                const s = 3 * (y * width + xs);
                const w = kernel[i + k];
                r += w * src.data[s];
                g += w * src.data[s + 1];
                b += w * src.data[s + 2];
            }
            const t = 3 * ((y - ty0) * tw + (x - x0));
            tmp[t] = r;
            tmp[t + 1] = g;
            tmp[t + 2] = b;
        }
    }
    for (let y = y0; y < y1; y++) {
        for (let x = x0; x < x1; x++) {
            let r = 0;
            let g = 0;
            let b = 0;
            for (let i = -k; i <= k; i++) {
                const ys = Math.min(height - 1, Math.max(0, y + i));
                const t = 3 * ((ys - ty0) * tw + (x - x0));
                const w = kernel[i + k];
                r += w * tmp[t];
                g += w * tmp[t + 1];
                b += w * tmp[t + 2];
            }
            const d = 3 * (y * width + x);
            dst.data[d] = r;
            dst.data[d + 1] = g;
            dst.data[d + 2] = b;
        }
    }
}

// one-channel version of blurRect, the fit's hot path: interior columns skip the edge clamping
function blurPlaneRect(src, dst, width, height, kernel, rect) {
    const k = (kernel.length - 1) / 2;
    const [x0, y0, x1, y1] = rect;
    const ty0 = Math.max(0, y0 - k);
    const ty1 = Math.min(height, y1 + k);
    const tw = x1 - x0;
    const tmp = new Float32Array(tw * (ty1 - ty0));
    for (let y = ty0; y < ty1; y++) {
        const row = y * width;
        for (let x = x0; x < x1; x++) {
            let s = 0;
            if (x - k >= 0 && x + k < width) {
                const base = row + x - k;
                for (let i = 0; i < kernel.length; i++)
                    s += kernel[i] * src[base + i];
            }
            else {
                for (let i = -k; i <= k; i++)
                    s += kernel[i + k] * src[row + Math.min(width - 1, Math.max(0, x + i))];
            }
            tmp[(y - ty0) * tw + x - x0] = s;
        }
    }
    for (let y = y0; y < y1; y++) {
        for (let x = x0; x < x1; x++) {
            let s = 0;
            for (let i = -k; i <= k; i++) {
                const ys = Math.min(ty1 - 1, Math.max(ty0, y + i));
                s += kernel[i + k] * tmp[(ys - ty0) * tw + x - x0];
            }
            dst[y * width + x] = s;
        }
    }
}

/**
 * An RgbImage filled with one color.
 *
 * @param {number} width
 * @param {number} height
 * @param {number[]} color  [r, g, b]
 * @returns {RgbImage}
 */
export function solidImage(width, height, color) {
    const data = new Float32Array(3 * width * height);
    for (let p = 0; p < width * height; p++) {
        data[3 * p] = color[0];
        data[3 * p + 1] = color[1];
        data[3 * p + 2] = color[2];
    }
    return { width, height, data };
}

/**
 * Round an RgbImage to an opaque RgbaImage (8-bit, as a PNG or canvas holds it).
 *
 * @param {RgbImage} image
 * @returns {import('./image.js').RgbaImage}
 */
export function toRgba(image) {
    const n = image.width * image.height;
    const data = new Uint8ClampedArray(4 * n);
    for (let p = 0; p < n; p++) {
        data[4 * p] = Math.round(image.data[3 * p]);
        data[4 * p + 1] = Math.round(image.data[3 * p + 1]);
        data[4 * p + 2] = Math.round(image.data[3 * p + 2]);
        data[4 * p + 3] = 255;
    }
    return { width: image.width, height: image.height, data };
}

/**
 * An RgbaImage as an RgbImage (alpha dropped).
 *
 * @param {import('./image.js').RgbaImage} image
 * @returns {RgbImage}
 */
export function fromRgba(image) {
    const n = image.width * image.height;
    const data = new Float32Array(3 * n);
    for (let p = 0; p < n; p++) {
        data[3 * p] = image.data[4 * p];
        data[3 * p + 1] = image.data[4 * p + 1];
        data[3 * p + 2] = image.data[4 * p + 2];
    }
    return { width: image.width, height: image.height, data };
}

/** The smallest rectangle [x0, y0, x1, y1) holding two. */
export function unionRect(a, b) {
    return [Math.min(a[0], b[0]), Math.min(a[1], b[1]), Math.max(a[2], b[2]), Math.max(a[3], b[3])];
}

/**
 * What an incremental model keeps beside its pixels: the dots' records by id, a spatial grid that finds the dots near
 * a rectangle, and snapshots of both over a rectangle, so that a rejected trial is undone by a copy instead of a
 * second render. createDotModel and outline.createOutlineModel differ in what a record holds and in how a rectangle
 * is rendered from the records; the rest is this.
 *
 * A record is any object with x and y (px); the models add what they render from (rect, stamps, z). A grid cell
 * lists its dots in the order they were put there, which is the order `near` returns them in and so the order a
 * model composites them in: put, drop and restore keep to one sequence of grid changes for that reason.
 *
 * @param {number} width  px
 * @param {number} height  px
 * @param {number} radius  px, the dots'
 * @param {number} kHalf  px, the blur kernel's half-width (0 without blur)
 * @param {{data: Float32Array, stride: number, blurred: boolean}[]} buffers  the model's planes that a snapshot
 *   saves: `stride` values per pixel; a `blurred` one is saved over the rectangle grown by kHalf
 */
export function createDotStore(width, height, radius, kHalf, buffers) {
    const cell = Math.max(1, GRID_CELL_RADII * radius);
    const gw = Math.ceil(width / cell) + 1;
    const gh = Math.ceil(height / cell) + 1;
    const grid = Array.from({ length: gw * gh }, () => new Set());
    const dots = new Map();
    let nextId = 0;

    function cellOf(x, y) {
        const gx = Math.min(gw - 1, Math.max(0, Math.floor(x / cell)));
        const gy = Math.min(gh - 1, Math.max(0, Math.floor(y / cell)));
        return gy * gw + gx;
    }

    // ids of the dots whose rectangles can reach the rectangle, in grid order
    function idsNear(rect) {
        const [x0, y0, x1, y1] = rect;
        const gx0 = Math.max(0, Math.floor((x0 - radius) / cell));
        const gy0 = Math.max(0, Math.floor((y0 - radius) / cell));
        const gx1 = Math.min(gw - 1, Math.floor((x1 + radius) / cell));
        const gy1 = Math.min(gh - 1, Math.floor((y1 + radius) / cell));
        const out = [];
        for (let gy = gy0; gy <= gy1; gy++) {
            for (let gx = gx0; gx <= gx1; gx++) {
                for (const id of grid[gy * gw + gx])
                    out.push(id);
            }
        }
        return out;
    }

    function blurredRect(rect) {
        const [x0, y0, x1, y1] = rect;
        return [Math.max(0, x0 - kHalf), Math.max(0, y0 - kHalf), Math.min(width, x1 + kHalf), Math.min(height, y1 + kHalf)];
    }

    function copyRect(src, stride, rect) {
        const [x0, y0, x1, y1] = rect;
        const w = Math.max(0, x1 - x0);
        const out = new Float32Array(w * Math.max(0, y1 - y0) * stride);
        for (let y = y0; y < y1; y++)
            out.set(src.subarray(stride * (y * width + x0), stride * (y * width + x1)), stride * (y - y0) * w);
        return out;
    }

    function pasteRect(dst, stride, rect, saved) {
        const [x0, y0, x1, y1] = rect;
        const w = Math.max(0, x1 - x0);
        for (let y = y0; y < y1; y++)
            dst.set(saved.subarray(stride * (y - y0) * w, stride * (y - y0 + 1) * w), stride * (y * width + x0));
    }

    return {
        /** @returns {number} an id no dot has had */
        newId() {
            return nextId++;
        },

        get(id) {
            return dots.get(id);
        },

        /** Set a dot's record, new or in place of its old one (which is returned). */
        put(id, rec) {
            const old = dots.get(id);
            if (old)
                grid[cellOf(old.x, old.y)].delete(id);
            dots.set(id, rec);
            grid[cellOf(rec.x, rec.y)].add(id);
            return old;
        },

        /** Take a dot out; returns its record. */
        drop(id) {
            const d = dots.get(id);
            grid[cellOf(d.x, d.y)].delete(id);
            dots.delete(id);
            return d;
        },

        /** The records of the dots whose rectangles can reach the rectangle, in grid order. */
        near(rect) {
            return idsNear(rect).map(id => dots.get(id));
        },

        /** @returns {[number, object][]}  every [id, record], in id order (restore can put a dot back at the Map's end) */
        entries() {
            return [...dots].sort((a, b) => a[0] - b[0]);
        },

        /** The pixel rectangle of the output that pixels in `rect` can affect, blur included. */
        blurredRect,

        /**
         * Save the state a change within `rect` (a footprint) can alter: the records of the dots that can reach the
         * rectangle and the buffers over it (grown by the blur for a blurred one). restore() puts it back without
         * re-rendering, so a rejected trial costs a copy instead of a second render. Only the dots near the rectangle
         * are saved (in id order, so restore() refills the grid cells in the order a full copy would): a trial's cost
         * must not grow with the number of dots in the image. Every dot added, moved or removed between the two
         * calls must have its footprint inside `rect`, before and after.
         */
        snapshot(rect) {
            const outRect = blurredRect(rect);
            return {
                rect, outRect,
                records: idsNear(rect).sort((a, b) => a - b).map(id => [id, dots.get(id)]),
                saved: buffers.map(b => copyRect(b.data, b.stride, b.blurred ? outRect : rect)),
            };
        },

        restore(snap) {
            const saved = new Map(snap.records);
            for (const id of idsNear(snap.rect)) {
                const d = dots.get(id);
                if (saved.get(id) === d)
                    continue;
                grid[cellOf(d.x, d.y)].delete(id);
                if (!saved.has(id))
                    dots.delete(id);
            }
            for (const [id, d] of snap.records) {
                if (dots.get(id) !== d) {
                    grid[cellOf(d.x, d.y)].add(id);
                    dots.set(id, d);
                }
            }
            buffers.forEach((b, k) => pasteRect(b.data, b.stride, b.blurred ? snap.outRect : snap.rect, snap.saved[k]));
        },
    };
}

/**
 * An incrementally updatable rendering of one dot family over a background. Adding, moving or removing a dot
 * recomputes only the pixels it touches (its bounding box, grown by the blur kernel), so a fit can try a change
 * and read the affected patch cheaply. A spatial grid finds the other dots overlapping that patch.
 *
 * Complexity per change: O((2r + 2)² · k) for k dots overlapping the patch, plus the blur, O((2r + 6σ)² · 6σ).
 *
 * @param {RgbImage} background  composited under the dots; not modified (a later family's background can be an
 *                               earlier family's rendering)
 * @param {DotStyle} style
 */
export function createDotModel(background, style) {
    const { width, height } = background;
    const { color, alpha, radius } = style;
    const blur = style.blur ?? 0;
    const kernel = blur > 0 ? gaussianKernel(blur) : null;
    const kHalf = kernel ? (kernel.length - 1) / 2 : 0;
    const T = new Float32Array(width * height).fill(1);
    const project = style.project ?? null;
    const CH = project ? 1 : 3;
    const proj = c => (project.B[0] - c[0]) * project.u[0] + (project.B[1] - c[1]) * project.u[1] + (project.B[2] - c[2]) * project.u[2];
    const initial = () => (project ? new Float32Array(width * height) : Float32Array.from(background.data));
    const composite = { width, height, data: initial() };
    const output = kernel ? { width, height, data: initial() } : composite;
    const table = style.levels ?? null;
    // Interpolating the table linearly in T between T_k = q^k and T_(k+1) reproduces each measured plateau, and
    // equals c + (B − c)·T exactly when the table is geometric, edges included (the color is linear in T).
    const q = 1 - alpha;
    const tPow = table ? table.map((_, k) => q ** k) : null;
    const logQ = Math.log(q);
    const tableT = table && project ? table.map(proj) : null;
    const spanT = project ? proj(color) : 0; // t of the full color: t = spanT · (1 − T) without a table
    function tableColor(Tp, out, o) {
        const K = table.length - 1;
        let k = 0;
        if (q > 0 && Tp < 1)
            k = Tp <= 0 ? K - 1 : Math.min(K - 1, Math.floor(Math.log(Tp) / logQ));
        const frac = Math.min(1, Math.max(0, (tPow[k] - Tp) / (tPow[k] - tPow[k + 1])));
        const a = table[k];
        const b = table[k + 1];
        out[o] = a[0] + frac * (b[0] - a[0]);
        out[o + 1] = a[1] + frac * (b[1] - a[1]);
        out[o + 2] = a[2] + frac * (b[2] - a[2]);
    }
    function tableValue(Tp) {
        const K = table.length - 1;
        let k = 0;
        if (q > 0 && Tp < 1)
            k = Tp <= 0 ? K - 1 : Math.min(K - 1, Math.floor(Math.log(Tp) / logQ));
        const frac = Math.min(1, Math.max(0, (tPow[k] - Tp) / (tPow[k] - tPow[k + 1])));
        return tableT[k] + frac * (tableT[k + 1] - tableT[k]);
    }
    // a record is { x, y, rect, stamp }: each dot keeps its coverage over its rectangle, so re-rendering a patch costs
    // a multiply per covered pixel for the dots that didn't change instead of recomputing their coverage
    const store = createDotStore(width, height, radius, kHalf, [
        { data: T, stride: 1, blurred: false },
        { data: composite.data, stride: CH, blurred: false },
        ...(kernel ? [{ data: output.data, stride: CH, blurred: true }] : []),
    ]);

    // pixel rectangle [x0, y0, x1, y1) a dot at (x, y) can touch, clipped to the image
    function dotRect(x, y) {
        return [
            Math.max(0, Math.floor(x - radius)),
            Math.max(0, Math.floor(y - radius)),
            Math.min(width, Math.floor(x + radius) + 1),
            Math.min(height, Math.floor(y + radius) + 1),
        ];
    }

    const ring = style.ring ?? null;
    const innerRadius = ring ? ring.inner * radius : 0;

    function makeRecord(x, y) {
        const rect = dotRect(x, y);
        const [x0, y0, x1, y1] = rect;
        const w = Math.max(0, x1 - x0);
        const stamp = new Float32Array(w * Math.max(0, y1 - y0));
        for (let yy = y0; yy < y1; yy++) {
            for (let xx = x0; xx < x1; xx++) {
                const m = pixelCoverage(x, y, radius, xx, yy);
                // the ring's coverage is the disc's less the inner disc's: exact antialiasing for the annulus
                const mr = ring && m > 0 ? m - pixelCoverage(x, y, innerRadius, xx, yy) : 0;
                stamp[(yy - y0) * w + xx - x0] = 1 - alpha * m - (ring ? ring.alpha * mr : 0);
            }
        }
        return { x, y, rect, stamp };
    }

    function recompute(rect) {
        const [x0, y0, x1, y1] = rect;
        if (x0 >= x1 || y0 >= y1)
            return;
        for (let y = y0; y < y1; y++)
            T.fill(1, y * width + x0, y * width + x1);
        for (const d of store.near(rect)) {
            const [dx0, dy0, dx1] = d.rect;
            const dw = dx1 - dx0;
            const ya = Math.max(y0, dy0);
            const yb = Math.min(y1, d.rect[3]);
            const xa = Math.max(x0, dx0);
            const xb = Math.min(x1, dx1);
            for (let y = ya; y < yb; y++) {
                const row = (y - dy0) * dw - dx0;
                for (let x = xa; x < xb; x++)
                    T[y * width + x] *= d.stamp[row + x];
            }
        }
        for (let y = y0; y < y1; y++) {
            for (let x = x0; x < x1; x++) {
                const p = y * width + x;
                if (project) {
                    composite.data[p] = table ? tableValue(T[p]) : spanT * (1 - T[p]);
                    continue;
                }
                if (table) {
                    tableColor(T[p], composite.data, 3 * p);
                    continue;
                }
                for (let ch = 0; ch < 3; ch++)
                    composite.data[3 * p + ch] = color[ch] + (background.data[3 * p + ch] - color[ch]) * T[p];
            }
        }
        if (kernel)
            blurRect(composite, output, kernel, store.blurredRect(rect));
    }

    return {
        /** With `project`: the value a pixel of transmittance T renders to (before blur). */
        valueOf(Tp) {
            return table ? tableValue(Tp) : spanT * (1 - Tp);
        },

        /** The rendered image (blurred if the style has blur); one value per pixel with `project`. Live. */
        image: output,
        /** The transmittance T before blur, one value per pixel. Live. */
        transmittance: T,

        /**
         * @param {number} x  px
         * @param {number} y  px
         * @returns {number} the new dot's id
         */
        add(x, y) {
            const id = store.newId();
            const rec = makeRecord(x, y);
            store.put(id, rec);
            recompute(rec.rect);
            return id;
        },

        /** Add many dots and recompute once (for a first render). */
        addAll(points) {
            const ids = [];
            for (const { x, y } of points) {
                const id = store.newId();
                store.put(id, makeRecord(x, y));
                ids.push(id);
            }
            recompute([0, 0, width, height]);
            return ids;
        },

        remove(id) {
            recompute(store.drop(id).rect);
        },

        move(id, x, y) {
            const rec = makeRecord(x, y);
            const old = store.put(id, rec);
            recompute(unionRect(old.rect, rec.rect));
        },

        /** Save and put back the state a change within a rectangle can alter (createDotStore's snapshot). */
        snapshot: store.snapshot,
        restore: store.restore,

        /** @returns {{id: number, x: number, y: number}[]} in id order */
        dots() {
            return store.entries().map(([id, d]) => ({ id, x: d.x, y: d.y }));
        },

        /** The pixel rectangle [x0, y0, x1, y1) a dot at (x, y) covers (before blur). */
        footprint: dotRect,

        /** The pixel rectangle [x0, y0, x1, y1) of the output a dot at (x, y) can affect, blur included. */
        influenceRect(x, y) {
            return store.blurredRect(dotRect(x, y));
        },
    };
}

/**
 * Render a set of dots of one family over a background in one go.
 *
 * @param {RgbImage} background
 * @param {{x: number, y: number}[]} points  dot centers, px
 * @param {DotStyle} style
 * @returns {RgbImage}
 */
export function renderDots(background, points, style) {
    const model = createDotModel(background, style);
    model.addAll(points);
    return model.image;
}

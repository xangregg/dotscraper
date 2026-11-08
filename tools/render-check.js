// Compare the forward model (src/core/render.js) with Chrome's 2D canvas drawing the same dots (PLAN.md phase 1).
// The canvas composites each fill in 8 bits, so interior pixels can differ from the model by rounding that grows
// with the number of overlapping dots; edge pixels show how the canvas antialiases. Run through check-page:
//
//   npm run check:page -- http://localhost:8003/tools/render-check.html --until 'window.renderCheckResult' \
//       --eval 'JSON.stringify(window.renderCheckResult, null, 1)' --screenshot out.png

import { createRng } from '../src/core/rng.js';
import { pixelCoverage, renderDots, solidImage } from '../src/core/render.js';

const SIZE = 320; // px, each test canvas
const DIFF_GAIN = 20; // diff images show |model − canvas| × this

function canvasRender(points, style, bg) {
    const canvas = new OffscreenCanvas(SIZE, SIZE);
    const ctx = canvas.getContext('2d', { colorSpace: 'srgb', willReadFrequently: true });
    ctx.fillStyle = `rgb(${bg.join(' ')})`;
    ctx.fillRect(0, 0, SIZE, SIZE);
    ctx.fillStyle = `rgb(${style.color.join(' ')} / ${style.alpha})`;
    for (const p of points) {
        ctx.beginPath();
        ctx.arc(p.x, p.y, style.radius, 0, 2 * Math.PI);
        ctx.fill();
    }
    return ctx.getImageData(0, 0, SIZE, SIZE).data;
}

// For each pixel: is any dot's edge in it (0 < coverage < 1)? Interiors are exact in the model, so they isolate
// compositing and rounding from antialiasing.
function edgeMask(points, r) {
    const mask = new Uint8Array(SIZE * SIZE);
    for (const p of points) {
        for (let y = Math.max(0, Math.floor(p.y - r)); y <= Math.min(SIZE - 1, Math.floor(p.y + r)); y++) {
            for (let x = Math.max(0, Math.floor(p.x - r)); x <= Math.min(SIZE - 1, Math.floor(p.x + r)); x++) {
                const m = pixelCoverage(p.x, p.y, r, x, y);
                if (m > 0 && m < 1)
                    mask[y * SIZE + x] = 1;
            }
        }
    }
    return mask;
}

function compare(name, points, style, bg) {
    const model = renderDots(solidImage(SIZE, SIZE, bg), points, style);
    const canvas = canvasRender(points, style, bg);
    const edges = edgeMask(points, style.radius);
    const diffImage = new ImageData(SIZE, SIZE);
    const stats = { interior: { n: 0, maxAbs: 0, sumAbs: 0 }, edge: { n: 0, maxAbs: 0, sumAbs: 0, sum: 0 }, far: { n: 0, maxAbs: 0, sumAbs: 0 } };
    // interior pixels more than one pixel from any edge pixel: the canvas's antialiasing reaches a little past
    // the exact disc, so only these isolate compositing and rounding
    const nearEdge = new Uint8Array(SIZE * SIZE);
    for (let y = 0; y < SIZE; y++) {
        for (let x = 0; x < SIZE; x++) {
            if (!edges[y * SIZE + x])
                continue;
            for (let dy = -1; dy <= 1; dy++) {
                for (let dx = -1; dx <= 1; dx++) {
                    if (x + dx >= 0 && x + dx < SIZE && y + dy >= 0 && y + dy < SIZE)
                        nearEdge[(y + dy) * SIZE + x + dx] = 1;
                }
            }
        }
    }
    for (let p = 0; p < SIZE * SIZE; p++) {
        let worst = 0;
        let signed = 0;
        for (let ch = 0; ch < 3; ch++) {
            const d = model.data[3 * p + ch] - canvas[4 * p + ch];
            signed += d;
            worst = Math.max(worst, Math.abs(d));
        }
        const s = edges[p] ? stats.edge : stats.interior;
        s.n++;
        s.maxAbs = Math.max(s.maxAbs, worst);
        s.sumAbs += worst;
        if (edges[p])
            s.sum += signed / 3;
        if (!nearEdge[p]) {
            stats.far.n++;
            stats.far.maxAbs = Math.max(stats.far.maxAbs, worst);
            stats.far.sumAbs += worst;
        }
        const v = Math.min(255, Math.round(DIFF_GAIN * worst));
        diffImage.data.set([v, v, v, 255], 4 * p);
    }
    // ink: total darkening per dot relative to the background, model vs canvas; for opaque dots, an effective
    // radius difference
    let inkModel = 0;
    let inkCanvas = 0;
    for (let p = 0; p < SIZE * SIZE; p++) {
        for (let ch = 0; ch < 3; ch++) {
            inkModel += bg[ch] - model.data[3 * p + ch];
            inkCanvas += bg[ch] - canvas[4 * p + ch];
        }
    }
    const out = {
        name,
        dots: points.length,
        farInterior: { pixels: stats.far.n, maxAbs: +stats.far.maxAbs.toFixed(2), meanAbs: +(stats.far.sumAbs / stats.far.n).toFixed(3) },
        interior: { pixels: stats.interior.n, maxAbs: +stats.interior.maxAbs.toFixed(2), meanAbs: +(stats.interior.sumAbs / stats.interior.n).toFixed(3) },
        edge: { pixels: stats.edge.n, maxAbs: +stats.edge.maxAbs.toFixed(2), meanAbs: +(stats.edge.sumAbs / stats.edge.n).toFixed(3), meanSigned: +(stats.edge.sum / stats.edge.n).toFixed(3) },
        inkRatio: +(inkCanvas / inkModel).toFixed(5),
    };
    return { out, model, canvas, diffImage };
}

// Interior levels of a stack of n identical dots, canvas vs the model's c + (B − c)(1 − α)^n.
function stackLevels(style, bg, maxN) {
    const rows = [];
    for (let n = 1; n <= maxN; n++) {
        const points = Array.from({ length: n }, () => ({ x: SIZE / 2, y: SIZE / 2 }));
        const canvas = canvasRender(points, style, bg);
        const p = 4 * ((SIZE / 2) * SIZE + SIZE / 2);
        const exact = style.color.map((c, ch) => c + (bg[ch] - c) * (1 - style.alpha) ** n);
        rows.push({ n, canvas: [canvas[p], canvas[p + 1], canvas[p + 2]], model: exact.map(v => +v.toFixed(2)) });
    }
    return rows;
}

// Coverage along rows through an isolated opaque dot, model vs canvas, to see the shape of the canvas's
// antialiasing: center (60.37, 60.21), r 8.37, rows y = 60 (through the center) and 53 (near the top)
function edgeProfileRows() {
    const style = { color: [0, 0, 0], alpha: 1, radius: 8.37 };
    const points = [{ x: 60.37, y: 60.21 }];
    const model = renderDots(solidImage(SIZE, SIZE, bg), points, style);
    const canvas = canvasRender(points, style, bg);
    const rows = {};
    for (const y of [60, 53, 52]) {
        const m = [];
        const c = [];
        for (let x = 48; x <= 73; x++) {
            m.push(+(1 - model.data[3 * (y * SIZE + x)] / 255).toFixed(3));
            c.push(+(1 - canvas[4 * (y * SIZE + x)] / 255).toFixed(3));
        }
        rows[y] = { model: m, canvas: c };
    }
    return rows;
}

function show(title, { model, canvas, diffImage }) {
    const fig = document.createElement('figure');
    const cap = document.createElement('figcaption');
    cap.textContent = title;
    const toCanvas = data => {
        const c = document.createElement('canvas');
        c.width = SIZE;
        c.height = SIZE;
        c.getContext('2d').putImageData(data, 0, 0);
        return c;
    };
    const m = new ImageData(SIZE, SIZE);
    for (let p = 0; p < SIZE * SIZE; p++)
        m.data.set([model.data[3 * p], model.data[3 * p + 1], model.data[3 * p + 2], 255], 4 * p);
    fig.append(cap, toCanvas(m), toCanvas(new ImageData(canvas, SIZE, SIZE)), toCanvas(diffImage));
    document.getElementById('figures').append(fig);
}

const bg = [255, 255, 255];
const green = [50, 161, 73];
const cases = [];
for (const [radius, alpha, n] of [[24.04, 0.31, 40], [8.37, 0.3, 120], [3.3, 0.5, 200], [11, 1, 60], [6.5, 0.7, 120]]) {
    const rng = createRng(Math.round(radius * 100) + n);
    const points = Array.from({ length: n }, () => ({ x: 20 + (SIZE - 40) * rng.uniform(), y: 20 + (SIZE - 40) * rng.uniform() }));
    const r = compare(`r ${radius}, α ${alpha}`, points, { color: green, alpha, radius }, bg);
    cases.push(r.out);
    show(`r ${radius}, α ${alpha}, ${n} dots: model, canvas, |diff| × ${DIFF_GAIN}`, r);
}
// isolated opaque dots: effective radius from the ink
const iso = [];
for (const radius of [2.5, 8.37, 24.04]) {
    const points = [];
    for (let i = 0; i < 9; i++)
        points.push({ x: 60 + 100 * (i % 3) + 0.37 * i, y: 60 + 100 * Math.floor(i / 3) + 0.21 * i });
    const r = compare(`isolated r ${radius}`, points, { color: [0, 0, 0], alpha: 1, radius }, bg);
    iso.push({ radius, inkRatio: r.out.inkRatio, effectiveRadius: +(radius * Math.sqrt(r.out.inkRatio)).toFixed(3), edgeMeanSigned: r.out.edge.meanSigned });
}

window.renderCheckResult = { cases, isolated: iso, stack: stackLevels({ color: green, alpha: 0.31, radius: 20 }, bg, 8), profile: edgeProfileRows() };
document.getElementById('summary').textContent = JSON.stringify(window.renderCheckResult, null, 1);

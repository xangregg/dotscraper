import { test } from 'node:test';
import assert from 'node:assert/strict';
import { existsSync } from 'node:fs';
import { createRng } from '../src/core/rng.js';
import { distanceTransform, dilate, opening, buildMask, CLASS_FAMILY, CLASS_OTHER, CLASS_BACKGROUND } from '../src/core/layers.js';
import { estimateStyle } from '../src/core/style.js';
import { synthGraph, SYNTH_PRESETS, scatterPoints } from '../src/core/synth.js';
import { readPng } from '../tools/png.mjs';

const near = (a, b, tol, what) => assert.ok(Math.abs(a - b) <= tol, `${what}: ${a} vs ${b} (±${tol})`);

test('distance transform equals brute force on random masks', () => {
    const rng = createRng(11);
    for (const [w, h, density] of [[17, 11, 0.05], [30, 25, 0.01], [9, 40, 0.3], [5, 5, 0]]) {
        const mask = new Uint8Array(w * h);
        for (let p = 0; p < w * h; p++)
            mask[p] = rng.uniform() < density ? 1 : 0;
        const d2 = distanceTransform(mask, w, h);
        for (let y = 0; y < h; y++) {
            for (let x = 0; x < w; x++) {
                let best = Infinity;
                for (let q = 0; q < w * h; q++) {
                    if (mask[q])
                        best = Math.min(best, (q % w - x) ** 2 + (Math.floor(q / w) - y) ** 2);
                }
                assert.equal(d2[y * w + x], best, `${w}×${h} at ${x},${y}`);
            }
        }
    }
});

test('opening removes a line thinner than 2r and keeps a disc larger than r', () => {
    const w = 80;
    const h = 60;
    const mask = new Uint8Array(w * h);
    for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
            if (Math.abs(y - 10) <= 1)
                mask[y * w + x] = 1; // 3 px line
            if ((x - 40) ** 2 + (y - 38) ** 2 <= 144)
                mask[y * w + x] = 1; // disc of radius 12
        }
    }
    const o = opening(mask, w, h, 4);
    assert.equal(o[10 * w + 20], 0);
    assert.equal(o[38 * w + 40], 1);
    assert.equal(o[38 * w + 51], 1); // near the disc's edge
    const grown = dilate(mask, w, h, 2);
    assert.equal(grown[13 * w + 20], 1);
    assert.equal(grown[14 * w + 20], 0);
});

// distance from (x, y) to a polyline
function lineDistance(x, y, pts) {
    let best = Infinity;
    for (let i = 0; i + 1 < pts.length; i++) {
        const [ax, ay] = pts[i];
        const [bx, by] = pts[i + 1];
        const dx = bx - ax;
        const dy = by - ay;
        const t = Math.max(0, Math.min(1, ((x - ax) * dx + (y - ay) * dy) / (dx * dx + dy * dy)));
        best = Math.min(best, Math.hypot(x - ax - t * dx, y - ay - t * dy));
    }
    return best;
}

test('synthetic curve graph: the lines are masked, dot pixels away from them are not', () => {
    const spec = SYNTH_PRESETS.curve(1);
    const { image, truth } = synthGraph(spec);
    const est = estimateStyle(image);
    const { mask, classes } = buildMask(image, est);
    let lineCore = 0;
    let lineMasked = 0;
    let dotFar = 0;
    let dotFarMasked = 0;
    for (let y = 0; y < image.height; y++) {
        for (let x = 0; x < image.width; x++) {
            const p = y * image.width + x;
            // the solid curve's core (dashes have gaps, so use the solid black line only)
            if (lineDistance(x + 0.5, y + 0.5, spec.lines[0].points) < spec.lines[0].width / 2 - 1) {
                lineCore++;
                lineMasked += mask[p];
            }
            if (!truth.some(t => Math.hypot(x + 0.5 - t.x, y + 0.5 - t.y) < 24 - 2))
                continue;
            const dLine = Math.min(...spec.lines.map(l => lineDistance(x + 0.5, y + 0.5, l.points) - l.width / 2));
            if (dLine > 6) {
                dotFar++;
                dotFarMasked += mask[p];
                assert.equal(classes[p], CLASS_FAMILY);
            }
        }
    }
    assert.equal(lineMasked, lineCore, 'every pixel of the solid line is masked');
    assert.ok(dotFarMasked / dotFar < 0.002, `${dotFarMasked} of ${dotFar} dot pixels masked`);
});

test('an opaque line in the family’s full color is masked as thin; dots under it are not', () => {
    const c = [246, 66, 60];
    const rng = createRng(4);
    const points = Array.from({ length: 40 }, () => ({ x: 30 + 340 * rng.uniform(), y: 30 + 240 * rng.uniform() }));
    const spec = {
        width: 400, height: 300, background: [255, 255, 255], blur: 0.6,
        families: [{ color: c, alpha: 0.3, radius: 8.4, points }],
        lines: [{ points: [[10, 250], [200, 120], [390, 60]], width: 7, color: c }],
    };
    const { image } = synthGraph(spec);
    const est = estimateStyle(image);
    assert.ok(Math.abs(est.style.alpha - 0.3) < 0.01);
    const { mask, thin, stats } = buildMask(image, est);
    // the line's core is found as thin; its blurred edge, a little lighter, is covered by the mask's growth
    let core = 0;
    let coreThin = 0;
    for (let y = 0; y < 300; y++) {
        for (let x = 0; x < 400; x++) {
            const d = lineDistance(x + 0.5, y + 0.5, spec.lines[0].points);
            if (d < 2) {
                core++;
                coreThin += thin[y * 400 + x];
            }
            if (d < spec.lines[0].width / 2)
                assert.equal(mask[y * 400 + x], 1, `line pixel ${x},${y} masked`);
        }
    }
    assert.ok(coreThin / core > 0.95, `${coreThin} of ${core} line core pixels thin`);
    // an isolated dot away from the line stays unmasked at its center
    const far = points.find(p => lineDistance(p.x, p.y, spec.lines[0].points) > 20 && points.every(q => q === p || Math.hypot(q.x - p.x, q.y - p.y) > 20));
    assert.equal(mask[Math.floor(far.y) * 400 + Math.floor(far.x)], 0);
    assert.ok(stats.thin > 0);
});

const DOG = new URL('../testgraphs/dog-ages.png', import.meta.url).pathname;
test('dog-ages: curve, dashes and text masked; dots and background not', { skip: !existsSync(DOG) && 'local test image absent' }, () => {
    const image = readPng(DOG);
    const est = estimateStyle(image);
    const { mask, classes, thin } = buildMask(image, est);
    const at = (x, y) => ({ m: mask[y * image.width + x], c: classes[y * image.width + x] });
    // dots beside the curve and dashes, whose pale halos read as background, are not "too thin to be dots"
    for (const [x0, y0, x1, y1] of [[389, 361, 407, 388], [414, 420, 442, 438], [419, 395, 424, 410]]) {
        for (let y = y0; y <= y1; y++) {
            for (let x = x0; x <= x1; x++)
                assert.equal(thin[y * image.width + x], 0, `${x},${y}`);
        }
    }
    assert.deepEqual(at(714, 508), { m: 0, c: CLASS_FAMILY }); // isolated dot at 14.5, 47
    assert.deepEqual(at(60, 60), { m: 0, c: CLASS_BACKGROUND });
    // every pure black or dash-gray pixel inside the plot area is masked
    let n = 0;
    for (let y = 230; y < 850; y++) {
        for (let x = 285; x < 790; x++) {
            const p = 4 * (y * image.width + x);
            const [r, g, b] = image.data.slice(p, p + 3);
            if ((r < 30 && g < 30 && b < 30) || (Math.abs(r - 147) < 4 && Math.abs(g - 149) < 4 && Math.abs(b - 152) < 4)) {
                n++;
                assert.equal(mask[y * image.width + x], 1, `${x},${y}`);
                assert.equal(classes[y * image.width + x], CLASS_OTHER);
            }
        }
    }
    assert.ok(n > 3000, `${n} line pixels checked`);
});

test('nearestTransform finds a nearest mask pixel (same distance as brute force)', async () => {
    const { nearestTransform } = await import('../src/core/layers.js');
    const rng = createRng(12);
    for (const [w, h, density] of [[23, 17, 0.04], [9, 31, 0.2], [6, 6, 0]]) {
        const mask = new Uint8Array(w * h);
        for (let p = 0; p < w * h; p++)
            mask[p] = rng.uniform() < density ? 1 : 0;
        const near = nearestTransform(mask, w, h);
        const d2 = distanceTransform(mask, w, h);
        for (let p = 0; p < w * h; p++) {
            if (d2[p] === Infinity) {
                assert.equal(near[p], -1);
                continue;
            }
            assert.equal(mask[near[p]], 1);
            const dx = near[p] % w - p % w;
            const dy = Math.floor(near[p] / w) - Math.floor(p / w);
            assert.equal(dx * dx + dy * dy, d2[p]);
        }
    }
});

test('glyph-sized strokes in a translucent family’s own color are masked; dots are not', () => {
    // JMP's 50% black dots share their color line with black text, whose strokes read as counts 1–8 and are too
    // short to be lines: the fit put dots on the tick labels and title
    const rng = createRng(6);
    const points = scatterPoints(rng, { n: 30, box: [30, 60, 370, 270] });
    // overlapping partners, so α can be measured
    for (const p of points.slice(0, 15))
        points.push({ x: p.x + 5, y: p.y + 3 });
    const strokes = [];
    for (let k = 0; k < 12; k++) {
        const x = 20 + 30 * k;
        strokes.push({ points: [[x, 15], [x, 27]], width: 1.6, color: [0, 0, 0] });
        strokes.push({ points: [[x + 3, 15], [x + 9, 27]], width: 1.6, color: [0, 0, 0] });
    }
    const spec = { width: 400, height: 300, background: [255, 255, 255], blur: 0.4, families: [{ color: [0, 0, 0], alpha: 0.5, radius: 8, points }], lines: strokes };
    const { image } = synthGraph(spec);
    const est = estimateStyle(image);
    near(est.style.alpha, 0.5, 0.02, 'α');
    const { mask } = buildMask(image, est);
    for (const s of strokes) {
        const [[x0, y0], [x1, y1]] = s.points;
        assert.equal(mask[Math.round((y0 + y1) / 2) * 400 + Math.round((x0 + x1) / 2)], 1, `stroke at ${x0}`);
    }
    for (const p of points)
        assert.equal(mask[Math.floor(p.y) * 400 + Math.floor(p.x)], 0, `dot at ${p.x}, ${p.y}`);
});

test('bounds: pixels more than a dot radius beyond the axes are masked, nothing nearer changes', () => {
    const { image } = synthGraph(SYNTH_PRESETS.sparse(1));
    const est = estimateStyle(image);
    const r = est.style.radius;
    const bounds = { left: 60.5, right: null, top: null, bottom: image.height - 40.5 };
    const plain = buildMask(image, est);
    const bounded = buildMask(image, est, { bounds });
    const { width, height } = image;
    let outside = 0;
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const p = y * width + x;
            const out = x + 0.5 < bounds.left - r || y + 0.5 > bounds.bottom + r;
            outside += out;
            assert.equal(bounded.mask[p], out ? 1 : plain.mask[p], `${x},${y}`);
        }
    }
    assert.equal(bounded.stats.outside, outside);
});

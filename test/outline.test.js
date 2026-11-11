import { test } from 'node:test';
import assert from 'node:assert/strict';
import { existsSync } from 'node:fs';
import { pixelCoverage } from '../src/core/render.js';
import { createOutlineModel, fillPlane, findOutlines } from '../src/core/outline.js';
import { estimateFamilies } from '../src/core/families.js';
import { matchDots } from '../src/core/metrics.js';
import { createRng } from '../src/core/rng.js';
import { readImage } from '../tools/png.mjs';
import { readTruthCsv, truthPixelsByTicks } from '../tools/truth-csv.mjs';
import { detectAxes } from '../src/core/axes.js';
import { extractAll, jitterRow } from '../tools/bench.mjs';

const WHITE = [255, 255, 255];
const W = 360;
const H = 240;
const RADIUS = 8; // px, the ring's outer edge
const RING_WIDTH = 1.5; // px
const RING_COLOR = [40, 40, 40];
const RING_ALPHA = 0.6;
const FILL_ALPHA = 0.75;
const GRID_GRAY = 234; // a gridline under the dots
const GRID_STEP = 60; // px

// a yellow-to-red scale on a planar curve through white (each stop a mix of two offsets from white)
function scaleColor(s) {
    const a = [-5, -40, -120];
    const b = [-60, -200, -160];
    return WHITE.map((v, ch) => v + (1 - s) * a[ch] + s * b[ch]);
}

// Outlined dots in drawing order: each a fill disc to the ring's middle, then a dark ring, both translucent, over
// white with gridlines; exact area coverage, no blur
function outlinedImage(dots, ringColor = RING_COLOR) {
    const img = new Float32Array(3 * W * H);
    for (let y = 0; y < H; y++) {
        for (let x = 0; x < W; x++) {
            const g = x % GRID_STEP < 2 || y % GRID_STEP < 2 ? GRID_GRAY : 255;
            img.fill(g, 3 * (y * W + x), 3 * (y * W + x) + 3);
        }
    }
    const inner = RADIUS - RING_WIDTH;
    const mid = RADIUS - RING_WIDTH / 2;
    for (const d of dots) {
        const c = scaleColor(d.s);
        for (let y = Math.floor(d.y - RADIUS - 1); y <= d.y + RADIUS + 1; y++) {
            for (let x = Math.floor(d.x - RADIUS - 1); x <= d.x + RADIUS + 1; x++) {
                if (x < 0 || y < 0 || x >= W || y >= H)
                    continue;
                const mf = FILL_ALPHA * pixelCoverage(d.x, d.y, mid, x, y);
                const mr = RING_ALPHA * (pixelCoverage(d.x, d.y, RADIUS, x, y) - pixelCoverage(d.x, d.y, inner, x, y));
                for (let ch = 0; ch < 3; ch++) {
                    const p = 3 * (y * W + x) + ch;
                    img[p] = img[p] + mf * (c[ch] - img[p]);
                    img[p] = img[p] + mr * (ringColor[ch] - img[p]);
                }
            }
        }
    }
    const data = new Uint8ClampedArray(4 * W * H);
    for (let p = 0; p < W * H; p++) {
        for (let ch = 0; ch < 3; ch++)
            data[4 * p + ch] = Math.round(img[3 * p + ch]);
        data[4 * p + 3] = 255;
    }
    return { width: W, height: H, data };
}

// isolated dots across the scale, and clusters where dots overlap and hide each other's rings
function scene() {
    const rng = createRng(7);
    const dots = [];
    for (let i = 0; i < 12; i++)
        dots.push({ x: 25 + i * 28, y: 30, s: i / 11 });
    for (let k = 0; k < 6; k++) {
        const cx = 40 + k * 55;
        const cy = 110 + (k % 2) * 60;
        for (let j = 0; j < 5; j++)
            dots.push({ x: cx + 14 * (rng.uniform() - 0.5) * 2, y: cy + 14 * (rng.uniform() - 0.5) * 2, s: rng.uniform() });
    }
    return dots;
}

test('createOutlineModel: a later fill dims the ring under it; drawn first, it is the one dimmed', () => {
    const base = new Float32Array(40 * 20);
    const style = { radius: 6, ring: { inner: 0.75, alpha: 0.5 }, fillAlpha: 0.8, ringT: 100 };
    // pixels 8 and 19 are both 5.5 px from (14, 10), on its ring (4.5–6 px); 19 is inside the fill of a dot at
    // (22, 10) (5.25 px around it), 8 is clear of it
    const at = (m, x) => m.image.data[10 * 40 + x];
    const under = createOutlineModel(base, 40, 20, style);
    under.add(14, 10);
    const b = under.add(22, 10); // drawn later, on top
    const free = at(under, 8);
    assert.ok(free > 20, `free ring ${free}`);
    assert.ok(Math.abs(at(under, 19) - (1 - 0.8) * free) < 1e-4, `under the later fill ${at(under, 19)}, free ${free}`);
    const over = createOutlineModel(base, 40, 20, style);
    over.add(22, 10, 1);
    over.add(14, 10, 2); // drawn later this time: over the other's fill (value 0), its ring reads as in the open
    assert.ok(Math.abs(at(over, 19) - free) < 1e-4, `on top ${at(over, 19)}`);
    // snapshot and restore undo a fill value; pixel 23 is inside the later dot's fill only
    const snap = under.snapshot(under.footprint(22, 10));
    under.setFill(b, 30);
    assert.ok(Math.abs(at(under, 23) - 0.8 * 30) < 1e-4, `fill value ${at(under, 23)}`);
    under.restore(snap);
    assert.ok(Math.abs(at(under, 23)) < 1e-4, 'restored');
});

test('findOutlines: the fill plane, the ring and αf of outlined dots whose fills vary', () => {
    const image = outlinedImage(scene());
    const plane = fillPlane(image, WHITE);
    assert.ok(plane, 'a fill plane');
    assert.ok(plane.rms < 1.5, `plane rms ${plane.rms}`);
    const found = findOutlines(image, WHITE);
    assert.ok(found, 'outlined dots');
    // the ring in P: K is black, the ring color (40, 40, 40) at α 0.6 is 0.6 · (1 − 40/255 off the plane's scale)
    assert.ok(Math.abs(found.ring.radius - RADIUS) <= 0.25, `radius ${found.ring.radius}`);
    assert.ok(Math.abs(found.ring.radius * (1 - found.ring.inner) - RING_WIDTH) <= 0.4, `ring width ${found.ring.radius * (1 - found.ring.inner)}`);
    assert.ok(Math.abs(found.fillAlpha - FILL_ALPHA) <= 0.1, `fill α ${found.fillAlpha} from ${found.pairs} pairs`);
});

test('outlined dots whose fills vary are fitted by their rings, in drawing order', () => {
    const dots = scene();
    const image = outlinedImage(dots);
    const { estimates } = estimateFamilies(image);
    assert.equal(estimates.length, 1);
    assert.equal(estimates[0].scale, 'outline');
    const results = extractAll(image);
    const fit = results[0].fit.dots;
    const m = matchDots(dots, fit, RADIUS / 2);
    // clusters of 5 hide some rings almost entirely under two or three fills (0.25 or less of their ink shows)
    assert.ok(m.recall >= 0.9, `recall ${m.recall} (${fit.length} found of ${dots.length})`);
    assert.ok(m.precision >= 0.95, `precision ${m.precision}`);
});

test('findOutlines: fills without a dark ring are not outlined dots', () => {
    // the same scene with white rings: the fills still lie in their plane, but nothing in P is a ring
    const image = outlinedImage(scene(), WHITE);
    assert.ok(fillPlane(image, WHITE), 'the fill plane is still found');
    assert.equal(findOutlines(image, WHITE), null);
});

test('truthPixelsByTicks: values through detected ticks, interpolated in log(v + 1)', () => {
    const truth = {
        xScale: { type: 'linear', jitter: true }, yScale: { type: 'log y+1', jitter: false },
        xTicks: [0, 10], yTicks: [0, 1, 3],
        rows: [{ x: 5, y: 2 }, { x: 10, y: 0 }],
    };
    // y ticks drawn at log(v + 1): 0 at 100 px, 1 at 100 − 40·ln 2, 3 at 100 − 40·ln 4
    const axes = { x: { ticks: [{ pos: 10 }, { pos: 110 }] }, y: { ticks: [{ pos: 100 - 40 * Math.log(4) }, { pos: 100 - 40 * Math.log(2) }, { pos: 100 }] } };
    const r = truthPixelsByTicks(truth, axes);
    assert.equal(r.jitter, true);
    assert.ok(Math.abs(r.dots[0].x - 60) < 1e-9);
    assert.ok(Math.abs(r.dots[0].y - (100 - 40 * Math.log(3))) < 1e-9, `y ${r.dots[0].y}`);
    assert.ok(Math.abs(r.dots[1].y - 100) < 1e-9);
});

const PATHOGEN = new URL('../testgraphs/pathogen-trade-time-scatter', import.meta.url).pathname;

test('regression: pathogen, outlined dots colored by a scale, found by their rings', { skip: !existsSync(PATHOGEN + '.csv') && 'pathogen truth absent' }, () => {
    // before: "No dot family found" (the fills vary by dot, and no color line held enough of them)
    const image = readImage(PATHOGEN + '.png');
    const px = truthPixelsByTicks(readTruthCsv(PATHOGEN + '.csv'), detectAxes(image));
    const r = jitterRow('pathogen', px.dots, extractAll(image), 0);
    // 583 dots, 286 of them in the jittered band at 0, up to 41 to a cell: many are under two or more fills
    assert.ok(r.found >= 440 && r.found <= 500, `found ${r.found}`);
    assert.ok(r.precision >= 0.95, `precision ${r.precision}`);
    assert.ok(r.recall >= 0.75, `recall ${r.recall}`);
});

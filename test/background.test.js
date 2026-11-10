import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createDotModel, solidImage, toRgba } from '../src/core/render.js';
import { flatMask } from '../src/core/style.js';
import { backgroundRegions, removeOverlays } from '../src/core/background.js';

const near = (a, b, tol, what) => assert.ok(Math.abs(a - b) <= tol, `${what}: ${a} vs ${b} (±${tol})`);

const WHITE = [255, 255, 255];
const ORANGE = { color: [230, 116, 7], alpha: 0.5, radius: 8 };
const BAND = [120, 140, 200]; // the layer's own color
const BAND_ALPHA = 0.25;
const RECT = [60, 0, 140, 80]; // x0, y0, x1, y1: the band
const W = 200;
const H = 80;

// a grid of orange dots, some inside the band and some outside
function dots() {
    const out = [];
    for (let y = 14; y < H - 10; y += 26) {
        for (let x = 14; x < W - 10; x += 22)
            out.push({ x, y });
    }
    return out;
}

function inRect(x, y) {
    return x >= RECT[0] && x < RECT[2] && y >= RECT[1] && y < RECT[3];
}

// the band composited over everything drawn before it
function overlay(im) {
    for (let y = 0; y < H; y++) {
        for (let x = 0; x < W; x++) {
            if (!inRect(x, y))
                continue;
            for (let ch = 0; ch < 3; ch++)
                im.data[3 * (y * W + x) + ch] = BAND[ch] * BAND_ALPHA + im.data[3 * (y * W + x) + ch] * (1 - BAND_ALPHA);
        }
    }
    return im;
}

const bandOver = WHITE.map((v, ch) => BAND[ch] * BAND_ALPHA + v * (1 - BAND_ALPHA));

function quantize(rgb) {
    const im = toRgba(rgb);
    return { ...im, data: Uint8ClampedArray.from(im.data, v => Math.round(v)) };
}

test('a band drawn over the dots is an overlay; dividing it out gives back the dots as drawn on white', () => {
    const model = createDotModel(solidImage(W, H, WHITE), ORANGE);
    model.addAll(dots());
    const clean = quantize(model.image);
    const drawn = quantize(overlay(model.image));
    const layers = backgroundRegions(drawn, WHITE, [ORANGE], flatMask(drawn), ORANGE.radius);
    assert.equal(layers.regions.length, 1);
    const r = layers.regions[0];
    assert.equal(r.mode, 'over');
    r.color.forEach((v, ch) => near(v, bandOver[ch], 1, 'band color'));
    near(r.s, 1 - BAND_ALPHA, 0.02, 's');
    const fixed = removeOverlays(drawn, WHITE, layers);
    // inside the band, away from its edges, the dots look as they would on white
    let worst = 0;
    for (let y = RECT[1] + 2; y < RECT[3] - 2; y++) {
        for (let x = RECT[0] + 2; x < RECT[2] - 2; x++) {
            for (let ch = 0; ch < 3; ch++)
                worst = Math.max(worst, Math.abs(fixed.data[4 * (y * W + x) + ch] - clean.data[4 * (y * W + x) + ch]));
        }
    }
    assert.ok(worst <= 3, `largest difference inside the band ${worst}`);
});

test('a band drawn under the dots is under them, and is left as it is', () => {
    const base = solidImage(W, H, WHITE);
    for (let y = 0; y < H; y++) {
        for (let x = 0; x < W; x++) {
            if (inRect(x, y))
                base.data.set(bandOver, 3 * (y * W + x));
        }
    }
    const model = createDotModel(base, ORANGE);
    model.addAll(dots());
    const drawn = quantize(model.image);
    const layers = backgroundRegions(drawn, WHITE, [ORANGE], flatMask(drawn), ORANGE.radius);
    assert.equal(layers.regions.length, 1);
    assert.equal(layers.regions[0].mode, 'under');
    // a dot inside the band belongs to the band's region; one outside to the main background
    assert.equal(layers.label[36 * W + 102], 1);
    assert.equal(layers.label[36 * W + 14], 0);
    assert.deepEqual(removeOverlays(drawn, WHITE, layers).data, drawn.data);
});

test('a thick stroke is not a region (dog-ages: its black curve and text)', () => {
    const im = solidImage(W, H, WHITE);
    for (const y0 of [10, 34, 58]) {
        for (let y = y0; y < y0 + 6; y++) {
            for (let x = 0; x < W; x++)
                im.data.set([0, 0, 0], 3 * (y * W + x));
        }
    }
    const drawn = quantize(im);
    let flat = 0;
    for (const v of flatMask(drawn))
        flat += v;
    assert.ok(flat >= 1500, `enough flat pixels for a region by count alone: ${flat}`);
    // strokes 6 px wide: narrower than the opening's disc (1.5 dot radii across)
    assert.equal(backgroundRegions(drawn, WHITE, [ORANGE], flatMask(drawn), ORANGE.radius).regions.length, 0);
});

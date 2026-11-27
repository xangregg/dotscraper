import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createDotModel, solidImage, toRgba } from '../src/core/render.js';
import { flatMask } from '../src/core/scan.js';
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

test('two bands drawn over the dots, one over the other: the upper is a layer by their overlap, which is divided out to the dots on white', () => {
    // red band, then blue band over it, both over the dots (step1's ribbons); the overlap is a color of its own
    const bands = [{ rect: [30, 0, 130, 80], color: [220, 60, 40], alpha: 0.4 }, { rect: [90, 0, 180, 80], color: [40, 110, 210], alpha: 0.35 }];
    const model = createDotModel(solidImage(W, H, WHITE), ORANGE);
    model.addAll(dots());
    const clean = quantize(model.image);
    const im = model.image;
    for (const { rect, color, alpha } of bands) {
        for (let y = rect[1]; y < rect[3]; y++) {
            for (let x = rect[0]; x < rect[2]; x++) {
                for (let ch = 0; ch < 3; ch++)
                    im.data[3 * (y * W + x) + ch] = color[ch] * alpha + im.data[3 * (y * W + x) + ch] * (1 - alpha);
            }
        }
    }
    const drawn = quantize(im);
    const layers = backgroundRegions(drawn, WHITE, [ORANGE], flatMask(drawn), ORANGE.radius);
    const red = layers.regions.findIndex(r => r.color[0] > 200 && !r.stack);
    const blue = layers.regions.findIndex(r => r.color[2] > 200 && r.color[0] < 200 && !r.stack);
    assert.ok(red >= 0 && blue >= 0, JSON.stringify(layers.regions.map(r => r.color)));
    assert.equal(layers.regions[blue].mode, 'over');
    near(layers.regions[blue].s, 0.65, 0.03, 'blue band s (from the overlap)');
    assert.equal(layers.regions[blue].stackedOver, red);
    const overlap = layers.regions.find(r => r.stack);
    assert.ok(overlap, 'the overlap is a region');
    assert.deepEqual(overlap.stack, [blue, red]);
    near(overlap.s, 0.6 * 0.65, 0.03, 'overlap s');
    const fixed = removeOverlays(drawn, WHITE, layers);
    // inside the overlap the dots look as they would on white (x 92–112: the dots at x 102, wholly inside it; the
    // ones at x 124 straddle the red band's edge at 130 and are split along it, as background.js says)
    let worst = 0;
    for (let y = 2; y < H - 2; y++) {
        for (let x = 92; x < 112; x++) {
            for (let ch = 0; ch < 3; ch++)
                worst = Math.max(worst, Math.abs(fixed.data[4 * (y * W + x) + ch] - clean.data[4 * (y * W + x) + ch]));
        }
    }
    assert.ok(worst <= 6, `largest difference inside the overlap ${worst}`);
});

test('a box under the dots holds its rectangle where a dot as wide as it is tall hides its pieces', () => {
    // protrusion1: boxes 37 px tall, dots 39 px across; pixels in such a dot nearer the page above or below than the
    // box's pieces beside it were read against the page
    const W = 300;
    const H = 120;
    const fam = { color: [119, 214, 167], alpha: 0.155, radius: 19.5 };
    const box = [210, 241, 226];
    const bg = solidImage(W, H, WHITE);
    for (let y = 42; y < 79; y++) {
        for (let x = 40; x < 260; x++)
            bg.data.set(box, 3 * (y * W + x));
    }
    const model = createDotModel(bg, fam);
    model.addAll([{ x: 150, y: 60.5 }, { x: 20, y: 20 }, { x: 280, y: 100 }]);
    const image = toRgba(model.image);
    const { regions, label } = backgroundRegions(image, WHITE, [{ ...fam, band: box, box: true }], flatMask(image), fam.radius);
    assert.equal(regions.length, 1);
    // inside the dot at the box's top and bottom rows, and outside the box above and below the dot
    for (const [x, y] of [[150, 43], [150, 77], [140, 44]])
        assert.equal(label[y * W + x], 1, `(${x}, ${y}) in the box`);
    for (const [x, y] of [[150, 41], [150, 79]])
        assert.equal(label[y * W + x], 0, `(${x}, ${y}) outside`);
});

test('regression: a band over the dots narrower than a dot, a line along it: the dots under it are divided out all the same (lung-tumor)', () => {
    // a band 8 px across at its waist with a line 4 px wide through it: strips 2 px wide, which are no seeds, so the
    // dots under them took the page's label and stayed as they were (734 dot pixels off, every one of the 160 under
    // the waist); and a pixel at the band's edge was left alone though its color says it is under the band
    const W = 420;
    const H = 120;
    const fam = { color: [71, 116, 164], alpha: 0.44, radius: 7 };
    const band = [179, 58, 45];
    const bandAlpha = 0.15;
    const half = x => 4 + 26 * Math.max(0, Math.abs(x - W / 2) - 70) / (W / 2 - 70);
    const inBand = (x, y) => Math.abs(y + 0.5 - H / 2) <= half(x + 0.5);
    const onLine = y => Math.abs(y + 0.5 - H / 2) < 2;
    const centers = [];
    for (let x = 14; x < W - 10; x += 19) {
        for (const y of [22, 41, 60, 79, 98])
            centers.push({ x: x + (y % 3), y: y + (x % 5) - 2 });
    }
    const model = createDotModel(solidImage(W, H, WHITE), fam);
    model.addAll(centers);
    const clean = quantize(model.image);
    for (let y = 0; y < H; y++) {
        for (let x = 0; x < W; x++) {
            if (!inBand(x, y))
                continue;
            for (let ch = 0; ch < 3; ch++)
                model.image.data[3 * (y * W + x) + ch] = onLine(y) ? band[ch] : band[ch] * bandAlpha + model.image.data[3 * (y * W + x) + ch] * (1 - bandAlpha);
        }
    }
    const drawn = quantize(model.image);
    const layers = backgroundRegions(drawn, WHITE, [fam], flatMask(drawn), fam.radius);
    assert.equal(layers.regions.length, 1);
    assert.equal(layers.regions[0].mode, 'over');
    const fixed = removeOverlays(drawn, WHITE, layers);
    // dot pixels off the line that differ from the dots as drawn on white by more than 6 units
    let off = 0;
    let offWaist = 0;
    for (let y = 0; y < H; y++) {
        for (let x = 0; x < W; x++) {
            const p = y * W + x;
            if (clean.data[4 * p] > 250 || onLine(y))
                continue;
            if (![0, 1, 2].some(ch => Math.abs(fixed.data[4 * p + ch] - clean.data[4 * p + ch]) > 6))
                continue;
            off++;
            if (Math.abs(x - W / 2) < 60 && inBand(x, y))
                offWaist++;
        }
    }
    assert.ok(off <= 40 && offWaist <= 20, `${off} dot pixels off, ${offWaist} under the waist`);
});

test('regression: ink a little off a dark region\'s color is no dot under a layer that lets a twentieth through (crimson)', () => {
    // crimson: a dark gray of 35 with 28 flat pixels of 23 to 26 in it, read as black dots under a layer with s = 0.04
    // and divided out
    const W = 200;
    const H = 100;
    const red = { color: [146, 29, 26], alpha: 1, radius: 8 };
    const dark = [35, 35, 35];
    const im = solidImage(W, H, WHITE);
    for (let y = 20; y < 80; y++) {
        for (let x = 20; x < 120; x++)
            im.data.set(dark, 3 * (y * W + x));
    }
    // patches a twentieth of the way from the region's color along the dots' offset from the page
    const ink = dark.map((v, ch) => v + 0.05 * (red.color[ch] - WHITE[ch]));
    for (const [x0, y0] of [[30, 30], [60, 40], [90, 60]]) {
        for (let y = y0; y < y0 + 6; y++) {
            for (let x = x0; x < x0 + 6; x++)
                im.data.set(ink, 3 * (y * W + x));
        }
    }
    const model = createDotModel(im, red);
    model.addAll([{ x: 150, y: 30 }, { x: 170, y: 70 }]);
    const drawn = quantize(model.image);
    const { regions } = backgroundRegions(drawn, WHITE, [red], flatMask(drawn), red.radius);
    assert.equal(regions.length, 1);
    assert.ok(regions[0].votes.over >= 20, JSON.stringify(regions[0].votes));
    assert.equal(regions[0].mode, 'unknown');
});

test('regression: two families\' own bands, one over the other: the overlap is a third region under the dots (bread2)', () => {
    // before: the upper band was a layer over everything by the overlap's color, whose s (0.78 for 0.85) undid the
    // dots under it a tenth too deep; then, in the app alone, when the dots' votes on the upper band weren't "under"
    const PANEL = [229, 229, 229];
    const W2 = 600;
    const H2 = 300;
    const blue = { color: [52, 137, 188], alpha: 0.5, band: [202, 214, 222] };
    const red = { color: [225, 74, 52], alpha: 0.5, band: [228, 205, 201] };
    // (as measured there: 2 units lighter than the blue band over the red at the 0.85 it lets through)
    const overlap = [203, 196, 200];
    const im = solidImage(W2, H2, PANEL);
    const bg = (x, y) => (y < 60 ? PANEL : x < 260 ? blue.band : x < 400 ? overlap : red.band);
    for (let y = 0; y < H2; y++) {
        for (let x = 0; x < W2; x++)
            im.data.set(bg(x, y), 3 * (y * W2 + x));
    }
    // each family's dots over the overlap and the red band, drawn over them; none over the blue band, which the dots'
    // votes then leave unknown (and a band in the dots' own hue can't be told by them: bread2's was under by 1887 to
    // 129 as Node decodes the JPEG, over by 1194 to 840 as Chrome does)
    for (const [f, y0] of [[blue, 110], [red, 200]]) {
        for (let k = 5; k < 11; k++) {
            const cx = 40 + 52 * k;
            for (let y = y0 - 11; y <= y0 + 11; y++) {
                for (let x = cx - 11; x <= cx + 11; x++) {
                    if (Math.hypot(x + 0.5 - cx, y + 0.5 - y0) <= 10)
                        im.data.set(bg(x, y).map((v, ch) => f.color[ch] + (v - f.color[ch]) * 0.5), 3 * (y * W2 + x));
                }
            }
        }
    }
    const image = toRgba(im);
    const layers = backgroundRegions(image, PANEL, [blue, red], flatMask(image), 10);
    assert.deepEqual(layers.regions.map(r => r.mode), ['under', 'under', 'under'], JSON.stringify(layers.regions.map(r => [r.color.map(Math.round), r.mode, r.votes])));
    assert.ok(layers.regions.some(r => r.color.every((v, ch) => Math.abs(v - overlap[ch]) <= 2)), 'the overlap is a region');
});

test('regression: in a gray theme, a gray band over black dots is a layer, not the dots under the white margin (gg-loess)', () => {
    // Every gray lies on the line from the white page margin toward black dots, so the band was read as black dots
    // under a white "layer" and dropped. Its pixels lie far from the margin; a dot under a layer is surrounded by it.
    const W2 = 240;
    const H2 = 140;
    const PANEL = [230, 230, 230];
    const black = { color: [0, 0, 0], alpha: 1, radius: 6 };
    const band = { color: [153, 153, 153], alpha: 0.4, y0: 50, y1: 90 };
    const im = solidImage(W2, H2, WHITE);
    for (let y = 20; y < 120; y++) {
        for (let x = 30; x < 210; x++)
            im.data.set(PANEL, 3 * (y * W2 + x));
    }
    const model = createDotModel(im, black);
    const at = [];
    for (let y = 32; y < 112; y += 19) {
        for (let x = 44; x < 200; x += 31)
            at.push({ x, y });
    }
    model.addAll(at);
    const out = model.image;
    for (let y = band.y0; y < band.y1; y++) {
        for (let x = 30; x < 210; x++) {
            for (let ch = 0; ch < 3; ch++)
                out.data[3 * (y * W2 + x) + ch] = band.color[ch] * band.alpha + out.data[3 * (y * W2 + x) + ch] * (1 - band.alpha);
        }
    }
    const drawn = quantize(out);
    const over = PANEL.map((v, ch) => band.color[ch] * band.alpha + v * (1 - band.alpha));
    const isBand = r => r.color.every((v, ch) => Math.abs(v - over[ch]) <= 1);
    const { regions } = backgroundRegions(drawn, PANEL, [black], flatMask(drawn), black.radius, { page: true });
    const found = regions.find(isBand);
    assert.ok(found, JSON.stringify(regions.map(r => [r.color, r.mode])));
    assert.equal(found.mode, 'over');
    near(found.s, 1 - band.alpha, 0.03, 's');
    // told from the dots' own background, a color off the regions found stays the dots' (violin-beeswarm)
    assert.ok(!backgroundRegions(drawn, PANEL, [black], flatMask(drawn), black.radius).regions.some(isBand), 'not from the page');
});

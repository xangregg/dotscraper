import { test } from 'node:test';
import assert from 'node:assert/strict';
import { existsSync } from 'node:fs';
import { createDotModel, solidImage, toRgba } from '../src/core/render.js';
import { flatMask, EXACT_TOLERANCES, JPEG_TOLERANCES, backgroundColor } from '../src/core/style.js';
import { findColormap, rayImage, opaqueImage } from '../src/core/colormap.js';
import { readImage } from '../tools/png.mjs';
import { readTruthCsv, truthPixels, truthPixelsByTicks } from '../tools/truth-csv.mjs';
import { extractAll, groupRow } from '../tools/bench.mjs';
import { detectAxes } from '../src/core/axes.js';
import { matchDots } from '../src/core/metrics.js';
import { restoreChroma } from '../src/core/jpeg.js';
import { createRng } from '../src/core/rng.js';

const WHITE = [255, 255, 255];
const ALPHA = 0.5;
const RADIUS = 8;
const W = 300;
const H = 120;

// a yellow-to-purple scale, like JMP's "Muted Yellow to Red"
function scaleColor(s) {
    const stops = [[240, 220, 120], [220, 120, 70], [150, 30, 80]];
    const k = Math.min(1, Math.floor(s * 2));
    const f = s * 2 - k;
    return stops[k].map((v, ch) => v + (stops[k + 1][ch] - v) * f);
}

// dots drawn one at a time in order, each its own color; pairs overlap so the second level shows
function scaleImage() {
    let im = solidImage(W, H, WHITE);
    const dots = [];
    for (let i = 0; i < 24; i++) {
        const x = 20 + (i % 12) * 23;
        const y = i < 12 ? 30 : 85;
        const s = i / 23;
        dots.push({ x, y, s });
        if (i % 3 === 0)
            dots.push({ x: x + 6, y: y + 5, s });
    }
    for (const d of dots) {
        const model = createDotModel(im, { color: scaleColor(d.s), alpha: ALPHA, radius: RADIUS });
        model.add(d.x, d.y);
        im = model.image;
    }
    const rgba = toRgba(im);
    return { image: { ...rgba, data: Uint8ClampedArray.from(rgba.data, v => Math.round(v)) }, dots };
}

test('findColormap: dots of a continuous scale give one curve and their α', () => {
    const { image } = scaleImage();
    const cm = findColormap(image, WHITE, flatMask(image), EXACT_TOLERANCES);
    assert.ok(cm, 'a color scale');
    assert.ok(Math.abs(cm.alpha - ALPHA) <= 0.03, `α ${cm.alpha}`);
    // on the virtual image every lone dot is the same color, and a lens of two is level 2 − α
    const { image: v, P } = rayImage(image, WHITE, cm);
    const at = (x, y) => [...v.data.subarray(4 * (y * W + x), 4 * (y * W + x) + 3)];
    for (const x of [43, 135, 250]) // dots 1, 5 and 10 of the first row, not overlapped
        at(x, 30).forEach((c, ch) => assert.ok(Math.abs(c - P[ch]) <= 3, `lone dot at ${x}: ${at(x, 30)} vs ${P}`));
    const lens = at(23, 32);
    const level2 = WHITE.map((b, ch) => b + (2 - ALPHA) * (P[ch] - b));
    lens.forEach((c, ch) => assert.ok(Math.abs(c - level2[ch]) <= 4, `lens ${lens} vs ${level2}`));
});

// A categorical palette (tab10-like, as the wages2025 test image's), translucent, overlapping at random, and one color
// far in direction from the rest with few dots (wages2025's pink: no chain reaches it)
const PALETTE = [[31, 119, 180], [44, 160, 44], [214, 39, 40], [255, 127, 14], [148, 103, 189], [140, 86, 75],
    [127, 127, 127], [188, 189, 34], [23, 190, 207], [57, 59, 121], [99, 121, 57], [189, 158, 57]];
const PINK = [227, 119, 194];
const PALETTE_RADIUS = 17.6; // px

function paletteImage() {
    const rng = createRng(7);
    const width = 900;
    const height = 520;
    const dots = [];
    // dense and at wages2025's scale (r 17.6): sparse or small dots of a palette leave few mixes to chain through,
    // and are families found one at a time
    PALETTE.forEach((color, family) => {
        for (let i = 0; i < 20; i++)
            dots.push({ x: 80 + rng.uniform() * 740, y: 40 + rng.uniform() * 380, family, color });
    });
    for (let i = 0; i < 4; i++)
        dots.push({ x: 120 + 220 * i, y: 480, family: PALETTE.length, color: PINK });
    let im = solidImage(width, height, WHITE);
    for (const d of dots) {
        const model = createDotModel(im, { color: d.color, alpha: 0.6, radius: PALETTE_RADIUS });
        model.add(d.x, d.y);
        im = model.image;
    }
    const rgba = toRgba(im);
    return { image: { ...rgba, data: Uint8ClampedArray.from(rgba.data, v => Math.round(v)) }, dots };
}

test('findColormap: a categorical palette is told from a scale; its dots split by their own colors', () => {
    // before: one family "colored by a continuous scale", and no pink (a palette's colors chain through overlaps of
    // unlike colors, but its votes are in peaks; pink links to none)
    const { image, dots } = paletteImage();
    const cm = findColormap(image, WHITE, flatMask(image), EXACT_TOLERANCES);
    assert.ok(cm?.palette, 'a palette');
    const scale = findColormap(scaleImage().image, WHITE, flatMask(scaleImage().image), EXACT_TOLERANCES);
    assert.equal(scale.palette, false, 'a scale stays a scale');
    const results = extractAll(image);
    assert.equal(results.length, PALETTE.length + 1, results.map(r => r.est.style.color.map(Math.round).join(',')).join('; '));
    // measured: 188 of 244 found in their own color's family (the scene is denser than wages2025: a fifth of the dots
    // are mostly hidden), every color at least half; of 177 dots marked sure, none on another color's dot
    let matched = 0;
    for (const res of results) {
        // the family's one-dot color names its palette color
        const one = c => WHITE.map((b, ch) => b + 0.6 * (c[ch] - b));
        const k = [...PALETTE, PINK].findIndex(c => Math.hypot(...one(c).map((v, ch) => v - res.est.style.color[ch])) <= 6);
        assert.ok(k >= 0, `family color ${res.est.style.color.map(Math.round)}`);
        const want = dots.filter(d => d.family === k);
        const m = matchDots(want, res.fit.dots, 2);
        matched += m.pairs.length;
        assert.ok(m.recall >= 0.5, `color ${k}: ${m.pairs.length} of ${want.length}`);
        const others = dots.filter(d => d.family !== k);
        const wrong = res.fit.dots.filter(d => d.familySure && !want.some(t => Math.hypot(t.x - d.x, t.y - d.y) <= 2)
            && others.some(t => Math.hypot(t.x - d.x, t.y - d.y) <= 2));
        assert.equal(wrong.length, 0, `color ${k}: sure dots on other colors ${JSON.stringify(wrong.map(d => [Math.round(d.x), Math.round(d.y)]))}`);
    }
    assert.ok(matched >= 0.75 * dots.length, `${matched} of ${dots.length}`);
});

const WAGES = new URL('../testgraphs/wages2025', import.meta.url).pathname;

test('regression: wages2025, 22 occupation groups in a palette are families, each dot by its own pixels', { skip: !existsSync(WAGES + '.png') && 'testgraphs wages2025 absent' }, () => {
    // before: one family "colored by a continuous scale" (P 0.860 R 0.845 pooled); split one family at a time, 9
    // families in 6 minutes; by palette, 38 colors (stacks, gridline grays) until the colors had to be uniform, 41
    // dots in the wrong family until each dot voted by its exact pixels, and no pink (8 dots, outside the chain).
    // Measured: 23 families (the 22, and the median line's purple), P 0.851 R 0.846 by group.
    const image = readImage(WAGES + '.png');
    const results = extractAll(image);
    assert.ok(results.length >= 22 && results.length <= 24, `${results.length} families`);
    const px = truthPixelsByTicks(readTruthCsv(WAGES + '.csv'), detectAxes(image));
    const r = groupRow('wages2025', px.dots, results, 0);
    assert.ok(r.precision >= 0.83 && r.recall >= 0.83, `precision ${r.precision}, recall ${r.recall}, found ${r.found}`);
});

test('findColormap: one translucent family is not a color scale', () => {
    let im = solidImage(W, H, WHITE);
    const model = createDotModel(im, { color: [220, 120, 70], alpha: ALPHA, radius: RADIUS });
    for (let i = 0; i < 12; i++)
        model.add(20 + i * 23, 40 + (i % 2) * 5);
    const rgba = toRgba(model.image);
    const image = { ...rgba, data: Uint8ClampedArray.from(rgba.data, v => Math.round(v)) };
    assert.equal(findColormap(image, WHITE, flatMask(image), EXACT_TOLERANCES), null);
});

const OUTLINE = 2; // px: outline width of the outlined variant
const OUTLINE_DARKEN = 0.75; // outline color = fill × this, toward black
const GAP_X = 28.75; // px: the middle of the 1 px gap between the last two dots, off the pixel grid
const GAP_Y = 105; // px

// opaque dots drawn in order, each its own color, in touching and overlapping clusters, one pair 1 px apart;
// with `outline`, each is an outline disc under a fill OUTLINE px smaller, as JMP draws outlined markers
function opaqueScaleImage(outline) {
    const centers = [];
    for (let i = 0; i < 24; i++)
        centers.push({ x: 20 + (i % 12) * 23 + (i % 2) * 6, y: (i < 12 ? 30 : 85) + (i % 3) * 5 });
    centers.push({ x: GAP_X - RADIUS - 0.5, y: GAP_Y }, { x: GAP_X + RADIUS + 0.5, y: GAP_Y });
    const dots = centers.map((c, i) => ({ ...c, s: i / (centers.length - 1) }));
    let im = solidImage(W, H, WHITE);
    for (const d of dots) {
        const fill = scaleColor(d.s);
        const layers = outline ? [[fill.map(v => v * OUTLINE_DARKEN), RADIUS], [fill, RADIUS - OUTLINE]] : [[fill, RADIUS]];
        for (const [color, radius] of layers) {
            const model = createDotModel(im, { color, alpha: 1, radius });
            model.add(d.x, d.y);
            im = model.image;
        }
    }
    const round = rgba => ({ ...rgba, data: Uint8ClampedArray.from(rgba.data, v => Math.round(v)) });
    const union = createDotModel(solidImage(W, H, WHITE), { color: [0, 0, 0], alpha: 1, radius: RADIUS });
    for (const d of dots)
        union.add(d.x, d.y);
    return { image: round(toRgba(im)), union: round(toRgba(union.image)) };
}

for (const outline of [false, true]) {
    test(`opaqueImage: opaque dots of a scale${outline ? ', outlined,' : ''} become the union of their discs`, () => {
        const { image, union } = opaqueScaleImage(outline);
        const cm = findColormap(image, WHITE, flatMask(image), EXACT_TOLERANCES);
        assert.ok(cm, 'a color scale');
        assert.equal(cm.alpha, null, 'opaque: no second level');
        const { image: v, P } = opaqueImage(image, WHITE, cm);
        // coverage of the virtual image along B → P, against the one-color rendering's
        const d = P.map((c, ch) => c - WHITE[ch]);
        const dd = d.reduce((sum, x) => sum + x * x, 0);
        let se = 0;
        let count = 0;
        let worst = 0;
        let worstGap = 0;
        for (let p = 0; p < W * H; p++) {
            let s = 0;
            for (let ch = 0; ch < 3; ch++)
                s += (v.data[4 * p + ch] - WHITE[ch]) * d[ch];
            const a = s / dd;
            const truth = 1 - union.data[4 * p] / 255;
            if (a === 0 && truth === 0)
                continue;
            se += (a - truth) ** 2;
            count++;
            worst = Math.max(worst, Math.abs(a - truth));
            const x = p % W;
            const y = Math.floor(p / W);
            if (Math.abs(x + 0.5 - GAP_X) <= 1 && Math.abs(y + 0.5 - GAP_Y) <= RADIUS / 2)
                worstGap = Math.max(worstGap, Math.abs(a - truth));
        }
        const rms = Math.sqrt(se / count);
        assert.ok(rms <= 0.03, `coverage rms ${rms.toFixed(4)} over ${count} pixels`);
        assert.ok(worst <= 0.5, `worst coverage error ${worst.toFixed(3)}`);
        // the gap between the pair has no background pixel, but is mostly uncovered
        assert.ok(worstGap <= 0.15, `worst coverage error in the gap ${worstGap.toFixed(3)}`);
    });
}

const dir = set => new URL(`../${set}/`, import.meta.url).pathname;
const absent = (set, name) => !existsSync(dir(set) + name) && `${set}/${name} absent`;

function row(set, name) {
    const truth = truthPixels(readTruthCsv(dir(set) + 'data.csv')).dots;
    const found = extractAll(readImage(dir(set) + name), name.endsWith('.jpg')).flatMap(r => r.fit.dots);
    return { found: found.length, m: matchDots(truth, found, 4.8) };
}

test('regression: dots colored by a scale are found (bench1 gradient-dots)', { skip: absent('bench1', 'gradient-dots.png') }, () => {
    // before: "The dot radius is unknown": no family's color line held more than a few dots
    const { found, m } = row('bench1', 'gradient-dots.png');
    assert.equal(found, 100);
    assert.equal(m.recall, 1);
});

test('regression: bench2 gradient-dots, with the scale’s darkest dot alone in its color', { skip: absent('bench2', 'gradient-dots.png') }, () => {
    // before: 199, the isolated darkest dot 0.116 from the next color off the curve; the 4 still missed are in
    // clusters of 4–6 dots of mixed colors
    const { found, m } = row('bench2', 'gradient-dots.png');
    assert.ok(found >= 199 && found <= 205, `found ${found}`);
    assert.ok(m.recall >= 0.975, `recall ${m.recall}`);
});

const OPAQUE_ROWS = [
    // before: 114 found, recall 0.82, as one opaque family of the most common fill
    ['bench1', 'gradient-opaque.png', 99, 102, 0.99],
    // before: "The dot radius is unknown"; bench2 solid-opaque, the same dots in one color: 196, 0.926
    ['bench2', 'gradient-opaque.png', 188, 200, 0.91],
    // the stroked renderings are fitted by their rings: outline.test.js
];

for (const [set, name, lo, hi, recall] of OPAQUE_ROWS) {
    test(`regression: opaque dots colored by a scale are found (${set} ${name})`, { skip: absent(set, name) }, () => {
        const { found, m } = row(set, name);
        assert.ok(found >= lo && found <= hi, `found ${found}`);
        assert.ok(m.recall >= recall, `recall ${m.recall}`);
    });
}

test('regression: a band’s edges are not an opaque color scale (bench1 smoother-ci.jpg)', { skip: absent('bench1', 'smoother-ci.jpg') }, () => {
    // its band edges chain to a spread of 0.215 with no second level; the opaque scales span 0.54–0.60
    const image = restoreChroma(readImage(dir('bench1') + 'smoother-ci.jpg'));
    const B = backgroundColor(image).color;
    assert.equal(findColormap(image, B, flatMask(image, JPEG_TOLERANCES), JPEG_TOLERANCES), null);
});

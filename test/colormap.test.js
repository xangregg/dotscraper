import { test } from 'node:test';
import assert from 'node:assert/strict';
import { existsSync } from 'node:fs';
import { createDotModel, solidImage, toRgba } from '../src/core/render.js';
import { flatMask, EXACT_TOLERANCES } from '../src/core/style.js';
import { findColormap, rayImage } from '../src/core/colormap.js';
import { readImage } from '../tools/png.mjs';
import { readTruthCsv, truthPixels } from '../tools/truth-csv.mjs';
import { extractAll } from '../tools/bench.mjs';
import { matchDots } from '../src/core/metrics.js';

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

test('findColormap: one translucent family is not a color scale', () => {
    let im = solidImage(W, H, WHITE);
    const model = createDotModel(im, { color: [220, 120, 70], alpha: ALPHA, radius: RADIUS });
    for (let i = 0; i < 12; i++)
        model.add(20 + i * 23, 40 + (i % 2) * 5);
    const rgba = toRgba(model.image);
    const image = { ...rgba, data: Uint8ClampedArray.from(rgba.data, v => Math.round(v)) };
    assert.equal(findColormap(image, WHITE, flatMask(image), EXACT_TOLERANCES), null);
});

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

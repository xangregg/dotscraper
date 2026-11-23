import { test } from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, readFileSync } from 'node:fs';
import { annotationBoxes } from '../src/core/annotations.js';
import { backgroundColors } from '../src/core/axes.js';
import { readImage } from '../tools/png.mjs';

const WHITE = [255, 255, 255];
const INK = [24, 37, 50];

// A canvas painted in order, supersampled 4 × 4: rectangles { rect: [x0, y0, x1, y1], color } and discs
// { x, y, r, color, alpha } (translucent over what is under them)
function paint(width, height, shapes, S = 4) {
    const W = width * S;
    const buf = new Float32Array(3 * W * height * S).fill(255);
    for (const sh of shapes) {
        const [x0, y0, x1, y1] = sh.rect ?? [sh.x - sh.r, sh.y - sh.r, sh.x + sh.r, sh.y + sh.r];
        const a = sh.alpha ?? 1;
        for (let Y = Math.max(0, Math.floor(y0 * S)); Y < Math.min(height * S, Math.ceil(y1 * S)); Y++) {
            for (let X = Math.max(0, Math.floor(x0 * S)); X < Math.min(W, Math.ceil(x1 * S)); X++) {
                if (!sh.rect && Math.hypot((X + 0.5) / S - sh.x, (Y + 0.5) / S - sh.y) > sh.r)
                    continue;
                for (let ch = 0; ch < 3; ch++) {
                    const q = 3 * (Y * W + X) + ch;
                    buf[q] = a * sh.color[ch] + (1 - a) * buf[q];
                }
            }
        }
    }
    const data = new Uint8ClampedArray(4 * width * height);
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            for (let ch = 0; ch < 3; ch++) {
                let sum = 0;
                for (let sy = 0; sy < S; sy++) {
                    for (let sx = 0; sx < S; sx++)
                        sum += buf[3 * ((y * S + sy) * W + x * S + sx) + ch];
                }
                data[4 * (y * width + x) + ch] = Math.round(sum / (S * S));
            }
            data[4 * (y * width + x) + 3] = 255;
        }
    }
    return { width, height, data };
}

// Block letters of strokes (w px wide, h px tall) on a baseline, left to right from x: E, H, L, T, I and F vary in
// shape, as letters do
function word(x, baseline, h, letters, w = 3, color = INK) {
    const shapes = [];
    const top = baseline - h;
    const cw = 0.6 * h;
    for (const ch of letters) {
        const r = (x0, y0, x1, y1) => shapes.push({ rect: [x + x0, y0, x + x1, y1], color });
        if (ch !== ' ') {
            if ('EHLFI'.includes(ch))
                r(0, top, w, baseline);
            if (ch === 'H')
                r(cw - w, top, cw, baseline), r(0, top + h / 2 - w / 2, cw, top + h / 2 + w / 2);
            if ('EF'.includes(ch))
                r(0, top, cw, top + w), r(0, top + h / 2 - w / 2, 0.8 * cw, top + h / 2 + w / 2);
            if ('EL'.includes(ch))
                r(0, baseline - w, cw, baseline);
            if (ch === 'T')
                r(0, top, cw, top + w), r(cw / 2 - w / 2, top, cw / 2 + w / 2, baseline);
        }
        x += ch === 'I' ? w + 5 : cw + 5;
    }
    return shapes;
}

const inside = (box, x, y) => x >= box[0] && x < box[2] && y >= box[1] && y < box[3];

test('annotationBoxes: lines of text and a legend column are boxed; dots, crescents, a row of rings and a frame aren\'t', () => {
    const R = 12;
    const shapes = [
        // a frame around the plot (thin for its height, as a glyph is: testgraphs violin-beeswarm's made the plot "text")
        { rect: [4, 4, 796, 6], color: INK }, { rect: [4, 314, 796, 316], color: INK }, { rect: [4, 4, 6, 316], color: INK }, { rect: [794, 4, 796, 316], color: INK },
        // a title, bold (strokes 5 px, thinner than the dots: 8 px, as thick as them, would be no text)
        ...word(20, 60, 44, 'THE TITLE', 5),
        // a note in the plot, a dot right beside it
        ...word(300, 160, 20, 'FILE HEFT'),
        { x: 285, y: 150, r: R, color: [214, 39, 40], alpha: 0.6 },
        // a legend: markers the dots' size, each before a word
        ...[0, 1, 2, 3].flatMap(k => [{ x: 620, y: 110 + 40 * k, r: R, color: [[31, 119, 180], [44, 160, 44], [255, 127, 14], [148, 103, 189]][k] },
            ...word(645, 110 + 40 * k + 10, 20, ['THE', 'FLIT', 'HEIL', 'LIFE'][k])]),
        // overlapping translucent dots with white rims: crescents
        ...[0, 1, 2, 3, 4, 5].flatMap(k => [{ x: 80 + 16 * k, y: 260, r: R + 1.5, color: WHITE, alpha: 0.9 }, { x: 80 + 16 * k, y: 260, r: R, color: [189, 158, 57], alpha: 0.6 }]),
        // a row of hollow rings, the dots' size
        ...[0, 1, 2, 3, 4].flatMap(k => [{ x: 320 + 30 * k, y: 260, r: R, color: INK }, { x: 320 + 30 * k, y: 260, r: R - 2.5, color: WHITE }]),
        // black dots touching in pairs, text's color, their bottoms in line: thin for their height, not for a dot
        // (bench-plos adults1's cluster)
        ...[0, 1, 2, 3].flatMap(k => [{ x: 480 + 28 * k, y: 240, r: R, color: INK }, { x: 480 + 28 * k + 4, y: 262, r: R, color: INK }]),
    ];
    const image = paint(800, 320, shapes);
    const { text, legend } = annotationBoxes(image, backgroundColors(image)[0], R);
    const covered = (x, y) => text.some(b => inside(b, x, y));
    assert.ok(covered(40, 40) && covered(200, 40), `the title, ${JSON.stringify(text)}`);
    assert.ok(covered(310, 150) && covered(380, 150), 'the note');
    for (const k of [0, 1, 2, 3])
        assert.ok(covered(660, 110 + 40 * k), `legend word ${k}`);
    assert.equal(legend.length, 4, JSON.stringify(legend));
    for (const k of [0, 1, 2, 3])
        assert.ok(legend.some(b => inside(b, 620, 110 + 40 * k)), `legend marker ${k}`);
    // the data
    assert.ok(!covered(285, 150) && !legend.some(b => inside(b, 285, 150)), 'the dot beside the note');
    for (let x = 70; x <= 170; x += 10)
        assert.ok(!covered(x, 260), `crescent at ${x}`);
    for (let x = 310; x <= 450; x += 10)
        assert.ok(!covered(x, 260), `ring at ${x}`);
    for (let x = 470; x <= 570; x += 10)
        assert.ok(!covered(x, 250), `touching dots at ${x}`);
    assert.ok(!covered(400, 290), 'the frame');
});

test('regression: bold text with strokes almost as thick as the dots is text where whole dots show their thickness (crimson)', () => {
    // crimson: "Productive" and "Latent" in strokes of 0.85–1.07 of the dots' radius as measured here, with the
    // antialiased edge, against 1.2 for the dots themselves; the crossings of the three t's were fitted as dots.
    // Here, without blur: strokes of 13 px measure 7 to 8.6 (at their joints), the dots 9.8
    const R = 10;
    const bold = word(40, 120, 80, 'THE LIFT', 13);
    const dots = [0, 1, 2, 3, 4].map(k => ({ x: 60 + 50 * k, y: 220, r: R, color: INK }));
    // touching in pairs, their bottoms in line: pieces thin for their height that hold whole dots (adults1)
    const pairs = [0, 1, 2, 3].flatMap(k => [{ x: 400 + 24 * k, y: 200, r: R, color: INK }, { x: 400 + 24 * k + 3, y: 218, r: R, color: INK }]);
    const image = paint(700, 300, [...bold, ...dots, ...pairs]);
    const { text } = annotationBoxes(image, backgroundColors(image)[0], R);
    const covered = (boxes, x, y) => boxes.some(b => inside(b, x, y));
    // (the T's crossing, and the word's far end)
    assert.ok(covered(text, 64, 46) && covered(text, 400, 80), JSON.stringify(text));
    for (const d of [...dots, ...pairs])
        assert.ok(!covered(text, d.x, d.y), `a box over the dot at ${d.x}, ${d.y}`);
    // without whole dots to measure, strokes over 0.6 of the radius stay unmasked, as before
    const alone = paint(700, 300, [...bold, ...pairs]);
    assert.deepEqual(annotationBoxes(alone, backgroundColors(alone)[0], R).text, []);
});

const CRIMSON = new URL('../bench-plos/crimson-10.1371-journal.pone.0361655.png', import.meta.url).pathname;

test('regression: crimson, "Productive" and "Latent" are masked, and no dot is', { skip: !existsSync(CRIMSON) && 'bench-plos crimson absent' }, () => {
    const image = readImage(CRIMSON);
    const { text } = annotationBoxes(image, backgroundColors(image)[0], 9.37);
    const covered = (x, y) => text.some(b => inside(b, x, y));
    // where the fit had dots: the crossings of the t's
    for (const [x, y] of [[693, 303], [1061, 863], [1166, 863]])
        assert.ok(covered(x, y), `text at ${x}, ${y}`);
    const truth = JSON.parse(readFileSync(new URL('../bench-plos/truth/crimson-10.1371-journal.pone.0361655.json', import.meta.url), 'utf8'));
    assert.equal(truth.dots.length, 180);
    for (const d of truth.dots)
        assert.ok(!covered(d.x, d.y), `a box over the dot at ${d.x}, ${d.y}`);
});

const WAGES = new URL('../testgraphs/wages2025.png', import.meta.url).pathname;

test('regression: wages2025, its title, subtitle, legend and the median line\'s label are masked', { skip: !existsSync(WAGES) && 'testgraphs wages2025 absent' }, () => {
    // before: 105 extras of 767 found on them (P 0.851). Its bold title, strokes as thick as the dots, stays.
    const image = readImage(WAGES);
    const { text, legend } = annotationBoxes(image, backgroundColors(image)[0], 17.6);
    assert.equal(legend.length, 22);
    const covered = (x, y) => text.some(b => inside(b, x, y));
    // in the subtitle, the legend's "Management" and the median line's label
    for (const [x, y] of [[250, 375], [3100, 632], [3100, 2720]])
        assert.ok(covered(x, y), `text at ${x}, ${y}`);
    // no box over the dense middle of the dots
    for (const [x, y] of [[1500, 2800], [1800, 3000], [2000, 2600]])
        assert.ok(!covered(x, y), `a box over the dots at ${x}, ${y}`);
});

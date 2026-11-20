import { test } from 'node:test';
import assert from 'node:assert/strict';
import { existsSync } from 'node:fs';
import { createDotModel, solidImage, toRgba } from '../src/core/render.js';
import { unmixFamilies, familyImage, regionLevels } from '../src/core/unmix.js';
import { flatMask } from '../src/core/scan.js';
import { readImage } from '../tools/png.mjs';
import { readTruthCsv, truthPixels } from '../tools/truth-csv.mjs';
import { extractAll } from '../tools/bench.mjs';
import { matchDots } from '../src/core/metrics.js';

const near = (a, b, tol, what) => assert.ok(Math.abs(a - b) <= tol, `${what}: ${a} vs ${b} (±${tol})`);

const WHITE = [255, 255, 255];
const RED = { color: [230, 40, 60], alpha: 0.5, radius: 8 };
const BLUE = { color: [40, 90, 240], alpha: 0.5, radius: 8 };
const T_TOL = 0.03; // 8-bit rounding of the composited colors

// dots drawn one at a time in the given order, as a renderer drawing rows in order would
function drawInOrder(width, height, dots) {
    let im = solidImage(width, height, WHITE);
    for (const { style, x, y } of dots) {
        const model = createDotModel(im, style);
        model.add(x, y);
        im = model.image;
    }
    return toRgba(im);
}

test('unmixFamilies: overlapping red and blue dots in either drawing order give each family one dot', () => {
    // left pair: blue drawn over red; right pair: red drawn over blue
    const image = drawInOrder(64, 30, [
        { style: RED, x: 14, y: 15 }, { style: BLUE, x: 20, y: 15 },
        { style: BLUE, x: 44, y: 15 }, { style: RED, x: 50, y: 15 },
    ]);
    const { T, mixed } = unmixFamilies(image, WHITE, [RED, BLUE]);
    const at = (x, y) => y * image.width + x;
    for (const [x, what] of [[17, 'blue over red'], [47, 'red over blue']]) {
        const p = at(x, 15);
        assert.equal(mixed[p], 1, `${what}: a pair explains the lens`);
        near(T[0][p], 0.5, T_TOL, `${what}: red T`);
        near(T[1][p], 0.5, T_TOL, `${what}: blue T`);
    }
    // one family alone: the other's transmittance is 1
    near(T[0][at(9, 15)], 0.5, T_TOL, 'red alone: red T');
    assert.equal(T[1][at(9, 15)], 1);
    near(T[1][at(25, 15)], 0.5, T_TOL, 'blue alone: blue T');
    assert.equal(T[0][at(25, 15)], 1);
    // red's image: one dot of red where red is, white where only blue is
    const red = familyImage(image, WHITE, RED, T[0], mixed);
    const px = (im, x, y) => [...im.data.subarray(4 * at(x, y), 4 * at(x, y) + 3)];
    const level1 = RED.color.map((c, ch) => c + (WHITE[ch] - c) * 0.5);
    for (const [x, y] of [[17, 15], [47, 15], [9, 15]])
        px(red, x, y).forEach((v, ch) => near(v, level1[ch], 2, `red image at ${x},${y}`));
    assert.deepEqual(px(red, 25, 15), WHITE);
});

test('unmixFamilies: a pixel no pair of families explains is left to the masks', () => {
    // green ink is neither red, blue, nor a mix of them over white
    const image = drawInOrder(20, 20, [{ style: { color: [20, 200, 40], alpha: 1, radius: 5 }, x: 10, y: 10 }]);
    const { T, mixed } = unmixFamilies(image, WHITE, [RED, BLUE]);
    const p = 10 * 20 + 10;
    assert.equal(mixed[p], 2);
    const red = familyImage(image, WHITE, RED, T[0], mixed);
    assert.deepEqual([...red.data.subarray(4 * p, 4 * p + 4)], [...image.data.subarray(4 * p, 4 * p + 4)]);
});

test('unmixFamilies: a flat pixel on one family’s line must be a whole number of its dots', () => {
    // bench2 group-smoother-ci: a blue dot over a light steel-blue band reads as 1.5 blue dots
    const fams = [{ color: [240, 49, 70], alpha: 0.4 }, { color: [39, 101, 252], alpha: 0.4 }];
    const flatImage = color => {
        const data = new Uint8ClampedArray(4 * 9);
        for (let p = 0; p < 9; p++)
            data.set([...color, 255], 4 * p);
        return { width: 3, height: 3, data };
    };
    const all = new Uint8Array(9).fill(1);
    assert.equal(unmixFamilies(flatImage([141, 167, 231]), WHITE, fams, all).mixed[4], 2, 'flat: left to the masks');
    // the same color at a rim (not flat) is a partly covered pixel
    const rim = unmixFamilies(flatImage([141, 167, 231]), WHITE, fams);
    assert.equal(rim.mixed[4], 0);
    assert.ok(rim.T[1][4] < 1);
    // one blue dot over white, flat: a whole count, kept
    const one = unmixFamilies(flatImage([169, 196, 253]), WHITE, fams, all); // 0.4 of blue over white
    assert.equal(one.mixed[4], 0);
    near(one.T[1][4], 0.6, T_TOL, 'one blue dot');
});

const BENCH1 = new URL('../bench1/', import.meta.url).pathname;
const BENCH2 = new URL('../bench2/', import.meta.url).pathname;

test('regression: the red–blue coincident pair is found (bench1 group)', { skip: !existsSync(BENCH1 + 'group.png') && 'bench1 absent' }, () => {
    // before: 98 found; the pair at (69.4, 52.2), one dot in each group, is purple, off both families' lines
    const truth = truthPixels(readTruthCsv(BENCH1 + 'data.csv')).dots;
    const found = extractAll(readImage(BENCH1 + 'group.png')).flatMap(r => r.fit.dots);
    const m = matchDots(truth, found, 1);
    assert.equal(found.length, 100);
    assert.equal(m.recall, 1);
});

test('regression: bench2 group finds all 204, overlapping stacks of both colors included', { skip: !existsSync(BENCH2 + 'group.png') && 'bench2 absent' }, () => {
    // before: 193 found, the 11 missed each within 6 px of a dot of the other color
    const truth = truthPixels(readTruthCsv(BENCH2 + 'data.csv')).dots;
    const found = extractAll(readImage(BENCH2 + 'group.png')).flatMap(r => r.fit.dots);
    const m = matchDots(truth, found, 1);
    assert.equal(found.length, 204);
    assert.equal(m.recall, 1);
});

test('regression: overlaps of two translucent families aren’t families of their own (bench2 group-large)', { skip: !existsSync(BENCH2 + 'group-large.png') && 'bench2 absent' }, () => {
    // before: red over blue and blue over red (α 0.33) read as two purple families with no radius, and unmixing
    // gave them the overlaps: red fitted 70 dots for 113, blue 39 for 91
    const truth = truthPixels(readTruthCsv(BENCH2 + 'data.csv')).dots;
    const results = extractAll(readImage(BENCH2 + 'group-large.png'));
    assert.equal(results.length, 2);
    const found = results.flatMap(r => r.fit.dots);
    const m = matchDots(truth, found, 1);
    assert.ok(Math.abs(found.length - truth.length) <= 2, `${found.length} found for ${truth.length}`);
    assert.ok(m.recall >= 0.98, `recall ${m.recall}`);
});

test('regionLevels: a family\'s levels over a box are measured there, where the sRGB model misses them', () => {
    // protrusion3's colors, painted flat by overlap count: the box 150, 183, 149, one dot over it 116, 163, 122 and
    // two 96, 153, 107. With c and α from the page, one dot over the box predicts 128, 167, 127, which read as 1.56
    // dots and was masked. Two dots on the page beside the box, 142, 178, 142, lie 12 units from it, and pixels of
    // theirs labeled the box's must not be taken for its first level.
    const W = 300;
    const H = 200;
    const R = 15;
    const page = [[255, 255, 255], [190, 208, 187], [142, 178, 142]];
    const boxed = [[150, 183, 149], [116, 163, 122], [96, 153, 107]];
    const inBox = (x, y) => x >= 100 && x < 280 && y >= 20 && y < 180;
    const dots = [{ x: 140, y: 60 }, { x: 220, y: 60 }, { x: 140, y: 130 }, { x: 160, y: 140 }, { x: 230, y: 140 }, { x: 245, y: 130 },
        { x: 40, y: 60 }, { x: 95, y: 100 }, { x: 110, y: 105 }];
    const data = new Uint8ClampedArray(4 * W * H);
    const index = new Uint8Array(W * H);
    for (let y = 0; y < H; y++) {
        for (let x = 0; x < W; x++) {
            const n = Math.min(2, dots.filter(d => Math.hypot(x + 0.5 - d.x, y + 0.5 - d.y) <= R).length);
            data.set([...(inBox(x, y) ? boxed : page)[n], 255], 4 * (y * W + x));
            // the region's label reaches a few pixels past the box, as nearest seeds give it
            index[y * W + x] = x >= 95 && x < 285 && y >= 15 && y < 185 ? 1 : 0;
        }
    }
    const image = { width: W, height: H, data };
    const levels = regionLevels(image, flatMask(image), index, 1, boxed[0], { color: [99, 146, 96], alpha: 0.424 });
    const dist = c => Math.hypot(...c.map((v, ch) => v - boxed[0][ch]));
    assert.equal(levels.ts.length, 2);
    near(levels.ts[0], dist(boxed[1]), 1, 'one dot');
    near(levels.ts[1], dist(boxed[2]), 1, 'two dots');
});

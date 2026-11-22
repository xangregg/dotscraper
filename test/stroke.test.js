import { test } from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, readFileSync } from 'node:fs';
import { createRng } from '../src/core/rng.js';
import { scanFamilies, extractAll, fitInput } from '../src/core/extract.js';
import { buildMask } from '../src/core/layers.js';
import { fitDots } from '../src/core/fit.js';
import { pageStroke } from '../src/core/stroke.js';
import { matchDots } from '../src/core/metrics.js';
import { readImage } from '../tools/png.mjs';

const TESTGRAPHS = new URL('../testgraphs/', import.meta.url).pathname;
const R = 10; // px, the dots' fill radius
const BLUE = [58, 111, 165];
const ORANGE = [223, 105, 59];
const SUPER = 4; // samples per pixel side

// Opaque dots drawn in order on a white page, antialiased by supersampling: each a disc of radius r + stroke in the
// stroke's color (if any) with its fill of radius r over it.
function draw(width, height, dots, stroke = 0, strokeColor = [255, 255, 255]) {
    const image = { width, height, data: new Uint8ClampedArray(4 * width * height).fill(255) };
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const sum = [0, 0, 0];
            for (let j = 0; j < SUPER; j++) {
                for (let i = 0; i < SUPER; i++) {
                    const sx = x + (i + 0.5) / SUPER;
                    const sy = y + (j + 0.5) / SUPER;
                    let c = [255, 255, 255];
                    for (const d of dots) {
                        const dist = Math.hypot(sx - d.x, sy - d.y);
                        if (dist <= R)
                            c = d.color;
                        else if (dist <= R + stroke)
                            c = strokeColor;
                    }
                    sum[0] += c[0];
                    sum[1] += c[1];
                    sum[2] += c[2];
                }
            }
            image.data.set(sum.map(v => v / (SUPER * SUPER)), 4 * (y * width + x));
        }
    }
    return image;
}

// Dots in groups: a few alone, and piles of two to four, each dot `apart` px from the one drawn before it in a
// direction of its own, each of a color drawn at random.
function piles(apart, colors = [BLUE, ORANGE]) {
    const rng = createRng(5);
    const dots = [];
    for (let gy = 0; gy < 4; gy++) {
        for (let gx = 0; gx < 6; gx++) {
            const size = 1 + (gx + 2 * gy) % 4;
            let x = 50 + 75 * gx + 6 * rng.uniform();
            let y = 50 + 75 * gy + 6 * rng.uniform();
            for (let i = 0; i < size; i++) {
                dots.push({ x, y, color: colors[Math.floor(colors.length * rng.uniform())] });
                const a = 2 * Math.PI * rng.uniform();
                x += apart * Math.cos(a);
                y += apart * Math.sin(a);
            }
        }
    }
    return dots;
}

const scan = image => {
    const { work, prepared } = scanFamilies(image);
    return pageStroke(work, prepared.estimates);
};
const found = image => extractAll(image).flatMap(r => r.fit.dots);
// … and as it was before the stroke was read: the families fitted with no stroke's pixels masked
const foundPlain = image => scanFamilies(image).prepared.estimates.flatMap(est => fitDots(fitInput(est.image, est, buildMask(est.image, est).mask)).dots);
const matched = (dots, got) => matchDots(dots, got, R / 2).pairs.length;

test('a white stroke around opaque dots: its width, and the dots under it are found', () => {
    const dots = piles(11);
    const image = draw(520, 370, dots, 2.5);
    const { work, prepared } = scanFamilies(image);
    const stroke = pageStroke(work, prepared.estimates);
    assert.ok(stroke.mask, 'the dots have a stroke');
    // read from opaque families only: a translucent family's overlaps show in its levels, and rings are fitted in
    // their drawing order
    const translucent = prepared.estimates.map(e => ({ ...e, style: { ...e.style, alpha: 0.6 } }));
    assert.equal(pageStroke(work, translucent).mask, null);
    assert.equal(pageStroke(work, prepared.estimates.map(e => ({ ...e, outlineMask: new Uint8Array(1) }))).mask, null);
    assert.ok(stroke.gaps >= 100 && stroke.share >= 0.9, `${stroke.gaps} gaps, ${stroke.share} of one width`);
    // from the fill's edge to the ink beyond, the edges' pixels by their coverage
    assert.ok(Math.abs(stroke.width - 2.5) <= 0.25, `width ${stroke.width}`);
    // the mask is the arcs: page-colored pixels a few px from dots on both sides, none inside a fill or far out
    let masked = 0;
    for (let p = 0; p < stroke.mask.length; p++) {
        if (!stroke.mask[p])
            continue;
        masked++;
        const x = p % image.width + 0.5;
        const y = Math.floor(p / image.width) + 0.5;
        const dist = dots.map(d => Math.hypot(x - d.x, y - d.y)).sort((a, b) => a - b);
        assert.ok(dist[1] <= R + 2.5 + 1.5, `a masked pixel ${dist[1].toFixed(1)} px from its second dot`);
    }
    assert.ok(masked > 100, `${masked} pixels masked`);
    // the dots cut by the strokes of those over them are found too: 58 of 60 (two others hide the rest), and no
    // dot that isn't there. Before the stroke was read, 29: the arc over a dot runs through it, often through its
    // center, and the opaque model pays for every pixel of the arc that a dot placed there covers.
    const got = found(image);
    assert.ok(matched(dots, got) >= 57, `${matched(dots, got)} of 60 found`);
    assert.equal(got.length, matched(dots, got), 'no dot that isn\'t there');
    assert.ok(matched(dots, foundPlain(image)) <= 32, `${matched(dots, foundPlain(image))} of 60 without the stroke`);
});

test('no stroke: plain overlapping dots, dots that only come near, and outlines of another color', () => {
    // the same piles without strokes: the dots meet with no page between
    assert.equal(scan(draw(520, 370, piles(11))).mask, null);
    // dots packed side by side, 2 px apart at the nearest (a beeswarm): the page between them widens along the gap
    const packed = [];
    for (let row = 0; row < 8; row++) {
        for (let col = 0; col < 12; col++)
            packed.push({ x: 60 + (2 * R + 2) * (col + (row % 2) / 2), y: 60 + (2 * R + 2) * Math.sqrt(3) / 2 * row, color: BLUE });
    }
    const swarm = scan(draw(400, 280, packed));
    assert.equal(swarm.mask, null);
    assert.equal(swarm.gaps, 0, 'whole dots beside whole dots tell nothing');
    // dark outlines as wide as the stroke was: one width all round, and not the page's color
    assert.equal(scan(draw(520, 370, piles(11), 2.5, [30, 30, 30])).mask, null);
});

// a rectangle [x0, y0, x1, y1) of other ink, or of `color`
function bar(image, [x0, y0, x1, y1], color = [40, 40, 40]) {
    for (let y = y0; y < y1; y++) {
        for (let x = x0; x < x1; x++)
            image.data.set([...color, 255], 4 * (y * image.width + x));
    }
}

test('too little to tell a stroke by: one pair of dots, and gaps of many widths beside cut dots', () => {
    // one stroked dot over another, among dots alone: an arc of one width, 25 directions of it
    const pair = [{ x: 100, y: 100, color: BLUE }, { x: 111, y: 104, color: BLUE }];
    for (let i = 0; i < 6; i++)
        pair.push({ x: 60 + 70 * i, y: 200, color: BLUE });
    const one = scan(draw(520, 300, pair, 2.5));
    assert.ok(one.gaps > 0 && one.gaps < 40 && one.share === 1, `${one.gaps} gaps`);
    assert.equal(one.mask, null);
    // no strokes: dots cut by bars of other ink that touch them, each with a whole dot 1 to 4 px from what is left
    // of it. The page between those is of every width, and widens along each gap.
    const dots = [];
    const bars = [];
    for (let i = 0; i < 12; i++) {
        const x = 70 + 110 * (i % 4);
        const y = 60 + 90 * Math.floor(i / 4);
        dots.push({ x, y, color: BLUE }, { x: x - (2 * R + 1 + i % 4), y: y - 3 + i % 5, color: BLUE });
        bars.push([x + 2, y - R - 3, x + R + 3, y + R + 3]);
    }
    for (let i = 0; i < 5; i++)
        dots.push({ x: 40 + 90 * i, y: 330, color: BLUE });
    const image = draw(520, 370, dots);
    for (const b of bars)
        bar(image, b);
    const near = scan(image);
    assert.ok(near.gaps >= 40, `${near.gaps} gaps`);
    assert.ok(near.share < 0.6, `${near.share} of one width`);
    assert.equal(near.mask, null);
});

test('a dash in the dots\' color beside each whole dot is no cut dot', () => {
    // a dashed line under a row of dots, a dash 2.5 px from each: the page between is one width, as under a stroke,
    // and a dash has a tenth of a dot's area. It has no pixel 2 px inside it, as what shows of a dot has.
    const dots = [];
    for (let i = 0; i < 14; i++)
        dots.push({ x: 40 + 34 * i, y: 80, color: BLUE }, { x: 40 + 34 * i, y: 160, color: BLUE });
    const image = draw(520, 240, dots);
    for (const d of dots)
        bar(image, [d.x - 5, d.y + R + 2, d.x + 5, d.y + R + 5], BLUE);
    const stroke = scan(image);
    assert.equal(stroke.gaps, 0);
    assert.equal(stroke.mask, null);
});

test('the stroke\'s pixels: beside a sliver of a dot too, and not other ink between two dots', () => {
    const dots = piles(11);
    // a dot 6 px from the one over it: what shows of it is 3.5 px wide at the most
    dots.push({ x: 60, y: 345, color: ORANGE }, { x: 66, y: 345, color: BLUE });
    // two dots 3 px apart with a line of other ink in between
    dots.push({ x: 440, y: 345, color: BLUE }, { x: 463, y: 345, color: BLUE });
    const image = draw(520, 370, dots, 2.5);
    bar(image, [450, 335, 453, 355]);
    const { mask } = scan(image);
    const at = (x, y) => mask[Math.floor(y) * image.width + Math.floor(x)];
    // the arc between the sliver and the dot over it, 0.5 to 2 px out from that dot's edge
    assert.deepEqual([0.5, 1, 1.5, 2].map(d => at(66 - R - d, 345)), [1, 1, 1, 1]);
    // the line between the two dots is no page
    assert.deepEqual([450, 451, 452].map(x => at(x + 0.5, 345)), [0, 0, 0]);
    // … nor is the page beside a dot alone, or a dot's fill
    assert.equal(at(66 + R + 1.5, 345), 0);
    assert.equal(at(66, 345), 0);
});

test('regression: song-cents, a white stroke 2.4 px wide, and the dots it cuts', { skip: !existsSync(TESTGRAPHS + 'song-cents.png') && 'local test image absent' }, () => {
    // 36 of its 325 dots were missed, most with half their fill in view. A part of the plot with piles of dots in
    // four fills, and the truth's dots in it (by hand, from the page's SVG):
    const image = readImage(TESTGRAPHS + 'song-cents.png');
    const [x0, y0, w, h] = [1700, 560, 560, 360];
    const crop = { width: w, height: h, data: new Uint8ClampedArray(4 * w * h) };
    for (let y = 0; y < h; y++)
        crop.data.set(image.data.subarray(4 * ((y0 + y) * image.width + x0), 4 * ((y0 + y) * image.width + x0 + w)), 4 * y * w);
    const stroke = scan(crop);
    assert.ok(stroke.mask);
    // the SVG's stroke is 1 unit, 2.36 px
    assert.ok(Math.abs(stroke.width - 2.36) <= 0.15, `width ${stroke.width}`);
    assert.ok(stroke.share >= 0.8, `${stroke.share} of one width`);
    const truth = JSON.parse(readFileSync(TESTGRAPHS + 'truth/song-cents.json', 'utf8')).dots
        .map(d => ({ x: d.x - x0, y: d.y - y0 }))
        .filter(d => d.x > 14 && d.y > 14 && d.x < w - 14 && d.y < h - 14);
    const hit = dots => matchDots(truth, dots, 6).pairs.length;
    // 72 dots: 66 found with the stroke's pixels ignored, 57 without
    assert.equal(truth.length, 72);
    const [now, before] = [hit(found(crop)), hit(foundPlain(crop))];
    assert.ok(now >= 65, `${now} of 72 found`);
    assert.ok(before <= 58, `${before} of 72 found without the stroke`);
});

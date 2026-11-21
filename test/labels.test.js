import { test } from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, readdirSync } from 'node:fs';
import { detectAxes } from '../src/core/axes.js';
import { labelGlyphs, glyphKinds, glyphGrid, classifyDigit, readLabel, consistentValues, readAxisLabels } from '../src/core/labels.js';
import { DIGIT_TEMPLATES, TEMPLATE_WIDTH, TEMPLATE_HEIGHT } from '../src/core/digit-templates.js';
import { synthGraph } from '../src/core/synth.js';
import { readImage } from '../tools/png.mjs';

const WHITE = [[255, 255, 255]];
const ARIAL = DIGIT_TEMPLATES.find(f => f.name === 'Arial');

// Text for the tests, drawn from the templates themselves (Node has no fonts): each digit is its template scaled to
// `height` px, a point is a square on the baseline, a comma a square with a tail below it, a minus a bar, and "$"
// a tall glyph that is no digit (a saltire). `x` is the text's left edge, or its right edge with `right`; `y` the
// digits' top. Returns the ink's box [x0, y0, x1, y1).
function drawText(image, text, x, y, height, { right = false, font = ARIAL } = {}) {
    const s = height / TEMPLATE_HEIGHT;
    const unit = Math.max(2, Math.round(0.14 * height)); // px, a point's side
    const gap = Math.max(1, Math.round(0.14 * height));
    const glyphs = [...text].map(ch => {
        if (ch === '.')
            return { width: unit, dark: (i, j) => (j >= height - unit ? 1 : 0) };
        if (ch === ',')
            return { width: unit, tail: unit, dark: (i, j) => (j >= height - unit && (j < height || i < unit / 2) ? 1 : 0) };
        if (ch === '-')
            return { width: Math.round(0.4 * height), dark: (i, j) => (Math.abs(j + 0.5 - 0.6 * height) <= unit / 2 ? 1 : 0) };
        if (ch === '$') {
            const w = Math.round(0.6 * height);
            const across = j => (j + 0.5) / height * w;
            return { width: w, dark: (i, j) => (Math.min(Math.abs(i + 0.5 - across(j)), Math.abs(i + 0.5 - w + across(j))) <= 1 ? 1 : 0) };
        }
        // a digit: its template's inked columns, sampled bilinearly
        const grid = font.digits[ch];
        const at = (c, r) => (c < 0 || r < 0 || c >= TEMPLATE_WIDTH || r >= TEMPLATE_HEIGHT ? 0 : grid[r * TEMPLATE_WIDTH + c]);
        const inked = Array.from({ length: TEMPLATE_WIDTH }, (_, c) => c).filter(c => Array.from({ length: TEMPLATE_HEIGHT }, (_, r) => at(c, r)).some(v => v >= 0.3));
        const c0 = inked[0];
        return {
            width: Math.round((inked[inked.length - 1] + 1 - c0) * s),
            dark: (i, j) => {
                const u = c0 + (i + 0.5) / s - 0.5;
                const v = (j + 0.5) / s - 0.5;
                const [c, r] = [Math.floor(u), Math.floor(v)];
                const [fu, fv] = [u - c, v - r];
                return (at(c, r) * (1 - fu) + at(c + 1, r) * fu) * (1 - fv) + (at(c, r + 1) * (1 - fu) + at(c + 1, r + 1) * fu) * fv;
            },
        };
    });
    const total = glyphs.reduce((sum, g) => sum + g.width, 0) + gap * (glyphs.length - 1);
    let left = Math.round(right ? x - total : x);
    const x0 = left;
    for (const g of glyphs) {
        for (let j = 0; j < height + (g.tail ?? 0); j++) {
            for (let i = 0; i < g.width; i++) {
                const p = 4 * ((y + j) * image.width + left + i);
                const v = Math.round(255 * (1 - Math.min(1, g.dark(i, j))));
                image.data[p] = image.data[p + 1] = image.data[p + 2] = Math.min(image.data[p], v);
            }
        }
        left += g.width + gap;
    }
    return [x0, y, left - gap, y + height];
}

function blank(width, height) {
    return { width, height, data: new Uint8ClampedArray(4 * width * height).fill(255) };
}

// A graph with real tick labels: synth's axes without its label blocks, and the labels drawn where a renderer puts
// them (centered under an x tick, right-aligned beside a y tick).
function labeled({ xLabels, yLabels, height = 14, before = image => image }) {
    const spec = {
        width: 520, height: 400, background: [255, 255, 255], blur: 0.5, families: [],
        axes: {
            x: { pos: 330.5, from: 79, to: 500, ticks: xLabels.map((_, i) => 100 + i * 380 / (xLabels.length - 1)) },
            y: { pos: 80.5, from: 20, to: 331.5, ticks: yLabels.map((_, i) => 310 - i * 270 / (yLabels.length - 1)) },
            label: [0, 0],
        },
    };
    const { image } = synthGraph(spec);
    xLabels.forEach((text, i) => {
        const probe = drawText(blank(200, 60), text, 0, 0, height);
        drawText(image, text, spec.axes.x.ticks[i] - (probe[2] - probe[0]) / 2, 348, height);
    });
    yLabels.forEach((text, i) => drawText(image, text, 64, Math.round(spec.axes.y.ticks[i] - height / 2), height, { right: true }));
    before(image);
    return { image, spec };
}

function axisValues(image, which) {
    const axes = detectAxes(image);
    const { read, scale } = readAxisLabels(image, axes.backgrounds, axes[which], which);
    return { texts: read.map(r => r?.text ?? null), scale };
}

test('templates: every font has ten digits, and each digit is told by the other fonts\' templates', () => {
    assert.ok(DIGIT_TEMPLATES.length >= 10);
    for (const font of DIGIT_TEMPLATES) {
        assert.equal(Object.keys(font.digits).join(''), '0123456789', font.name);
        const others = DIGIT_TEMPLATES.filter(f => f !== font);
        for (const [digit, grid] of Object.entries(font.digits)) {
            assert.equal(grid.length, TEMPLATE_WIDTH * TEMPLATE_HEIGHT);
            const found = classifyDigit(grid, others);
            assert.equal(found.digit, digit, `${font.name} ${digit}`);
            assert.ok(found.score >= 0.7, `${font.name} ${digit}: ${found.score}`);
        }
    }
});

test('glyphKinds: digits, a point on the baseline, a comma below it, a minus at mid-height', () => {
    const glyph = (x0, y0, x1, y1) => ({ x0, y0, x1, y1 });
    const found = {
        height: 20, top: 100,
        glyphs: [glyph(0, 109, 8, 112), glyph(10, 100, 22, 120), glyph(24, 117, 27, 124), glyph(30, 100, 42, 120), glyph(44, 116, 48, 120), glyph(50, 100, 62, 121), glyph(64, 106, 76, 120)],
    };
    // (the last is an x-height letter: neither tall nor small)
    assert.deepEqual(glyphKinds(found), ['-', 'digit', ',', 'digit', '.', 'digit', '?']);
});

test('regression: a point is told from a comma by the label\'s own baseline, not the axis\'s line', () => {
    // fibers-distance's "3.7": the line its axis's labels share is 18 px from 105, this label's digits end a pixel
    // lower, and so does its point, which read as a comma hanging below the line
    const glyph = (x0, y0, x1, y1) => ({ x0, y0, x1, y1 });
    const found = { height: 18, top: 105, glyphs: [glyph(79, 105, 91, 124), glyph(95, 120, 98, 124), glyph(101, 106, 113, 124)] };
    assert.deepEqual(glyphKinds(found), ['digit', '.', 'digit']);
});

test('readLabel: a number with a minus sign, a thousands separator and a decimal point, at 12 to 28 px', () => {
    for (const height of [12, 14, 20, 28]) {
        const image = blank(400, 80);
        const box = drawText(image, '-1,250.75', 40, 20, height);
        const read = readLabel(image, WHITE, box);
        assert.equal(read.text, '-1,250.75', `${height} px`);
        assert.equal(read.value, -1250.75);
        assert.ok(read.score >= 0.8, `${height} px: score ${read.score}`);
    }
});

test('readLabel: every font\'s digits read as drawn; a box cut short still finds the whole label', () => {
    for (const font of DIGIT_TEMPLATES) {
        const image = blank(420, 60);
        const box = drawText(image, '1234567890', 20, 20, 21, { font });
        assert.equal(readLabel(image, WHITE, box).text, '1234567890', font.name);
    }
    const image = blank(200, 60);
    const box = drawText(image, '100', 20, 20, 14);
    // the box of the two zeros alone, as axes.js gives it when its search band ends in the label
    assert.equal(readLabel(image, WHITE, [box[0] + 9, box[1], box[2], box[3]]).value, 100);
});

test('readLabel: a letter makes the label unread, and a dash after the number is not a minus', () => {
    const image = blank(300, 60);
    const box = drawText(image, '$320', 20, 20, 14);
    const read = readLabel(image, WHITE, box);
    assert.equal(read.text, '?320');
    assert.equal(read.value, null);
    const dashed = blank(300, 60);
    assert.equal(readLabel(dashed, WHITE, drawText(dashed, '40-', 20, 20, 14)).value, 40);
});

test('labelGlyphs: touching digits are cut apart', () => {
    const image = blank(200, 60);
    const box = drawText(image, '40', 20, 20, 14);
    // close the gap between the two: a column of gray, as small text has
    const found0 = labelGlyphs(image, WHITE, box);
    assert.equal(found0.glyphs.length, 2);
    for (let x = found0.glyphs[0].x1; x < found0.glyphs[1].x0; x++) {
        for (let y = 26; y < 30; y++)
            image.data.fill(140, 4 * (y * image.width + x), 4 * (y * image.width + x) + 3);
    }
    const found = labelGlyphs(image, WHITE, box);
    assert.equal(found.glyphs.length, 2);
    assert.deepEqual(found.glyphs.map(g => classifyDigit(glyphGrid(found.cover, g)).digit), ['4', '0']);
});

test('consistentValues: the scale fills an unread tick and overrules a misread one, with round values', () => {
    // ticks half a pixel off even places, as detected ones are
    const pos = [30.5, 103.4, 176.5, 249.6, 322.5, 395.4, 468.5];
    const read = [null, 80, 70, 80, 50, 40, 30];
    const scale = consistentValues(pos.map((p, i) => ({ pos: p, value: read[i] })));
    assert.equal(scale.type, 'linear');
    assert.deepEqual(scale.values, [90, 80, 70, 60, 50, 40, 30]);
    assert.deepEqual(scale.agree, [false, true, true, false, true, true, true]);
    assert.equal(scale.count, 5);
});

test('consistentValues: a log scale, and decimals', () => {
    const log = consistentValues([320, 160, 80, 40, 20, 10].map((v, i) => ({ pos: 50 + 60 * i, value: i === 1 || i === 5 ? null : v })));
    assert.equal(log.type, 'log');
    assert.deepEqual(log.values, [320, 160, 80, 40, 20, 10]);
    const decimals = consistentValues([7.5, 5, 2.5, null].map((v, i) => ({ pos: 40.5 + 91 * i, value: v })));
    assert.equal(decimals.type, 'linear');
    assert.deepEqual(decimals.values, [7.5, 5, 2.5, 0]);
});

test('consistentValues: no scale from two labels, from one value repeated, or from doses that are names', () => {
    assert.equal(consistentValues([{ pos: 10, value: 1 }, { pos: 90, value: 4 }]), null);
    assert.equal(consistentValues([10, 10, 10, 10].map((v, i) => ({ pos: 50 * i, value: v }))), null);
    // (hair3's x axis: 50, 100, 200 at even steps are on a log scale, and 0, 5, 20 are not on it)
    assert.equal(consistentValues([0, 5, 20, 50, 100, 200].map((v, i) => ({ pos: 100 + 80 * i, value: v }))), null);
    // neither linear nor log: left alone (pathogen's y axis runs from 64 to 1 and 0 on log(1 + y))
    assert.equal(consistentValues([64, 32, 16, 8, 4, 2, 1, 0].map(v => ({ pos: 400 - 50 * Math.log2(1 + v), value: v }))), null);
});

test('readAxisLabels: a linear x axis and a y axis in thousands, read and confirmed', () => {
    const { image } = labeled({ xLabels: ['-10', '0', '10', '20', '30'], yLabels: ['0', '500', '1,000', '1,500', '2,000'] });
    const x = axisValues(image, 'x');
    assert.deepEqual(x.texts, ['-10', '0', '10', '20', '30']);
    assert.deepEqual(x.scale.values, [-10, 0, 10, 20, 30]);
    const y = axisValues(image, 'y');
    assert.deepEqual(y.scale.values, [2000, 1500, 1000, 500, 0]);
    assert.equal(y.scale.count, 5);
});

test('readAxisLabels: unit signs are left out ("$" before, a letter after), on a log and a linear axis', () => {
    const { image } = labeled({ xLabels: ['0$', '25$', '50$', '75$', '100$'], yLabels: ['$10', '$20', '$40', '$80', '$160', '$320'] });
    const y = axisValues(image, 'y');
    assert.equal(y.scale.type, 'log');
    assert.deepEqual(y.scale.values, [320, 160, 80, 40, 20, 10]);
    assert.deepEqual(axisValues(image, 'x').scale.values, [0, 25, 50, 75, 100]);
});

test('readAxisLabels: a neighbor\'s glyphs are told from the label\'s by the axis\'s shared line', () => {
    // as in JMP's graphs: a rotated title's letter beside one y label, taller than the digits and within a word's
    // gap, and a title's descender hanging into the top one
    const { image, spec } = labeled({
        xLabels: ['0', '1', '2', '3'], yLabels: ['30', '40', '50', '60', '70', '80', '90'],
        before: image => {
            const y60 = Math.round(310 - 3 * 270 / 6);
            for (let y = y60 - 11; y < y60 + 10; y++) {
                for (let x = 30; x < 40; x++)
                    image.data.fill(0, 4 * (y * image.width + x), 4 * (y * image.width + x) + 3);
            }
            for (let y = 20; y < 40; y++) {
                for (let x = 52; x < 55; x++)
                    image.data.fill(0, 4 * (y * image.width + x), 4 * (y * image.width + x) + 3);
            }
        },
    });
    assert.equal(spec.axes.y.ticks.length, 7);
    const y = axisValues(image, 'y');
    assert.equal(y.texts[3], '60');
    assert.deepEqual(y.scale.values, [90, 80, 70, 60, 50, 40, 30]);
});

test('readAxisLabels: category names give no scale', () => {
    const { image } = labeled({ xLabels: ['$-$', '$$', '$-$$'], yLabels: ['0', '5', '10'] });
    assert.equal(axisValues(image, 'x').scale, null);
});

const PLOS = new URL('../bench-plos/', import.meta.url).pathname;
const plos = name => existsSync(PLOS) && readdirSync(PLOS).find(f => f.startsWith(`${name}-`) && f.endsWith('.png'));

test('bench-plos: step1\'s decimals, score2\'s thousands and droplet1\'s percent are read', { skip: !plos('step1') && 'bench-plos absent' }, () => {
    const values = (name, which) => axisValues(readImage(PLOS + plos(name)), which).scale?.values;
    assert.deepEqual(values('step1', 'x'), [40, 50, 60, 70, 80, 90]);
    assert.deepEqual(values('step1', 'y'), [1.4, 1.3, 1.2, 1.1, 1, 0.9, 0.8, 0.7, 0.6]);
    assert.deepEqual(values('score2', 'x'), [0, 2000, 4000, 6000, 8000, 10000]);
    assert.deepEqual(values('score2', 'y'), [600, 500, 400, 300, 200, 100]);
    // "100%": the axis's own unit
    assert.deepEqual(values('droplet1', 'y'), [100, 75, 50, 25, 0]);
    assert.deepEqual(values('droplet1', 'x'), [0, 0.25, 0.5, 0.75, 1]);
    // bold digits (landing1), and names under the other axis
    assert.deepEqual(values('landing1', 'y'), [7.5, 5, 2.5]);
    assert.equal(values('landing1', 'x'), undefined);
});

const BENCH1 = new URL('../bench1/', import.meta.url).pathname;

test('bench1: JMP\'s labels, the title\'s "p" through "90" and the rotated title beside "60"', { skip: !existsSync(BENCH1 + 'medium-dots.png') && 'bench1 absent' }, () => {
    for (const file of ['medium-dots.png', 'medium-dots.jpg']) {
        const image = readImage(BENCH1 + file);
        const x = axisValues(image, 'x');
        assert.deepEqual(x.texts, ['65', '70', '75', '80', '85', '90', '95', '100'], file);
        assert.deepEqual(x.scale.values, [65, 70, 75, 80, 85, 90, 95, 100], file);
        const y = axisValues(image, 'y');
        assert.deepEqual(y.scale.values, [90, 80, 70, 60, 50, 40, 30], file);
        assert.deepEqual(y.texts.slice(1), ['80', '70', '60', '50', '40', '30'], file);
    }
});

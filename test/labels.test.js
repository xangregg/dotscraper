import { test } from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, readdirSync } from 'node:fs';
import { detectAxes } from '../src/core/axes.js';
import { labelGlyphs, glyphKinds, glyphGrid, classifyDigit, classifySign, readLabel, consistentValues, readAxisLabels } from '../src/core/labels.js';
import { DIGIT_TEMPLATES, TEMPLATE_WIDTH, TEMPLATE_HEIGHT } from '../src/core/digit-templates.js';
import { synthGraph } from '../src/core/synth.js';
import { readImage } from '../tools/png.mjs';

const WHITE = [[255, 255, 255]];
const EXPONENT_SIZE = 0.66; // × the digits' height: an exponent's digits
const EXPONENT_RISE = 0.38; // × the digits' height: how far above the digits' top an exponent starts
const ARIAL = DIGIT_TEMPLATES.find(f => f.name === 'Arial');
// The signs of scientific notation as R's Helvetica sets them, [height, bottom] × the digits' height from their top:
// an "e" and an "x" on the baseline, a plus about the middle, the multiplication sign on the baseline (plotmath's)
const SIGN_PLACE = { e: [0.72, 1], E: [1, 1], '+': [0.66, 0.84], '×': [0.58, 1], x: [0.75, 1] };
const SPACE = 0.39; // × the digits' height: a typed space, which with the glyphs' own gaps is 0.67 between two glyphs

// Text for the tests, drawn from the templates themselves (Node has no fonts): each digit is its template scaled to
// `height` px, a point is a square on the baseline, a comma a square with a tail below it, a minus a bar, and "$"
// a tall glyph that is no digit (a saltire). "e", "E", "+", "×" and "x" are their templates at SIGN_PLACE, a space
// is SPACE wide, and "~" draws the glyph after it touching the one before it, as small type has them. What follows
// "^" is an exponent: smaller and raised, as Nature's figures set it (EXPONENT_SIZE, EXPONENT_RISE), and after "^~"
// touching its base, no empty column between the two. `x` is the text's left edge, or its right edge with
// `right`; `y` the digits' top, which an exponent rises above by `lift` × the height. Returns the ink's box
// [x0, y0, x1, y1).
function drawText(image, text, x, y, height, { right = false, font = ARIAL, lift = EXPONENT_RISE } = {}) {
    if (text.includes('^')) {
        const [number, raised] = text.split('^');
        const power = raised.replace(/^~/, '');
        const small = Math.round(EXPONENT_SIZE * height);
        const rise = Math.round(lift * height);
        const gap = raised.startsWith('~') ? 0 : Math.max(1, Math.round(0.14 * height));
        const widthOf = (s, h) => {
            const b = drawText(blank(400, h), s, 0, 0, h, { font });
            return b[2] - b[0];
        };
        const a = drawText(image, number, right ? x - widthOf(power, small) - gap : x, y, height, { right, font });
        const b = drawText(image, power, a[2] + gap, y - rise, small, { font });
        return [a[0], y - rise, b[2], a[3]];
    }
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
        if (ch === ' ')
            return { width: Math.round(SPACE * height), dark: () => 0 };
        if (ch === '~')
            return { width: -2 * gap - 1, dark: () => 0 };
        // a digit or a sign: its template's inked columns, sampled bilinearly, over the rows [from, from + rows)
        const [size, foot] = SIGN_PLACE[ch] ?? [1, 1];
        const rows = Math.round(size * height);
        const from = Math.round(foot * height) - rows;
        const s = rows / TEMPLATE_HEIGHT;
        const grid = font.digits[ch] ?? font.signs[ch];
        const at = (c, r) => (c < 0 || r < 0 || c >= TEMPLATE_WIDTH || r >= TEMPLATE_HEIGHT ? 0 : grid[r * TEMPLATE_WIDTH + c]);
        const inked = Array.from({ length: TEMPLATE_WIDTH }, (_, c) => c).filter(c => Array.from({ length: TEMPLATE_HEIGHT }, (_, r) => at(c, r)).some(v => v >= 0.3));
        const c0 = inked[0];
        return {
            width: Math.round((inked[inked.length - 1] + 1 - c0) * s),
            dark: (i, j) => {
                if (j < from || j >= from + rows)
                    return 0;
                const u = c0 + (i + 0.5) / s - 0.5;
                const v = (j - from + 0.5) / s - 0.5;
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

// An image from rows of gray values, two hex digits a pixel: text as Chrome's canvas drew it, kept for the regressions
// that the templates' own shapes, drawn here, don't show.
function gray(rows) {
    const image = blank(rows[0].length / 2, rows.length);
    rows.forEach((row, y) => {
        for (let x = 0; x < image.width; x++)
            image.data.fill(parseInt(row.slice(2 * x, 2 * x + 2), 16), 4 * (y * image.width + x), 4 * (y * image.width + x) + 3);
    });
    return image;
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
        const probe = drawText(blank(200, 60), text, 0, 20, height);
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

test('templates: every font has the signs of scientific notation, told by the other fonts\' templates; no digit is more like one', () => {
    for (const font of DIGIT_TEMPLATES) {
        assert.equal(Object.keys(font.signs).join(''), 'eE+×x', font.name);
        const others = DIGIT_TEMPLATES.filter(f => f !== font);
        for (const [sign, grid] of Object.entries(font.signs)) {
            assert.equal(grid.length, TEMPLATE_WIDTH * TEMPLATE_HEIGHT);
            const found = classifySign(grid, 'eE+×x', others);
            // (a typed "x" is read as the multiplication sign it stands for)
            assert.equal(found.sign, sign === 'x' ? '×' : sign, `${font.name} ${sign}`);
            assert.ok(found.score >= 0.75, `${font.name} ${sign}: ${found.score}`);
        }
        for (const [digit, grid] of Object.entries(font.digits))
            assert.ok(classifySign(grid, 'eE+×x').score < classifyDigit(grid, others).score, `${font.name} ${digit}`);
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

test('readLabel: a number with an exponent is its power, at 14 to 28 px', () => {
    for (const height of [14, 20, 28]) {
        for (const [text, value] of [['10^3', 1000], ['10^-2', 0.01], ['10^-5', 0.00001], ['10^0', 1], ['10^12', 1e12], ['2^10', 1024], ['-10^1', -10]]) {
            const image = blank(300, 90);
            const read = readLabel(image, WHITE, drawText(image, text, 40, 40, height));
            assert.equal(read.text, text, `${text} at ${height} px`);
            assert.equal(read.value, value, `${text} at ${height} px`);
        }
    }
    // the box of the "10" alone, as axes.js gave it before it took an exponent into a label's word
    const image = blank(300, 90);
    const box = drawText(image, '10^4', 40, 40, 20);
    const read = readLabel(image, WHITE, [box[0], 40, box[0] + 24, 60]);
    assert.equal(read.value, 10000);
    assert.deepEqual(read.box, box);
});

test('regression: a raised minus sign touching its digit is cut from it', () => {
    // primates2's "10" with a raised "-2": no empty column between the minus and the 2, and the two were one glyph
    // that read as no digit
    const image = blank(300, 90);
    const box = drawText(image, '10^-2', 40, 40, 28);
    const found = labelGlyphs(image, WHITE, box);
    assert.equal(found.raised.length, 2);
    const [minus, two] = found.raised;
    for (let x = minus.x1; x < two.x0; x++) {
        for (let y = minus.y0; y < minus.y1; y++)
            image.data.fill(0, 4 * (y * image.width + x), 4 * (y * image.width + x) + 3);
    }
    const read = readLabel(image, WHITE, box);
    assert.equal(read.text, '10^-2');
    assert.equal(read.value, 0.01);
    // a raised 1 starts flat too, with its flag, above mid-height (phase1's and primates2's): it stays one glyph
    const one = blank(300, 90);
    const ten = drawText(one, '10', 40, 40, 28);
    const ink = (x0, y0, x1, y1) => {
        for (let y = y0; y < y1; y++) {
            for (let x = x0; x < x1; x++)
                one.data.fill(0, 4 * (y * one.width + x), 4 * (y * one.width + x) + 3);
        }
    };
    ink(ten[2] + 4, 32, ten[2] + 11, 35);
    ink(ten[2] + 11, 29, ten[2] + 14, 48);
    assert.deepEqual(labelGlyphs(one, WHITE, ten).raised, [{ x0: ten[2] + 4, y0: 29, x1: ten[2] + 14, y1: 48 }]);
});

test('regression: an exponent touching its base is cut from it, and the base is a digit of its line', () => {
    // plotmath's "4 × 10" with a raised 5 at 72 dpi (bench-r's times-r72, digits 9 px): the 5 follows the 0 with no
    // empty column, the pair was one glyph taller than the digits, the line was taken from it and nothing read
    const r72 = gray([
        'ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff',
        'ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff',
        'ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff',
        'fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff7294b4bf6ffffff',
        'ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffca7effffffffffff',
        'fffffffffffffffffffffffffffffffffffffffffffffffffbfeffffffffffffffff9a45537cffffffff',
        'ffffffffffffb911ffffffffffffffffffffffffffffffe81ddcffffee551b2cccffe7edff5dd7ffffff',
        'ffffffffffef2b10ffffffffffffffffffffffffffffa42b05dcffff5779fbb31af7b0d3ff63d8ffffff',
        'ffffffffff5e9810fffffffffffffffffffffffffff943d715dcffff0debffff31badc516c82ffffffff',
        'ffffffffae5ce010fffffffff6cdfffffec7ffffffffffff15dcffef06ffffff4aa4fffff9ffffffffff',
        'ffffffea2bece010ffffffffe732c0fe6777ffffffffffff15dcffdb11ffffff5498ffffffffffffffff',
        'ffffff576cbea70cbef9ffffffe6324e77ffffffffffffff15dcffee04ffffff47b1ffffffffffffffff',
        'ffffff4c2b2b26022becfffffffe5e17c1ffffffffffffff15dcffff0ae6ffff29ceffffffffffffffff',
        'ffffffffffffe010fffffffffe6478e734c0ffffffffffff15dcffff5e78fdb824f8ffffffffffffffff',
        'ffffffffffffe010ffffffffe083ffffe77cffffffffffff15dcffffdd292023beffffffffffffffffff',
        'ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffeaffffffffffffffffffffff',
        'ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff',
        'ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff',
        'ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff',
    ]);
    const read = readLabel(r72, WHITE, [3, 3, r72.width - 3, r72.height - 3]);
    assert.equal(read.text, '4×10^5');
    assert.equal(read.value, 400000);
    // drawn: powers of one and two digits, and a number before the sign
    for (const height of [14, 20, 28]) {
        for (const [text, reads, value] of [['10^~5', '10^5', 100000], ['10^~12', '10^12', 1e12], ['2 × 10^~4', '2×10^4', 20000]]) {
            const image = blank(400, 90);
            const box = drawText(image, text, 40, 40, height);
            const drawn = readLabel(image, WHITE, box, { near: (box[0] + box[2]) / 2 });
            assert.equal(drawn.text, reads, `${text} at ${height} px`);
            assert.equal(drawn.value, value, `${text} at ${height} px`);
        }
    }
    // STIX's "10" with a raised 9 a pixel closer than touching, 16 px, as Chrome drew it: the 9's bowl hangs over the
    // 0's last columns, and the 0 with that ink above it read as a 6. The base's rows are the line's.
    const stix = gray([
        'ffffffffffffffffffffffffffffffffffffffffffffffffff',
        'ffffffffffffffffffffffffffffffffffffffffffffffffff',
        'ffffffffffffffffffffffffffffffffffffffffffffffffff',
        'ffffffffffffffffffffffffffffffffffff977c69d8ffffff',
        'ffffffffffffffffffffffffffffffffffe32effc940ffffff',
        'ffffffffffffffffffffffffffffffffffb24bfff100fbffff',
        'ffffffffffffffffffffffffffffffffffee0db99400fbffff',
        'fffffffcca8571fffffffffdbc686cc0ffffcb726c2effffff',
        'ffffff5558002affffffff7e0dd6c70097fffff747a7ffffff',
        'ffffffffff402affffffea0085ffff6d0df59f4d9bfeffffff',
        'ffffffffff472affffffb100c2ffffaa00c9efffffffffffff',
        'ffffffffff472affffff8c00deffffc700a3ffffffffffffff',
        'ffffffffff472affffff6800ecffffd5007fffffffffffffff',
        'ffffffffff472affffff7000e7ffffd10083ffffffffffffff',
        'ffffffffff472affffff8800d8ffffc2009effffffffffffff',
        'ffffffffff472affffffc700a7ffff9200dcffffffffffffff',
        'fffffffffb220df3fffffe2e45fdf92e4bffffffffffffffff',
        'ffffff782a00002266ffffe4383d3549f0ffffffffffffffff',
        'ffffffffffffffffffffffffffeef1ffffffffffffffffffff',
        'ffffffffffffffffffffffffffffffffffffffffffffffffff',
        'ffffffffffffffffffffffffffffffffffffffffffffffffff',
        'ffffffffffffffffffffffffffffffffffffffffffffffffff',
    ]);
    const nine = readLabel(stix, WHITE, [3, 3, stix.width - 3, stix.height - 3]);
    assert.equal(nine.text, '10^9');
    assert.equal(nine.value, 1e9);
    // an exponent set lower, a quarter of the digits' height above them (the 1 is then nearly as tall as the pair)
    for (const height of [20, 28]) {
        const low = blank(400, 90);
        assert.equal(readLabel(low, WHITE, drawText(low, '10^~5', 40, 40, height, { lift: 0.24 })).value, 100000, `${height} px`);
    }
    // a raised minus sign touching the base doesn't rise above the line, as a digit does: left alone, and unread
    const minus = blank(400, 90);
    assert.equal(readLabel(minus, WHITE, drawText(minus, '10^~-3', 40, 40, 20)).value, null);
});

test('labelGlyphs: a glyph above the line is cut only where what is left keeps to the line', () => {
    const ink = (image, x0, y0, x1, y1) => {
        for (let y = y0; y < y1; y++) {
            for (let x = x0; x < x1; x++)
                image.data.fill(0, 4 * (y * image.width + x), 4 * (y * image.width + x) + 3);
        }
    };
    // a tall letter after "12", its stem from above the digits to their feet and a flag from its top into the
    // digits' rows: no base and exponent, since the stem rises too. The letter stays whole, and is no part of the
    // number.
    const image = blank(300, 90);
    const box = drawText(image, '12', 40, 40, 20);
    ink(image, box[2] + 3, 32, box[2] + 6, 60);
    ink(image, box[2] + 6, 32, box[2] + 14, 46);
    const found = labelGlyphs(image, WHITE, box);
    assert.equal(found.glyphs.length, 2);
    assert.deepEqual(found.raised, []);
    assert.equal(readLabel(image, WHITE, box).text, '12');
    // a slash between two digits, from above them to below their feet: its top end has no ink below it, and what is
    // left of it keeps under the digits' top, but it is on no baseline of theirs. The line stays the slash's, as
    // it was before exponents were cut.
    const slashed = blank(300, 90);
    const one = drawText(slashed, '1', 40, 40, 20);
    for (let y = 34; y < 66; y++)
        ink(slashed, one[2] + 11 - Math.floor((y - 34) / 4), y, one[2] + 14 - Math.floor((y - 34) / 4), y + 1);
    const two = drawText(slashed, '2', one[2] + 17, 40, 20);
    const line = labelGlyphs(slashed, WHITE, [one[0], 34, two[2], 66]);
    assert.deepEqual([line.top, line.height], [34, 32]);
    // a 7's bar ends beyond its stem, with no ink below it, but the 7 doesn't rise above the line: it stays whole
    const seven = blank(300, 90);
    assert.equal(labelGlyphs(seven, WHITE, drawText(seven, '77', 40, 40, 20)).glyphs.length, 2);
});

test('readLabel: a small raised mark is no exponent, and an exponent that is no number makes the label unread', () => {
    // a degree sign's size: under RAISED_MIN_HEIGHT of the digits
    const image = blank(300, 90);
    const box = drawText(image, '30', 40, 40, 20);
    for (let y = 40; y < 47; y++) {
        for (let x = box[2] + 3; x < box[2] + 10; x++)
            image.data.fill(0, 4 * (y * image.width + x), 4 * (y * image.width + x) + 3);
    }
    const read = readLabel(image, WHITE, box);
    assert.equal(read.text, '30?');
    assert.equal(read.value, null);
    const lettered = blank(300, 90);
    const exponent = readLabel(lettered, WHITE, drawText(lettered, '10^$', 40, 40, 20));
    assert.equal(exponent.text, '10^?');
    assert.equal(exponent.value, null);
    // a stroke that comes down from far above the digits (a title's descender) is not raised: it starts elsewhere
    const hung = blank(300, 90);
    const under = drawText(hung, '90', 40, 40, 20);
    for (let y = 10; y < 47; y++) {
        for (let x = under[2] + 3; x < under[2] + 6; x++)
            hung.data.fill(0, 4 * (y * hung.width + x), 4 * (y * hung.width + x) + 3);
    }
    assert.equal(readLabel(hung, WHITE, under).text, '90');
});

test('readLabel: a number with its power of ten after an "e", as R, JMP and Excel write it, at 12 to 28 px', () => {
    const cases = [
        ['1e+05', 1e5], ['1e-03', 0.001], ['0e+00', 0], ['2.5e+03', 2500], ['-4e-02', -0.04], ['1e5', 1e5], ['1e-7', 1e-7], ['6.0E+03', 6000],
        ['2.E+05', 2e5],
    ];
    for (const height of [12, 14, 20, 28]) {
        for (const [text, value] of cases) {
            const image = blank(400, 80);
            const read = readLabel(image, WHITE, drawText(image, text, 40, 20, height));
            assert.equal(read.text, text, `${text} at ${height} px`);
            assert.equal(read.value, value, `${text} at ${height} px`);
        }
    }
});

test('readLabel: a number times a power of ten, spaced or not, with "×" or an "x" typed for it', () => {
    const cases = [
        ['2 × 10^3', '2×10^3', 2000], ['2×10^3', '2×10^3', 2000], ['1.5 × 10^-4', '1.5×10^-4', 0.00015], ['-3 × 10^2', '-3×10^2', -300],
        ['6 x 10^5', '6×10^5', 600000], ['1 × 10^0', '1×10^0', 1],
    ];
    for (const height of [14, 20, 28]) {
        for (const [text, reads, value] of cases) {
            const image = blank(400, 90);
            const box = drawText(image, text, 40, 40, height);
            // (as an x axis's label is looked for, at its tick: the middle of the text)
            const read = readLabel(image, WHITE, box, { near: (box[0] + box[2]) / 2 });
            assert.equal(read.text, reads, `${text} at ${height} px`);
            assert.equal(read.value, value, `${text} at ${height} px`);
            // the number before the sign is the label's too (the box's right edge is the exponent's, to a pixel of gray)
            assert.equal(read.box[0], box[0], `${text} at ${height} px`);
            assert.ok(Math.abs(read.box[2] - box[2]) <= 1, `${text} at ${height} px`);
        }
    }
});

test('regression: a power of ten keeps the number it is multiplied by, across the spaces beside the sign', () => {
    // R's "8 × 10³" beside a y axis, the label looked for from the axis's side: the spaces are wider than a word's
    // gaps, so the label was the "10³" alone and read 1000
    const image = blank(400, 90);
    const box = drawText(image, '8 × 10^3', 40, 40, 20);
    assert.equal(readLabel(image, WHITE, box, { near: box[2] }).value, 8000);
    // a number as far from a power, with no sign between them, is a neighbor: not its factor
    const apart = blank(400, 90);
    const two = drawText(apart, '8  10^3', 40, 40, 20);
    const read = readLabel(apart, WHITE, two, { near: two[2] });
    assert.equal(read.text, '10^3');
    assert.equal(read.value, 1000);
});

test('readLabel: signs that make no number leave the label unread', () => {
    const cases = [['1e', '1e'], ['e5', 'e5'], ['1e+', '1e+'], ['1e-', '1e'], ['2+3', '2+3'], ['+5', '+5'], ['2×10', '2×10'], ['2 × 5^3', '2×5^3'], ['2 × 10^$', '2×10^?'], ['1e+0.5', '1e+0.5']];
    for (const [text, reads] of cases) {
        const image = blank(400, 90);
        const box = drawText(image, text, 40, 40, 20);
        const read = readLabel(image, WHITE, box, { near: (box[0] + box[2]) / 2 });
        assert.equal(read.text, reads, text);
        assert.equal(read.value, null, text);
    }
    // a glyph of a sign's height that is none: an "n", two stems under an arch, where an "e" would stand
    const image = blank(400, 90);
    const one = drawText(image, '1', 40, 40, 20);
    const ink = (x0, y0, x1, y1) => {
        for (let y = y0; y < y1; y++) {
            for (let x = x0; x < x1; x++)
                image.data.fill(0, 4 * (y * image.width + x), 4 * (y * image.width + x) + 3);
        }
    };
    const left = one[2] + 4;
    ink(left, 46, left + 3, 60);
    ink(left + 9, 46, left + 12, 60);
    ink(left, 46, left + 12, 49);
    const five = drawText(image, '5', left + 16, 40, 20);
    const read = readLabel(image, WHITE, [one[0], 40, five[2], 60]);
    assert.equal(read.text, '1?5');
    assert.equal(read.value, null);
});

test('regression: a minus sign touching the "e" before it or the digit after it is cut off', () => {
    // R's png at its default size (12 px type, digits 9 px tall) and at 96 dpi: the minus of "1e-01" ran into the
    // "e" or into the "0", and the pair was one glyph, no sign and no digit
    for (const height of [12, 14, 20]) {
        for (const text of ['1e-~03', '1e~-03']) {
            const image = blank(400, 80);
            const read = readLabel(image, WHITE, drawText(image, text, 40, 20, height));
            assert.equal(read.text, '1e-03', `${text} at ${height} px`);
            assert.equal(read.value, 0.001, `${text} at ${height} px`);
        }
    }
    // the labels themselves, as R drew them (Helvetica): "1e-03" with digits 12 px tall, its minus run into the 0,
    // and "1e-01" at the default size, digits 9 px tall, its "e" run into the minus
    const minusZero = gray([
        'ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff',
        'ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff',
        'ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff',
        'fffffffffffffff6fffffffffffffffffffffffffffffffffffffffffffefffffffffffffffffeffffffffffffff',
        'ffffffffffff9c0affffffffffffffffffffffffffffffffffffffa03f0835befffffff98e370a3ec8ffffffffff',
        'ffffffffffac0609ffffffffffffffffffffffffffffffffffffc10695d06a07d0ffff930f97c7590ad6ffffffff',
        'fffffff05a002709ffffffffffffffffffffffffffffffffffff295bffffff2c62fff61876fffff20b84ffffffff',
        'ffffffbd1eaf8e0afffffffff4691f134fe1ffffffffffffffe300acffffff7b18ffffeaf2ffffe10489ffffffff',
        'fffffff0f8ff8e0afffffffe38147bab3c15eeffffffffffffbf00c7ffffff9503fdfffffffcc54b17deffffffff',
        'ffffffffffff8d09ffffffa900b4fffff50b7affffffffffffb100d9ffffffa600eaffffff8900009fffffffffff',
        'ffffffffffff8d09ffffff6b0fc9cccccc3b40ffeae6e6e6f3b500dcffffffa800e4ffffffdecc6f0688ffffffff',
        'ffffffffffff8d09ffffff4b00000000000020ff2c00000081cd00cbffffff9700f1ffffffffffff4e1bffffffff',
        'ffffffffffff8d09ffffff5b20f4f5f5f5f5f6ffa7949494cae500b4ffffff8012fff3bef2ffffff9600e2ffffff',
        'ffffffffffff8d09ffffff8600cafffffe6491fffffffffffffe286bffffff3651ffd5008cffffff4f08fcffffff',
        'ffffffffffff8d09ffffffed1138b9ec7f02cbffffffffffffff940bb4f68f08daffff3b1cbbeb8b0579ffffffff',
        'ffffffffffff8d09ffffffffcc3000013298fffffffffffffffffb6f040129a0ffffffe54800000679faffffffff',
        'ffffffffffffffffffffffffffffeee7fffffffffffffffffffffffff7e1ffffffffffffffedd8faffffffffffff',
        'ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff',
        'ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff',
        'ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff',
    ]);
    assert.equal(readLabel(minusZero, WHITE, [0, 0, minusZero.width, minusZero.height]).text, '1e-03');
    const eMinus = gray([
        'ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff',
        'ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff',
        'fffffffffffffffffbfefffffffffffffffffffffffffffffffefffffffffffffbfeffffff',
        'ffffffffffffffe81ddcffffffffffffffffffffffffffbf391853f3ffffffe81ddcffffff',
        'ffffffffffffa42b05dcffffffffffffffffffffffffec19c6f66866ffffa42b05dcffffff',
        'fffffffffff943d715dcffffeb510f2ec3ffffffffffb840ffffdc0ffff943d715dcffffff',
        'ffffffffffffffff15dcffff4e5ce49a23ffffffffff9a5bfffff500f9ffffff15dcffffff',
        'ffffffffffffffff15dcffee02ddf9f93ed3ffffffff8666fffffe00edffffff15dcffffff',
        'ffffffffffffffff15dcffcf010d0d0d0d936a0e0e699959fffff107ffffffff15dcffffff',
        'ffffffffffffffff15dcffe005effffff4ffd9c1c1d9b43bffffd423ffffffff15dcffffff',
        'ffffffffffffffff15dcfffd268bffee17cdfffffffff615c7fb6872ffffffff15dcffffff',
        'ffffffffffffffff15dcffffc923252787ffffffffffffa60d2141f2ffffffff15dcffffff',
        'ffffffffffffffffffffffffffffeafefffffffffffffffffbeeffffffffffffffffffffff',
        'ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff',
        'ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff',
        'ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff',
    ]);
    assert.equal(readLabel(eMinus, WHITE, [0, 0, eMinus.width, eMinus.height]).text, '1e-01');
    // a plus's arms are flat and at mid-height too, a minus's width each (R's "1e+05", 12 px): it stays whole, since
    // what the first leaves is no digit and what the second leaves no "e"
    const plus = gray([
        'ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff',
        'ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff',
        'ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff',
        'fffffffffffffff6fffffffffffffffffffffffffffffffffffffffffffffffffffeffffffffffffffffffffffffffffffff',
        'ffffffffffff9c0affffffffffffffffffffffffffffffffffffffffffffffa03f0835befffffff4282727272764ffffffff',
        'ffffffffffac0609ffffffffffffffffffffffffffffffffffffffffffffc10695d06a07d0ffffc700737f7f7fa3ffffffff',
        'fffffff05a002709ffffffffffffffffffffffffffffffb00affffffffff295bffffff2c62ffff990cfeffffffffffffffff',
        'ffffffbd1eaf8e0affffffffff9d300c35b2ffffffffffac00ffffffffe300acffffff7b18ffff6b3bfffeffffffffffffff',
        'fffffff0f8ff8e0affffffff8b0357ad6e01aeffffffffac00ffffffffbf00c7ffffff9503fdff3d40610f1e8bfeffffffff',
        'ffffffffffff8d09ffffffef0f5fffffff5525ffb557573a00565757ebb100d9ffffffa600eaff0f0a7c99360083ffffffff',
        'ffffffffffff8d09ffffffc00094cccccc7f00ebab40402b003f4040e8b500dcffffffa800e4fb9fc6ffffff3d09f7ffffff',
        'ffffffffffff8d09ffffffa000000000000000cbffffffac00ffffffffcd00cbffffff9700f1ffffffffffff9300d0ffffff',
        'ffffffffffff8d09ffffffb100c3f5f5f5f5f5fdffffffac00ffffffffe500b4ffffff8012fff5cff4ffffffa000d2ffffff',
        'ffffffffffff8d09ffffffda0075ffffff9f5df8ffffffac00fffffffffe286bffffff3651ffd10088ffffff4e0df7ffffff',
        'ffffffffffff8d09ffffffff541395eeb90d79fffffffff5e0ffffffffff940bb4f68f08dafffe351ab8ea820681ffffffff',
        'ffffffffffff8d09ffffffffef5e04001965f6fffffffffffffffffffffffb6f040129a0ffffffe24500000c83feffffffff',
        'fffffffffffffffffffffffffffff9defefffffffffffffffffffffffffffffff7e1ffffffffffffffefdbfdffffffffffff',
        'ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff',
        'ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff',
        'ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff',
    ]);
    const read = readLabel(plus, WHITE, [0, 0, plus.width, plus.height]);
    assert.equal(read.text, '1e+05');
    assert.equal(read.value, 100000);
});

test('regression: a "4" keeps the tip of its bar, which is as wide and as flat as a minus sign', () => {
    // Gill Sans's "1454" at 17 px: the bar of its 4 starts 3 px left of the diagonal, a minus sign's width at
    // mid-height, and the cut for a minus touching its digit made the number "1-154"
    const gill = gray([
        'ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff',
        'ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff',
        'ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff',
        'ffffffffffffffffffffffffffffedf7ffffffffffffffffffffffffffffffedf7ffffffff',
        'ffffff6000d4ffffffffffffffe00db6ffffa80000000000ffffffffffffe00db6ffffffff',
        'ffffff5f00d2fffffffffffff53200b6ffffa7003b585860fffffffffff53200b6ffffffff',
        'ffffff5f00d2ffffffffffff5d0000b6ffffa700b4ffffffffffffffff5d0000b6ffffffff',
        'ffffff5f00d2ffffffffff90001c00b6ffffa700b4ffffffffffffff90001c00b6ffffffff',
        'ffffff5f00d2ffffffffbf00558600b6ffffa70058a1e2ffffffffbf00558600b6ffffffff',
        'ffffff5f00d2ffffffe10d26f18800b6ffffa700000000a7ffffe10d26f18800b6ffffffff',
        'ffffff5f00d2fffff53200558142005ab4ffe5c5b05a0000def53200558142005ab4ffffff',
        'ffffff5f00d2ffffa3000000000000006dffffffffff42009da3000000000000006dffffff',
        'ffffff5f00d2fffff6e7e7e7e77a00a4f2ffffffffff820087f6e7e7e7e77a00a4f2ffffff',
        'ffffff5f00d2ffffffffffffff8800b6ffe5cbffffdf1600b0ffffffffff8800b6ffffffff',
        'ffffff5f00d2ffffffffffffff8800b6ffc5002e3800002ef9ffffffffff8800b6ffffffff',
        'ffffff6000d4ffffffffffffff8900b7ffe5692200005ae6ffffffffffff8900b7ffffffff',
        'fffffffffffffffffffffffffffffffffffffffff0fcffffffffffffffffffffffffffffff',
        'ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff',
        'ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff',
        'ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff',
    ]);
    const read = readLabel(gill, WHITE, [0, 0, gill.width, gill.height]);
    assert.equal(read.text, '1454');
    assert.equal(read.value, 1454);
    // Arial's "10" with a raised 4, 16 px: the same tip in an exponent read as a negative power, "10^-?"
    const arial = gray([
        'ffffffffffffffffffffffffffffffffffffffffffffffffffffffffff',
        'ffffffffffffffffffffffffffffffffffffffffffffffffffffffffff',
        'ffffffffffffffffffffffffffffffffffffffffffffffffffffffffff',
        'ffffffffffffffffffffffffffffffffffffffffffffffdd6cffffffff',
        'fffffffffffffff7fffffffffffffffcfffffffffffffb3b16ffffffff',
        'ffffffffffff8e00f4ffffffff9b3d0035b6ffffffff854d16ffffffff',
        'ffffffffffa30000f4ffffffb20060a33b00c4ffffca2ed516ffffffff',
        'ffffffeb51000000f4fffffa1638fffff40d51fff42ad0de16ffffffff',
        'ffffffae007c7900f4ffffd00094ffffff6200fc9f0d3d35004fffffff',
        'ffffffdedfff8000f4ffffae00b4ffffff8400ebefd4d4b716d8ffffff',
        'ffffffffffff8000f4ffffa100c5ffffff9500d6ffffffde16ffffffff',
        'ffffffffffff8000f4ffffa600c8ffffff9700d1ffffffffffffffffff',
        'ffffffffffff8000f4ffffbb00b9ffffff8600dcffffffffffffffffff',
        'ffffffffffff8000f4ffffd1009effffff6900f5ffffffffffffffffff',
        'ffffffffffff8000f4fffff8164bfffffb163bffffffffffffffffffff',
        'ffffffffffff8000f4ffffff840080c85a00cbffffffffffffffffffff',
        'ffffffffffff8100f5fffffff76800002699ffffffffffffffffffffff',
        'fffffffffffffffffffffffffffff5dfffffffffffffffffffffffffff',
        'ffffffffffffffffffffffffffffffffffffffffffffffffffffffffff',
        'ffffffffffffffffffffffffffffffffffffffffffffffffffffffffff',
        'ffffffffffffffffffffffffffffffffffffffffffffffffffffffffff',
    ]);
    const power = readLabel(arial, WHITE, [3, 5, 19, 17]);
    assert.equal(power.text, '10^4');
    assert.equal(power.value, 10000);
});

test('readLabel: digits lower than their line are no signs', () => {
    // 0.85 of the line the axis's labels share, as a taller neighbor leaves them: an "e" may be as tall, and a 6 or
    // an 8 is something like one, but more like itself
    for (const height of [12, 15, 17, 20, 22]) {
        for (const text of ['606', '808', '909']) {
            const image = blank(400, 90);
            const box = drawText(image, text, 40, 40, height, { font: DIGIT_TEMPLATES.find(f => f.name === 'Helvetica') });
            const read = readLabel(image, WHITE, box, { line: [40 - Math.round(0.17 * height), 40 + height] });
            assert.equal(read.text, text, `${text} at ${height} px`);
        }
    }
});

test('readAxisLabels: axes in scientific notation, a log axis with "e" and a linear one with "×"', () => {
    const { image } = labeled({ xLabels: ['1e-03', '1e-01', '1e+01', '1e+03'], yLabels: ['0', '2 × 10^5', '4 × 10^5', '6 × 10^5'], height: 14 });
    const x = axisValues(image, 'x');
    assert.deepEqual(x.texts, ['1e-03', '1e-01', '1e+01', '1e+03']);
    assert.equal(x.scale.type, 'log');
    assert.deepEqual(x.scale.values, [0.001, 0.1, 10, 1000]);
    const y = axisValues(image, 'y');
    assert.deepEqual(y.texts, ['6×10^5', '4×10^5', '2×10^5', '0']);
    assert.equal(y.scale.type, 'linear');
    assert.deepEqual(y.scale.values, [600000, 400000, 200000, 0]);
});

test('readAxisLabels: an exponent touching its base is cut on the line the axis\'s labels share, too', () => {
    const { image } = labeled({ xLabels: ['10^0', '10^~1', '10^2', '10^~3'], yLabels: ['10^-2', '10^-1', '10^~0', '10^1', '10^~2'], height: 16 });
    const x = axisValues(image, 'x');
    assert.deepEqual(x.texts, ['10^0', '10^1', '10^2', '10^3']);
    assert.deepEqual(x.scale.values, [1, 10, 100, 1000]);
    const y = axisValues(image, 'y');
    assert.deepEqual(y.texts, ['10^2', '10^1', '10^0', '10^-1', '10^-2']);
    assert.equal(y.scale.count, 5);
});

test('readAxisLabels: log axes labeled with powers of ten', () => {
    const { image } = labeled({ xLabels: ['10^0', '10^1', '10^2', '10^3'], yLabels: ['10^-2', '10^-1', '10^0', '10^1', '10^2'], height: 16 });
    const x = axisValues(image, 'x');
    assert.deepEqual(x.texts, ['10^0', '10^1', '10^2', '10^3']);
    assert.equal(x.scale.type, 'log');
    assert.deepEqual(x.scale.values, [1, 10, 100, 1000]);
    const y = axisValues(image, 'y');
    assert.equal(y.scale.type, 'log');
    assert.deepEqual(y.scale.values, [100, 10, 1, 0.1, 0.01]);
    assert.equal(y.scale.count, 5);
});

test('regression: a label with an exponent is its own tick\'s, and the scale gives the minor ticks round values', () => {
    // primates1's x axis: "10" was boxed without its raised 2, the box's center was nearer the minor tick of 90 than
    // its own, and three labels 174 and 159 px apart fit no scale
    const decade = 160; // px
    const at = v => 120 + decade * Math.log10(v / 100);
    const values = [80, 90, 100, 200, 500, 900, 1000, 2000, 5000, 9000, 10000];
    const spec = {
        width: 520, height: 400, background: [255, 255, 255], blur: 0.5, families: [],
        axes: { x: { pos: 330.5, from: 79, to: 500, ticks: values.map(at) }, y: { pos: 80.5, from: 20, to: 331.5, ticks: [310, 220, 130, 40] }, label: [0, 0] },
    };
    const { image } = synthGraph(spec);
    for (const [text, v] of [['10^2', 100], ['10^3', 1000], ['10^4', 10000]]) {
        const probe = drawText(blank(200, 60), text, 0, 20, 20);
        drawText(image, text, at(v) - (probe[2] - probe[0]) / 2, 352, 20);
    }
    ['0', '1', '2', '3'].forEach((text, i) => drawText(image, text, 64, spec.axes.y.ticks[i] - 7, 14, { right: true }));
    const axes = detectAxes(image);
    assert.deepEqual(axes.x.ticks.map(t => Boolean(t.label)), values.map(v => [100, 1000, 10000].includes(v)));
    const { scale } = readAxisLabels(image, axes.backgrounds, axes.x, 'x');
    assert.equal(scale.type, 'log');
    assert.deepEqual(scale.values, values);
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

// A log axis as primates2's x axis has it: 426.6 px a decade, a tick at each multiple of a power of ten given, and
// a value read at some of them. `read` maps a tick's true value to what its label reads as.
function logTicks(multiples, read, decade = 426.6) {
    return multiples.map(v => ({ pos: 228.8 + decade * Math.log10(v / 0.01), value: read[v] ?? null }));
}
const MINOR = [0.009, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3];

test('consistentValues: two labels on a log axis are believed when its minor ticks fall on multiples of powers of ten', () => {
    const scale = consistentValues(logTicks(MINOR, { 0.01: 0.01, 0.1: 0.1 }));
    assert.equal(scale.type, 'log');
    assert.deepEqual(scale.values, MINOR);
    assert.deepEqual(scale.agree, MINOR.map(v => v === 0.01 || v === 0.1));
    assert.equal(scale.count, 2);
    assert.equal(scale.minor, 11);
    // a false tick among them is the scale's to value (4.5 of a decade is no multiple), as one label off a scale is
    const withFalse = consistentValues([...logTicks(MINOR, { 0.01: 0.01, 0.1: 0.1 }), { pos: 228.8 + 426.6 * Math.log10(4.5), value: null }].sort((a, b) => a.pos - b.pos));
    assert.equal(withFalse.minor, 11);
    assert.equal(withFalse.values[5], 0.045);
    // three labels need no minor ticks
    assert.equal(consistentValues([1, 10, 100].map((v, i) => ({ pos: 50 + 100 * i, value: v }))).minor, 0);
});

test('consistentValues: two labels misread, or too little to check them by, give no scale', () => {
    const none = (read, multiples = MINOR, decade) => assert.equal(consistentValues(logTicks(multiples, read, decade)), null, JSON.stringify(read));
    // backwards, three decades for one, the leading digit wrong in both
    none({ 0.01: 0.1, 0.1: 0.01 });
    none({ 0.01: 0.01, 0.1: 10 });
    none({ 0.01: 0.07, 0.1: 0.7 });
    none({ 0.01: 0.02, 0.1: 0.2 });
    // a third label that reads off the scale of the other two: one in three, as without minor ticks
    none({ 0.01: 0.01, 0.1: 0.1, 0.05: 0.3 });
    // minor ticks at 2 and 5 alone: read backwards they fall on 5 and 2, so they check nothing, right labels or not
    const sparse = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1];
    none({ 0.01: 0.01, 0.1: 0.1 }, sparse);
    none({ 0.01: 0.1, 0.1: 0.01 }, sparse);
    // a decade of 40 px: a pixel is 0.025 of it, and 9 and 10 are 0.046 apart
    none({ 0.01: 0.01, 0.1: 0.1 }, MINOR, 40);
    // as many ticks off the multiples as on them
    const half = logTicks([...MINOR, 0.015, 0.025, 0.035, 0.045, 0.055, 0.065], { 0.01: 0.01, 0.1: 0.1 }).sort((a, b) => a.pos - b.pos);
    assert.equal(consistentValues(half), null);
    // a linear axis from 1 to 10 with a tick at each whole number, labeled at its ends
    assert.equal(consistentValues([1, 2, 3, 4, 5, 6, 7, 8, 9, 10].map(v => ({ pos: 100 + 45 * v, value: v === 1 || v === 10 ? v : null }))), null);
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
    // grader1's labels stand off the ends of its gridlines (no axis lines, no ticks)
    assert.deepEqual(values('grader1', 'x'), [0, 15, 30, 45]);
    assert.deepEqual(values('grader1', 'y'), [45, 30, 15, 0]);
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

const BENCH_R = new URL('../bench-r/', import.meta.url).pathname;

test('bench-r: R\'s labels in scientific notation, with "e" and with "×", down to its default size', { skip: !existsSync(BENCH_R + 'log-e.png') && 'bench-r absent' }, () => {
    const read = (name, which) => axisValues(readImage(`${BENCH_R}${name}.png`), which);
    const decades = [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000];
    for (const name of ['log-e', 'log-e-hi', 'log-e-default']) {
        const x = read(name, 'x');
        // (R leaves out the labels that would overlap: every other one)
        assert.deepEqual(x.texts.filter(Boolean), ['1e-03', '1e-01', '1e+01', '1e+03', '1e+05'], name);
        assert.equal(x.scale.type, 'log');
        assert.deepEqual(x.scale.values, decades, name);
        // the minus of "1e-01" touches the 0 at 96 dpi and the "e" at the default size
        assert.deepEqual(read(name, 'y').texts, ['1e+06', '1e+05', '1e+04', '1e+03', '1e+02', '1e+01', '1e+00', '1e-01'], name);
    }
    assert.deepEqual(read('linear-e', 'x').scale.values, [0, 100000, 200000, 300000, 400000, 500000, 600000]);
    assert.deepEqual(read('jmp-e', 'x').scale.values, [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]);
    assert.deepEqual(read('excel-E', 'y').texts, ['8.0E+03', '6.0E+03', '4.0E+03', '2.0E+03']);
    for (const name of ['times-r96', 'times-r144', 'times-r220']) {
        // plotmath's spacing below, typed spaces beside: wider than a word's gaps
        assert.deepEqual(read(name, 'x').texts, ['0', '2×10^5', '4×10^5', '6×10^5'], name);
        assert.deepEqual(read(name, 'y').scale.values, [8000, 6000, 4000, 2000], name);
    }
    assert.deepEqual(read('times-x', 'x').scale.values, [0, 150000, 300000, 450000, 600000]);
    assert.deepEqual(read('times-x', 'y').texts, ['8×10^3', '6×10^3', '4×10^3', '2×10^3']);
    // at 72 dpi two of plotmath's exponents touch the 0 under them, and are cut from it
    assert.deepEqual(read('times-r72', 'x').texts, ['0', '2×10^5', '4×10^5', '6×10^5']);
    assert.deepEqual(read('times-log', 'x').scale.values, [300, 1000, 3000, 10000]);
    // … and the typed "³" is under an exponent's height, so the labels stay unread: no scale, no wrong value
    assert.equal(read('times-r72', 'y').scale, null);
});

const NATURE = new URL('../bench-nature/', import.meta.url).pathname;
const nature = name => existsSync(NATURE) && readdirSync(NATURE).find(f => f.startsWith(`${name}-`) && f.endsWith('.png'));

test('bench-nature: phase1\'s and primates\' powers of ten, with minor ticks and a negative exponent', { skip: !nature('primates1') && 'bench-nature absent' }, () => {
    const read = (name, which) => axisValues(readImage(NATURE + nature(name)), which);
    assert.deepEqual(read('phase1', 'x').scale.values, [10, 100, 1000, 10000]);
    const p1 = read('primates1', 'x');
    assert.deepEqual(p1.texts.filter(Boolean), ['10^2', '10^3', '10^4']);
    assert.deepEqual(p1.scale.values, [70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 20000]);
    const p2 = read('primates2', 'y');
    assert.deepEqual(p2.texts.filter(Boolean), ['10^2', '10^1', '10^0', '10^-1']);
    assert.equal(p2.scale.type, 'log');
    assert.deepEqual([p2.scale.values[0], p2.scale.values[1], p2.scale.values[19], p2.scale.values[28], p2.scale.values[34]], [200, 100, 1, 0.1, 0.04]);
    // two labels, the minus sign touching its 2: read, and checked by the eleven minor ticks
    const x = read('primates2', 'x');
    assert.deepEqual(x.texts.filter(Boolean), ['10^-2', '10^-1']);
    assert.deepEqual(x.scale.values, [0.009, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3]);
    assert.equal(x.scale.minor, 11);
});

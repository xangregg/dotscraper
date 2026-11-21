// The shape of each digit in the fonts figures are set in, for labels.js to tell digits apart by: each digit drawn by
// the browser at several sizes and subpixel places, cut out by the reader's own segmentation (labelGlyphs) and
// averaged on its grid (glyphGrid). With them the signs of a number in scientific notation (SIGNS). tools/make-digit-templates.mjs opens this page and writes the result to
// src/core/digit-templates.js.
//
// The fonts are this machine's: one that is missing is left out and named in `missing`. (A canvas draws a missing
// font in its fallback without saying so, which is why each is tested by its width against the fallbacks'.)

import { labelGlyphs, glyphGrid } from '../src/core/labels.js';
import { TEMPLATE_WIDTH, TEMPLATE_HEIGHT } from '../src/core/digit-templates.js';

// Sans-serifs first, the ones figures are mostly set in: Helvetica and Arial (R, Prism, Illustrator, JMP), Verdana
// (close to matplotlib's DejaVu Sans), with their bolds; then serifs and monospaced. No font with old-style figures
// (Georgia): their digits aren't of one height, which the reader takes digits to be.
const FONTS = [
    ['Helvetica', 400], ['Helvetica', 700], ['Arial', 400], ['Arial', 700], ['Helvetica Neue', 400], ['Arial Narrow', 400],
    ['Verdana', 400], ['Tahoma', 400], ['Trebuchet MS', 400], ['Lucida Grande', 400], ['Roboto', 400], ['Gill Sans', 400],
    ['Avenir', 400], ['Times New Roman', 400], ['Times New Roman', 700], ['STIX Two Text', 400], ['Courier New', 400],
    ['Menlo', 400],
];
const SIZES = [12, 14, 16, 18, 21, 25, 30, 40, 56]; // px, font size: digits 0.7 of it tall, 8–40 px as in figures
const OFFSETS = [0, 0.5]; // px, each way: where the digit falls on the pixel grid
// "1e+05" and "2 × 10³": the letter of the exponent in both cases, the plus, and the multiplication sign with the
// letter typed for it
const SIGNS = 'eE+×x';
const PROBE = '0123456789mmWWll';
const WHITE = [[255, 255, 255]];

const canvas = document.createElement('canvas');
const ctx = canvas.getContext('2d', { willReadFrequently: true });

// whether a family is installed: its text is not as wide as both fallbacks'
function installed(family) {
    return ['monospace', 'serif'].some(fallback => {
        ctx.font = `40px ${fallback}`;
        const width = ctx.measureText(PROBE).width;
        ctx.font = `40px "${family}", ${fallback}`;
        return ctx.measureText(PROBE).width !== width;
    });
}

// One character's grid at one size and place, or null when the reader finds no single glyph. A sign is taken
// `whole`, however the reader cuts it: alone, with no digit to give the line's height, a plus has the shape of a
// minus touching a digit.
function drawn(family, weight, digit, size, dx, dy, whole = false) {
    canvas.width = 3 * size;
    canvas.height = 3 * size;
    ctx.fillStyle = '#fff';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = '#000';
    ctx.font = `${weight} ${size}px "${family}"`;
    ctx.fillText(digit, size + dx, 2 * size + dy);
    const image = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const found = labelGlyphs(image, WHITE, [0, 0, canvas.width, canvas.height]);
    if (!found || (!whole && found.glyphs.length !== 1))
        return null;
    const all = found.glyphs;
    return glyphGrid(found.cover, {
        x0: Math.min(...all.map(g => g.x0)), y0: Math.min(...all.map(g => g.y0)),
        x1: Math.max(...all.map(g => g.x1)), y1: Math.max(...all.map(g => g.y1)),
    });
}

// a character's mean grid over the sizes and places
function meanGrid(family, weight, ch, whole) {
    const grids = [];
    for (const size of SIZES) {
        for (const dx of OFFSETS) {
            for (const dy of OFFSETS) {
                const grid = drawn(family, weight, ch, size, dx, dy, whole);
                if (grid)
                    grids.push(grid);
            }
        }
    }
    const mean = new Float32Array(TEMPLATE_WIDTH * TEMPLATE_HEIGHT);
    for (const grid of grids) {
        for (let p = 0; p < mean.length; p++)
            mean[p] += grid[p] / grids.length;
    }
    return mean;
}

// a template as the reader has it, enlarged
function view(mean) {
    const out = document.createElement('canvas');
    out.width = TEMPLATE_WIDTH;
    out.height = TEMPLATE_HEIGHT;
    out.style.width = `${3 * TEMPLATE_WIDTH}px`;
    const data = new ImageData(TEMPLATE_WIDTH, TEMPLATE_HEIGHT);
    mean.forEach((v, p) => data.data.set([255 * (1 - v), 255 * (1 - v), 255 * (1 - v), 255], 4 * p));
    out.getContext('2d').putImageData(data, 0, 0);
    return out;
}

const fonts = [];
const missing = [];
const figures = document.getElementById('figures');
for (const [family, weight] of FONTS) {
    const name = weight === 400 ? family : `${family} bold`;
    if (!installed(family)) {
        missing.push(name);
        continue;
    }
    const digits = {};
    const signs = {};
    const row = document.createElement('div');
    row.append(`${name} `);
    for (const digit of '0123456789') {
        const mean = meanGrid(family, weight, digit, false);
        digits[digit] = [...mean];
        row.append(view(mean));
    }
    for (const sign of SIGNS) {
        const mean = meanGrid(family, weight, sign, true);
        signs[sign] = [...mean];
        row.append(view(mean));
    }
    figures.append(row);
    fonts.push({ name, digits, signs });
}
document.getElementById('summary').textContent = `${fonts.length} fonts${missing.length ? `; missing: ${missing.join(', ')}` : ''}`;
window.digitFonts = { fonts, missing };

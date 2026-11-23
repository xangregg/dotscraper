// Reading the numbers of tick labels, so that an axis's values needn't be typed: a small reader of our own, not
// general OCR. A tick label is a few glyphs from a small alphabet (digits, a decimal point, a minus sign, thousands
// separators) in a clean font, and the axis checks the reading: the values of ticks follow a linear or log scale,
// so three labels read right give the scale, a misreading shows as a tick that doesn't fit, and a label that
// can't be read gets the scale's value (consistentValues). Without a scale nothing is checked, and a caller should
// offer nothing as read: two labels and category names end there. A number with an exponent ("10" with a raised 3,
// as log axes are labeled) is read as its power, and so is scientific notation: "1e+05", and "2 × 10" with a raised 3.
//
// A label's box from axes.js is where to look, not what to read: boxes are cut short by the search band (cells1's
// "100" boxed as "00"), lose a part across a decimal point (autocor1's "1.0" as "0") and take in a neighbor (JMP's
// "60" with two letters of the rotated axis title). So the label is found again from the pixels: the line of text
// through the box, cut into glyphs at empty columns and into words at wide gaps, and the word at the tick. The
// labels of one axis share a line (labelOptions), which tells a foreign glyph by its reach beyond it: a letter of
// the plot's title hanging into "90", the rotated title beside "60", a currency sign.
//
// Digits are told apart by correlation with templates of the fonts figures are set in (digit-templates.js), each
// glyph scaled to one height, and so are the signs of scientific notation (e, E, +, ×). Anything else that is no
// point, comma or minus makes its label unread, never guessed; a sign alike in every label of the axis (a "$"
// before, a "%" after) is left out (unitSigns).

import { DIGIT_TEMPLATES, TEMPLATE_WIDTH, TEMPLATE_HEIGHT } from './digit-templates.js';
import { median, upperMedian } from './stats.js';

const INK_MIN_CONTRAST = 60; // 0–255 units from the page: a label's darkest pixel is at least this far, or it has no ink
const GLYPH_INK = 0.3; // 0–1 of the label's darkest: a row or column with a pixel this dark has ink
// Glyphs of one label are closer than this × the digits' height (Helvetica: digits 0.08–0.15 apart, 0.3 across a
// decimal point, a minus sign 0.2 before its number); the next word, a rotated title or a tick mark is farther
const WORD_GAP = 0.45;
// Small digits touch: at 13 px a "4" and a "0" share a column of gray. A glyph wider than SPLIT_MIN_WIDTH × H and
// not flat (a dash) is cut into round(width / DIGIT_PITCH × H) at its lightest columns, looked for within SPLIT_SEARCH
// × H of even cuts (Helvetica's digits: ink 0.65–0.75 H wide, 0.77 H from one to the next)
const SPLIT_MIN_WIDTH = 1.1;
const DIGIT_PITCH = 0.77;
const SPLIT_SEARCH = 0.2;
const SPLIT_MIN_HEIGHT = 0.5; // × H: lower glyphs are dashes, wide as they are
const GROW_HEIGHTS = 5; // a label is looked for this many box heights beyond its box, along its line
// A glyph running farther than this × H beyond the digits' line, above or below, is no part of a number
const LINE_OVER = 0.15;
const LINE_OVER_MIN = 2; // px, at least
const LINE_REACH = 0.4; // × H: how far beyond the line a glyph is followed (a comma's tail is 0.2 below it)
// An exponent's glyphs are raised: each ends at most RAISED_MAX_BOTTOM × H below the digits' top and starts at most
// RAISED_MAX_RISE × H above it, and one of them is RAISED_MIN_HEIGHT × H tall or more (a digit, where a degree sign
// or a tick mark at the top isn't). Measured on Nature's figures (phase1, primates1–2): digits 0.65–0.68 H tall, from
// 0.36–0.39 H above the top to 0.25–0.32 H below it, and a negative exponent's minus at the top itself. The limits
// leave room for exponents set lower, as matplotlib's are said to be: not measured, no local figure has one
const RAISED_MAX_BOTTOM = 0.6;
const RAISED_MAX_RISE = 0.6;
const RAISED_MIN_HEIGHT = 0.45;
// A glyph by its size against the digits' height H and its place in their line (0 at the top, 1 at the baseline):
const DIGIT_MIN_HEIGHT = 0.8; // × H: digits are the tall glyphs
const POINT_MAX_SIZE = 0.3; // × H, each way: a decimal point, on the baseline
const POINT_MIN_BOTTOM = 0.8;
const COMMA_MAX_WIDTH = 0.3; // × H: a thousands separator, hanging below the baseline
const COMMA_MAX_HEIGHT = 0.5;
const COMMA_MIN_TOP = 0.6;
// A comma's tail ends at least this × H (and a pixel) below the digits' feet, a point at most (a comma's is 0.2)
const COMMA_MIN_UNDER = 0.06;
const MINUS_MAX_HEIGHT = 0.25; // × H: a minus sign (hyphen, en dash or minus), wide and flat at mid-height
const MINUS_MIN_WIDTH = 0.25;
const MINUS_CENTER = [0.3, 0.8];
// Correlation with the best template below which a tall glyph is no digit. Of the 1,282 labels of the local images
// that their axis's scale confirms (2026-11-21), the least sure digit scores 0.72 and all but that one 0.82 or more;
// letters reach 0.6
const DIGIT_MIN_SCORE = 0.7;
// Scientific notation, "1e+05" as R writes it and "2 × 10³": its signs (e, E, +, ×, and the x typed for ×) are told
// by templates as digits are, a glyph being the sign it is most like when that is at least SIGN_MIN_SCORE and more
// than its likeness to any digit. "e", "+" and "×" are lower than digits, at most SIGN_MAX_HEIGHT × H and at least
// SIGN_MIN_HEIGHT, and "E" as tall as they are. Measured on bench-r, R's renderings with digits 9–28 px tall, in the
// 160 labels read (2026-11-21): e scores 0.87–0.97 at 0.68–0.78 H, + 0.90–0.99 at 0.60–0.78, × 0.83–0.99 at
// 0.53–0.65 (a typed x 0.97 at 0.75), E 0.95–0.97. The score alone doesn't tell a sign from every letter (of 16
// fonts' letters a "c" reaches 0.85 as an e, an "F" 0.83 as an E, a "÷" 0.89 as a plus): what a label reads as
// must still be a number, and the axis's scale must agree. (Was 0.75: the e of R's labels turned along the y axis,
// 7 px tall, scores 0.74999 once turned back, and log-e-turned's eight labels read "1?+06". At 0.72 the label
// benchmark differs in that axis alone.)
const SIGN_MIN_SCORE = 0.72;
const SIGN_MAX_HEIGHT = 0.9;
const SIGN_MIN_HEIGHT = 0.45;
// A multiplication sign stands apart from its numbers: glyphs this × H apart or less are one word when one of them is
// the sign (bench-r: 0.30–0.65, plotmath's spacing and typed spaces in Helvetica; 0.67 in another rendering), where
// WORD_GAP ends any other word
const TIMES_MAX_GAP = 0.9;
const SIGN_READ = { x: '×' }; // a template's character to the sign it is read as, where they differ
const SIGN_LIKENESS = 0.85; // correlation of two labels' glyphs at which they are one unit sign
// A label found cleanly on its own, its digits within this of the shared line's height, is read where it was found
const OWN_LINE_TOL = 1; // px
const SHARED_LINE_MIN = 2; // labels read cleanly, of one axis, for their line to be taken for all of its labels
// consistentValues: a read value agrees with a scale when the scale puts it within SCALE_TOL of the least gap
// between ticks, or SCALE_POS_TOL if that is more (a tick's position is known to about a pixel), of its tick
const SCALE_TOL = 0.02;
const SCALE_POS_TOL = 1; // px
const SCALE_MIN_AGREE = 3; // labels that must agree for a scale to be believed (two always agree)
// and of the labels read as numbers, those that don't fit it are at most this share of those that do: a category
// axis with numbers for names (doses 0, 5, 20, 50, 100, 200) has three on a log scale and three off it
const SCALE_MAX_DISAGREE = 1 / 3;
// Two labels always agree, so a scale through two needs another check, and a log axis has one: its minor ticks, at 2
// to 9 times each power of ten, fall on those values only under the right scale (logByMinorTicks). The unread ticks
// that fall on them must show at least this many of the eight multiples: read backwards (10 to 1 for 1 to 10) a scale
// still puts the ticks of 2 and 5 on 5 and 2, and one of twice the decades puts 2 and 3 on 4 and 9
const MINOR_MIN_MULTIPLES = 3;
// … with the tolerance at most this many decades: 9 and 10 are 0.046 apart, and the wrong scales above miss a
// multiple by 0.046 or more
const MINOR_MAX_TOL = 0.02;

/** @typedef {import('./image.js').RgbaImage} RgbaImage */

/**
 * How dark each pixel of a rectangle of the image is.
 *
 * @typedef {object} Cover
 * @property {number} x0      px, the rectangle's left column in the image
 * @property {number} y0      px, its top row
 * @property {number} width   px
 * @property {number} height  px
 * @property {Float32Array} c  row-major, 0–1 of the rectangle's darkest pixel against the page
 */

/**
 * A glyph's box, [x0, x1) × [y0, y1), px in the image.
 *
 * @typedef {object} Glyph
 * @property {number} x0
 * @property {number} y0
 * @property {number} x1
 * @property {number} y1
 */

/**
 * A label as found in the pixels (labelGlyphs).
 *
 * @typedef {object} FoundLabel
 * @property {Glyph[]} glyphs  left to right
 * @property {Glyph[]} raised  the glyphs of its exponent, left to right: none without one
 * @property {number} height   px, the digits'
 * @property {number} top      px, the first row of the digits' line
 * @property {number} bottom   px, the row after its last
 * @property {Cover} cover     the coverage the glyphs were cut from
 */

/**
 * Where to look for a label (labelGlyphs' options).
 *
 * @typedef {object} LabelSearch
 * @property {number} [anchor]    px, the y the label is centered on (a y axis's tick), to pick its line when the box
 *   holds more than one
 * @property {number} [near]      px, the x the label is at (an x axis's tick) or nearest to (a y axis's line); the
 *   box's middle when not given
 * @property {number[]} [limits]  [x0, x1), px: columns not to look beyond (the axis and its tick marks)
 * @property {number[]} [line]    [top, bottom), px: the digits' rows, when known from the axis's other labels
 */

/**
 * A label as read.
 *
 * @typedef {object} LabelReading
 * @property {string} text         the glyphs as read, "?" for one that is no digit, point, comma, minus or sign of
 *   scientific notation ("1e+05"); an exponent after "^" ("10^-3", "2×10^3")
 * @property {number|null} value   the number (a power for a number with an exponent, times the number before a
 *   "×"), or null when the text isn't one
 * @property {number} score        the least sure digit's or sign's correlation with its template (1 without any)
 * @property {number[]} box        [x0, y0, x1, y1), px: the glyphs read, an exponent's included
 */

/**
 * The values of an axis's ticks as its scale gives them (consistentValues).
 *
 * @typedef {object} TickValues
 * @property {'linear'|'log'} type
 * @property {number[]} values   per tick: the value read where that agrees with the scale, elsewhere the scale's,
 *   rounded
 * @property {boolean[]} agree   per tick, whether its read value fits the scale
 * @property {number} count      how many do
 * @property {number} minor      how many unread ticks confirm the scale, where two labels gave it (a log scale: see
 *   logByMinorTicks); 0 where three or more labels agree, which need no other check
 */

/**
 * An axis's labels as read (readAxisLabels).
 *
 * @typedef {object} AxisReading
 * @property {(LabelReading|null)[]} read  per tick, what its label reads as, unit signs left out; null without a
 *   label
 * @property {TickValues|null} scale       consistentValues of those: null when no scale confirms them
 */

/**
 * How dark each pixel of a rectangle is against the page, 0–1 of the rectangle's darkest: the label's ink whatever
 * its color. Null when nothing in it is INK_MIN_CONTRAST from the page.
 *
 * @param {RgbaImage} image
 * @param {number[][]} backgrounds  0–255, the page's colors (axes.js backgroundColors)
 * @param {number[]} rect  [x0, y0, x1, y1), px, inside the image
 * @returns {Cover|null}
 */
function coverage(image, backgrounds, [x0, y0, x1, y1]) {
    const width = x1 - x0;
    const height = y1 - y0;
    const c = new Float32Array(width * height);
    let darkest = 0;
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const p = 4 * ((y0 + y) * image.width + x0 + x);
            let d = Infinity;
            for (const B of backgrounds)
                d = Math.min(d, Math.hypot(image.data[p] - B[0], image.data[p + 1] - B[1], image.data[p + 2] - B[2]));
            c[y * width + x] = d;
            darkest = Math.max(darkest, d);
        }
    }
    if (darkest < INK_MIN_CONTRAST)
        return null;
    for (let i = 0; i < c.length; i++)
        c[i] = Math.min(1, c[i] / darkest);
    return { x0, y0, width, height, c };
}

// runs [a, b) of true in a boolean per index
function runs(has) {
    const out = [];
    let start = -1;
    for (let i = 0; i <= has.length; i++) {
        if (i < has.length && has[i]) {
            if (start < 0)
                start = i;
        }
        else if (start >= 0) {
            out.push([start, i]);
            start = -1;
        }
    }
    return out;
}

/**
 * The glyphs of the label a box points at: the line of text through the box (or the line given), cut at empty
 * columns into glyphs and at wide gaps and foreign glyphs into words (a multiplication sign keeps the numbers beside
 * it in its word), and of the words the one nearest `near`, with the raised glyphs after it, its exponent. Touching
 * digits are cut apart, and a minus sign from the glyph it touches.
 *
 * @param {RgbaImage} image
 * @param {number[][]} backgrounds
 * @param {number[]} box  [x0, y0, x1, y1), px: the label's box from axes.js
 * @param {LabelSearch} [opts]
 * @returns {FoundLabel|null}  null without ink under the box
 */
export function labelGlyphs(image, backgrounds, box, opts = {}) {
    const rows = opts.line ?? [Math.floor(box[1]), Math.ceil(box[3])];
    const over = Math.max(LINE_OVER_MIN, Math.round(LINE_OVER * (rows[1] - rows[0])));
    const grow = Math.ceil(GROW_HEIGHTS * (rows[1] - rows[0]));
    const reach = Math.max(over + 1, Math.ceil(LINE_REACH * (rows[1] - rows[0])));
    // (above the line, a row beyond the highest exponent: whether a glyph ends there is what tells one)
    const up = Math.max(reach, Math.ceil(RAISED_MAX_RISE * (rows[1] - rows[0])) + 1);
    const rect = [
        Math.max(opts.limits?.[0] ?? 0, 0, Math.floor(box[0]) - grow), Math.max(0, rows[0] - up),
        Math.min(opts.limits?.[1] ?? Infinity, image.width, Math.ceil(box[2]) + grow), Math.min(image.height, rows[1] + reach),
    ];
    if (rect[2] <= rect[0] || rect[3] <= rect[1])
        return null;
    const cover = coverage(image, backgrounds, rect);
    if (!cover)
        return null;
    const { width, height, c } = cover;
    const sx0 = Math.max(0, Math.floor(box[0]) - rect[0]);
    const sx1 = Math.min(width, Math.ceil(box[2]) - rect[0]);
    const inked = (y, a, b) => {
        for (let x = a; x < b; x++) {
            if (c[y * width + x] >= GLYPH_INK)
                return true;
        }
        return false;
    };
    let line = opts.line && [opts.line[0] - rect[1], opts.line[1] - rect[1]];
    if (!line) {
        // the rows with ink under the box's columns, nearest the anchor (or the tallest)
        const lines = runs(Array.from({ length: height }, (_, y) => inked(y, sx0, sx1)));
        if (!lines.length)
            return null;
        const anchor = opts.anchor === undefined ? null : opts.anchor - rect[1];
        line = lines.reduce((best, l) => {
            if (anchor === null)
                return l[1] - l[0] > best[1] - best[0] ? l : best;
            return Math.abs((l[0] + l[1]) / 2 - anchor) < Math.abs((best[0] + best[1]) / 2 - anchor) ? l : best;
        });
    }
    // a glyph's box over columns [a, b): the rows of the line with ink there, and on from the line's edges while
    // there is ink (how far a glyph runs beyond the line is what tells a foreign one)
    const glyphOver = (a, b) => {
        let y0 = line[1];
        let y1 = line[0];
        for (let y = line[0]; y < line[1]; y++) {
            if (inked(y, a, b)) {
                y0 = Math.min(y0, y);
                y1 = Math.max(y1, y + 1);
            }
        }
        if (y0 === line[0]) {
            while (y0 > 0 && inked(y0 - 1, a, b))
                y0--;
        }
        if (y1 === line[1]) {
            while (y1 < height && inked(y1, a, b))
                y1++;
        }
        return { x0: a, x1: b, y0, y1 };
    };
    const inkedColumn = (x, y0, y1) => {
        for (let y = y0; y < y1; y++) {
            if (c[y * width + x] >= GLYPH_INK)
                return true;
        }
        return false;
    };
    const cut = runs(Array.from({ length: width }, (_, x) => inkedColumn(x, line[0], line[1]))).map(([a, b]) => glyphOver(a, b));
    const seed = cut.filter(g => g.x1 > sx0 && g.x0 < sx1);
    if (!seed.length)
        return null;
    // the tall glyphs among some, and their rows
    const tallOf = glyphs => {
        const tallest = Math.max(...glyphs.map(g => g.y1 - g.y0));
        const tall = glyphs.filter(g => g.y1 - g.y0 >= DIGIT_MIN_HEIGHT * tallest);
        return { tall, top: Math.min(...tall.map(g => g.y0)), bottom: Math.max(...tall.map(g => g.y1)) };
    };
    // An exponent touching its base (plotmath's "10" with a raised 5 at 9 px: no empty column between the 0 and the
    // 5, so the two are one glyph, taller than the digits). Given the digits' line, the glyph rises above it, and
    // its last columns have no ink below an exponent's rows: they are cut off as a glyph of their own when what is
    // left then keeps to the line (it doesn't for a tall letter). From there on the exponent is found as one
    // standing apart is, and anything else raised is what it would be standing apart.
    const cutRaised = (g, top, bottom) => {
        if (g.y0 >= top - over)
            return [g];
        const bandEnd = top + Math.floor(RAISED_MAX_BOTTOM * (bottom - top));
        let at = g.x1;
        while (at > g.x0 && !inkedColumn(at - 1, bandEnd, bottom))
            at--;
        if (at === g.x0 || at === g.x1)
            return [g];
        const base = glyphOver(g.x0, at);
        if (base.y0 < top - over)
            return [g];
        // (the base's rows are the line's: a raised 4's tip may hang over its last columns)
        return [{ ...base, y0: Math.max(base.y0, top) }, glyphOver(at, g.x1)];
    };
    // the digits' line: the one given, or that of the tall glyphs under the box
    let top = line[0];
    let bottom = line[1];
    if (!opts.line) {
        const own = tallOf(seed);
        [top, bottom] = [own.top, own.bottom];
        // … unless the one that starts it is a base with its exponent: the line is then the other glyphs', on its
        // baseline
        const highest = own.tall.filter(g => g.y0 === top);
        const others = seed.filter(g => g !== highest[0]);
        if (highest.length === 1 && others.length) {
            const lower = tallOf(others);
            if (Math.abs(lower.bottom - highest[0].y1) <= over && cutRaised(highest[0], lower.top, lower.bottom).length === 2)
                [top, bottom] = [lower.top, lower.bottom];
        }
    }
    const all = cut.flatMap(g => cutRaised(g, top, bottom));
    const H = bottom - top;
    // a glyph running beyond the line is not of a number (a letter with a descender, a rotated title, a currency
    // sign), a comma's tail aside: it ends a word, as a wide gap does
    const foreign = g => g.y0 < top - over || (g.y1 > bottom + over && g.y1 - g.y0 > COMMA_MAX_HEIGHT * H);
    const inImage = g => ({ x0: g.x0 + rect[0], x1: g.x1 + rect[0], y0: g.y0 + rect[1], y1: g.y1 + rect[1] });
    const times = g => glyphSign(cover, inImage(g), H)?.sign === '×';
    // (a multiplication sign holds the number before it and the power after it together, spaced as they are)
    const apart = (a, b) => b.x0 - a.x1 > WORD_GAP * H && (b.x0 - a.x1 > TIMES_MAX_GAP * H || !(times(a) || times(b)));
    const words = [];
    let word = [];
    for (const g of all) {
        if (word.length && (foreign(g) || apart(word[word.length - 1], g))) {
            words.push(word);
            word = [];
        }
        if (!foreign(g))
            word.push(g);
    }
    if (word.length)
        words.push(word);
    if (!words.length)
        return null;
    const near = opts.near === undefined ? (box[0] + box[2]) / 2 - rect[0] : opts.near - rect[0];
    const away = w => Math.max(0, w[0].x0 - near, near - w[w.length - 1].x1);
    const nearest = words.reduce((best, w) => (away(w) < away(best) ? w : best));
    // touching digits: cut at the lightest column near each even division
    const splitWide = g => {
        const w = g.x1 - g.x0;
        if (w <= SPLIT_MIN_WIDTH * H || g.y1 - g.y0 < SPLIT_MIN_HEIGHT * H)
            return [g];
        const parts = Math.round(w / (DIGIT_PITCH * H));
        const ink = x => {
            let s = 0;
            for (let y = top; y < bottom; y++)
                s += c[y * width + x];
            return s;
        };
        const cuts = [g.x0];
        for (let k = 1; k < parts; k++) {
            const at = g.x0 + k * w / parts;
            let best = Math.round(at);
            for (let x = Math.ceil(at - SPLIT_SEARCH * H); x <= Math.floor(at + SPLIT_SEARCH * H); x++) {
                if (x > cuts[cuts.length - 1] && x < g.x1 && ink(x) < ink(best))
                    best = x;
            }
            cuts.push(best);
        }
        cuts.push(g.x1);
        // (the cut's column is the gray between the two: it goes to neither)
        return cuts.slice(1).map((b, k) => glyphOver(cuts[k] + (k > 0 ? 1 : 0), b));
    };
    const digitScore = g => classifyDigit(glyphGrid(cover, inImage(g))).score;
    // A minus sign touching its neighbor, as small type has it ("1e-03" at R's default size: the minus runs into the
    // "0", or the "e" into the minus): the glyph's first columns, flat and at mid-height, when what is left is more
    // of a digit than the whole, or its last, when what is left is an "e". A "4" starts flat too, with the tip of
    // its bar (as wide as a minus in Gill Sans), and is less of a 4 without it; a plus's arms leave neither.
    const cutMinus = g => {
        const rowsAt = x => {
            let y0 = g.y1;
            let y1 = g.y0;
            for (let y = g.y0; y < g.y1; y++) {
                if (c[y * width + x] >= GLYPH_INK) {
                    y0 = Math.min(y0, y);
                    y1 = Math.max(y1, y + 1);
                }
            }
            return [y0, y1];
        };
        // how many columns from `from` on, by `step`, are flat together and at mid-height: 0 unless a minus's width
        const flat = (from, step) => {
            let rows = [g.y1, g.y0];
            let n = 0;
            for (let x = from; x >= g.x0 && x < g.x1; x += step) {
                const [a, b] = rowsAt(x);
                if (Math.max(rows[1], b) - Math.min(rows[0], a) > MINUS_MAX_HEIGHT * H)
                    break;
                rows = [Math.min(rows[0], a), Math.max(rows[1], b)];
                n++;
            }
            const middle = ((rows[0] + rows[1]) / 2 - top) / H;
            return n < g.x1 - g.x0 && n >= MINUS_MIN_WIDTH * H && middle >= MINUS_CENTER[0] && middle <= MINUS_CENTER[1] ? n : 0;
        };
        const start = flat(g.x0, 1);
        if (start) {
            const rest = glyphOver(g.x0 + start, g.x1);
            if (rest.y1 - rest.y0 >= DIGIT_MIN_HEIGHT * H && digitScore(rest) > digitScore(g))
                return [glyphOver(g.x0, g.x0 + start), rest];
        }
        const end = flat(g.x1 - 1, -1);
        if (end) {
            const rest = glyphOver(g.x0, g.x1 - end);
            if (/e/i.test(glyphSign(cover, inImage(rest), H)?.sign ?? ''))
                return [rest, glyphOver(g.x1 - end, g.x1)];
        }
        return [g];
    };
    // the rows an exponent's glyphs lie in, [bandStart, bandEnd)
    const bandStart = top - Math.floor(RAISED_MAX_RISE * H);
    const bandEnd = top + Math.floor(RAISED_MAX_BOTTOM * H);
    // The raised glyphs from a column on: runs of columns with ink in the band and none in the line below it, each
    // with its rows, up to a wide gap, a glyph of the line (the next word) or one that comes down from above the
    // band (a title's descender). Cut from the band's own rows, not the line's: a raised "1" has its flag and a
    // minus sign may have all of itself above the line's first row.
    const raisedFrom = from => {
        const above = Math.max(0, bandStart - 1);
        const high = x => inkedColumn(x, above, bandEnd) && !inkedColumn(x, bandEnd, bottom);
        // the band's rows with ink over columns [a, b)
        const rowsOf = (a, b) => {
            let y0 = bandEnd;
            let y1 = above;
            for (let y = above; y < bandEnd; y++) {
                if (inked(y, a, b)) {
                    y0 = Math.min(y0, y);
                    y1 = Math.max(y1, y + 1);
                }
            }
            return [y0, y1];
        };
        const out = [];
        let last = from;
        for (let x = from; x < width && x - last <= WORD_GAP * H; x++) {
            if (inkedColumn(x, bandEnd, bottom))
                break;
            if (!high(x))
                continue;
            let x1 = x + 1;
            while (x1 < width && high(x1))
                x1++;
            if (x1 < width && inkedColumn(x1, bandEnd, bottom))
                break;
            const [y0, y1] = rowsOf(x, x1);
            if (y0 < bandStart)
                break;
            // a minus sign touching the digit after it (primates2's raised "-2") is the run's first columns, flat
            // and at mid-height: a "1" starts flat with its flag, higher up, a round digit is flat for a column,
            // and a "4" with the tip of its bar is less of a digit without it
            let cut = x;
            let flat = [y1, y0];
            while (cut < x1) {
                const [a, b] = rowsOf(cut, cut + 1);
                if (Math.max(flat[1], b) - Math.min(flat[0], a) > MINUS_MAX_HEIGHT * (y1 - y0))
                    break;
                flat = [Math.min(flat[0], a), Math.max(flat[1], b)];
                cut++;
            }
            const middle = ((flat[0] + flat[1]) / 2 - y0) / (y1 - y0);
            const [r0, r1] = cut < x1 ? rowsOf(cut, x1) : [y0, y1];
            const minus = cut < x1 && cut - x >= MINUS_MIN_WIDTH * (y1 - y0) && middle >= MINUS_CENTER[0] && middle <= MINUS_CENTER[1];
            if (minus && digitScore({ x0: cut, x1, y0: r0, y1: r1 }) > digitScore({ x0: x, x1, y0, y1 })) {
                out.push({ x0: x, x1: cut, y0: flat[0], y1: flat[1] }, { x0: cut, x1, y0: r0, y1: r1 });
            }
            else {
                out.push({ x0: x, x1, y0, y1 });
            }
            last = x1;
            x = x1 - 1;
        }
        return out;
    };
    // an exponent follows the number's last glyph on the line. A raised glyph that the line's rows show ends the
    // word as a glyph of it (a minus at the digits' top): it is found again, whole, among the raised ones
    let number = nearest;
    while (number.length > 1 && number[number.length - 1].y0 >= bandStart && number[number.length - 1].y1 <= bandEnd)
        number = number.slice(0, -1);
    const after = raisedFrom(number[number.length - 1].x1);
    const exponent = after.some(g => g.y1 - g.y0 >= RAISED_MIN_HEIGHT * H);
    return {
        glyphs: (exponent ? number : nearest).flatMap(splitWide).flatMap(cutMinus).map(inImage), raised: exponent ? after.map(inImage) : [],
        height: H, top: top + rect[1], bottom: bottom + rect[1], cover,
    };
}

/**
 * What each glyph of a label is by its size and place in the line, before any digit is told from another.
 *
 * @param {{glyphs: Glyph[], height: number, top: number}} found  labelGlyphs'
 * @returns {string[]}  per glyph: 'digit' (a tall glyph), '.', ',', '-', or '?' for anything else
 */
export function glyphKinds({ glyphs, height: H, top }) {
    // the baseline is the label's own digits' feet: the line given for the whole axis can be a pixel off it
    const feet = glyphs.filter(g => g.y1 - g.y0 >= DIGIT_MIN_HEIGHT * H).map(g => g.y1);
    const base = feet.length ? Math.max(...feet) : top + H;
    return glyphs.map(g => {
        const h = g.y1 - g.y0;
        const w = g.x1 - g.x0;
        const from = (g.y0 - top) / H;
        const to = (g.y1 - top) / H;
        const under = g.y1 - base >= Math.max(1, COMMA_MIN_UNDER * H);
        if (h >= DIGIT_MIN_HEIGHT * H)
            return 'digit';
        if (h <= MINUS_MAX_HEIGHT * H && w >= MINUS_MIN_WIDTH * H && (from + to) / 2 >= MINUS_CENTER[0] && (from + to) / 2 <= MINUS_CENTER[1])
            return '-';
        if (h <= POINT_MAX_SIZE * H && w <= POINT_MAX_SIZE * H && to >= POINT_MIN_BOTTOM && !under)
            return '.';
        if (w <= COMMA_MAX_WIDTH * H && h <= COMMA_MAX_HEIGHT * H && from >= COMMA_MIN_TOP && under)
            return ',';
        return '?';
    });
}

/**
 * A glyph on the templates' grid: its own height scaled to TEMPLATE_HEIGHT, its width by the same factor and
 * centered (a "1" stays narrow), sampled bilinearly from the coverage. A glyph wider than the grid loses its sides
 * (a plus, as wide as it is tall, the ends of its arms).
 *
 * @param {Cover} cover
 * @param {Glyph} g
 * @returns {Float32Array}  TEMPLATE_WIDTH × TEMPLATE_HEIGHT, 0–1
 */
export function glyphGrid(cover, g) {
    const out = new Float32Array(TEMPLATE_WIDTH * TEMPLATE_HEIGHT);
    const scale = (g.y1 - g.y0) / TEMPLATE_HEIGHT;
    const cx = (g.x0 + g.x1) / 2 - cover.x0;
    const at = (x, y) => (x < 0 || y < 0 || x >= cover.width || y >= cover.height ? 0 : cover.c[y * cover.width + x]);
    for (let j = 0; j < TEMPLATE_HEIGHT; j++) {
        // pixel centers: grid cell (i, j) samples the glyph at its own center
        const sy = g.y0 - cover.y0 + (j + 0.5) * scale - 0.5;
        const y = Math.floor(sy);
        const fy = sy - y;
        for (let i = 0; i < TEMPLATE_WIDTH; i++) {
            const sx = cx + (i + 0.5 - TEMPLATE_WIDTH / 2) * scale - 0.5;
            // (beyond the glyph's own columns is the next glyph: nothing of it)
            if (sx < g.x0 - cover.x0 - 1 || sx > g.x1 - cover.x0)
                continue;
            const x = Math.floor(sx);
            const fx = sx - x;
            out[j * TEMPLATE_WIDTH + i] = (at(x, y) * (1 - fx) + at(x + 1, y) * fx) * (1 - fy) + (at(x, y + 1) * (1 - fx) + at(x + 1, y + 1) * fx) * fy;
        }
    }
    return out;
}

// correlation of two grids, each about its own mean: −1 to 1
function correlation(a, b) {
    let ma = 0;
    let mb = 0;
    for (let i = 0; i < a.length; i++) {
        ma += a[i];
        mb += b[i];
    }
    ma /= a.length;
    mb /= b.length;
    let ab = 0;
    let aa = 0;
    let bb = 0;
    for (let i = 0; i < a.length; i++) {
        ab += (a[i] - ma) * (b[i] - mb);
        aa += (a[i] - ma) ** 2;
        bb += (b[i] - mb) ** 2;
    }
    return aa > 0 && bb > 0 ? ab / Math.sqrt(aa * bb) : 0;
}

/**
 * The digit a glyph's grid is most like, over every template of every font.
 *
 * @param {Float32Array} grid  glyphGrid's
 * @param {{name: string, digits: Object<string, ArrayLike<number>>}[]} [templates]  each font's digits on the grid
 * @returns {{digit: string, score: number, margin: number}}  score: the best correlation; margin: how far it is above
 *   the best of any other digit
 */
export function classifyDigit(grid, templates = DIGIT_TEMPLATES) {
    const best = new Map();
    for (const font of templates) {
        for (const [digit, t] of Object.entries(font.digits))
            best.set(digit, Math.max(best.get(digit) ?? -1, correlation(grid, t)));
    }
    const ranked = [...best].sort((a, b) => b[1] - a[1]);
    return { digit: ranked[0][0], score: ranked[0][1], margin: ranked[0][1] - (ranked[1]?.[1] ?? -1) };
}

/**
 * The sign of scientific notation a glyph's grid is most like, among those given.
 *
 * @param {Float32Array} grid  glyphGrid's
 * @param {string} signs  the templates' characters to try: of "eE+×x"
 * @param {{name: string, signs: Object<string, ArrayLike<number>>}[]} [templates]  each font's signs on the grid
 * @returns {{sign: string, score: number}}  sign: "e", "E", "+" or "×" (an "x" is one); score: the best correlation
 */
export function classifySign(grid, signs, templates = DIGIT_TEMPLATES) {
    let best = { sign: '', score: -1 };
    for (const font of templates) {
        for (const ch of signs) {
            const score = correlation(grid, font.signs[ch]);
            if (score > best.score)
                best = { sign: SIGN_READ[ch] ?? ch, score };
        }
    }
    return best;
}

// The sign a glyph is, of those its height against the digits' H allows, or null when it is like none: "E" is as
// tall as a digit, the others lower.
function glyphSign(cover, g, H) {
    const h = g.y1 - g.y0;
    const signs = (h >= DIGIT_MIN_HEIGHT * H ? 'E' : '') + (h <= SIGN_MAX_HEIGHT * H && h >= SIGN_MIN_HEIGHT * H ? 'e+×x' : '');
    if (!signs)
        return null;
    const s = classifySign(glyphGrid(cover, g), signs);
    return s.score >= SIGN_MIN_SCORE ? s : null;
}

/**
 * The glyphs of one line as text: each by its kind, a digit or a sign of scientific notation by its template.
 *
 * @param {Cover} cover
 * @param {{glyphs: Glyph[], height: number, top: number}} line  the glyphs, with their digits' height and top, px
 * @returns {{text: string, score: number}}  score: the least sure digit's or sign's correlation with its template
 */
function lineText(cover, line) {
    let text = '';
    let score = 1;
    glyphKinds(line).forEach((kind, i) => {
        if (kind !== 'digit' && kind !== '?') {
            text += kind;
            return;
        }
        const g = line.glyphs[i];
        const d = kind === 'digit' ? classifyDigit(glyphGrid(cover, g)) : null;
        const s = glyphSign(cover, g, line.height);
        if (s && (!d || s.score > d.score)) {
            text += s.sign;
            score = Math.min(score, s.score);
            return;
        }
        if (!d) {
            text += '?';
            return;
        }
        text += d.score >= DIGIT_MIN_SCORE ? d.digit : '?';
        score = Math.min(score, d.score);
    });
    return { text, score };
}

/**
 * A label's glyphs as text and as a number.
 *
 * @param {FoundLabel} found
 * @param {number[]} [signs]  glyphs to leave out at the start and at the end (unitSigns)
 * @returns {LabelReading}
 */
function glyphsRead(found, signs = [0, 0]) {
    const glyphs = found.glyphs.slice(signs[0], found.glyphs.length - signs[1]);
    const all = [...glyphs, ...found.raised];
    const box = [Math.min(...all.map(g => g.x0)), Math.min(...all.map(g => g.y0)), Math.max(...all.map(g => g.x1)), Math.max(...all.map(g => g.y1))];
    const number = lineText(found.cover, { glyphs, height: found.height, top: found.top });
    if (found.raised.length) {
        // the exponent is a line of its own, of smaller digits: an integer, the power of an integer
        const tallest = Math.max(...found.raised.map(g => g.y1 - g.y0));
        const tall = found.raised.filter(g => g.y1 - g.y0 >= DIGIT_MIN_HEIGHT * tallest);
        const top = Math.min(...tall.map(g => g.y0));
        const power = lineText(found.cover, { glyphs: found.raised, height: Math.max(...tall.map(g => g.y1)) - top, top });
        const base = /^(-?)(\d+)$/.exec(number.text);
        // … or of ten, times a number before it: "2×10" with a raised 3
        const times = /^(-?)(\d+\.?\d*|\.\d+)×10$/.exec(number.text);
        let value = null;
        if ((base || times) && /^-?\d+$/.test(power.text)) {
            // (ten's powers from their decimal text: 10 ** -5 is a bit off 0.00001)
            const v = times ? Number(`${times[2]}e${power.text}`) : base[2] === '10' ? Number(`1e${power.text}`) : Number(base[2]) ** Number(power.text);
            if (Number.isFinite(v))
                value = (times ?? base)[1] ? -v : v;
        }
        return { text: `${number.text}^${power.text}`, value, score: Math.min(number.score, power.score), box };
    }
    // (a dash after the number is no minus sign: a tick mark, or the dash of a range)
    const trimmed = number.text.replace(/-+$/, '');
    // thousands separators go; what is left must be a plain number
    const plain = /^-?\d{1,3}(,\d{3})+(\.\d+)?$/.test(trimmed) ? trimmed.replace(/,/g, '') : trimmed;
    // (… or one with its power of ten after an "e", which Number reads as written: "1e+05", "2.5E-3")
    return { text: trimmed, value: /^-?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$/.test(plain) ? Number(plain) : null, score: number.score, box };
}

/**
 * Read a tick label as a number.
 *
 * @param {RgbaImage} image
 * @param {number[][]} backgrounds  0–255, the page's colors
 * @param {number[]} box  [x0, y0, x1, y1), px: the label's box from axes.js
 * @param {LabelSearch} [opts]
 * @returns {LabelReading|null}  null without ink
 */
export function readLabel(image, backgrounds, box, opts = {}) {
    const found = labelGlyphs(image, backgrounds, box, opts);
    return found && found.height > 0 ? glyphsRead(found) : null;
}

/**
 * The unit signs of an axis's labels: glyphs that are no digit and are alike in every label, before the number (a
 * currency sign) or after it (a percent sign, a letter). They are left out of the reading: "$320" is 320 and "75%"
 * is 75, in the unit the axis states.
 *
 * @param {FoundLabel[]} founds  each label of the axis (at least SCALE_MIN_AGREE: fewer could be anything alike)
 * @returns {number[]}  how many glyphs at the start and at the end of each label are signs
 */
function unitSigns(founds) {
    const signs = [0, 0];
    if (founds.length < SCALE_MIN_AGREE)
        return signs;
    for (const end of [0, 1]) {
        for (;;) {
            // (a number is left in every label)
            if (founds.some(f => f.glyphs.length <= signs[0] + signs[1] + 1))
                break;
            const at = f => f.glyphs[end ? f.glyphs.length - 1 - signs[1] : signs[0]];
            const grids = founds.map(f => glyphGrid(f.cover, at(f)));
            const sign = founds.every((f, i) => {
                const g = at(f);
                const tall = g.y1 - g.y0 >= DIGIT_MIN_HEIGHT * f.height;
                return tall && classifyDigit(grids[i]).score < DIGIT_MIN_SCORE && correlation(grids[i], grids[0]) >= SIGN_LIKENESS;
            });
            if (!sign)
                break;
            signs[end]++;
        }
    }
    return signs;
}

/**
 * The roundest number within a tolerance of a value: the one with the fewest significant digits. A tick's value
 * from the scale is 89.9083 when its position is half a pixel off; its label says 90.
 *
 * @param {number} v
 * @param {(r: number) => boolean} near  whether r is within the tolerance of v
 * @returns {number}
 */
function roundest(v, near) {
    if (near(0))
        return 0;
    for (let digits = 1; digits < 15; digits++) {
        const r = Number(v.toPrecision(digits));
        if (near(r))
            return r;
    }
    return v;
}

// per tick, whether the value read for it is within tol px of where a scale puts it: value = a + b · pos, or its
// log10 on a log scale
function agreeing(ticks, type, a, b, tol) {
    const f = type === 'log' ? Math.log10 : v => v;
    return ticks.map(t => t.value !== null && (type === 'linear' || t.value > 0) && Math.abs(f(t.value) - (a + b * t.pos)) <= tol * Math.abs(b));
}

/**
 * A log scale that two read values give and the axis's other ticks confirm. Minor ticks of a log axis stand at 2 to
 * 9 times each power of ten, bunched toward the next power, and only the right scale puts them there: one with the
 * wrong number of decades between the two labels, or run backwards, or with the labels' leading digit misread,
 * leaves most of them between multiples. What it can't tell is both labels off by one same power of ten.
 *
 * Believed when the unread ticks that fall on a multiple show at least MINOR_MIN_MULTIPLES of 2 to 9, those that
 * fall on none are at most SCALE_MAX_DISAGREE of those that do (a false tick), and so are the read values off the
 * scale. O(ticks³).
 *
 * @param {{pos: number, value: number|null}[]} ticks
 * @param {number} tol  px, how near a tick a value must be put
 * @returns {{fitted: {pos: number, value: number}[], minor: number}|null}  the ticks to fit the scale to, those
 *   read and those on a multiple with it as their value, and how many the latter are
 */
function logByMinorTicks(ticks, tol) {
    const read = ticks.filter(t => t.value !== null);
    const usable = read.filter(t => t.value > 0);
    let best = null;
    for (let p = 0; p < usable.length; p++) {
        for (let q = p + 1; q < usable.length; q++) {
            const [s, t] = [usable[p], usable[q]];
            if (s.pos === t.pos || s.value === t.value)
                continue;
            const b = (Math.log10(t.value) - Math.log10(s.value)) / (t.pos - s.pos);
            const a = Math.log10(s.value) - b * s.pos;
            const near = tol * Math.abs(b); // decades
            const agree = agreeing(ticks, 'log', a, b, tol);
            const count = agree.filter(Boolean).length;
            if (near > MINOR_MAX_TOL || read.length - count > SCALE_MAX_DISAGREE * count)
                continue;
            const on = [];
            const multiples = new Set();
            let off = 0;
            for (const u of ticks.filter(v => v.value === null)) {
                const at = a + b * u.pos;
                const power = Math.floor(at + near);
                const k = Math.round(10 ** (at - power));
                if (Math.abs(at - power - Math.log10(k)) > near) {
                    off++;
                    continue;
                }
                on.push({ pos: u.pos, value: k * 10 ** power });
                if (k > 1)
                    multiples.add(k);
            }
            if (multiples.size >= MINOR_MIN_MULTIPLES && off <= SCALE_MAX_DISAGREE * on.length && (!best || on.length > best.minor))
                best = { fitted: [...ticks.filter((u, i) => agree[i]), ...on], minor: on.length };
        }
    }
    return best;
}

/**
 * The values of an axis's ticks from those read, checked by the scale: among linear and log scales through two read
 * values, the one that the most read values agree with, fitted again to all that do. Two read values alone are
 * believed on a log scale that the axis's minor ticks confirm (logByMinorTicks). Positions are px along the axis; a
 * tick's value on a linear scale is a + b · pos, on a log scale 10^(a + b · pos). O(ticks³).
 *
 * @param {{pos: number, value: number|null}[]} ticks  each tick's position and the value read for it, or null
 * @returns {TickValues|null}  null when fewer than SCALE_MIN_AGREE read values fit one scale, or too many don't
 *   (SCALE_MAX_DISAGREE), and no log scale through two is confirmed by the other ticks
 */
export function consistentValues(ticks) {
    if (ticks.length < SCALE_MIN_AGREE)
        return null;
    const tol = Math.max(SCALE_POS_TOL, SCALE_TOL * Math.min(...ticks.slice(1).map((u, k) => Math.abs(u.pos - ticks[k].pos)))); // px
    let best = null;
    for (const type of ['linear', 'log']) {
        const f = type === 'log' ? Math.log10 : v => v;
        const usable = ticks.filter(t => t.value !== null && (type === 'linear' || t.value > 0));
        for (let p = 0; p < usable.length; p++) {
            for (let q = p + 1; q < usable.length; q++) {
                const [s, t] = [usable[p], usable[q]];
                if (s.pos === t.pos || s.value === t.value)
                    continue;
                const b = (f(t.value) - f(s.value)) / (t.pos - s.pos);
                const agree = agreeing(ticks, type, f(s.value) - b * s.pos, b, tol);
                const count = agree.filter(Boolean).length;
                // (a tie goes to linear, tried first)
                if (!best || count > best.count)
                    best = { type, count, fitted: ticks.filter((u, i) => agree[i]), minor: 0 };
            }
        }
    }
    const read = ticks.filter(t => t.value !== null).length;
    if (!best || best.count < SCALE_MIN_AGREE || read - best.count > SCALE_MAX_DISAGREE * best.count) {
        const confirmed = logByMinorTicks(ticks, tol);
        if (!confirmed)
            return null;
        best = { type: 'log', ...confirmed };
    }
    const { type, fitted, minor } = best;
    const f = type === 'log' ? Math.log10 : v => v;
    // least squares through those that agree: two ticks' half pixels are a poor scale far from them
    const mp = fitted.reduce((sum, t) => sum + t.pos, 0) / fitted.length;
    const mv = fitted.reduce((sum, t) => sum + f(t.value), 0) / fitted.length;
    const b = fitted.reduce((sum, t) => sum + (t.pos - mp) * (f(t.value) - mv), 0) / fitted.reduce((sum, t) => sum + (t.pos - mp) ** 2, 0);
    const a = mv - b * mp;
    const agree = agreeing(ticks, type, a, b, tol);
    const g = type === 'log' ? v => 10 ** v : v => v;
    return {
        type, agree, count: agree.filter(Boolean).length, minor,
        values: ticks.map((t, i) => {
            if (agree[i])
                return t.value;
            const v = a + b * t.pos;
            return roundest(g(v), r => (type === 'linear' || r > 0) && Math.abs(f(r) - v) <= tol * Math.abs(b));
        }),
    };
}

/**
 * Where to look for each of an axis's labels (labelGlyphs' options). The labels are found twice: each from its own
 * box, and then on the line the axis's labels share (one row of text below an x axis, one height and one place
 * against its tick beside a y axis), which tells a neighbor's glyphs from the label's where the first finding takes
 * them in.
 *
 * A label found cleanly at the shared height keeps its own line: it has no neighbor's glyphs to tell, and the shared
 * line may not be where it is. A label set by hand stands a few pixels off its row (crimson's "10²", 5 px lower than
 * the other three: on their line its exponent ended below an exponent's rows, and it read "10?"), and a y axis's
 * line, placed by each tick's position, is a row off at some ticks (small-e-r72: a point one pixel square in the
 * digits' last row was left out of two labels of six, "000025").
 *
 * @param {RgbaImage} image
 * @param {number[][]} backgrounds  0–255, the page's colors
 * @param {import('./axes.js').Axis} axis
 * @param {'x'|'y'} which  the axis it is: an x axis runs across, its labels in one row
 * @returns {(LabelSearch|null)[]}  per tick, null without a label
 */
export function labelOptions(image, backgrounds, axis, which) {
    const vertical = which === 'y';
    // (a y axis's label is not to be followed across the axis or into its tick marks: no farther than the longest
    // mark's end, or than its box where that reaches nearer the axis, beside a shorter mark: axes.js uncutLabel)
    const reach = (axis.line?.thickness ?? 0) / 2 + Math.max(0, ...axis.ticks.map(t => t.length));
    const options = t => {
        if (!vertical)
            return { near: t.pos };
        if (!axis.line)
            return { anchor: t.pos };
        const left = t.label[2] <= axis.line.pos;
        const limits = left ? [0, Math.max(Math.floor(axis.line.pos - reach), Math.ceil(t.label[2]))] : [Math.min(Math.ceil(axis.line.pos + reach), Math.floor(t.label[0])), image.width];
        return { anchor: t.pos, limits, near: left ? limits[1] : limits[0] };
    };
    const clean = [];
    for (const t of axis.ticks) {
        const found = t.label && labelGlyphs(image, backgrounds, t.label, options(t));
        if (found && found.height > 0 && glyphKinds(found).every(kind => kind !== '?'))
            clean.push({ found, t });
    }
    let lineOf = null;
    if (clean.length >= SHARED_LINE_MIN) {
        const H = upperMedian(clean.map(({ found }) => found.height));
        if (vertical) {
            const offset = Math.round(median(clean.map(({ found, t }) => found.top - t.pos)));
            lineOf = t => [Math.floor(t.pos) + offset, Math.floor(t.pos) + offset + H];
        }
        else {
            const top = upperMedian(clean.map(({ found }) => found.top));
            lineOf = () => [top, top + H];
        }
    }
    const own = new Map(clean.map(({ found, t }) => [t, found]));
    return axis.ticks.map(t => {
        if (!t.label)
            return null;
        if (!lineOf || (own.has(t) && Math.abs(own.get(t).height - (lineOf(t)[1] - lineOf(t)[0])) <= OWN_LINE_TOL))
            return options(t);
        return { ...options(t), line: lineOf(t) };
    });
}

/**
 * Read an axis's tick labels.
 *
 * @param {RgbaImage} image
 * @param {number[][]} backgrounds  0–255, the page's colors
 * @param {import('./axes.js').Axis} axis
 * @param {'x'|'y'} which  the axis it is
 * @returns {AxisReading}
 */
export function readAxisLabels(image, backgrounds, axis, which) {
    const options = labelOptions(image, backgrounds, axis, which);
    const founds = axis.ticks.map((t, i) => {
        const found = t.label && labelGlyphs(image, backgrounds, t.label, options[i]);
        return found && found.height > 0 ? found : null;
    });
    const signs = unitSigns(founds.filter(Boolean));
    const read = founds.map(found => found && glyphsRead(found, signs));
    return { read, scale: consistentValues(axis.ticks.map((t, i) => ({ pos: t.pos, value: read[i]?.value ?? null }))) };
}

/**
 * Read the tick labels of both axes of an image.
 *
 * @param {RgbaImage} image
 * @param {import('./axes.js').Axes} axes  axes.js detectAxes'
 * @returns {{x: AxisReading|null, y: AxisReading|null}}  null for an axis that wasn't found
 */
export function readTickLabels(image, axes) {
    let y = axes.y ? readAxisLabels(image, axes.backgrounds, axes.y, 'y') : null;
    // a y axis's labels written along it: read turned, when upright they fit no scale
    if (y && !y.scale) {
        const turned = readTurned(image, axes.backgrounds, axes.y);
        if (turned.scale)
            y = turned;
    }
    return { x: axes.x ? readAxisLabels(image, axes.backgrounds, axes.x, 'x') : null, y };
}

/**
 * A y axis's labels written along the axis, bottom to top, as Stata and base R write them (funnel: "0", ".05",
 * ".1", ".15" on their sides): read on the image turned a quarter clockwise, where they run left to right in one
 * row over the axis. The readings' boxes are turned back; the scale is checked on the ticks as they are.
 *
 * @param {RgbaImage} image
 * @param {number[][]} backgrounds  0–255, the page's colors
 * @param {import('./axes.js').Axis} axis  the y axis
 * @returns {AxisReading}
 */
function readTurned(image, backgrounds, axis) {
    const { width, height, data } = image;
    // pixel (x, y) goes to (height − 1 − y, x)
    const turned = { width: height, height: width, data: new Uint8ClampedArray(data.length) };
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++)
            turned.data.set(data.subarray(4 * (y * width + x), 4 * (y * width + x) + 4), 4 * (x * height + height - 1 - y));
    }
    const turn = b => b && [height - b[3], b[0], height - b[1], b[2]];
    const back = b => b && [b[1], height - b[2], b[3], height - b[0]];
    const line = axis.line && { ...axis.line, dir: 'h', from: height - axis.line.to, to: height - axis.line.from };
    const ticks = axis.ticks.map(t => ({ ...t, pos: height - t.pos, label: turn(t.label) }));
    const read = readAxisLabels(turned, backgrounds, { ...axis, line, ticks }, 'x').read.map(r => r && { ...r, box: back(r.box) });
    return { read, scale: consistentValues(axis.ticks.map((t, i) => ({ pos: t.pos, value: read[i]?.value ?? null }))) };
}

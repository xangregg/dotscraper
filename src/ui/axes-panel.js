// The Axes section: detected ticks with their label crops, typed values, the scale they imply, and pixel → data.
//
// An axis entry is { type: 'linear' | 'log' | 'category', texts: [string per tick] } and is remembered per image
// (by pixel hash) in localStorage, so reopening a test image doesn't mean retyping its ticks.

import { fitScale, toData, resolution, guessValues } from '../core/axes.js';

const STORAGE_PREFIX = 'dotscraper:axes:';
const CROP_HEIGHT = 20; // CSS px, label crops in the tick table
const CROP_PAD = 2; // px of image around a label box
const AGREE_PX = 1; // typed ticks off their fitted position by more than this are flagged

/**
 * @typedef {object} AxisEntry  what the user typed for an axis
 * @property {'linear'|'log'|'category'} type
 * @property {string[]} texts  per tick: its value, or its category's name
 */

/**
 * The scale an entry implies (axisScale).
 *
 * @typedef {{kind: 'numeric', scale: {type: string, a: number, b: number}, known: object[], guesses: (number|null)[]}
 *   | {kind: 'category', positions: number[], names: string[]}
 *   | {kind: 'none', guesses: (number|null)[]}} AxisScale
 */

/**
 * Parse a typed tick value: accepts a Unicode minus and thousands separators.
 *
 * @param {string|null|undefined} text
 * @returns {number|null}  null when empty, NaN when it isn't a number
 */
export function parseValue(text) {
    const t = (text ?? '').trim().replace(/−/g, '-').replace(/,/g, '');
    if (t === '')
        return null;
    const v = Number(t);
    return Number.isFinite(v) ? v : NaN;
}

/**
 * An entry with nothing typed.
 *
 * @param {import('../core/axes.js').Axis|null} axis
 * @returns {AxisEntry}
 */
export function defaultEntry(axis) {
    return { type: 'linear', texts: axis ? axis.ticks.map(() => '') : [] };
}

/**
 * The entries remembered for an image, or empty ones; one whose tick count no longer matches the detected ticks is
 * dropped.
 *
 * @param {string} hash  image.js pixelHash of the image
 * @param {import('../core/axes.js').Axes} axes
 * @returns {{x: AxisEntry, y: AxisEntry}}
 */
export function loadEntries(hash, axes) {
    let stored = null;
    try {
        stored = JSON.parse(localStorage.getItem(STORAGE_PREFIX + hash) ?? 'null');
    }
    catch {
        // storage unavailable or corrupt: start fresh
    }
    const out = {};
    for (const k of ['x', 'y']) {
        const axis = axes[k];
        const s = stored?.[k];
        // keep stored values only if the ticks are the same in number (detection may have changed)
        out[k] = s && axis && s.texts.length === axis.ticks.length ? s : defaultEntry(axis);
    }
    return out;
}

/**
 * Remember an image's entries (in localStorage; quietly not, where storage is blocked).
 *
 * @param {string} hash  image.js pixelHash of the image
 * @param {{x: AxisEntry, y: AxisEntry}} entries
 */
export function saveEntries(hash, entries) {
    try {
        localStorage.setItem(STORAGE_PREFIX + hash, JSON.stringify(entries));
    }
    catch {
        // private window or blocked storage: values just aren't remembered
    }
}

/**
 * The scale an axis entry implies: numeric from at least two typed values (scale: core/axes.js fitScale's; known:
 * the ticks with a value; guesses: a value for every tick, guessValues'), categories (the ticks' positions, px, and
 * names), or none yet (guesses only). Null without an axis.
 *
 * @param {import('../core/axes.js').Axis|null} axis
 * @param {AxisEntry} entry
 * @returns {AxisScale|null}
 */
export function axisScale(axis, entry) {
    if (!axis)
        return null;
    if (entry.type === 'category')
        return { kind: 'category', positions: axis.ticks.map(t => t.pos), names: entry.texts.map((t, i) => t.trim() || String(i + 1)) };
    const ticks = axis.ticks.map((t, i) => ({ pos: t.pos, value: parseValue(entry.texts[i]), text: entry.texts[i] }));
    const known = ticks.filter(t => t.value !== null && !Number.isNaN(t.value));
    const scale = fitScale(known, entry.type);
    const guesses = guessValues(ticks.map(t => ({ ...t, value: Number.isNaN(t.value) ? null : t.value })), entry.type);
    return scale ? { kind: 'numeric', scale, known, guesses } : { kind: 'none', guesses };
}

/**
 * A pixel position (along the axis) in data terms: a number, a category name, or null.
 *
 * @param {AxisScale|null} s
 * @param {number} px
 * @returns {number|string|null}
 */
export function toAxisData(s, px) {
    if (!s)
        return null;
    if (s.kind === 'numeric')
        return toData(s.scale, px);
    if (s.kind === 'category') {
        let best = 0;
        s.positions.forEach((p, i) => {
            if (Math.abs(p - px) < Math.abs(s.positions[best] - px))
                best = i;
        });
        return s.names[best];
    }
    return null;
}

/**
 * Data units per pixel at a position, or null.
 *
 * @param {AxisScale|null} s
 * @param {number} px
 * @returns {number|null}
 */
export function axisResolution(s, px) {
    return s?.kind === 'numeric' ? resolution(s.scale, px) : null;
}

/**
 * Format a data value with about as many significant digits as one pixel resolves, plus one.
 *
 * @param {number|string|null|undefined} v  a value, a category name, or nothing ('–')
 * @param {number|null} [res]  data units per px
 * @returns {string}
 */
export function formatData(v, res) {
    if (v === null || v === undefined)
        return '–';
    if (typeof v === 'string')
        return v;
    if (!res || !Number.isFinite(res))
        return v.toPrecision(4);
    const decimals = Math.max(0, Math.min(8, 1 - Math.floor(Math.log10(res))));
    return v.toFixed(decimals);
}

function labelCrop(image, box) {
    const c = document.createElement('canvas');
    if (!box)
        return c;
    const x0 = Math.max(0, box[0] - CROP_PAD);
    const y0 = Math.max(0, box[1] - CROP_PAD);
    const x1 = Math.min(image.width, box[2] + CROP_PAD);
    const y1 = Math.min(image.height, box[3] + CROP_PAD);
    const w = x1 - x0;
    const h = y1 - y0;
    c.width = w;
    c.height = h;
    const data = new ImageData(w, h);
    for (let y = 0; y < h; y++)
        data.data.set(image.data.subarray(4 * ((y0 + y) * image.width + x0), 4 * ((y0 + y) * image.width + x1)), 4 * y * w);
    c.getContext('2d').putImageData(data, 0, 0);
    c.style.height = `${CROP_HEIGHT}px`;
    c.style.width = `${Math.round(CROP_HEIGHT * w / h)}px`;
    return c;
}

/**
 * Render one axis's block: header with type menu, a row per tick (label crop, value, px), and the scale summary.
 *
 * @param {HTMLElement} el
 * @param {string} name  'x' or 'y'
 * @param {import('../core/axes.js').Axis|null} axis  from detectAxes
 * @param {AxisEntry} entry
 * @param {import('../core/image.js').RgbaImage} image
 * @param {function(AxisEntry): void} onChange  the new entry
 */
export function renderAxis(el, name, axis, entry, image, onChange) {
    el.replaceChildren();
    const head = document.createElement('h3');
    head.textContent = `${name.toUpperCase()} axis`;
    el.append(head);
    if (!axis) {
        const p = document.createElement('p');
        p.className = 'fine';
        p.textContent = 'No axis line with ticks or gridlines found.';
        el.append(p);
        return;
    }
    const type = document.createElement('select');
    type.setAttribute('aria-label', `${name} scale type`);
    for (const [v, label] of [['linear', 'linear'], ['log', 'log'], ['category', 'categories']])
        type.add(new Option(label, v));
    type.value = entry.type;
    type.addEventListener('change', () => onChange({ ...entry, type: type.value }));
    head.append(type);
    const source = axis.ticks[0]?.source === 'grid' ? 'gridlines' : 'ticks';
    const info = document.createElement('p');
    info.className = 'fine';
    info.textContent = `${axis.ticks.length} ${source}, ${axis.regular ? 'evenly' : 'unevenly'} spaced (mean ${axis.spacing.toFixed(2)} px).`;
    el.append(info);

    const s = axisScale(axis, entry);
    const table = document.createElement('table');
    axis.ticks.forEach((t, i) => {
        const tr = table.insertRow();
        tr.insertCell().append(labelCrop(image, t.label));
        const input = document.createElement('input');
        input.type = 'text';
        input.inputMode = entry.type === 'category' ? 'text' : 'decimal';
        input.value = entry.texts[i];
        input.setAttribute('aria-label', `${name} tick at ${t.pos.toFixed(1)} px`);
        const guess = s?.guesses?.[i];
        input.placeholder = entry.type === 'category' ? String(i + 1) : (guess === null || guess === undefined ? 'value' : String(guess));
        input.addEventListener('change', () => {
            const texts = [...entry.texts];
            texts[i] = input.value;
            onChange({ ...entry, texts });
        });
        tr.insertCell().append(input);
        const px = tr.insertCell();
        px.className = 'px';
        px.textContent = `${t.pos.toFixed(1)} px`;
    });
    el.append(table);

    const summary = document.createElement('p');
    summary.className = 'fine summary';
    if (!s || s.kind === 'none') {
        summary.textContent = entry.type === 'log' ? 'Type two positive tick values.' : 'Type two tick values; the others are guessed from them.';
    }
    else if (s.kind === 'category') {
        summary.textContent = 'Each dot gets the category of the nearest tick.';
    }
    else {
        const mid = (axis.ticks[0].pos + axis.ticks[axis.ticks.length - 1].pos) / 2;
        const res = axisResolution(s, mid);
        const parts = [`1 px = ${res.toPrecision(3)} ${entry.type === 'log' ? 'at mid-axis (log scale)' : 'data units'}.`];
        if (s.known.length > 2) {
            parts.push(`Typed ticks within ${s.scale.maxResidualPx.toFixed(2)} px of a ${entry.type} scale.`);
            if (s.scale.maxResidualPx > AGREE_PX)
                summary.classList.add('warn');
        }
        else {
            parts.push('Two values fix the scale; type a third to check it.');
        }
        summary.textContent = parts.join(' ');
    }
    el.append(summary);
}

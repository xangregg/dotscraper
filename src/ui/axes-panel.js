// The Axes section: detected ticks with their label crops, their values, the scale they imply, and pixel → data.
//
// An axis entry is { type: 'linear' | 'log' | 'category', texts: [string per tick] }. It starts as the values read
// from the labels, when a scale confirms them (core/labels.js), and is the user's from the first change: remembered
// per image (by pixel hash) in localStorage, so reopening a test image doesn't mean retyping its ticks. Nothing in
// an entry says where a value came from: a tick is marked as read while its value is the reading's (tickStatus).
//
// Detection misses ticks and finds ones that aren't, so the ticks are the user's to change too: once one is added
// (a click in the image) or removed, the entry lists every tick's position (`ticks`) and those are the axis's, in
// place of the detected ones, for the scale only. The mask and the plot's bounds stay the detected axis's.

import { fitScale, toData, resolution, guessValues } from '../core/axes.js';

const STORAGE_PREFIX = 'dotscraper:axes:';
const CROP_HEIGHT = 20; // CSS px, label crops in the tick table
const CROP_PAD = 2; // px of image around a label box
const AGREE_PX = 1; // typed ticks off their fitted position by more than this are flagged

/**
 * @typedef {object} AxisEntry  an axis's values: read from its labels, or typed
 * @property {'linear'|'log'|'category'} type
 * @property {string[]} texts  per tick: its value, or its category's name
 * @property {number[]} [ticks]  px, ascending: every tick's position once the user has added or removed one, in
 *   place of the detected ticks'
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
 * An entry with nothing typed: linear, or categories for an axis without tick marks, whose labels are the positions
 * (core/axes.js labelAxis). Categories are numbered 1, 2, … along the axis: names until the user types theirs, and
 * values at once if the type is changed to linear (dots then get positions between the categories).
 *
 * @param {import('../core/axes.js').Axis|null} axis
 * @returns {AxisEntry}
 */
export function defaultEntry(axis) {
    const category = axis?.ticks[0]?.source === 'label';
    return { type: category ? 'category' : 'linear', texts: axis ? axis.ticks.map((_, i) => (category ? String(i + 1) : '')) : [] };
}

/**
 * The axis an entry's ticks are: the detected one, or, once the user has added or removed a tick, one with the
 * entry's ticks (a detected tick as detected, one placed by hand with its position alone).
 *
 * @param {import('../core/axes.js').Axis|null} axis  the detected axis
 * @param {AxisEntry} entry
 * @returns {import('../core/axes.js').Axis|null}  null with no detected axis and no tick placed
 */
export function axisInEffect(axis, entry) {
    if (!entry.ticks)
        return axis;
    if (!entry.ticks.length && !axis)
        return null;
    const ticks = entry.ticks.map(pos => axis?.ticks.find(t => t.pos === pos) ?? { pos, label: null, source: 'user' });
    const gaps = ticks.slice(1).map((t, i) => t.pos - ticks[i].pos);
    const spacing = gaps.length ? gaps.reduce((a, b) => a + b, 0) / gaps.length : null;
    return { line: axis?.line ?? null, ticks, spacing, regular: gaps.every(g => Math.abs(g - spacing) <= AGREE_PX) };
}

/**
 * An axis's reading for the ticks an entry has: the detected ticks' own, nothing for a tick placed by hand.
 *
 * @param {import('../core/axes.js').Axis|null} axis  the detected axis
 * @param {import('../core/labels.js').AxisReading|null} reading  of the detected ticks
 * @param {AxisEntry} entry
 * @returns {import('../core/labels.js').AxisReading|null}
 */
export function readingInEffect(axis, reading, entry) {
    if (!entry.ticks || !reading)
        return reading;
    const from = entry.ticks.map(pos => axis.ticks.findIndex(t => t.pos === pos));
    const pick = (list, none) => from.map(j => (j < 0 ? none : list[j]));
    return {
        read: pick(reading.read, null),
        scale: reading.scale && { ...reading.scale, values: pick(reading.scale.values, null), agree: pick(reading.scale.agree, false) },
    };
}

/**
 * An entry with one more tick, placed by hand, without a value yet.
 *
 * @param {import('../core/axes.js').Axis|null} axis  the detected axis
 * @param {AxisEntry} entry
 * @param {number} pos  px along the axis
 * @returns {AxisEntry}
 */
export function addTick(axis, entry, pos) {
    const ticks = [...(entry.ticks ?? axis?.ticks.map(t => t.pos) ?? [])];
    const texts = [...entry.texts];
    const i = ticks.filter(p => p < pos).length;
    ticks.splice(i, 0, pos);
    texts.splice(i, 0, '');
    return { ...entry, texts, ticks };
}

/**
 * An entry without one of its ticks.
 *
 * @param {import('../core/axes.js').Axis|null} axis  the detected axis
 * @param {AxisEntry} entry
 * @param {number} i  the tick, among the entry's
 * @returns {AxisEntry}
 */
export function removeTick(axis, entry, i) {
    const ticks = entry.ticks ?? axis.ticks.map(t => t.pos);
    return { ...entry, texts: entry.texts.filter((_, j) => j !== i), ticks: ticks.filter((_, j) => j !== i) };
}

/**
 * The entry an axis's labels give: the values of the scale that confirms them, or null without one.
 *
 * @param {import('../core/labels.js').AxisReading|null} reading
 * @returns {AxisEntry|null}
 */
export function readEntry(reading) {
    return reading?.scale ? { type: reading.scale.type, texts: reading.scale.values.map(v => (v === null ? '' : String(v))) } : null;
}

/**
 * Where a tick's value comes from, while it is the reading's (null once it is typed over, or without a reading):
 * 'read' from its own label, which fits the scale of the others; 'scale' when its label wasn't read or it has none,
 * and the value is the scale's there; 'differs' when its label reads as another number (`label`), which the scale
 * of the others contradicts.
 *
 * @param {AxisEntry} entry
 * @param {import('../core/labels.js').AxisReading|null} reading
 * @param {number} i  the tick
 * @returns {{kind: 'read'|'scale'|'differs', label?: string}|null}
 */
export function tickStatus(entry, reading, i) {
    const scale = reading?.scale;
    if (!scale || entry.type !== scale.type || entry.texts[i] !== String(scale.values[i]))
        return null;
    if (scale.agree[i])
        return { kind: 'read' };
    const label = reading.read[i];
    return label && label.value !== null ? { kind: 'differs', label: label.text } : { kind: 'scale' };
}

/**
 * The entries remembered for an image, or else those read from its labels, or else empty ones. A remembered one
 * whose tick count no longer matches the detected ticks is dropped, unless it has ticks of its own.
 *
 * @param {string} hash  image.js pixelHash of the image
 * @param {import('../core/axes.js').Axes} axes
 * @param {{x: import('../core/labels.js').AxisReading|null, y: import('../core/labels.js').AxisReading|null}} readings
 *   core/labels.js readTickLabels'
 * @returns {{x: AxisEntry, y: AxisEntry}}
 */
export function loadEntries(hash, axes, readings) {
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
        // keep stored values only if the ticks are the same in number (detection may have changed) or the entry's own
        const fits = s && (s.ticks ? s.texts.length === s.ticks.length : axis && s.texts.length === axis.ticks.length);
        // (categories nobody named, as they were stored before they were numbered: numbered now)
        const unnamed = s && !s.ticks && s.type === 'category' && s.texts.every(t => t === '');
        out[k] = fits && !unnamed ? s : readEntry(readings[k]) ?? defaultEntry(axis);
    }
    return out;
}

/**
 * Remember an image's entries (in localStorage; quietly not, where storage is blocked). An entry that is just what
 * its labels read as isn't kept: it is read again next time, by the reader as it then is.
 *
 * @param {string} hash  image.js pixelHash of the image
 * @param {{x: AxisEntry, y: AxisEntry}} entries
 * @param {{x: import('../core/labels.js').AxisReading|null, y: import('../core/labels.js').AxisReading|null}} readings
 */
export function saveEntries(hash, entries, readings) {
    const kept = {};
    for (const k of ['x', 'y']) {
        const asRead = readEntry(readings[k]);
        const same = asRead && !entries[k].ticks && asRead.type === entries[k].type && asRead.texts.every((t, i) => t === entries[k].texts[i]);
        kept[k] = same ? null : entries[k];
    }
    try {
        localStorage.setItem(STORAGE_PREFIX + hash, JSON.stringify(kept));
    }
    catch {
        // private window or blocked storage: values just aren't remembered
    }
}

/**
 * The scale an axis entry implies: numeric from at least two values (scale: core/axes.js fitScale's; known:
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

// back from ticks changed by hand to the detected ones, with the values their labels give
function detectedTicksButton(detected, reading, onChange) {
    const button = document.createElement('button');
    button.type = 'button';
    button.textContent = 'Detected ticks';
    button.title = 'Drop the ticks placed by hand and bring back the removed ones';
    button.addEventListener('click', () => onChange(readEntry(reading) ?? defaultEntry(detected)));
    return button;
}

// a tick's label as the image has it: its box from the axis and the glyphs read, which the box can cut short
function labelCrop(image, tickBox, readBox) {
    const c = document.createElement('canvas');
    const boxes = [tickBox, readBox].filter(Boolean);
    if (!boxes.length)
        return c;
    const box = [Math.min(...boxes.map(b => b[0])), Math.min(...boxes.map(b => b[1])), Math.max(...boxes.map(b => b[2])), Math.max(...boxes.map(b => b[3]))];
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
 * Render one axis's block: header with type menu, a row per tick (label crop, value, where the value is from, px,
 * a button to remove it), and the scale summary.
 *
 * @param {HTMLElement} el
 * @param {string} name  'x' or 'y'
 * @param {import('../core/axes.js').Axis|null} detected  from detectAxes
 * @param {AxisEntry} entry
 * @param {import('../core/labels.js').AxisReading|null} detectedReading  the detected axis's labels as read
 * @param {import('../core/image.js').RgbaImage} image
 * @param {function(AxisEntry): void} onChange  the new entry
 * @param {{on: boolean, toggle: function(): void}} adding  whether a click in the image adds a tick to this axis
 *   now, and the switch for it
 */
export function renderAxis(el, name, detected, entry, detectedReading, image, onChange, adding) {
    el.replaceChildren();
    const head = document.createElement('h3');
    head.textContent = `${name.toUpperCase()} axis`;
    el.append(head);
    const axis = axisInEffect(detected, entry);
    const reading = readingInEffect(detected, detectedReading, entry);
    const add = document.createElement('button');
    add.type = 'button';
    add.textContent = 'Add tick';
    add.title = `Then click in the image where a tick is: its ${name} is the tick's position. Esc to finish.`;
    add.setAttribute('aria-pressed', String(adding.on));
    add.addEventListener('click', adding.toggle);
    if (!axis) {
        head.append(add);
        const p = document.createElement('p');
        p.className = 'fine';
        p.textContent = `${detected ? 'Every tick removed.' : 'No axis line with ticks or gridlines found.'} To place ticks by hand: Add tick, then click in the image at two or more labeled positions, and type their values.`;
        el.append(p);
        if (detected)
            el.append(detectedTicksButton(detected, detectedReading, onChange));
        return;
    }
    const type = document.createElement('select');
    type.setAttribute('aria-label', `${name} scale type`);
    for (const [v, label] of [['linear', 'linear'], ['log', 'log'], ['category', 'categories']])
        type.add(new Option(label, v));
    type.value = entry.type;
    type.addEventListener('change', () => onChange({ ...entry, type: type.value }));
    head.append(type);
    const statuses = axis.ticks.map((_, i) => tickStatus(entry, reading, i));
    const asRead = readEntry(reading);
    if (asRead) {
        // back to the reading after typing over it, or rid of it
        const whole = statuses.every(Boolean);
        const button = document.createElement('button');
        button.type = 'button';
        button.textContent = whole ? 'Clear' : 'Use read values';
        // (the ticks stay as the user has them)
        const ticks = entry.ticks ? { ticks: entry.ticks } : {};
        button.addEventListener('click', () => onChange(whole ? { ...defaultEntry(axis), type: entry.type, ...ticks } : { ...asRead, ...ticks }));
        head.append(button);
    }
    head.append(add);
    const source = { grid: 'gridlines', label: 'labels' }[detected?.ticks[0]?.source] ?? 'ticks';
    const info = document.createElement('p');
    info.className = 'fine';
    const spaced = axis.ticks.length > 1 ? `, ${axis.regular ? 'evenly' : 'unevenly'} spaced (mean ${axis.spacing.toFixed(2)} px)` : '';
    if (entry.ticks) {
        const placed = axis.ticks.filter(t => t.source === 'user').length;
        const removed = (detected?.ticks.length ?? 0) - (axis.ticks.length - placed);
        const parts = [placed ? `${placed} placed by hand` : '', removed ? `${removed} of the ${detected.ticks.length} detected removed` : ''].filter(Boolean);
        info.textContent = `${axis.ticks.length} ${axis.ticks.length === 1 ? 'tick' : 'ticks'}${spaced}${parts.length ? `: ${parts.join(', ')}` : ''}. `;
        if (detected)
            info.append(detectedTicksButton(detected, detectedReading, onChange));
    }
    else {
        // (a label's center is where its category is, to a pixel or two: bench3's "1" and "2" lie 1 and 1.5 px left
        // of their swarms' centers)
        info.textContent = `${axis.ticks.length} ${source}${spaced}.`
            + (source === 'labels' ? ' No tick marks: each position is its label\'s center.' : '');
    }
    el.append(info);

    const s = axisScale(axis, entry);
    const table = document.createElement('table');
    axis.ticks.forEach((t, i) => {
        const tr = table.insertRow();
        tr.insertCell().append(labelCrop(image, t.label, reading?.read[i]?.box));
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
        const from = tr.insertCell();
        from.className = 'from';
        const status = statuses[i];
        if (status?.kind === 'read') {
            from.textContent = 'read';
            from.title = 'Read from this label; it fits the scale of the axis\'s labels.';
        }
        else if (status?.kind === 'scale') {
            from.textContent = 'scale';
            from.title = t.label ? 'This label wasn\'t read: the value is what the scale of the other labels gives here.'
                : 'No label here: the value is what the scale of the axis\'s labels gives.';
        }
        else if (status?.kind === 'differs') {
            from.textContent = `label: ${status.label}`;
            from.classList.add('warn');
            from.title = `This label reads ${status.label}, which the scale of the other labels contradicts: the value is the scale's here. Check the tick and its label.`;
        }
        else if (t.source === 'user') {
            from.textContent = 'placed';
            from.title = 'Placed by hand, at the pixel clicked.';
        }
        const px = tr.insertCell();
        px.className = 'px';
        px.textContent = `${t.pos.toFixed(1)} px`;
        const remove = document.createElement('button');
        remove.type = 'button';
        remove.className = 'remove';
        remove.textContent = '×';
        remove.title = 'Remove this tick';
        remove.setAttribute('aria-label', `Remove the ${name} tick at ${t.pos.toFixed(1)} px`);
        remove.addEventListener('click', () => onChange(removeTick(detected, entry, i)));
        tr.insertCell().append(remove);
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
        const read = statuses.filter(st => st?.kind === 'read').length;
        const fromReading = statuses.filter(Boolean).length;
        if (fromReading) {
            const labels = axis.ticks.filter(t => t.label).length;
            const rest = fromReading - read;
            parts.push(`Read from the labels: ${read} of ${labels} fit a ${entry.type} scale${rest ? `; the scale gives the other ${rest === 1 ? 'tick' : `${rest} ticks`}` : ''}.`);
            // two labels fit any scale: say what checked this one, and what that check can't see
            if (reading.scale.minor)
                parts.push(`Two labels alone prove no scale: this one is checked by ${reading.scale.minor} ticks without labels, which it puts on 2 to 9 times a power of ten. Both labels off by the same power of ten would pass too.`);
            parts.push('Type over any that is wrong.');
            if (statuses.some(st => st?.kind === 'differs'))
                summary.classList.add('warn');
        }
        if (s.known.length > 2) {
            parts.push(`${fromReading ? 'Ticks' : 'Typed ticks'} within ${s.scale.maxResidualPx.toFixed(2)} px of a ${entry.type} scale.`);
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

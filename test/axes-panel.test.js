import { test } from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, readdirSync } from 'node:fs';
import { detectAxes } from '../src/core/axes.js';
import { readTickLabels } from '../src/core/labels.js';
import { readEntry, tickStatus, loadEntries, axisScale, toAxisData } from '../src/ui/axes-panel.js';
import { readImage } from '../tools/png.mjs';

// (Node has no localStorage: loadEntries finds nothing remembered, as in a private window)

const near = (a, b, tol, what) => assert.ok(Math.abs(a - b) <= tol, `${what}: ${a} vs ${b} (±${tol})`);

// a reading as labels.js gives it: kite2's y axis, whose "2" sits off the scale of the others, with an unread label
const reading = {
    read: [{ text: '12', value: 12 }, { text: '1?', value: null }, { text: '8', value: 8 }, { text: '6', value: 6 }, { text: '2', value: 2 }, null],
    scale: { type: 'linear', values: [12, 10, 8, 6, 1.9, 0], agree: [true, false, true, true, false, false], count: 3 },
};

test('readEntry: the scale\'s values and type, and nothing without a scale', () => {
    assert.deepEqual(readEntry(reading), { type: 'linear', texts: ['12', '10', '8', '6', '1.9', '0'] });
    assert.equal(readEntry({ read: reading.read, scale: null }), null);
    assert.equal(readEntry(null), null);
});

test('tickStatus: read, from the scale, or against its label; nothing once typed over', () => {
    const entry = readEntry(reading);
    assert.deepEqual(entry.texts.map((_, i) => tickStatus(entry, reading, i)), [
        { kind: 'read' }, { kind: 'scale' }, { kind: 'read' }, { kind: 'read' }, { kind: 'differs', label: '2' }, { kind: 'scale' },
    ]);
    // the user takes the label's word for the fifth tick: that value is theirs, the others still the reading's
    const typed = { ...entry, texts: ['12', '10', '8', '6', '2', '0'] };
    assert.equal(tickStatus(typed, reading, 4), null);
    assert.deepEqual(tickStatus(typed, reading, 0), { kind: 'read' });
    // on another scale type the values aren't the reading's
    assert.equal(tickStatus({ ...entry, type: 'log' }, reading, 0), null);
    assert.equal(tickStatus(entry, { read: reading.read, scale: null }, 0), null);
    assert.equal(tickStatus(entry, null, 0), null);
});

test('loadEntries: an axis starts as read; one without a confirmed scale starts empty, a labels-only one as categories', () => {
    const ticks = n => Array.from({ length: n }, (_, i) => ({ pos: 100 + 50 * i, length: 6, side: 'before', label: null, source: 'tick' }));
    const axes = { x: { ticks: ticks(2).map(t => ({ ...t, source: 'label' })) }, y: { ticks: ticks(6) } };
    const entries = loadEntries('no-such-image', axes, { x: { read: [], scale: null }, y: reading });
    assert.deepEqual(entries.x, { type: 'category', texts: ['', ''] });
    assert.deepEqual(entries.y, readEntry(reading));
    assert.deepEqual(loadEntries('no-such-image', { x: null, y: axes.y }, { x: null, y: { read: [], scale: null } }), {
        x: { type: 'linear', texts: [] }, y: { type: 'linear', texts: ['', '', '', '', '', ''] },
    });
});

const PLOS = new URL('../bench-plos/', import.meta.url).pathname;
const plos = name => existsSync(PLOS) && readdirSync(PLOS).find(f => f.startsWith(`${name}-`) && f.endsWith('.png'));

test('bench-plos step1: from the image to data units with nothing typed', { skip: !plos('step1') && 'bench-plos absent' }, () => {
    const image = readImage(PLOS + plos('step1'));
    const axes = detectAxes(image);
    const readings = readTickLabels(image, axes);
    const entries = loadEntries('no-such-image', axes, readings);
    assert.deepEqual(entries.x.texts, ['40', '50', '60', '70', '80', '90']);
    assert.deepEqual(entries.y.texts, ['1.4', '1.3', '1.2', '1.1', '1', '0.9', '0.8', '0.7', '0.6']);
    const x = axisScale(axes.x, entries.x);
    const y = axisScale(axes.y, entries.y);
    assert.equal(x.kind, 'numeric');
    assert.equal(y.kind, 'numeric');
    // each tick's pixel position maps to its value, and the ticks fit their scales within a pixel
    near(toAxisData(x, axes.x.ticks[2].pos), 60, 0.05, 'x at the third tick');
    near(toAxisData(y, axes.y.ticks[0].pos), 1.4, 0.002, 'y at the top tick');
    assert.ok(x.scale.maxResidualPx < 1 && y.scale.maxResidualPx < 1, `${x.scale.maxResidualPx}, ${y.scale.maxResidualPx}`);
    for (const k of ['x', 'y'])
        assert.ok(axes[k].ticks.every((_, i) => tickStatus(entries[k], readings[k], i)?.kind === 'read'), k);
});

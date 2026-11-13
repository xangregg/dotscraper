// Ground truth in data units: a directory of renderings of one data set (bench1/, bench2/, bench3/) with the data
// in data.csv.
//
// data.csv: '#' header lines, then a header row and one row per dot (x, y, optional group), duplicates kept:
//   # x ticks: 65, 70, 75, 80, 85, 90, 95, 100
//   # x scale: linear, min=65, max=100
//   # y scale: linear, min=30, max=90
//   # plot area: left=70, top=30, width=546, height=436
//   Score,Response,group
//   97.2,74,1
//
// The plot area is in px and is the same in every image of the set: its edge pixels' centers are at the scale's
// min and max (measured on bench1: fitted dots regress onto the data with x(65) = 0.50 px into the plot area and
// x(100) = width − 0.50 px, residuals 0.01–0.03 px). Through it, truth is exact; through detected ticks it would
// be off by up to 0.25 px, because JMP draws ticks on whole pixels.
//
// A test image's own truth (testgraphs/<name>.csv) may have no plot area: its tick values are then paired with the
// ticks detected in the image (truthPixelsByTicks), and positions are interpolated between ticks in the scale's
// transform. Scale types: "linear" and "log y+1" (log(v + 1)); "with jitter" after the type means each dot was
// moved by a random amount, so the truth gives its cell, not its position.
//
// When the x tick list is exactly the set of distinct x values (bench3: x is a group, ticks 1 and 2), x names a
// lane: a beeswarm or jitter layout chose each dot's x, and only its lane and its y are known. Rows without an x
// are not drawn and are dropped.
//
// Ticks may be category names (# x ticks: early, late) when the data names its groups instead of numbering them:
// the names become the lanes 1, 2, … in the listed order (left to right, or bottom to top for y), and each row's
// name is replaced by its number.

import { readFileSync } from 'node:fs';

/**
 * @param {string} path
 * @returns {{xScale: {type: string, min: number, max: number}, yScale: {type: string, min: number, max: number},
 *            xTicks: number[]|null, yTicks: number[]|null, plotArea: {left: number, top: number, width: number,
 *            height: number}|null, lanes: boolean, columns: string[], rows: {x: number, y: number, group: string|null}[]}}
 *   data units; plotArea in px
 */
export function readTruthCsv(path) {
    const lines = readFileSync(path, 'utf8').replace(/^﻿/, '').split(/\r?\n/);
    const header = {};
    const body = [];
    for (const line of lines) {
        if (line.startsWith('#')) {
            const m = line.slice(1).match(/^\s*([^:]+):\s*(.*)$/);
            if (m)
                header[m[1].trim().toLowerCase()] = m[2].trim();
        }
        else if (line.trim()) {
            body.push(line.split(',').map(s => s.trim()));
        }
    }
    const pairs = text => Object.fromEntries(text.split(',').map(s => s.split('=').map(t => t.trim())).filter(kv => kv.length === 2));
    const scale = text => {
        if (!text)
            return null;
        const kv = pairs(text);
        return { type: text.split(',')[0].trim(), min: Number(kv.min), max: Number(kv.max), jitter: /with jitter/i.test(text) };
    };
    const names = text => (text ? text.split(',').map(s => s.trim()) : null);
    const xNames = names(header['x ticks']);
    const yNames = names(header['y ticks']);
    // category names: every tick isn't a number
    const categories = list => (list?.some(s => !Number.isFinite(Number(s))) ? list : null);
    const xCategories = categories(xNames);
    const yCategories = categories(yNames);
    const ticks = (list, cats) => (cats ? cats.map((_, i) => i + 1) : list?.map(Number) ?? null);
    const value = (cell, cats, axis) => {
        if (!cats)
            return Number(cell);
        const i = cats.indexOf(cell);
        if (i < 0)
            throw new Error(`${path}: ${axis} value "${cell}" isn't one of the tick names ${cats.join(', ')}`);
        return i + 1;
    };
    const area = text => {
        if (!text)
            return null;
        const kv = pairs(text);
        return { left: Number(kv.left), top: Number(kv.top), width: Number(kv.width), height: Number(kv.height) };
    };
    const [columns, ...rows] = body;
    const dots = rows.filter(r => r[0] !== '' && r[1] !== '')
        .map(r => ({ x: value(r[0], xCategories, 'x'), y: value(r[1], yCategories, 'y'), group: r[2] || null }));
    const xTicks = ticks(xNames, xCategories);
    const distinctX = [...new Set(dots.map(d => d.x))].sort((a, b) => a - b);
    const lanes = !!xTicks && xTicks.length === distinctX.length && xTicks.every((t, i) => t === distinctX[i]);
    return {
        xScale: scale(header['x scale']), yScale: scale(header['y scale']), xTicks, yTicks: ticks(yNames, yCategories),
        plotArea: area(header['plot area']), lanes, columns, rows: dots,
    };
}

/**
 * Data units to px through the plot area.
 *
 * @param {ReturnType<typeof readTruthCsv>} truth
 * @returns {{x: (v: number) => number, y: (v: number) => number}}  px
 */
export function plotAreaMapping(truth) {
    const { xScale, yScale, plotArea: a } = truth;
    const sx = (a.width - 1) / (xScale.max - xScale.min);
    const sy = (a.height - 1) / (yScale.max - yScale.min);
    return { x: v => a.left + 0.5 + (v - xScale.min) * sx, y: v => a.top + 0.5 + (yScale.max - v) * sy };
}

/**
 * Truth dots in pixels of a rendering. For lane data, x is the lane's center and `lane` names the lane.
 *
 * @param {ReturnType<typeof readTruthCsv>} truth
 * @returns {{dots: {x: number, y: number, group: string|null, lane?: number}[], laneCenters: number[]|null}|{error: string}}
 *   px
 */
export function truthPixels(truth) {
    if (truth.xScale?.type !== 'linear' || truth.yScale?.type !== 'linear')
        return { error: 'only linear scales so far' };
    if (!truth.plotArea)
        return { error: 'no plot area in the header' };
    const map = plotAreaMapping(truth);
    const dots = truth.rows.map(r => ({ x: map.x(r.x), y: map.y(r.y), group: r.group, ...(truth.lanes ? { lane: r.x } : {}) }));
    return { dots, laneCenters: truth.lanes ? truth.xTicks.map(map.x) : null };
}

// the scale's transform: values are interpolated linearly in it between ticks
function transformOf(type) {
    if (type === 'linear')
        return v => v;
    if (/^log y\s*\+\s*1$/i.test(type))
        return v => Math.log1p(v);
    return null;
}

// px of v from (value, px) tick pairs sorted by value, linear in f between neighbors, extrapolated at the ends
function interpolate(ticks, f, v) {
    let i = 0;
    while (i < ticks.length - 2 && f(v) > f(ticks[i + 1][0]))
        i++;
    const [a, pa] = ticks[i];
    const [b, pb] = ticks[i + 1];
    return pa + (f(v) - f(a)) / (f(b) - f(a)) * (pb - pa);
}

/**
 * Truth dots in pixels through the image's detected ticks (axes.detectAxes), paired in order with the header's tick
 * values: x ticks left to right, y ticks bottom to top.
 *
 * @param {ReturnType<typeof readTruthCsv>} truth
 * @param {{x: {ticks: {pos: number}[]}|null, y: {ticks: {pos: number}[]}|null}} axes  tick positions in px
 * @returns {{dots: {x: number, y: number, cell: number[], lane?: number}[], jitter: boolean, laneCenters: number[]|null}
 *   |{error: string}}  px; cell: the data values; for lane data (truth.lanes), each dot's lane and the lanes' x
 */
export function truthPixelsByTicks(truth, axes) {
    const fx = transformOf(truth.xScale?.type);
    const fy = transformOf(truth.yScale?.type);
    if (!fx || !fy)
        return { error: `scale types ${truth.xScale?.type} and ${truth.yScale?.type} aren't supported` };
    const xs = (axes.x?.ticks ?? []).map(t => t.pos);
    const ys = (axes.y?.ticks ?? []).map(t => t.pos);
    if (!truth.xTicks || !truth.yTicks || xs.length !== truth.xTicks.length || ys.length !== truth.yTicks.length)
        return { error: `${xs.length} x and ${ys.length} y ticks detected for ${truth.xTicks?.length} and ${truth.yTicks?.length} in the header` };
    const xt = [...truth.xTicks].sort((a, b) => a - b).map((v, i) => [v, [...xs].sort((a, b) => a - b)[i]]);
    const yt = [...truth.yTicks].sort((a, b) => a - b).map((v, i) => [v, [...ys].sort((a, b) => b - a)[i]]);
    const dots = truth.rows.map(r => ({ x: interpolate(xt, fx, r.x), y: interpolate(yt, fy, r.y), cell: [r.x, r.y], ...(truth.lanes ? { lane: r.x } : {}) }));
    return { dots, jitter: !!(truth.xScale.jitter || truth.yScale.jitter), laneCenters: truth.lanes ? xt.map(([, px]) => px) : null };
}

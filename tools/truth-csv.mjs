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
// When the x tick list is exactly the set of distinct x values (bench3: x is a group, ticks 1 and 2), x names a
// lane: a beeswarm or jitter layout chose each dot's x, and only its lane and its y are known. Rows without an x
// are not drawn and are dropped.

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
        return { type: text.split(',')[0].trim(), min: Number(kv.min), max: Number(kv.max) };
    };
    const ticks = text => (text ? text.split(',').map(Number) : null);
    const area = text => {
        if (!text)
            return null;
        const kv = pairs(text);
        return { left: Number(kv.left), top: Number(kv.top), width: Number(kv.width), height: Number(kv.height) };
    };
    const [columns, ...rows] = body;
    const dots = rows.filter(r => r[0] !== '' && r[1] !== '').map(r => ({ x: Number(r[0]), y: Number(r[1]), group: r[2] || null }));
    const xTicks = ticks(header['x ticks']);
    const distinctX = [...new Set(dots.map(d => d.x))].sort((a, b) => a - b);
    const lanes = !!xTicks && xTicks.length === distinctX.length && xTicks.every((t, i) => t === distinctX[i]);
    return {
        xScale: scale(header['x scale']), yScale: scale(header['y scale']), xTicks, yTicks: ticks(header['y ticks']),
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

// Ground truth placed by hand, in image px: the app's Save truth JSON (testgraphs/truth/, bench-nature/truth/).
//
//   { "image": "kite1-….png", "hash": "53f2db19", "note": "…",
//     "families": [{ "family": 0, "color": [r, g, b], "radius": 7.5 }, …],
//     "dots": [{ "x": 197.4, "y": 106.81, "family": 0 }, { …, "fill": 1, "fillSure": true }, …] }
//
// x and y are px (pixel i spans i to i + 1); a dot's group for scoring is its family, and its fill where an outlined
// family's dots have several (outline.js dotFamilies). The hash is image.js pixelHash of the image as decoded, so a
// truth file can't silently pair with another rendering; JPEG decoders differ, so for a JPEG it can't be checked.
//
// Run directly, it turns one of the app's Save CSV exports into a truth file (for exports made before Save truth
// was the way to record truth):
//   node tools/truth-json.mjs bench-nature/engagement1-….csv bench-nature/engagement1-….png > bench-nature/truth/engagement1-….json

import { readFileSync } from 'node:fs';
import { basename } from 'node:path';
import { fileURLToPath } from 'node:url';
import { pixelHash } from '../src/core/image.js';
import { readImage } from './png.mjs';

/**
 * @param {string} path
 * @returns {{image: string|undefined, hash: string|undefined, families: {family: number, color: number[]}[],
 *            dots: {x: number, y: number, family: number, group: string}[]}}  px; group: family, and fill if any
 */
export function readTruthJson(path) {
    const truth = JSON.parse(readFileSync(path, 'utf8'));
    const dots = truth.dots.map(d => ({ ...d, family: d.family ?? 0, group: d.fill === undefined ? `${d.family ?? 0}` : `${d.family ?? 0}.${d.fill}` }));
    return { image: truth.image, hash: truth.hash, families: truth.families ?? [], dots };
}

/**
 * Whether a CSV is the app's dot export (Save CSV, Copy table) rather than truth in data units.
 *
 * @param {string} text
 * @returns {boolean}
 */
export function isDotsExport(text) {
    const first = text.replace(/^﻿/, '').split(/\r?\n/).find(l => l.trim() && !l.startsWith('#')) ?? '';
    return /^family,x_px,y_px\b/.test(first);
}

/**
 * Truth JSON from a dots export: families from the first column ('2 #bab8f7'), fills from the fill column
 * ('1 #a0c4ff', '1? …' where the fill was mostly hidden).
 *
 * @param {string} text  the export
 * @param {string} imageName
 * @param {string} hash  pixelHash of the image
 * @returns {{image: string, hash: string, note: string, families: {family: number, color: number[]}[],
 *            dots: {x: number, y: number, family: number, fill?: number, fillSure?: boolean}[]}}  the truth, as Save
 *   truth writes it (see the top of this file)
 */
export function truthFromExport(text, imageName, hash) {
    const rows = text.replace(/^﻿/, '').trim().split(/\r?\n/).map(l => l.split(','));
    const [head, ...body] = rows;
    const col = name => head.indexOf(name);
    const families = new Map();
    const dots = body.map(r => {
        const [num, color] = r[col('family')].split(' ');
        const family = Number(num) - 1;
        if (!families.has(family))
            families.set(family, color ? [1, 3, 5].map(i => parseInt(color.slice(i, i + 2), 16)) : null);
        const d = { x: Number(r[col('x_px')]), y: Number(r[col('y_px')]), family };
        const fill = col('fill') >= 0 ? r[col('fill')] : '';
        if (fill)
            Object.assign(d, { fill: parseInt(fill, 10) - 1, fillSure: !fill.split(' ')[0].endsWith('?') });
        return d;
    });
    return {
        image: imageName, hash,
        note: 'Dot centers in image px, converted from a dots export. Review every dot before using this as ground truth.',
        families: [...families].sort((a, b) => a[0] - b[0]).map(([family, color]) => ({ family, color })),
        dots,
    };
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
    const [csv, png] = process.argv.slice(2);
    const text = readFileSync(csv, 'utf8');
    if (!isDotsExport(text))
        throw new Error(`${csv} isn't a dots export (its header should start family,x_px,y_px)`);
    process.stdout.write(JSON.stringify(truthFromExport(text, basename(png), pixelHash(readImage(png))), null, 1) + '\n');
}

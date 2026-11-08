// Benchmark the extraction (PLAN.md §2.6): synthetic presets over several seeds, plus every local test image with
// ground truth in testgraphs/truth/<name>.json ({ "dots": [{ "x", "y" }, …] } in px). Prints precision, recall,
// count error, position RMSE and time; tolerance r/2 for matching.
//
// Usage: node tools/bench.mjs [--seeds n] [--preset name] [--verbose]

import { existsSync, readdirSync, readFileSync } from 'node:fs';
import { join, dirname, basename } from 'node:path';
import { fileURLToPath } from 'node:url';
import { readPng } from './png.mjs';
import { synthGraph, SYNTH_PRESETS } from '../src/core/synth.js';
import { estimateStyle } from '../src/core/style.js';
import { buildMask } from '../src/core/layers.js';
import { fitDots, houghCenters } from '../src/core/fit.js';
import { matchDots } from '../src/core/metrics.js';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const TRUTH_DIR = join(ROOT, 'testgraphs', 'truth');
const MATCH_TOL_RADII = 0.5;

function parseArgs(argv) {
    const opts = { seeds: 5, preset: null, verbose: false };
    for (let i = 0; i < argv.length; i++) {
        if (argv[i] === '--seeds')
            opts.seeds = Number(argv[++i]);
        else if (argv[i] === '--preset')
            opts.preset = argv[++i];
        else if (argv[i] === '--verbose')
            opts.verbose = true;
    }
    return opts;
}

/** Style, mask and fit for one image; also scores the Hough candidates alone. */
export function extract(image) {
    const est = estimateStyle(image);
    const layers = buildMask(image, est);
    const input = { image, style: est.style, line: est.familyMeasure.line, mask: layers.mask, D: est.D };
    const fit = fitDots(input);
    const hough = houghCenters(est.D, layers.mask.map(m => 1 - m), image.width, image.height, est.style.radius);
    return { est, layers, fit, hough };
}

function score(truth, found, r) {
    const m = matchDots(truth, found, MATCH_TOL_RADII * r);
    return m;
}

function row(name, truth, res, ms) {
    const r = res.est.style.radius;
    const m = score(truth, res.fit.dots, r);
    const h = score(truth, res.hough, r);
    const flagged = res.fit.dots.filter(d => d.flags.length).length;
    return { name, n: truth.length, found: res.fit.dots.length, precision: m.precision, recall: m.recall, countError: m.countError, rmse: m.rmse, houghRecall: h.recall, houghPrecision: h.precision, flagged, ms, radius: res.fit.radius, m };
}

function print(rows) {
    const f = (v, d = 3) => (Number.isNaN(v) ? '–' : v.toFixed(d));
    console.log('case                     truth found  prec  recall  count  rmse px  hough P/R    flagged  radius   ms');
    for (const r of rows)
        console.log(`${r.name.padEnd(24)} ${String(r.n).padStart(5)} ${String(r.found).padStart(5)} ${f(r.precision)} ${f(r.recall)} ${String(r.countError).padStart(6)} ${f(r.rmse).padStart(8)}  ${f(r.houghPrecision, 2)}/${f(r.houghRecall, 2)}  ${String(r.flagged).padStart(7)} ${f(r.radius, 2).padStart(7)} ${String(r.ms).padStart(5)}`);
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
    const opts = parseArgs(process.argv.slice(2));
    const rows = [];
    for (const name of Object.keys(SYNTH_PRESETS)) {
        if (opts.preset && name !== opts.preset)
            continue;
        for (let seed = 1; seed <= opts.seeds; seed++) {
            const { image, truth } = synthGraph(SYNTH_PRESETS[name](seed));
            const t0 = Date.now();
            const res = extract(image);
            const r = row(`${name} #${seed}`, truth, res, Date.now() - t0);
            rows.push(r);
            if (opts.verbose) {
                console.log(r.name, 'missed', r.m.missed.map(i => `${truth[i].x.toFixed(1)},${truth[i].y.toFixed(1)}`).join(' '), 'extra', r.m.extra.map(j => `${res.fit.dots[j].x.toFixed(1)},${res.fit.dots[j].y.toFixed(1)}`).join(' '));
                console.log('   log', JSON.stringify(res.fit.log));
            }
        }
    }
    if (existsSync(TRUTH_DIR) && !opts.preset) {
        for (const file of readdirSync(TRUTH_DIR).filter(f => f.endsWith('.json'))) {
            const truth = JSON.parse(readFileSync(join(TRUTH_DIR, file), 'utf8'));
            const img = join(ROOT, 'testgraphs', truth.image ?? basename(file, '.json') + '.png');
            const t0 = Date.now();
            const res = extract(readPng(img));
            rows.push(row(basename(file, '.json'), truth.dots, res, Date.now() - t0));
        }
    }
    print(rows);
}

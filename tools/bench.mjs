// Benchmark the extraction (PLAN.md §2.6): synthetic presets over several seeds, every local test image with
// ground truth in testgraphs/truth/<name>.json ({ "dots": [{ "x", "y" }, …] } in px), and every PNG and JPEG of a
// truth set (a directory with data.csv in data units, tools/truth-csv.mjs; bench1–3/). Prints precision, recall,
// count error, position RMSE and time; tolerance r/2 for matching.
//
// Usage: node tools/bench.mjs [--seeds n] [--preset name] [--set dir] [--only substring] [--verbose]

import { existsSync, readdirSync, readFileSync } from 'node:fs';
import { join, dirname, basename } from 'node:path';
import { fileURLToPath } from 'node:url';
import { readPng, readImage } from './png.mjs';
import { synthGraph, SYNTH_PRESETS } from '../src/core/synth.js';
import { estimateStyle } from '../src/core/style.js';
import { buildMask } from '../src/core/layers.js';
import { fitDots, houghCenters } from '../src/core/fit.js';
import { matchDots } from '../src/core/metrics.js';
import { restoreChroma } from '../src/core/jpeg.js';
import { readTruthCsv, truthPixels } from './truth-csv.mjs';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const TRUTH_DIR = join(ROOT, 'testgraphs', 'truth');
const TRUTH_SETS = ['bench1', 'bench2', 'bench3'];
const MATCH_TOL_RADII = 0.5;

function parseArgs(argv) {
    const opts = { seeds: 5, preset: null, set: null, only: null, verbose: false };
    for (let i = 0; i < argv.length; i++) {
        if (argv[i] === '--seeds')
            opts.seeds = Number(argv[++i]);
        else if (argv[i] === '--preset')
            opts.preset = argv[++i];
        else if (argv[i] === '--set')
            opts.set = argv[++i];
        else if (argv[i] === '--only')
            opts.only = argv[++i];
        else if (argv[i] === '--verbose')
            opts.verbose = true;
    }
    return opts;
}

/**
 * Style, mask and fit for one family of an image (the first by default); also the Hough candidates alone. For a
 * JPEG, pass the image through restoreChroma first and set `jpeg` (style.js JPEG_TOLERANCES).
 */
export function extract(image, family = 0, candidates = undefined, jpeg = false) {
    const est = estimateStyle(image, { family, candidates, jpeg });
    if (!est.style)
        throw new Error(est.notes.join(' '));
    const layers = buildMask(image, est);
    const input = { image, style: est.style, line: est.familyMeasure.line, mask: layers.mask, D: est.D, imgT: est.fitT };
    const fit = fitDots(input);
    const hough = houghCenters(est.D, layers.mask.map(m => 1 - m), image.width, image.height, est.style.radius);
    return { est, layers, fit, hough };
}

/** Every dot family of an image, each fitted with the others masked. */
export function extractAll(image, jpeg = false) {
    if (jpeg)
        image = restoreChroma(image);
    const first = extract(image, 0, undefined, jpeg);
    const out = [first];
    for (let k = 1; k < first.est.candidates.length; k++)
        out.push(extract(image, k, first.est.candidates, jpeg));
    return out;
}

// truth families matched to found families by color (each found family to the nearest truth color)
function familyRows(name, truth, spec, results, ms) {
    const found = [];
    const houghs = [];
    for (const res of results) {
        const c = res.est.style.color;
        let best = 0;
        spec.families.forEach((f, k) => {
            if (Math.hypot(...f.color.map((v, ch) => v - c[ch])) < Math.hypot(...spec.families[best].color.map((v, ch) => v - c[ch])))
                best = k;
        });
        found.push(...res.fit.dots.map(d => ({ ...d, family: best })));
        houghs.push(...res.hough.map(d => ({ ...d, family: best })));
    }
    // match within each family and pool the counts
    let pairs = 0;
    let sq = 0;
    let hpairs = 0;
    const tol = 0.5 * Math.min(...results.map(r => r.est.style.radius));
    for (let k = 0; k < spec.families.length; k++) {
        const m = matchDots(truth.filter(t => t.family === k), found.filter(d => d.family === k), tol);
        pairs += m.pairs.length;
        sq += m.pairs.reduce((s, p) => s + p.d * p.d, 0);
        hpairs += matchDots(truth.filter(t => t.family === k), houghs.filter(d => d.family === k), tol).pairs.length;
    }
    return {
        name, n: truth.length, found: found.length, precision: found.length ? pairs / found.length : 1, recall: pairs / truth.length,
        countError: found.length - truth.length, rmse: Math.sqrt(sq / pairs), houghRecall: hpairs / truth.length,
        houghPrecision: houghs.length ? hpairs / houghs.length : 1, flagged: found.filter(d => d.flags.length).length, ms,
        radius: results.reduce((s, r) => s + r.fit.radius, 0) / results.length, families: results.length,
    };
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

// a found family explains a truth group if it takes at least this share of the group's dots when all are pooled
const GROUP_LINK_SHARE = 0.2;

/**
 * Score every family of an image against truth dots with groups. Truth groups have no colors, and an image may draw
 * them in one color (then one family explains both) or split one color in two (ellipse: on white and on the
 * ellipse). So families are linked to the groups they explain in a pooled match, and the dots are matched again
 * within each connected set of families and groups: a dot found in the wrong group's color counts as wrong.
 */
export function groupRow(name, truth, results, ms) {
    const found = results.flatMap((res, k) => res.fit.dots.map(d => ({ ...d, family: k })));
    const tol = MATCH_TOL_RADII * Math.min(...results.map(r => r.fit.radius));
    const groups = [...new Set(truth.map(t => t.group))];
    const pooled = matchDots(truth, found, tol);
    const taken = results.map(() => new Map());
    for (const p of pooled.pairs) {
        const g = truth[p.truth].group;
        const v = taken[found[p.found].family];
        v.set(g, (v.get(g) ?? 0) + 1);
    }
    // union-find over families (0…F−1) and groups (F…)
    const F = results.length;
    const parent = [...Array(F + groups.length).keys()];
    const root = i => (parent[i] === i ? i : (parent[i] = root(parent[i])));
    taken.forEach((v, k) => {
        for (const [g, c] of v) {
            const gi = groups.indexOf(g);
            if (c >= GROUP_LINK_SHARE * truth.filter(t => t.group === g).length)
                parent[root(k)] = root(F + gi);
        }
    });
    let pairs = 0;
    let sq = 0;
    const missed = [];
    const extra = [];
    for (const set of new Set(parent.map((p, i) => root(i)))) {
        const ti = truth.map((t, i) => i).filter(i => root(F + groups.indexOf(truth[i].group)) === set);
        const fi = found.map((f, j) => j).filter(j => root(found[j].family) === set);
        const m = matchDots(ti.map(i => truth[i]), fi.map(j => found[j]), tol);
        pairs += m.pairs.length;
        sq += m.pairs.reduce((s, p) => s + p.d * p.d, 0);
        missed.push(...m.missed.map(i => ti[i]));
        extra.push(...m.extra.map(j => fi[j]));
    }
    return {
        name, n: truth.length, found: found.length, precision: found.length ? pairs / found.length : 1, recall: pairs / truth.length,
        countError: found.length - truth.length, rmse: Math.sqrt(sq / pairs), houghRecall: NaN, houghPrecision: NaN,
        flagged: found.filter(d => d.flags.length).length, ms, radius: results.reduce((s, r) => s + r.fit.radius, 0) / results.length,
        families: results.length, missed: missed.map(i => truth[i]), extra: extra.map(j => found[j]),
    };
}

/**
 * Score lane data (beeswarm, jitter): the layout chose each dot's x, so a found dot belongs to the nearest lane
 * center and is matched by y alone within its lane. RMSE is in y only.
 */
export function laneRow(name, truth, laneCenters, results, ms) {
    const found = results.flatMap(res => res.fit.dots);
    const tol = MATCH_TOL_RADII * Math.min(...results.map(r => r.fit.radius));
    const lanes = [...new Set(truth.map(t => t.lane))];
    const laneOf = d => lanes[laneCenters.reduce((b, c, k) => (Math.abs(d.x - c) < Math.abs(d.x - laneCenters[b]) ? k : b), 0)];
    let pairs = 0;
    let sq = 0;
    const missed = [];
    const extra = [];
    for (const lane of lanes) {
        const t = truth.filter(d => d.lane === lane);
        const f = found.filter(d => laneOf(d) === lane);
        const m = matchDots(t.map(d => ({ x: 0, y: d.y })), f.map(d => ({ x: 0, y: d.y })), tol);
        pairs += m.pairs.length;
        sq += m.pairs.reduce((s, p) => s + p.d * p.d, 0);
        missed.push(...m.missed.map(i => t[i]));
        extra.push(...m.extra.map(j => f[j]));
    }
    return {
        name, n: truth.length, found: found.length, precision: found.length ? pairs / found.length : 1, recall: pairs / truth.length,
        countError: found.length - truth.length, rmse: Math.sqrt(sq / pairs), houghRecall: NaN, houghPrecision: NaN,
        flagged: found.filter(d => d.flags.length).length, ms, radius: results.reduce((s, r) => s + r.fit.radius, 0) / results.length,
        families: results.length, missed, extra,
    };
}

function print(rows) {
    const f = (v, d = 3) => (Number.isNaN(v) ? '–' : v.toFixed(d));
    console.log('case                           truth found  prec  recall  count  rmse px  hough P/R    flagged  radius   ms');
    for (const r of rows)
        console.log(`${r.name.padEnd(30)} ${String(r.n).padStart(5)} ${String(r.found).padStart(5)} ${f(r.precision)} ${f(r.recall)} ${String(r.countError).padStart(6)} ${f(r.rmse).padStart(8)}  ${f(r.houghPrecision, 2)}/${f(r.houghRecall, 2)}  ${String(r.flagged).padStart(7)} ${f(r.radius, 2).padStart(7)} ${String(r.ms).padStart(5)}`);
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
    const opts = parseArgs(process.argv.slice(2));
    const rows = [];
    for (const name of Object.keys(SYNTH_PRESETS)) {
        if (opts.preset && name !== opts.preset)
            continue;
        for (let seed = 1; seed <= opts.seeds; seed++) {
            const spec = SYNTH_PRESETS[name](seed);
            const { image, truth } = synthGraph(spec);
            const t0 = Date.now();
            if (spec.families.length > 1) {
                const results = extractAll(image);
                rows.push(familyRows(`${name} #${seed} (${results.length} fam)`, truth, spec, results, Date.now() - t0));
                continue;
            }
            const res = extract(image);
            const r = row(`${name} #${seed}`, truth, res, Date.now() - t0);
            rows.push(r);
            if (opts.verbose) {
                console.log(r.name, 'missed', r.m.missed.map(i => `${truth[i].x.toFixed(1)},${truth[i].y.toFixed(1)}`).join(' '), 'extra', r.m.extra.map(j => `${res.fit.dots[j].x.toFixed(1)},${res.fit.dots[j].y.toFixed(1)}`).join(' '));
                console.log('   log', JSON.stringify(res.fit.log));
            }
        }
    }
    for (const set of opts.set ? [opts.set] : TRUTH_SETS) {
        const dir = join(ROOT, set);
        if (opts.preset || !existsSync(join(dir, 'data.csv')))
            continue;
        const px = truthPixels(readTruthCsv(join(dir, 'data.csv')));
        if (px.error) {
            console.log(`${set}: ${px.error}`);
            continue;
        }
        for (const file of readdirSync(dir).filter(f => /\.(png|jpe?g)$/i.test(f)).sort()) {
            if (opts.only && !`${set}/${file}`.includes(opts.only))
                continue;
            const image = readImage(join(dir, file));
            const label = `${set}/${file.replace(/\.png$/, '')}`;
            const t0 = Date.now();
            let results;
            try {
                results = extractAll(image, /\.jpe?g$/i.test(file));
            }
            catch (e) {
                console.log(`${label}: ${e.message}`);
                continue;
            }
            const ms = Date.now() - t0;
            const r = px.laneCenters ? laneRow(label, px.dots, px.laneCenters, results, ms) : groupRow(label, px.dots, results, ms);
            rows.push(r);
            if (opts.verbose) {
                console.log(label, results.map(q => `${q.est.style.alpha < 1 ? 'translucent' : 'opaque'} ${q.est.style.color.map(Math.round)} r ${q.fit.radius.toFixed(2)}`).join('; '));
                console.log('   missed', r.missed.map(t => `${t.x.toFixed(1)},${t.y.toFixed(1)}`).join(' '), ' extra', r.extra.map(d => `${d.x.toFixed(1)},${d.y.toFixed(1)}`).join(' '));
            }
        }
    }
    if (existsSync(TRUTH_DIR) && !opts.preset && !opts.set) {
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

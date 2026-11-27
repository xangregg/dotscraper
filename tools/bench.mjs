// Benchmark the extraction (PLAN.md §2.6): synthetic presets over several seeds, every image with truth placed by hand
// in bench-local/truth/, bench-nature/truth/, bench-plos/truth/ or bench-pmc/truth/ (<name>.json, the app's Save truth in px, tools/truth-json.mjs; every
// family fitted, scored by truth family and fill), and every PNG and JPEG of a
// truth set (a directory with data.csv in data units, tools/truth-csv.mjs; bench1–3/), and every test image with a
// CSV of its own (bench-local/<name>.csv, bench-nature/<name>.csv, bench-plos/<name>.csv, bench-r/<name>.csv and bench-r2/<name>.csv, mapped through the detected ticks; lane data
// scored by lane, "(J)" after a name when JPEG noise turned JPEG mode on). Prints precision, recall,
// count error, position RMSE and time; tolerance r/2 for matching.
//
// Usage: node tools/bench.mjs [--seeds n] [--preset name] [--set dir] [--only substring] [--verbose] [--json file]
//                             [--jobs n] [--fewer-doubles]
//   --json  also write every row, and every image that fails, as JSON (tools/results.mjs builds RESULTS.md from it)
//   --jobs  fit n images at a time, in worker threads (default 5, or the machine's cores if fewer; 1 for one at a
//           time, in this process). The rows are the same; each image's time is longer than alone, so compare times
//           only between runs at one job count.
//   --fewer-doubles  fit with the app's "Fewer doubles" (fit.js fitDots' fewerDoubles)

import { existsSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join, dirname, basename } from 'node:path';
import { fileURLToPath } from 'node:url';
import { Worker, isMainThread, parentPort, workerData } from 'node:worker_threads';
import { availableParallelism } from 'node:os';
import { readImage } from './png.mjs';
import { synthGraph, SYNTH_PRESETS } from '../src/core/synth.js';
import { houghCenters } from '../src/core/fit.js';
import { matchDots } from '../src/core/metrics.js';
import { restoreChroma } from '../src/core/jpeg.js';
import { detectAxes } from '../src/core/axes.js';
import { extractAll } from '../src/core/extract.js';
import { laneCentersFromDots, readTruthCsv, truthPixels, truthPixelsByTicks } from './truth-csv.mjs';
import { readTruthJson, isDotsExport } from './truth-json.mjs';
import { pixelHash } from '../src/core/image.js';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const TRUTH_SETS = ['bench1', 'bench2', 'bench3'];
// directories of images with a CSV each (data units, ticks in the header), mapped through the detected ticks
const CSV_SETS = ['bench-local', 'bench-nature', 'bench-plos', 'bench-pmc', 'bench-r', 'bench-r2'];
const MATCH_TOL_RADII = 0.5;
// Images fitted at a time unless --jobs says otherwise. Measured on 12 performance cores, the full bench: 1,115 s one
// at a time, 269 s at 6, 207 s at 10, where the slowest image (about 2 minutes) begins to set the time. At 10 the
// process now passes the memory guard's 16 GB (2026-11-27); 5 stays under it and takes about 10 minutes.
const DEFAULT_JOBS = 5;
// Jittered truth (a test image's CSV with "with jitter"): the truth is each dot's cell, and a dot matches a cell
// within this box. Measured on pathogen: x up to about ±1 year (12.2 px; dots reach 0.83 year left of the first
// year), y ±25 px about rows 0–2.
const JITTER_BOX = { x: 14, y: 27 }; // px, half-widths

// the fit's options, the same in the main thread and in each worker (which gets the parsed arguments)
const FIT_OPTS = { fewerDoubles: Boolean(isMainThread ? process.argv.includes('--fewer-doubles') : workerData?.benchOpts?.fewerDoubles) };

function parseArgs(argv) {
    const opts = { seeds: 5, preset: null, set: null, only: null, verbose: false, json: null, jobs: Math.min(DEFAULT_JOBS, availableParallelism()) };
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
        else if (argv[i] === '--json')
            opts.json = argv[++i];
        else if (argv[i] === '--jobs')
            opts.jobs = Math.max(1, Math.floor(Number(argv[++i])) || 1);
        else if (argv[i] === '--fewer-doubles')
            opts.fewerDoubles = true;
    }
    return opts;
}

// the Hough candidates alone, for the Hough-only precision and recall
function houghOnly(image, est, layers) {
    return houghCenters(est.D, layers.mask.map(m => 1 - m), image.width, image.height, est.style.radius);
}

// Every image, synthetic ones too, goes through extract.js extractAll, as in the app; here each result also gets its
// Hough candidates. A family split by fill color has none of its own.
function extractWithHough(image, jpeg = false) {
    return extractAll(image, jpeg, FIT_OPTS).map(r => ({ ...r, hough: r.fill === undefined ? houghOnly(r.est.image, r.est, r.layers) : [] }));
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

// a found family explains a truth group if it takes at least this share of the group's dots when all are pooled, or
// of its own matched dots if fewer: a group can be mostly hidden (degradation1: 127 rows, about 60 visible dots in
// two colors with no colors in the data, so neither family reached 20% of the group and every match was dropped),
// while a family's few strays in another color's group stay short of either
const GROUP_LINK_SHARE = 0.2;

/**
 * Score every family of an image against truth dots with groups. Truth groups have no colors, and an image may draw
 * them in one color (then one family explains both) or split one color in two (ellipse: on white and on the
 * ellipse). So families are linked to the groups they explain in a pooled match, and the dots are matched again
 * within each connected set of families and groups: a dot found in the wrong group's color counts as wrong.
 *
 * @param {string} name  the row's
 * @param {{x: number, y: number, group?: string}[]} truth  px
 * @param {{est: object, fit: {dots: object[], radius: number}}[]} results  extract.js extractAll's, one per family
 * @param {number} ms  the extraction's time
 * @returns {object}  the row: name, n (truth dots), found, precision, recall, countError (found − n), rmse (px, over
 *   matched dots), houghPrecision and houghRecall (NaN: not scored by group), flagged, ms, radius (the families'
 *   mean, px), families, and the truth dots missed and found dots extra
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
        const mine = [...v.values()].reduce((t, c) => t + c, 0);
        for (const [g, c] of v) {
            const gi = groups.indexOf(g);
            if (c >= GROUP_LINK_SHARE * Math.min(truth.filter(t => t.group === g).length, mine))
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
 *
 * @param {string} name  the row's
 * @param {{x: number, y: number, lane: number}[]} truth  px
 * @param {number[]} laneCenters  px, each lane's x, left to right
 * @param {{fit: {dots: object[], radius: number}}[]} results  extract.js extractAll's
 * @param {number} ms  the extraction's time
 * @returns {object}  the row, as groupRow's
 */
export function laneRow(name, truth, laneCenters, results, ms) {
    const found = results.flatMap(res => res.fit.dots);
    const tol = MATCH_TOL_RADII * Math.min(...results.map(r => r.fit.radius));
    // left to right, as laneCenters are (truth rows can start in any lane: hair1's first row is lane 2)
    const lanes = [...new Set(truth.map(t => t.lane))].sort((a, b) => a - b);
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

/**
 * Score every family's dots against jittered truth cells: matched within JITTER_BOX (scaled so one tolerance fits);
 * positions can't be scored, so rmse is NaN.
 *
 * @param {string} name  the row's
 * @param {{x: number, y: number}[]} truth  px, each dot's cell
 * @param {{fit: {dots: object[], radius: number}}[]} results  extract.js extractAll's
 * @param {number} ms  the extraction's time
 * @returns {object}  the row, as groupRow's
 */
export function jitterRow(name, truth, results, ms) {
    const found = results.flatMap(r => r.fit.dots);
    const k = JITTER_BOX.x / JITTER_BOX.y;
    const m = matchDots(truth.map(t => ({ x: t.x, y: t.y * k })), found.map(d => ({ x: d.x, y: d.y * k })), JITTER_BOX.x);
    return {
        name, n: truth.length, found: found.length, precision: m.precision, recall: m.recall, countError: m.countError,
        rmse: NaN, houghRecall: NaN, houghPrecision: NaN, flagged: found.filter(d => d.flags.length).length, ms,
        radius: results.reduce((s, r) => s + r.fit.radius, 0) / results.length, families: results.length,
        missed: m.missed.map(i => truth[i]), extra: m.extra.map(j => found[j]),
    };
}

function print(rows) {
    const f = (v, d = 3) => (Number.isNaN(v) ? '–' : v.toFixed(d));
    console.log('case                           truth found  prec  recall  count  rmse px  hough P/R    flagged  radius   ms');
    for (const r of rows)
        console.log(`${r.name.padEnd(30)} ${String(r.n).padStart(5)} ${String(r.found).padStart(5)} ${f(r.precision)} ${f(r.recall)} ${String(r.countError).padStart(6)} ${f(r.rmse).padStart(8)}  ${f(r.houghPrecision, 2)}/${f(r.houghRecall, 2)}  ${String(r.flagged).padStart(7)} ${f(r.radius, 2).padStart(7)} ${String(r.ms).padStart(5)}`);
}

/**
 * The benchmark's jobs, in the order of its rows: one per synthetic graph, then one per image of each truth set, per
 * image with a CSV of its own, and per image with truth placed by hand. A job is a plain object, so that it can be
 * sent to a worker; a truth set whose data.csv can't be read is one job that only reports it.
 */
function listJobs(opts) {
    const jobs = [];
    for (const name of Object.keys(SYNTH_PRESETS)) {
        if (opts.preset && name !== opts.preset)
            continue;
        for (let seed = 1; seed <= opts.seeds; seed++)
            jobs.push({ kind: 'preset', name, seed });
    }
    for (const set of opts.set ? [opts.set] : TRUTH_SETS) {
        const dir = join(ROOT, set);
        if (opts.preset || !existsSync(join(dir, 'data.csv')))
            continue;
        const px = truthPixels(readTruthCsv(join(dir, 'data.csv')));
        if (px.error) {
            jobs.push({ kind: 'failed', meta: { set, image: 'data.csv', truth: 'data' }, message: px.error });
            continue;
        }
        for (const file of readdirSync(dir).filter(f => /\.(png|jpe?g)$/i.test(f)).sort()) {
            if (!opts.only || `${set}/${file}`.includes(opts.only))
                jobs.push({ kind: 'set', set, file });
        }
    }
    // test images with a CSV of their own (data units, ticks in the header)
    for (const set of CSV_SETS) {
        const testDir = join(ROOT, set);
        if (!existsSync(testDir) || opts.preset || (opts.set && opts.set !== set))
            continue;
        for (const file of readdirSync(testDir).filter(f => f.endsWith('.csv')).sort()) {
            const name = basename(file, '.csv');
            const img = ['.png', '.jpg', '.jpeg'].map(e => join(testDir, name + e)).find(existsSync);
            if (img && (!opts.only || `${set}/${name}`.includes(opts.only)))
                jobs.push({ kind: 'csv', set, file, img });
        }
    }
    // truth placed by hand (Save truth JSON, px): every family fitted, scored by group (each truth family, and fill)
    for (const set of CSV_SETS) {
        const truthDir = join(ROOT, set, 'truth');
        if (!existsSync(truthDir) || opts.preset || (opts.set && opts.set !== set))
            continue;
        for (const file of readdirSync(truthDir).filter(f => f.endsWith('.json')).sort()) {
            if (!opts.only || `${set}/${basename(file, '.json')}`.includes(opts.only))
                jobs.push({ kind: 'hand', set, file });
        }
    }
    return jobs;
}

/**
 * One job: the image read, its dots extracted and scored against its truth. Returns the rows it adds to the table
 * (each with `meta`: what it is, for --json), the failures, and the lines to print, so that the jobs can run in any
 * order, or in workers, and be reported in the table's order.
 */
function runJob(job, opts) {
    const rows = [];
    const failures = [];
    const lines = [];
    const out = { rows, failures, lines };
    const record = (row, meta) => {
        rows.push(row);
        row.meta = meta;
        return row;
    };
    const fail = (meta, message) => {
        lines.push(`${meta.set}/${meta.image}: ${message}`);
        failures.push({ ...meta, error: message });
        return out;
    };
    const modeOf = (results, jpeg) => (jpeg ? 'jpeg' : results[0]?.jpegAuto ? 'noisy' : 'exact');
    if (job.kind === 'failed')
        return fail(job.meta, job.message);
    if (job.kind === 'preset') {
        const { name, seed } = job;
        const spec = SYNTH_PRESETS[name](seed);
        const { image, truth } = synthGraph(spec);
        const meta = { set: 'synthetic', image: name, seed, truth: 'exact', mode: 'exact' };
        const t0 = Date.now();
        const results = extractWithHough(image);
        // by family when the graph has several, or when one was read as several: the row's name then says how many
        if (spec.families.length > 1 || results.length !== 1) {
            record(familyRows(`${name} #${seed} (${results.length} fam)`, truth, spec, results, Date.now() - t0), meta);
            return out;
        }
        const [res] = results;
        const r = record(row(`${name} #${seed}`, truth, res, Date.now() - t0), meta);
        if (opts.verbose) {
            lines.push(`${r.name} missed ${r.m.missed.map(i => `${truth[i].x.toFixed(1)},${truth[i].y.toFixed(1)}`).join(' ')} extra ${r.m.extra.map(j => `${res.fit.dots[j].x.toFixed(1)},${res.fit.dots[j].y.toFixed(1)}`).join(' ')}`);
            lines.push(`   log ${JSON.stringify(res.fit.log)}`);
        }
        return out;
    }
    if (job.kind === 'set') {
        const { set, file } = job;
        const dir = join(ROOT, set);
        const px = truthPixels(readTruthCsv(join(dir, 'data.csv')));
        const jpeg = /\.jpe?g$/i.test(file);
        const image = readImage(join(dir, file));
        const label = `${set}/${file.replace(/\.png$/, '')}`;
        const meta = { set, image: file, truth: 'exact' };
        const t0 = Date.now();
        let results;
        try {
            results = extractWithHough(image, jpeg);
        }
        catch (e) {
            return fail(meta, e.message);
        }
        const ms = Date.now() - t0;
        const r = record(px.laneCenters ? laneRow(label, px.dots, px.laneCenters, results, ms) : groupRow(label, px.dots, results, ms),
            { ...meta, mode: modeOf(results, jpeg) });
        if (opts.verbose) {
            lines.push(`${label} ${results.map(q => `${q.est.style.alpha < 1 ? 'translucent' : 'opaque'} ${q.est.style.color.map(Math.round)} r ${q.fit.radius.toFixed(2)}`).join('; ')}`);
            lines.push(`   missed ${r.missed.map(t => `${t.x.toFixed(1)},${t.y.toFixed(1)}`).join(' ')}  extra ${r.extra.map(d => `${d.x.toFixed(1)},${d.y.toFixed(1)}`).join(' ')}`);
        }
        return out;
    }
    if (job.kind === 'csv') {
        const { set, file, img } = job;
        const testDir = join(ROOT, set);
        const name = basename(file, '.csv');
        // the app's dot export isn't data-unit truth: hand truth belongs in truth/<name>.json (tools/truth-json.mjs)
        if (isDotsExport(readFileSync(join(testDir, file), 'utf8'))) {
            lines.push(`${set}/${name}: a dots export, not truth in data units; skipped (truth/${name}.json is scored)`);
            return out;
        }
        const jpeg = /\.jpe?g$/i.test(img);
        const meta = { set, image: basename(img), truth: 'data' };
        const image = readImage(img);
        const t0 = Date.now();
        const px = truthPixelsByTicks(readTruthCsv(join(testDir, file)), detectAxes(jpeg ? restoreChroma(image) : image));
        if (px.error)
            return fail(meta, px.error);
        let results;
        try {
            results = extractWithHough(image, jpeg);
        }
        catch (e) {
            return fail(meta, e.message);
        }
        const mark = results[0]?.jpegAuto ? ' (J)' : '';
        const label = `${set}/${name}`.slice(0, 30 - mark.length) + mark;
        const ms = Date.now() - t0;
        if (px.lanesFromDots) {
            px.laneCenters = laneCentersFromDots(results.flatMap(r => r.fit.dots.map(d => d.x)), new Set(px.dots.map(d => d.lane)).size);
            if (!px.laneCenters)
                return fail(meta, 'fewer dots found than lanes');
        }
        const r = record(px.laneCenters ? laneRow(label, px.dots, px.laneCenters, results, ms)
            : px.jitter ? jitterRow(label, px.dots, results, ms) : groupRow(label, px.dots, results, ms), { ...meta, mode: modeOf(results, jpeg) });
        if (opts.verbose && r.missed) {
            lines.push(`${label} ${results.map(q => `${q.est.style.alpha < 1 ? 'translucent' : 'opaque'} ${q.est.style.color.map(Math.round)} r ${q.fit.radius.toFixed(2)}`).join('; ')}`);
            lines.push(`   missed ${r.missed.map(t => `${t.x.toFixed(1)},${t.y.toFixed(1)}`).join(' ')}  extra ${r.extra.map(d => `${d.x.toFixed(1)},${d.y.toFixed(1)}`).join(' ')}`);
        }
        return out;
    }
    const { set, file } = job;
    const name = basename(file, '.json');
    const truth = readTruthJson(join(ROOT, set, 'truth', file));
    const img = join(ROOT, set, truth.image ?? name + '.png');
    const meta = { set, image: basename(img), truth: 'hand' };
    if (!existsSync(img))
        return fail(meta, `image ${truth.image} absent`);
    const jpeg = /\.jpe?g$/i.test(img);
    const image = readImage(img);
    // JPEG decoders differ, so only a PNG's hash can be checked
    if (!jpeg && truth.hash && pixelHash(image) !== truth.hash)
        return fail(meta, `the image's pixel hash ${pixelHash(image)} isn't the truth's ${truth.hash}; skipped`);
    const t0 = Date.now();
    let results;
    try {
        results = extractWithHough(image, jpeg);
    }
    catch (e) {
        return fail(meta, e.message);
    }
    const mark = results[0]?.jpegAuto ? ' (J)' : '';
    const label = `${set}/${name}`.slice(0, 30 - mark.length) + mark;
    const r = record(groupRow(label, truth.dots, results, Date.now() - t0), { ...meta, mode: modeOf(results, jpeg) });
    if (opts.verbose) {
        lines.push(`${label} ${results.map(q => `${q.est.style.alpha < 1 ? 'translucent' : 'opaque'} ${q.est.style.color.map(Math.round)} r ${q.fit.radius.toFixed(2)}`).join('; ')}`);
        lines.push(`   missed ${r.missed.map(t => `${t.x.toFixed(1)},${t.y.toFixed(1)}`).join(' ')}  extra ${r.extra.map(d => `${d.x.toFixed(1)},${d.y.toFixed(1)}`).join(' ')}`);
    }
    return out;
}

/**
 * Run the jobs, `opts.jobs` at a time, and hand each one's result to `report` in the jobs' order. One at a time they
 * run here; more, in that many worker threads (this file again, in its worker role below), each taking the next job
 * as it finishes one. The images are independent, so the rows are the same either way; the times are not: a fit
 * shares the machine's memory and caches with the others, so compare times only between runs at one job count.
 */
function runJobs(jobs, opts, report) {
    if (opts.jobs <= 1 || jobs.length <= 1) {
        for (const job of jobs)
            report(runJob(job, opts));
        return Promise.resolve();
    }
    return new Promise((resolve, reject) => {
        const results = new Array(jobs.length);
        const workers = [];
        let next = 0;
        let reported = 0;
        const stop = err => {
            for (const w of workers)
                w.terminate();
            if (err)
                reject(err);
            else
                resolve();
        };
        for (let k = 0; k < Math.min(opts.jobs, jobs.length); k++) {
            const worker = new Worker(new URL(import.meta.url), { workerData: { benchOpts: opts } });
            let current = -1;
            const take = () => {
                current = next++;
                worker.postMessage(jobs[current]);
            };
            worker.on('message', ({ result, error }) => {
                if (error) {
                    stop(new Error(error));
                    return;
                }
                results[current] = result;
                while (reported < jobs.length && results[reported])
                    report(results[reported++]);
                if (reported === jobs.length)
                    stop();
                else if (next < jobs.length)
                    take();
            });
            worker.on('error', stop);
            workers.push(worker);
            take();
        }
    });
}

async function main() {
    const opts = parseArgs(process.argv.slice(2));
    const rows = [];
    const failures = [];
    await runJobs(listJobs(opts), opts, result => {
        for (const line of result.lines)
            console.log(line);
        rows.push(...result.rows);
        failures.push(...result.failures);
    });
    print(rows);
    if (opts.json) {
        const keep = ({ name, n, found, precision, recall, countError, rmse, flagged, radius, ms, families, meta }) =>
            ({ ...meta, name, n, found, precision, recall, countError, rmse, flagged, radius, ms, families });
        writeFileSync(opts.json, JSON.stringify({ seeds: opts.seeds, jobs: opts.jobs, rows: rows.map(keep), failures }, null, 1) + '\n');
    }
}

if (!isMainThread && workerData?.benchOpts) {
    // a worker of runJobs: one job at a time, as they are sent
    parentPort.on('message', job => {
        try {
            parentPort.postMessage({ result: runJob(job, workerData.benchOpts) });
        }
        catch (e) {
            parentPort.postMessage({ error: `${job.set ?? job.name}/${job.file ?? job.seed}: ${e.stack}` });
        }
    });
}
else if (isMainThread && process.argv[1] === fileURLToPath(import.meta.url)) {
    main();
}

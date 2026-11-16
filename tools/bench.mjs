// Benchmark the extraction (PLAN.md §2.6): synthetic presets over several seeds, every image with truth placed by hand
// in testgraphs/truth/, bench-nature/truth/ or bench-plos/truth/ (<name>.json, the app's Save truth in px, tools/truth-json.mjs; every
// family fitted, scored by truth family and fill), and every PNG and JPEG of a
// truth set (a directory with data.csv in data units, tools/truth-csv.mjs; bench1–3/), and every test image with a
// CSV of its own (testgraphs/<name>.csv, bench-nature/<name>.csv and bench-plos/<name>.csv, mapped through the detected ticks; lane data
// scored by lane, "(J)" after a name when JPEG noise turned JPEG mode on). Prints precision, recall,
// count error, position RMSE and time; tolerance r/2 for matching.
//
// Usage: node tools/bench.mjs [--seeds n] [--preset name] [--set dir] [--only substring] [--verbose] [--json file]
//   --json  also write every row, and every image that fails, as JSON (tools/results.mjs builds RESULTS.md from it)

import { existsSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join, dirname, basename } from 'node:path';
import { fileURLToPath } from 'node:url';
import { readImage } from './png.mjs';
import { synthGraph, SYNTH_PRESETS } from '../src/core/synth.js';
import { estimateStyle } from '../src/core/style.js';
import { estimateFamilies } from '../src/core/families.js';
import { buildMask } from '../src/core/layers.js';
import { fitDots, houghCenters } from '../src/core/fit.js';
import { matchDots } from '../src/core/metrics.js';
import { restoreChroma, jpegNoise } from '../src/core/jpeg.js';
import { dotFamilies } from '../src/core/outline.js';
import { detectAxes, plotBounds } from '../src/core/axes.js';
import { laneCentersFromDots, readTruthCsv, truthPixels, truthPixelsByTicks } from './truth-csv.mjs';
import { readTruthJson, isDotsExport } from './truth-json.mjs';
import { pixelHash } from '../src/core/image.js';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const TRUTH_SETS = ['bench1', 'bench2', 'bench3'];
// directories of images with a CSV each (data units, ticks in the header), mapped through the detected ticks
const CSV_SETS = ['testgraphs', 'bench-nature', 'bench-plos'];
const MATCH_TOL_RADII = 0.5;
// Jittered truth (a test image's CSV with "with jitter"): the truth is each dot's cell, and a dot matches a cell
// within this box. Measured on pathogen: x up to about ±1 year (12.2 px; dots reach 0.83 year left of the first
// year), y ±25 px about rows 0–2.
const JITTER_BOX = { x: 14, y: 27 }; // px, half-widths

function parseArgs(argv) {
    const opts = { seeds: 5, preset: null, set: null, only: null, verbose: false, json: null };
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
    }
    return opts;
}

/**
 * Style, mask and fit for one family of an image (the first by default); also the Hough candidates alone. For a
 * JPEG, pass the image through restoreChroma first and set `jpeg` (style.js JPEG_TOLERANCES). `bounds` masks what
 * lies beyond the axes (axes.plotBounds).
 */
export function extract(image, family = 0, candidates = undefined, jpeg = false, bounds = undefined) {
    const est = estimateStyle(image, { family, candidates, jpeg });
    if (!est.style)
        throw new Error(est.notes.join(' '));
    return fitEstimate(image, est, bounds);
}

// mask and fit for one family's estimate
function fitEstimate(image, est, bounds) {
    const layers = buildMask(image, est, { bounds });
    const input = { image, style: est.style, line: est.familyMeasure.line, mask: layers.mask, D: est.D, imgT: est.fitT, base: est.base };
    const fit = fitDots(input);
    const hough = houghCenters(est.D, layers.mask.map(m => 1 - m), image.width, image.height, est.style.radius);
    return { est, layers, fit, hough };
}

/**
 * Every dot family of an image, as families.js prepares them (overlays divided out, translucent families on their
 * own images), each fitted with what lies beyond the axes masked.
 */
export function extractAll(image, jpeg = false) {
    const source = image;
    if (jpeg)
        image = restoreChroma(image);
    let { estimates } = estimateFamilies(image, { jpeg });
    // as the app does: a PNG with JPEG noise is analyzed as a JPEG when that finds more families with a radius (all
    // of them, when none shows with exact colors)
    let jpegAuto = false;
    if (!jpeg && jpegNoise(source).likely) {
        const restored = restoreChroma(source);
        const noisy = estimateFamilies(restored, { jpeg: true }).estimates;
        const found = es => es.filter(e => e.style?.radius).length;
        if (found(noisy) > found(estimates)) {
            jpegAuto = true;
            image = restored;
            estimates = noisy;
        }
    }
    const missing = estimates.find(e => !e.style);
    if (missing)
        throw new Error(missing.notes.join(' '));
    const bounds = plotBounds(detectAxes(image));
    return estimates.flatMap(e => splitByFill(image, fitEstimate(e.image, e, bounds))).map(r => ({ ...r, jpegAuto }));
}

// Outlined dots of a few fill colors are fitted as one family of rings; each dot's family comes from its fill
// (outline.dotFamilies), and each family becomes a result of its own, so the group scoring checks the colors.
function splitByFill(image, res) {
    const fams = res.est.fillFamilies;
    if (!fams)
        return [res];
    const st = res.est.style;
    const which = dotFamilies(image, res.fit.dots, fams, res.fit.radius, res.fit.radius * (1 + st.ring.inner) / 2);
    return fams.map((color, k) => ({
        ...res,
        est: { ...res.est, style: { ...st, color } },
        fit: { ...res.fit, dots: res.fit.dots.filter((_, i) => which[i].family === k).map(d => ({ ...d, familySure: which[res.fit.dots.indexOf(d)].sure })) },
        hough: [],
    }));
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

if (process.argv[1] === fileURLToPath(import.meta.url)) {
    const opts = parseArgs(process.argv.slice(2));
    const rows = [];
    // for --json: what each row is, and what failed
    const record = (row, meta) => {
        rows.push(row);
        row.meta = meta;
        return row;
    };
    const failures = [];
    const fail = (meta, message) => {
        console.log(`${meta.set}/${meta.image}: ${message}`);
        failures.push({ ...meta, error: message });
    };
    const modeOf = (results, jpeg) => (jpeg ? 'jpeg' : results[0]?.jpegAuto ? 'noisy' : 'exact');
    for (const name of Object.keys(SYNTH_PRESETS)) {
        if (opts.preset && name !== opts.preset)
            continue;
        for (let seed = 1; seed <= opts.seeds; seed++) {
            const spec = SYNTH_PRESETS[name](seed);
            const { image, truth } = synthGraph(spec);
            const t0 = Date.now();
            if (spec.families.length > 1) {
                const results = extractAll(image);
                record(familyRows(`${name} #${seed} (${results.length} fam)`, truth, spec, results, Date.now() - t0), { set: 'synthetic', image: name, seed, truth: 'exact', mode: 'exact' });
                continue;
            }
            const res = extract(image, 0, undefined, false, plotBounds(detectAxes(image)));
            const r = record(row(`${name} #${seed}`, truth, res, Date.now() - t0), { set: 'synthetic', image: name, seed, truth: 'exact', mode: 'exact' });
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
            fail({ set, image: 'data.csv', truth: 'data' }, px.error);
            continue;
        }
        for (const file of readdirSync(dir).filter(f => /\.(png|jpe?g)$/i.test(f)).sort()) {
            if (opts.only && !`${set}/${file}`.includes(opts.only))
                continue;
            const image = readImage(join(dir, file));
            const label = `${set}/${file.replace(/\.png$/, '')}`;
            const meta = { set, image: file, truth: 'exact' };
            const t0 = Date.now();
            let results;
            try {
                results = extractAll(image, /\.jpe?g$/i.test(file));
            }
            catch (e) {
                fail(meta, e.message);
                continue;
            }
            const ms = Date.now() - t0;
            const r = record(px.laneCenters ? laneRow(label, px.dots, px.laneCenters, results, ms) : groupRow(label, px.dots, results, ms),
                { ...meta, mode: modeOf(results, /\.jpe?g$/i.test(file)) });
            if (opts.verbose) {
                console.log(label, results.map(q => `${q.est.style.alpha < 1 ? 'translucent' : 'opaque'} ${q.est.style.color.map(Math.round)} r ${q.fit.radius.toFixed(2)}`).join('; '));
                console.log('   missed', r.missed.map(t => `${t.x.toFixed(1)},${t.y.toFixed(1)}`).join(' '), ' extra', r.extra.map(d => `${d.x.toFixed(1)},${d.y.toFixed(1)}`).join(' '));
            }
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
            if (!img || (opts.only && !`${set}/${name}`.includes(opts.only)))
                continue;
            // the app's dot export isn't data-unit truth: hand truth belongs in truth/<name>.json (tools/truth-json.mjs)
            if (isDotsExport(readFileSync(join(testDir, file), 'utf8'))) {
                console.log(`${set}/${name}: a dots export, not truth in data units; skipped (truth/${name}.json is scored)`);
                continue;
            }
            const jpeg = /\.jpe?g$/i.test(img);
            const meta = { set, image: basename(img), truth: 'data' };
            const image = readImage(img);
            const t0 = Date.now();
            const px = truthPixelsByTicks(readTruthCsv(join(testDir, file)), detectAxes(jpeg ? restoreChroma(image) : image));
            if (px.error) {
                fail(meta, px.error);
                continue;
            }
            let results;
            try {
                results = extractAll(image, jpeg);
            }
            catch (e) {
                fail(meta, e.message);
                continue;
            }
            const mark = results[0]?.jpegAuto ? ' (J)' : '';
            const label = `${set}/${name}`.slice(0, 30 - mark.length) + mark;
            const ms = Date.now() - t0;
            if (px.lanesFromDots) {
                px.laneCenters = laneCentersFromDots(results.flatMap(r => r.fit.dots.map(d => d.x)), new Set(px.dots.map(d => d.lane)).size);
                if (!px.laneCenters) {
                    fail(meta, 'fewer dots found than lanes');
                    continue;
                }
            }
            record(px.laneCenters ? laneRow(label, px.dots, px.laneCenters, results, ms)
                : px.jitter ? jitterRow(label, px.dots, results, ms) : groupRow(label, px.dots, results, ms), { ...meta, mode: modeOf(results, jpeg) });
        }
    }
    // truth placed by hand (Save truth JSON, px): every family fitted, scored by group (each truth family, and fill)
    for (const set of CSV_SETS) {
        const truthDir = join(ROOT, set, 'truth');
        if (!existsSync(truthDir) || opts.preset || (opts.set && opts.set !== set))
            continue;
        for (const file of readdirSync(truthDir).filter(f => f.endsWith('.json')).sort()) {
            const name = basename(file, '.json');
            if (opts.only && !`${set}/${name}`.includes(opts.only))
                continue;
            const truth = readTruthJson(join(truthDir, file));
            const img = join(ROOT, set, truth.image ?? name + '.png');
            const meta = { set, image: basename(img), truth: 'hand' };
            if (!existsSync(img)) {
                fail(meta, `image ${truth.image} absent`);
                continue;
            }
            const jpeg = /\.jpe?g$/i.test(img);
            const image = readImage(img);
            // JPEG decoders differ, so only a PNG's hash can be checked
            if (!jpeg && truth.hash && pixelHash(image) !== truth.hash) {
                fail(meta, `the image's pixel hash ${pixelHash(image)} isn't the truth's ${truth.hash}; skipped`);
                continue;
            }
            const t0 = Date.now();
            let results;
            try {
                results = extractAll(image, jpeg);
            }
            catch (e) {
                fail(meta, e.message);
                continue;
            }
            const mark = results[0]?.jpegAuto ? ' (J)' : '';
            const label = `${set}/${name}`.slice(0, 30 - mark.length) + mark;
            record(groupRow(label, truth.dots, results, Date.now() - t0), { ...meta, mode: modeOf(results, jpeg) });
        }
    }
    print(rows);
    if (opts.json) {
        const keep = ({ name, n, found, precision, recall, countError, rmse, flagged, radius, ms, families, meta }) =>
            ({ ...meta, name, n, found, precision, recall, countError, rmse, flagged, radius, ms, families });
        writeFileSync(opts.json, JSON.stringify({ seeds: opts.seeds, rows: rows.map(keep), failures }, null, 1) + '\n');
    }
}

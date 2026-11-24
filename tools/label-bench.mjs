// Benchmark the reading of tick labels (src/core/labels.js) against the tick values the truth files list: bench1–3's
// data.csv for every rendering, each real figure's own CSV, and bench-r's (R's renderings in scientific notation). Per set: the labels read right one at a time, and
// the axes, which is what a user sees:
//   solved     a scale found, and every labeled tick's value right
//   wrong      a scale found that gives a wrong value: the failure that matters, a wrong number offered as checked
//   unchecked  no scale, but every label read as a number (two labels, or values on no linear or log scale); the
//              count in parentheses is those with every value right
//   none       the rest: left to the user, as before
//
// An axis in percent ("75%") whose header is in fractions (0.75) is solved when every value is 100 × the header's:
// the reading is the axis's own unit. Category axes are in the headers as 1, 2, 3, ...: their names don't read as
// numbers (none), or do and aren't those (hair3's doses 0, 5, 20, ...: unchecked, and not "right").
//
// An axis whose header values aren't numbers is skipped; so is one whose ticks can't be paired with them.
//
// Usage: node tools/label-bench.mjs [--set dir] [--verbose]

import { existsSync, readdirSync, readFileSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { readImage } from './png.mjs';
import { detectAxes } from '../src/core/axes.js';
import { readTickLabels } from '../src/core/labels.js';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const TRUTH_SETS = ['bench1', 'bench2', 'bench3'];
const CSV_SETS = ['bench-local', 'bench-nature', 'bench-plos', 'bench-r'];
const VALUE_TOL = 1e-9; // relative: a value is right when it is the header's

// the header's tick values as written, or null when one isn't a number
function headerValues(csv, axis) {
    const line = readFileSync(csv, 'utf8').replace(/^\uFEFF/, '').split(/\r?\n/).find(l => l.toLowerCase().startsWith(`# ${axis} ticks:`));
    if (!line)
        return null;
    const values = line.slice(line.indexOf(':') + 1).split(',').map(s => Number(s.trim().replace(/−/g, '-')));
    return values.every(Number.isFinite) ? values : null;
}

const same = (a, b) => Math.abs(a - b) <= VALUE_TOL * Math.max(1, Math.abs(b));

/**
 * One image's axes against the header's values.
 *
 * @returns {{axis: string, labels: number, right: number, outcome: 'solved'|'wrong'|'unchecked'|'none', allRight: boolean, detail: string}[]}
 */
function score(imagePath, csv) {
    const image = readImage(imagePath);
    const axes = detectAxes(image);
    // (as the app reads them: a y axis's labels turned, when upright they fit no scale)
    const readings = readTickLabels(image, axes);
    const out = [];
    for (const k of ['x', 'y']) {
        const values = headerValues(csv, k);
        const ticks = axes[k]?.ticks ?? [];
        const labeled = ticks.filter(t => t.label);
        if (!values || !axes[k] || !labeled.length)
            continue;
        // the header lists every tick, or the labeled ones
        const listed = ticks.length === values.length ? ticks : labeled.length === values.length ? labeled : null;
        if (!listed)
            continue;
        const sorted = [...values].sort((a, b) => a - b);
        // x ticks run left to right, y ticks bottom to top
        const truth = new Map((k === 'x' ? listed : [...listed].reverse()).map((t, i) => [t, sorted[i]]));
        const { read, scale } = readings[k];
        const at = ticks.map((t, i) => ({ t, i })).filter(({ t }) => t.label && truth.has(t));
        // (percent against fractions: see the header)
        const unit = scale && at.every(({ t, i }) => same(scale.values[i], 100 * truth.get(t))) ? 100 : 1;
        const isRight = (v, t) => v !== null && v !== undefined && same(v, unit * truth.get(t));
        const right = at.filter(({ t, i }) => isRight(read[i]?.value, t)).length;
        const allRead = at.every(({ i }) => read[i] && read[i].value !== null);
        let outcome = allRead ? 'unchecked' : 'none';
        if (scale)
            outcome = at.every(({ t, i }) => isRight(scale.values[i], t)) ? 'solved' : 'wrong';
        const detail = at.map(({ t, i }) => `${truth.get(t)}→${read[i]?.text ?? '∅'}${scale && !isRight(scale.values[i], t) ? `≠${Number(scale.values[i].toPrecision(6))}` : ''}`).join(' ');
        out.push({
            axis: k, labels: at.length, right, outcome, allRight: right === at.length,
            detail: `${scale ? scale.type + ` ${scale.count} agree` : 'no scale'}${unit === 100 ? ', percent' : ''}: ${detail}`,
        });
    }
    return out;
}

const args = process.argv.slice(2);
const only = args.includes('--set') ? args[args.indexOf('--set') + 1] : null;
const verbose = args.includes('--verbose');
const jobs = [];
for (const set of TRUTH_SETS) {
    const dir = join(ROOT, set);
    if (existsSync(join(dir, 'data.csv'))) {
        for (const f of readdirSync(dir).filter(f => /\.(png|jpe?g)$/i.test(f)).sort())
            jobs.push({ set, name: f, image: join(dir, f), csv: join(dir, 'data.csv') });
    }
}
for (const set of CSV_SETS) {
    const dir = join(ROOT, set);
    if (!existsSync(dir))
        continue;
    for (const f of readdirSync(dir).filter(f => f.endsWith('.csv')).sort()) {
        const image = ['.png', '.jpg', '.jpeg'].map(e => join(dir, f.replace(/\.csv$/, e))).find(existsSync);
        if (image)
            jobs.push({ set, name: f.replace(/\.csv$/, '').replace(/-10\.1[03].*$/, ''), image, csv: join(dir, f) });
    }
}
const totals = new Map();
for (const job of jobs.filter(j => !only || j.set === only)) {
    const t = totals.get(job.set) ?? { images: 0, axes: 0, labels: 0, right: 0, solved: 0, wrong: 0, unchecked: 0, uncheckedRight: 0, none: 0 };
    totals.set(job.set, t);
    t.images++;
    for (const a of score(job.image, job.csv)) {
        t.axes++;
        t.labels += a.labels;
        t.right += a.right;
        t[a.outcome]++;
        if (a.outcome === 'unchecked' && a.allRight)
            t.uncheckedRight++;
        if (verbose || a.outcome !== 'solved')
            console.log(`${job.set}/${job.name} ${a.axis}: ${a.outcome.toUpperCase()} (${a.right} of ${a.labels} read) ${a.detail}`);
    }
}
console.log('\nset            images  axes  labels read right   axes solved  wrong  unchecked (right)  none');
const sum = { images: 0, axes: 0, labels: 0, right: 0, solved: 0, wrong: 0, unchecked: 0, uncheckedRight: 0, none: 0 };
for (const [set, t] of [...totals, ['all', sum]]) {
    if (set !== 'all') {
        for (const key of Object.keys(sum))
            sum[key] += t[key];
    }
    const pct = (a, b) => (b ? `${(100 * a / b).toFixed(1)}%` : '–');
    const cell = (v, width) => String(v).padStart(width);
    console.log([
        set.padEnd(14), cell(t.images, 6), cell(t.axes, 5), ` ${cell(t.right, 5)} of ${String(t.labels).padEnd(5)}`, cell(pct(t.right, t.labels), 6),
        ` ${cell(t.solved, 5)}`, cell(pct(t.solved, t.axes), 6), cell(t.wrong, 6), ` ${cell(t.unchecked, 9)}`, `(${t.uncheckedRight})`.padEnd(7), cell(t.none, 5),
    ].join(' '));
}

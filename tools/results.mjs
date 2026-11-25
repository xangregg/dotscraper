// RESULTS.md: the benchmark's result for every test and bench image, one table per set. Runs tools/bench.mjs (every
// set it finds: the synthetic presets, bench1–3, bench-plos, bench-r, and the local-only bench-local and bench-nature images) with --json, and
// writes the tables. The numbers are regenerated each time; the Notes column is hand-written and kept, matched by
// set, image and kind of truth, so notes can be edited in RESULTS.md directly.
//
// Usage: node tools/results.mjs [--seeds n] [--jobs n]   (npm run results; the bench fits 10 images at a time unless
//        --jobs says otherwise, and the header says how many, since the times are longer than an image's alone)
//        node tools/results.mjs --from-json bench.json [--out file]   (tables from a saved bench --json run)
// Run it on the full bench: rows (and their notes) of images a run leaves out are dropped.

import { spawnSync, execSync } from 'node:child_process';
import { existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
let OUT = join(ROOT, 'RESULTS.md');
const SET_ORDER = ['synthetic', 'bench1', 'bench2', 'bench3', 'bench-local', 'bench-nature', 'bench-plos', 'bench-pmc', 'bench-r'];
const SET_TITLES = {
    synthetic: 'Synthetic presets',
    bench1: 'bench1: JMP renderings, 100 dots',
    bench2: 'bench2: JMP renderings, 204 dots with more ties',
    bench3: 'bench3: JMP renderings, 204 dots as beeswarms',
    'bench-local': 'bench-local: real figures (local only)',
    'bench-nature': 'bench-nature: Nature figures (local only)',
    'bench-plos': 'bench-plos: PLOS ONE figures (CC BY)',
    'bench-pmc': 'bench-pmc: figures from articles in PubMed Central (local only)',
    'bench-r': 'bench-r: R renderings, 40 dots (scientific notation, ggplot2\'s gray theme)',
};
const TRUTH = { exact: 'exact', data: 'data', hand: 'by hand' };
const MODE = { exact: 'exact', noisy: 'noisy (auto)', jpeg: 'JPEG' };

// short reasons for the failures the bench reports
function reason(error) {
    if (/No dot family found/i.test(error))
        return 'no family';
    if (/radius is unknown/i.test(error))
        return 'radius unknown';
    if (/ticks detected/i.test(error))
        return 'ticks unmatched';
    if (/pixel hash/i.test(error))
        return 'hash differs';
    if (/absent/i.test(error))
        return 'image absent';
    return 'error';
}

// the image's name as shown: no .png (a JPEG keeps .jpg, beside its PNG in bench1–2), no DOI suffix (bench-nature,
// bench-plos, bench-pmc)
const shortName = image => image.replace(/\.png$/i, '').replace(/-10\.\d{4,5}[-_:].*?(\.jpe?g)?$/, '$1');

const key = (set, image, truth) => `${set}|${image}|${truth}`;

// hand-written notes from the current RESULTS.md: the last cell of each table row, by set (its heading), image and truth
function readNotes() {
    const notes = new Map();
    if (!existsSync(OUT))
        return notes;
    let set = null;
    for (const line of readFileSync(OUT, 'utf8').split('\n')) {
        const h = line.match(/^## (\S+)/);
        if (h) {
            set = h[1].replace(/:$/, '');
            continue;
        }
        if (!set || !line.startsWith('|') || /^\|\s*-/.test(line))
            continue;
        const cells = line.split('|').slice(1, -1).map(c => c.trim());
        const truth = set === 'synthetic' ? TRUTH.exact : cells[1];
        if (cells[0] && cells[0] !== 'Image' && cells[0] !== 'Preset' && cells.at(-1))
            notes.set(key(set, cells[0], truth), cells.at(-1));
    }
    return notes;
}

const f = (v, d) => (v === undefined || v === null || Number.isNaN(v) ? '–' : v.toFixed(d));
const mean = xs => xs.reduce((t, x) => t + x, 0) / xs.length;

function syntheticTable(rows, notes) {
    const out = ['| Preset | Seeds | Dots | Precision mean (min) | Recall mean (min) | RMSE px | Flagged | Time s | Notes |',
        '|---|---|---|---|---|---|---|---|---|'];
    const byPreset = new Map();
    for (const r of rows)
        byPreset.set(r.image, [...(byPreset.get(r.image) ?? []), r]);
    for (const [preset, rs] of byPreset) {
        const p = rs.map(r => r.precision);
        const rc = rs.map(r => r.recall);
        const note = notes.get(key('synthetic', preset, TRUTH.exact)) ?? '';
        out.push(`| ${preset} | ${rs.length} | ${rs[0].n} | ${f(mean(p), 3)} (${f(Math.min(...p), 3)}) | ${f(mean(rc), 3)} (${f(Math.min(...rc), 3)}) | `
            + `${f(mean(rs.map(r => r.rmse).filter(Number.isFinite)), 3)} | ${f(mean(rs.map(r => r.flagged)), 1)} | ${f(mean(rs.map(r => r.ms)) / 1000, 1)} | ${note} |`);
    }
    return out;
}

function imageTable(set, rows, failures, notes) {
    const out = ['| Image | Truth | Mode | Dots | Found | Precision | Recall | Count | RMSE px | Flagged | r px | Time s | Notes |',
        '|---|---|---|---|---|---|---|---|---|---|---|---|---|'];
    const entries = [
        ...rows.map(r => ({ name: shortName(r.image), truth: TRUTH[r.truth], r })),
        ...failures.map(e => ({ name: shortName(e.image), truth: TRUTH[e.truth], e })),
    ].sort((a, b) => a.name.localeCompare(b.name) || a.truth.localeCompare(b.truth));
    for (const { name, truth, r, e } of entries) {
        const note = notes.get(key(set, name, truth)) ?? '';
        if (r) {
            out.push(`| ${name} | ${truth} | ${MODE[r.mode] ?? r.mode} | ${r.n} | ${r.found} | ${f(r.precision, 3)} | ${f(r.recall, 3)} | ${r.countError > 0 ? '+' : ''}${r.countError} | `
                + `${f(r.rmse, 2)} | ${r.flagged} | ${f(r.radius, 1)} | ${f(r.ms / 1000, 1)} | ${note} |`);
        }
        else {
            out.push(`| ${name} | ${truth} | | | **${reason(e.error)}** | | | | | | | | ${note} |`);
        }
    }
    return out;
}

// the bench's rows: from a saved --json run, or a fresh run of tools/bench.mjs with the other arguments
function benchRows(args) {
    const from = args.indexOf('--from-json');
    if (from >= 0)
        return JSON.parse(readFileSync(args[from + 1], 'utf8'));
    const dir = mkdtempSync(join(tmpdir(), 'dotscraper-results-'));
    const json = join(dir, 'bench.json');
    const run = spawnSync(process.execPath, [join(ROOT, 'tools/bench.mjs'), ...args, '--json', json], { cwd: ROOT, stdio: ['ignore', 'inherit', 'inherit'] });
    if (run.status !== 0 || !existsSync(json))
        throw new Error('the bench failed');
    const result = JSON.parse(readFileSync(json, 'utf8'));
    rmSync(dir, { recursive: true, force: true });
    return result;
}

function main() {
    const args = process.argv.slice(2);
    const out = args.indexOf('--out');
    if (out >= 0) {
        OUT = args[out + 1];
        args.splice(out, 2);
    }
    const { seeds, jobs = 1, rows, failures } = benchRows(args);
    const notes = readNotes();
    let commit = '';
    try {
        commit = execSync('git rev-parse --short HEAD', { cwd: ROOT }).toString().trim();
        if (execSync('git status --porcelain -- src tools', { cwd: ROOT }).toString().trim())
            commit += ' plus uncommitted changes';
    }
    catch {
        commit = 'unknown';
    }
    const lines = [
        '# Results',
        '',
        `Generated by \`npm run results\` on ${new Date().toISOString().slice(0, 10)} at commit ${commit}, ${seeds} seed${seeds === 1 ? '' : 's'} per synthetic preset.`,
        // (a time is an image's own only when it was fitted alone)
        jobs > 1 ? `The images were fitted ${jobs} at a time (\`--jobs ${jobs}\`): each **Time** is longer than the image takes alone.` : 'The images were fitted one at a time: each **Time** is the image\'s alone.',
        'The numbers are rewritten each run; edit only the Notes column, which is kept (matched by set, image and truth).',
        '',
        'Columns: **Truth** is *exact* (rendered from known data), *data* (published values placed through the detected',
        'ticks) or *by hand* (Save truth JSON). **Mode** is how colors were matched: *exact*, *noisy (auto)* when JPEG noise',
        'turned noisy color matching on, or *JPEG* for a JPEG file. A dot matches truth within half a radius; precision is',
        'matched over found, recall matched over truth, **Count** is found minus truth, **RMSE** is over matched dots,',
        '**Flagged** counts dots with any flag (weak, hidden, deep, close). A failure shows its reason in **Found**.',
        '',
    ];
    const sets = [...new Set([...rows.map(r => r.set), ...failures.map(e => e.set)])]
        .sort((a, b) => (SET_ORDER.indexOf(a) + 1 || 99) - (SET_ORDER.indexOf(b) + 1 || 99));
    for (const set of sets) {
        lines.push(`## ${set}`, '', SET_TITLES[set] ?? set, '');
        const rs = rows.filter(r => r.set === set);
        lines.push(...(set === 'synthetic' ? syntheticTable(rs, notes) : imageTable(set, rs, failures.filter(e => e.set === set), notes)), '');
    }
    writeFileSync(OUT, lines.join('\n'));
    console.log(`wrote ${OUT}: ${rows.length} rows, ${failures.length} failures`);
}

main();

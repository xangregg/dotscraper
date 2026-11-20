// tools/bench.mjs as a program: its rows don't depend on how many images it fits at a time.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { mkdtempSync, readFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

const BENCH = fileURLToPath(new URL('../tools/bench.mjs', import.meta.url));

function bench(dir, name, args) {
    const json = join(dir, `${name}.json`);
    const run = spawnSync(process.execPath, [BENCH, ...args, '--json', json], { encoding: 'utf8' });
    assert.equal(run.status, 0, run.stderr);
    return { ...JSON.parse(readFileSync(json, 'utf8')), printed: run.stdout };
}

test('bench --jobs: the same rows, in the same order, as one image at a time', () => {
    const dir = mkdtempSync(join(tmpdir(), 'dotscraper-bench-'));
    try {
        // four synthetic graphs: more jobs than workers, so a worker takes a second one
        const args = ['--preset', 'sparse', '--seeds', '4'];
        const serial = bench(dir, 'serial', [...args, '--jobs', '1']);
        const parallel = bench(dir, 'parallel', [...args, '--jobs', '3']);
        assert.equal(serial.jobs, 1);
        assert.equal(parallel.jobs, 3);
        assert.equal(serial.rows.length, 4);
        // every number but the time, which is the run's own
        const timeless = rows => rows.map(({ ms, ...row }) => row);
        assert.deepEqual(timeless(parallel.rows), timeless(serial.rows));
        assert.deepEqual(parallel.rows.map(r => r.name), ['sparse #1', 'sparse #2', 'sparse #3', 'sparse #4']);
        assert.deepEqual(parallel.failures, serial.failures);
        const table = out => out.split('\n').map(line => line.replace(/\s+\d+$/, ''));
        assert.deepEqual(table(parallel.printed), table(serial.printed));
        for (const r of serial.rows)
            assert.ok(r.precision >= 0.95 && r.recall >= 0.95, `${r.name}: P ${r.precision} R ${r.recall}`);
    }
    finally {
        rmSync(dir, { recursive: true, force: true });
    }
});

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';

const GUARD = new URL('../tools/mem-guard.mjs', import.meta.url).pathname;

// node -e `code` with the guard preloaded at a limit of `gb`
function guarded(code, gb) {
    return spawnSync(process.execPath, ['-e', code], {
        env: { ...process.env, NODE_OPTIONS: `--import=${GUARD}`, MEM_GUARD_GB: String(gb) },
        encoding: 'utf8',
    });
}

test('mem-guard: a process stuck allocating in one synchronous loop is killed past its limit, and says so', () => {
    const run = guarded('const a = []; for (;;) a.push(Buffer.alloc(16 << 20, 1));', 0.5);
    assert.equal(run.signal, 'SIGKILL');
    assert.match(run.stderr, /mem-guard: killed pid \d+ at [\d.]+ GB \(limit 0\.5 GB\)/);
});

test('mem-guard: a process under its limit runs to its end, with worker threads, and exits by itself', () => {
    const run = guarded("const { Worker } = require('node:worker_threads'); new Worker('1', { eval: true }).on('exit', () => console.log('done'));", 0.5);
    assert.equal(run.status, 0);
    assert.equal(run.signal, null);
    assert.equal(run.stdout.trim(), 'done');
});

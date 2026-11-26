// A memory guard for long node runs (tests, mutation checks, the bench): preloaded into a process, it kills the
// process when its resident memory passes a limit, and says so on stderr first. Twice a run gone wrong took the
// machine's memory (a failing assertion printing image-sized arrays, PLAN.md); clean runs stay far below the limit
// (each test file under 0.75 GB, the full bench at 10 jobs 9 GB, measured 2026-11-26).
//
//   NODE_OPTIONS="--import=$PWD/tools/mem-guard.mjs" npm test
//   MEM_GUARD_GB=4 NODE_OPTIONS="--import=$PWD/tools/mem-guard.mjs" node --test test/outline.test.js
//
// NODE_OPTIONS reaches the processes `node --test` starts for each test file, so each one is guarded. The watch runs
// in a worker thread: a runaway is often one long synchronous call (building an assertion's message), during which
// the main thread's timers never fire. RSS is the whole process's, the bench's worker threads included, so only the
// main thread starts a watch. --max-old-space-size doesn't do this: typed arrays live outside the JavaScript heap.

import { Worker, isMainThread } from 'node:worker_threads';

const DEFAULT_LIMIT = 16; // GB of resident memory (MEM_GUARD_GB overrides it)
const CHECK_INTERVAL = 100; // ms between checks (a runaway can allocate 14 GB/s: it overshoots by about that × this)

// the watch, a worker's code (CommonJS, from a string, so that the guard is one file to preload)
const WATCH = `
const { workerData } = require('node:worker_threads');
const { writeSync } = require('node:fs');
setInterval(() => {
    const rss = process.memoryUsage.rss();
    if (rss <= workerData.limit)
        return;
    writeSync(2, \`\\nmem-guard: killed pid \${workerData.pid} at \${(rss / 2 ** 30).toFixed(1)} GB (limit \${workerData.limitGb} GB): \${workerData.what}\\n\`);
    process.kill(workerData.pid, 'SIGKILL');
}, workerData.interval);
`;

if (isMainThread) {
    const limitGb = Number(process.env.MEM_GUARD_GB) || DEFAULT_LIMIT;
    const what = process.argv.slice(1).join(' ');
    new Worker(WATCH, { eval: true, workerData: { limit: limitGb * 2 ** 30, limitGb, interval: CHECK_INTERVAL, pid: process.pid, what } }).unref();
}

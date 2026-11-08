// Fit two renderings of the same graph independently and compare the dots: a consistency check when there is no
// ground truth (dog-ages.png is dog-ages-actual-size.png enlarged 3× and cropped). Estimates x' = a·x + b and
// y' = c·y + d from the second image to the first (coarse offset search, then least squares on dots unflagged in
// both), matches within r/2 of the first image, and reports agreement.
//
// Usage: node tools/compare-fits.mjs first.png second.png

import { readPng } from './png.mjs';
import { extract } from './bench.mjs';
import { matchDots } from '../src/core/metrics.js';

const OFFSET_RANGE = 12; // px of the first image, coarse search either way
const OFFSET_STEP = 0.5; // px
const COARSE_TOL = 3; // px, matching tolerance during the search
const REFINE_TOL = 2; // px, while refining the transform
const REFINE_ITERATIONS = 5;

const [fileA, fileB] = process.argv.slice(2);
const imA = readPng(fileA);
const imB = readPng(fileB);
const A = extract(imA).fit;
const B = extract(imB).fit;
const t = [imA.width / imB.width, 0, imA.height / imB.height, 0];
const mapped = () => B.dots.map(d => ({ x: t[0] * d.x + t[1], y: t[2] * d.y + t[3] }));
let best = null;
for (let tx = -OFFSET_RANGE; tx <= OFFSET_RANGE; tx += OFFSET_STEP) {
    for (let ty = -OFFSET_RANGE; ty <= OFFSET_RANGE; ty += OFFSET_STEP) {
        t[1] = tx;
        t[3] = ty;
        const n = matchDots(mapped(), A.dots, COARSE_TOL).pairs.length;
        if (!best || n > best.n)
            best = { n, tx, ty };
    }
}
t[1] = best.tx;
t[3] = best.ty;
const clean = m => m.pairs.filter(p => !B.dots[p.truth].flags.length && !A.dots[p.found].flags.length);
for (let it = 0; it < REFINE_ITERATIONS; it++) {
    const pairs = clean(matchDots(mapped(), A.dots, REFINE_TOL));
    for (const [k, key] of [[0, 'x'], [2, 'y']]) {
        let n = 0;
        let sx = 0;
        let sy = 0;
        let sxx = 0;
        let sxy = 0;
        for (const p of pairs) {
            const X = B.dots[p.truth][key];
            const Y = A.dots[p.found][key];
            n++;
            sx += X;
            sy += Y;
            sxx += X * X;
            sxy += X * Y;
        }
        t[k] = (n * sxy - sx * sy) / (n * sxx - sx * sx);
        t[k + 1] = (sy - t[k] * sx) / n;
    }
}
const m = matchDots(mapped(), A.dots, 0.5 * A.radius);
const pairs = clean(m);
const rmse = ps => Math.sqrt(ps.reduce((s, p) => s + p.d * p.d, 0) / ps.length);
console.log(`${fileA}: ${A.dots.length} dots (r ${A.radius.toFixed(2)}); ${fileB}: ${B.dots.length} dots (r ${B.radius.toFixed(2)})`);
console.log(`second → first: x' = ${t[0].toFixed(5)}·x + ${t[1].toFixed(3)}, y' = ${t[2].toFixed(5)}·y + ${t[3].toFixed(3)}`);
console.log(`matched ${m.pairs.length}; only in the first ${m.extra.length}, only in the second ${m.missed.length}`);
console.log(`position agreement (px of the first image): unflagged in both ${pairs.length} dots, rmse ${rmse(pairs).toFixed(3)}; all matched rmse ${m.rmse.toFixed(3)}`);
for (const j of m.extra)
    console.log(`  only first: ${A.dots[j].x.toFixed(1)}, ${A.dots[j].y.toFixed(1)} [${A.dots[j].flags.join(' ')}]`);
for (const i of m.missed)
    console.log(`  only second: ${(t[0] * B.dots[i].x + t[1]).toFixed(1)}, ${(t[2] * B.dots[i].y + t[3]).toFixed(1)} [${B.dots[i].flags.join(' ')}]`);

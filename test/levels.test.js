import { test } from 'node:test';
import assert from 'node:assert/strict';
import { alongLine, colorLine, projectOnLine, fitGeometric } from '../src/core/levels.js';

const near = (a, b, tol, what) => assert.ok(Math.abs(a - b) <= tol, `${what}: ${a} vs ${b} (±${tol})`);

test('fitGeometric recovers α and c from exact levels, and flags a single level as underdetermined', () => {
    const B = [255, 250, 245];
    const c = [40, 120, 200];
    const levels = Array.from({ length: 5 }, (_, n) => c.map((v, ch) => v + (B[ch] - v) * 0.7 ** n));
    const fit = fitGeometric(levels);
    near(fit.alpha, 0.3, 1e-6, 'α');
    for (let ch = 0; ch < 3; ch++)
        near(fit.c[ch], c[ch], 1e-3, 'c');
    assert.ok(fitGeometric(levels.slice(0, 2)).underdetermined);
});

test('fitGeometric with a missing level: counts 1, 2 and 4 fit exactly; read as 1, 2, 3 they do not', () => {
    const B = [255, 255, 255];
    const c = [239, 49, 70];
    const at = n => c.map((v, ch) => v + (B[ch] - v) * 0.5 ** n);
    const colors = [B, at(1), at(2), at(4)];
    const fit = fitGeometric(colors, [0, 1, 2, 4]);
    near(fit.alpha, 0.5, 1e-6, 'α');
    for (let ch = 0; ch < 3; ch++)
        near(fit.c[ch], c[ch], 1e-3, 'c');
    assert.ok(fitGeometric(colors).rms > 1, 'consecutive counts should not fit');
});

test('alongLine: a color 30% of the way from one to another, 5 units off the line', () => {
    const w = [60, -20, 30];
    const perp = [1, 3, 0]; // w · perp = 0
    const k = 5 / Math.hypot(...perp);
    const { m, off } = alongLine(w.map((x, ch) => 0.3 * x + k * perp[ch]), w);
    near(m, 0.3, 1e-12, 'm');
    near(off, 5, 1e-12, 'off');
});

test('colorLine and projectOnLine: distance along the line from B toward a color, and off it', () => {
    const B = [255, 255, 255];
    const line = colorLine(B, [55, 155, 255]);
    near(Math.hypot(...line.u), 1, 1e-12, '|u|');
    // B itself, the seed, a color halfway, and one 6 units off the line at the seed (0, 0, 6 is across u)
    const colors = [B, [55, 155, 255], [155, 205, 255], [55, 155, 249]];
    const image = { width: 4, height: 1, data: Uint8ClampedArray.from(colors.flatMap(c => [...c, 255])) };
    const { t, perp } = projectOnLine(image, line);
    const length = Math.hypot(200, 100, 0);
    near(t[0], 0, 1e-4, 't at B');
    near(t[1], length, 1e-3, 't at the seed');
    near(t[2], length / 2, 1e-3, 't halfway');
    near(perp[1], 0, 1e-3, 'perp at the seed');
    near(perp[3], 6, 1e-3, 'perp off the line');
});

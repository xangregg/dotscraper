import { test } from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, readFileSync } from 'node:fs';
import { pixelCoverage, createDotModel, solidImage } from '../src/core/render.js';
import { createOutlineModel, fillPlane, findOutlines, fillFamilies, relativeDarkness, findHollowRings, lineOpening, paletteValue, findPaletteOutlines } from '../src/core/outline.js';
import { estimateFamilies } from '../src/core/families.js';
import { matchDots } from '../src/core/metrics.js';
import { createRng } from '../src/core/rng.js';
import { readImage } from '../tools/png.mjs';
import { readTruthCsv, truthPixels, truthPixelsByTicks } from '../tools/truth-csv.mjs';
import { detectAxes } from '../src/core/axes.js';
import { extractAll } from '../src/core/extract.js';
import { jitterRow, groupRow } from '../tools/bench.mjs';
import { refitNear } from '../src/core/fit.js';

const WHITE = [255, 255, 255];
const W = 360;
const H = 240;
const RADIUS = 8; // px, the ring's outer edge
const RING_WIDTH = 1.5; // px
const RING_COLOR = [40, 40, 40];
const RING_ALPHA = 0.6;
const FILL_ALPHA = 0.75;
const GRID_GRAY = 234; // a gridline under the dots
const GRID_STEP = 60; // px

// a yellow-to-red scale on a planar curve through white (each stop a mix of two offsets from white)
function scaleColor(s) {
    const a = [-5, -40, -120];
    const b = [-60, -200, -160];
    return WHITE.map((v, ch) => v + (1 - s) * a[ch] + s * b[ch]);
}

// Outlined dots in drawing order: each a fill disc to the ring's middle, then a dark ring, both translucent, over
// white with gridlines; exact area coverage, no blur
function outlinedImage(dots, { ringColor = RING_COLOR, fillAlpha = FILL_ALPHA, colorOf = scaleColor, glyph = null } = {}) {
    const img = new Float32Array(3 * W * H);
    for (let y = 0; y < H; y++) {
        for (let x = 0; x < W; x++) {
            const g = x % GRID_STEP < 2 || y % GRID_STEP < 2 ? GRID_GRAY : 255;
            img.fill(g, 3 * (y * W + x), 3 * (y * W + x) + 3);
        }
    }
    const inner = RADIUS - RING_WIDTH;
    const mid = RADIUS - RING_WIDTH / 2;
    for (const d of dots) {
        const c = colorOf(d.s);
        for (let y = Math.floor(d.y - RADIUS - 1); y <= d.y + RADIUS + 1; y++) {
            for (let x = Math.floor(d.x - RADIUS - 1); x <= d.x + RADIUS + 1; x++) {
                if (x < 0 || y < 0 || x >= W || y >= H)
                    continue;
                const mf = fillAlpha * pixelCoverage(d.x, d.y, mid, x, y);
                const mr = RING_ALPHA * (pixelCoverage(d.x, d.y, RADIUS, x, y) - pixelCoverage(d.x, d.y, inner, x, y));
                for (let ch = 0; ch < 3; ch++) {
                    const p = 3 * (y * W + x) + ch;
                    img[p] = img[p] + mf * (c[ch] - img[p]);
                    img[p] = img[p] + mr * (ringColor[ch] - img[p]);
                }
            }
        }
    }
    // a black "o": a stroke 1.5 px wide around a 4 px hole, like a letter in a title
    if (glyph) {
        for (let y = Math.floor(glyph.y - 6); y <= glyph.y + 6; y++) {
            for (let x = Math.floor(glyph.x - 6); x <= glyph.x + 6; x++) {
                const m = pixelCoverage(glyph.x, glyph.y, 5.5, x, y) - pixelCoverage(glyph.x, glyph.y, 4, x, y);
                for (let ch = 0; ch < 3; ch++)
                    img[3 * (y * W + x) + ch] *= 1 - m;
            }
        }
    }
    const data = new Uint8ClampedArray(4 * W * H);
    for (let p = 0; p < W * H; p++) {
        for (let ch = 0; ch < 3; ch++)
            data[4 * p + ch] = Math.round(img[3 * p + ch]);
        data[4 * p + 3] = 255;
    }
    return { width: W, height: H, data };
}

// isolated dots across the scale, and clusters where dots overlap and hide each other's rings
function scene() {
    const rng = createRng(7);
    const dots = [];
    for (let i = 0; i < 12; i++)
        dots.push({ x: 25 + i * 28, y: 30, s: i / 11 });
    for (let k = 0; k < 6; k++) {
        const cx = 40 + k * 55;
        const cy = 110 + (k % 2) * 60;
        for (let j = 0; j < 5; j++)
            dots.push({ x: cx + 14 * (rng.uniform() - 0.5) * 2, y: cy + 14 * (rng.uniform() - 0.5) * 2, s: rng.uniform() });
    }
    return dots;
}

test('createOutlineModel: a later fill dims the ring under it; drawn first, it is the one dimmed', () => {
    const base = new Float32Array(40 * 20);
    const style = { radius: 6, ring: { inner: 0.75, alpha: 0.5 }, fillAlpha: 0.8, ringT: 100 };
    // pixels 8 and 19 are both 5.5 px from (14, 10), on its ring (4.5–6 px); 19 is inside the fill of a dot at
    // (22, 10) (5.25 px around it), 8 is clear of it
    const at = (m, x) => m.image.data[10 * 40 + x];
    const under = createOutlineModel(base, 40, 20, style);
    under.add(14, 10);
    const b = under.add(22, 10); // drawn later, on top
    const free = at(under, 8);
    assert.ok(free > 20, `free ring ${free}`);
    assert.ok(Math.abs(at(under, 19) - (1 - 0.8) * free) < 1e-4, `under the later fill ${at(under, 19)}, free ${free}`);
    const over = createOutlineModel(base, 40, 20, style);
    over.add(22, 10, 1);
    over.add(14, 10, 2); // drawn later this time: over the other's fill (value 0), its ring reads as in the open
    assert.ok(Math.abs(at(over, 19) - free) < 1e-4, `on top ${at(over, 19)}`);
    // snapshot and restore undo a fill value; pixel 23 is inside the later dot's fill only
    const snap = under.snapshot(under.footprint(22, 10));
    under.setFill(b, 30);
    assert.ok(Math.abs(at(under, 23) - 0.8 * 30) < 1e-4, `fill value ${at(under, 23)}`);
    under.restore(snap);
    assert.ok(Math.abs(at(under, 23)) < 1e-4, 'restored');
});

// as the dot model's: snapshot() saves only the dots near its rectangle
test('createOutlineModel: snapshot and restore among many dots leave the dots, their order and the image as they were', () => {
    const MW = 200;
    const MH = 140;
    const COUNT = 50;
    const base = new Float32Array(MW * MH);
    const style = { radius: 6, ring: { inner: 0.75, alpha: 0.5 }, fillAlpha: 0.8, ringT: 100 };
    const rng = createRng(12);
    const union = (a, b) => [Math.min(a[0], b[0]), Math.min(a[1], b[1]), Math.max(a[2], b[2]), Math.max(a[3], b[3])];
    const model = createOutlineModel(base, MW, MH, style);
    model.addAll(Array.from({ length: COUNT }, (_, i) => ({ x: 10 + (MW - 20) * rng.uniform(), y: 10 + (MH - 20) * rng.uniform(), z: i + 1 })));
    for (let t = 0; t < 40; t++) {
        const image = Float32Array.from(model.image.data);
        const dots = JSON.stringify(model.dots());
        const d = model.dots()[Math.floor(COUNT * rng.uniform())];
        const to = { x: d.x + 8 * (rng.uniform() - 0.5), y: d.y + 8 * (rng.uniform() - 0.5) };
        const extra = { x: d.x + 10 * (rng.uniform() - 0.5), y: d.y + 10 * (rng.uniform() - 0.5) };
        const snap = model.snapshot(union(union(model.footprint(d.x, d.y), model.footprint(to.x, to.y)), model.footprint(extra.x, extra.y)));
        if (t % 4 === 0)
            model.move(d.id, to.x, to.y);
        else if (t % 4 === 1)
            model.remove(d.id);
        else if (t % 4 === 2)
            model.setFill(d.id, 25);
        else {
            model.add(extra.x, extra.y);
            model.move(d.id, to.x, to.y);
        }
        model.restore(snap);
        assert.equal(JSON.stringify(model.dots()), dots, `dots after trial ${t}`);
        assert.deepEqual(model.image.data, image, `image after trial ${t}`);
        if (t % 5 === 4)
            model.move(d.id, to.x, to.y);
    }
    // a rejected birth does not raise the drawing order of the next one
    assert.equal(model.zOf(model.add(100, 70)), COUNT + 1);
    // the grid still knows where every dot is: moving each one agrees with a model built from the dots
    for (const d of model.dots())
        model.move(d.id, d.x + 1.3, d.y - 0.7);
    const fresh = createOutlineModel(base, MW, MH, style);
    fresh.addAll(model.dots());
    for (let i = 0; i < base.length; i++)
        assert.ok(Math.abs(fresh.image.data[i] - model.image.data[i]) < 1e-3);
});

test('findOutlines: the fill plane, the ring and αf of outlined dots whose fills vary', () => {
    const image = outlinedImage(scene());
    const plane = fillPlane(image, WHITE);
    assert.ok(plane, 'a fill plane');
    assert.ok(plane.rms < 1.5, `plane rms ${plane.rms}`);
    const found = findOutlines(image, WHITE);
    assert.ok(found, 'outlined dots');
    // the ring in P: K is black, the ring color (40, 40, 40) at α 0.6 is 0.6 · (1 − 40/255 off the plane's scale)
    assert.ok(Math.abs(found.ring.radius - RADIUS) <= 0.25, `radius ${found.ring.radius}`);
    assert.ok(Math.abs(found.ring.radius * (1 - found.ring.inner) - RING_WIDTH) <= 0.4, `ring width ${found.ring.radius * (1 - found.ring.inner)}`);
    assert.ok(Math.abs(found.fillAlpha - FILL_ALPHA) <= 0.1, `fill α ${found.fillAlpha} from ${found.pairs} pairs`);
});

test('outlined dots whose fills vary are fitted by their rings, in drawing order', () => {
    const dots = scene();
    const image = outlinedImage(dots);
    const { estimates } = estimateFamilies(image);
    assert.equal(estimates.length, 1);
    assert.equal(estimates[0].scale, 'outline');
    const results = extractAll(image);
    const fit = results[0].fit.dots;
    const m = matchDots(dots, fit, RADIUS / 2);
    // clusters of 5 hide some rings almost entirely under two or three fills (0.25 or less of their ink shows)
    assert.ok(m.recall >= 0.9, `recall ${m.recall} (${fit.length} found of ${dots.length})`);
    assert.ok(m.precision >= 0.95, `precision ${m.precision}`);
});

test('regression: refitNear keeps outlined dots in drawing order and puts an added one where it belongs', () => {
    // before: edits re-added every dot without its z, so all were redrawn in list order, and an added dot went on top
    const image = outlinedImage(scene());
    const { est, layers, fit } = extractAll(image)[0];
    const input = { image: est.image, style: { ...est.style, radius: fit.radius }, line: est.familyMeasure.line, mask: layers.mask, D: est.D, imgT: est.fitT, base: est.base };
    const overlapping = d => fit.dots.filter(e => e !== d && Math.hypot(e.x - d.x, e.y - d.y) < 1.5 * fit.radius);
    // the dot with the most overlapping neighbors, and some drawn above it and some below
    const gone = fit.dots.filter(d => overlapping(d).some(e => e.z > d.z) && overlapping(d).some(e => e.z < d.z))
        .sort((a, b) => overlapping(b).length - overlapping(a).length)[0];
    assert.ok(gone, 'a dot with neighbors above and below');
    const kept = fit.dots.filter(d => d !== gone);
    const res = refitNear(input, [...kept.map(d => ({ x: d.x, y: d.y, z: d.z })), { x: gone.x + 0.6, y: gone.y - 0.6, added: true }], gone.x, gone.y);
    assert.equal(res.dots.length, fit.dots.length);
    const at = d => res.dots.reduce((b, e) => (Math.hypot(e.x - d.x, e.y - d.y) < Math.hypot(b.x - d.x, b.y - d.y) ? e : b));
    const back = at(gone);
    assert.ok(Math.hypot(back.x - gone.x, back.y - gone.y) < 0.3, `added dot at ${back.x}, ${back.y}, fitted ${gone.x}, ${gone.y}`);
    for (const e of overlapping(gone))
        assert.equal(Math.sign(at(e).z - back.z), Math.sign(e.z - gone.z), `order against the neighbor at ${e.x}, ${e.y}`);
});

const GREEN = [106, 255, 107]; // JMP's light green
const GRAY_RING = [150, 150, 150];
const GLYPH = { x: 340, y: 225 }; // px, clear of the scene's dots

test('one opaque fill color under gray rings: the plane through the fill line that sees black best', () => {
    // before: no fill plane (one color spans no plane), and with opaque fills (αf 1) the fill fit divided by 1 − αf,
    // so every loss was NaN and the fit froze on its first candidates (bench1 solid-graystroke: 172 for 100)
    // plus a row of overlapping pairs, 11 px apart, where αf shows
    const dots = [...scene(), ...Array.from({ length: 16 }, (_, i) => ({ x: 20 + 40 * (i >> 1) + 11 * (i & 1), y: 218, s: 0 }))];
    const image = outlinedImage(dots, { ringColor: GRAY_RING, fillAlpha: 1, colorOf: () => GREEN, glyph: GLYPH });
    const plane = fillPlane(image, WHITE);
    assert.ok(plane && plane.spread < 1, `a plane through one fill line: ${JSON.stringify(plane)}`);
    const found = findOutlines(image, WHITE);
    assert.ok(found, 'outlined dots');
    assert.ok(found.fillAlpha > 0.9, `opaque fills: αf ${found.fillAlpha}`);
    const fit = extractAll(image)[0].fit.dots;
    const m = matchDots(dots, fit, RADIUS / 2);
    assert.ok(m.recall >= 0.95, `recall ${m.recall} (${fit.length} found of ${dots.length})`);
    assert.ok(m.precision >= 0.97, `precision ${m.precision}`);
    // the black "o" is masked, not a ring (bench1 solid-graystroke: 81 phantom dots on the title's letters)
    assert.ok(fit.every(d => Math.hypot(d.x - GLYPH.x, d.y - GLYPH.y) > RADIUS), 'no dot on the glyph');
});

// JMP's two group colors, nearly complementary: the plane through white that holds both almost holds gray
const ORANGE = [255, 145, 56];
const BLUE = [78, 156, 255];

test('two nearly complementary fill colors under black rings: fitted by the rings, each dot its color', () => {
    // before: no plane (black only 0.05 of |K − B| off it, under the 0.2 required), though P's noise is small
    const dots = [...scene(), ...Array.from({ length: 16 }, (_, i) => ({ x: 20 + 40 * (i >> 1) + 11 * (i & 1), y: 218, s: (i >> 1) % 2 }))];
    const colorOf = s => (s < 0.5 ? ORANGE : BLUE);
    const image = outlinedImage(dots, { ringColor: [0, 0, 0], fillAlpha: 1, colorOf });
    const plane = fillPlane(image, WHITE);
    assert.ok(plane && plane.kOff < 0.1, `a plane nearly holding gray: ${JSON.stringify(plane)}`);
    const fills = fillFamilies(image);
    assert.equal(fills?.length, 2, `fill families ${JSON.stringify(fills)}`);
    const results = extractAll(image);
    assert.equal(results.length, 2, 'one result per fill color');
    const found = results.flatMap(r => r.fit.dots);
    const m = matchDots(dots, found, RADIUS / 2);
    assert.ok(m.recall >= 0.95, `recall ${m.recall} (${found.length} found of ${dots.length})`);
    // every dot read where its fill shows takes its own color
    for (const [k, r] of results.entries()) {
        const color = fills[k];
        for (const d of r.fit.dots.filter(q => q.familySure)) {
            const t = dots.reduce((a, b) => (Math.hypot(b.x - d.x, b.y - d.y) < Math.hypot(a.x - d.x, a.y - d.y) ? b : a));
            const want = colorOf(t.s);
            assert.ok(Math.hypot(...want.map((v, ch) => v - color[ch])) < 10, `dot at ${d.x.toFixed(1)}, ${d.y.toFixed(1)} read as ${color}`);
        }
    }
});

test('fillFamilies: a scale and one color are not families', () => {
    assert.equal(fillFamilies(outlinedImage(scene())), null, 'a scale');
    assert.equal(fillFamilies(outlinedImage(scene(), { fillAlpha: 1, colorOf: () => GREEN })), null, 'one color');
});

test('findOutlines: fills without a dark ring are not outlined dots', () => {
    // the same scene with white rings: the fills still lie in their plane, but nothing in P is a ring
    const image = outlinedImage(scene(), { ringColor: WHITE });
    assert.ok(fillPlane(image, WHITE), 'the fill plane is still found');
    assert.equal(findOutlines(image, WHITE), null);
});

test('truthPixelsByTicks: values through detected ticks, interpolated in log(v + 1)', () => {
    const truth = {
        xScale: { type: 'linear', jitter: true }, yScale: { type: 'log y+1', jitter: false },
        xTicks: [0, 10], yTicks: [0, 1, 3],
        rows: [{ x: 5, y: 2 }, { x: 10, y: 0 }],
    };
    // y ticks drawn at log(v + 1): 0 at 100 px, 1 at 100 − 40·ln 2, 3 at 100 − 40·ln 4
    const axes = { x: { ticks: [{ pos: 10 }, { pos: 110 }] }, y: { ticks: [{ pos: 100 - 40 * Math.log(4) }, { pos: 100 - 40 * Math.log(2) }, { pos: 100 }] } };
    const r = truthPixelsByTicks(truth, axes);
    assert.equal(r.jitter, true);
    assert.ok(Math.abs(r.dots[0].x - 60) < 1e-9);
    assert.ok(Math.abs(r.dots[0].y - (100 - 40 * Math.log(3))) < 1e-9, `y ${r.dots[0].y}`);
    assert.ok(Math.abs(r.dots[1].y - 100) < 1e-9);
});

const PATHOGEN = new URL('../bench-local/pathogen-trade-time-scatter', import.meta.url).pathname;

test('regression: pathogen, outlined dots colored by a scale, found by their rings', { skip: !existsSync(PATHOGEN + '.csv') && 'pathogen truth absent' }, () => {
    // before: "No dot family found" (the fills vary by dot, and no color line held enough of them)
    const image = readImage(PATHOGEN + '.png');
    const px = truthPixelsByTicks(readTruthCsv(PATHOGEN + '.csv'), detectAxes(image));
    const r = jitterRow('pathogen', px.dots, extractAll(image), 0);
    // 583 dots, 286 of them in the jittered band at 0, up to 41 to a cell: many are under two or more fills
    assert.ok(r.found >= 440 && r.found <= 500, `found ${r.found}`);
    assert.ok(r.precision >= 0.95, `precision ${r.precision}`);
    assert.ok(r.recall >= 0.75, `recall ${r.recall}`);
});

const benchFile = (set, name) => new URL(`../${set}/${name}`, import.meta.url).pathname;

for (const [set, lo, hi, recall] of [['bench1', 96, 101, 0.97], ['bench2', 182, 200, 0.9]]) {
    test(`regression: one opaque fill under gray rings (${set} solid-graystroke)`, { skip: !existsSync(benchFile(set, 'solid-graystroke.png')) && `${set} absent` }, () => {
        // before: "No dot family found"; then 172 for 100 (the frozen fit), then 210 for 100 (phantoms on the title).
        // JMP draws these markers half a pixel up and left of the plot area's mapping (fill centroids and rings
        // agree to 0.05 px), so positions are 0.7 px off the truth; matching is within r/2.
        const truth = truthPixels(readTruthCsv(benchFile(set, 'data.csv'))).dots;
        const fit = extractAll(readImage(benchFile(set, 'solid-graystroke.png'))).flatMap(r => r.fit.dots);
        const m = matchDots(truth, fit, RADIUS / 2);
        assert.ok(fit.length >= lo && fit.length <= hi, `found ${fit.length}`);
        assert.ok(m.precision >= 0.98, `precision ${m.precision}`);
        assert.ok(m.recall >= recall, `recall ${m.recall}`);
    });
}

for (const [set, name, precision, recall] of [
    ['bench1', 'group-graystroke', 1, 1], ['bench1', 'group-solid-graystroke', 1, 0.97],
    ['bench2', 'group-graystroke', 0.9, 0.87], ['bench2', 'group-solid-graystroke', 0.94, 0.86],
]) {
    test(`regression: two fill colors under gray rings, each dot its group (${set} ${name})`, { skip: !existsSync(benchFile(set, name + '.png')) && `${set} absent` }, () => {
        // before: "No dot family found" or "The dot radius is unknown". Scored by group: a dot of the wrong color is
        // wrong. bench2's misses and wrong colors are near-coincident pairs, where the lower fill doesn't show.
        const truth = truthPixels(readTruthCsv(benchFile(set, 'data.csv'))).dots;
        const results = extractAll(readImage(benchFile(set, name + '.png')));
        const r = groupRow(name, truth, results, 0);
        assert.ok(r.precision >= precision, `precision ${r.precision}`);
        assert.ok(r.recall >= recall, `recall ${r.recall}`);
        // the dots whose fill shows are all in the right group
        const sure = groupRow(name, truth, results.map(q => ({ ...q, fit: { ...q.fit, dots: q.fit.dots.filter(d => d.familySure) } })), 0);
        assert.equal(sure.precision, 1, `sure dots' precision ${sure.precision}`);
    });
}

test('relativeDarkness: fills and the edge between them read 0, a ring its RGB distance from its own fill', () => {
    // bench2 gradient-opaque-stroke's darkest and palest fills, each with a 1 px line of its own ring color
    const DARK = [189, 90, 78];
    const DARK_RING = [120, 53, 45];
    const PALE = [242, 196, 114];
    const PALE_RING = [232, 157, 22];
    const width = 40;
    const height = 20;
    const data = new Uint8ClampedArray(4 * width * height);
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const c = x === 8 ? DARK_RING : x === 30 ? PALE_RING : x < 20 ? DARK : PALE;
            data.set([...c, 255], 4 * (y * width + x));
        }
    }
    const P = relativeDarkness({ width, height, data });
    const at = x => P[10 * width + x];
    const WHITE_LENGTH = 255 * Math.sqrt(3);
    for (const x of [3, 14, 19, 20, 25, 36])
        assert.ok(at(x) < 1e-6, `fill or edge at x ${x}: ${at(x)}`);
    assert.ok(Math.abs(at(8) - Math.hypot(...DARK.map((v, ch) => v - DARK_RING[ch])) / WHITE_LENGTH) < 1e-6, `dark ring ${at(8)}`);
    assert.ok(Math.abs(at(30) - Math.hypot(...PALE.map((v, ch) => v - PALE_RING[ch])) / WHITE_LENGTH) < 1e-6, `pale ring ${at(30)}`);
});

for (const [set, lo, hi, precision, recall, rmse] of [['bench1', 97, 101, 0.99, 0.97, 0.1], ['bench2', 165, 182, 0.97, 0.8, 0.2]]) {
    test(`regression: outlines in each fill's own darker shade, on a scale (${set} gradient-opaque-stroke)`, { skip: !existsSync(benchFile(set, 'gradient-opaque-stroke.png')) && `${set} absent` }, () => {
        // before: the color-scale path, every dot pixel covered and the outlines unused: bench1 100 found, RMSE
        // 0.478 px; bench2 177, precision 0.955, RMSE 0.885, 18 misses that show and 8 dots 6–8 px off in its stacks
        const image = readImage(benchFile(set, 'gradient-opaque-stroke.png'));
        const est = estimateFamilies(image).estimates[0];
        assert.equal(est.scale, 'outline');
        const truth = truthPixels(readTruthCsv(benchFile(set, 'data.csv'))).dots;
        const fit = extractAll(image).flatMap(r => r.fit.dots);
        const m = matchDots(truth, fit, est.style.radius / 2);
        assert.ok(fit.length >= lo && fit.length <= hi, `found ${fit.length}`);
        assert.ok(m.precision >= precision, `precision ${m.precision}`);
        assert.ok(m.recall >= recall, `recall ${m.recall}`);
        assert.ok(m.rmse <= rmse, `rmse ${m.rmse}`);
    });
}

const DROPLET1 = new URL('../bench-plos/droplet1-10.1371-journal.pone.0362653', import.meta.url).pathname;

test('regression: droplet1, opaque black outlines around fills of a diverging scale', { skip: !existsSync(DROPLET1 + '.png') && 'bench-plos droplet1 absent' }, () => {
    // before: no family. The rings, 0.93 in P at their cores, were masked as black text (αk measured 0.028); without
    // the mask, a glyph at r 15 among the isolated rings and a free αk (1.26, a thin blurred ring for a sharp one)
    // stacked 2–3 dots on single dots (68 found, P 0.60). Still missed: the two piles near the origin, whose rings
    // merge into dark areas the openings read as under the dots.
    const image = readImage(DROPLET1 + '.png');
    const est = estimateFamilies(image).estimates[0];
    assert.equal(est.scale, 'outline');
    assert.equal(est.style.ring.alpha, 1, 'αk');
    const truth = truthPixelsByTicks(readTruthCsv(DROPLET1 + '.csv'), detectAxes(image)).dots;
    const fit = extractAll(image).flatMap(r => r.fit.dots);
    const m = matchDots(truth, fit, est.style.radius / 2);
    assert.ok(m.precision >= 0.85, `precision ${m.precision}, found ${fit.length}`);
    assert.ok(m.recall >= 0.7, `recall ${m.recall}, found ${fit.length}`);
});

const HOLLOW_RED = [200, 60, 40];

test('findHollowRings: red rings with nothing inside, over gridlines and beside a black "o", are one hollow family', () => {
    const dots = scene();
    const image = outlinedImage(dots, { ringColor: HOLLOW_RED, fillAlpha: 0, glyph: { x: 330, y: 220 } });
    const found = findHollowRings(image, WHITE);
    assert.equal(found.length, 1);
    const [f] = found;
    assert.equal(f.fillAlpha, 0);
    // (no fill is measured for a hollow ring)
    assert.equal(f.fillAlphaMeasured, false);
    assert.equal(f.pairs, 0);
    assert.ok(Math.abs(f.ring.radius - RADIUS) <= 0.5, `radius ${f.ring.radius}`);
    // the darkening's direction is the ring color's from white
    const d = WHITE.map((v, ch) => v - HOLLOW_RED[ch]);
    const cos = d.reduce((s, v, ch) => s + v * f.hollow.dir[ch], 0) / Math.hypot(...d);
    assert.ok(cos > 0.99, `direction ${f.hollow.dir}`);
    // rings in a found family's color are that family's own ink (its outlines), not rings of their own: the color
    // as the rings show it on the page
    const shown = WHITE.map((v, ch) => Math.round(v + RING_ALPHA * (HOLLOW_RED[ch] - v)));
    assert.equal(findHollowRings(image, WHITE, [shown]).length, 0);
    // … which a family of another color doesn't make them, however near its pixels they lie: pale blue hairlines
    // every 6 px, within reach of every ring pixel, darken the page another way
    const PALE = [215, 225, 255];
    const lined = { ...image, data: new Uint8ClampedArray(image.data) };
    for (let p = 0; p < W * H; p++) {
        if (p % W % 6 === 0 && lined.data[4 * p] === 255 && lined.data[4 * p + 1] === 255 && lined.data[4 * p + 2] === 255)
            lined.data.set(PALE, 4 * p);
    }
    assert.equal(findHollowRings(lined, WHITE).length, 1);
    assert.equal(findHollowRings(lined, WHITE, [PALE]).length, 1);
});

// Small hollow rings as a low-resolution figure shows them (bench-pmc cycle): a wall of 2 px around a ring 13 px
// across, blurred, so that above ISO_LEVEL the ring fills about two thirds of its box
const SMALL_RING = { radius: 6.5, inner: 0.69, blur: 0.9, color: [100, 105, 135] };

function smallRingsImage() {
    const rng = createRng(11);
    const dots = [];
    while (dots.length < 40) {
        const d = { x: 20 + rng.uniform() * (W - 40), y: 20 + rng.uniform() * (H - 40) };
        if (dots.some(e => Math.hypot(e.x - d.x, e.y - d.y) < 2 * SMALL_RING.radius + 6))
            continue;
        dots.push(d);
    }
    const model = createDotModel(solidImage(W, H, WHITE), { color: SMALL_RING.color, alpha: 0, radius: SMALL_RING.radius, ring: { inner: SMALL_RING.inner, alpha: 1 }, blur: SMALL_RING.blur });
    for (const d of dots)
        model.add(d.x, d.y);
    const data = new Uint8ClampedArray(4 * W * H);
    for (let p = 0; p < W * H; p++) {
        for (let ch = 0; ch < 3; ch++)
            data[4 * p + ch] = Math.round(model.image.data[3 * p + ch]);
        data[4 * p + 3] = 255;
    }
    return { image: { width: W, height: H, data }, dots };
}

test('findHollowRings: small blurred rings whose walls fill two thirds of their box are hollow rings', () => {
    const { image, dots } = smallRingsImage();
    const found = findHollowRings(image, WHITE);
    assert.equal(found.length, 1);
    assert.ok(Math.abs(found[0].ring.radius - SMALL_RING.radius) <= 0.5, `radius ${found[0].ring.radius}`);
    const results = extractAll(image);
    assert.equal(results.length, 1);
    const m = matchDots(dots, results[0].fit.dots, SMALL_RING.radius / 2);
    assert.equal(m.recall, 1);
    assert.equal(results[0].fit.dots.length, dots.length);
});

const CYCLE = new URL('../bench-pmc/cycle-10.1371-journal.pone.0305300.g002.png', import.meta.url).pathname;
const CYCLE_TRUTH = new URL('../bench-pmc/truth/cycle-10.1371-journal.pone.0305300.g002.json', import.meta.url).pathname;

test('regression: cycle, small hollow rings at low resolution are a family', { skip: !existsSync(CYCLE) && 'bench-pmc absent' }, () => {
    // before: no family: 2 of its 212 rings passed as hollow. Still missed: rings the smoother crosses, and those
    // stacked two to four deep in a column
    const truth = JSON.parse(readFileSync(CYCLE_TRUTH, 'utf8')).dots;
    const results = extractAll(readImage(CYCLE));
    assert.equal(results.length, 1);
    assert.ok(Math.abs(results[0].fit.radius - 6.3) <= 0.5, `radius ${results[0].fit.radius}`);
    const m = matchDots(truth, results[0].fit.dots, results[0].fit.radius / 2);
    assert.ok(m.precision >= 0.99 && m.recall >= 0.75, `${results[0].fit.dots.length} dots, P ${m.precision.toFixed(3)} R ${m.recall.toFixed(3)}`);
});

test('regression: gray rings are found beside a family whose color darkens the page the same way', () => {
    // song-cents: navy dots, 15, 39, 66, darken the page 9° from the gray of its hollow rings, and every direction
    // within 18° of a found family's was left out, so finding the navy lost the rings. The navy's own thin ink is
    // its dots' rims; the rings are nowhere near them.
    const NAVY = [15, 39, 66];
    const image = outlinedImage(scene(), { fillAlpha: 0 });
    for (let i = 0; i < 8; i++) {
        const [cx, cy] = [30 + i * 40, 222];
        for (let y = cy - 10; y <= cy + 10; y++) {
            for (let x = cx - 10; x <= cx + 10; x++) {
                const m = pixelCoverage(cx, cy, RADIUS, x, y);
                for (let ch = 0; ch < 3; ch++)
                    image.data[4 * (y * W + x) + ch] = Math.round(image.data[4 * (y * W + x) + ch] * (1 - m) + NAVY[ch] * m);
            }
        }
    }
    const found = findHollowRings(image, WHITE, [NAVY]);
    assert.equal(found.length, 1);
    assert.ok(Math.abs(found[0].ring.radius - RADIUS) <= 0.5, `radius ${found[0].ring.radius}`);
    const gray = WHITE.map((v, ch) => v - RING_COLOR[ch]);
    const navy = WHITE.map((v, ch) => v - NAVY[ch]);
    const angle = Math.acos(gray.reduce((s, v, ch) => s + v * navy[ch], 0) / Math.hypot(...gray) / Math.hypot(...navy)) * 180 / Math.PI;
    assert.ok(angle < 18, `the two directions are ${angle.toFixed(1)}° apart`);
});

test('findHollowRings: outlined dots with fills are not hollow', () => {
    const image = outlinedImage(scene(), { ringColor: HOLLOW_RED, fillAlpha: 1 });
    assert.equal(findHollowRings(image, WHITE).length, 0);
});

test('lineOpening is the minimum, then the maximum, over the rounded segment, to the bit: at the edges, with zeros of both signs and NaN', () => {
    // the definition, as the function was first written: every tap tested against the image's bounds
    const plain = (src, width, height, ang, len) => {
        const h = (len - 1) / 2;
        const taps = [];
        for (let k = -h; k <= h; k++)
            taps.push([Math.round(k * Math.cos(ang)), Math.round(k * Math.sin(ang))]);
        const pass = (a, sign, pick, start) => {
            const out = new Float32Array(a.length);
            for (let y = 0; y < height; y++) {
                for (let x = 0; x < width; x++) {
                    let m = start;
                    for (const [dx, dy] of taps) {
                        const xx = x + sign * dx;
                        const yy = y + sign * dy;
                        if (xx >= 0 && xx < width && yy >= 0 && yy < height)
                            m = pick(m, a[yy * width + xx]);
                    }
                    out[y * width + x] = m;
                }
            }
            return out;
        };
        return pass(pass(src, 1, Math.min, Infinity), -1, Math.max, -Infinity);
    };
    const rng = createRng(11);
    const bytes = a => Buffer.from(a.buffer, a.byteOffset, a.byteLength);
    let cases = 0;
    // sizes smaller than the segment, one pixel, and wide enough to have an interior
    for (const [width, height] of [[1, 1], [3, 7], [7, 3], [40, 31], [5, 60], [60, 5]]) {
        for (const len of [1, 5, 9, 21, 51]) {
            const src = new Float32Array(width * height);
            for (let p = 0; p < src.length; p++) {
                const u = rng.uniform();
                src[p] = u < 0.1 ? 0 : u < 0.2 ? -0 : u < 0.21 ? NaN : Math.round(40 * rng.uniform() - 20) / 4;
            }
            for (let k = 0; k < 12; k++) {
                const ang = Math.PI * k / 12;
                assert.ok(bytes(lineOpening(src, width, height, ang, len)).equals(bytes(plain(src, width, height, ang, len))),
                    `${width} × ${height}, ${len} px, direction ${k}`);
                cases++;
            }
        }
    }
    assert.equal(cases, 360);
    // an opening removes what is narrower than the segment and keeps what is longer
    const width = 30;
    const P = new Float32Array(width * 9);
    for (let x = 5; x < 25; x++)
        P[4 * width + x] = 1;
    P[1 * width + 3] = 1;
    const o = lineOpening(P, width, 9, 0, 7);
    assert.equal(o[4 * width + 12], 1);
    assert.equal(o[1 * width + 3], 0);
});

// Filled dots and open circles in one black, the circles' walls thick, all blurred: drawn at 4× and averaged down, then
// a Gaussian of sd `sigma` px. A circle's hole is the page's color, over whatever was drawn before it.
function openAndFilled(width, height, dots, { r, rIn, rOut, sigma }) {
    const S = 4;
    const sum = new Float32Array(width * height);
    for (let y = 0; y < height * S; y++) {
        for (let x = 0; x < width * S; x++) {
            let v = 255;
            for (const d of dots) {
                const q = Math.hypot((x + 0.5) / S - d.x, (y + 0.5) / S - d.y);
                if (d.open ? q <= rOut : q <= r)
                    v = d.open && q < rIn ? 255 : 0;
            }
            sum[Math.floor(y / S) * width + Math.floor(x / S)] += v / (S * S);
        }
    }
    const half = Math.ceil(3 * sigma);
    const kernel = Array.from({ length: 2 * half + 1 }, (_, i) => Math.exp(-((i - half) ** 2) / (2 * sigma * sigma)));
    const norm = kernel.reduce((a, b) => a + b, 0);
    const pass = (src, dx, dy) => {
        const dst = new Float32Array(src.length);
        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
                for (let k = -half; k <= half; k++) {
                    const xx = Math.min(width - 1, Math.max(0, x + k * dx));
                    const yy = Math.min(height - 1, Math.max(0, y + k * dy));
                    dst[y * width + x] += kernel[k + half] / norm * src[yy * width + xx];
                }
            }
        }
        return dst;
    };
    const out = pass(pass(sum, 1, 0), 0, 1);
    const data = new Uint8ClampedArray(4 * width * height);
    for (let p = 0; p < width * height; p++)
        data.set([out[p], out[p], out[p], 255], 4 * p);
    return { width, height, data };
}

test('regression: open circles with thick walls beside filled dots of the same black are two families (corneal)', () => {
    // before: the filled dots alone. The search for hollow rings left the dots' color to them, and walls this thick
    // (4.8 px of 9.5) are no thin ink for it: against the fill around them they read unevenly, half as dark as text
    const rng = createRng(6);
    const dots = [];
    // two swarms: open circles at the left, filled dots at the right, some touching
    for (const [cx, open] of [[150, true], [400, false]]) {
        for (let n = 0; n < 30;) {
            const d = { x: cx + 110 * (rng.uniform() - 0.5), y: 50 + 300 * rng.uniform(), open };
            if (dots.every(e => Math.hypot(e.x - d.x, e.y - d.y) > 19)) {
                dots.push(d);
                n++;
            }
        }
    }
    const results = extractAll(openAndFilled(550, 400, dots, { r: 8.3, rIn: 4.7, rOut: 9.5, sigma: 1 }));
    assert.equal(results.length, 2);
    for (const r of results) {
        const hollow = !!r.est.candidates[0].hollow;
        const m = matchDots(dots.filter(d => d.open === hollow), r.fit.dots, 4.5);
        assert.equal(m.pairs.length, 30, `${hollow ? 'circles' : 'dots'}: ${m.pairs.length} of 30 found`);
        assert.equal(r.fit.dots.length, 30, `${hollow ? 'circles' : 'dots'}: ${r.fit.dots.length} in all`);
    }
    const rings = results.find(r => r.est.candidates[0].hollow).est.style;
    assert.ok(Math.abs(rings.radius - 9.5) < 0.8 && rings.ring.inner < 0.7, `ring from ${rings.ring.inner * rings.radius} to ${rings.radius}`);
});

const VIABILITY = new URL('../bench-plos/viability-10.1371-journal.pone.0361912.png', import.meta.url).pathname;

test('thick-walled rings in a solid family\'s color: not the o\'s of text in that color (viability)', { skip: !existsSync(VIABILITY) && 'bench-plos absent' }, () => {
    // the reading for open circles beside filled ones finds thin rings too, along a solid family's direction: the o's
    // of the legend's red text (inner 0.73 of their radius) were a family of red rings, and song-cents's 46 gray rings
    // a second family of 42. Its rings must be thick-walled
    const { estimates } = estimateFamilies(readImage(VIABILITY));
    const rings = estimates.filter(e => e.candidates?.[0]?.hollow);
    assert.ok(rings.every(e => Math.max(...e.style.color) - Math.min(...e.style.color) < 30), `ring families: ${rings.map(e => e.style.color).join('; ')}`);
    assert.equal(estimates.filter(e => e.familyMeasure?.kind === 'opaque').length, 2);
});

// grapes' fills as its color clusters give them: saturated colors that share no plane through white. The dark green
// is half the light one, and its pixels (DARK_DRAWN: the legend marker's) lie nearer the light green's line to black,
// 6 units off, than to their own cluster's color, 12.
const DARK = [9, 123, 11];
const DARK_DRAWN = [2, 127, 2];
const PALETTE = [[10, 247, 13], [243, 5, 10], [246, 7, 243], DARK, [4, 2, 181], [68, 127, 243]];
const DRAWN = PALETTE.map(c => (c === DARK ? DARK_DRAWN : c));
const PW = 560;
const PH = 380;
const PR = 10; // px, the dots' radius to the ring's outer edge
const PRING = 2; // px, their rings
const LEGEND_RING = 2.9; // px, the legend markers' (heavier, and still under half their box: isolated rings)
const LEGEND_X = 480;
const LINE = { x0: 30, y0: 322, x1: 330, y1: 364, width: 2 }; // 8° off the rows

// Opaque fills in opaque black rings, in drawing order: dots { x, y, k (its color's index), ring? }; then black
// rectangles and a thin black line
function paletteImage(dots, { rects = [], line = null } = {}) {
    const img = new Float32Array(3 * PW * PH).fill(255);
    for (const d of dots) {
        const ring = d.ring ?? PRING;
        const c = d.k === null ? WHITE : DRAWN[d.k];
        for (let y = Math.floor(d.y - PR - 1); y <= d.y + PR + 1; y++) {
            for (let x = Math.floor(d.x - PR - 1); x <= d.x + PR + 1; x++) {
                const mf = d.hollow ? 0 : pixelCoverage(d.x, d.y, PR - ring / 2, x, y);
                const mr = pixelCoverage(d.x, d.y, PR, x, y) - pixelCoverage(d.x, d.y, PR - ring, x, y);
                for (let ch = 0; ch < 3; ch++) {
                    const p = 3 * (y * PW + x) + ch;
                    img[p] = (1 - mr) * (img[p] + mf * (c[ch] - img[p]));
                }
            }
        }
    }
    for (const [x0, y0, x1, y1] of rects) {
        for (let y = y0; y < y1; y++)
            img.fill(0, 3 * (y * PW + x0), 3 * (y * PW + x1));
    }
    if (line) {
        const len = Math.hypot(line.x1 - line.x0, line.y1 - line.y0);
        const ux = (line.x1 - line.x0) / len;
        const uy = (line.y1 - line.y0) / len;
        for (let y = 0; y < PH; y++) {
            for (let x = 0; x < PW; x++) {
                const dx = x + 0.5 - line.x0;
                const dy = y + 0.5 - line.y0;
                const along = dx * ux + dy * uy;
                const m = along < 0 || along > len ? 0 : Math.min(1, Math.max(0, line.width / 2 + 0.5 - Math.abs(dx * uy - dy * ux)));
                for (let ch = 0; ch < 3; ch++)
                    img[3 * (y * PW + x) + ch] *= 1 - m;
            }
        }
    }
    const data = new Uint8ClampedArray(4 * PW * PH);
    for (let p = 0; p < PW * PH; p++) {
        for (let ch = 0; ch < 3; ch++)
            data[4 * p + ch] = Math.round(img[3 * p + ch]);
        data[4 * p + 3] = 255;
    }
    return { width: PW, height: PH, data };
}

// one isolated dot of each color, and overlapping groups in mixed colors
function paletteDots() {
    const dots = PALETTE.map((_, k) => ({ x: 40 + 60 * k, y: 40.5, k }));
    for (let g = 0; g < 6; g++) {
        const cx = 50 + 70 * g;
        const cy = 110 + 70 * (g % 3);
        for (let j = 0; j < 4; j++)
            dots.push({ x: cx + 9 * Math.cos(1.7 * j + g) + 0.3 * j, y: cy + 9 * Math.sin(2.3 * j + g) + 0.2 * g, k: (g + (j >> 1)) % PALETTE.length });
    }
    return dots;
}

// a legend: a marker for each color, of the dots' size and outlined heavier, at even steps, each before a letter or
// two (blocks 3 px wide). With it each color has two isolated fills: one more, and it would be found by its color.
function legend(n = PALETTE.length) {
    const markers = [];
    const rects = [];
    for (let i = 0; i < n; i++) {
        const y = 60 + 34 * i;
        markers.push({ x: LEGEND_X, y: y + 0.5, k: i, ring: LEGEND_RING });
        // an L, and for every other entry an I after it
        rects.push([LEGEND_X + 22, y - 8, LEGEND_X + 25, y + 8], [LEGEND_X + 22, y + 5, LEGEND_X + 31, y + 8]);
        if (i % 2)
            rects.push([LEGEND_X + 35, y - 8, LEGEND_X + 38, y + 8]);
    }
    return { markers, rects };
}

test('paletteValue: fills of colors that share no plane read 0 and black rings 1; a dark fill half a lighter one is no ring', () => {
    const dots = paletteDots();
    const image = paletteImage(dots);
    assert.equal(fillPlane(image, WHITE), null, 'no fill plane');
    const P = paletteValue(image, WHITE, PALETTE);
    const at = (x, y) => P[Math.floor(y) * PW + Math.floor(x)];
    for (const d of dots.slice(0, PALETTE.length)) {
        assert.ok(Math.abs(at(d.x, d.y)) < 0.05, `fill ${d.k} reads ${at(d.x, d.y)}`);
        assert.ok(at(d.x + PR - PRING / 2, d.y) > 0.9, `ring of ${d.k} reads ${at(d.x + PR - PRING / 2, d.y)}`);
    }
    assert.ok(at(300, 300) === 0, 'the page');
});

test('findPaletteOutlines: black rings around fills are read at full darkness; rings around the page alone are none', () => {
    const dots = paletteDots();
    const found = findPaletteOutlines(paletteImage(dots), WHITE, PALETTE, () => true);
    assert.ok(found, 'found');
    assert.ok(Math.abs(found.ring.radius - PR) <= 0.3, `radius ${found.ring.radius}`);
    assert.ok(Math.abs(found.ring.radius * (1 - found.ring.inner) - PRING) <= 0.4, `ring width ${found.ring.radius * (1 - found.ring.inner)}`);
    assert.equal(found.ring.alpha, 1);
    // plain dots of two colors and four letter o's: rings, with the page inside
    const plain = [0, 1, 2, 3, 4, 5, 6, 7].map(i => ({ x: 40 + 50 * i, y: 200.5, k: i % 2, ring: 0 }));
    const os = [0, 1, 2, 3].map(i => ({ x: 60 + 40 * i, y: 60.5, k: null, hollow: true }));
    assert.equal(findPaletteOutlines(paletteImage([...plain, ...os]), WHITE, PALETTE.slice(0, 2), () => true), null);
});

test('black rings read unmasked: a thin black line off the rows and columns is under the rings, not ring ink', () => {
    const found = findPaletteOutlines(paletteImage(paletteDots(), { line: LINE }), WHITE, PALETTE, () => true);
    assert.ok(found, 'found');
    let unread = 0;
    let total = 0;
    for (let s = 0.15; s <= 0.85; s += 0.01) {
        const x = Math.floor(LINE.x0 + s * (LINE.x1 - LINE.x0));
        const y = Math.floor(LINE.y0 + s * (LINE.y1 - LINE.y0));
        total++;
        if (Number.isNaN(found.D[y * PW + x]))
            unread++;
    }
    assert.ok(unread >= 0.95 * total, `${unread} of ${total} points along the line unread`);
});

test('outlined dots of six fill colors off any plane, with a legend outlined heavier: one family of rings, each dot its color', () => {
    const dots = paletteDots();
    const { markers, rects } = legend();
    const image = paletteImage([...dots, ...markers], { rects, line: LINE });
    const { estimates } = estimateFamilies(image);
    assert.equal(estimates.length, 1);
    const est = estimates[0];
    assert.equal(est.scale, 'outline', est.notes?.join(' '));
    assert.equal(est.fillFamilies.length, PALETTE.length);
    const width = est.style.radius * (1 - est.style.ring.inner);
    assert.ok(Math.abs(width - PRING) <= 0.4, `ring width ${width}: the dots', not the legend's`);
    assert.ok(est.isolated.length >= 3 && !est.isolated.some(c => Math.abs(c.cx - LEGEND_X) < PR), `measured from the plot's rings alone: ${JSON.stringify(est.isolated)}`);
    const results = extractAll(image);
    const found = results.flatMap(r => r.fit.dots.map(d => ({ ...d, color: r.est.style.color })));
    const m = matchDots(dots, found, PR / 2);
    assert.ok(m.recall >= 0.9, `recall ${m.recall} (${found.length} found of ${dots.length})`);
    assert.ok(m.precision >= 0.97, `precision ${m.precision}: extras ${m.extra.map(i => `${found[i].x.toFixed(0)},${found[i].y.toFixed(0)}`)}`);
    for (const d of found.filter(q => q.familySure)) {
        const t = dots.reduce((a, b) => (Math.hypot(b.x - d.x, b.y - d.y) < Math.hypot(a.x - d.x, a.y - d.y) ? b : a));
        assert.ok(Math.hypot(...DRAWN[t.k].map((v, ch) => v - d.color[ch])) < 12, `dot at ${d.x.toFixed(1)}, ${d.y.toFixed(1)} read as ${d.color.map(Math.round)}`);
    }
});

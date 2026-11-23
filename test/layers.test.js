import { test } from 'node:test';
import assert from 'node:assert/strict';
import { existsSync } from 'node:fs';
import { createRng } from '../src/core/rng.js';
import { distanceTransform, dilate, opening, buildMask, components8, plotClips, CLASS_FAMILY, CLASS_OTHER, CLASS_BACKGROUND } from '../src/core/layers.js';
import { estimateStyle } from '../src/core/style.js';
import { synthGraph, SYNTH_PRESETS, scatterPoints } from '../src/core/synth.js';
import { readImage, readPng } from '../tools/png.mjs';
import { extractAll } from '../src/core/extract.js';
import { readTruthJson } from '../tools/truth-json.mjs';
import { matchDots } from '../src/core/metrics.js';
import { detectAxes, plotBounds, clipEdges } from '../src/core/axes.js';
import { fitDots } from '../src/core/fit.js';

const near = (a, b, tol, what) => assert.ok(Math.abs(a - b) <= tol, `${what}: ${a} vs ${b} (±${tol})`);

test('distance transform equals brute force on random masks', () => {
    const rng = createRng(11);
    for (const [w, h, density] of [[17, 11, 0.05], [30, 25, 0.01], [9, 40, 0.3], [5, 5, 0]]) {
        const mask = new Uint8Array(w * h);
        for (let p = 0; p < w * h; p++)
            mask[p] = rng.uniform() < density ? 1 : 0;
        const d2 = distanceTransform(mask, w, h);
        for (let y = 0; y < h; y++) {
            for (let x = 0; x < w; x++) {
                let best = Infinity;
                for (let q = 0; q < w * h; q++) {
                    if (mask[q])
                        best = Math.min(best, (q % w - x) ** 2 + (Math.floor(q / w) - y) ** 2);
                }
                assert.equal(d2[y * w + x], best, `${w}×${h} at ${x},${y}`);
            }
        }
    }
});

test('opening removes a line thinner than 2r and keeps a disc larger than r', () => {
    const w = 80;
    const h = 60;
    const mask = new Uint8Array(w * h);
    for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
            if (Math.abs(y - 10) <= 1)
                mask[y * w + x] = 1; // 3 px line
            if ((x - 40) ** 2 + (y - 38) ** 2 <= 144)
                mask[y * w + x] = 1; // disc of radius 12
        }
    }
    const o = opening(mask, w, h, 4);
    assert.equal(o[10 * w + 20], 0);
    assert.equal(o[38 * w + 40], 1);
    assert.equal(o[38 * w + 51], 1); // near the disc's edge
    const grown = dilate(mask, w, h, 2);
    assert.equal(grown[13 * w + 20], 1);
    assert.equal(grown[14 * w + 20], 0);
});

// distance from (x, y) to a polyline
function lineDistance(x, y, pts) {
    let best = Infinity;
    for (let i = 0; i + 1 < pts.length; i++) {
        const [ax, ay] = pts[i];
        const [bx, by] = pts[i + 1];
        const dx = bx - ax;
        const dy = by - ay;
        const t = Math.max(0, Math.min(1, ((x - ax) * dx + (y - ay) * dy) / (dx * dx + dy * dy)));
        best = Math.min(best, Math.hypot(x - ax - t * dx, y - ay - t * dy));
    }
    return best;
}

test('synthetic curve graph: the lines are masked, dot pixels away from them are not', () => {
    const spec = SYNTH_PRESETS.curve(1);
    const { image, truth } = synthGraph(spec);
    const est = estimateStyle(image);
    const { mask, classes } = buildMask(image, est);
    let lineCore = 0;
    let lineMasked = 0;
    let dotFar = 0;
    let dotFarMasked = 0;
    for (let y = 0; y < image.height; y++) {
        for (let x = 0; x < image.width; x++) {
            const p = y * image.width + x;
            // the solid curve's core (dashes have gaps, so use the solid black line only)
            if (lineDistance(x + 0.5, y + 0.5, spec.lines[0].points) < spec.lines[0].width / 2 - 1) {
                lineCore++;
                lineMasked += mask[p];
            }
            if (!truth.some(t => Math.hypot(x + 0.5 - t.x, y + 0.5 - t.y) < 24 - 2))
                continue;
            const dLine = Math.min(...spec.lines.map(l => lineDistance(x + 0.5, y + 0.5, l.points) - l.width / 2));
            if (dLine > 6) {
                dotFar++;
                dotFarMasked += mask[p];
                assert.equal(classes[p], CLASS_FAMILY);
            }
        }
    }
    assert.equal(lineMasked, lineCore, 'every pixel of the solid line is masked');
    assert.ok(dotFarMasked / dotFar < 0.002, `${dotFarMasked} of ${dotFar} dot pixels masked`);
});

test('an opaque line in the family’s full color is masked as thin; dots under it are not', () => {
    const c = [246, 66, 60];
    const rng = createRng(4);
    const points = Array.from({ length: 40 }, () => ({ x: 30 + 340 * rng.uniform(), y: 30 + 240 * rng.uniform() }));
    const spec = {
        width: 400, height: 300, background: [255, 255, 255], blur: 0.6,
        families: [{ color: c, alpha: 0.3, radius: 8.4, points }],
        lines: [{ points: [[10, 250], [200, 120], [390, 60]], width: 7, color: c }],
    };
    const { image } = synthGraph(spec);
    const est = estimateStyle(image);
    assert.ok(Math.abs(est.style.alpha - 0.3) < 0.01);
    const { mask, thin, stats } = buildMask(image, est);
    // the line's core is found as thin; its blurred edge, a little lighter, is covered by the mask's growth
    let core = 0;
    let coreThin = 0;
    for (let y = 0; y < 300; y++) {
        for (let x = 0; x < 400; x++) {
            const d = lineDistance(x + 0.5, y + 0.5, spec.lines[0].points);
            if (d < 2) {
                core++;
                coreThin += thin[y * 400 + x];
            }
            if (d < spec.lines[0].width / 2)
                assert.equal(mask[y * 400 + x], 1, `line pixel ${x},${y} masked`);
        }
    }
    assert.ok(coreThin / core > 0.95, `${coreThin} of ${core} line core pixels thin`);
    // an isolated dot away from the line stays unmasked at its center
    const far = points.find(p => lineDistance(p.x, p.y, spec.lines[0].points) > 20 && points.every(q => q === p || Math.hypot(q.x - p.x, q.y - p.y) > 20));
    assert.equal(mask[Math.floor(far.y) * 400 + Math.floor(far.x)], 0);
    assert.ok(stats.thin > 0);
});

test('regression: a line darker than the full color the levels give is past any stack of dots, and masked as thin (richness)', () => {
    // richness: c, from two levels at α 0.22, came out 25 units short of the regression line's color on the family's
    // line. More than 8 from c, the line wasn't "saturated": nothing of it was masked, and the fit read it as dots
    // 14 deep. Here c comes out 16 units short of the dots' own, and the line, 4% farther from the page than that, is
    // 27 past it
    const c = [246, 66, 60];
    const dark = c.map(v => Math.round(255 + 1.04 * (v - 255)));
    const rng = createRng(4);
    const points = Array.from({ length: 40 }, () => ({ x: 30 + 340 * rng.uniform(), y: 30 + 240 * rng.uniform() }));
    const spec = {
        width: 400, height: 300, background: [255, 255, 255], blur: 0.6,
        families: [{ color: c, alpha: 0.22, radius: 8.4, points }],
        lines: [{ points: [[10, 250], [200, 120], [390, 60]], width: 7, color: dark }],
    };
    const { image } = synthGraph(spec);
    const est = estimateStyle(image);
    const short = Math.hypot(...est.style.color.map((v, ch) => v - dark[ch]));
    assert.ok(short > 20 && short < 30, `the line's color is ${short} from the full color read`);
    const { mask, thin } = buildMask(image, est);
    let core = 0;
    let coreThin = 0;
    for (let y = 0; y < 300; y++) {
        for (let x = 0; x < 400; x++) {
            const d = lineDistance(x + 0.5, y + 0.5, spec.lines[0].points);
            if (d < 2) {
                core++;
                coreThin += thin[y * 400 + x];
            }
            if (d < spec.lines[0].width / 2)
                assert.equal(mask[y * 400 + x], 1, `line pixel ${x},${y} masked`);
        }
    }
    assert.ok(coreThin / core > 0.95, `${coreThin} of ${core} line core pixels thin`);
    const far = points.find(p => lineDistance(p.x, p.y, spec.lines[0].points) > 20 && points.every(q => q === p || Math.hypot(q.x - p.x, q.y - p.y) > 20));
    assert.equal(mask[Math.floor(far.y) * 400 + Math.floor(far.x)], 0);
});

const DOG = new URL('../testgraphs/dog-ages.png', import.meta.url).pathname;
test('dog-ages: curve, dashes and text masked; dots and background not', { skip: !existsSync(DOG) && 'local test image absent' }, () => {
    const image = readPng(DOG);
    const est = estimateStyle(image);
    const { mask, classes, thin } = buildMask(image, est);
    const at = (x, y) => ({ m: mask[y * image.width + x], c: classes[y * image.width + x] });
    // dots beside the curve and dashes, whose pale halos read as background, are not "too thin to be dots"
    for (const [x0, y0, x1, y1] of [[389, 361, 407, 388], [414, 420, 442, 438], [419, 395, 424, 410]]) {
        for (let y = y0; y <= y1; y++) {
            for (let x = x0; x <= x1; x++)
                assert.equal(thin[y * image.width + x], 0, `${x},${y}`);
        }
    }
    assert.deepEqual(at(714, 508), { m: 0, c: CLASS_FAMILY }); // isolated dot at 14.5, 47
    assert.deepEqual(at(60, 60), { m: 0, c: CLASS_BACKGROUND });
    // every pure black or dash-gray pixel inside the plot area is masked
    let n = 0;
    for (let y = 230; y < 850; y++) {
        for (let x = 285; x < 790; x++) {
            const p = 4 * (y * image.width + x);
            const [r, g, b] = image.data.slice(p, p + 3);
            if ((r < 30 && g < 30 && b < 30) || (Math.abs(r - 147) < 4 && Math.abs(g - 149) < 4 && Math.abs(b - 152) < 4)) {
                n++;
                assert.equal(mask[y * image.width + x], 1, `${x},${y}`);
                assert.equal(classes[y * image.width + x], CLASS_OTHER);
            }
        }
    }
    assert.ok(n > 3000, `${n} line pixels checked`);
});

test('nearestTransform finds a nearest mask pixel (same distance as brute force)', async () => {
    const { nearestTransform } = await import('../src/core/layers.js');
    const rng = createRng(12);
    for (const [w, h, density] of [[23, 17, 0.04], [9, 31, 0.2], [6, 6, 0]]) {
        const mask = new Uint8Array(w * h);
        for (let p = 0; p < w * h; p++)
            mask[p] = rng.uniform() < density ? 1 : 0;
        const near = nearestTransform(mask, w, h);
        const d2 = distanceTransform(mask, w, h);
        for (let p = 0; p < w * h; p++) {
            if (d2[p] === Infinity) {
                assert.equal(near[p], -1);
                continue;
            }
            assert.equal(mask[near[p]], 1);
            const dx = near[p] % w - p % w;
            const dy = Math.floor(near[p] / w) - Math.floor(p / w);
            assert.equal(dx * dx + dy * dy, d2[p]);
        }
    }
});

test('glyph-sized strokes in a translucent family’s own color are masked; dots are not', () => {
    // JMP's 50% black dots share their color line with black text, whose strokes read as counts 1–8 and are too
    // short to be lines: the fit put dots on the tick labels and title
    const rng = createRng(6);
    const points = scatterPoints(rng, { n: 30, box: [30, 60, 370, 270] });
    // overlapping partners, so α can be measured
    for (const p of points.slice(0, 15))
        points.push({ x: p.x + 5, y: p.y + 3 });
    const strokes = [];
    for (let k = 0; k < 12; k++) {
        const x = 20 + 30 * k;
        strokes.push({ points: [[x, 15], [x, 27]], width: 1.6, color: [0, 0, 0] });
        strokes.push({ points: [[x + 3, 15], [x + 9, 27]], width: 1.6, color: [0, 0, 0] });
    }
    const spec = { width: 400, height: 300, background: [255, 255, 255], blur: 0.4, families: [{ color: [0, 0, 0], alpha: 0.5, radius: 8, points }], lines: strokes };
    const { image } = synthGraph(spec);
    const est = estimateStyle(image);
    near(est.style.alpha, 0.5, 0.02, 'α');
    const { mask } = buildMask(image, est);
    for (const s of strokes) {
        const [[x0, y0], [x1, y1]] = s.points;
        assert.equal(mask[Math.round((y0 + y1) / 2) * 400 + Math.round((x0 + x1) / 2)], 1, `stroke at ${x0}`);
    }
    for (const p of points)
        assert.equal(mask[Math.floor(p.y) * 400 + Math.floor(p.x)], 0, `dot at ${p.x}, ${p.y}`);
});

test('clipped dots: where the plot cuts the dots at the axis and the x axis line\'s end, beyond is masked and they are found whole', () => {
    // before (autocor1): the cut read as the dots' edge; at the right end the fit moved them in and stacked three
    // where two were cut
    const rng = createRng(5);
    const points = scatterPoints(rng, { n: 24, box: [90, 40, 380, 260] });
    const cut = [{ x: 63, y: 120 }, { x: 61, y: 200 }, { x: 403, y: 90 }, { x: 406, y: 170 }];
    const spec = {
        width: 440, height: 340, background: [255, 255, 255], blur: 0.5,
        families: [{ color: [193, 59, 141], alpha: 1, radius: 9, points: [...points, ...cut] }],
        axes: {
            x: { pos: 290.5, from: 59, to: 405, ticks: [80, 140, 200, 260, 320, 380] },
            y: { pos: 60.5, from: 20, to: 291.5, ticks: [270, 220, 170, 120, 70] },
        },
    };
    const { image } = synthGraph(spec);
    // the plot clips its dots: beyond the y axis line and the x axis line's end, the dots' pixels are the page
    const { width, height, data } = image;
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const p = 4 * (y * width + x);
            const chroma = Math.max(data[p], data[p + 1], data[p + 2]) - Math.min(data[p], data[p + 1], data[p + 2]);
            if (chroma > 30 && (x < 60 || x >= 405))
                data.set([255, 255, 255], p);
        }
    }
    const axes = detectAxes(image);
    const est = estimateStyle(image);
    const fit = opts => {
        const layers = buildMask(image, est, opts);
        return { layers, dots: fitDots({ image, style: est.style, line: est.familyMeasure.line, mask: layers.mask, D: est.D, imgT: est.fitT }).dots };
    };
    const clipped = fit({ bounds: plotBounds(axes), clips: clipEdges(axes) });
    assert.deepEqual(clipped.layers.stats.clipped.map(c => [c.dir, c.out]).sort(), [['v', -1], ['v', 1]]);
    for (const c of cut)
        assert.ok(clipped.dots.some(d => Math.hypot(d.x - c.x, d.y - c.y) <= 1.5), `cut dot at ${c.x}, ${c.y}: ${JSON.stringify(clipped.dots.filter(d => Math.abs(d.y - c.y) < 15).map(d => [+d.x.toFixed(1), +d.y.toFixed(1)]))}`);
    assert.equal(clipped.dots.length, points.length + cut.length);
    // without the cut, the dots at the right end were moved in
    const plain = fit({ bounds: plotBounds(axes) });
    assert.ok(cut.slice(2).some(c => !plain.dots.some(d => Math.hypot(d.x - c.x, d.y - c.y) <= 1.5)), 'unclipped, a cut dot is misplaced');
});

test('a bar in the dots\' color, cut into short pieces by the dots on it, is masked as a line', () => {
    // before (testgraphs Laser-Asyllable-Maturity): the bar's pieces between the dots, 5–10 px each, were too short
    // for the length test and stayed unmasked
    const color = [58, 152, 176];
    const R = 5.7;
    const loose = [[40, 30], [90, 25], [140, 35], [190, 30], [240, 40], [70, 70], [200, 80], [250, 110], [215, 150], [260, 170]];
    const onBar = [[67, 155], [89, 156], [111, 155], [133, 156]];
    const spec = {
        width: 300, height: 200, background: [255, 255, 255], blur: 0.2,
        families: [{ color, alpha: 1, radius: R, points: [...loose, ...onBar].map(([x, y]) => ({ x, y })) }],
    };
    const { image } = synthGraph(spec);
    // a bracket: two sides and a bottom bar, 5 px thick, through the dots on it
    const { width, data } = image;
    const rect = (x0, y0, x1, y1) => {
        for (let y = y0; y < y1; y++) {
            for (let x = x0; x < x1; x++)
                data.set(color, 4 * (y * width + x));
        }
    };
    rect(55, 90, 60, 158);
    rect(140, 90, 145, 158);
    rect(55, 153, 145, 158);
    const est = estimateStyle(image);
    near(est.style.radius, R, 0.5, 'radius');
    const layers = buildMask(image, est);
    for (const x of [78, 100, 122])
        assert.ok(layers.mask[155 * width + x], `the bar between the dots at x ${x}`);
    const { dots } = fitDots({ image, style: est.style, line: est.familyMeasure.line, mask: layers.mask, D: est.D, imgT: est.fitT });
    for (const [x, y] of [...loose, ...onBar])
        assert.ok(dots.some(d => Math.hypot(d.x - x, d.y - y) <= 1.5), `dot at ${x}, ${y}`);
    assert.equal(dots.length, loose.length + onBar.length, JSON.stringify(dots.map(d => [+d.x.toFixed(1), +d.y.toFixed(1)])));
});

const LASER = new URL('../testgraphs/Laser-Asyllable-Maturity.png', import.meta.url).pathname;
const LASER_TRUTH = new URL('../testgraphs/truth/Laser-Asyllable-Maturity.json', import.meta.url).pathname;

test('regression: Laser, no dots fitted on the brackets\' bars between the dots on them', { skip: !(existsSync(LASER) && existsSync(LASER_TRUTH)) && 'testgraphs Laser absent' }, () => {
    // before: 80 found for 70, all 70 matched, 10 extras on the bars (P 0.875)
    const results = extractAll(readImage(LASER));
    const truth = readTruthJson(LASER_TRUTH).dots;
    const found = results.flatMap(r => r.fit.dots);
    const m = matchDots(truth, found, 0.5 * results[0].fit.radius);
    assert.ok(m.extra.length <= 5, `${m.extra.length} extras: ${JSON.stringify(m.extra.map(j => [Math.round(found[j].x), Math.round(found[j].y)]))}`);
    assert.ok(m.missed.length <= 2, `${m.missed.length} missed`);
});

test('bounds: pixels more than a dot radius beyond the axes are masked, nothing nearer changes', () => {
    const { image } = synthGraph(SYNTH_PRESETS.sparse(1));
    const est = estimateStyle(image);
    const r = est.style.radius;
    const bounds = { left: 60.5, right: null, top: null, bottom: image.height - 40.5 };
    const plain = buildMask(image, est);
    const bounded = buildMask(image, est, { bounds });
    const { width, height } = image;
    let outside = 0;
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const p = y * width + x;
            const out = x + 0.5 < bounds.left - r || y + 0.5 > bounds.bottom + r;
            outside += out;
            assert.equal(bounded.mask[p], out ? 1 : plain.mask[p], `${x},${y}`);
        }
    }
    assert.equal(bounded.stats.outside, outside);
});

test('components8: pixels joined at a corner are one component; boxes include their last row and column', () => {
    const width = 7;
    const rows = [
        '#......',
        '.#...##',
        '..#..##',
        '.......',
        '#......',
    ];
    const mask = Uint8Array.from(rows.join(''), c => (c === '#' ? 1 : 0));
    const comps = components8(mask, width, rows.length);
    // in the order of their first pixels
    assert.deepEqual(comps.map(c => [c.x0, c.y0, c.x1, c.y1, c.pixels.length]), [[0, 0, 2, 2, 3], [5, 1, 6, 2, 4], [0, 4, 0, 4, 1]]);
    assert.deepEqual([...comps[0].pixels].sort((a, b) => a - b), [0, 8, 16]);
    assert.deepEqual(components8(new Uint8Array(12), 4, 3), []);
});

test('plotClips: a cut that the families show together is every family\'s; ink they share is counted once', () => {
    const width = 200;
    const height = 160;
    const r = 10;
    const cutAt = 150;
    // a disc of radius r, cut straight at the column cutAt (nothing of it from there on)
    const plane = discs => {
        const D = new Float32Array(width * height);
        for (const [cx, cy] of discs) {
            for (let y = 0; y < height; y++) {
                for (let x = 0; x < Math.min(width, cutAt); x++) {
                    if ((x + 0.5 - cx) ** 2 + (y + 0.5 - cy) ** 2 <= r * r)
                        D[y * width + x] = 1;
                }
            }
        }
        return D;
    };
    const edge = { dir: 'v', pos: cutAt, out: 1, thickness: 0, from: null, to: null };
    // each family: whole dots, and one cut 8 px past its middle, a chord of 1.5 r: less than the 2 r a family must show
    const a = { D: plane([[40, 30], [142, 40]]), radius: r };
    const b = { D: plane([[60, 120], [142, 90]]), radius: r };
    // (the cut is the last column of the dots: the boundary pixel may be antialiased, so it is masked too)
    const cut = cutAt - 1;
    const inside = { D: plane([[80, 80]]), radius: r };
    assert.deepEqual(plotClips([a], width, height, [edge]), [[]], 'one family alone: 1.5 r of cut rows is not enough');
    const clips = plotClips([a, b, inside], width, height, [edge]);
    assert.deepEqual(clips.map(c => c.map(e => e.cut)), [[cut], [cut], [cut]], 'together 3 r: cut, also for the family with no dot there');
    assert.equal(clips[0][0].dir, 'v');
    // buildMask's own reading is kept for a family that shows the cut alone
    const two = { D: plane([[142, 40], [142, 90]]), radius: r };
    assert.deepEqual(plotClips([two], width, height, [edge]).map(c => c.map(e => e.cut)), [[cut]]);
    assert.deepEqual(plotClips([two, inside], width, height, [edge]).map(c => c.map(e => e.cut)), [[cut], [cut]]);
    // the same cut dot in two families' coverage (ink in a color they share) is one dot, 1.5 r: no cut
    const shared = plotClips([{ D: a.D, radius: r }, { D: a.D, radius: r }], width, height, [edge]);
    assert.deepEqual(shared, [[], []]);
    // a family without a radius has no clip, and gives none
    assert.deepEqual(plotClips([a, { D: b.D, radius: undefined }], width, height, [edge]), [[], []]);
    // whole dots at the edge are no cut: the union only adds rows that go on beyond it
    const through = new Float32Array(width * height);
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            if ((x + 0.5 - 150) ** 2 + (y + 0.5 - 130) ** 2 <= 30 * 30)
                through[y * width + x] = 1;
        }
    }
    assert.deepEqual(plotClips([a, b, { D: through, radius: 30 }], width, height, [edge]), [[], [], []], 'a large dot across the edge: under 70% of the full rows are cut');
});

// Phase 0 risk check: measure the style of one translucent dot family in a graph image, to test the forward model
// of PLAN.md §2.1 on real pixels before building on it. Reports the background, the overlap levels along the dot
// color line, how well a geometric (constant-alpha) model fits those levels in sRGB values and in linear light,
// the dot radius and edge blur from isolated dots, and signs of upscaling. Optionally writes the overlap count
// image D as a PNG for viewing.
//
// Usage: node tools/measure-dots.mjs image.png [--seed r,g,b] [--out dir]
//   --seed  a color of one dot over the background (default: the most common non-background color)
//   --out   write D (overlap count, 0 black to MAX_LEVEL_SHOWN white) and its legend values to this directory

import { mkdirSync } from 'node:fs';
import { join, basename } from 'node:path';
import { readPng, writePng } from './png.mjs';
import { median } from '../src/core/stats.js';

const PERP_TOL = 4; // 0–255 units: max distance of a pixel from the color line to count as dot-colored
// Near the background every color is close to the line in absolute terms (antialiased gray text one step from white
// is within PERP_TOL), so the distance off the line must also be small relative to the distance along it.
const PERP_REL_TOL = 0.12; // max perp / t
// Levels are plateaus, so only flat pixels vote for them: with small dots, antialiased edge pixels outnumber the
// interiors and would make false peaks.
const FLAT_TOL = 3; // 0–255 units: max channel difference within a pixel's 3×3 neighborhood to count as flat
const BIN = 1; // 0–255 units along the color line, for the level histogram
const PEAK_MIN_FRACTION = 0.004; // a level's peak bin must hold this fraction of the dot-colored pixels
const PEAK_SEPARATION = 10; // 0–255 units: minimum distance between levels
const PEAK_HALF_WIDTH = 3; // bins on each side summed into a level's mean
const ISOLATED_MAX_LEVEL = 1.25; // components whose level never exceeds this are single, unoverlapped dots
const ISOLATED_MIN_AREA = 50; // px², to skip specks
// Dots partly covered by a line or by another family stay below the level limit but aren't whole discs.
const ISOLATED_MAX_ASPECT = 1.05; // max ratio of the principal second-moment radii
const ISOLATED_MAX_MOMENT_RATIO = 1.03; // max r(moment) / r(area); equal for a uniform disc
const EDGE_RING = 4; // px beyond the area radius sampled for the edge profile
const PROFILE_BIN = 0.25; // px, radial profile bins
const MAX_LEVEL_SHOWN = 6; // overlap count shown as white in the D image
const MAX_STEP_RATIO = 0.95; // a level whose step exceeds this ratio to the previous step ends the overlap sequence
const DEEP_LEVELS = 6; // predicted levels reported beyond the detected ones
const DEEP_WINDOW = 1.5; // 0–255 units around each predicted level

function parseArgs(argv) {
    const opts = { file: null, seed: null, out: null };
    for (let i = 0; i < argv.length; i++) {
        if (argv[i] === '--seed')
            opts.seed = argv[++i].split(',').map(Number);
        else if (argv[i] === '--out')
            opts.out = argv[++i];
        else
            opts.file = argv[i];
    }
    return opts;
}

function srgbToLinear(v) {
    const s = v / 255;
    return s <= 0.04045 ? s / 12.92 : ((s + 0.055) / 1.055) ** 2.4;
}

function mostCommonColors(image) {
    const counts = new Map();
    const n = image.width * image.height;
    for (let p = 0; p < n; p++) {
        const key = (image.data[4 * p] << 16) | (image.data[4 * p + 1] << 8) | image.data[4 * p + 2];
        counts.set(key, (counts.get(key) ?? 0) + 1);
    }
    return [...counts].sort((a, b) => b[1] - a[1]).map(([k, v]) => ({ color: [k >> 16, (k >> 8) & 255, k & 255], count: v }));
}

/**
 * Project every pixel onto the line from the background B toward the seed color. Returns t (distance from B along
 * the line, 0–255 units) and perp (distance off the line) per pixel.
 */
function projectOnLine(image, B, seed) {
    const d = [B[0] - seed[0], B[1] - seed[1], B[2] - seed[2]];
    const len = Math.hypot(...d);
    const u = d.map(v => v / len);
    const n = image.width * image.height;
    const t = new Float32Array(n);
    const perp = new Float32Array(n);
    for (let p = 0; p < n; p++) {
        const e0 = B[0] - image.data[4 * p];
        const e1 = B[1] - image.data[4 * p + 1];
        const e2 = B[2] - image.data[4 * p + 2];
        const tp = e0 * u[0] + e1 * u[1] + e2 * u[2];
        t[p] = tp;
        perp[p] = Math.hypot(e0 - tp * u[0], e1 - tp * u[1], e2 - tp * u[2]);
    }
    return { t, perp, u };
}

function onLine(t, perp, tol) {
    return perp <= tol && perp <= PERP_REL_TOL * t;
}

/** 1 where every channel of the pixel's 3×3 neighborhood is within FLAT_TOL of the pixel itself. */
function flatPixels(image) {
    const { width, height, data } = image;
    const flat = new Uint8Array(width * height);
    for (let y = 1; y < height - 1; y++) {
        for (let x = 1; x < width - 1; x++) {
            const p = y * width + x;
            let ok = true;
            for (let dy = -1; dy <= 1 && ok; dy++) {
                for (let dx = -1; dx <= 1 && ok; dx++) {
                    const q = p + dy * width + dx;
                    for (let ch = 0; ch < 3; ch++) {
                        if (Math.abs(data[4 * q + ch] - data[4 * p + ch]) > FLAT_TOL)
                            ok = false;
                    }
                }
            }
            flat[p] = ok ? 1 : 0;
        }
    }
    return flat;
}

/**
 * Overlap levels: peaks of the histogram of t over dot-colored pixels. Each level's color is the mean of the flat
 * pixels within PEAK_HALF_WIDTH bins of its peak.
 */
function findLevels(image, t, perp) {
    const nb = Math.ceil(450 / BIN);
    const hist = new Float64Array(nb);
    const flat = flatPixels(image);
    let total = 0;
    for (let p = 0; p < t.length; p++) {
        if (flat[p] && onLine(t[p], perp[p], PERP_TOL) && t[p] > PEAK_SEPARATION / 2) {
            hist[Math.round(t[p] / BIN)]++;
            total++;
        }
    }
    const peaks = [];
    for (let b = 1; b < nb - 1; b++) {
        if (hist[b] < PEAK_MIN_FRACTION * total || hist[b] < hist[b - 1] || hist[b] < hist[b + 1])
            continue;
        const last = peaks[peaks.length - 1];
        if (last !== undefined && (b - last) * BIN < PEAK_SEPARATION) {
            if (hist[b] > hist[last])
                peaks[peaks.length - 1] = b;
            continue;
        }
        peaks.push(b);
    }
    return peaks.map(b => {
        const sum = [0, 0, 0];
        let count = 0;
        for (let p = 0; p < t.length; p++) {
            if (flat[p] && onLine(t[p], perp[p], PERP_TOL) && Math.abs(t[p] / BIN - b) <= PEAK_HALF_WIDTH) {
                sum[0] += image.data[4 * p];
                sum[1] += image.data[4 * p + 1];
                sum[2] += image.data[4 * p + 2];
                count++;
            }
        }
        return { t: b * BIN, peakCount: hist[b], count, color: sum.map(s => s / count) };
    });
}

/**
 * Least-squares fit of L_n = c + (B − c) q^n per channel, with one q shared by the channels, to the levels
 * L_0 = B, L_1, …: a grid over q, and c in closed form per channel for each q. Returns q, c and the residuals.
 */
function fitGeometric(levels) {
    let best = null;
    for (let q = 0.3; q <= 0.95; q += 0.0005) {
        const c = [];
        let sse = 0;
        const res = [];
        for (let ch = 0; ch < 3; ch++) {
            const B = levels[0][ch];
            // L_n − B q^n = c (1 − q^n): regression through the origin on w_n = 1 − q^n
            let sw = 0;
            let swy = 0;
            for (let n = 0; n < levels.length; n++) {
                const w = 1 - q ** n;
                sw += w * w;
                swy += w * (levels[n][ch] - B * q ** n);
            }
            c.push(swy / sw);
            for (let n = 0; n < levels.length; n++) {
                const r = levels[n][ch] - (c[ch] + (B - c[ch]) * q ** n);
                res.push(r);
                sse += r * r;
            }
        }
        if (best === null || sse < best.sse)
            best = { q, c, sse, res };
    }
    best.rms = Math.sqrt(best.sse / best.res.length);
    best.maxAbs = Math.max(...best.res.map(Math.abs));
    return best;
}

/** Fractional overlap level of each pixel: piecewise-linear interpolation of t between the level positions. */
function levelImage(t, perp, levelTs) {
    const n = t.length;
    const D = new Float32Array(n);
    const K = levelTs.length - 1;
    for (let p = 0; p < n; p++) {
        if (!onLine(t[p], perp[p], 3 * PERP_TOL))
            continue; // not dot-colored: left at 0 here; the real pipeline masks such pixels
        const tp = t[p];
        if (tp <= 0)
            continue;
        let k = 0;
        while (k < K - 1 && tp > levelTs[k + 1])
            k++;
        D[p] = k + (tp - levelTs[k]) / (levelTs[k + 1] - levelTs[k]);
    }
    return D;
}

/** Connected components (4-neighbor) of pixels with D > 0.5. Returns arrays of pixel indices. */
function components(D, width, height) {
    const label = new Int32Array(D.length).fill(-1);
    const comps = [];
    const stack = [];
    for (let s = 0; s < D.length; s++) {
        if (D[s] <= 0.5 || label[s] >= 0)
            continue;
        const comp = [];
        label[s] = comps.length;
        stack.push(s);
        while (stack.length) {
            const p = stack.pop();
            comp.push(p);
            const x = p % width;
            const y = (p - x) / width;
            const nbrs = [x > 0 ? p - 1 : -1, x < width - 1 ? p + 1 : -1, y > 0 ? p - width : -1, y < height - 1 ? p + width : -1];
            for (const q of nbrs) {
                if (q >= 0 && D[q] > 0.5 && label[q] < 0) {
                    label[q] = comps.length;
                    stack.push(q);
                }
            }
        }
        comps.push(comp);
    }
    return comps;
}

function erfc(x) {
    // Abramowitz & Stegun 7.1.26, |error| < 1.5e-7
    const z = Math.abs(x);
    const t = 1 / (1 + 0.3275911 * z);
    const y = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429)))) * Math.exp(-z * z);
    return x >= 0 ? y : 2 - y;
}

/**
 * Isolated dots: components that never exceed ISOLATED_MAX_LEVEL. For each, the coverage-weighted centroid and area
 * (Σ min(D, 1) over a window around it, so edge pixels count fractionally), the area radius, the moment radius
 * (√(2·mean ρ²), equal to r for a uniform disc) and the aspect ratio of the second moments.
 */
function isolatedDots(D, width, height) {
    const dots = [];
    for (const comp of components(D, width, height)) {
        if (comp.length < ISOLATED_MIN_AREA)
            continue;
        let maxD = 0;
        let x0 = width;
        let x1 = 0;
        let y0 = height;
        let y1 = 0;
        for (const p of comp) {
            maxD = Math.max(maxD, D[p]);
            const x = p % width;
            const y = (p - x) / width;
            x0 = Math.min(x0, x);
            x1 = Math.max(x1, x);
            y0 = Math.min(y0, y);
            y1 = Math.max(y1, y);
        }
        if (maxD > ISOLATED_MAX_LEVEL)
            continue;
        // window with a margin, so the soft edge outside the D > 0.5 component is included
        const m = EDGE_RING;
        if (x0 - m < 0 || y0 - m < 0 || x1 + m >= width || y1 + m >= height)
            continue;
        let a = 0;
        let sx = 0;
        let sy = 0;
        for (let y = y0 - m; y <= y1 + m; y++) {
            for (let x = x0 - m; x <= x1 + m; x++) {
                const w = Math.min(1, D[y * width + x]);
                a += w;
                sx += w * (x + 0.5);
                sy += w * (y + 0.5);
            }
        }
        const cx = sx / a;
        const cy = sy / a;
        let mxx = 0;
        let myy = 0;
        let mxy = 0;
        for (let y = y0 - m; y <= y1 + m; y++) {
            for (let x = x0 - m; x <= x1 + m; x++) {
                const w = Math.min(1, D[y * width + x]);
                const dx = x + 0.5 - cx;
                const dy = y + 0.5 - cy;
                mxx += w * dx * dx;
                myy += w * dy * dy;
                mxy += w * dx * dy;
            }
        }
        mxx /= a;
        myy /= a;
        mxy /= a;
        const tr = mxx + myy;
        const det = mxx * myy - mxy * mxy;
        const l1 = tr / 2 + Math.sqrt(Math.max(0, tr * tr / 4 - det));
        const l2 = tr / 2 - Math.sqrt(Math.max(0, tr * tr / 4 - det));
        const rArea = Math.sqrt(a / Math.PI);
        if (Math.sqrt(l1 / l2) > ISOLATED_MAX_ASPECT || Math.sqrt(2 * tr) / rArea > ISOLATED_MAX_MOMENT_RATIO)
            continue;
        dots.push({ cx, cy, area: a, rArea, rMoment: Math.sqrt(2 * tr), aspect: Math.sqrt(l1 / l2), box: [x0, y0, x1, y1] });
    }
    return dots;
}

/**
 * Edge profile of the isolated dots: D against distance from each dot's centroid, normalized by subtracting the dot's
 * own area radius, pooled and binned. Fit D(ρ) = ½ erfc((ρ − r) / (σ√2)), a disc edge blurred by a Gaussian of sd σ.
 */
function edgeProfile(D, width, dots) {
    const nb = Math.ceil(2 * EDGE_RING / PROFILE_BIN) + 1;
    const sum = new Float64Array(nb);
    const cnt = new Float64Array(nb);
    for (const dot of dots) {
        const [x0, y0, x1, y1] = dot.box;
        for (let y = y0 - EDGE_RING; y <= y1 + EDGE_RING; y++) {
            for (let x = x0 - EDGE_RING; x <= x1 + EDGE_RING; x++) {
                const rel = Math.hypot(x + 0.5 - dot.cx, y + 0.5 - dot.cy) - dot.rArea;
                const b = Math.round((rel + EDGE_RING) / PROFILE_BIN);
                if (b < 0 || b >= nb)
                    continue;
                sum[b] += Math.min(1, D[y * width + x]);
                cnt[b]++;
            }
        }
    }
    const profile = [];
    for (let b = 0; b < nb; b++) {
        if (cnt[b] > 0)
            profile.push({ rel: b * PROFILE_BIN - EDGE_RING, d: sum[b] / cnt[b] });
    }
    let best = null;
    for (let off = -1.5; off <= 1.5; off += 0.01) {
        for (let s = 0.05; s <= 3; s += 0.01) {
            let sse = 0;
            for (const { rel, d } of profile) {
                const r = d - 0.5 * erfc((rel - off) / (s * Math.SQRT2));
                sse += r * r;
            }
            if (best === null || sse < best.sse)
                best = { offset: off, sigma: s, sse };
        }
    }
    return { profile, ...best };
}

/**
 * Signs of integer upscaling: if an image was enlarged k× by pixel replication, the intensity changes between
 * horizontal neighbors fall on one phase of x mod k. Reports, for k = 2–4, the share of change energy at each phase
 * (uniform 1/k means no sign of k× replication).
 */
function upscalePhases(image) {
    const { width, height, data } = image;
    const out = {};
    for (let k = 2; k <= 4; k++) {
        const e = new Float64Array(k);
        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width - 1; x++) {
                const p = 4 * (y * width + x);
                const d = Math.abs(data[p] - data[p + 4]) + Math.abs(data[p + 1] - data[p + 5]) + Math.abs(data[p + 2] - data[p + 6]);
                e[x % k] += d;
            }
        }
        const tot = e.reduce((s, v) => s + v, 0);
        out[k] = [...e].map(v => v / tot);
    }
    return out;
}

function fmt(v, d = 1) {
    return v.toFixed(d);
}

function main() {
    const opts = parseArgs(process.argv.slice(2));
    const image = readPng(opts.file);
    const { width, height } = image;
    const common = mostCommonColors(image);
    const B = common[0].color;
    const seed = opts.seed ?? common[1].color;
    console.log(`${basename(opts.file)}: ${width} × ${height} px`);
    console.log(`background ${B.join(',')} (${fmt(100 * common[0].count / (width * height))}% of pixels); seed ${seed.join(',')}`);

    const { t, perp } = projectOnLine(image, B, seed);
    const found = findLevels(image, t, perp);
    // Overlap steps shrink by (1 − α) each time. A level reached by a larger step than the one before it is not the
    // next overlap: typically an opaque mark in the dot color (economist: the trend line is the dots' full color).
    let nSeq = found.length;
    for (let i = 1; i < found.length; i++) {
        const prevStep = found[i - 1].t - (i > 1 ? found[i - 2].t : 0);
        if ((found[i].t - found[i - 1].t) / prevStep > MAX_STEP_RATIO) {
            nSeq = i;
            break;
        }
    }
    const levels = found.slice(0, nSeq);
    console.log('\noverlap levels along the color line (t: distance from the background, 0–255 units):');
    console.log('   n      t   color (mean)           pixels   step   step ratio');
    let prevT = 0;
    let prevStep = NaN;
    found.forEach((L, i) => {
        const step = L.t - prevT;
        const n = i < nSeq ? String(i + 1).padStart(2) : ' –';
        const note = i < nSeq ? '' : '   not an overlap level (step grew)';
        console.log(`  ${n} ${fmt(L.t).padStart(6)}   ${L.color.map(v => fmt(v)).join(', ').padEnd(20)} ${String(L.count).padStart(8)} ${fmt(step).padStart(6)}   ${Number.isNaN(prevStep) ? '     ' : fmt(step / prevStep, 3)}${note}`);
        prevStep = step;
        prevT = L.t;
    });

    const levelColors = [B, ...levels.map(L => L.color)];
    const srgb = fitGeometric(levelColors);
    const lin = fitGeometric(levelColors.map(c => c.map(srgbToLinear)));
    console.log('\ngeometric model L_n = c + (B − c)(1 − α)^n, one α for all channels:');
    console.log(`  sRGB values:  α = ${fmt(1 - srgb.q, 3)}, c = ${srgb.c.map(v => fmt(v)).join(', ')}; residual rms ${fmt(srgb.rms, 2)}, max ${fmt(srgb.maxAbs, 2)} (0–255)`);
    console.log(`  linear light: α = ${fmt(1 - lin.q, 3)}, c = ${lin.c.map(v => fmt(255 * v)).join(', ')} (×255); residual rms ${fmt(255 * lin.rms, 2)}, max ${fmt(255 * lin.maxAbs, 2)} (×255, linear units)`);
    if (levels.length >= 3) {
        const later = fitGeometric(levelColors.slice(1));
        console.log(`  sRGB, levels 1+ only (B left out): α = ${fmt(1 - later.q, 3)}, c = ${later.c.map(v => fmt(v)).join(', ')}; residual rms ${fmt(later.rms, 2)}, max ${fmt(later.maxAbs, 2)}`);
        const predictedB = later.c.map((c, ch) => c + (levelColors[1][ch] - c) / later.q);
        console.log(`    → the background those levels imply: ${predictedB.map(v => fmt(v)).join(', ')} (actual ${B.join(',')})`);
    }
    for (const L of found.slice(nSeq))
        console.log(`  the level at t = ${fmt(L.t)} (${L.color.map(v => fmt(v)).join(', ')}) is ${fmt(Math.hypot(...L.color.map((v, ch) => v - srgb.c[ch])), 1)} units from the fitted full color c`);

    // Deeper levels hold too few pixels to make histogram peaks; the fit predicts where they would be. Their steps
    // shrink by (1 − α) each time, which bounds how many stacked dots the colors can distinguish at all.
    const T = Math.hypot(...B.map((b, ch) => b - srgb.c[ch]));
    let maxT = 0;
    for (let p = 0; p < t.length; p++) {
        if (onLine(t[p], perp[p], PERP_TOL))
            maxT = Math.max(maxT, t[p]);
    }
    console.log(`  deeper levels predicted by the sRGB fit (pixels on the line within ±${DEEP_WINDOW} of each; deepest on-line t = ${fmt(maxT)}):`);
    for (let n = levels.length + 1; n <= levels.length + DEEP_LEVELS; n++) {
        const tn = T * (1 - srgb.q ** n);
        let near = 0;
        for (let p = 0; p < t.length; p++) {
            if (onLine(t[p], perp[p], PERP_TOL) && Math.abs(t[p] - tn) <= DEEP_WINDOW)
                near++;
        }
        console.log(`    n = ${String(n).padStart(2)}: t ${fmt(tn).padStart(6)}, step ${fmt(T * srgb.q ** (n - 1) * (1 - srgb.q), 1).padStart(4)}, ${near} px`);
    }

    const D = levelImage(t, perp, [0, ...levels.map(L => L.t)]);
    const dots = isolatedDots(D, width, height);
    console.log(`\nisolated dots (no pixel above level ${ISOLATED_MAX_LEVEL}, aspect ≤ ${ISOLATED_MAX_ASPECT}, r(moment)/r(area) ≤ ${ISOLATED_MAX_MOMENT_RATIO}): ${dots.length}`);
    for (const d of dots)
        console.log(`  center ${fmt(d.cx).padStart(6)}, ${fmt(d.cy).padStart(6)}   r(area) ${fmt(d.rArea, 2)}   r(moment) ${fmt(d.rMoment, 2)}   aspect ${fmt(d.aspect, 3)}`);
    if (dots.length) {
        const rs = dots.map(d => d.rArea);
        console.log(`  r(area): median ${fmt(median(rs), 2)} px, range ${fmt(Math.min(...rs), 2)}–${fmt(Math.max(...rs), 2)}`);
        const edge = edgeProfile(D, width, dots);
        console.log(`\nedge profile (pooled over isolated dots, ρ relative to each dot's area radius):`);
        for (const { rel, d } of edge.profile)
            console.log(`  ${fmt(rel, 2).padStart(6)}  ${fmt(d, 3)}  ${'#'.repeat(Math.round(40 * d))}`);
        console.log(`  erf fit: σ = ${fmt(edge.sigma, 2)} px (10–90% width ${fmt(2.563 * edge.sigma, 2)} px), edge at ${fmt(edge.offset, 2)} px from the area radius`);
    }

    console.log('\nhorizontal change energy by phase of x mod k (uniform = no sign of k× pixel replication):');
    for (const [k, ph] of Object.entries(upscalePhases(image)))
        console.log(`  k = ${k}: ${ph.map(v => fmt(v, 3)).join('  ')}`);

    if (opts.out) {
        mkdirSync(opts.out, { recursive: true });
        const img = { width, height, data: new Uint8ClampedArray(4 * width * height) };
        for (let p = 0; p < D.length; p++) {
            const v = Math.round(255 * Math.min(1, D[p] / MAX_LEVEL_SHOWN));
            img.data.set([v, v, v, 255], 4 * p);
        }
        const path = join(opts.out, basename(opts.file, '.png') + '-D.png');
        writePng(path, img);
        console.log(`\nwrote ${path} (black 0 … white ${MAX_LEVEL_SHOWN} overlapping dots)`);
    }
}

main();

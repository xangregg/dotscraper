// Every dot family of an image, ready to fit (PLAN.md phase 5): the style estimates, the shaded regions under or
// over the dots (background.js), overlays divided out, and each translucent family on its own image (unmix.js).
// The bench and the app both go through here, so they fit the same thing. Dots colored by a continuous scale are
// one family on a virtual image of their counts (colormap.js).

import { estimateStyle, measureFamily, backgroundColor, flatMask, JPEG_TOLERANCES, EXACT_TOLERANCES } from './style.js';
import { backgroundRegions, removeOverlays } from './background.js';
import { unmixEstimates } from './unmix.js';
import { findColormap, rayImage, colormapCandidate } from './colormap.js';

const DEFAULT_RADIUS = 5; // px, for the region test when no family has a radius yet
// One translucent family over two backgrounds looks like two opaque families when it has too few overlaps to show
// its levels (bench1 ellipse: orange on white and orange on lavender). Its colors there differ by q·(B − B₂), with
// q = 1 − α, which gives α and c without any overlap. A color within this of the prediction is that family.
const MERGE_TOL = 6; // 0–255 units (RGB distance)
const MERGE_Q = [0.05, 0.95]; // a plausible 1 − α

// a candidate's full color and α, as background.js wants them
function candidateStyle(c) {
    return c.kind === 'opaque' ? { color: c.seed, alpha: 1 } : { color: c.fit.c, alpha: c.fit.alpha };
}

const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

/**
 * Candidates that are one translucent family seen over the main background B and over a region under the dots:
 * an opaque candidate that is a translucent candidate's dot over a region color is dropped, and two opaque
 * candidates whose colors differ by q·(B − B₂) become one translucent candidate with α = 1 − q.
 *
 * @returns {object[]|null}  the new candidates, or null if nothing merged
 */
function mergeAcrossRegions(image, B, candidates, regions, flat, tol) {
    const under = regions.filter(r => r.mode === 'under').map(r => r.color);
    if (!under.length)
        return null;
    let out = [...candidates];
    let changed = false;
    // an opaque cluster that is a translucent family's level 1 over B₂
    out = out.filter(c => {
        if (c.kind !== 'opaque')
            return true;
        const covered = out.some(t => t.kind === 'translucent' && under.some(B2 => {
            const q = 1 - t.fit.alpha;
            const level1 = t.fit.c.map((v, ch) => v + (B2[ch] - v) * q);
            return Math.hypot(...sub(c.seed, level1)) <= MERGE_TOL;
        }));
        changed ||= covered;
        return !covered;
    });
    // two opaque clusters that are one translucent family over B and over B₂
    for (const B2 of under) {
        const d = sub(B, B2);
        const dd = dot(d, d);
        for (let i = 0; i < out.length; i++) {
            for (let j = 0; j < out.length; j++) {
                const a = out[i];
                const b = out[j];
                if (i === j || a.kind !== 'opaque' || b.kind !== 'opaque')
                    continue;
                // a over B, b over B₂: a − b = q (B − B₂)
                const diff = sub(a.seed, b.seed);
                const q = dot(diff, d) / dd;
                if (q < MERGE_Q[0] || q > MERGE_Q[1] || Math.hypot(...sub(diff, d.map(v => q * v))) > MERGE_TOL)
                    continue;
                const c = a.seed.map((v, ch) => (v - q * B[ch]) / (1 - q));
                const m = measureFamily(image, B, a.seed, flat, tol);
                if (!m.levels.length)
                    continue;
                const merged = {
                    ...m, kind: 'translucent', translucent: true,
                    fit: { alpha: 1 - q, c, rms: 0, maxAbs: 0, underdetermined: false },
                };
                out = out.filter((_, k) => k !== i && k !== j);
                out.unshift(merged);
                changed = true;
                break;
            }
        }
    }
    return changed ? out : null;
}

/**
 * The estimate for dots colored by a continuous scale, fitted on the virtual image (colormap.rayImage), or null
 * when the image has no such family.
 */
function colormapEstimate(image, jpeg) {
    const tol = jpeg ? JPEG_TOLERANCES : EXACT_TOLERANCES;
    const B = backgroundColor(image).color;
    const cm = findColormap(image, B, flatMask(image, tol), tol);
    if (!cm)
        return null;
    const { image: virtual, P } = rayImage(image, B, cm);
    const measured = measureFamily(virtual, B, P, flatMask(virtual, tol), tol);
    const est = estimateStyle(virtual, { family: 0, candidates: [colormapCandidate(measured, B, P, cm.alpha)], jpeg });
    const note = `Dots colored by a continuous scale (${cm.points.length} colors sampled): each pixel is counted along its own direction from the background, so the fit sees one family. Outlines are fitted as part of the dot.`;
    return { ...est, image: virtual, notes: [note, ...est.notes] };
}

/**
 * @param {import('./image.js').RgbaImage} image  as analyzed (a JPEG's chroma already restored)
 * @param {object} [opts]
 * @param {boolean} [opts.jpeg]
 * @returns {{image: import('./image.js').RgbaImage, layers: object|null, estimates: object[]}}  image: with
 *   overlays removed; estimates: one per family candidate, each with the `image` to fit it on (estimates[0] is
 *   the only one, without a style, when no family is found)
 */
export function estimateFamilies(image, opts = {}) {
    const jpeg = !!opts.jpeg;
    const scaled = colormapEstimate(image, jpeg);
    if (scaled)
        return { image, layers: null, estimates: [scaled] };
    let first = estimateStyle(image, { jpeg });
    if (!first.candidates.length)
        return { image, layers: null, estimates: [{ ...first, image }] };
    const B = first.background.color;
    const radius = first.style?.radius ?? first.candidates.find(c => c.radius)?.radius ?? DEFAULT_RADIUS;
    const layers = backgroundRegions(image, B, first.candidates.map(candidateStyle), first.flat, radius);
    const overlays = layers.regions.filter(r => r.mode === 'over');
    let work = image;
    if (overlays.length) {
        work = removeOverlays(image, B, layers);
        first = estimateStyle(work, { jpeg });
    }
    const tol = jpeg ? JPEG_TOLERANCES : EXACT_TOLERANCES;
    const merged = mergeAcrossRegions(work, B, first.candidates, layers.regions, first.flat, tol);
    if (merged)
        first = estimateStyle(work, { family: 0, candidates: merged, jpeg });
    const estimates = [first];
    for (let k = 1; k < first.candidates.length; k++)
        estimates.push(estimateStyle(work, { family: k, candidates: first.candidates, jpeg }));
    const separated = unmixEstimates(work, estimates, tol, layers);
    if (overlays.length) {
        const note = overlays.map(r => `A translucent layer of color ${r.color.map(Math.round).join(', ')} covers part of the plot, letting ${Math.round(100 * r.s)}% of what's under it through; it was divided out before finding dots.`);
        for (const e of separated)
            e.notes = [...e.notes, ...note];
    }
    return { image: work, layers, estimates: separated };
}

// Every dot family of an image, ready to fit (PLAN.md phase 5): the style estimates, the shaded regions under or
// over the dots (background.js), overlays divided out, and each translucent family on its own image (unmix.js).
// The bench and the app both go through here, so they fit the same thing. Dots colored by a continuous scale are
// one family on a virtual image of their counts (colormap.js). Dots whose fills vary under a dark outline are fitted
// by their rings, in drawing order, with the fill colors projected out (outline.js).

import { estimateStyle, measureFamily, backgroundColor, flatMask, familyRings, JPEG_TOLERANCES, EXACT_TOLERANCES, SPECK_RADIUS_RATIO } from './style.js';
import { backgroundRegions, removeOverlays } from './background.js';
import { unmixEstimates } from './unmix.js';
import { findColormap, rayImage, opaqueImage, colormapCandidate } from './colormap.js';
import { findOutlines, findHueOutlines, fillFamilies, outlinesInP } from './outline.js';
import { backgroundColors } from './axes.js';

const T_BLACK = 255 * Math.sqrt(3); // t of black from white along the gray line: the outline fit's units
const OUTLINE_MASK_DILATE = 1; // px around what can't be read (P_bg above outline.js MASK_BG)
const DEFAULT_RADIUS = 5; // px, for the region test when no family has a radius yet
// The hue-outline reading (outline.js findHueOutlines, after colormap.js finds opaque fills on a scale) sees outlines
// darker than each fill; dots of a few discrete fills in one dark stroke look like that too (engagement1: five blues
// read as a scale fitted by rings found 22 of 49). Families
// found one at a time win when at least PER_FAMILY_MIN of them are stroked or outlined and their fills hold
// PER_FAMILY_INK_SHARE of the colored ink (engagement1: 4 families, 0.87; the gradient renders: none, 0).
const PER_FAMILY_MIN = 2;
const PER_FAMILY_INK_SHARE = 0.6;
const INK_MIN_CHROMA = 40; // 0–255, max − min channel: colored ink, not the page, text or a dark stroke
const FILL_TOL = 40; // 0–255 units (RGB distance) from a fill's color
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
                // outlined fills are dots of their own: grays over white and black always fit a − b = q (B − B₂)
                // (phase1: gray 187 and 82 in black outlines, with a black colorbar block for B₂)
                if (i === j || a.kind !== 'opaque' || b.kind !== 'opaque' || a.outline || b.outline)
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
 * The estimate for dots colored by a continuous scale, fitted on the virtual image (colormap.rayImage for
 * translucent dots, opaqueImage for opaque ones), or null when the image has no such family. `scale` marks it,
 * 'translucent' or 'opaque': its style's color is the virtual family's, which no dot has.
 */
function colormapEstimate(image, jpeg) {
    const tol = jpeg ? JPEG_TOLERANCES : EXACT_TOLERANCES;
    const B = backgroundColor(image).color;
    const cm = findColormap(image, B, flatMask(image, tol), tol);
    if (!cm)
        return null;
    if (cm.alpha === null) {
        const { image: virtual } = opaqueImage(image, B, cm);
        const est = estimateStyle(virtual, { jpeg });
        const note = `Opaque dots colored by a continuous scale (${cm.points.length} colors sampled): every dot pixel is counted as covered, so the fit sees one opaque family. Outlines are fitted as part of the dot, and edges between dots of different colors aren't used.`;
        return { ...est, image: virtual, scale: 'opaque', notes: [note, ...est.notes] };
    }
    const { image: virtual, P } = rayImage(image, B, cm);
    const measured = measureFamily(virtual, B, P, flatMask(virtual, tol), tol);
    const est = estimateStyle(virtual, { family: 0, candidates: [colormapCandidate(measured, B, P, cm.alpha)], jpeg });
    const note = `Dots colored by a continuous scale (${cm.points.length} colors sampled): each pixel is counted along its own direction from the background, so the fit sees one family. Outlines are fitted as part of the dot.`;
    return { ...est, image: virtual, scale: 'translucent', notes: [note, ...est.notes] };
}

/**
 * The estimate for outlined dots with fills that vary (outline.js), or null. Its image is P drawn in gray (what the
 * fit sees), its fitT and base are P and what lies under the rings in t units, and it brings its own mask.
 */
function outlineEstimate(image) {
    const background = backgroundColor(image);
    const found = findOutlines(image, background.color);
    return found && outlinedEstimate(image, background, found);
}

// what can't be read in P (D NaN), dilated by OUTLINE_MASK_DILATE: the fit's mask for outlined dots
function unreadMask(D, width, height) {
    const mask = new Uint8Array(width * height);
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            if (!Number.isNaN(D[y * width + x]))
                continue;
            for (let yy = Math.max(0, y - OUTLINE_MASK_DILATE); yy <= Math.min(height - 1, y + OUTLINE_MASK_DILATE); yy++) {
                for (let xx = Math.max(0, x - OUTLINE_MASK_DILATE); xx <= Math.min(width - 1, x + OUTLINE_MASK_DILATE); xx++)
                    mask[yy * width + xx] = 1;
            }
        }
    }
    return mask;
}

/**
 * An outlined opaque family, set up to be fitted by its rings in drawing order (outline.js's layered
 * model) instead of as a union of discs: in the union, fill and outline alike are "dot", so two dots of the family
 * overlapping are one blob, and the upper one's outline crossing the lower, which tells them apart, is invisible
 * (degradation1: 6 mint dots missed this way). The family keeps its fill color and outline for display; its ring
 * image (style.familyRings) gives the fit's image, base, coverage and mask, and its ring style. Fills are opaque, so
 * a fill drawn later hides the rings under it entirely. Only when its isolated rings fit the annulus as well as
 * outline.js asks of a whole image: where the rings are poorly measured, most of each ring is hidden or another
 * family's, and the union, which also reads the visible fill, places the dots better (degradation1's mint: 3 rings at
 * rms 0.095, and by rings it lost 5 of the dots the union found; the orange, 5 rings at 0.044, gained 3).
 * The estimate as it was otherwise.
 *
 * @param {import('./image.js').RgbaImage} image
 * @param {object} est  an opaque family's estimate with an outline
 * @param {number[][]} others  0–255, the other families' fill colors
 * @param {number[][]} backgrounds  0–255, the page's first
 * @returns {object}  the estimate
 */
function ringEstimate(image, est, others, backgrounds) {
    const fam = est.familyMeasure;
    const outline = fam.outline;
    const { width, height } = image;
    const n = width * height;
    const { P, unknown } = familyRings(image, est.style.color, outline, backgrounds, others);
    const found = outlinesInP(P, width, height, unknown);
    if (!found)
        return est;
    const { ring, under, D } = found;
    const fitT = new Float32Array(n);
    const base = new Float32Array(n);
    for (let p = 0; p < n; p++) {
        fitT[p] = T_BLACK * P[p];
        base[p] = T_BLACK * Math.min(under[p], 1);
    }
    return {
        ...est,
        familyMeasure: { ...fam, line: { B: [255, 255, 255], u: [1, 1, 1].map(v => v / Math.sqrt(3)) } },
        style: {
            ...est.style, alpha: 0, radius: ring.radius, blur: ring.blur,
            ring: { inner: ring.inner, alpha: ring.alpha },
            layered: { fillAlpha: 1, ringT: T_BLACK },
            outline: { color: [...outline], inner: ring.inner },
        },
        D, fitT, base, outlineMask: unreadMask(D, width, height),
        isolated: found.isolated.map(c => ({ cx: c.x, cy: c.y, r: c.r })),
        // the rings gave the radius: the union's note about having no isolated whole dot no longer applies
        notes: [...est.notes.filter(t => !t.startsWith('No isolated whole dot')), `Fitted by its outlines in drawing order: an outline crossing another dot of the family is the edge of the dot on top. Ring from ${(ring.inner * ring.radius).toFixed(2)} to ${ring.radius.toFixed(2)} px.`],
    };
}

// the estimate from outline.findOutlines or findHueOutlines
function outlinedEstimate(image, background, found) {
    const { width, height } = image;
    const { P, under, D, ring } = found;
    const n = width * height;
    const data = new Uint8ClampedArray(4 * n);
    const fitT = new Float32Array(n);
    const base = new Float32Array(n);
    for (let p = 0; p < n; p++) {
        const v = Math.round(255 * (1 - Math.min(1, Math.max(0, P[p]))));
        data[4 * p] = data[4 * p + 1] = data[4 * p + 2] = v;
        data[4 * p + 3] = 255;
        fitT[p] = T_BLACK * P[p];
        base[p] = T_BLACK * Math.min(under[p], 1);
    }
    const mask = unreadMask(D, width, height);
    const style = {
        color: [0, 0, 0], alpha: 0, radius: ring.radius, blur: ring.blur, measuredLevels: 1,
        ring: { inner: ring.inner, alpha: ring.alpha },
        layered: { fillAlpha: found.fillAlpha, ringT: T_BLACK },
    };
    const line = { B: [255, 255, 255], u: [1, 1, 1].map(v => v / Math.sqrt(3)) };
    // discrete fill colors (two groups under gray rings), for dotFamilies after the fit; null for a scale or one
    const oneColor = found.plane?.oneColor ?? false;
    const fills = !found.plane || oneColor ? null : fillFamilies(image);
    const notes = [
        !found.plane
            ? 'Outlined dots whose outlines are a darker shade of their own fills, colored by a scale: each pixel is read as how much darker it is than the fill around it, so the fills drop out, and the dots are fitted by their rings, in drawing order.'
            : fills
            ? `Outlined dots of ${fills.length} fill colors (${fills.map(c => c.map(Math.round).join(', ')).join('; ')}): the fills are projected out, the dots are fitted by their rings in drawing order, and each dot's color is read from its fill where it shows.`
            : found.plane.oneColor
            ? `Outlined dots of one fill color (${found.plane.pixels} fill pixels): the fill is projected out and the dots are fitted by their rings, in drawing order.`
            : `Outlined dots whose fills vary (${found.plane.pixels} fill pixels within ${found.plane.rms.toFixed(1)} units of one plane): the fill colors are projected out and the dots are fitted by their rings, in drawing order.`,
        `Ring from ${(ring.inner * ring.radius).toFixed(2)} to ${ring.radius.toFixed(2)} px, ${found.plane ? `${Math.round(100 * ring.alpha)}% of the way to black` : `darker than its fill by ${Math.round(100 * ring.alpha)}% of white's distance from black (the median of the isolated rings)`}; fills ${Math.round(100 * found.fillAlpha)}% opaque${found.fillAlphaMeasured ? ` (from ${found.pairs} overlapping pairs)` : ' (assumed: too few overlapping pairs to measure)'}.`,
    ];
    return {
        background, family: 0, flat: null, image: { width, height, data },
        // one candidate, so the app fits one family
        candidates: [{ kind: 'outline', radius: ring.radius, levels: [], fit: { alpha: ring.alpha }, oneColor, fillColors: fills?.length ?? 0 }],
        familyMeasure: { kind: 'outline', line, levels: [] },
        style, D, fitT, base, outlineMask: mask,
        fillFamilies: fills,
        isolated: found.isolated.map(c => ({ cx: c.x, cy: c.y, r: c.r })),
        edge: null, scale: 'outline', notes,
    };
}

// whether stroked or outlined families found one at a time account for the image's colored dots (see PER_FAMILY_MIN)
function familiesExplainInk(image, candidates) {
    const fills = candidates.filter(c => c.kind === 'opaque' && (c.stroke || c.outline)).map(c => c.seed);
    if (fills.length < PER_FAMILY_MIN)
        return false;
    const { data } = image;
    let ink = 0;
    let inFill = 0;
    for (let p = 0; p < data.length; p += 4) {
        const r = data[p];
        const g = data[p + 1];
        const b = data[p + 2];
        if (Math.max(r, g, b) - Math.min(r, g, b) < INK_MIN_CHROMA)
            continue;
        ink++;
        if (fills.some(c => Math.hypot(r - c[0], g - c[1], b - c[2]) <= FILL_TOL))
            inFill++;
    }
    return ink > 0 && inFill >= PER_FAMILY_INK_SHARE * ink;
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
    // a dark ring off the fill plane decides for outlined dots even when the fills also read as a translucent
    // scale (their overlaps show levels too); JMP's outlines, in the fill's own hue, never pass
    const outlined = outlineEstimate(image);
    if (outlined)
        return { image, layers: null, estimates: [outlined] };
    let first = null;
    const scaled = colormapEstimate(image, jpeg);
    if (scaled?.scale === 'opaque') {
        // opaque fills on a scale whose outlines are a darker shade of each fill: the rings place the dots where
        // the union of discs can't (bench2 gradient-opaque-stroke's dense stacks); unless a few stroked families
        // found one at a time explain the fills (engagement1: five blues in one dark stroke)
        const hue = findHueOutlines(image);
        if (hue) {
            first = estimateStyle(image, { jpeg });
            if (!familiesExplainInk(image, first.candidates))
                return { image, layers: null, estimates: [outlinedEstimate(image, backgroundColor(image), hue)] };
        }
    }
    if (scaled && !first)
        return { image, layers: null, estimates: [scaled] };
    first ??= estimateStyle(image, { jpeg });
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
    const kept = withoutSpeckLevels(separated);
    // outlined opaque families are fitted by their rings (ringEstimate)
    const backgrounds = backgroundColors(work);
    const fills = kept.filter(e => e.style).map(e => e.style.color);
    // outlines only: a stroke too thin to show its own color is a ring the image barely resolves (engagement1, strokes
    // under a pixel wide: 54 dots for 49 by the union, 40 by rings)
    const ringed = kept.map(e => (e.familyMeasure?.kind === 'opaque' && e.familyMeasure.outline && e.style
        ? ringEstimate(work, e, fills.filter(c => c !== e.style.color), backgrounds)
        : e));
    return { image: work, layers, estimates: ringed };
}

// A translucent family's radius is known only now: one far smaller than the others is text's antialiased grays
// read as levels, its isolated "dots" decimal points (engagement1: black at r 3.5 beside fills at r 14.3, 1362 dots).
// Opaque specks went earlier (style.js withoutSpecks), with their coverage.
function withoutSpeckLevels(estimates) {
    const largest = Math.max(...estimates.map(e => e.style?.radius ?? 0));
    // a family whose radius wasn't measured stays: the user can type it
    const kept = estimates.filter(e => e.familyMeasure?.kind !== 'translucent' || !(e.style?.radius > 0) || e.style.radius >= SPECK_RADIUS_RATIO * largest);
    return kept.length ? kept : estimates;
}

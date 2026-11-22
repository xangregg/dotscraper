// Every dot family of an image, ready to fit (PLAN.md phase 5): the style estimates, the shaded regions under or
// over the dots (background.js), overlays divided out, and each translucent family on its own image (unmix.js).
// The bench and the app both go through here, so they fit the same thing. Dots colored by a continuous scale are
// one family on a virtual image of their counts (colormap.js). Dots whose fills vary under a dark outline are fitted
// by their rings, in drawing order, with the fill colors projected out (outline.js).

import { colorDistance, sub } from './image.js';
import { scanOf, backgroundColor, flatMask, solidRegion } from './scan.js';
import { measureFamily, alongLine } from './levels.js';
import { familyRings, siblingRadius, SPECK_RADIUS_RATIO } from './opaque.js';
import { estimateStyle, NO_ISOLATED_NOTE } from './style.js';
import { backgroundRegions, removeOverlays } from './background.js';
import { unmixEstimates } from './unmix.js';
import { findColormap, rayImage, opaqueImage, colormapCandidate } from './colormap.js';
import { findOutlines, findHueOutlines, fillFamilies, outlinesInP, findHollowRings } from './outline.js';

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
// The scan's stages and the share of its time each takes, for a progress bar (measured on 11 bench images of 0.2–58
// s: the dots' colors 0.31–0.97 of the time, outlines and hollow rings 0.18–0.49, the dots' colors over the regions
// 0.20–0.32 where they run; score2, dots on a color scale, 0.68 checking for the scale, most of it the outlines in
// its fills). A stage that doesn't run is passed over, so the bar can jump.
const SCAN_STAGES = [
    { name: 'outlines', label: 'Looking for outlined dots', weight: 0.01 },
    { name: 'scale', label: 'Checking for a color scale', weight: 0.04 },
    { name: 'hue outlines', label: 'Looking for outlines around the scale’s fills', weight: 0.12 },
    { name: 'colors', label: 'Finding the dots’ colors', weight: 0.38 },
    { name: 'regions', label: 'Finding shaded regions', weight: 0.05 },
    { name: 'colors again', label: 'Finding the dots’ colors over the regions', weight: 0.12 },
    { name: 'families', label: 'Measuring each family', weight: 0.06 },
    { name: 'rings', label: 'Measuring outlines and rings', weight: 0.22 },
];

// a candidate's full color and α, as background.js wants them
function candidateStyle(c) {
    return c.kind === 'opaque' ? { color: c.seed, alpha: 1 } : { color: c.fit.c, alpha: c.fit.alpha, ...(c.band ? { band: c.band, box: !!c.box } : {}) };
}

/**
 * Candidates that are one translucent family seen over the main background B and over a region under the dots:
 * an opaque candidate that is a translucent candidate's dot over a region color is dropped, and two opaque
 * candidates whose colors differ by q·(B − B₂) become one translucent candidate with α = 1 − q.
 *
 * @param {import('./scan.js').Scan} scan  of the image the candidates were found in
 * @param {number[]} B  the background the regions were told from (the page, or the first family's)
 * @param {import('./style.js').FamilyMeasure[]} candidates  the families found
 * @param {{color: number[], mode: string}[]} regions  background.js backgroundRegions': shaded regions, 'under' or
 *   'over' the dots
 * @returns {import('./style.js').FamilyMeasure[]|null}  the new candidates, or null if nothing merged
 */
function mergeAcrossRegions(scan, B, candidates, regions) {
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
            return colorDistance(c.seed, level1) <= MERGE_TOL;
        }));
        changed ||= covered;
        return !covered;
    });
    // two opaque clusters that are one translucent family over B and over B₂
    for (const B2 of under) {
        const d = sub(B, B2);
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
                const { m: q, off } = alongLine(diff, d);
                if (q < MERGE_Q[0] || q > MERGE_Q[1] || off > MERGE_TOL)
                    continue;
                const c = a.seed.map((v, ch) => (v - q * B[ch]) / (1 - q));
                const m = measureFamily({ ...scan, B }, a.seed);
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
function colormapEstimate(scan, jpeg) {
    const { image, B, tol } = scan;
    const cm = findColormap(image, B, scan.flat, tol);
    if (!cm)
        return null;
    // a categorical palette is fitted the same way; each dot's color is read from its own pixels after the fit
    // (colormap.js paletteColors), and splits the dots into families
    const colored = cm.palette ? 'Dots of many distinct colors (a categorical palette, not a scale)' : `Dots colored by a continuous scale (${cm.points.length} colors sampled)`;
    const after = cm.palette ? ' After the fit, each dot\'s color is read where no other dot covers it, and the dots are split by color; a dot mostly covered is marked unsure.' : '';
    if (cm.alpha === null) {
        const { image: virtual } = opaqueImage(image, B, cm);
        const est = estimateStyle(virtual, { jpeg });
        const note = `Opaque ${colored.charAt(0).toLowerCase()}${colored.slice(1)}: every dot pixel is counted as covered, so the fit sees one opaque family. Outlines are fitted as part of the dot, and edges between dots of different colors aren't used.${after}`;
        return { ...est, image: virtual, scale: 'opaque', palette: cm.palette, notes: [note, ...est.notes] };
    }
    const { image: virtual, P } = rayImage(image, B, cm);
    const measured = measureFamily({ image: virtual, B, flat: flatMask(virtual, tol), tol }, P);
    const est = estimateStyle(virtual, { family: 0, candidates: [colormapCandidate(measured, B, P, cm.alpha)], jpeg });
    const note = `${colored}: each pixel is counted along its own direction from the background, so the fit sees one family. Outlines are fitted as part of the dot.${after}`;
    return { ...est, image: virtual, scale: 'translucent', palette: cm.palette, notes: [note, ...est.notes] };
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
 * @param {import('./style.js').Estimate} est  an opaque family's estimate with an outline
 * @param {number[][]} others  0–255, the other families' fill colors
 * @param {number[][]} backgrounds  0–255, the page's first
 * @returns {import('./style.js').Estimate}  set up for its rings (style.ring and layered, D, fitT, base,
 *   outlineMask), or `est` itself
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

// Hollow-ring families (outline.findHollowRings), fitted by their rings in drawing order as outlined dots are, with
// nothing inside. `colors`: the found families' colors (fills, outlines, strokes), whose own thin ink is no ring.
function hollowEstimates(image, background, colors) {
    const B = background.color;
    return findHollowRings(image, B, colors).map(found => {
        const est = outlinedEstimate(image, background, found);
        const { ring, hollow } = found;
        est.style.color = hollow.color;
        est.candidates[0].hollow = true;
        // (the fit sees the ring ink in gray, as for other outlined dots)
        est.style.fitColor = [0, 0, 0];
        est.notes = [
            `Hollow rings (${hollow.color.join(', ')} at their darkest on the page): each pixel is read as how much darker it is than what is around it, in the rings' own color, so other ink drops out, and the rings are fitted in drawing order, nothing inside them.`,
            `Ring from ${(ring.inner * ring.radius).toFixed(2)} to ${ring.radius.toFixed(2)} px, darker than what is under it by ${Math.round(100 * ring.alpha)}% of white's distance from black (the median of ${found.isolated.length} isolated rings).`,
        ];
        return est;
    });
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
 * An image's dot families. Three readings come first, each of which can settle the image and end the scan:
 * outlined dots whose fills vary (outlineEstimate); dots colored by a scale (scaleReading); and, when no family is
 * found by its color (estimateStyle), hollow rings. Then, for the families found by color: the shaded regions over
 * and under the dots (shadedRegions), each family's estimate on the image without the layers over it
 * (familyEstimates), the outlined ones fitted by their rings (withRingFits), and hollow rings in colors no family
 * has.
 *
 * @param {import('./image.js').RgbaImage} image  as analyzed (a JPEG's chroma already restored)
 * @param {object} [opts]
 * @param {boolean} [opts.jpeg]
 * @param {(label: string, fraction: number) => void} [opts.onProgress]  what is being done and the share of the
 *   whole done (0–1, by SCAN_STAGES; stages that don't run are skipped over)
 * @returns {{image: import('./image.js').RgbaImage, layers: object|null, estimates: import('./style.js').Estimate[]}}
 *   image: with overlays removed; layers: background.js backgroundRegions' (the shaded regions and each pixel's), or
 *   null when a reading settled the image before them; estimates: one per family candidate, each with the `image`
 *   to fit it on (estimates[0] is
 *   the only one, without a style, when no family is found)
 */
export function estimateFamilies(image, opts = {}) {
    const jpeg = !!opts.jpeg;
    const stage = scanStages(opts.onProgress);
    const settled = estimates => ({ image, layers: null, estimates });
    // a dark ring off the fill plane decides for outlined dots even when the fills also read as a translucent
    // scale (their overlaps show levels too); JMP's outlines, in the fill's own hue, never pass
    stage('outlines');
    const outlined = outlineEstimate(image);
    if (outlined)
        return settled([outlined]);
    stage('scale');
    // the page, the flat mask and what is asked about the image's colors, once for every estimate of this image
    const scan = scanOf(image, { jpeg });
    const scaled = scaleReading(image, scan, jpeg, stage);
    if (scaled.estimate)
        return settled([scaled.estimate]);
    let first = scaled.first ?? estimateStyle(image, { jpeg, scan, onProgress: stage('colors') });
    if (!first.candidates.length) {
        const hollow = hollowEstimates(image, first.background, []);
        return settled(hollow.length ? hollow : [{ ...first, image }]);
    }
    const radius = first.style?.radius ?? first.candidates.find(c => c.radius)?.radius ?? DEFAULT_RADIUS;
    stage('regions');
    const { B, layers, overlays, regions } = shadedRegions(image, scan, first, radius);
    const work = overlays.length ? removeOverlays(image, B, layers) : image;
    const workScan = work === image ? scan : scanOf(work, { jpeg });
    if (overlays.length || regions)
        first = estimateStyle(work, { jpeg, regions, scan: workScan, onProgress: stage('colors again') });
    const each = stage('families');
    const merged = mergeAcrossRegions(workScan, B, first.candidates, layers.regions);
    if (merged)
        first = estimateStyle(work, { family: 0, candidates: merged, jpeg, scan: workScan });
    const separated = familyEstimates(work, workScan, first, layers, jpeg, each);
    const ring = stage('rings');
    const ringed = withRingFits(work, workScan, withoutSpeckLevels(withSiblingRadii(work, separated)), ring);
    // hollow rings of a color no family found has (its fill, outline or stroke)
    const known = ringed.filter(e => e.style).flatMap(e => [e.style.color, e.familyMeasure?.outline, e.familyMeasure?.stroke].filter(Boolean));
    const hollow = hollowEstimates(work, first.background, known);
    stage('done');
    return { image: work, layers, estimates: [...ringed, ...hollow] };
}

// Dots colored by a scale: `estimate` when the scale settles the image, `first` (the families by their colors, which
// the caller needs next) when a few families explain it instead, neither when there is no scale.
function scaleReading(image, scan, jpeg, stage) {
    const scaled = colormapEstimate(scan, jpeg);
    if (!scaled)
        return {};
    if (scaled.scale === 'opaque') {
        // opaque fills on a scale whose outlines are a darker shade of each fill: the rings place the dots where
        // the union of discs can't (bench2 gradient-opaque-stroke's dense stacks); unless a few stroked families
        // found one at a time explain the fills (engagement1: five blues in one dark stroke)
        stage('hue outlines');
        const hue = findHueOutlines(image);
        if (hue) {
            const first = estimateStyle(image, { jpeg, scan, onProgress: stage('colors') });
            return familiesExplainInk(image, first.candidates) ? { first } : { estimate: outlinedEstimate(image, scan.background, hue) };
        }
    }
    return { estimate: scaled };
}

// The shaded regions of the image: `layers` (background.js), the color B they were told from, the `overlays` (layers
// over the dots, to divide out) and `regions` (the colors of solid regions under opaque dots, or undefined).
//
// A layer over the dots is divided out against the page, so regions are first told from the page. An opaque
// family's background is its dots' surroundings, and a band most of them lie in (kite1: the red dots inside the
// red band, 254, 227, 225) made the page a "region" under the dots and the band none, so the cyan dots under it
// were never seen. Without a layer over the dots, regions are told from the first family's background as before:
// the regions under the dots are read against their own colors either way, and the fits were tuned with it
// (from the page, violin-beeswarm's violins under the dots gave 18 more extras, P 0.985 to 0.960).
function shadedRegions(image, scan, first, radius) {
    const page = scan.B;
    let B = page;
    let layers = backgroundRegions(image, B, first.candidates.map(candidateStyle), first.flat, radius);
    if (!layers.regions.some(r => r.mode === 'over') && first.background.color.some((v, ch) => v !== page[ch])) {
        B = first.background.color;
        layers = backgroundRegions(image, B, first.candidates.map(candidateStyle), first.flat, radius);
    }
    const overlays = layers.regions.filter(r => r.mode === 'over');
    // Opaque dots over a shaded region are read against its color, as against the page; otherwise the region is
    // other ink and the mask around it takes the dots' edges (boxdots1: black dots over a red box, recall 0.74)
    const under = layers.regions.filter(r => r.mode === 'under' && solidRegion(scan, r.color, radius)).map(r => r.color);
    const regions = under.length && first.candidates.some(c => c.kind === 'opaque') ? under : undefined;
    return { B, layers, overlays, regions };
}

// One estimate per family candidate (`first` is the first family's), overlapping translucent families separated
// (unmix.js), each with a note of the layers divided out of the image. `each` gets the share of the families done.
function familyEstimates(work, workScan, first, layers, jpeg, each) {
    const estimates = [first];
    for (let k = 1; k < first.candidates.length; k++) {
        each(k / first.candidates.length);
        estimates.push(estimateStyle(work, { family: k, candidates: first.candidates, jpeg, scan: workScan }));
    }
    const separated = unmixEstimates(work, estimates, workScan.tol, layers);
    const overlays = layers.regions.filter(r => r.mode === 'over');
    if (overlays.length) {
        const rgb = c => c.map(Math.round).join(', ');
        const note = overlays.map(r => (r.stack
            ? `Where the layer of color ${rgb(layers.regions[r.stack[0]].color)} lies over that of ${rgb(layers.regions[r.stack[1]].color)} (${rgb(r.color)}), both were divided out, in that order.`
            : `A translucent layer of color ${rgb(r.color)} covers part of the plot, letting ${Math.round(100 * r.s)}% of what's under it through${r.stackedOver !== undefined ? `, and lies over the layer of ${rgb(layers.regions[r.stackedOver].color)}` : ''}; it was divided out before finding dots.`));
        for (const e of separated)
            e.notes = [...e.notes, ...note];
    }
    return separated;
}

// Outlined opaque families are fitted by their rings (ringEstimate). `ring` gets the share done.
function withRingFits(work, workScan, kept, ring) {
    const backgrounds = workScan.pageBackgrounds;
    // (a fill in the page's color is no other family's ring owner: it can't be told from the page)
    const fills = kept.filter(e => e.style && !e.familyMeasure?.pageFill).map(e => e.style.color);
    // outlines only: a stroke too thin to show its own color is a ring the image barely resolves (engagement1, strokes
    // under a pixel wide: 54 dots for 49 by the union, 40 by rings)
    // (not a fill in the page's color: its ring image would read the whole page as fill)
    // (the hollow rings after them take about as long as one family's)
    const ringed = kept.map((e, k) => {
        ring(k / (kept.length + 1));
        return e.familyMeasure?.kind === 'opaque' && e.familyMeasure.outline && !e.familyMeasure.pageFill && e.style
            ? ringEstimate(work, e, fills.filter(c => c !== e.style.color), backgrounds)
            : e;
    });
    ring(kept.length / (kept.length + 1));
    return ringed;
}

// estimateFamilies' progress: stage(name) reports the stage's label at the share of the whole where it starts, and
// returns a reporter of the share of it done (0–1). Stages are in order; one that doesn't run is passed over.
function scanStages(onProgress) {
    const report = onProgress ?? (() => {});
    const at = name => SCAN_STAGES.slice(0, SCAN_STAGES.findIndex(s => s.name === name)).reduce((t, s) => t + s.weight, 0);
    return name => {
        const { label = '', weight = 0 } = SCAN_STAGES.find(s => s.name === name) ?? {};
        const start = name === 'done' ? 1 : at(name);
        report(label, start);
        return f => report(label, start + weight * f);
    };
}

// A translucent family with no radius of its own takes another's when its dots show at it (opaque.js siblingRadius).
function withSiblingRadii(image, estimates) {
    const radii = estimates.map(e => e.style?.radius).filter(r => r > 0);
    return estimates.map(e => {
        if (e.familyMeasure?.kind !== 'translucent' || !e.style || e.style.radius || !e.D)
            return e;
        const sib = siblingRadius(e.D, image.width, image.height, radii);
        if (!sib)
            return e;
        const note = `No isolated whole dot: the radius, ${sib.radius.toFixed(1)} px, is another family's, at which ${sib.centers} of these dots show half their edge or more; the blur is read from the dots' boundaries. Check it against a few dots.`;
        return { ...e, style: { ...e.style, radius: sib.radius, blur: sib.blur }, notes: [...e.notes.filter(n => n !== NO_ISOLATED_NOTE), note] };
    });
}

// A translucent family's radius is known only now: one far smaller than the others is text's antialiased grays
// read as levels, its isolated "dots" decimal points (engagement1: black at r 3.5 beside fills at r 14.3, 1362 dots).
// Opaque specks went earlier (opaque.js withoutSpecks), with their coverage.
function withoutSpeckLevels(estimates) {
    const largest = Math.max(...estimates.map(e => e.style?.radius ?? 0));
    // a family whose radius wasn't measured stays: the user can type it
    const kept = estimates.filter(e => e.familyMeasure?.kind !== 'translucent' || !(e.style?.radius > 0) || e.style.radius >= SPECK_RADIUS_RATIO * largest);
    return kept.length ? kept : estimates;
}

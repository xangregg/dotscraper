// Dot style estimation (PLAN.md §2.2, stage 4): the dot families of an image and one family's style (color, alpha,
// levels, radius, edge blur), ready for the fit. estimateStyle and findCandidates put together the translucent
// families (levels.js) and the opaque ones (opaque.js), with the rules that tell bands and boxes under the dots from
// the dots' own levels; scan.js has the Scan they pass around, shape.js the measures of dots as shapes.

import { distanceTransform } from './layers.js';
import { nearColor, colorDistance } from './image.js';
import { LEVEL_SEPARATION, CLUSTER_TOL, BOX_COLOR_TOL, isThin, scanOf, solidRegion, straightShare } from './scan.js';
import { components, isolatedDots, measureRing, edgeBlur, boundaryBlur } from './shape.js';
import { median } from './stats.js';
import {
    FIT_C_TOL, alongLine, colorLine, projectOnLine, levelTable, countImage, familyCandidates, measureFamily,
} from './levels.js';
import {
    CLUSTER_ON_LINE_TOL, DOT_MIN_ISOLATED, STROKE_RING, measureOpaque, isDotFamily, opaqueFamilies, strokedFamily,
} from './opaque.js';

/** @typedef {import('./scan.js').Scan} Scan */

const RINGED_MIN_SHARE = 0.5; // of the isolated dots, for the family to be outlined
// A translucent "family" whose levels are all stacks of two other families' dots is their overlaps (bench2
// group-large, α 0.33: red over blue and blue over red as two purple families with no isolated dot). Measured: the
// blends' levels within 1.3 of a stack, the real families' 86 or more away.
const BLEND_TOL = 12; // 0–255 units
const OVER_BAND_Q = [0.05, 0.95]; // 1 − α for a family's dots over a band of their hue (overBand)
const BLEND_DEPTH = 4; // dots in a stack
// findCandidates' progress: the share of the work before the opaque clusters (measured 0.07–0.28 on five bench
// images, 0.73 on protrusion1, whose four translucent families each test their colors for a box)
const TRANSLUCENT_SHARE = 0.25;
// A box under the dots (boxBand): straight boundary runs of BOX_RUN_RADII dot radii hold BOX_MIN_STRAIGHT of its
// boundary. Measured on every translucent candidate's levels on the bench: dots' levels 0 (a few grays at radius under
// 4, axis text: up to 0.063); boxes 0.145–0.274 (protrusion1–3), and score2's colorbar 0.124.
const BOX_RUN_RADII = 2; // × the dots' radius
const BOX_MIN_STRAIGHT = 0.08;
// Before any radius is known (beyondBox), a box is told from the dots by comparison: straighter than the first level
// by this factor in runs of BEYOND_BOX_RUN (protrusion1: 3.5–4.0; landing1's violins, a band under the dots, are the
// straighter, 0.155 to 0.000)
const BEYOND_BOX_RUN = 20; // px
const BEYOND_BOX_RATIO = 2;
// A box is an area, not strokes (landing1: black axes, box outlines and text beyond a gray of their antialiasing are
// straight too): solidRegion at this radius (measured: the protrusion boxes keep 0.92–0.99, the black 0.000)
const BOX_MIN_HALF_WIDTH = 4; // px
// Two dots overlap inside each dot, so a translucent family's level 2 lies in level 1: its pieces have level 1's flat
// pixels close by (flat pixels stop a pixel or two short of an edge on each side). A gray of text on the line to
// black dots is no level 1 (molar1: the "K_d" label at 62, 62, 62 and black dots as level 2, α 0.78, no dot whole).
// Real families had 9–100% of their pieces touching; text's levels 0%, over 73–281 pieces.
const NEST_REACH = 6; // px from a level-2 piece to level 1's flat pixels
const NEST_MIN_PIECES = 20; // level-2 pieces of NEST_MIN_PIECE_AREA px needed to judge (real families at 0: 7 or fewer)
const NEST_MIN_PIECE_AREA = 3; // px
const NEST_MIN_SHARE = 0.05; // of the pieces near level 1

/** The note on a translucent family whose radius no isolated dot gave. */
export const NO_ISOLATED_NOTE = 'No isolated whole dot found, so the radius and edge blur must be set by hand.';

/**
 * An opaque family measured as the user describes it (the app's outline edits): its fill color, its outline or
 * none, and the other families' fills. With rings to others, a ring pixel nearer another fill is that dot's, as in
 * opaqueFamilies' final pass, so every family can be measured again when one's outline changes.
 *
 * @param {import('./image.js').RgbaImage} image
 * @param {number[]} seed  the fill, 0–255
 * @param {object} [opts]
 * @param {{color: number[], fillRadius: number}|null} [opts.stroke]  0–255, px
 * @param {number[]|null} [opts.outline]  a JMP-style outline color (measureOpaque), kept from the estimate
 * @param {number[][]} [opts.others]  the other families' fills, 0–255
 * @param {boolean} [opts.ringsToOthers]
 * @param {boolean} [opts.jpeg]
 * @returns {object}  an estimate, as estimateStyle's; style.radius is null when no dot shape shows
 */
export function opaqueFamilyEstimate(image, seed, opts = {}) {
    const scan = scanOf(image, opts);
    const others = opts.others ?? [];
    const stroke = opts.stroke ?? null;
    const rings = opts.ringsToOthers && others.length ? others.map(fill => ({ fill, reach: STROKE_RING })) : null;
    const fam = measureOpaque(scan, seed, others, stroke ? null : opts.outline ?? null,
        { stroke: stroke?.color ?? null, fillRadius: stroke?.fillRadius, rings });
    const measured = { ...fam, fillRadius: stroke?.fillRadius ?? fam.fillRadius };
    return opaqueEstimate(scan, measured, [measured], 0);
}

/**
 * Estimate the style of one translucent dot family.
 *
 * @param {import('./image.js').RgbaImage} image
 * @param {object} [opts]
 * @param {number[]} [opts.seed]    a color of the family (e.g. picked by the user); default: the best candidate
 * @param {number} [opts.family]    index into the candidates, instead of a seed
 * @param {boolean} [opts.jpeg]     the image is JPEG-compressed: noise-tolerant thresholds (JPEG_TOLERANCES); candidates
 *   passed in must have been found with the same setting
 * @param {number[][]} [opts.regions]  0–255, colors of shaded regions under the dots (background.js), which opaque
 *   dots are read against as against the page
 * @param {Scan} [opts.scan]  a scan of this image from an earlier call (scanOf, with the same `jpeg`): its page, flat
 *   mask and remembered answers are used again
 * @param {(fraction: number) => void} [opts.onProgress]  the share of the candidate search done (0–1)
 * @returns {object}  { background, candidates, family, style, D, isolated, edge, notes }
 *   style: { color, alpha, levels, radius, blur } ready for render.js; radius and blur are null when no isolated
 *   dot was found. notes: human-readable caveats for display.
 */
export function estimateStyle(image, opts = {}) {
    const { width, height } = image;
    const made = opts.scan?.image === image ? opts.scan : scanOf(image, opts);
    const scan = opts.regions ? { ...made, backgrounds: [...made.pageBackgrounds, ...opts.regions] } : made;
    const { background, B, tol, flat } = scan;
    // candidates are the slow part (seconds on large images): reuse them when estimating another family
    const candidates = opts.candidates ?? findCandidates(scan, opts.onProgress);
    let fam;
    let familyIndex = opts.family ?? 0;
    if (opts.seed) {
        fam = measureFamily(scan, opts.seed);
        fam.kind = 'translucent';
        if (!fam.translucent) {
            const plain = measureOpaque(scan, opts.seed);
            fam = isDotFamily(plain) ? plain : strokedFamily(scan, opts.seed, plain) ?? plain;
        }
        familyIndex = -1;
    }
    else {
        fam = candidates[familyIndex];
    }
    const notes = [];
    if (!fam || !fam.levels.length)
        return { background, candidates, family: familyIndex, style: null, D: null, isolated: [], edge: null, notes: ['No dot family found.'] };
    if (fam.kind === 'opaque')
        return opaqueEstimate(scan, fam, candidates, familyIndex);
    const table = levelTable(B, fam.levels, fam.fit);
    const D = countImage(fam.proj, fam.line, table, tol);
    const isolated = isolatedDots(D, width, height, tol);
    let radius = null;
    let blur = null;
    let edge = null;
    let ring = null;
    if (isolated.length) {
        radius = median(isolated.map(d => d.r));
        edge = edgeBlur(D, width, isolated);
        // the profile's σ includes the pixel box, which the renderer's exact coverage already supplies
        blur = Math.sqrt(Math.max(0, edge.sigma ** 2 - 1 / 12));
        const ringed = isolated.filter(d => d.ringed);
        if (ringed.length >= RINGED_MIN_SHARE * isolated.length) {
            const { B: lb, u } = fam.line;
            const tC = (lb[0] - fam.fit.c[0]) * u[0] + (lb[1] - fam.fit.c[1]) * u[1] + (lb[2] - fam.fit.c[2]) * u[2];
            ring = measureRing(fam.proj.t, width, ringed, radius, fam.fit.alpha, tC);
            if (ring)
                notes.push(`Outlined dots: a ring from ${(ring.inner * radius).toFixed(1)} to ${radius.toFixed(1)} px, ${Math.round(100 * (fam.fit.alpha + ring.alpha))}% opaque against ${Math.round(100 * fam.fit.alpha)}% inside, drawn in the fill's color line (an approximation where outlines cross other dots).`);
        }
    }
    else {
        notes.push(NO_ISOLATED_NOTE);
    }
    const inferredLevels = fam.levels.flatMap((L, i) => L.inferred ? [i + 1] : []);
    if (inferredLevels.length)
        notes.push(`Too few flat pixels show ${inferredLevels.length > 1 ? 'levels' : 'level'} ${inferredLevels.join(', ')} (overlap count), so its color was inferred from the levels around it.`);
    return {
        background, candidates, family: familyIndex, flat,
        familyMeasure: fam,
        style: { color: fam.fit.c, alpha: fam.fit.alpha, levels: table, measuredLevels: fam.levels.length, inferredLevels, radius, blur, ...(ring ? { ring } : {}) },
        D, isolated, edge, notes,
    };
}

/**
 * All dot families: translucent ones (levels along a color direction) first, then opaque ones (color clusters).
 * `onProgress` gets the share of the work done (0–1), by stage: the opaque clusters are most of it on large images.
 *
 * @param {Scan} scan
 */
export function findCandidates(scan, onProgress = null) {
    const { B } = scan;
    // A translucent family's first level fills the insides of single dots, so it is thick. A first level that is
    // only lines is a line's color that happens to lie on the line to an opaque family's color (microplastics1: a
    // 3 px border bar at 48 and black dots read as levels 1 and 3, α 0.81, with no isolated dot, which hid the black
    // dots as an opaque family).
    // A candidate whose level 2 lies outside its level 1 is no family (levelsNest); its colors stay other ink for the
    // opaque families, as when it was dropped later as text (engagement1: black text and the blues' strokes; without
    // them the blues' fit gave 55 dots for 49, P 0.873)
    const nesting = familyCandidates(scan).map(c => beyondBox(scan, c) ?? overBand(scan, c) ?? c)
        .filter(c => c.translucent && !isThin(scan, c.levels[0].color))
        .map(c => withoutBand(scan, c));
    const measured = nesting.filter(c => levelsNest(scan, c));
    const ink = nesting.filter(c => !measured.includes(c)).flatMap(c => c.levels.filter(L => !L.inferred).map(L => L.color));
    // A candidate without a lone dot whose first level is a solid area at the other families' radius is a band, with
    // a few pixels of other ink on its line for a weak second level (step1: the red band, 9359 flat pixels, and 2 at
    // a red ring's core); dropped, it is a region (background.js)
    const lone = measured.map(c => lonelyDots(scan, c));
    const radii = lone.filter(l => l.length >= DOT_MIN_ISOLATED).map(l => median(l.map(d => d.r)));
    const kept = radii.length
        ? measured.filter((c, i) => lone[i].length || !solidRegion(scan, c.levels[0].color, Math.max(...radii)))
        : measured;
    // a box under the dots in their hue, read as a level (boxBand), at the family's own lone dots' radius or else
    // the other families', or none
    const anyLone = lone.flat();
    const translucent = kept.map(c => {
        const own = lone[measured.indexOf(c)];
        const radius = own.length ? median(own.map(d => d.r)) : anyLone.length ? median(anyLone.map(d => d.r)) : null;
        return boxBand(scan, c, radius);
    });
    for (const c of translucent)
        c.kind = 'translucent';
    dropBlends(translucent, B);
    onProgress?.(TRANSLUCENT_SHARE);
    const opaque = opaqueFamilies(scan, translucent, ink, f => onProgress?.(TRANSLUCENT_SHARE + (1 - TRANSLUCENT_SHARE) * f));
    // a translucent "family" with a level in an opaque family's outline color is those outlines (degradation1: the
    // gray outlines and the gray line as two levels, no isolated dot, no radius)
    const rims = opaque.flatMap(f => [f.outline, f.stroke].filter(Boolean));
    const real = translucent.filter(t => !t.levels.some(L => rims.some(c => colorDistance(c, L.color) <= CLUSTER_TOL)));
    return [...real, ...opaque];
}

// A translucent candidate's first level can be a band in the family's own hue under or over the dots (step1: the
// blue CI band, 181, 207, 240 at α 0.46 of the dots' color, 4800 flat pixels against the lone dots' 820, read as level
// 1 with the dots as level 2, so no dot was a whole one, count 1, and the radius was unknown). When no lone dot shows,
// the family is measured again without that level's pixels; it is a band if that gives lone dots and their radius
// shows the level's color as a solid region. The band is then a region of its own (background.js).
function withoutBand(scan, c) {
    const { image, flat, tol } = scan;
    if (c.levels.length < 2 || lonelyDots(scan, c).length)
        return c;
    const L1 = c.levels[0].color;
    const { data } = image;
    const rest = flat.slice();
    const isBand = nearColor(data, L1, tol.minLevelT);
    for (let p = 0; p < rest.length; p++) {
        if (isBand(p))
            rest[p] = 0;
    }
    const alt = measureFamily({ ...scan, flat: rest }, c.levels[1].color);
    if (!alt.translucent || !alt.levels.length)
        return c;
    const lone = lonelyDots(scan, alt);
    if (lone.length < DOT_MIN_ISOLATED || !solidRegion(scan, L1, median(lone.map(d => d.r))))
        return c;
    return { ...alt, band: L1 };
}

// A family's dots over a band of their own hue drawn under them (landing1: violins, 193, 206, 227 at 0.35 of the blue
// dots' offset, under dots at α 0.75): the band is the one level found, and the dots' colors come beyond it at steps no
// sequence of levels takes. A dot over the page (the first of them, X₁) and one over the band (X₂) differ by
// (1 − α)·(band − B), as across regions (families.js mergeAcrossRegions), so α = 1 − q and the color is
// B + (X₁ − B) / α. The colors beyond are all on one line, so X₂ is the one, of most pixels, that puts the color
// inside the cube (blue 121, 149, 196 → 105, 137, 189: α 0.747, 76, 113, 176; deeper colors gave −210 and −22). The
// candidate has X₁ as its one level and the band recorded (background.js reads it as a region under the dots).
function overBand(scan, c) {
    const { image, B } = scan;
    if (c.levels.length !== 1 || c.beyond.length < 2 || isThin(scan, c.levels[0].color))
        return null;
    const band = c.levels[0].color;
    const d = band.map((v, ch) => v - B[ch]);
    const [X1, ...rest] = c.beyond;
    let best = null;
    for (const X2 of rest) {
        const e = X2.color.map((v, ch) => v - X1.color[ch]);
        const { m: q, off } = alongLine(e, d);
        // (the step must be a level's apart: beeswarm-boxplot's gray dots under a gray panel, all colors on one line,
        // gave a "pair" 5 units apart, α 0.78, and 2129 dots for 711)
        if (q <= OVER_BAND_Q[0] || q >= OVER_BAND_Q[1] || Math.hypot(...e) < LEVEL_SEPARATION || off > CLUSTER_ON_LINE_TOL)
            continue;
        const alpha = 1 - q;
        const color = X1.color.map((v, ch) => B[ch] + (v - B[ch]) / alpha);
        if (color.every(v => v >= -FIT_C_TOL && v <= 255 + FIT_C_TOL) && (!best || X2.count > best.count))
            best = { alpha, color, count: X2.count };
    }
    if (!best)
        return null;
    const line = colorLine(B, X1.color);
    return {
        seed: X1.color, line, proj: projectOnLine(image, line), levels: [X1], beyond: rest,
        fit: { alpha: best.alpha, c: best.color, rms: 0, maxAbs: 0 }, pixels: X1.count, translucent: true, band,
    };
}

// Whether a translucent candidate's level 2 lies in its level 1 (NEST_REACH). Pixels are a level's when flat and
// nearer its color than half the step between the two levels. True when there are too few pieces to judge, or level 1
// or 2 was inferred.
function levelsNest(scan, c) {
    const { image, flat } = scan;
    const [L1, L2] = c.levels;
    if (!L2 || L1.inferred || L2.inferred)
        return true;
    const { width, height, data } = image;
    const half = colorDistance(L1.color, L2.color) / 2;
    const in1 = new Uint8Array(width * height);
    const in2 = new Uint8Array(width * height);
    for (let p = 0; p < in1.length; p++) {
        if (!flat[p])
            continue;
        const [r, g, b] = [data[4 * p], data[4 * p + 1], data[4 * p + 2]];
        if (Math.hypot(r - L1.color[0], g - L1.color[1], b - L1.color[2]) < half)
            in1[p] = 1;
        else if (Math.hypot(r - L2.color[0], g - L2.color[1], b - L2.color[2]) < half)
            in2[p] = 1;
    }
    const to1 = distanceTransform(in1, width, height);
    const pieces = components(in2, width, height).filter(cp => cp.length >= NEST_MIN_PIECE_AREA);
    const near = pieces.filter(cp => cp.some(p => to1[p] <= NEST_REACH * NEST_REACH)).length;
    return pieces.length < NEST_MIN_PIECES || near >= NEST_MIN_SHARE * pieces.length;
}

// A box plot's box drawn under its dots in their own hue is a flat color on the family's line, where a deeper level
// or a color beyond the levels is expected (protrusion2: the green box, 198, 237, 219, as level 2 beside the dots' own
// 193, 235, 216, and the orange box as level 3; protrusion3's as level 2). A level's boundary is arcs of the dots'
// radius, and no straight run of it is longer than a dot's diameter; a box's edges are (straightShare). The family
// is measured again without the box's pixels, and the box is a region under the dots (background.js). When no second
// level shows without it (protrusion3: two dots on the page, 142, 178, 142, lie 8 units off the line to the first
// level, past PERP_TOL), the levels stay as they were, the box's color standing in for level 2's.
function boxBand(scan, c, radius) {
    if (c.band)
        return c;
    // with no radius known (protrusion1: every dot on a whisker), by comparison with level 1, as in beyondBox
    const firstShare = radius ? 0 : straightShare(scan, c.levels[0].color, BEYOND_BOX_RUN);
    const isBox = color => {
        const share = straightShare(scan, color, radius ? BOX_RUN_RADII * radius : BEYOND_BOX_RUN);
        return share >= BOX_MIN_STRAIGHT && share >= BEYOND_BOX_RATIO * firstShare;
    };
    const box = [...c.levels.slice(1), ...c.beyond].find(L => !L.inferred && isBox(L.color) && solidRegion(scan, L.color, BOX_MIN_HALF_WIDTH));
    if (!box)
        return c;
    const band = box.color;
    const alt = measureWithout(scan, c.levels[0].color, band);
    return alt.translucent && alt.levels.length >= 2 ? { ...alt, band, box: true } : { ...c, band, box: true };
}

// A candidate with one level whose next color is a box under the dots, not a dot (protrusion1: very light dots, α
// about 0.1–0.2, whose boxes, 6 units past level 2, took its peak, and whose step ratio then ended the sequence at
// level 1). Its two-level reading as dots over a band of their hue (overBand) fits as well with the roles swapped,
// so shape decides: the box is the straighter (BEYOND_BOX_RATIO). The family is measured again without the box's
// pixels, as in boxBand. Null when that isn't so.
function beyondBox(scan, c) {
    if (c.levels.length !== 1 || !c.beyond.length)
        return null;
    const box = c.beyond[0].color;
    const boxShare = straightShare(scan, box, BEYOND_BOX_RUN);
    if (boxShare < BOX_MIN_STRAIGHT || boxShare < BEYOND_BOX_RATIO * straightShare(scan, c.levels[0].color, BEYOND_BOX_RUN)
        || !solidRegion(scan, box, BOX_MIN_HALF_WIDTH))
        return null;
    const alt = measureWithout(scan, c.levels[0].color, box);
    return alt.translucent && alt.levels.length >= 2 ? { ...alt, band: box, box: true } : null;
}

// a family measured from seed with the pixels of one color (a box's) taken out of the flat mask
function measureWithout(scan, seed, color) {
    const { data } = scan.image;
    const rest = scan.flat.slice();
    const isBox = nearColor(data, color, BOX_COLOR_TOL);
    for (let p = 0; p < rest.length; p++) {
        if (isBox(p))
            rest[p] = 0;
    }
    return measureFamily({ ...scan, flat: rest }, seed);
}

// isolated whole dots under a translucent candidate's levels
function lonelyDots(scan, c) {
    const D = countImage(c.proj, c.line, levelTable(c.line.B, c.levels, c.fit), scan.tol);
    return isolatedDots(D, scan.image.width, scan.image.height, scan.tol);
}

// Removes, in place, translucent candidates whose every measured level is a stack of up to BLEND_DEPTH dots of two
// or more larger candidates kept before them, composited in sRGB over the page. O(families^BLEND_DEPTH · levels).
function dropBlends(translucent, B) {
    const over = (f, under) => under.map((v, ch) => f.fit.alpha * f.fit.c[ch] + (1 - f.fit.alpha) * v);
    const kept = [];
    for (const X of [...translucent]) {
        let stacks = [{ color: B, used: new Set() }];
        const mixed = [];
        for (let d = 0; d < BLEND_DEPTH && kept.length >= 2; d++) {
            stacks = stacks.flatMap(s => kept.map(f => ({ color: over(f, s.color), used: new Set([...s.used, f]) })));
            mixed.push(...stacks.filter(s => s.used.size >= 2).map(s => s.color));
        }
        const blend = mixed.length && X.levels.filter(L => !L.inferred).every(L => mixed.some(c => colorDistance(c, L.color) <= BLEND_TOL));
        if (blend)
            translucent.splice(translucent.indexOf(X), 1);
        else
            kept.push(X);
    }
}

// the estimate for an opaque family: its own background, radius from isolated dots or the distance transform, blur
// from isolated dots or the family's boundaries
function opaqueEstimate(scan, fam, candidates, familyIndex) {
    const { image, flat, background: pageBackground } = scan;
    const { width, height } = image;
    const notes = ['Opaque dots: overlapping dots hide each other, so a dot entirely behind others can’t be found.'];
    if (fam.stroke && fam.fillRadius)
        notes.push(`Outlined: a fill of radius ${fam.fillRadius.toFixed(1)} px inside a stroke too thin to show its own color; `
            + `its darkest mixes average ${fam.stroke.map(Math.round).join(', ')}. The ring counts as part of the dot, and a ring nearer another family's fill is that dot's.`);
    if (fam.pageFill)
        notes.push(`Filled with the page's color inside an outline (${fam.outline.map(Math.round).join(', ')}): each fill is a page region its outline encloses, and the fit reads the outline.`);
    let edge;
    if (fam.isolated.length) {
        edge = edgeBlur(fam.D, width, fam.isolated);
    }
    else {
        edge = boundaryBlur(fam.D, width, height);
        notes.push('No isolated whole dot: the radius comes from the dots’ shapes and is refit by Find dots.');
    }
    const blur = Math.sqrt(Math.max(0, edge.sigma ** 2 - 1 / 12));
    const count = fam.background.every((v, ch) => v === pageBackground.color[ch]) ? pageBackground.fraction : null;
    // what the fit compares: coverage scaled to color units along the line from the family's background to its
    // color, which is what the renderer produces for the table [B, c] at α = 1 (NaN pixels are masked)
    const fitColor = fam.fitColor ?? fam.seed;
    const span = colorDistance(fam.background, fitColor);
    const fitT = new Float32Array(fam.D.length);
    for (let p = 0; p < fitT.length; p++)
        fitT[p] = Number.isNaN(fam.D[p]) ? 0 : span * fam.D[p];
    return {
        background: { color: fam.background, fraction: count },
        candidates, family: familyIndex, flat,
        familyMeasure: fam,
        style: {
            color: [...fam.seed], alpha: 1, levels: fam.table, measuredLevels: 1, radius: fam.radius, blur,
            // a page-colored fill is fitted as its outline's color (what the fit sees); color stays the fill, for display
            ...(fam.fitColor ? { fitColor: [...fam.fitColor] } : {}),
            // for display only: the fit sees an outlined dot as one disc (measureOpaque); a stroke is drawn in the
            // darkest mix seen, not its true color, which no pixel shows
            ...((fam.outline ?? fam.stroke) && fam.fillRadius ? { outline: { color: [...(fam.outline ?? fam.stroke)], inner: fam.fillRadius / fam.radius } } : {}),
        },
        D: fam.D, fitT, isolated: fam.isolated, edge, notes,
        // a page-colored fill's crescents, for the fit's first centers: their edges lie inside the union of discs, where
        // its Hough can't see them (hair2: rows of five dots 0.47 r apart, the middle ones never proposed)
        ...(fam.fillCenters ? { seeds: fam.fillCenters } : {}),
    };
}

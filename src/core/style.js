// Dot style estimation (PLAN.md §2.2, stage 4): the dot families of an image and one family's style (color, alpha,
// levels, radius, edge blur), ready for the fit. estimateStyle and findCandidates put together the translucent
// families (levels.js) and the opaque ones (opaque.js), with the rules that tell bands and boxes under the dots from
// the dots' own levels; scan.js has the Scan they pass around, shape.js the measures of dots as shapes.

import { distanceTransform } from './layers.js';
import { nearColor, colorDistance, sub } from './image.js';
import { LEVEL_SEPARATION, CLUSTER_TOL, BOX_COLOR_TOL, isThin, scanOf, solidRegion, straightShare } from './scan.js';
import { components, isolatedDots, measureRing, edgeBlur, boundaryBlur } from './shape.js';
import { median } from './stats.js';
import {
    FIT_C_TOL, alongLine, colorLine, projectOnLine, levelTable, countImage, familyCandidates, measureFamily, lonelyDots,
    firstLevelFills,
} from './levels.js';
import {
    CLUSTER_ON_LINE_TOL, DOT_MIN_ISOLATED, STROKE_RING, measureOpaque, isDotFamily, opaqueFamilies, strokedFamily,
} from './opaque.js';

/** @typedef {import('./scan.js').Scan} Scan */
/** @typedef {import('./levels.js').LevelMeasure|import('./opaque.js').OpaqueMeasure} FamilyMeasure */

/**
 * One dot family of an image, measured and ready to be masked and fitted (estimateStyle). families.js prepares one
 * per family and adds the fields marked so.
 *
 * @typedef {object} Estimate
 * @property {{color: number[], fraction: number}} background  what the dots lie on: 0–255, and its share of the image
 * @property {FamilyMeasure[]} candidates  every family found in the image
 * @property {number} family  this one's index among them
 * @property {Uint8Array|null} [flat]  scan.js flatMask of the image
 * @property {FamilyMeasure} [familyMeasure]  candidates[family]; absent when no family was found
 * @property {import('./render.js').DotStyle|null} style  what the fit draws the dots with: color, alpha, levels,
 *   radius and blur (px; null when no dot shape shows), measuredLevels, and for outlined dots ring, outline or
 *   layered; null when no family was found
 * @property {Float32Array|null} D  per pixel, the overlap count of the family's dots (translucent) or their
 *   coverage, 0–1 (opaque); NaN where it can't be read
 * @property {Float32Array} [fitT]  per pixel, what the fit compares its model with, when it isn't the image along
 *   the family's line (opaque: the coverage in color units; outlines: the ring image)
 * @property {{cx: number, cy: number, r: number}[]} isolated  whole single dots (shape.js isolatedDots), px
 * @property {{sigma: number, offset: number, profile: object[]}|null} edge  the blur of the dots' edges (shape.js
 *   edgeBlur or boundaryBlur)
 * @property {string[]} notes  caveats, for display
 * @property {{x: number, y: number}[]} [seeds]  px, centers for the fit to try first
 * @property {import('./image.js').RgbaImage} [image]  families.js: the image to fit the family on (layers over the
 *   dots divided out, or the family's own virtual image)
 * @property {Float32Array} [base]  families.js, outlines: per pixel, what lies under the rings
 * @property {Uint8Array} [outlineMask]  families.js, outlines: the mask the rings bring (layers.js buildMask)
 * @property {number[][]|null} [fillFamilies]  0–255, the fill colors of one family's dots (outlined dots of a few
 *   colors; a palette's, once fitted)
 * @property {boolean} [palette]  families.js: dots of many distinct colors, split by color after the fit
 * @property {'opaque'|'translucent'|'outline'} [scale]  families.js: the dots' colors vary, and how they were read
 */

const RINGED_MIN_SHARE = 0.5; // of the isolated dots, for the family to be outlined
// A translucent "family" whose levels are all stacks of two other families' dots is their overlaps (bench2
// group-large, α 0.33: red over blue and blue over red as two purple families with no isolated dot). Measured: the
// blends' levels within 1.3 of a stack, the real families' 86 or more away.
const BLEND_TOL = 12; // 0–255 units
const OVER_BAND_Q = [0.05, 0.95]; // 1 − α for a family's dots over a band of their hue (overBand)
// A figure's lines are drawn in its dots' full color. An opaque plateau beyond the dots on their line is that color
// when, as the color, it puts the dots over the band where they are found (bread2: a JPEG, α 0.5; the blue dots' 438
// and 191 flat pixels on the page and over the band gave α 0.53 and the red's 74 and 64 gave 0.61, with the lines, of
// 1518 and 690 pixels, at 0.507 and 0.510 of the way; the dots over the bands 1.5 and 3.6 units from where the lines'
// colors put them, and 10.3 from where the next nearest plateau would).
const PLATEAU_TOL = 6; // 0–255 units (RGB distance)
// … and a color line with no dots over a band to measure is a family at the α of one found, when a plateau beyond
// its first fill gives that α (bread2's purple, 0.496: two dots clear of the pile, none over its band)
const SIBLING_ALPHA_TOL = 0.05;
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
const NEST_MAX_RETRIES = 3; // colors set aside as other ink on a family's line before the candidate is given up

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
 * @param {number} [opts.radius]  px, the radius of the image's other families, for a family with no whole dot of
 *   its own (families.js pickedEstimate)
 * @returns {Estimate}  style.radius is null when no dot shape shows
 */
export function opaqueFamilyEstimate(image, seed, opts = {}) {
    const scan = scanOf(image, opts);
    const others = opts.others ?? [];
    const stroke = opts.stroke ?? null;
    const rings = opts.ringsToOthers && others.length ? others.map(fill => ({ fill, reach: STROKE_RING })) : null;
    const fam = measureOpaque(scan, seed, others, stroke ? null : opts.outline ?? null,
        { stroke: stroke?.color ?? null, fillRadius: stroke?.fillRadius, rings, radius: opts.radius });
    const measured = { ...fam, fillRadius: stroke?.fillRadius ?? fam.fillRadius, siblingRadius: opts.radius };
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
 * @returns {Estimate}  style is ready for render.js; its radius and blur are null when no isolated dot was found
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
        // (a picked color's line can hold other ink as the search's candidates' do)
        fam = (fam.translucent && nestedLevels(scan, fam).family) || fam;
        if (!fam.translucent) {
            const plain = measureOpaque(scan, opts.seed);
            fam = isDotFamily(plain) ? plain : strokedFamily(scan, opts.seed) ?? plain;
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
    let isolated = isolatedDots(D, width, height, tol);
    let radius = null;
    let blur = null;
    let edge = null;
    let ring = null;
    const shape = () => {
        radius = isolated.length ? median(isolated.map(d => d.r)) : null;
        edge = isolated.length ? edgeBlur(D, width, isolated) : null;
        // the profile's σ includes the pixel box, which the renderer's exact coverage already supplies
        blur = edge ? Math.sqrt(Math.max(0, edge.sigma ** 2 - 1 / 12)) : null;
    };
    shape();
    if (isolated.length) {
        const ringed = isolated.filter(d => d.ringed);
        if (ringed.length >= RINGED_MIN_SHARE * isolated.length) {
            const { B: lb, u } = fam.line;
            const tC = (lb[0] - fam.fit.c[0]) * u[0] + (lb[1] - fam.fit.c[1]) * u[1] + (lb[2] - fam.fit.c[2]) * u[2];
            ring = measureRing(fam.proj.t, width, height, ringed, radius, fam.fit.alpha, tC, blur);
            if (ring)
                notes.push(`Outlined dots: a ring from ${(ring.inner * radius).toFixed(1)} to ${(ring.outer * radius).toFixed(1)} px of a radius of ${radius.toFixed(1)}, ${Math.round(100 * (fam.fit.alpha + ring.alpha))}% opaque against ${Math.round(100 * fam.fit.alpha)}% inside, drawn in the fill's color line (an approximation where outlines cross other dots).`);
            // Deeper in its outer part with no ring there to measure, a piece is not a whole dot (expression with noisy
            // matching: opaque red dots read at α 0.96, the outer half of each a level deeper by JPEG noise, which
            // left a ring 0.04 of opacity; two such gave the family a radius, a gray one took it as a sibling's, and
            // the noisy scan won with two families: 1033 dots for 46)
            if (!ring) {
                isolated = isolated.filter(d => !d.ringed);
                shape();
            }
        }
    }
    if (!isolated.length)
        notes.push(NO_ISOLATED_NOTE);
    if (fam.plateau) {
        const ink = `the solid ink in their hue (${fam.plateau.color.map(Math.round).join(', ')}, ${fam.plateau.count} flat pixels: a line or a key drawn in the dots' color)`;
        notes.push(fam.plateau.sibling
            ? `No overlap of these dots shows how opaque they are: their full color is taken to be ${ink}, which gives them the opacity of the image's other families, ${Math.round(100 * fam.fit.alpha)}%.`
            : `The dots' full color is taken to be ${ink}, which makes them ${Math.round(100 * fam.fit.alpha)}% opaque; the dots over the band of their hue agree with it.`);
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
 * @param {((fraction: number) => void)|null} [onProgress]
 * @returns {FamilyMeasure[]}  translucent families, then opaque ones
 */
export function findCandidates(scan, onProgress = null) {
    const { translucent, ink } = translucentFamilies(scan);
    onProgress?.(TRANSLUCENT_SHARE);
    // (the translucent families' own whole dots give their size, for the colors with too few dots to show theirs)
    const radii = translucent.map(t => lonelyDots(scan, t)).filter(lone => lone.length).map(lone => median(lone.map(d => d.r)));
    const opaque = opaqueFamilies(scan, translucent, ink, f => onProgress?.(TRANSLUCENT_SHARE + (1 - TRANSLUCENT_SHARE) * f), radii);
    // a translucent "family" with a level in an opaque family's outline color is those outlines (degradation1: the
    // gray outlines and the gray line as two levels, no isolated dot, no radius)
    const rims = opaque.flatMap(f => [f.outline, f.stroke].filter(Boolean));
    const real = translucent.filter(t => !t.levels.some(L => rims.some(c => colorDistance(c, L.color) <= CLUSTER_TOL)));
    // … and one with no dot alone whose first level is a solid area at the opaque dots' radius is what those dots lie
    // on (mpo: black dots over gray boxes of 160 and 127, whose colors with the page fit a gray of 107 at α 0.64; it
    // had no radius of its own and took the black dots' as a sibling's). As withoutSolidBands, which has only the
    // translucent families' radii to go by.
    const radius = Math.max(0, ...opaque.map(f => f.radius ?? 0));
    const dots = radius ? real.filter(t => lonelyDots(scan, t).length || !solidRegion(scan, t.levels[0].color, radius)) : real;
    return [...dots, ...opaque];
}

// The translucent families, and `ink`: the level colors of the candidates that weren't families, other ink for the
// opaque ones. The stages: the candidates whose levels can be a family's (levelCandidates), less those that are a
// solid band (withoutSolidBands), each read with a box under its dots if it has one (boxBand), less the blends of two.
function translucentFamilies(scan) {
    const { measured, ink } = levelCandidates(scan);
    const lone = measured.map(c => lonelyDots(scan, c));
    const kept = withoutSolidBands(scan, measured, lone);
    // a box under the dots in their hue, read as a level (boxBand), at the family's own lone dots' radius or else
    // the other families', or none
    const anyLone = lone.flat();
    const translucent = kept.map(c => {
        const own = lone[measured.indexOf(c)];
        const radius = own.length ? median(own.map(d => d.r)) : anyLone.length ? median(anyLone.map(d => d.r)) : null;
        return boxBand(scan, c, radius);
    });
    dropBlends(translucent, scan.B);
    return { translucent, ink };
}

// The color lines whose levels can be a translucent family's (`measured`), and the level colors of those that
// can't (`ink`).
// A translucent family's first level fills the insides of single dots, so it is thick, or whole dots show on it
// (levels.js firstLevelFills). A first level that is
// only lines is a line's color that happens to lie on the line to an opaque family's color (microplastics1: a
// 3 px border bar at 48 and black dots read as levels 1 and 3, α 0.81, with no isolated dot, which hid the black
// dots as an opaque family).
// A candidate whose level 2 lies outside its level 1, however it is measured, is no family (nestedLevels); its colors
// stay other ink for the opaque families, as when it was dropped later as text (engagement1: black text and the
// blues' strokes; without them the blues' fit gave 55 dots for 49, P 0.873)
function levelCandidates(scan) {
    const banded = familyCandidates(scan).map(c => beyondBox(scan, c) ?? overBand(scan, c) ?? c);
    const alphas = banded.filter(c => c.translucent && c.plateau).map(c => c.fit.alpha);
    const nesting = banded.map(c => (c.translucent ? c : atSiblingAlpha(scan, c, alphas) ?? c))
        .filter(c => c.translucent && firstLevelFills(scan, c))
        .map(c => withoutBand(scan, c));
    const measured = [];
    const ink = [];
    for (const c of nesting) {
        const { family, others } = nestedLevels(scan, c);
        if (family)
            measured.push(family);
        ink.push(...others);
    }
    return { measured, ink };
}

// A translucent candidate with levels that nest, and the colors that aren't its levels (`others`). A level 2 that
// lies outside level 1 (levelsNest) is other ink on the family's line, not two of its dots, and with it the fit is of
// colors that have nothing to do with each other. The family is measured again without that color's pixels, up to
// NEST_MAX_RETRIES times (bread1: grey15 dots at α 0.6 over ggplot2's gray panel, 114, with black dots, 92, and the
// axis title, 85, all on the page's own gray line; the three fit a family of 85 at α 0.79, which had no dots, and
// picked by hand put 509 on a box's outline. Without 92 and 85 the levels are 114, 72, 52, 44: α 0.60, color 40).
// `family` is null when nothing nests; a candidate read with a band or a box keeps its one reading.
function nestedLevels(scan, c) {
    const { image } = scan;
    const others = [];
    let m = c;
    let rest = scan.flat;
    for (let i = 0; i < NEST_MAX_RETRIES && m.translucent && !levelsNest(scan, m) && !c.band; i++) {
        const [L1, L2] = m.levels;
        others.push(L2.color);
        const isOther = nearColor(image.data, L2.color, colorDistance(L1.color, L2.color) / 2);
        rest = rest.slice();
        for (let p = 0; p < rest.length; p++) {
            if (isOther(p))
                rest[p] = 0;
        }
        m = measureFamily({ ...scan, flat: rest }, c.seed);
    }
    if (m.translucent && (m === c || firstLevelFills(scan, m)) && levelsNest(scan, m))
        return { family: m, others };
    return { family: null, others: [...others, ...m.levels.filter(L => !L.inferred).map(L => L.color)] };
}

// A candidate without a lone dot whose first level is a solid area at the other families' radius is a band, with
// a few pixels of other ink on its line for a weak second level (step1: the red band, 9359 flat pixels, and 2 at
// a red ring's core); dropped, it is a region (background.js). `lone` is each candidate's lone dots.
function withoutSolidBands(scan, measured, lone) {
    const radii = lone.filter(l => l.length >= DOT_MIN_ISOLATED).map(l => median(l.map(d => d.r)));
    return radii.length
        ? measured.filter((c, i) => lone[i].length || !solidRegion(scan, c.levels[0].color, Math.max(...radii)))
        : measured;
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
    if (c.levels.length !== 1 || isThin(scan, c.levels[0].color))
        return null;
    const band = c.levels[0].color;
    const beyond = beyondBand(scan, c);
    if (beyond.length < 2)
        return null;
    const d = band.map((v, ch) => v - B[ch]);
    const [X1, ...rest] = beyond;
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
            best = { alpha, color, count: X2.count, over: X2.color };
    }
    if (!best)
        return null;
    // the plateau beyond that, as the full color, best puts the dots over the band where they are (landing1's orange:
    // of three colors past the dots, the first is 3.7 units off at α 0.80, the last, the dots' own, 0.4 off at 0.753)
    const full = rest.map(P => {
        const { m, off } = alongLine(sub(X1.color, B), sub(P.color, B));
        return { P, m, off, miss: colorDistance(X1.color.map((v, ch) => v + (1 - m) * d[ch]), best.over) };
    }).filter(f => f.m > 1 - OVER_BAND_Q[1] && f.m < 1 - OVER_BAND_Q[0] && f.off <= CLUSTER_ON_LINE_TOL && f.miss <= PLATEAU_TOL)
        .sort((a, b) => a.miss - b.miss)[0];
    const fit = full ? { alpha: full.m, c: [...full.P.color] } : { alpha: best.alpha, c: best.color };
    const line = colorLine(B, X1.color);
    return {
        kind: 'translucent', seed: X1.color, line, proj: projectOnLine(image, line), levels: [X1], beyond: rest,
        fit: { ...fit, rms: 0, maxAbs: 0 }, pixels: X1.count, translucent: true, band,
        ...(full ? { plateau: { color: [...full.P.color], count: full.P.count } } : {}),
    };
}

// The colors beyond a candidate's one level, a band: as measured when two or more show. A band that is a solid area
// can hold a hundred times the pixels of the dots over it, and a level's peak must hold a share of the line's flat
// pixels (levels.js PEAK_MIN_FRACTION), so the dots' own colors made no peak (bread2: the blue band's 129,883 flat
// pixels set the bar at 520, and the dots' fills, 438 on the page and 191 over the band, were not seen). Then the line
// is measured again without the band's pixels, aimed at the first color found: the band's own color lies up to 2° off
// the dots' (its few units from the page are rounded), which at the line's far end is past the tolerance.
// Only colors past the band count: nearer the page are its antialiased rims (nicest language: a panel of 241, 240, 233
// with 61 pixels of 244, 243, 237 and 9 of 238, 238, 229 around it, which made a "family" at α 0.69).
function beyondBand(scan, c) {
    const { B, tol } = scan;
    const band = c.levels[0].color;
    if (c.beyond.length >= 2 || !solidRegion(scan, band, BOX_MIN_HALF_WIDTH))
        return c.beyond;
    const past = L => colorDistance(L.color, B) >= colorDistance(band, B) + LEVEL_SEPARATION;
    const first = measureWithout(scan, band, band, tol.minLevelT).levels.find(past);
    if (!first)
        return c.beyond;
    const again = measureWithout(scan, first.color, band, tol.minLevelT);
    return [...again.levels, ...again.beyond].filter(past);
}

// A candidate that is no translucent family as measured, read as one at a found family's α: its first color X₁ that
// lies at that share (SIBLING_ALPHA_TOL) of the way from the page to a plateau beyond it is one dot, the plateau the
// full color. Figures draw their groups alike, and a group with too few dots clear of the others shows no second
// level and no dot over a band to measure α by. `alphas` are those of the families whose own full color is a plateau
// in their hue (overBand): what this one is assumed to share with them. Null when no color does.
function atSiblingAlpha(scan, c, alphas) {
    const { image, B } = scan;
    const colors = [...c.levels, ...c.beyond];
    for (let i = 0; i < colors.length && alphas.length; i++) {
        const X1 = colors[i];
        for (const P of colors.slice(i + 1)) {
            const { m, off } = alongLine(sub(X1.color, B), sub(P.color, B));
            if (off > CLUSTER_ON_LINE_TOL || !alphas.some(a => Math.abs(a - m) <= SIBLING_ALPHA_TOL))
                continue;
            const line = colorLine(B, X1.color);
            return {
                kind: 'translucent', seed: X1.color, line, proj: projectOnLine(image, line), levels: [X1], beyond: colors.slice(i + 1),
                fit: { alpha: m, c: [...P.color], rms: 0, maxAbs: 0 }, pixels: X1.count, translucent: true,
                plateau: { color: [...P.color], count: P.count, sibling: true },
            };
        }
    }
    return null;
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

// a family measured from seed with the pixels of one color (a box's or a band's, within `tol`) taken out of the flat mask
function measureWithout(scan, seed, color, tol = BOX_COLOR_TOL) {
    const { data } = scan.image;
    const rest = scan.flat.slice();
    const isBox = nearColor(data, color, tol);
    for (let p = 0; p < rest.length; p++) {
        if (isBox(p))
            rest[p] = 0;
    }
    return measureFamily({ ...scan, flat: rest }, seed);
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
        const shown = `${fam.dots + fam.dotsUnder} of these dots show, whole or partly under other ink`;
        const others = fam.siblingRadius ? `No isolated whole dot: the radius, ${fam.siblingRadius.toFixed(1)} px, is that of the image's other families` : null;
        notes.push(!others ? 'No isolated whole dot: the radius comes from the dots’ shapes and is refit by Find dots.'
            : fam.foundRadius ? `${others}; the dots' own shapes gave ${fam.foundRadius.toFixed(1)} px, at which ${shown}.`
            : `${others}, at which ${shown}.`);
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

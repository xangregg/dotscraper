// Opaque dot families (PLAN.md phase 5): color clusters whose pixels form dots. measureOpaque reads one color's
// coverage image, radius and centers, as a plain, outlined, stroked or page-colored fill; opaqueFamilies goes
// through the image's color clusters and decides which are families; the radius of dots that show no isolated
// whole one comes from their shapes (shapeRadius, siblingRadius).

import { houghCenters, houghGradient } from './fit.js';
import { distanceTransform, dilate, nearestTransform, opening } from './layers.js';
import { nearColor, nearAnyColor, colorMask, colorDistance } from './image.js';
import { CLUSTER_TOL, isThin, solidRegion } from './scan.js';
import {
    ISOLATED_MIN_AREA, ISOLATED_MAX_ASPECT, ISOLATED_MAX_MOMENT_RATIO, components, isolatedDots, boundaryBlur,
    discShare,
} from './shape.js';
import { median, quantile } from './stats.js';
import { alongLine, colorLine } from './levels.js';

/** @typedef {import('./scan.js').Scan} Scan */

/**
 * A color measured as an opaque dot family (measureOpaque), with the fields a translucent one has (levels.js
 * LevelMeasure), so that the two are handled alike after.
 *
 * @typedef {object} OpaqueMeasure
 * @property {'opaque'} kind
 * @property {number[]} seed  the fill, 0–255
 * @property {number[]|null} outline  0–255: an outline that shows its own color
 * @property {number[]|null} stroke  0–255: a stroke too thin to show its color, in the darkest mix seen
 * @property {import('./levels.js').ColorLine} line  from the family's background to the color it is fitted in
 * @property {number[]} background  0–255, what the dots usually lie on
 * @property {number[][]} table  [background, fit color]: the levels a render needs
 * @property {Float32Array} D  per pixel, the dots' coverage, 0–1; NaN where it can't be read
 * @property {{cx: number, cy: number, r: number}[]} isolated  whole single dots (shape.js isolatedDots)
 * @property {number|null} radius  px, the outline's or stroke's included
 * @property {number} dots  dot-shaped centers found at that radius
 * @property {number} strongDots  those seen over DOT_STRONG_ARC of their edge
 * @property {number} coverage  0–1, the share of the family's pixels inside those dots
 * @property {number} dotsUnder  more centers at that radius: dots partly under other ink (DOT_MIN_SEEN)
 * @property {number} coverageUnder  0–1, the share of the family's pixels inside the dots, those included
 * @property {{x: number, y: number}[]} centersUnder  px, those centers
 * @property {number} pixels  pixels of the fill's color (and the outline's within reach)
 * @property {Uint8Array} edgeInk  per pixel, other ink read as the dots' background: 1 + its index in inkColors
 * @property {number[][]} inkColors  0–255
 * @property {{x: number, y: number}[]} centers  px, the centers found
 * @property {number} [fillRadius]  px, the fill inside an outline or stroke
 * @property {number} [siblingRadius]  px, a radius taken from the image's other families, for a family with no whole
 *   dot of its own
 * @property {number} [foundRadius]  px, with siblingRadius: the radius of its own shapes that the family was found
 *   at, which `dots` and `dotsUnder` are counted at
 * @property {boolean} [pageFill]  the fill is the page's color, inside an outline; then fitColor (the outline's,
 *   0–255) and fillCenters (px, the fills' centers, for the fit to start from)
 * @property {import('./levels.js').Level[]} levels  one: the fit color, at its distance from the background
 * @property {import('./levels.js').Level[]} beyond  none
 * @property {{alpha: number, c: number[], rms: number, maxAbs: number, underdetermined: boolean}} fit  α 1, c the fill
 * @property {boolean} translucent  false
 * @property {number} [interior]  0–1, the share of the color's pixels with all eight neighbors that color
 *   (opaqueFamilies)
 * @property {number[][]} [under]  0–255, the families drawn over this one (opaqueFamilies)
 */

// Opaque families (PLAN.md phase 5) are color clusters, not directions: a beeswarm's dots and the violin shading
// under them can share a hue, and compressed images (violin) have dot interiors too noisy to be flat.
const CLUSTER_BIN = 8; // 0–255 units per channel, color histogram bins
const CLUSTER_MIN_FRACTION = 0.001; // a cluster holds at least this share of the image …
const CLUSTER_MIN_PIXELS = 100; // … and at least this many pixels
const CLUSTER_MAX_MIN_PIXELS = 1500;
export const CLUSTER_ON_LINE_TOL = 8; // 0–255 units: a cluster this close to a translucent family's line is one of its levels
const LEVEL_LINE_BEYOND = 0.25; // × the family's full color's distance from its background (the fitted color can be short)
const MIX_SHARE = [0.05, 0.95]; // of the way from one color to another: a mix of the two lies between these
const INK_TOL = 15; // 0–255 units from every background color for a pixel to be ink
const RING = 3; // px: the ring around a family's pixels whose most common color is its background
// A dot family (measured on every test image, PLAN.md phase 5): radius ≥ DOT_MIN_RADIUS (text and thin lines give
// 2.5–2.74), at least DOT_MIN_STRONG filled centers with DOT_STRONG_ARC of their edge seen (nicest language's red:
// 3 of 9), and either discs that explain DOT_MIN_COVERAGE of the cluster's pixels (dots: 0.44–1.00; text, dashes,
// gridlines: ≤ 0.23) or
// DOT_MIN_ISOLATED whole isolated dots (black and gray dots share their color with text: coverage 0.18–0.23, but
// 9 and 17 isolated dots).
const DOT_MIN_RADIUS = 3; // px
// Smaller dots count when isolated whole dots show them round (bench1 tiny-dots: r 2.6, 58 isolated; text clusters
// measure r 2.5–2.74 but never have an isolated round component)
const SMALL_DOT_MIN_RADIUS = 2; // px
const SMALL_DOT_MIN_ISOLATED = 5;
export const SPECK_RADIUS_RATIO = 0.4; // × the largest family's radius: below it, a mostly-text color's dots are specks
const DOT_MIN_ARC = 0.5; // fraction of the circumference for a Hough center to count
const DOT_STRONG_ARC = 0.75;
const DOT_MIN_STRONG = 3;
const DOT_MIN_COVERAGE = 0.35;
// … and with isolated dots instead, at least this much: a color 98% text with a few round specks (hair2–3: decimal
// points and a colon's dots, coverage 0.02) is no family, even alone; real ones sharing text's color measured 0.18
const DOT_MIN_COVERAGE_ANY = 0.1;
// … or many strong, filled centers: in a packed beeswarm Hough finds only part of the dots, so coverage stays low, and
// no dot is isolated (beeswarm-boxplot's gray: 30 strong, coverage 0.27; round glyphs are hollow and fail the fill)
const DOT_MANY_STRONG = 10;
export const DOT_MIN_ISOLATED = 3;
// … and this many whole isolated dots stand in for the strong centers: a blurred dot's edge, 3 or 4 px wide, votes
// for its center from every radius across it, and no arc reaches DOT_STRONG_ARC (corneal's black: 62 centers, none
// strong, 25 whole dots of radius 8.22 to 8.31, coverage 0.28 beside bold labels)
const DOT_MANY_ISOLATED = 10;
// A color whose filled centers lie at the radius of a family already found needs no strong arc: figures draw every
// group with one marker size, and a small group can be all touching dots crossed by lines (myeloid1's teal: 9 dots
// in pairs and a stack, gray pair lines reaching each, arcs at most 0.72; 5 filled centers at its radius, coverage
// 0.64, radius 13.84 as the orange family's). It still needs DOT_MIN_STRONG centers of DOT_MIN_ARC and coverage.
export const SIBLING_RADIUS_TOL = 0.1; // × the found family's radius
const SIBLING_MIN_ISOLATED = 2; // whole dots of its own, for its thick pixels alone to count (siblingEvidence)
// … and a color that is all dots of that size needs only two of them, one seen whole or by a strong arc: a group can be
// two points (bench-plos synthetic's teal: 2 dots of 16, one alone and one touching a salmon dot, radius 11.5 beside the
// translucent families' 11.7, 0.99 of the color inside the two discs, one strong arc). FEW_MIN_COVERAGE of the color lies in the discs: a glyph or
// a line in it would be most of its pixels.
const FEW_MIN_DOTS = 2;
const FEW_MIN_COVERAGE = 0.9;
// A layer's color lies this far off the line from the page to the family's color, or "seen through it" says nothing:
// on one line (phase1's grays, white to black) every color is any other seen through a third. The bands measured
// 48 units off (bench3 ANOVA 206, 213, 231 and bench1 group-smoother-ci 205, 213, 232 against pink and red).
const LAYER_MIN_OFF = 20; // 0–255 units
// … and a layer is an area a dot can lie under, solid at this half width: a color of strokes is none (song-cents: its
// navy dots, 15, 39, 66, are the labels' gray 88 plus 0.45 of the mid blue's offset from the page, to 0.3 units,
// and were no family for being "the mid blue seen through the gray")
const LAYER_MIN_HALF_WIDTH = 4; // px
// The pixels next to an opaque family's dots that are another ink (measureOpaque's edgeInk): its index in the others'
// colors plus 1, or this for an ink not among them
const EDGE_INK_UNLISTED = 255;
// Drawing order of two opaque families (drawnUnder): seam pixels on the upper one's circles, and how many times those
// on the lower one's they must be
const ORDER_RING = 2; // px from a found dot's circle
const ORDER_MIN_PIXELS = 20;
const ORDER_RATIO = 3;
// Text looks like dots to a Hough vote (round glyphs, and every gray of its antialiasing is a color cluster), but
// it is strokes: mostly edge pixels, and the inside of an "o" is empty.
const DOT_MIN_INTERIOR = 0.1; // share of a cluster's pixels with all 8 neighbors in the cluster
const BLEND_OUTLINE_MAX_INTERIOR = 0.3; // a cluster between a fill and its outline with more interior is a family
// A cluster between a family and the page with interior pixels is ink drawn in that color, not a dot's antialiased rim
// (1–2 px wide, interior 0.00 on every bench image): boxdots1's gray box outlines and median lines at 52, 4–5 px
// wide (interior 0.18), read as 81% covered by the black dots, took 64 false dots along them; violin-beeswarm's
// violin fill (182, 187, 224, interior 0.62) under the blue dots, 734 → 712 found for 702.
const BLEND_INK_MIN_INTERIOR = 0.05; // share of the cluster's pixels
// … or a family in a tint of the other's color, when it is the insides of dots: with this share of interior pixels
// the cluster is read as any other is, and is a family if it reads as dots (song-cents: light blue 144, 180, 226,
// 0.69 of the way from the page to the mid blue's 93, 146, 205, interior 0.53, 30 dots; a violin's fill doesn't).
// Not a cluster within a found family's own color or its outline's: a rim between a fill and its outline takes in
// fill pixels by its tolerance, and reads as the dots it rims (a synthetic rim at 149, 6, 220, interior 0.73).
const BLEND_DOT_MIN_INTERIOR = 0.3;
const DOT_FILL_RADII = 0.75; // × r: the disc that must be filled around a Hough center …
const DOT_MIN_FILL = 0.85; // … to this fraction
// A dot partly under other ink (another family's dot over it): the part of that disc the other ink doesn't hold is
// filled to DOT_MIN_FILL, and DOT_MIN_SEEN of the whole disc is the family's fill. Such centers count only for a
// color at a found family's radius (siblingFills).
const DOT_MIN_SEEN = 0.5;
// distance-transform peaks underestimate the radius where dots touch or overlap; the upper quartile of the peaks,
// plus the half pixel the transform loses between pixel centers, matched synthetic swarms within a few tenths
const DT_RADIUS_QUANTILE = 0.75;
const DT_RADIUS_OFFSET = 0.5; // px
// When that radius finds no family, Hough tries larger ones (measureOpaque): packed dots' peaks are short of r
const RADIUS_SCAN_STEP = 0.5; // px
const RADIUS_SCAN_MAX = 1.6; // × the distance transform's radius
const RADIUS_SCAN_KEEP = 0.75; // of the most strong centers, at the largest radius that keeps them
// A line in the dots' own color puts most distance-transform peaks on its ridge, at its half width (kite1: the cyan
// line gave 3.5 px for dots of 12.3, and the scan to 1.6 times that found no dot). Each connected piece's deepest point
// is a dot's center wherever a dot touches the line, so measureOpaque tries again around the radii the pieces give
// (componentRadii), ± this fraction.
const COMPONENT_SCAN = 0.25;
const HUE_RING_MIN_CHROMA = 30; // 0–255 units (max − min channel) a fill needs for a ring in its hue (kite2: 127)
const RING_PAGE_REACH = 3; // px …
const RING_PAGE_MIN_SHARE = 0.3; // … of the page, for this share of a ring's color (hueRings)
const RING_FILL_REACH = 4; // px inside a ring from which its fill's color is read (opaqueFamilies' reading (4))
const COMPACT_MAX_AREA = 2.5; // × π·depth²: a disc is 1, two overlapping discs up to 2; kite2's curve 50

// A dot color can also be text's (nicest language: its gray dots' color is the gray of antialiased black text on the
// white page, which made white their "background" and text strokes their radius). Text is thin: the family's
// background comes from its thick parts (an opening), and distance peaks below DOT_MIN_RADIUS are ignored.
const THICK_OPENING = 1.5; // px
// Opaque coverage is measured against the background where each pixel is (shading, white, a neighboring dot of
// another color), estimated from nearby pixels that are not the family's: violin's dots sit on shading and on
// white, and against a single background color the edges on the other one were unreadable.
const NEAR_FAMILY = 2; // px: pixels this close to the family's color don't inform its background
const MIN_CONTRAST = 30; // 0–255 units between a dot's color and the local background for its coverage to be read
// A blurred dot's edge is wider than NEAR_FAMILY, and beyond it passes through the color of any other ink that lies
// between the dots' color and their background's (biomarker: black dots with an edge 4 px wide, in a gray frame). Out
// there every rim had a ring of pixels read as the frame's, and the pixels nearer in were read against that ring as
// their background, so no dot was whole and the radius, from the distance transform, was 4.5 for 6.0: the fit put
// 216 dots in for 83. For a family with no whole dot, read again as blurred dots (softGround): the near zone goes on
// outward along a dot's own slope, through pixels that are a mix of the family's color and its background, each with
// less of the color than its neighbor nearer in by this much, for this many pixels. Other ink is flat along itself,
// and a crisp dot's edge has ended within NEAR_FAMILY: where the zone grows by less than SOFT_EDGE_MIN_GROWTH × the
// family's own pixels the dots aren't blurred, and the first reading stands, as it does when the second shows no
// whole dot either. (The growth: biomarker 0.29, its black being text and lines too; blurred dots alone, the test's,
// 1.65; biomarker's grays and the test's, 0.00 to 0.02.) Read so always, dots with whole ones moved too: violin-beeswarm 722 found for
// 712 (702 there), cattle 298 for 300.
const SOFT_EDGE_MIN_STEP = 0.05; // of the family's color, per px
const SOFT_EDGE_MIN_SHARE = 0.05; // of the family's color: less is the background
const SOFT_EDGE_MAX_REACH = 3; // px beyond NEAR_FAMILY
const SOFT_EDGE_MIN_GROWTH = 0.15;
// A pixel is a mix of the dot and its background only if it lies close to the line between them, within OFF_MIX_ABS
// or OFF_MIX_REL × its distance along the line. An absolute 25 units let another family's antialiased rims through
// (synthetic swarm: blue–white rims read as gray coverage 0.89 at 20.5 units off, about 13° off the gray mix, and
// births put phantom gray dots on the blue ones); a real rim pixel is off only by noise.
const OFF_MIX_ABS = 10; // 0–255 units, at least …
const OFF_MIX_NOISE = 2.5; // … this many times the family's own color noise (compressed images: violin, nicest)
const OFF_MIX_REL = 0.12;
// … and so is one well past the dot's color along the mix (black text against light gray dots: on the same gray
// axis, it read as coverage 4, clipped to 1, and text strokes became the family's radius)
const MAX_MIX = 1.3;
// Some renderers draw every overlap of opaque dots in one dark color, as dark for three as for two (fanc2's PDF:
// gray 86 dots, overlaps 19, the text's black). A region beyond the family's color on its line, enclosed by the
// family and not another family's color, is such an overlap: covered, not unknown. Text and lines border mostly on
// the page; a darker family's dot packed among this family's in a swarm also borders gaps and other dots (at 0.6
// the synthetic swarm's maroon filled in).
const ISOLATED_FILL_RADII = 0.5; // × r: the middle of an isolated dot that must be the family's fill
const OVERLAP_BORDER_SHARE = 0.9; // of a region's border pixels that must be the family's
// … and a lens has a middle: a line drawn over a swarm in the same dark ink is enclosed by dots too (the synthetic
// swarm's black median line over its light gray dots, 2 px wide), but has no square this deep inside it
const OVERLAP_MIN_DEPTH = 2; // px, half the side of a square that must fit inside the region
// An overlap is darker than one dot by at most a second dot's contrast (fanc2: 19 at 1.4 of the way from the page to
// 86). Unbounded, a pale family's line reached black text (economist: a light gray "family" at 226 filled its text).
const OVERLAP_MAX_MIX = 2; // × the family's contrast with its background
const OUTLINE_SLACK = 0.1; // a pixel this far beyond either end of the fill–outline segment still lies on it
// An outline is the family's only next to its fill: farther out, the outline's color is other ink (phase1: black
// outlines around gray fills, and black text, axes and a black family, which counted as dots everywhere)
const OUTLINE_REACH = 4; // px from the family's fill
const RING_TAIL_MIN_GROWTH = 1; // × the pixels at the reach's edge (familyRings; stiffness: 2.55, degradation1: 0.16 and 0.38)
// an outlined family's coverage counts its fill's color where a disc of this × the fill radius fits
const THICK_FILL_RADII = 0.5;
// Fills in the page's color (white dots in black outlines on a white page) are the page regions their outlines
// enclose: round ones within this share of one radius are one family's whole fills (at least PAGE_FILL_MIN_WHOLE of
// them, and DOT_MIN_STRONG fills in all, whole or partly hidden: hair2's nine white dots lie in two rows, each with one
// whole dot in front) …
const PAGE_FILL_SIZE_TOL = 0.1;
const PAGE_FILL_MIN_WHOLE = 2;
// … and every enclosed region up to this × a whole fill's area is a fill, whole or partly hidden (crescents where
// dots overlap); larger ones (a bar's inside) aren't
const PAGE_FILL_AREA_SLACK = 1.2;
// the outline's color is read in this band outside the whole fills
const PAGE_RING_BAND = [0.5, 2]; // px beyond the fill radius
// A region is a fill only inside a fill-sized circle whose edge it shows: a Hough center at the fill radius with at
// least PAGE_FILL_MIN_ARC of its edge seen, holding this share of the region within PAGE_FILL_DISC_SLACK px beyond
// the fill radius (letters' counters and the gaps between other dots are enclosed page too: hair1's P, B, D and the
// gaps among its pink dots each became a dot). The arc is low for crescents: a dot mostly hidden shows a quarter of
// its edge or less (tumour1's lane of six at zero, hair2's rows).
const PAGE_FILL_IN_DISC = 0.9;
const PAGE_FILL_DISC_SLACK = 1.5; // px
const PAGE_FILL_MIN_ARC = 0.2;
// … and its edge must be its own circle (within PAGE_FILL_EDGE_TOL of the fill radius, PAGE_FILL_OWN_EDGE of it at
// least) or the outlines of fills over it (within the widest outline of the other centers more than
// PAGE_FILL_SAME_RADII × the fill radius away), PAGE_FILL_EXPLAINED of it in all. A glyph's counter is bounded by its
// stroke, which is neither: measured, every dot of tumour1, hair1 and hair2 is explained at 1.00 (own 0.43 or more),
// counters at 0.61 at most (the 0s and 8s of p-values, 0.03–0.38; 0.9964's 6 was a dot, at 0)
const PAGE_FILL_EDGE_TOL = 1.5; // px
const PAGE_FILL_OWN_EDGE = 0.2;
const PAGE_FILL_EXPLAINED = 0.9;
const PAGE_FILL_SAME_RADII = 0.25;
// A marker's outline is thin beside its fill, which it must show; a glyph's stroke is thick beside its counter, which
// encloses page as round as a dot's (comment1's geometric font: o, 0, d and a gave 11 "dots", stroke 0.56 × the counter's
// radius; hair1's outlines are 0.19 × the fill's, tumour1's 0.27)
const PAGE_RING_MAX_WIDTH = 0.4; // × the fill radius
// A family's ring image (familyRings) reads a full outline as this, below outline.js's MASK_INK (0.8), which masks
// near-black ink as text: the reach already masks text here, and black outlines are exactly text's color
const FAMILY_RING_P = 0.45;
const COLLINEAR_SIN2 = 0.01; // sin² of the angle between page→fill and page→outline below which they're one line
// an outline's own color beyond the darkest seen (outlineItself): read from the pixels with this much of the color
// seen, no more fill than this and no more of the two together (the rest the page's) …
const OUTLINE_MIX_MIN_OUTLINE = 0.2;
const OUTLINE_MIX_MAX_FILL = 0.05;
const OUTLINE_MIX_MAX_COVER = 0.9;
// … this many of them or more, when the share of fill they lack, at this quantile, is this or more (stiffness: 0.32;
// with black given as its outline: 0.05), and taken as no more than this
const OUTLINE_MIX_QUANTILE = 0.9;
const OUTLINE_MIX_MIN_PIXELS = 50;
const OUTLINE_MIX_MIN_SHARE = 0.1;
const OUTLINE_MIX_MAX_SHARE = 0.6;
// Stroked fills (PLAN.md open question 10): a fill inside a stroke so thin and blurred that it never shows its own
// color (bench-nature engagement1: on lavender 188, 186, 234 its darkest pixel is about 80, 80, 105). Its edge
// pixels mix three colors, background, stroke and fill, where the opaque test accepts two, so the edge read as
// unknown and no dot was found. The stroke is read from the rings of isolated fills, as the mean of the ring pixels
// farthest off the background–fill line: not its true color, which no pixel shows, but the far corner of the
// colors seen, so every ring pixel lies inside the triangle background–fill–stroke.
// px beyond an outline's measured width that its antialiased rim reaches, read as the ring's
const RING_SLACK = 1.5;
export const STROKE_RING = 4; // px outside a fill read as its ring (engagement1: dark out to 3–4 px past the fill's color, background from 5)
const STROKE_CLEAR = 7; // px: beyond the ring, out to this, a fill must have background around it to count as isolated …
const STROKE_CLEAR_SHARE = 0.9; // … for this share of those pixels
const STROKE_COMPONENT = [12, 5000]; // px, sizes of the fill components whose rings are read
const STROKE_MIN_RINGS = 3; // isolated fills needed
const STROKE_TOP = 0.8; // quantile of the ring pixels' distance off the line: those beyond are the stroke
const STROKE_MIN_OFF = 30; // 0–255 units: the stroke must lie this far off the background–fill line
const STROKE_SLACK = 0.15; // barycentric weights may go this far below 0
// A ring belongs to the fill it surrounds: in each family's coverage, a ring pixel nearer another family's fill (by
// more than STROKE_TIE) is that dot's, and reads 0, so a disc of this family can't be placed to explain it.
const STROKE_TIE = 0.5; // px

// whether color c is a mix of P and F: on the line between them (CLUSTER_ON_LINE_TOL) and clear of both ends
function isMixOf(c, P, F) {
    const { m, off } = alongLine(c.map((x, ch) => x - P[ch]), F.map((x, ch) => x - P[ch]));
    return m > MIX_SHARE[0] && m < MIX_SHARE[1] && off <= CLUSTER_ON_LINE_TOL;
}

/**
 * Color clusters: peaks of a 3-D color histogram (CLUSTER_BIN bins, smoothed over 3×3×3 neighbors) of the ink
 * pixels, each with the mean color of the pixels in its neighborhood of bins.
 *
 * @param {import('./image.js').RgbaImage} image
 * @param {number[][]} backgrounds
 * @returns {{color: number[], count: number}[]}  largest first
 */
export function colorClusters(image, backgrounds) {
    const nb = 256 / CLUSTER_BIN;
    const n = image.width * image.height;
    const count = new Float64Array(nb ** 3);
    const sum = new Float64Array(3 * nb ** 3);
    const d = image.data;
    const isBackground = nearAnyColor(d, backgrounds, INK_TOL);
    for (let p = 0; p < n; p++) {
        if (isBackground(p))
            continue;
        const r = d[4 * p];
        const g = d[4 * p + 1];
        const b = d[4 * p + 2];
        const k = ((r / CLUSTER_BIN | 0) * nb + (g / CLUSTER_BIN | 0)) * nb + (b / CLUSTER_BIN | 0);
        count[k]++;
        sum[3 * k] += r;
        sum[3 * k + 1] += g;
        sum[3 * k + 2] += b;
    }
    const at = (i, j, k) => (i < 0 || j < 0 || k < 0 || i >= nb || j >= nb || k >= nb ? -1 : (i * nb + j) * nb + k);
    const smooth = new Float64Array(nb ** 3);
    for (let i = 0; i < nb; i++) {
        for (let j = 0; j < nb; j++) {
            for (let k = 0; k < nb; k++) {
                let t = 0;
                for (let a = -1; a <= 1; a++) {
                    for (let b = -1; b <= 1; b++) {
                        for (let c = -1; c <= 1; c++) {
                            const q = at(i + a, j + b, k + c);
                            if (q >= 0)
                                t += count[q];
                        }
                    }
                }
                smooth[at(i, j, k)] = t;
            }
        }
    }
    const min = Math.max(CLUSTER_MIN_PIXELS, Math.min(CLUSTER_MAX_MIN_PIXELS, CLUSTER_MIN_FRACTION * n));
    const out = [];
    for (let i = 0; i < nb; i++) {
        for (let j = 0; j < nb; j++) {
            for (let k = 0; k < nb; k++) {
                const v = smooth[at(i, j, k)];
                if (v < min)
                    continue;
                let peak = true;
                const rgb = [0, 0, 0];
                for (let a = -1; a <= 1 && peak; a++) {
                    for (let b = -1; b <= 1 && peak; b++) {
                        for (let c = -1; c <= 1; c++) {
                            const q = at(i + a, j + b, k + c);
                            if (q < 0)
                                continue;
                            if (smooth[q] > v || (smooth[q] === v && q > at(i, j, k))) {
                                peak = false;
                                break;
                            }
                            rgb[0] += sum[3 * q];
                            rgb[1] += sum[3 * q + 1];
                            rgb[2] += sum[3 * q + 2];
                        }
                    }
                }
                if (peak)
                    out.push({ color: rgb.map(x => x / v), count: v });
            }
        }
    }
    return out.sort((a, b) => b.count - a.count);
}

/** The most common color in a ring RING px wide around a mask, and how many ring pixels have it. */
function ringBackground(image, mask) {
    const { width, height, data } = image;
    const grown = dilate(mask, width, height, RING);
    const counts = new Map();
    for (let p = 0; p < mask.length; p++) {
        if (!grown[p] || mask[p])
            continue;
        const key = (data[4 * p] << 16) | (data[4 * p + 1] << 8) | data[4 * p + 2];
        counts.set(key, (counts.get(key) ?? 0) + 1);
    }
    let best = null;
    for (const [k, c] of counts) {
        if (!best || c > best.c)
            best = { k, c };
    }
    return best ? [best.k >> 16, (best.k >> 8) & 255, best.k & 255] : null;
}

/**
 * Radius from the distance transform of a family's pixels: its local maxima sit at dot centers, at about r where a
 * dot is whole and less where neighbors overlap it; the upper quartile, plus the half pixel lost between pixel
 * centers.
 */
function radiusFromDistance(D, width, height) {
    // distance to the nearest pixel known to be outside the family, at the family's pixels: a masked line across a
    // dot isn't its edge (protrusion1: every dot on a whisker read as two halves, 9.5–10 for 19.5)
    const outside = new Uint8Array(D.length);
    for (let p = 0; p < D.length; p++)
        outside[p] = D[p] > 0.5 || Number.isNaN(D[p]) ? 0 : 1;
    const d2 = distanceTransform(outside, width, height);
    const peaks = [];
    for (let y = 1; y < height - 1; y++) {
        for (let x = 1; x < width - 1; x++) {
            const p = y * width + x;
            const v = d2[p];
            if (!(D[p] > 0.5) || v < DOT_MIN_RADIUS * DOT_MIN_RADIUS || v === Infinity)
                continue;
            let isMax = true;
            for (let dy = -1; dy <= 1 && isMax; dy++) {
                for (let dx = -1; dx <= 1; dx++) {
                    // (a masked pixel is no peak, nor does it hide one: a line through a dot's center)
                    if ((dx || dy) && d2[p + dy * width + dx] > v && D[p + dy * width + dx] > 0.5) {
                        isMax = false;
                        break;
                    }
                }
            }
            if (isMax)
                peaks.push(Math.sqrt(v));
        }
    }
    return peaks.length ? quantile(peaks, DT_RADIUS_QUANTILE) + DT_RADIUS_OFFSET : null;
}

// The number of Hough centers with DOT_MIN_ARC of their edge seen at a radius, on the count clamped to one dot.
function edgeCenters(D, width, height) {
    const C = new Float32Array(D.length);
    const weight = new Uint8Array(D.length);
    for (let p = 0; p < D.length; p++) {
        weight[p] = Number.isNaN(D[p]) ? 0 : 1;
        C[p] = weight[p] ? Math.min(1, D[p]) : 0;
    }
    const grad = houghGradient(C, weight, width, height);
    return r => houghCenters(C, weight, width, height, r, grad).filter(c => c.arc >= DOT_MIN_ARC).length;
}

/**
 * Another family's radius for a translucent family with no whole isolated dot, when its own dots show at that
 * radius: figures draw every group with one marker size, and a small group can have every dot crossed by a whisker
 * or touching its box (protrusion2's orange: 3 centers with half their edge seen at the green's 30.1 px, none at 25
 * or 35). The same test as an opaque family's sibling (DOT_MIN_STRONG centers of DOT_MIN_ARC); the blur is read
 * from the dots' boundaries. Null when no radius shows them.
 *
 * @param {Float32Array} D  overlap count, NaN where unknown
 * @param {number} width
 * @param {number} height
 * @param {number[]} radii  px, the other families'
 * @returns {{radius: number, blur: number, centers: number}|null}  px
 */
export function siblingRadius(D, width, height, radii) {
    if (!radii.length)
        return null;
    const centersAt = edgeCenters(D, width, height);
    const best = radii.map(radius => ({ radius, centers: centersAt(radius) })).sort((a, b) => b.centers - a.centers)[0];
    if (best.centers < DOT_MIN_STRONG)
        return null;
    const edge = boundaryBlur(D, width, height);
    return { ...best, blur: Math.sqrt(Math.max(0, edge.sigma ** 2 - 1 / 12)) };
}

/**
 * A translucent family's radius and edge blur when no dot is a whole isolated one (landing1: every dot touches a
 * gray line, a box or a violin's outline): the distance transform's radius, then larger radii by Hough, as for
 * opaque families (the largest that keeps RADIUS_SCAN_KEEP of the most centers), on the count clamped to one dot (a
 * stack or a dot's darker rim is still that dot's disc). Centers with half their edge seen count: dots that dense
 * show few strong arcs (landing1, at 10–11 px: 2 and 3 strong, 9 and 11 with half an arc, none at other radii).
 * Null without DOT_MIN_STRONG such centers.
 *
 * @param {Float32Array} D  overlap count, NaN where unknown
 * @param {number} width
 * @param {number} height
 * @returns {{radius: number, blur: number, centers: number}|null}  px
 */
export function shapeRadius(D, width, height) {
    const r0 = radiusFromDistance(D, width, height);
    if (!r0)
        return null;
    const centersAt = edgeCenters(D, width, height);
    const scan = [];
    for (let r = r0; r <= RADIUS_SCAN_MAX * r0; r += RADIUS_SCAN_STEP)
        scan.push({ r, n: centersAt(r) });
    const most = Math.max(...scan.map(t => t.n));
    if (most < DOT_MIN_STRONG)
        return null;
    const pick = scan.filter(t => t.n >= RADIUS_SCAN_KEEP * most).pop();
    const edge = boundaryBlur(D, width, height);
    return { radius: pick.r, blur: Math.sqrt(Math.max(0, edge.sigma ** 2 - 1 / 12)), centers: pick.n };
}

// Radii to try from the connected pieces of a family's pixels, each piece by its deepest point (distance to outside)
// plus the half pixel the transform loses between pixel centers; pieces shallower than DOT_MIN_RADIUS are left out.
// (1) The median weighted by size: the whole dots and the dots joined by a line decide, not the slivers of partly
// hidden ones (kite1 online, its red band divided out: pieces 5.8–8 px deep, 2500 px in all, against 12.0 and 12.2,
// 5900 px). (2) The deepest compact piece (area within COMPACT_MAX_AREA of a disc's at its depth): where a line of the
// dots' color is as large as the dots, the weighted median falls between them (kite2: its curve, 6.4–7 px deep and
// 15 700 px, 50 times a disc's area, against 21 dots 16–16.6 px deep, 18 000 px, gave 9.6). (3) The deepest piece:
// dots are usually their color's thickest ink, even when a line joins them all into one piece with the axes and
// error bars (delta1: that piece 14.1 px deep, 47% of the black, against text 3–5 px deep, 49%, so the median fell on
// the bold panel label "g", 8.1 px, and the deepest compact piece was a scrap of text).
function componentRadii(D, width, height) {
    const inside = new Uint8Array(D.length);
    for (let p = 0; p < D.length; p++)
        inside[p] = D[p] > 0.5 ? 0 : 1;
    const d2 = distanceTransform(inside, width, height);
    const pieces = [];
    for (const comp of components(D, width, height)) {
        let most = 0;
        for (const p of comp)
            most = Math.max(most, d2[p]);
        if (most >= DOT_MIN_RADIUS * DOT_MIN_RADIUS && most !== Infinity)
            pieces.push({ depth: Math.sqrt(most), size: comp.length });
    }
    if (!pieces.length)
        return [];
    pieces.sort((a, b) => a.depth - b.depth);
    const half = pieces.reduce((t, q) => t + q.size, 0) / 2;
    let sum = 0;
    let median = null;
    for (const q of pieces) {
        sum += q.size;
        if (median === null && sum >= half)
            median = q.depth;
    }
    const compact = pieces.filter(q => q.size <= COMPACT_MAX_AREA * Math.PI * q.depth * q.depth);
    const deepest = compact.length ? compact[compact.length - 1].depth : null;
    return [...new Set([median, deepest, pieces[pieces.length - 1].depth].filter(d => d !== null))].map(d => d + DT_RADIUS_OFFSET);
}

// A plain family's ground for blurred dots: the near zone grown along their edges, and the family's background and
// the local one read beyond them (right beside the family's pixels the commonest color can be a tone of the edge
// itself: 187 for black dots on white, in a rendering with a Gaussian's smooth edge). Null where the zone hardly
// grows (SOFT_EDGE_MIN_GROWTH).
function softGround(scan, ground) {
    const { image, color, mask, own, others } = ground;
    const beyond = dilate(mask, image.width, image.height, NEAR_FAMILY + SOFT_EDGE_MAX_REACH);
    const B = ringBackground(image, beyond) ?? ground.B;
    const near = ground.near.slice();
    const grown = alongSoftEdges(image, near, B, color);
    if (grown < SOFT_EDGE_MIN_GROWTH * own.reduce((sum, v) => sum + v, 0))
        return null;
    const pageColors = [...scan.backgrounds, B].filter(P => colorDistance(P, color) > CLUSTER_TOL);
    return { ...ground, B, near, pageColors, bg: localBackground(image, near, pageColors, others) };
}

// The near zone of a family's pixels, grown outward along blurred dots' edges (SOFT_EDGE_MIN_STEP), in place; returns
// the pixels added. O(pixels · SOFT_EDGE_MAX_REACH).
function alongSoftEdges(image, near, B, color) {
    const { data } = image;
    const w = color.map((v, ch) => v - B[ch]);
    const ww = w[0] * w[0] + w[1] * w[1] + w[2] * w[2];
    // a pixel's share of the family's color against its usual background, or -1 off the line between them
    return downhill(near, image.width, p => {
        const v = [data[4 * p] - B[0], data[4 * p + 1] - B[1], data[4 * p + 2] - B[2]];
        const m = (v[0] * w[0] + v[1] * w[1] + v[2] * w[2]) / ww;
        return Math.hypot(v[0] - m * w[0], v[1] - m * w[1], v[2] - m * w[2]) <= OFF_MIX_ABS ? m : -1;
    });
}

// A zone grown outward, in place, down the slope of a color's share (0–1 per pixel; below 0: not readable as one):
// through pixels with SOFT_EDGE_MIN_SHARE of it or more and less than a neighbor in the zone by SOFT_EDGE_MIN_STEP,
// for SOFT_EDGE_MAX_REACH layers. With `toEnd`, on through the slope's faint end too (less than
// SOFT_EDGE_MIN_SHARE, and no more than the neighbor's). Returns the pixels added.
function downhill(zone, width, share, toEnd = false) {
    const n = zone.length;
    const near = zone;
    let front = [];
    for (let p = 0; p < n; p++) {
        if (near[p])
            front.push(p);
    }
    let grown = 0;
    for (let layer = 0; layer < SOFT_EDGE_MAX_REACH && front.length; layer++) {
        const next = [];
        for (const p of front) {
            const from = share(p);
            const x = p % width;
            for (const q of [x > 0 ? p - 1 : -1, x < width - 1 ? p + 1 : -1, p - width, p + width]) {
                if (q < 0 || q >= n || near[q])
                    continue;
                const m = share(q);
                if (m >= SOFT_EDGE_MIN_SHARE ? from - m >= SOFT_EDGE_MIN_STEP : toEnd && m >= 0 && m <= from) {
                    near[q] = 1;
                    next.push(q);
                }
            }
        }
        // (the first layer starts from the whole zone; the later ones from what the last one added)
        front = next;
        grown += next.length;
    }
    return grown;
}

/**
 * The background under and around a family, per pixel: the color of the nearest pixel that is not close to the
 * family's color. For an edge pixel that is the pixel just outside the dot, which is what the dot's edge was
 * blended with (a window mean mixed in other marks: the synthetic swarms' median lines).
 */
function localBackground(image, near, colors, others) {
    const { width, height, data } = image;
    const n = width * height;
    // Only background colors and other families' colors can be a dot's background. Other ink can't: a median line
    // across a cantons dot took the line beyond the dot as its "background" and read as no dot there.
    const known = new Uint8Array(n);
    const isBackground = nearAnyColor(data, colors, INK_TOL);
    const isOther = nearAnyColor(data, others, CLUSTER_TOL);
    for (let p = 0; p < n; p++) {
        if (!near[p])
            known[p] = isBackground(p) || isOther(p) ? 1 : 0;
    }
    const nearest = nearestTransform(known, width, height);
    const bg = new Float32Array(3 * n);
    for (let p = 0; p < n; p++) {
        const q = nearest[p] >= 0 ? nearest[p] : p;
        bg[3 * p] = data[4 * q];
        bg[3 * p + 1] = data[4 * q + 1];
        bg[3 * p + 2] = data[4 * q + 2];
    }
    return bg;
}

// What a page-filled family's dots lie on, per pixel: the color of the nearest pixel away from the family (`near`)
// in a flat patch (its 4 neighbors within INK_TOL of it) and not of the outline's color. Flat, as a bar's inside or
// the page is, so that text and lines beside a dot aren't its background. 0–255, 3 per pixel.
function surroundings(image, near, outline) {
    const { width, height, data } = image;
    const n = width * height;
    const known = new Uint8Array(n);
    const same = (p, q) => Math.hypot(data[4 * p] - data[4 * q], data[4 * p + 1] - data[4 * q + 1], data[4 * p + 2] - data[4 * q + 2]) <= INK_TOL;
    const isOutline = nearColor(data, outline, CLUSTER_TOL);
    for (let y = 1; y < height - 1; y++) {
        for (let x = 1; x < width - 1; x++) {
            const p = y * width + x;
            if (near[p] || isOutline(p))
                continue;
            known[p] = same(p, p - 1) && same(p, p + 1) && same(p, p - width) && same(p, p + width) ? 1 : 0;
        }
    }
    const nearest = nearestTransform(known, width, height);
    const bg = new Float32Array(3 * n);
    for (let p = 0; p < n; p++) {
        const q = nearest[p] >= 0 ? nearest[p] : p;
        bg[3 * p] = data[4 * q];
        bg[3 * p + 1] = data[4 * q + 1];
        bg[3 * p + 2] = data[4 * q + 2];
    }
    return bg;
}

// 1 where a pixel is another family's ring: within `reach` of that family's fill and nearer it (by STROKE_TIE) than to
// this family's fill (`own`, its thick parts), or, for a ring marked `whole`, wherever it is within reach
// (outlineRings); null without rings. O(families · pixels).
function ringOwners(image, own, rings) {
    if (!rings?.length)
        return null;
    const { width, height } = image;
    const n = width * height;
    const dist = idx => {
        const out = new Float32Array(n);
        for (let p = 0; p < n; p++) {
            const q = idx[p];
            out[p] = q < 0 ? Infinity : Math.hypot(p % width - q % width, Math.floor(p / width) - Math.floor(q / width));
        }
        return out;
    };
    const dOwn = dist(nearestTransform(own, width, height));
    const out = new Uint8Array(n);
    for (const { fill, reach, whole } of rings) {
        const theirs = thickMembers(image, [fill]);
        for (let p = 0; p < n; p++)
            theirs[p] = own[p] ? 0 : theirs[p];
        const d = dist(nearestTransform(theirs, width, height));
        for (let p = 0; p < n; p++) {
            if (!own[p] && !theirs[p] && d[p] <= reach && (whole || d[p] < dOwn[p] - STROKE_TIE))
                out[p] = 1;
        }
    }
    return out;
}

/**
 * One outlined or stroked family's ring image, for fitting it by its rings in drawing order (outline.js): each pixel's
 * weight of the outline in the triangle page–fill–outline, scaled so a full outline reads FAMILY_RING_P. The fill and
 * the page read 0, so an outline crossing a lower dot of the family is the upper dot's edge, which the opaque union
 * can't see (degradation1: two mint dots 6–10 px apart read as one solid blob, fill and outline alike "dot").
 * Unknown: pixels off the triangle, other families' fills and the rings nearer them, and outline-colored ink beyond
 * OUTLINE_REACH of this family's thick fill (text, axes).
 *
 * @param {import('./image.js').RgbaImage} image
 * @param {number[]} fill  0–255
 * @param {number[]} outline  0–255
 * @param {number[][]} backgrounds  0–255, the page's first
 * @param {number[][]} others  0–255, the other families' fill colors
 * @returns {{P: Float32Array, unknown: Uint8Array}}  P 0–FAMILY_RING_P
 */
export function familyRings(image, fill, outline, backgrounds, others) {
    const { width, height, data } = image;
    const n = width * height;
    const own = colorMask(image, fill, CLUSTER_TOL);
    let reach = dilate(opening(own, width, height, THICK_OPENING), width, height, OUTLINE_REACH);
    const theirs = ringOwners(image, own, others.map(c => ({ fill: c, reach: STROKE_RING })));
    const otherFill = others.length ? thickMembers(image, others) : null;
    const B = backgrounds[0];
    const isPage = p => backgrounds.some(Q => Math.hypot(data[4 * p] - Q[0], data[4 * p + 1] - Q[1], data[4 * p + 2] - Q[2]) <= INK_TOL);
    const notOurs = p => (otherFill && !own[p] && otherFill[p]) || (theirs && theirs[p]);
    const a1 = fill.map((v, ch) => v - B[ch]);
    const m11 = a1[0] * a1[0] + a1[1] * a1[1] + a1[2] * a1[2];
    // A pixel's weights of the fill and of an outline color in the triangle with the page, and whether it lies in
    // the triangle (`on`).
    const weightsFor = edge => {
        const a2 = edge.map((v, ch) => v - B[ch]);
        const m12 = a1[0] * a2[0] + a1[1] * a2[1] + a1[2] * a2[2];
        const m22 = a2[0] * a2[0] + a2[1] * a2[1] + a2[2] * a2[2];
        const det = m11 * m22 - m12 * m12;
        const planar = det > COLLINEAR_SIN2 * m11 * m22;
        const fo = fill.map((c, ch) => c - edge[ch]);
        const weights = p => {
            const r = data[4 * p];
            const g = data[4 * p + 1];
            const b = data[4 * p + 2];
            const v = [r - B[0], g - B[1], b - B[2]];
            let wf;
            let ws;
            if (planar) {
                const r1 = v[0] * a1[0] + v[1] * a1[1] + v[2] * a1[2];
                const r2 = v[0] * a2[0] + v[1] * a2[1] + v[2] * a2[2];
                wf = (m22 * r1 - m12 * r2) / det;
                ws = (m11 * r2 - m12 * r1) / det;
            }
            else {
                // page, fill and outline on one line (gray fills in black outlines): the position on the outline–fill
                // segment, or beyond the fill on the page–outline one (an outline's rim on the page passes through the
                // fill's color too, and reads as fill: unavoidable on one line)
                const e = [r - edge[0], g - edge[1], b - edge[2]];
                const k = (e[0] * fo[0] + e[1] * fo[1] + e[2] * fo[2]) / (fo[0] * fo[0] + fo[1] * fo[1] + fo[2] * fo[2]);
                if (k <= 1) {
                    wf = Math.max(0, k);
                    ws = 1 - Math.max(0, k);
                }
                else {
                    wf = 0;
                    ws = (v[0] * a2[0] + v[1] * a2[1] + v[2] * a2[2]) / m22;
                }
            }
            const off = Math.hypot(v[0] - wf * a1[0] - ws * a2[0], v[1] - wf * a1[1] - ws * a2[1], v[2] - wf * a1[2] - ws * a2[2]);
            const span = Math.hypot(wf * a1[0] + ws * a2[0], wf * a1[1] + ws * a2[1], wf * a1[2] + ws * a2[2]);
            return { wf, ws, on: off <= Math.max(OFF_MIX_ABS, OFF_MIX_REL * span) && wf >= -STROKE_SLACK && ws >= -STROKE_SLACK && wf + ws <= MAX_MIX };
        };
        return { planar, weights };
    };
    const seen = weightsFor(outline);
    // (on one line there is no telling a darker outline from more of this one)
    const { weights } = seen.planar ? weightsFor(outlineItself(seen.weights, reach, own, isPage, notOurs, fill, outline)) : seen;
    // A blurred ring's tail runs on beyond the reach, falling: the outline's still, and not other ink, which is flat
    // (stiffness: rings 6 px from the fill's color to the page). To its faint end: a last pixel left unknown is
    // masked with its neighbors, and with the rings' rims in the mask their radius read 9.6 for 8.7. Only where the
    // reach's edge runs through the rings, so that the tails add RING_TAIL_MIN_GROWTH × that edge's pixels or more:
    // around crisp rings the edge is on the page, and what lies beyond it is other ink in the outline's color
    // touching a ring (degradation1: gray error bars' edges joined the gray rings, and none was whole).
    const tails = reach.slice();
    const grown = downhill(tails, width, p => {
        if (isPage(p) || notOurs(p))
            return -1;
        const w = weights(p);
        return w.on ? Math.max(0, w.ws) : -1;
    }, true);
    let edge = 0;
    for (let p = width; p < n - width; p++)
        edge += reach[p] && !(reach[p - 1] && reach[p + 1] && reach[p - width] && reach[p + width]) ? 1 : 0;
    if (grown >= RING_TAIL_MIN_GROWTH * edge)
        reach = tails;
    const P = new Float32Array(n);
    const unknown = new Uint8Array(n);
    for (let p = 0; p < n; p++) {
        if (isPage(p))
            continue;
        if (notOurs(p) || !reach[p]) {
            unknown[p] = 1;
            continue;
        }
        const w = weights(p);
        if (!w.on) {
            unknown[p] = 1;
            continue;
        }
        P[p] = FAMILY_RING_P * Math.min(1, Math.max(0, w.ws));
    }
    return { P, unknown };
}

// An outline's own color, where the darkest color seen in it is a mix: a stroke a pixel or two wide, blurred, is its
// own color nowhere, and its core is mixed with the fill (stiffness: black around a blue of 49, 179, 253, seen as 16,
// 57, 83 at its darkest). Its inner side then lies on the line from the fill to the color seen, but its outer side,
// mixed with the page, doesn't lie on the line from the page to it: it reads as the color seen less some fill (gray,
// for stiffness, where that line is blue), every ring pixel on the page's side was off the triangle and unknown, and
// the mask took the rings whole (39 dots found of 305). The color itself lies beyond the one seen, away from the
// fill, by the share of fill the outer side lacks: -wf / ws, over the pixels within reach that are part page and no
// fill, at their OUTLINE_MIX_QUANTILE, when that is OUTLINE_MIX_MIN_SHARE or more over OUTLINE_MIX_MIN_PIXELS
// pixels; kept within 0–255. Otherwise the color seen. The upper end, since most of a blurred ring's outer pixels
// have some fill in them too and lack less (stiffness: 0.24 at the median, 0.32 at the ninth decile, which is black;
// with the median's color the rings fitted their model at rms 0.057, with black at 0.019). Too far is the safer
// side: every mix of the page, the fill and the outline stays within the triangle of a color beyond the outline's.
function outlineItself(weights, reach, own, isPage, notOurs, fill, outline) {
    const lacks = [];
    for (let p = 0; p < reach.length; p++) {
        if (!reach[p] || own[p] || isPage(p) || notOurs(p))
            continue;
        const { wf, ws } = weights(p);
        if (ws >= OUTLINE_MIX_MIN_OUTLINE && wf <= OUTLINE_MIX_MAX_FILL && wf + ws <= OUTLINE_MIX_MAX_COVER)
            lacks.push(-wf / ws);
    }
    if (lacks.length < OUTLINE_MIX_MIN_PIXELS)
        return outline;
    const share = Math.min(OUTLINE_MIX_MAX_SHARE, quantile(lacks, OUTLINE_MIX_QUANTILE));
    if (share < OUTLINE_MIX_MIN_SHARE)
        return outline;
    // the color seen is (1 - k) × the outline's + k × the fill's with k = share / (1 + share)… as far as 0–255 allows
    let t = share / (1 - share);
    for (let ch = 0; ch < 3; ch++) {
        const d = outline[ch] - fill[ch];
        if (d < 0)
            t = Math.min(t, outline[ch] / -d);
        else if (d > 0)
            t = Math.min(t, (255 - outline[ch]) / d);
    }
    return outline.map((v, ch) => v + t * (v - fill[ch]));
}

// 1 on the thick parts (an opening) of the pixels within CLUSTER_TOL of any of `colors`
function thickMembers(image, colors) {
    const { width, height, data } = image;
    const m = new Uint8Array(width * height);
    const isMember = nearAnyColor(data, colors, CLUSTER_TOL);
    for (let p = 0; p < m.length; p++)
        m[p] = isMember(p) ? 1 : 0;
    return opening(m, width, height, THICK_OPENING);
}

/**
 * The stroke around a fill color, read from the rings of its isolated components, or null when they show none.
 *
 * @param {import('./image.js').RgbaImage} image
 * @param {number[]} color  the fill, 0–255
 * @param {number[]} B  the page background, 0–255
 * @returns {{color: number[], off: number, rings: number, fillRadius: number}|null}  color 0–255; off: its distance
 *   from the background–fill line, 0–255 units; rings: isolated fills read; fillRadius: px, the radius of a disc
 *   of their median area (the fill's own color: the rim's mixes with the stroke are left out)
 */
export function strokeColor(image, color, B) {
    const { width, height, data } = image;
    const n = width * height;
    const own = new Float32Array(n);
    const isOwn = nearColor(data, color, CLUSTER_TOL);
    const onPage = nearColor(data, B, INK_TOL);
    for (let p = 0; p < n; p++)
        own[p] = isOwn(p) ? 1 : 0;
    const comps = components(own, width, height);
    const label = new Int32Array(n).fill(-1);
    comps.forEach((c, k) => {
        for (const p of c)
            label[p] = k;
    });
    const nearest = nearestTransform(Uint8Array.from(own), width, height);
    const clear = new Float64Array(comps.length);
    const clearBg = new Float64Array(comps.length);
    const ring = comps.map(() => []);
    for (let p = 0; p < n; p++) {
        const q = nearest[p];
        if (own[p] || q < 0)
            continue;
        const d = Math.hypot(p % width - q % width, Math.floor(p / width) - Math.floor(q / width));
        const k = label[q];
        if (d <= STROKE_RING) {
            ring[k].push(p);
        }
        else if (d <= STROKE_CLEAR) {
            clear[k]++;
            if (onPage(p))
                clearBg[k]++;
        }
    }
    const u = color.map((v, ch) => v - B[ch]);
    const uu = u[0] * u[0] + u[1] * u[1] + u[2] * u[2];
    const pixels = [];
    const sizes = [];
    comps.forEach((c, k) => {
        if (c.length < STROKE_COMPONENT[0] || c.length > STROKE_COMPONENT[1] || !clear[k] || clearBg[k] < STROKE_CLEAR_SHARE * clear[k])
            return;
        sizes.push(c.length);
        for (const p of ring[k]) {
            const v = [0, 1, 2].map(ch => data[4 * p + ch] - B[ch]);
            if (Math.hypot(...v) <= INK_TOL)
                continue;
            const m = (v[0] * u[0] + v[1] * u[1] + v[2] * u[2]) / uu;
            pixels.push({ p, off: Math.hypot(v[0] - m * u[0], v[1] - m * u[1], v[2] - m * u[2]) });
        }
    });
    if (sizes.length < STROKE_MIN_RINGS || !pixels.length)
        return null;
    pixels.sort((a, b) => a.off - b.off);
    const top = pixels.slice(Math.floor(STROKE_TOP * (pixels.length - 1)));
    const S = [0, 1, 2].map(ch => top.reduce((t, q) => t + data[4 * q.p + ch], 0) / top.length);
    const { off } = alongLine(S.map((x, ch) => x - B[ch]), u);
    return off >= STROKE_MIN_OFF ? { color: S, off, rings: sizes.length, fillRadius: Math.sqrt(median(sizes) / Math.PI) } : null;
}

/**
 * An opaque family from a color: the coverage image D (0 its local background … 1 the color; NaN where a pixel is
 * neither, or contrast is too low to tell), its typical background (the most common color around it), the radius
 * (isolated dots, else the distance transform), and how many filled, dot-shaped edges a Hough vote finds.
 *
 * In four steps: where the family's pixels are and what they lie on (opaqueGround), each pixel's coverage by one of
 * four readings (opaqueCoverage: a plain fill, an outlined one, a stroked one, a fill in the page's color), the
 * radius (opaqueRadius) and the centers that pass the dot test (dotCenters).
 *
 * With an outline color, a dot is its fill inside an opaque ring of the outline (JMP's outlined markers: an outline
 * disc 1.7 px larger in radius under the fill): every pixel between the two colors is covered, and the outer rim is
 * read as the outline's coverage of the background. The boundary between fill and outline isn't used.
 *
 * With a stroke (strokeColor), a pixel inside the triangle background–fill–stroke is dot by the weight it takes off
 * the background, so the ring counts as dot; the fill's own weight decides whether a center is this family's (its
 * inner disc must be the fill: a darker family's fill can lie inside the triangle, near the stroke). With `rings`,
 * the other families' fills, a ring pixel nearer one of them than to this family's fill reads 0.
 *
 * Near the family, a pixel is read against its nearest other color, which keeps the edges where dots touch other ink;
 * where that color isn't the page, the pixel is other ink, recorded in `edgeInk` (1 + the index of the `others` color
 * it is, as `inkColors` lists them, or EDGE_INK_UNLISTED). Whether this family's dots go on under it depends on which
 * is drawn on top (opaqueFamilies' `under`; layers.js buildMask).
 *
 * @param {Scan} scan  its image, backgrounds (the page first) and tolerances
 * @param {number[]} color  the fill, 0–255
 * @param {number[][]} [others]  other families' colors and other ink, 0–255
 * @param {number[]|null} [outline]  0–255
 * @param {object} [opts]
 * @param {number[]|null} [opts.stroke]  0–255
 * @param {number} [opts.fillRadius]  px, the fill inside the stroke
 * @param {{fill: number[], reach: number, whole?: boolean}[]|null} [opts.rings]  other families' fills, and how far
 *   (px) their rings reach beyond them; whole: every thin pixel of this color within reach is theirs (outlineRings)
 * @param {number} [opts.radius]  px, the dots' radius when known otherwise (a found family's outline color as a family
 *   of its own: its isolated components are text's specks)
 * @returns {OpaqueMeasure}
 */
export function measureOpaque(scan, color, others = [], outline = null, opts = {}) {
    const { width, height } = scan.image;
    const stroke = opts.stroke ?? null;
    let ground = opaqueGround(scan, color, others, outline, stroke, opts.rings ?? null);
    let cover = opaqueCoverage(ground);
    if (!stroke)
        fillSaturatedOverlaps(scan.image, cover.D, ground.B, color, ground.others);
    let shape = opaqueRadius(ground, cover, scan.tol, opts.radius);
    // a plain family with no whole dot: read again as blurred dots, and so taken if that shows whole ones
    const soft = !shape.isolated.length && !outline && !stroke && !ground.paged ? softGround(scan, ground) : null;
    if (soft) {
        const softCover = opaqueCoverage(soft);
        fillSaturatedOverlaps(scan.image, softCover.D, soft.B, color, soft.others);
        const softShape = opaqueRadius(soft, softCover, scan.tol, opts.radius, true);
        if (softShape.isolated.length) {
            ground = soft;
            cover = softCover;
            shape = softShape;
        }
    }
    const { paged, own, mask, B } = ground;
    const { D, edgeInk, fillW } = cover;
    const fitColor = paged ? outline : color;
    const line = colorLine(B, fitColor);
    const table = [B, fitColor];
    const { isolated, byFill } = shape;
    let { radius } = shape;
    let dots = 0;
    let strongDots = 0;
    let coverage = 0;
    let dotsUnder = 0;
    let coverageUnder = 0;
    let found = [];
    let under = [];
    if (radiusPlausible(radius, isolated.length)) {
        const strong = cs => cs.filter(c => c.arc >= DOT_STRONG_ARC).length;
        const at = dotCenters(scan, ground, cover, shape, opts.fillRadius);
        ({ radius } = at);
        dots = at.centers.length;
        strongDots = strong(at.centers);
        found = at.centers.map(c => ({ x: c.x, y: c.y }));
        // a stroked family's coverage is its fill's: D also covers the stroke's color, which is text and axes too
        // (hair3: the black ink diluted it to 0.26); so is one measured by its fills, and an outlined one's only where
        // a disc of THICK_FILL_RADII × the fill radius fits: a line of the fill's color thinner than a fill is no dot
        // (phase1: gray 136's curves, 13 px wide beside fills of 17, held most of its pixels; coverage 0.29 → 0.63)
        const ofFamily = stroke ? fillW : byFill ? (outline ? opening(own, width, height, THICK_FILL_RADII * byFill.fillRadius) : own) : D;
        coverage = shareInDiscs(ofFamily, at.centers, radius, width, height);
        dotsUnder = at.under.length;
        under = at.under.map(c => ({ x: c.x, y: c.y }));
        coverageUnder = dotsUnder ? shareInDiscs(ofFamily, [...at.centers, ...at.under], radius, width, height) : coverage;
    }
    const pixels = mask.reduce((t, v) => t + v, 0);
    return {
        kind: 'opaque', seed: color, outline, stroke, line, background: B, table, D, isolated, radius, dots, strongDots, coverage, pixels,
        dotsUnder, coverageUnder,
        edgeInk, inkColors: ground.others, centers: found, centersUnder: under,
        ...(byFill ? { fillRadius: byFill.fillRadius } : {}),
        ...(paged ? { pageFill: true, fitColor, fillCenters: paged.centers } : {}),
        levels: [{ t: colorDistance(B, fitColor), color: fitColor, count: pixels }],
        beyond: [], fit: { alpha: 1, c: [...color], rms: 0, maxAbs: 0, underdetermined: true }, translucent: false,
    };
}

// Where an opaque family's pixels are and what they lie on, for the readings of opaqueCoverage: the fill's pixels
// (own), with the outline's in reach of them (mask) and its thick parts, the family's usual background B and the
// local one near it (bg), how far a pixel may lie off a mix (offAbs, from the color's noise), and the pixels that are
// other families' (theirs, otherFill). `others` comes back without colors the page hides.
function opaqueGround(scan, color, others, outline, stroke, rings) {
    const { image, backgrounds } = scan;
    const { width, height, data } = image;
    const n = width * height;
    const isColor = nearColor(data, color, CLUSTER_TOL);
    const isOutline = outline ? nearColor(data, outline, CLUSTER_TOL) : null;
    const page = backgrounds[0];
    // (the page's color itself: a light fill near it, as tumour1's 237 gray, is a color of its own)
    const isPage = c => colorDistance(c, page) <= INK_TOL;
    // an outlined fill in the page's color is the page its outlines enclose (pageFills); for the fit it is the outline's
    // color on the page, as the fill shows no contrast
    const paged = outline && isPage(color) ? pageFills(image, color) ?? { mask: new Uint8Array(n), discs: [], centers: [] } : null;
    // another family's fill in the page's color can't be told from the page by color: those dots are found by shape
    others = others.filter(O => !isPage(O));
    const ringFills = rings?.filter(r => !isPage(r.fill)) ?? null;
    const own = paged ? paged.mask : new Uint8Array(n);
    // a pixel nearer another family's color is that family's, though within this color's tolerance too (score1: the
    // teal dots, 39.9 units from the gray of a dashed line and label leaders, made the gray a family of 150 dots)
    const nearer = p => {
        const dOwn = Math.hypot(data[4 * p] - color[0], data[4 * p + 1] - color[1], data[4 * p + 2] - color[2]);
        return others.some(O => Math.hypot(data[4 * p] - O[0], data[4 * p + 1] - O[1], data[4 * p + 2] - O[2]) < dOwn);
    };
    if (!paged) {
        for (let p = 0; p < n; p++)
            own[p] = isColor(p) && !nearer(p) ? 1 : 0;
    }
    // the reach is from the fill's thick parts: an antialiased edge between the outline's color and a lighter one
    // passes through the fill's color (phase1: every black edge, text's and other grays' outlines, had 82-gray
    // pixels, and the reach covered them all)
    // (page-colored fills are exact, enclosed regions with no stray rims, so they reach from their edge, a pixel
    // farther: hair1's outer rims lay 20.4–21.5 px out on fills of 17.2)
    const thickOwn = outline ? opening(own, width, height, THICK_OPENING) : null;
    const ownReach = outline ? dilate(paged ? own : thickOwn, width, height, OUTLINE_REACH + (paged ? 1 : 0)) : null;
    const mask = new Uint8Array(n);
    for (let p = 0; p < n; p++)
        mask[p] = own[p] || (outline && ownReach[p] && isOutline(p)) ? 1 : 0;
    const thick = opening(mask, width, height, THICK_OPENING);
    // a stroked fill's surroundings are its ring: its background lies beyond
    const body = thick.some(v => v) ? thick : mask;
    const B = paged ? page : ringBackground(image, stroke ? dilate(body, width, height, STROKE_RING) : body) ?? backgrounds[0];
    const near = dilate(mask, width, height, NEAR_FAMILY + (stroke ? STROKE_RING : 0));
    // the family's color noise: RMS distance from its color over interior pixels (all 8 neighbors in the family)
    let noiseSum = 0;
    let noiseN = 0;
    for (let y = 1; y < height - 1; y++) {
        for (let x = 1; x < width - 1; x++) {
            const p = y * width + x;
            if (!(thickOwn ?? thick)[p] || !own[p - 1] || !own[p + 1] || !own[p - width] || !own[p + width])
                continue;
            noiseSum += (data[4 * p] - color[0]) ** 2 + (data[4 * p + 1] - color[1]) ** 2 + (data[4 * p + 2] - color[2]) ** 2;
            noiseN++;
        }
    }
    const offAbs = Math.max(OFF_MIX_ABS, OFF_MIX_NOISE * (noiseN ? Math.sqrt(noiseSum / noiseN) : 0));
    // background colors, but not this family's own (a dot color can be common enough to count as a background)
    const pageColors = [...backgrounds, B].filter(P => colorDistance(P, color) > CLUSTER_TOL
        && (!outline || colorDistance(P, outline) > CLUSTER_TOL)
        && (!stroke || colorDistance(P, stroke) > CLUSTER_TOL));
    const bg = localBackground(image, near, pageColors, others);
    // from the family's thick parts: a stroke near its color (engagement1's navy) is thin, and joins its color mask
    const theirs = ringOwners(image, thick.some(v => v) ? thick : own, ringFills);
    // other families' fills, thick parts only: a stroke near another family's color is still this dot's ring
    const otherFill = stroke ? thickMembers(image, others) : null;
    // a page-colored fill's antialiased inner rim, next to it, is inside the dot; farther out the outline is mixed with
    // the page
    const besideFill = paged ? dilate(own, width, height, 1.5) : null;
    // … whatever the dot lies on: the page, or a bar (tumour1's gray, 171, lies on the line from white to black, and
    // read as 0.36 of the outline: every dot over a bar read a pixel larger, and the fit put in pairs)
    const rimBg = paged ? surroundings(image, near, outline) : null;
    return {
        image, color, outline, stroke, others, page, paged, own, mask, ownReach, B, near, offAbs, pageColors, bg, theirs,
        otherFill, besideFill, rimBg,
    };
}

/**
 * Each pixel's coverage by the family's dots, by the reading its kind of dot needs.
 *
 * @param {object} ground  opaqueGround's
 * @returns {{D: Float32Array, edgeInk: Uint8Array, fillW: Float32Array|null, fillK: Float32Array|null}}  D: 0–1
 *   or NaN; fillW: a stroked fill's own weight; fillK: with an outline, how far each pixel is from the outline
 *   toward the fill (the fill's antialiased coverage)
 */
function opaqueCoverage(ground) {
    const n = ground.image.width * ground.image.height;
    const cover = {
        D: new Float32Array(n),
        edgeInk: new Uint8Array(n),
        fillW: ground.stroke ? new Float32Array(n) : null,
        fillK: ground.outline ? new Float32Array(n) : null,
    };
    if (ground.paged)
        readPageFills(ground, cover);
    else if (ground.stroke)
        readStroked(ground, cover);
    else if (ground.outline)
        readOutlined(ground, cover);
    else
        readPlain(ground, cover);
    return cover;
}

// whether no dot of the family can be at a pixel: away from the family on a background color, or on the ring of
// another family's dot
function noDotAt(ground) {
    const { near, theirs } = ground;
    const onPageColor = nearAnyColor(ground.image.data, ground.pageColors, INK_TOL);
    return p => (!near[p] && onPageColor(p)) || Boolean(theirs && theirs[p]);
}

// A fill in the page's color inside an outline: the fill's own pixels are the dot; in the outline's reach a pixel is
// the outline mixed with whatever the dot lies on (rimBg).
function readPageFills(ground, { D, fillK }) {
    const { color, outline, own, ownReach, besideFill, rimBg, offAbs } = ground;
    const { data } = ground.image;
    const isColor = nearColor(data, color, CLUSTER_TOL);
    const onPage = nearColor(data, ground.page, INK_TOL);
    for (let p = 0; p < D.length; p++) {
        const r = data[4 * p];
        const g = data[4 * p + 1];
        const b = data[4 * p + 2];
        if (onPage(p)) {
            D[p] = own[p];
            fillK[p] = own[p];
            continue;
        }
        // beyond the outlines' reach, a faint pixel is the page (antialiasing), other ink unknown
        if (!ownReach[p]) {
            D[p] = isColor(p) ? 0 : NaN;
            continue;
        }
        const w = [rimBg[3 * p], rimBg[3 * p + 1], rimBg[3 * p + 2]];
        const po = outline.map((v, ch) => v - w[ch]);
        const poo = po[0] * po[0] + po[1] * po[1] + po[2] * po[2];
        const v = [r - w[0], g - w[1], b - w[2]];
        const m = (v[0] * po[0] + v[1] * po[1] + v[2] * po[2]) / poo;
        const off = Math.hypot(v[0] - m * po[0], v[1] - m * po[1], v[2] - m * po[2]);
        if (poo < MIN_CONTRAST * MIN_CONTRAST || off > Math.max(offAbs, OFF_MIX_REL * Math.abs(m) * Math.sqrt(poo)) || m > MAX_MIX) {
            D[p] = NaN;
            continue;
        }
        D[p] = besideFill[p] ? 1 : Math.min(1, Math.max(0, m));
        // the fill's antialiased edge, for the center search (a binary edge votes weak arcs)
        if (besideFill[p])
            fillK[p] = Math.min(1, Math.max(0, 1 - m));
    }
}

// A fill inside a stroke too thin to show its own color: a pixel in the triangle background–fill–stroke is dot by
// the weight it takes off the background; the fill's own weight goes to fillW.
function readStroked(ground, { D, fillW }) {
    const { color, stroke, own, otherFill, near, bg, B, offAbs } = ground;
    const { data } = ground.image;
    const noDot = noDotAt(ground);
    for (let p = 0; p < D.length; p++) {
        if (noDot(p)) {
            D[p] = 0;
            continue;
        }
        // another family's fill is that dot, never this one's partial coverage
        if (!own[p] && otherFill[p]) {
            D[p] = NaN;
            continue;
        }
        const r = data[4 * p];
        const g = data[4 * p + 1];
        const b = data[4 * p + 2];
        const w = near[p] ? [bg[3 * p], bg[3 * p + 1], bg[3 * p + 2]] : B;
        const a1 = color.map((v, ch) => v - w[ch]);
        const a2 = stroke.map((v, ch) => v - w[ch]);
        const v = [r - w[0], g - w[1], b - w[2]];
        const m11 = a1[0] * a1[0] + a1[1] * a1[1] + a1[2] * a1[2];
        const m12 = a1[0] * a2[0] + a1[1] * a2[1] + a1[2] * a2[2];
        const m22 = a2[0] * a2[0] + a2[1] * a2[1] + a2[2] * a2[2];
        const det = m11 * m22 - m12 * m12;
        if (m11 < MIN_CONTRAST * MIN_CONTRAST || det <= 0) {
            D[p] = NaN;
            continue;
        }
        const r1 = v[0] * a1[0] + v[1] * a1[1] + v[2] * a1[2];
        const r2 = v[0] * a2[0] + v[1] * a2[1] + v[2] * a2[2];
        const wf = (m22 * r1 - m12 * r2) / det;
        const ws = (m11 * r2 - m12 * r1) / det;
        const off = Math.hypot(v[0] - wf * a1[0] - ws * a2[0], v[1] - wf * a1[1] - ws * a2[1], v[2] - wf * a1[2] - ws * a2[2]);
        const reach = Math.hypot(wf * a1[0] + ws * a2[0], wf * a1[1] + ws * a2[1], wf * a1[2] + ws * a2[2]);
        if (off > Math.max(offAbs, OFF_MIX_REL * reach) || wf < -STROKE_SLACK || ws < -STROKE_SLACK || wf + ws > MAX_MIX) {
            D[p] = NaN;
            continue;
        }
        D[p] = Math.min(1, Math.max(0, wf + ws));
        fillW[p] = Math.max(0, wf);
    }
}

// A fill inside an opaque ring of the outline's color: a pixel between the two colors is inside a dot (its place
// between them goes to fillK); any other pixel in the outline's reach is read as the outline mixed with the
// background (mixReader), and ink beyond the reach of every fill isn't this family's.
function readOutlined(ground, cover) {
    const { color, outline, ownReach, offAbs } = ground;
    const { D, fillK } = cover;
    const { data } = ground.image;
    const noDot = noDotAt(ground);
    const readMix = mixReader(ground, cover);
    const fo = color.map((v, ch) => v - outline[ch]);
    const foo = fo[0] * fo[0] + fo[1] * fo[1] + fo[2] * fo[2];
    for (let p = 0; p < D.length; p++) {
        if (noDot(p)) {
            D[p] = 0;
            continue;
        }
        if (!ownReach[p]) {
            D[p] = NaN;
            continue;
        }
        const e0 = data[4 * p] - outline[0];
        const e1 = data[4 * p + 1] - outline[1];
        const e2 = data[4 * p + 2] - outline[2];
        const k = (e0 * fo[0] + e1 * fo[1] + e2 * fo[2]) / foo;
        if (k >= -OUTLINE_SLACK && k <= 1 + OUTLINE_SLACK && Math.hypot(e0 - k * fo[0], e1 - k * fo[1], e2 - k * fo[2]) <= offAbs) {
            D[p] = 1;
            fillK[p] = Math.min(1, Math.max(0, k));
            continue;
        }
        readMix(p);
    }
}

// a plain fill: every pixel is the color mixed with its background
function readPlain(ground, cover) {
    const { D } = cover;
    const noDot = noDotAt(ground);
    const readMix = mixReader(ground, cover);
    for (let p = 0; p < D.length; p++) {
        if (noDot(p))
            D[p] = 0;
        else
            readMix(p);
    }
}

// The reading of one pixel as the family's edge color (the outline's, if there is one, else the fill's) mixed with
// its background: D is the share of the color, NaN off the line between them or where another family's color is
// nearer; where the background is other ink, edgeInk records which.
function mixReader(ground, { D, edgeInk }) {
    const { color, outline, others, near, bg, B, offAbs, pageColors } = ground;
    const { data } = ground.image;
    // the outer rim mixes the background with the outline, if there is one
    const edge = outline ?? color;
    return p => {
        const r = data[4 * p];
        const g = data[4 * p + 1];
        const b = data[4 * p + 2];
        // away from the family, another family's color is that family's pixel, not partial coverage of this one
        // (near it, pixels are read against their actual neighbor, which keeps the edges where dots touch)
        const dOwn = Math.min(Math.hypot(r - color[0], g - color[1], b - color[2]), outline ? Math.hypot(r - outline[0], g - outline[1], b - outline[2]) : Infinity);
        if (!near[p] && others.some(O => { const d = Math.hypot(r - O[0], g - O[1], b - O[2]); return d <= CLUSTER_TOL && d < dOwn; })) {
            D[p] = NaN;
            return;
        }
        // near the family, against the nearest other color; away from it, against the family's usual background
        // (faint antialiased rims just beyond the near zone were left unknown and broke the dots' edges)
        const b0 = near[p] ? bg[3 * p] : B[0];
        const b1 = near[p] ? bg[3 * p + 1] : B[1];
        const b2 = near[p] ? bg[3 * p + 2] : B[2];
        const w0 = edge[0] - b0;
        const w1 = edge[1] - b1;
        const w2 = edge[2] - b2;
        const ww = w0 * w0 + w1 * w1 + w2 * w2;
        if (ww < MIN_CONTRAST * MIN_CONTRAST) {
            D[p] = NaN;
            return;
        }
        const v0 = r - b0;
        const v1 = g - b1;
        const v2 = b - b2;
        const m = (v0 * w0 + v1 * w1 + v2 * w2) / ww;
        const off = Math.hypot(v0 - m * w0, v1 - m * w1, v2 - m * w2);
        const tol = Math.max(offAbs, OFF_MIX_REL * Math.abs(m) * Math.sqrt(ww));
        D[p] = off > tol || m > MAX_MIX ? NaN : Math.min(1, Math.max(0, m));
        if (near[p] && D[p] < 0.5 && !pageColors.some(P => Math.hypot(b0 - P[0], b1 - P[1], b2 - P[2]) <= INK_TOL)) {
            const k = others.findIndex(O => Math.hypot(b0 - O[0], b1 - O[1], b2 - O[2]) <= INK_TOL);
            edgeInk[p] = k >= 0 && k < EDGE_INK_UNLISTED - 1 ? k + 1 : EDGE_INK_UNLISTED;
        }
    };
}

/**
 * An opaque family's radius: the one given, else the isolated dots' median, else the fills' (outlined or stroked
 * dots), else the distance transform's. With `blurred`, blurred discs are isolated dots too (shape.js).
 *
 * @returns {{isolated: object[], byFill: object|null, fillCover: Float32Array|null, radius: number,
 *   fromDistance: boolean}}  radius in px
 */
function opaqueRadius(ground, cover, tol, given, blurred = false) {
    const { outline, stroke, own } = ground;
    const { D, fillW, fillK } = cover;
    const { width, height } = ground.image;
    // an isolated dot of this family has its fill at its middle: specks of another ink that D reads as covered
    // (hair2: the stroke's black, so the decimal points and a colon's dots, r 4.9, outvoted the one whole dot, r 30.3)
    // are not its dots
    const fill = stroke ? fillW : own;
    const isolated = isolatedDots(D, width, height, tol, blurred).filter(d => discShare(fill, width, height, d.cx, d.cy, ISOLATED_FILL_RADII * d.r) >= DOT_MIN_FILL);
    // outlined or stroked dots without an isolated whole one: their fills give the radius, and the center search runs
    // on the fills, whose edges are each dot's own wherever it's on top (D's edge is the union's, joined to lines of
    // the fill's color and to touching dots; cells1's magenta, chained to its neighbors by gray lines that read as
    // half the stroke, got 4.5 for 20 from the distance transform)
    const fillCover = outline ? fillK : stroke ? fillW : null;
    const byFill = !isolated.length && fillCover ? outlinedRadius(fillCover, own, D, width, height) : null;
    const radius = given ?? (isolated.length ? median(isolated.map(d => d.r)) : byFill?.radius ?? radiusFromDistance(D, width, height));
    return { isolated, byFill, fillCover, radius, fromDistance: !given && !isolated.length && !byFill };
}

/**
 * The centers that pass the dot test at the family's radius, and that radius, which a scan of larger radii may
 * raise when it came from the distance transform. O(pixels · radii tried) (houghCenters).
 *
 * @returns {{centers: {x: number, y: number, arc: number}[], radius: number,
 *            under: {x: number, y: number, arc: number}[]}}  px; under: the centers of dots partly under other ink,
 *   which aren't among `centers`
 */
function dotCenters(scan, ground, cover, shape, fillRadius) {
    const { outline, stroke, own, B } = ground;
    const { D, fillW } = cover;
    const { byFill, fillCover, fromDistance } = shape;
    const { width, height } = ground.image;
    const n = width * height;
    let { radius } = shape;
    const weight = new Uint8Array(n);
    for (let p = 0; p < n; p++)
        weight[p] = Number.isNaN(D[p]) ? 0 : 1;
    // a center counts if most of its edge is seen and its disc is filled with the family's own color (text:
    // round glyphs are hollow; fanc2: a rim gray between dot and overlap read black text as beyond its color,
    // so D was full there, and every glyph was a dot); a stroked family's inner disc must be its fill
    const rfOf = r => DOT_FILL_RADII * (byFill?.fillRadius ?? (stroke ? fillRadius ?? r * DOT_FILL_RADII : r));
    const inside = stroke ? fillW : own;
    // the pixels of the disc of radius rf at a center: all of them, the fill's, and those of other ink (unread)
    const disc = (c, rf) => {
        let count = 0;
        let on = 0;
        let other = 0;
        for (let y = Math.max(0, Math.floor(c.y - rf)); y < Math.min(height, Math.ceil(c.y + rf)); y++) {
            for (let x = Math.max(0, Math.floor(c.x - rf)); x < Math.min(width, Math.ceil(c.x + rf)); x++) {
                if ((x + 0.5 - c.x) ** 2 + (y + 0.5 - c.y) ** 2 > rf * rf)
                    continue;
                count++;
                if (inside[y * width + x] > 0.5)
                    on++;
                else if (Number.isNaN(D[y * width + x]))
                    other++;
            }
        }
        return { count, on, other };
    };
    const filled = (c, rf) => {
        const { count, on } = disc(c, rf);
        return count > 0 && on >= DOT_MIN_FILL * count;
    };
    // … and one partly under other ink, which the test above leaves out
    const partlyFilled = (c, rf) => {
        const { count, on, other } = disc(c, rf);
        return other > 0 && on < DOT_MIN_FILL * count && on >= DOT_MIN_FILL * (count - other) && on >= DOT_MIN_SEEN * count;
    };
    // (the gradient is the radius's alone, and by fill the centers are too)
    let grad = null;
    let fillCenters = null;
    const houghAt = r => {
        if (byFill) {
            fillCenters ??= houghCenters(fillCover, weight, width, height, byFill.fillRadius);
            return fillCenters;
        }
        grad ??= houghGradient(D, weight, width, height);
        return houghCenters(D, weight, width, height, r, grad);
    };
    // (the last radius's centers are kept for the dots partly under other ink at the radius chosen)
    let last = null;
    const centersAt = r => {
        last = { r, all: houghAt(r) };
        return last.all.filter(c => c.arc >= DOT_MIN_ARC && filled(c, rfOf(r)));
    };
    const strong = cs => cs.filter(c => c.arc >= DOT_STRONG_ARC).length;
    let centers = centersAt(radius);
    // The distance transform underestimates the radius where dots touch; packed past the dot test, larger radii
    // are tried by Hough. Its evidence leans small in a packed swarm (neighbors cut more of each arc as the radius
    // grows), so the largest radius keeping RADIUS_SCAN_KEEP of the most strong, filled centers is taken
    // (beeswarm-boxplot's gray: 6.16 from the transform, no center; 30 strong at 8–8.5 px, 23 at 9; the user's
    // truth has 9.0, and at 8.2 the fit put a second dot in nearly every dot).
    // Only for a fill on the page: one surrounded by another ink is an outlined or stroked dot, measured whole by
    // those readings (hair3, cells1: the scan passed their fills alone, 26.4 px for 28.3)
    const onBackground = scan.backgrounds.some(P => colorDistance(P, B) <= INK_TOL);
    if (fromDistance && !stroke && !outline && onBackground && strong(centers) < DOT_MIN_STRONG) {
        const tried = [];
        for (let r = radius + RADIUS_SCAN_STEP; r <= RADIUS_SCAN_MAX * radius; r += RADIUS_SCAN_STEP) {
            const cs = centersAt(r);
            tried.push({ r, cs, n: strong(cs) });
        }
        const most = Math.max(0, ...tried.map(t => t.n));
        // a line in the dots' color sets the transform's radius to its half width (kite1: cyan 3.5 px, the dots
        // 12.3): then the scan runs again around each of the pieces' radii (componentRadii), keeping the one with
        // the most strong centers (kite1: 4 at 11.5, 3 at 12, none below 11 or above 12.5)
        // (around each, the radii the first scan did not reach: a piece's scan was left out whenever it began
        // within the first one's, and whole dots at 10 px among the crescents of dots cut by a stroke, which set the
        // transform's radius to 4.97, were never tried: the first scan ended at 7.95, theirs began at 7.88)
        if (most < DOT_MIN_STRONG) {
            const reached = RADIUS_SCAN_MAX * radius;
            for (const r1 of componentRadii(D, width, height)) {
                const around = [];
                for (let r = Math.max((1 - COMPONENT_SCAN) * r1, reached + RADIUS_SCAN_STEP); r <= (1 + COMPONENT_SCAN) * r1; r += RADIUS_SCAN_STEP) {
                    const cs = centersAt(r);
                    around.push({ r, cs, n: strong(cs) });
                }
                if (Math.max(0, ...around.map(t => t.n)) > Math.max(0, ...tried.map(t => t.n)))
                    tried.splice(0, tried.length, ...around);
            }
        }
        const pick = tried.filter(t => t.n >= RADIUS_SCAN_KEEP * Math.max(0, ...tried.map(t => t.n))).pop();
        if (pick && pick.n > strong(centers)) {
            centers = pick.cs;
            radius = pick.r;
        }
    }
    const all = last.r === radius ? last.all : houghAt(radius);
    return { centers, radius, under: all.filter(c => c.arc >= DOT_MIN_ARC && partlyFilled(c, rfOf(radius))) };
}

// the share of a plane's pixels (> 0.5) that lie inside the discs of the given radius at the centers
function shareInDiscs(plane, centers, radius, width, height) {
    const inDisc = new Uint8Array(plane.length);
    for (const c of centers) {
        for (let y = Math.max(0, Math.floor(c.y - radius)); y < Math.min(height, Math.ceil(c.y + radius)); y++) {
            for (let x = Math.max(0, Math.floor(c.x - radius)); x < Math.min(width, Math.ceil(c.x + radius)); x++) {
                if ((x + 0.5 - c.x) ** 2 + (y + 0.5 - c.y) ** 2 <= radius * radius)
                    inDisc[y * width + x] = 1;
            }
        }
    }
    let fam = 0;
    let covered = 0;
    for (let p = 0; p < plane.length; p++) {
        if (plane[p] > 0.5) {
            fam++;
            covered += inDisc[p];
        }
    }
    return fam ? covered / fam : 0;
}

/**
 * Mark saturated overlaps covered (D = 1), in place: 8-connected regions of unknown pixels whose colors lie beyond
 * the family's color on its line from B (up to OVERLAP_MAX_MIX, and within OFF_MIX_ABS of it, not the family's
 * noise allowance: an overlap is one flat color, and on the noisy economist the allowance reached its pink and blue
 * dots) and aren't another family's, with at least OVERLAP_BORDER_SHARE of their border covered (D > 0.5) and a
 * (2·OVERLAP_MIN_DEPTH + 1)² square inside. O(pixels · OVERLAP_MIN_DEPTH²).
 *
 * @param {import('./image.js').RgbaImage} image
 * @param {Float32Array} D  coverage, NaN unknown
 * @param {number[]} B  0–255, the family's background
 * @param {number[]} color  0–255
 * @param {number[][]} others  0–255, the other families' colors
 */
function fillSaturatedOverlaps(image, D, B, color, others) {
    const { width, height, data } = image;
    const w = color.map((v, ch) => v - B[ch]);
    const ww = w[0] * w[0] + w[1] * w[1] + w[2] * w[2];
    if (ww < MIN_CONTRAST * MIN_CONTRAST)
        return;
    const isOther = nearAnyColor(data, others, CLUSTER_TOL);
    const beyond = p => {
        if (!Number.isNaN(D[p]))
            return false;
        if (isOther(p))
            return false;
        const v = [data[4 * p] - B[0], data[4 * p + 1] - B[1], data[4 * p + 2] - B[2]];
        const m = (v[0] * w[0] + v[1] * w[1] + v[2] * w[2]) / ww;
        return m > 1 && m <= OVERLAP_MAX_MIX && Math.hypot(v[0] - m * w[0], v[1] - m * w[1], v[2] - m * w[2]) <= Math.max(OFF_MIX_ABS, OFF_MIX_REL * m * Math.sqrt(ww));
    };
    const seen = new Uint8Array(width * height);
    for (let start = 0; start < seen.length; start++) {
        if (seen[start] || !beyond(start))
            continue;
        const comp = [start];
        seen[start] = 1;
        let border = 0;
        let covered = 0;
        for (let k = 0; k < comp.length; k++) {
            const p = comp[k];
            const x = p % width;
            const y = (p - x) / width;
            for (let dy = -1; dy <= 1; dy++) {
                for (let dx = -1; dx <= 1; dx++) {
                    if ((dx || dy) && x + dx >= 0 && x + dx < width && y + dy >= 0 && y + dy < height) {
                        const q = p + dy * width + dx;
                        if (seen[q])
                            continue;
                        if (beyond(q)) {
                            seen[q] = 1;
                            comp.push(q);
                        }
                        else {
                            border++;
                            if (D[q] > 0.5)
                                covered++;
                        }
                    }
                }
            }
        }
        if (border && covered >= OVERLAP_BORDER_SHARE * border && hasSquare(comp, width, height, OVERLAP_MIN_DEPTH)) {
            for (const p of comp)
                D[p] = 1;
        }
    }
}

/**
 * Radius of outlined dots without an isolated whole one: each dot's fill is bounded by its own outline, so the
 * topmost dots' fills are whole discs even where the dots touch others or a line of their own color drawn behind
 * them (phase1: most dots sit on their fitted curve, whose distance-transform ridge outvoted them, r 4–6 against
 * 18). The fill radius is the median of the round fill components; the outline's width comes from the covered
 * pixels in a ring just outside each, as the area they add to the disc. Null without a round fill. fillK is the fill's
 * antialiased coverage (0 at the outline's color, 1 at the fill's), or a stroked family's fill weight.
 *
 * @returns {{radius: number, fillRadius: number}|null}  px
 */
function outlinedRadius(fillK, own, D, width, height) {
    const discs = roundFills(fillK, width, height).filter(d => d.r >= DOT_MIN_RADIUS);
    if (!discs.length)
        return null;
    const rf = median(discs.map(d => d.r));
    const widths = discs.map(d => {
        let added = 0;
        const outer = d.r + OUTLINE_REACH;
        for (let y = Math.max(0, Math.floor(d.cy - outer)); y < Math.min(height, Math.ceil(d.cy + outer)); y++) {
            for (let x = Math.max(0, Math.floor(d.cx - outer)); x < Math.min(width, Math.ceil(d.cx + outer)); x++) {
                const q = Math.hypot(x + 0.5 - d.cx, y + 0.5 - d.cy);
                const p = y * width + x;
                if (q > d.r && q <= outer && !own[p] && D[p] > 0)
                    added += Math.min(1, D[p]);
            }
        }
        return Math.sqrt(d.r * d.r + added / Math.PI) - d.r;
    });
    return { radius: rf + median(widths), fillRadius: rf };
}

/**
 * Round fill components: those of fill > 0.5, away from the image border, round by their second moments over their
 * own pixels and their antialiased rim (fill 0–0.5 next to them), weighted by min(fill, 1). Unlike isolatedDots, a
 * window around the component isn't read: fills in a cluster sit 2 px apart, across their strokes (cells1's magenta:
 * every fill failed when its neighbors' fills fell in the window). O(pixels).
 *
 * @returns {{cx: number, cy: number, r: number}[]}  px
 */
function roundFills(fill, width, height) {
    const out = [];
    const taken = new Uint8Array(fill.length);
    for (const comp of components(fill, width, height)) {
        for (const p of comp)
            taken[p] = 1;
    }
    for (const comp of components(fill, width, height)) {
        const pixels = new Set(comp);
        for (const p of comp) {
            const x = p % width;
            for (const q of [p - 1, p + 1, p - width, p + width, p - width - 1, p - width + 1, p + width - 1, p + width + 1]) {
                const qx = q % width;
                if (q >= 0 && q < fill.length && Math.abs(qx - x) <= 1 && !taken[q] && fill[q] > 0)
                    pixels.add(q);
            }
        }
        let a = 0;
        let sx = 0;
        let sy = 0;
        let edge = false;
        for (const p of pixels) {
            const x = p % width;
            const y = (p - x) / width;
            edge ||= x === 0 || y === 0 || x === width - 1 || y === height - 1;
            const w = Math.min(1, fill[p]);
            a += w;
            sx += w * (x + 0.5);
            sy += w * (y + 0.5);
        }
        if (edge || a < ISOLATED_MIN_AREA)
            continue;
        const cx = sx / a;
        const cy = sy / a;
        let mxx = 0;
        let myy = 0;
        let mxy = 0;
        for (const p of pixels) {
            const x = p % width;
            const y = (p - x) / width;
            const w = Math.min(1, fill[p]);
            mxx += w * (x + 0.5 - cx) ** 2;
            myy += w * (y + 0.5 - cy) ** 2;
            mxy += w * (x + 0.5 - cx) * (y + 0.5 - cy);
        }
        mxx /= a;
        myy /= a;
        mxy /= a;
        const tr = mxx + myy;
        const disc = Math.sqrt(Math.max(0, tr * tr / 4 - (mxx * myy - mxy * mxy)));
        const r = Math.sqrt(a / Math.PI);
        if (Math.sqrt((tr / 2 + disc) / (tr / 2 - disc)) <= ISOLATED_MAX_ASPECT && Math.sqrt(2 * tr) / r <= ISOLATED_MAX_MOMENT_RATIO)
            out.push({ cx, cy, r });
    }
    return out;
}

/**
 * Fills in the page's own color (within INK_TOL: bench2 stroked-dots' translucent fills, about 230 on white, are a
 * color of their own): page-colored regions cut off from the rest of the page (not touching the image
 * border), which is what an outline does to a white dot on a white page. Round regions of one size (the largest
 * group within PAGE_FILL_SIZE_TOL; PAGE_FILL_MIN_WHOLE of them at least) are whole fills. An enclosed region up to
 * PAGE_FILL_AREA_SLACK × their area (a bar's inside is larger) is a fill, whole or partly hidden, when it lies in the
 * disc of a Hough center at the fill radius (PAGE_FILL_IN_DISC) and its edge is that circle or the outlines of fills
 * over it (PAGE_FILL_EXPLAINED): a hidden dot's crescent shows an arc of its own edge, which a letter's counter or a
 * gap between other dots (bounded by arcs curving the other way) doesn't. Null without DOT_MIN_STRONG fills.
 * O(pixels) plus O(centers · r²) and O(edge pixels · centers).
 *
 * @param {import('./image.js').RgbaImage} image
 * @param {number[]} page  0–255
 * @returns {{mask: Uint8Array, discs: {cx: number, cy: number, r: number}[], centers: {x: number, y: number}[]}|null}
 *   discs: the whole fills, px; centers: one per dot the fills show, whole or partly hidden, px
 */
function pageFills(image, page) {
    const { width, height, data } = image;
    const n = width * height;
    const member = new Float32Array(n);
    const onPage = nearColor(data, page, INK_TOL);
    for (let p = 0; p < n; p++)
        member[p] = onPage(p) ? 1 : 0;
    const round = roundFills(member, width, height).filter(d => d.r >= DOT_MIN_RADIUS);
    const within = (d, e) => Math.abs(e.r - d.r) <= PAGE_FILL_SIZE_TOL * d.r;
    let best = null;
    for (const d of round) {
        const group = round.filter(e => within(d, e));
        if (!best || group.length > best.length || (group.length === best.length && d.r > best.r))
            best = Object.assign(group, { r: d.r });
    }
    if (!best || best.length < PAGE_FILL_MIN_WHOLE)
        return null;
    const rf = median(best.map(d => d.r));
    const maxArea = PAGE_FILL_AREA_SLACK * Math.PI * Math.max(...best.map(d => d.r)) ** 2;
    const label = new Int32Array(n).fill(-1);
    const candidates = [];
    for (const comp of components(member, width, height)) {
        const border = comp.some(p => {
            const x = p % width;
            const y = (p - x) / width;
            return x === 0 || y === 0 || x === width - 1 || y === height - 1;
        });
        if (border || comp.length > maxArea)
            continue;
        for (const p of comp)
            label[p] = candidates.length;
        candidates.push(comp);
    }
    // the candidates' coverage with an antialiased edge (lightness of the pixels next to them), for the vote
    const pageL = luminance(page);
    const cover = new Float32Array(n);
    for (let p = 0; p < n; p++) {
        if (label[p] >= 0) {
            cover[p] = 1;
            continue;
        }
        const x = p % width;
        const y = (p - x) / width;
        if (x > 0 && x < width - 1 && y > 0 && y < height - 1 && [p - 1, p + 1, p - width, p + width].some(q => label[q] >= 0))
            cover[p] = Math.min(1, Math.max(0, luminance([data[4 * p], data[4 * p + 1], data[4 * p + 2]]) / pageL));
    }
    // the Hough centers whose disc holds a region
    const holding = candidates.map(() => []);
    const reach = rf + PAGE_FILL_DISC_SLACK;
    for (const c of houghCenters(cover, new Uint8Array(n).fill(1), width, height, rf).filter(c => c.arc >= PAGE_FILL_MIN_ARC)) {
        const seen = new Map();
        for (let y = Math.max(0, Math.floor(c.y - reach)); y < Math.min(height, Math.ceil(c.y + reach)); y++) {
            for (let x = Math.max(0, Math.floor(c.x - reach)); x < Math.min(width, Math.ceil(c.x + reach)); x++) {
                const k = label[y * width + x];
                if (k >= 0 && (x + 0.5 - c.x) ** 2 + (y + 0.5 - c.y) ** 2 <= reach * reach)
                    seen.set(k, (seen.get(k) ?? 0) + 1);
            }
        }
        for (const [k, count] of seen) {
            if (count >= PAGE_FILL_IN_DISC * candidates[k].length)
                holding[k].push(c);
        }
    }
    // each region's edge (pixels beside another), and its center: of those holding it, the one whose circle is most of
    // that edge (dots closer than their radius: a crescent lies in the disc of the dot beside it too)
    const edges = candidates.map((comp, k) => comp.filter(p => ![p - 1, p + 1, p - width, p + width].every(q => label[q] === k)));
    const onCircle = (p, c) => {
        const x = p % width;
        const y = (p - x) / width;
        return Math.abs(Math.hypot(x + 0.5 - c.x, y + 0.5 - c.y) - rf) <= PAGE_FILL_EDGE_TOL;
    };
    const own = (k, c) => edges[k].filter(p => onCircle(p, c)).length;
    const center = holding.map((cs, k) => cs.reduce((a, c) => (a && own(k, a) >= own(k, c) ? a : c), null));
    // dots that may lie over a region: the whole fills and the other regions' centers
    const dots = [...best.map(d => ({ x: d.cx, y: d.cy })), ...center.filter(c => c)];
    const outer = rf * (1 + PAGE_RING_MAX_WIDTH) + PAGE_FILL_EDGE_TOL;
    const mask = new Uint8Array(n);
    // dots, not regions: a line across a crescent cuts it in two, and specks of page lie between outlines
    const fills = new Set();
    candidates.forEach((comp, k) => {
        const c = center[k];
        if (!c)
            return;
        const over = dots.filter(o => Math.hypot(o.x - c.x, o.y - c.y) > PAGE_FILL_SAME_RADII * rf);
        const edge = edges[k];
        const explained = edge.filter(p => {
            const x = p % width;
            const y = (p - x) / width;
            return onCircle(p, c) || over.some(o => Math.hypot(x + 0.5 - o.x, y + 0.5 - o.y) <= outer);
        }).length;
        if (own(k, c) < PAGE_FILL_OWN_EDGE * edge.length || explained < PAGE_FILL_EXPLAINED * edge.length)
            return;
        fills.add(c);
        for (const p of comp)
            mask[p] = 1;
    });
    return fills.size >= DOT_MIN_STRONG ? { mask, discs: [...best], centers: [...fills].map(c => ({ x: c.x, y: c.y })) } : null;
}

/**
 * A family of dots filled with the page's color inside an outline (hair1, tumour1: white dots in black outlines),
 * which the color clusters can't show: the page's color is left out of them. The outline's color is the median ink
 * in a band just outside the whole fills, and it must be thin beside them (PAGE_RING_MAX_WIDTH: text's counters
 * aren't dots). Null when no such family shows.
 */
function pageFilledFamily(scan) {
    const { image, backgrounds } = scan;
    const { width, height, data } = image;
    const page = backgrounds[0];
    const fills = pageFills(image, page);
    if (!fills)
        return null;
    const ink = [[], [], []];
    const onPage = nearColor(data, page, INK_TOL);
    for (const d of fills.discs) {
        const outer = d.r + PAGE_RING_BAND[1];
        for (let y = Math.max(0, Math.floor(d.cy - outer)); y < Math.min(height, Math.ceil(d.cy + outer)); y++) {
            for (let x = Math.max(0, Math.floor(d.cx - outer)); x < Math.min(width, Math.ceil(d.cx + outer)); x++) {
                const q = Math.hypot(x + 0.5 - d.cx, y + 0.5 - d.cy) - d.r;
                const p = 4 * (y * width + x);
                if (q < PAGE_RING_BAND[0] || q > PAGE_RING_BAND[1] || onPage(y * width + x))
                    continue;
                for (let ch = 0; ch < 3; ch++)
                    ink[ch].push(data[p + ch]);
            }
        }
    }
    if (!ink[0].length)
        return null;
    const outline = ink.map(median);
    if (colorDistance(outline, page) <= CLUSTER_TOL)
        return null;
    const fam = measureOpaque(scan, page, [], outline);
    const rf = median(fills.discs.map(d => d.r));
    // the fills are the evidence where few dots show half their edge (hair2: two whole dots and seven crescents)
    const enough = isDotFamily(fam) || (radiusPlausible(fam.radius, fam.isolated.length) && fam.coverage >= DOT_MIN_COVERAGE);
    return enough && fam.radius - rf <= PAGE_RING_MAX_WIDTH * rf ? fam : null;
}

// whether a (2·depth + 1)² square of pixels fits inside a region (pixel indices)
function hasSquare(comp, width, height, depth) {
    const inside = new Set(comp);
    return comp.some(p => {
        const x = p % width;
        const y = (p - x) / width;
        if (x < depth || y < depth || x + depth >= width || y + depth >= height)
            return false;
        for (let dy = -depth; dy <= depth; dy++) {
            for (let dx = -depth; dx <= depth; dx++) {
                if (!inside.has(p + dy * width + dx))
                    return false;
            }
        }
        return true;
    });
}

// relative luminance proxy, 0–255 (Rec. 709 weights on the sRGB values): only compared, never displayed
function luminance(c) {
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
}

function radiusPlausible(radius, isolatedCount) {
    return radius >= DOT_MIN_RADIUS || (radius >= SMALL_DOT_MIN_RADIUS && isolatedCount >= SMALL_DOT_MIN_ISOLATED);
}

/**
 * Whether a measured color is a dot family: a plausible radius, enough strong centers, and its pixels mostly inside
 * the dots, or isolated dots or many strong centers to make up for it.
 *
 * @param {OpaqueMeasure} f
 * @returns {boolean}
 */
export function isDotFamily(f) {
    return radiusPlausible(f.radius, f.isolated.length) && (f.strongDots >= DOT_MIN_STRONG || f.isolated.length >= DOT_MANY_ISOLATED) && f.coverage >= DOT_MIN_COVERAGE_ANY
        && (f.coverage >= DOT_MIN_COVERAGE || f.isolated.length >= DOT_MIN_ISOLATED || f.strongDots >= DOT_MANY_STRONG);
}

/**
 * Opaque dot families: color clusters that aren't a background, aren't levels of a translucent family, and show
 * at least DOT_MIN_COUNT dot-shaped edges at a radius of at least DOT_MIN_RADIUS. `ink` is colors of other marks
 * (levels of rejected translucent candidates): other ink for every family but one of their own color. `onProgress`
 * gets the share of color clusters examined (0–1).
 *
 * The stages, in order: each color cluster is screened, measured and read as a plain, outlined or stroked fill
 * (tryCluster); the clusters that failed are tried again with what the families found since can tell about them
 * (fillsInFoundOutlines, solidsInOutlineColor, siblingFills, fewSiblings); a fill in the page's color is looked for; specks go;
 * and every family is measured again against the others for the fit (measuredForFit).
 *
 * @param {Scan} scan  its image, backgrounds (the page first, then regions under the dots) and tolerances
 * @param {import('./levels.js').LevelMeasure[]} translucent  the translucent families found
 * @param {number[][]} [ink]  0–255
 * @param {((fraction: number) => void)|null} [onProgress]
 * @param {number[]} [radii]  px, the radii of the translucent families' whole dots: a color in a few dots of that size
 *   is a family (fewSiblings)
 * @returns {OpaqueMeasure[]}  the families, each measured against the others, most dots first
 */
export function opaqueFamilies(scan, translucent, ink = [], onProgress = null, radii = []) {
    // The search's state, passed through its stages. `out` is the families found so far, which the screens of later
    // clusters ask about; `rejected` the clusters measured and found no family, for the stages after the loop.
    const search = {
        scan, translucent, ink, radii,
        // translucent families' level colors are theirs (see measureOpaque's `others`)
        levelColors: translucent.flatMap(f => f.levels.filter(L => !L.inferred).map(L => L.color)),
        // Only the dominant color is excluded up front: a dot color can cover 5% of an image (sparse #3: 5.9%) and
        // so count as a background color; the dot test tells it from a panel.
        clusters: colorClusters(scan.image, scan.backgrounds.slice(0, 1)),
        out: [],
        rejected: [],
        // ink in a blend of a family's color and the page (BLEND_INK_MIN_INTERIOR): other ink for the final measure
        blendInk: [],
    };
    for (const [i, cl] of search.clusters.entries()) {
        onProgress?.(i / search.clusters.length);
        tryCluster(search, cl.color);
    }
    // (after the loop: clusters are taken largest first, so the family that explains a cluster may come later)
    fillsInFoundOutlines(search);
    solidsInOutlineColor(search);
    siblingFills(search);
    fewSiblings(search);
    const paged = pageFilledFamily(scan);
    if (paged)
        search.out.push(paged);
    // Specks go before the others are measured again: a speck family's color can be a real family's stroke (hair3:
    // the black of the decimal points is the pink dots' outline, which excluded left their radius 18 for 28)
    const final = measuredForFit(search, withoutSpecks(search.out, translucent.length > 0));
    for (const f of final)
        f.under = final.filter(g => g !== f && drawnUnder(f, g, scan.image.width)).map(g => g.seed);
    return final.sort((a, b) => b.dots - a.dots);
}

function sameCluster(a, b) {
    return colorDistance(a, b) <= CLUSTER_TOL;
}

// the colors of other marks that aren't a seed's own
function otherInk(search, seed) {
    return search.ink.filter(c => colorDistance(c, seed) > CLUSTER_TOL);
}

// Whether a color lies on a translucent family's color line, no farther along it than the family's full color and
// LEVEL_LINE_BEYOND more: its levels approach that color and never pass it. mpo: black dots over gray boxes of 160
// and 127, which with the page fit the levels of a gray of 107 at α 0.64; black, 185 units beyond that gray on the
// same line, was taken for the grays' and never tried as a family.
function onLevelLine(translucent, c) {
    return translucent.some(f => {
        const { B, u } = f.line;
        const e = B.map((v, ch) => v - c[ch]);
        const t = e[0] * u[0] + e[1] * u[1] + e[2] * u[2];
        const full = colorDistance(B, f.fit.c);
        return Math.hypot(e[0] - t * u[0], e[1] - t * u[1], e[2] - t * u[2]) <= CLUSTER_ON_LINE_TOL
            && t <= full + Math.max(CLUSTER_TOL, LEVEL_LINE_BEYOND * full);
    });
}

// One color cluster: the screens, cheapest first, then the measure and the outline readings. A family goes to
// search.out; a cluster measured and found none to search.rejected.
function tryCluster(search, c) {
    const { scan, out } = search;
    if (onLevelLine(search.translucent, c) || out.some(f => f.outline && sameCluster(f.outline, c)))
        return;
    // a second peak of one family's color (two bins' peaks over one pure gray, 5 units apart: two families of the
    // same dots)
    if (out.some(f => colorDistance(f.seed, c) <= INK_TOL))
        return;
    // cheap test first: a dot family's pixels are mostly interior; text's are mostly edges
    const { count, share } = interiorShare(scan.image, c);
    if (isBlend(search, c, share)) {
        const own = out.some(f => sameCluster(c, f.seed) || (f.outline && sameCluster(c, f.outline)));
        const tint = !own && share >= BLEND_DOT_MIN_INTERIOR ? outlineReadings(search, c) : null;
        if (tint?.found) {
            out.push(Object.assign(tint.fam, { interior: share }));
            return;
        }
        // (not near the page or a family's color: those pixels are the page's, or the family's own shading)
        if (count && share >= BLEND_INK_MIN_INTERIOR && ![...scan.backgrounds, ...out.map(f => f.seed)].some(P => sameCluster(P, c)))
            search.blendInk.push(c);
        return;
    }
    if (throughLayer(search, c))
        return;
    if (!count || share < DOT_MIN_INTERIOR)
        return;
    const { fam, plain, ringed, found } = outlineReadings(search, c);
    fam.interior = share;
    if (found)
        out.push(fam);
    else
        search.rejected.push(Object.assign(plain, { interior: share, ringed }));
}

// A color's pixels (within CLUSTER_TOL, off the image's border) and the share of them whose eight neighbors are its
// color too (0 when it has none).
function interiorShare(image, color) {
    const { width, height, data } = image;
    const member = nearColor(data, color, CLUSTER_TOL);
    let count = 0;
    let interior = 0;
    for (let y = 1; y < height - 1; y++) {
        for (let x = 1; x < width - 1; x++) {
            const p = y * width + x;
            if (!member(p))
                continue;
            count++;
            let all = true;
            for (let dy = -1; dy <= 1 && all; dy++) {
                for (let dx = -1; dx <= 1; dx++) {
                    if (!member(p + dy * width + dx)) {
                        all = false;
                        break;
                    }
                }
            }
            if (all)
                interior++;
        }
    }
    return { count, share: count ? interior / count : 0 };
}

// A cluster between a family found already and that family's background, its outline or a page color is an
// antialiased rim, not a family (synthetic swarm: maroon's blend with white made a "family"). Between a fill and
// its outline, only if it is mostly edge, as a rim is: other fills of an outlined scale sit there too (phase1:
// grays 136 and 82 between gray 187 and the black outline, 62% interior; a rim is 1–2 px wide, unless the fill's
// own tolerance takes in fill pixels: a synthetic rim at 149, 6, 220 was 39 units from its fill)
function isBlend(search, c, interiorShare) {
    const rimOf = f => interiorShare < BLEND_OUTLINE_MAX_INTERIOR || sameCluster(c, f.seed) || sameCluster(c, f.outline);
    // (an outlined family's background is often its outline's color, all around its dots: as the outline, a rim only
    // when rimOf says; else every gray between a gray fill and its black outline was a blend: a synthetic copy of
    // phase1 without curves, its 82 gray dots beside 187s)
    const bases = f => {
        const rim = f.outline && rimOf(f);
        const ownOutline = f.outline && sameCluster(f.background, f.outline);
        return [...(ownOutline && !rim ? [] : [f.background]), ...(rim ? [f.outline] : []), ...search.scan.backgrounds];
    };
    return search.out.some(f => bases(f).some(P => isMixOf(c, P, f.seed)));
}

// A family's dots seen through a translucent layer of another color P (a band drawn over them) are no family of
// their own: c' − P = s·(c − page), with c a found family's color or a translucent family's level. Their dots
// share the family's radius by construction, and as a candidate they keep the band from being divided out
// (background.js). bench3 ANOVA: pink under a lavender band, 197, 106, 136; bench1 group-smoother-ci: the red
// levels under the CI band, 197, 106, 135, a family of 10 once the levels' pixels stopped counting as its own.
function throughLayer(search, c) {
    const page = search.scan.backgrounds[0];
    // (P is neither color, and off the family's line from the page: see LAYER_MIN_OFF)
    return [...search.levelColors, ...search.out.map(g => g.seed)].some(F => search.clusters.some(({ color: P }) => {
        if (sameCluster(P, F) || sameCluster(P, c))
            return false;
        const w = F.map((v, ch) => v - page[ch]);
        if (alongLine(P.map((x, ch) => x - page[ch]), w).off < LAYER_MIN_OFF)
            return false;
        const { m, off } = alongLine(c.map((x, ch) => x - P[ch]), w);
        return m > MIX_SHARE[0] && m < MIX_SHARE[1] && off <= CLUSTER_ON_LINE_TOL && solidRegion(search.scan, P, LAYER_MIN_HALF_WIDTH);
    }));
}

// A cluster's color measured as a plain fill, and when that is no dot family, as an outlined or stroked one.
// Returns the reading that stands (`fam`), the plain one, and whether it is a family.
//
// Outlined or stroked fills, in order: (1) a ring whose color is thin over the whole image is an outline that
// shows its own color (JMP's 1.7 px outlines); (2) else a fill inside a stroke too thin to show its color, off
// the line from the page to the fill (engagement1, cells1: it places colored fills better than the outline
// model, rmse 0.28 px against 0.95); (3) else an outline whose color is also other ink, counted only next to
// the fill (measureOpaque's OUTLINE_REACH), for a fill that is thick as an outline is not (phase1: black
// outlines around grays, which the stroke model can't read, white, gray and black being on one line, with
// text, axes and a black colorbar block in the outline's color; hair2's pink, one strong arc as stroked).
// Not around a translucent family's level (economist: a red–blue lens inside red dots made a phantom
// "outlined" family). In (3) the outline is darker than the fill, as ink shared with text is: a lighter ring is
// a band or region the dots lie under (bench3 ANOVA: pink under a pale blue band read as fills "outlined" in
// the band's color, a second family of 31). That guard and the thick fill stop economist's lens, so (3) may
// take a ring on a translucent family's line: gray outlines and a gray line can read as one (degradation1:
// outline 136 and line 166 as two levels of a "family" with no dots).
function outlineReadings(search, color) {
    const { scan } = search;
    const { image, backgrounds } = scan;
    const plain = measureOpaque(scan, color, [...search.levelColors, ...otherInk(search, color)]);
    let fam = plain;
    const ring = plain.background;
    const notPage = !backgrounds.some(P => colorDistance(P, ring) <= CLUSTER_TOL);
    const ringable = !onLevelLine(search.translucent, ring) && notPage;
    // (the reading in the ring's color, measured once and kept when it is no family: fewSiblings asks it again)
    let ringed = null;
    const outlinedFamily = () => {
        if (!ringed) {
            const outlined = measureOpaque(scan, color, [], ring);
            ringed = { ...outlined, fillRadius: outlined.fillRadius ?? plain.radius };
        }
        return isDotFamily(ringed) ? ringed : null;
    };
    if (!isDotFamily(fam) && ringable && isThin(scan, ring))
        fam = outlinedFamily() ?? fam;
    if (!isDotFamily(fam))
        fam = strokedFamily(scan, color) ?? fam;
    if (!isDotFamily(fam) && notPage && !isThin(scan, color) && luminance(ring) < luminance(color))
        fam = outlinedFamily() ?? fam;
    // (4) else a ring in a darker shade of the fill's own hue, when the fill's surroundings are the page: a line
    // in nearly the fill's color, joined to it, can border the page longer than the dots' rings do (kite2: a curve
    // at 211, 136, 88 beside fills at 209, 131, 82 in rings of 193, 88, 32, the ring on the line from the page
    // through the fill, 6 units off it; text and axes are far off an orange fill's line)
    // The fill is the commonest color just inside the rings, not the cluster's mean, which the line pulls off it
    // (kite2: 210, 134, 85; read against it the fills came out 0.94 inside and failed as round). The round fills
    // in rings of their darker shade are the evidence, so the coverage need only be DOT_MIN_COVERAGE_ANY: the line
    // is the fill's color too (kite2: 0.32, half its fill-colored pixels the curve's; 8 strong centers at 19.3 px)
    let byHueRing = false;
    if (!isDotFamily(fam) && !notPage) {
        for (const hue of hueRings(image, color, search.clusters, backgrounds[0])) {
            const fill = ringFill(image, color, hue);
            const outlined = fill && measureOpaque(scan, fill, [], hue);
            if (outlined && (isDotFamily(outlined) || (outlined.fillRadius && radiusPlausible(outlined.radius, outlined.isolated.length)
                && outlined.strongDots >= DOT_MIN_STRONG && outlined.coverage >= DOT_MIN_COVERAGE_ANY))) {
                fam = { ...outlined, fillRadius: outlined.fillRadius ?? plain.radius };
                byHueRing = true;
                break;
            }
        }
    }
    return { fam, plain, ringed, found: byHueRing || isDotFamily(fam) };
}

// Fills in a found family's outline: a plot draws every family's outline alike, so a fill whose surroundings read
// as the page, its line of its own color outweighing its rings, is tried in it (phase1: gray 228's 13 px curve made
// its ring white; in the 187s' black, 18 strong dots). Darker than the fill, as reading (3) asks.
function fillsInFoundOutlines(search) {
    const { scan, out, rejected } = search;
    const outlines = [];
    for (const f of out) {
        if (f.outline && !f.pageFill && !outlines.some(O => sameCluster(O, f.outline)))
            outlines.push(f.outline);
    }
    for (const [i, f] of [...rejected.entries()].reverse()) {
        for (const O of outlines) {
            if (sameCluster(O, f.seed) || luminance(O) >= luminance(f.seed) || isThin(scan, f.seed))
                continue;
            const outlined = measureOpaque(scan, f.seed, [], O);
            if (isDotFamily(outlined)) {
                out.push({ ...outlined, fillRadius: outlined.fillRadius ?? f.radius, interior: f.interior });
                rejected.splice(i, 1);
                break;
            }
        }
    }
}

// Dots filled with a found family's outline color (phase1's black: fill and outline one color, beside grays in
// black outlines). That color is also the other dots' rings and, usually, text and axes: its own components give
// text's radius (periods, 4.9) and its coverage is tiny (0.01). So it is measured at the outlined families' radius,
// their rings theirs (within their outline's width and RING_SLACK of their fills), and its evidence is centers
// filled with it at that radius; a glyph holds no filled disc that size.
function solidsInOutlineColor(search) {
    const { scan, out, rejected } = search;
    for (const [i, f] of [...rejected.entries()].reverse()) {
        const owners = out.filter(g => g.outline && !g.pageFill && g.fillRadius && sameCluster(g.outline, f.seed));
        if (!owners.length)
            continue;
        const rings = outlineRings(out);
        const radius = median(owners.map(g => g.radius));
        const solid = measureOpaque(scan, f.seed, out.map(g => g.seed), null, { radius, rings });
        if (solid.strongDots >= DOT_MIN_STRONG && solid.coverage >= DOT_MIN_COVERAGE_ANY) {
            out.push({ ...solid, interior: f.interior, solidRadius: radius, solidRings: rings });
            rejected.splice(i, 1);
        }
    }
}

// The outlined families' rings, for the coverage of a family filled with their outlines' color: each fill, and how
// far its outline reaches from it. Whole: every thin pixel of that color within reach of the fill is its outline, also
// one nearer this family's dots. Decided by which is nearer, as a stroke between two fills is, the outline of a gray
// dot running beside a black dot read as black dot, and a second black disc a pixel or two off explained it (phase1:
// 16 black dots fitted as 24, once the radius refit had brought the discs back to size).
function outlineRings(families) {
    return families.filter(g => g.outline && g.fillRadius).map(g => ({ fill: g.seed, reach: g.radius - g.fillRadius + RING_SLACK, whole: true }));
}

// Plain fills at a found family's radius (the sibling with strong arcs may come later in the clusters); each needs
// its own centers, filled and seen over half their edge. Here the dots partly under other ink count too: a family
// drawn first can have nearly every dot under another's (14 orange dots, 12 of them a third under blue ones: 2
// centers, coverage 0.21, and no family).
function siblingFills(search) {
    const { out } = search;
    const sibling = f => f.radius && out.some(g => !g.pageFill && Math.abs(f.radius - g.radius) <= SIBLING_RADIUS_TOL * g.radius) && !throughLayer(search, f.seed);
    for (const f of search.rejected) {
        if (sibling(f) && radiusPlausible(f.radius, f.isolated.length) && siblingEvidence(search, f))
            out.push(f);
    }
    siblingsAtFoundRadii(search);
}

// a color's measure at the first of the radii where it shows a sibling's evidence, without or with the found
// families' colors as other dots; null when at none
function atFoundRadius(search, seed, radii) {
    const ink = [...search.levelColors, ...otherInk(search, seed)];
    const found = search.out.map(g => g.seed);
    for (const radius of radii) {
        for (const others of [ink, [...ink, ...found]]) {
            const m = measureOpaque(search.scan, seed, others, null, { radius });
            if (siblingEvidence(search, m))
                return m;
        }
    }
    return null;
}

// A sibling's evidence: DOT_MIN_STRONG centers at the radius, those partly under other ink included, and
// DOT_MIN_COVERAGE of its pixels inside them. With SIBLING_MIN_ISOLATED whole dots of its own, that share is asked
// of its pixels in pieces as thick as a dot: the color of a figure's text and axes can be a family of dots too, and
// then most of its pixels are letters (reward: black dots at the teal ones' radius, 2 of them whole, 15 centers
// holding 0.11 of the black pixels and 0.47 of those where a disc of half the radius fits). Not without whole dots:
// three centers at a found radius are in black text too (fanc2: 0.05 of its pixels; bench2's ellipse: 4 centers of
// a second color with 0.12 of its pixels). Nor for a color past a found family's on its line from the page (tsne:
// gray dots of 154 and, where two lie on each other, of 127, 5 of those whole; as a family of its own the 127 got
// 3 dots and cost the gray 7).
function siblingEvidence(search, m) {
    if (m.dots + m.dotsUnder < DOT_MIN_STRONG)
        return false;
    return m.coverageUnder >= DOT_MIN_COVERAGE
        || (m.isolated.length >= SIBLING_MIN_ISOLATED && !beyondFound(search, m.seed) && thickCoverage(search.scan, m) >= DOT_MIN_COVERAGE);
}

// whether a color lies past a found family's on its line from the page: where that family's dots lie over each other
// it is read as theirs (fillSaturatedOverlaps)
function beyondFound(search, c) {
    const page = search.scan.backgrounds[0];
    return search.out.some(g => {
        const { m, off } = alongLine(c.map((v, ch) => v - page[ch]), g.seed.map((v, ch) => v - page[ch]));
        return m > 1 && off <= CLUSTER_ON_LINE_TOL;
    });
}

// the share of a measured color's thick pixels (a disc of THICK_FILL_RADII × the radius fits) inside its dots, 0–1
function thickCoverage(scan, m) {
    const { width, height } = scan.image;
    const own = Uint8Array.from(m.D, v => v > 0.5 ? 1 : 0);
    return shareInDiscs(opening(own, width, height, THICK_FILL_RADII * m.radius), [...m.centers, ...m.centersUnder], m.radius, width, height);
}

// A rejected color measured again at the radius of the families found: a color whose own shapes gave another radius
// can still be dots of the size the image's other dots have (aerobic: seven colors of dots at 8.6 px; four had no
// whole dot alone, and their thickest piece was the legend's square, 9.6–11.1 px; at 8.6 px blue shows 12 centers).
// The same evidence is asked as of a sibling. Each radius is tried without and with the found families' colors as
// other dots: known, they free the color's own edge (aerobic's yellow: 0.30 of it in its discs alone, 0.34 with
// three families known, 0.37 with six), but a dot under one of them is seen as under other ink only while that ink
// is unknown (every green dot a third under a blue one: 8 dots without, none with). Again while a family is added.
// O(rejected colors · radii · passes) measures, twice each.
function siblingsAtFoundRadii(search) {
    const { out, rejected } = search;
    for (let added = true; added;) {
        added = false;
        const radii = [];
        for (const g of out) {
            if (!g.pageFill && g.radius && !radii.some(r => Math.abs(r - g.radius) <= SIBLING_RADIUS_TOL * r))
                radii.push(g.radius);
        }
        for (const f of rejected) {
            if (out.some(g => sameCluster(g.seed, f.seed) || (g.outline && sameCluster(g.outline, f.seed))) || throughLayer(search, f.seed))
                continue;
            const m = atFoundRadius(search, f.seed, radii);
            if (m) {
                // (kept for the fit's measure: measured again without it, the radius is the shapes' again)
                out.push({ ...m, interior: f.interior, siblingRadius: m.radius });
                added = true;
            }
        }
    }
}

// A color that is all dots of a found family's size, too few for the dot test (FEW_MIN_DOTS): read plain, or in the
// ring around its fills. The size is a family's found here or a translucent family's (`radii`).
// Not a color within a found family's cluster: read in the ring, its thick fills are that family's dots (engagement1:
// the gray of text's edges, 209, is 33 units from the lavender fills, whose 10 dots in their dark stroke it "had").
function fewSiblings(search) {
    const { out, rejected } = search;
    const radii = [...search.radii, ...out.filter(g => !g.pageFill).map(g => g.radius)];
    const few = m => m && m.radius && radii.some(r => Math.abs(m.radius - r) <= SIBLING_RADIUS_TOL * r)
        && m.dots >= FEW_MIN_DOTS && m.coverage >= FEW_MIN_COVERAGE && (m.strongDots > 0 || m.isolated.length > 0);
    for (const f of rejected) {
        const m = [f, f.ringed].find(few);
        if (m && !out.some(g => g === f || sameCluster(g.seed, f.seed)) && !throughLayer(search, f.seed))
            out.push(m === f ? f : { ...m, interior: f.interior });
    }
}

// Each family measured again with the other families' colors excluded, now that they are known, for the fit. The
// first measurement decides what is a family: a small one surrounded by others (nicest language's red and dark
// blue, about 10 dots each) can fail the test once its neighbors' colors are excluded.
function measuredForFit(search, families) {
    const { scan } = search;
    const colors = families.map(f => f.seed);
    // with stroked families, every family's coverage gives the other fills' rings to their dots (strokes are usually
    // drawn on every dot, also those whose fill passed as plain: engagement1's navy took the others' rings)
    // (the reach is from the fill's color mask, as strokeColor reads the ring)
    const stroked = families.some(f => f.stroke);
    return families.map(f => {
        // a family found by its own dots, in the color of other families' outlines, as one found by solidsInOutlineColor
        const inOutlineColor = !f.outline && families.some(g => g.outline && g.fillRadius && sameCluster(g.outline, f.seed));
        const rings = f.solidRings ?? (inOutlineColor ? outlineRings(families)
            : stroked ? families.filter(g => g !== f).map(g => ({ fill: g.seed, reach: STROKE_RING })) : null);
        const others = [...colors.filter(c => c !== f.seed), ...search.levelColors, ...otherInk(search, f.seed), ...search.blendInk];
        const measure = radius => measureOpaque(scan, f.seed, others, f.outline, { stroke: f.stroke, fillRadius: f.fillRadius, rings, radius });
        const given = f.solidRadius ?? f.siblingRadius;
        let g = measure(given);
        let { siblingRadius } = f;
        let foundRadius;
        // With no whole dot the radius comes from the shapes, and among the other colors the shapes can be other
        // ones. When it leaves the radius the family was found at by more than SIBLING_RADIUS_TOL, the family takes
        // the radius of one with whole dots that was within that of its own (figures draw every group with one
        // marker size), else keeps the one it was found at. kite1, cyan dots on a line of their color: found at
        // 11.4 px, where 10 centers show, and measured again at 4.5, the line's (194 dots fitted for 15); the red
        // dots' 12.3 px is the cyan ones' too. A family with every dot a third under another's: 5.9 px for 10 from
        // what shows between, and 60 dots fitted for 12. Not otherwise: a family's own shapes place its dots best
        // (rvat: blue at 6.72 px beside orange at 6.93; started at the orange's, rmse 0.004 → 0.058 px).
        if (given === undefined && !g.isolated.length && !f.outline && !f.stroke && !f.pageFill && f.radius && Math.abs(g.radius - f.radius) > SIBLING_RADIUS_TOL * f.radius) {
            const like = f.isolated.length ? null : families.find(h => h !== f && !h.pageFill && h.isolated.length && Math.abs(f.radius - h.radius) <= SIBLING_RADIUS_TOL * h.radius);
            siblingRadius = like?.radius;
            // (for the note: the counts are those it was found with, at its own radius)
            foundRadius = like ? f.radius : undefined;
            g = measure(siblingRadius ?? f.radius);
        }
        return { ...g, radius: g.radius ?? f.radius, fillRadius: f.fillRadius, dots: f.dots, strongDots: f.strongDots, coverage: f.coverage, dotsUnder: f.dotsUnder, coverageUnder: f.coverageUnder, siblingRadius, foundRadius };
    });
}

// Whether opaque family f is drawn under g, from their seam: the pixels of g's ink next to f's dots (f's edgeInk).
// The seam is the edge of the dot on top, so it lies on that family's circles and inside the other's discs; the dots
// on top are whole and their centers found, while those underneath are partly hidden and often not. Seam pixels
// within ORDER_RING of a found circle of each family are counted, and f is under g when g's circles carry
// ORDER_RATIO times as many as f's (delta1: purple under black).
function drawnUnder(f, g, width) {
    if (!f.radius || !g.radius || !f.edgeInk)
        return false;
    const k = f.inkColors.findIndex(c => colorDistance(c, g.seed) <= INK_TOL);
    if (k < 0)
        return false;
    const onCircle = (h, x, y) => h.centers.some(c => Math.abs(Math.hypot(x - c.x, y - c.y) - h.radius) <= ORDER_RING);
    let onF = 0;
    let onG = 0;
    for (let p = 0; p < f.edgeInk.length; p++) {
        if (f.edgeInk[p] !== k + 1)
            continue;
        const x = p % width + 0.5;
        const y = Math.floor(p / width) + 0.5;
        onF += onCircle(f, x, y);
        onG += onCircle(g, x, y);
    }
    return onG >= ORDER_MIN_PIXELS && onG >= ORDER_RATIO * onF;
}

// Color clusters in a darker shade of a fill's hue: on the line from the page through the fill, within
// CLUSTER_ON_LINE_TOL, and beyond the fill (opaqueFamilies' outline reading (4)). Not only thin ones: kite2's rings
// are 3 px wide, and its overlaps' lenses, 15 units from the ring color, join them; the outlined measure decides.
// A ring lies at its dots' rims, next to the page wherever they border it; an overlap of translucent dots, the same
// darker shade, lies inside them: RING_PAGE_MIN_SHARE of the cluster within RING_PAGE_REACH of the page (kite2:
// 0.50; bench1 medium-dots.jpg read as exact, its α 0.5 family unseen through JPEG noise and its overlaps taken
// for rings: 0.11).
function hueRings(image, fill, clusters, page) {
    // a gray's darker shade is black, which is also text and axes (primates2: gray antialiasing "outlined" in black,
    // 136 dots for 15; fanc2, phase1, economist likewise): grays in black outlines are reading (3)'s
    if (Math.max(...fill) - Math.min(...fill) < HUE_RING_MIN_CHROMA)
        return [];
    const u = fill.map((v, ch) => v - page[ch]);
    const shades = clusters.map(c => c.color).filter(c => {
        const { m, off } = alongLine(c.map((x, ch) => x - page[ch]), u);
        return m > 1 && colorDistance(c, fill) > CLUSTER_TOL && off <= CLUSTER_ON_LINE_TOL;
    });
    if (!shades.length)
        return [];
    const { width, height, data } = image;
    const n = width * height;
    const toPage = distanceTransform(colorMask(image, page, INK_TOL), width, height);
    return shades.filter(c => {
        const isShade = nearColor(data, c, CLUSTER_TOL);
        let total = 0;
        let near = 0;
        for (let p = 0; p < n; p++) {
            if (!isShade(p))
                continue;
            total++;
            near += toPage[p] <= RING_PAGE_REACH * RING_PAGE_REACH;
        }
        return total > 0 && near >= RING_PAGE_MIN_SHARE * total;
    });
}

// The commonest exact color of a cluster within RING_FILL_REACH of its ring color's pixels (those within
// CLUSTER_ON_LINE_TOL of `ring`), or null when none is
function ringFill(image, color, ring) {
    const { width, height, data } = image;
    const n = width * height;
    const ringPx = colorMask(image, ring, CLUSTER_ON_LINE_TOL);
    const isColor = nearColor(data, color, CLUSTER_TOL);
    const reach = dilate(ringPx, width, height, RING_FILL_REACH);
    const counts = new Map();
    for (let p = 0; p < n; p++) {
        if (!reach[p] || ringPx[p] || !isColor(p))
            continue;
        const key = (data[4 * p] << 16) | (data[4 * p + 1] << 8) | data[4 * p + 2];
        counts.set(key, (counts.get(key) ?? 0) + 1);
    }
    let best = null;
    for (const [key, c] of counts) {
        if (!best || c > best.c)
            best = { key, c };
    }
    return best ? [best.key >> 16, (best.key >> 8) & 255, best.key & 255] : null;
}

/**
 * The family of a fill color inside a stroke (strokeColor), or null when no stroke shows or it isn't a dot family.
 *
 * @param {Scan} scan
 * @param {number[]} color  the fill, 0–255
 * @returns {OpaqueMeasure|null}  with fillRadius, and strokeOff: how far the stroke's color lies off the line from
 *   the page to the fill, 0–255 units
 */
export function strokedFamily(scan, color) {
    const stroke = strokeColor(scan.image, color, scan.backgrounds[0]);
    if (!stroke)
        return null;
    // The fill's radius is its isolated fills', not the plain reading's: that one is no family here, and its radius
    // can pass the whole dot's (funnel: 6.33 px for fills of 3.7 in dots of 5.5, so no center's inner disc was fill,
    // and no family)
    const fillRadius = stroke.fillRadius;
    const fam = measureOpaque(scan, color, [], null, { stroke: stroke.color, fillRadius });
    return isDotFamily(fam) ? { ...fam, fillRadius, strokeOff: stroke.off } : null;
}

// Small dots count only when nothing else does: beside regular dots, a few round specks are a dotted gridline, the
// dots of i's or decimal points (comment1: 9 black specks at r 2.5 beside orange dots at r 9.2; wave1: 27 decimal
// points at r 3.23, coverage 0.19, beside dots at r 14.9). Small is under DOT_MIN_RADIUS, or under SPECK_RADIUS_RATIO
// of the largest family's radius with most of the color not dot-shaped (text). A translucent family's radius isn't
// measured yet, but its levels already show real dots.
function withoutSpecks(opaque, anyTranslucent) {
    const largest = Math.max(...opaque.map(c => c.radius ?? 0), 0);
    const small = c => c.radius < DOT_MIN_RADIUS || (c.radius < SPECK_RADIUS_RATIO * largest && c.coverage < DOT_MIN_COVERAGE);
    const kept = opaque.filter(c => !small(c));
    return anyTranslucent || kept.length ? kept : opaque;
}

// From an image to its dots. The app takes the steps one at a time, with the user's changes between them: the scan
// in a worker (scanFamilies), each family's mask on the page (excludedBoxes, familyClips, maskOptions), the fit in
// another worker (fitInput), and the dots' fill colors after it (dotFills). The bench and the tests take them all
// at once (extractAll). Both take them from here, so that the bench measures what the app does.

import { estimateFamilies } from './families.js';
import { restoreChroma, jpegNoise } from './jpeg.js';
import { detectAxes, plotBounds, clipEdges, backgroundColors } from './axes.js';
import { annotationBoxes } from './annotations.js';
import { buildMask, plotClips } from './layers.js';
import { fitDots } from './fit.js';
import { dotFamilies } from './outline.js';
import { paletteColors, paletteFamilies } from './colormap.js';
import { pageStroke } from './stroke.js';

/** @typedef {import('./image.js').RgbaImage} RgbaImage */
/** @typedef {import('./style.js').Estimate} Estimate */
/** @typedef {import('./axes.js').Axes} Axes */
/** @typedef {import('./axes.js').ClipEdge} ClipEdge */

// A family has ink inside the plot. One with next to none there is the axes' own ink: on ggplot2's gray theme at 300
// dpi the tick labels' gray and the tick marks', darker, are the two levels of a translucent "family" beside the
// dots', with no whole dot and so no radius, which stood in the app's list and stopped extractAll. Its ink is all
// beyond the axes.
const PLOT_MIN_INK = 0.01; // share of the family's ink that lies within the plot's bounds
// … its ink being the pixels its count image gives half a dot or more: the same image gave 0.1 or less to 8,000
// pixels of the gridlines' rims inside the plot, grays between the panel's and the page's
const INK_MIN_COUNT = 0.5;

// families with a radius: what can be fitted
function fittable(prepared) {
    return prepared.estimates.filter(e => e.style?.radius).length;
}

// The scan's result without the families that have no ink inside the plot (PLOT_MIN_INK). As it is when the axes
// bound nothing, or when no family would be left: then the axes are in doubt, not the families.
function insidePlot(prepared, bounds) {
    if (!bounds)
        return prepared;
    const { left, right, top, bottom } = bounds;
    const kept = prepared.estimates.filter(e => {
        if (!e.D)
            return true;
        const { width } = e.image;
        let ink = 0;
        let inside = 0;
        for (let p = 0; p < e.D.length; p++) {
            if (!(e.D[p] >= INK_MIN_COUNT))
                continue;
            ink++;
            const x = p % width + 0.5;
            const y = Math.floor(p / width) + 0.5;
            if ((left === null || x > left) && (right === null || x < right) && (top === null || y > top) && (bottom === null || y < bottom))
                inside++;
        }
        return inside >= PLOT_MIN_INK * ink;
    });
    return kept.length ? { ...prepared, estimates: kept } : prepared;
}

/**
 * The scan for an image's dot families (families.js estimateFamilies), with the choice of color matching.
 *
 * A PNG made from a JPEG (a screenshot of a web figure) fails the exact-color tests. With `tryNoisy`, which the
 * caller sets for an image that has JPEG noise (jpeg.js jpegNoise) and isn't a JPEG file, the image is scanned a
 * second time with its chroma restored and noisy color matching, and that scan wins when it finds more families
 * with a radius (all of them, when none shows with exact colors). Not whenever the noise shows: restoring chroma
 * cost pathogen precision (0.972 → 0.637), which worked as it was. A family counts once it has a radius: without
 * one it can't be fitted (step1: the blue dots, radius unknown with exact colors, 5.4 px with noisy matching).
 *
 * With the plot's `bounds`, a family whose ink lies beyond them is left out (insidePlot).
 *
 * @param {RgbaImage} source  the image as decoded
 * @param {object} [opts]
 * @param {boolean} [opts.jpeg]  noisy color matching from the start (a JPEG file, or the user's choice)
 * @param {boolean} [opts.tryNoisy]
 * @param {{left: number|null, right: number|null, top: number|null, bottom: number|null}} [opts.bounds]  px, the
 *   plot's sides that the axes show (axes.js plotBounds)
 * @param {(label: string, fraction: number) => void} [opts.onProgress]  what is being done and the share of the
 *   whole done (0–1)
 * @returns {{work: RgbaImage, prepared: {image: RgbaImage, layers: object|null, estimates: Estimate[]},
 *            switched: boolean, exact?: number, found?: number}}
 *   work: the image as analyzed (the source, or its chroma restored); prepared: estimateFamilies' result on it;
 *   switched: the noisy scan won, with `found` fittable families against `exact`
 */
export function scanFamilies(source, opts = {}) {
    const { jpeg = false, tryNoisy = false } = opts;
    const report = opts.onProgress ?? (() => {});
    // the noisy pass takes about as long as the first (beeswarm-boxplot: 19 s and 33 s), so each gets half the bar
    const share = tryNoisy ? 0.5 : 1;
    const work = jpeg ? restoreChroma(source) : source;
    const prepared = insidePlot(estimateFamilies(work, { jpeg, bounds: opts.bounds, onProgress: (label, f) => report(label, share * f) }), opts.bounds);
    if (!tryNoisy)
        return { work, prepared, switched: false };
    const restored = restoreChroma(source);
    const noisy = insidePlot(estimateFamilies(restored, {
        jpeg: true,
        bounds: opts.bounds,
        onProgress: (label, f) => report(`Trying noisy color matching: ${label.charAt(0).toLowerCase()}${label.slice(1)}`, 0.5 + 0.5 * f),
    }), opts.bounds);
    const exact = fittable(prepared);
    const found = fittable(noisy);
    return found > exact ? { work: restored, prepared: noisy, switched: true, exact, found } : { work, prepared, switched: false };
}

/**
 * Text lines and legend markers, which no dot of a family lies in (annotations.js). They are read on the image as
 * analyzed, not a family's virtual one, against the family's radius.
 *
 * @param {RgbaImage} work  the image as analyzed
 * @param {number} radius  px
 * @returns {number[][]}  boxes [x0, y0, x1, y1) px
 */
export function excludedBoxes(work, radius) {
    const { text, legend } = annotationBoxes(work, backgroundColors(work)[0], radius);
    return [...text, ...legend];
}

/**
 * The edges at which the plot clips each family's dots: the axis lines and their far ends (axes.js clipEdges), where
 * a family's coverage, or all the families' together, is cut straight (layers.js plotClips).
 *
 * @param {Axes} axes  axes.js detectAxes' result
 * @param {{D: ArrayLike<number>|undefined, radius: number|undefined}[]} families  every family of the image: its
 *   coverage or count image and the radius in effect, px
 * @param {number} width  px, the image's
 * @param {number} height
 * @returns {ClipEdge[][]}  per family, its clip edges, each with `cut`
 */
export function familyClips(axes, families, width, height) {
    return plotClips(families, width, height, clipEdges(axes));
}

/**
 * What layers.js buildMask is told to mask around a family, besides other ink: what lies beyond the axes, the edges
 * the plot clips its dots at, and the boxes to leave out.
 *
 * @param {Axes} axes  axes.js detectAxes' result
 * @param {number[][]} exclude  boxes [x0, y0, x1, y1) px (excludedBoxes)
 * @param {ClipEdge[]} [clips]  the family's clip edges (familyClips); without them the mask reads the cuts from the
 *   family's own coverage alone
 * @param {Uint8Array|null} [strokes]  the pixels of a stroke in the page's color around the dots (stroke.js
 *   pageStroke's mask)
 * @returns {{bounds: {left: number|null, right: number|null, top: number|null, bottom: number|null},
 *            clips: ClipEdge[], exclude: number[][], strokes: Uint8Array|null}}  bounds: px, the plot area's sides
 *   (axes.js plotBounds)
 */
export function maskOptions(axes, exclude, clips = clipEdges(axes), strokes = null) {
    return { bounds: plotBounds(axes), clips, exclude, strokes };
}

/**
 * What fit.js fitDots needs for one family.
 *
 * @param {RgbaImage} image  the image the family is fitted on (its estimate's own, when it has one)
 * @param {Estimate} est  the family's estimate
 * @param {Uint8Array} mask  buildMask's
 * @param {import('./render.js').DotStyle} [style]  the style to fit with, when it isn't the estimate's (the user's
 *   radius, α or blur)
 * @returns {import('./fit.js').FitInput}
 */
export function fitInput(image, est, mask, style = est.style) {
    return { image, style, line: est.familyMeasure.line, mask, D: est.D, imgT: est.fitT, base: est.base, seeds: est.seeds };
}

/**
 * Each fitted dot's fill color, for the families whose dots are fitted together and told apart afterwards: outlined
 * dots of a few fill colors, fitted as one family of rings (outline.js dotFamilies), and a palette's dots, whose
 * colors are read from the fitted dots themselves (colormap.js paletteColors, paletteFamilies).
 *
 * @param {RgbaImage} work  the image as analyzed
 * @param {Estimate} est  the family's estimate
 * @param {{x: number, y: number}[]} dots  px
 * @param {number} radius  px, the fit's
 * @returns {{colors: number[][], which: {family: number, sure: boolean}[]}|null}  the fill colors (0–255) and, per
 *   dot, its color's index and whether enough of its fill shows to be sure; null for a family of one color, or
 *   when no color could be read
 */
export function dotFills(work, est, dots, radius) {
    const colors = est.palette ? paletteColors(work, dots, radius, est.background.color) : est.fillFamilies;
    if (!colors?.length)
        return null;
    const which = est.palette ? paletteFamilies(work, dots, radius, colors)
        : dotFamilies(work, dots, colors, radius, radius * (1 + est.style.ring.inner) / 2);
    return { colors, which };
}

/**
 * Every dot family of an image, scanned, masked and fitted as the app does it with nothing changed by hand:
 * overlays divided out, translucent families on their own images, what lies beyond the axes and in text and
 * legends masked. A PNG with JPEG noise is tried with noisy color matching (scanFamilies).
 *
 * A family told apart by fill color after the fit (dotFills) gives one result per color, so that scoring by group
 * checks the colors: each has the dots of that color (`familySure`: enough of the fill showed), the color as its
 * style's, and `fill`, the color's index.
 *
 * @param {RgbaImage} source  the image as decoded
 * @param {boolean} [jpeg]  a JPEG file
 * @returns {{est: Estimate, layers: object, fit: import('./fit.js').FitResult, jpegAuto: boolean, fill?: number}[]}
 *   per family: its estimate, layers.js buildMask's result (classes, thin, mask, stats) and fit.js fitDots';
 *   jpegAuto: noisy color matching was turned on for the JPEG noise; fill: its color's index, for a family told
 *   apart by fill color
 * @throws {Error} when a family has no style to fit with (its notes say why)
 */
export function extractAll(source, jpeg = false) {
    // The axes are the image's as decoded, as in the app, where they are found before the scan and carry the tick
    // values the user types. Read on the image as analyzed instead, the plot's sides moved by up to 0.08 px on six
    // PNGs with JPEG noise, and no dot of any test image changed.
    const axes = detectAxes(source);
    const { work, prepared, switched } = scanFamilies(source, { jpeg, tryNoisy: !jpeg && jpegNoise(source).likely, bounds: plotBounds(axes) });
    const missing = prepared.estimates.find(e => !e.style);
    if (missing)
        throw new Error(missing.notes.join(' '));
    const clips = familyClips(axes, prepared.estimates.map(e => ({ D: e.D, radius: e.style.radius })), work.width, work.height);
    const stroke = pageStroke(work, prepared.estimates);
    return prepared.estimates.flatMap((est, k) => {
        const layers = buildMask(est.image, est, maskOptions(axes, excludedBoxes(work, est.style.radius), clips[k], stroke.mask));
        const fit = fitDots(fitInput(est.image, est, layers.mask));
        return byFill(work, { est, layers, fit });
    }).map(r => ({ ...r, jpegAuto: switched }));
}

// a family's result as one per fill color, when its dots are told apart by color (extractAll)
function byFill(work, res) {
    const fills = dotFills(work, res.est, res.fit.dots, res.fit.radius);
    if (!fills)
        return [res];
    const { colors, which } = fills;
    return colors.map((color, k) => ({
        ...res,
        est: { ...res.est, fillFamilies: colors, style: { ...res.est.style, color } },
        fit: { ...res.fit, dots: res.fit.dots.filter((_, i) => which[i].family === k).map(d => ({ ...d, familySure: which[res.fit.dots.indexOf(d)].sure })) },
        fill: k,
    }));
}

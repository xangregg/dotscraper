// App entry: image source controls, the view, and the details panel. The current image name is kept in the URL
// hash (#image=<test image file>) so a reload or a shared local link reopens it.
//
// On each new image the app estimates the dot style (src/core/style.js) and builds the mask (src/core/layers.js);
// the panel shows both and lets the user pick another family, seed one by clicking a dot, or override the
// radius, alpha and blur.

import { pixelAt, pixelHash } from '../core/image.js';
import { restoreChroma } from '../core/jpeg.js';
import { estimateStyle } from '../core/style.js';
import { estimateFamilies } from '../core/families.js';
import { buildMask, CLASS_BACKGROUND, CLASS_FAMILY } from '../core/layers.js';
import { renderDots, solidImage } from '../core/render.js';
import { refitNear } from '../core/fit.js';
import { detectAxes, plotBounds } from '../core/axes.js';
import { axisScale, toAxisData, axisResolution, formatData, loadEntries, saveEntries, renderAxis } from './axes-panel.js';
import { decodeImage, listTestImages, loadTestImage, wireImageInput } from './input.js';
import { createView, MARK_COLORS } from './view.js';
import {
    countOverlay, maskOverlay, classOverlay, residualOverlay, rgbOverlay, cssColor,
    COUNT_MAX_SHOWN, CLASS_COLORS, OFF_LINE_COLOR, MASK_TINT, RESIDUAL_FULL, RESIDUAL_DARKER, RESIDUAL_LIGHTER, RESIDUAL_MASKED,
} from './overlays.js';

const SAMPLE_SIZE = 80; // CSS px, each sample canvas
const SAMPLE_MARGIN = 4; // px around the dot in the sample crop
const SAMPLE_DIFF_GAIN = 10; // the diff sample shows |image − model| × this
// what the dot sample shows for dots colored by a scale (families.js `scale`)
const SCALE_CAPTION = {
    translucent: 'image as one virtual family (counts along each pixel’s color)',
    opaque: 'image as one virtual opaque family (every dot pixel covered, outlines included)',
    outline: 'image with the fill colors projected out (rings and their occlusion), model of the ring alone',
};
const EXTRAPOLATED_SWATCHES = 3; // level swatches shown beyond the measured ones
const EDIT_HIT_RADII = 0.6; // × r: a click this close to a dot's center removes it in edit mode

const JPEG_NAME = /\.jpe?g$/i;

const $ = id => document.getElementById(id);

const state = {
    name: null, // display name of the current image
    image: null, // RgbaImage
    jpeg: false, // JPEG-compressed: noise-tolerant style thresholds (from the file type, or set by the user)
    work: null, // the image the analysis sees: for a JPEG, its chroma restored (src/core/jpeg.js); else the image
    // every family's estimate (src/core/families.js), each with the image to fit it on; null for a picked color
    familyEstimates: null,
    base: null, // state.work with any translucent layer over the dots divided out (src/core/background.js)
    hash: null,
    estimate: null, // from estimateStyle
    seed: null, // a user-picked family color, or null
    family: 0, // candidate index when no seed
    overrides: {}, // radius, alpha, blur set by the user
    dilate: null, // user's mask dilation, px, or null for the default
    layers: null, // from buildMask
    overlay: 'image',
    picking: false,
    fit: null, // { dots, radius, sigma, lambda, log, ms, edited } once a fit has run
    worker: null,
    fitStart: 0,
    editing: false,
    showDots: true,
    axes: null, // from detectAxes
    axisEntries: { x: null, y: null }, // typed tick values and scale types
    showAxes: true,
};

const view = createView({
    stage: $('stage'),
    canvas: $('view'),
    onHover: showPixel,
    onClick: pickAt,
    onZoom: scale => {
        $('zoom-level').textContent = `${scale >= 1 ? Math.round(scale * 100) : (scale * 100).toFixed(1)}%`;
    },
});

const fmt = (v, d) => (v === null || v === undefined || Number.isNaN(v) ? '–' : v.toFixed(d));

/** The style in effect: the estimate with the user's overrides. */
function currentStyle() {
    const s = state.estimate?.style;
    if (!s)
        return null;
    return { ...s, ...state.overrides };
}

/** The family the panel shows: a candidate index, or −1 for a picked color. */
function currentFamily() {
    return state.seed ? -1 : state.family;
}

/** One family's fit result, or null. */
function familyFit(k = currentFamily()) {
    return state.fit?.families.find(f => f.family === k) ?? null;
}

/** Set the fit from per-family results; every dot carries its family and radius. */
function setFits(families, extra = {}) {
    const dots = families.flatMap(f => f.dots.map(d => ({ ...d, family: f.family, r: f.radius })));
    state.fit = { families, dots, ...extra };
}

/** The style the current family's fitted dots were drawn with: the style in effect, with the fit's radius. */
function fittedStyle() {
    const style = currentStyle();
    const f = familyFit();
    return style && f ? { ...style, radius: f.radius } : style;
}

/** The image the current family is analyzed on: its own image from families.js, else the base image. */
function analysisImage() {
    return state.estimate?.image ?? state.base ?? state.work;
}

/** What the fit needs for one family: the current one as shown (overrides applied), others as estimated. */
function fitInputFor(k) {
    if (k === currentFamily()) {
        return {
            image: analysisImage(), style: currentStyle(), line: state.estimate.familyMeasure.line,
            mask: state.layers.mask, D: state.estimate.D, imgT: state.estimate.fitT, base: state.estimate.base,
        };
    }
    const est = state.familyEstimates?.[k] ?? estimateStyle(state.base ?? state.work, { family: k, candidates: state.estimate.candidates, jpeg: state.jpeg });
    const image = est.image ?? state.base ?? state.work;
    const layers = buildMask(image, est, { bounds: plotBounds(state.axes) });
    return { image, style: est.style, line: est.familyMeasure.line, mask: layers.mask, D: est.D, imgT: est.fitT, base: est.base };
}

function familyLabel(k) {
    if (k < 0)
        return 'picked color';
    const c = state.estimate?.candidates[k]?.seed;
    return c ? `family ${k + 1} (${c.map(Math.round).join(', ')})` : `family ${k + 1}`;
}

/** The scales in effect: { x, y }, each from axisScale or null. */
function scales() {
    return {
        x: state.axes ? axisScale(state.axes.x, state.axisEntries.x) : null,
        y: state.axes ? axisScale(state.axes.y, state.axisEntries.y) : null,
    };
}

/** Data coordinates of an image point, with per-axis resolution (data units per px). */
function dataAt(x, y) {
    const s = scales();
    return {
        x: toAxisData(s.x, x), y: toAxisData(s.y, y),
        resX: axisResolution(s.x, x), resY: axisResolution(s.y, y),
    };
}

// index of the nearest dot within `within` × its radius, among `dots`
function nearestDot(dots, x, y, within = 1) {
    let best = -1;
    let bestD = Infinity;
    dots.forEach((d, i) => {
        const dd = Math.hypot(d.x - x, d.y - y);
        if (dd <= within * d.r && dd < bestD) {
            bestD = dd;
            best = i;
        }
    });
    return best;
}

function showPixel(x, y) {
    if (x === null || !state.image) {
        for (const id of ['pixel-pos', 'pixel-rgba', 'pixel-count', 'pixel-class', 'pixel-dot', 'pixel-data'])
            $(id).textContent = '–';
        $('pixel-swatch').style.background = 'transparent';
        return;
    }
    const [r, g, b, a] = pixelAt(state.image, x, y);
    $('pixel-pos').textContent = `${x}, ${y} px`;
    $('pixel-rgba').textContent = `${r}, ${g}, ${b}, ${a}`;
    $('pixel-swatch').style.background = `rgb(${r} ${g} ${b})`;
    const p = y * state.image.width + x;
    const D = state.estimate?.D;
    $('pixel-count').textContent = !D ? '–' : (Number.isNaN(D[p]) ? 'not the dot color' : D[p].toFixed(2));
    const L = state.layers;
    if (L) {
        const cls = L.thin[p] ? 'thin mark in dot color' : (L.classes[p] === CLASS_BACKGROUND ? 'background' : (L.classes[p] === CLASS_FAMILY ? 'dot color' : 'other'));
        $('pixel-class').textContent = `${cls}${L.mask[p] ? ', masked' : ''}`;
    }
    const pd = dataAt(x + 0.5, y + 0.5);
    $('pixel-data').textContent = pd.x === null && pd.y === null ? '–' : `${formatData(pd.x, pd.resX)}, ${formatData(pd.y, pd.resY)}`;
    const i = state.fit ? nearestDot(state.fit.dots, x + 0.5, y + 0.5) : -1;
    if (i < 0) {
        $('pixel-dot').textContent = '–';
        return;
    }
    const d = state.fit.dots[i];
    const fam = state.fit.families.length > 1 ? `${familyLabel(d.family)}: ` : '';
    // the curvature standard errors reflect pixel noise only; systematic error (resampled rims, model mismatch) is
    // several times larger (PLAN.md phase 4: two renderings of dog-ages differ by 3–6× these), so say so
    const se = d.seX === undefined ? '' : ` ± ${d.seX.toFixed(2)}, ${d.seY.toFixed(2)} (noise only)`;
    const extra = d.visible === undefined ? '' : `; ${Math.round(100 * d.visible)}% visible, depth ${d.depth}, margin ${d.margin.toFixed(0)}λ`;
    const dd = dataAt(d.x, d.y);
    const data = dd.x === null && dd.y === null ? '' : ` = data ${formatData(dd.x, dd.resX)}, ${formatData(dd.y, dd.resY)}`;
    $('pixel-dot').textContent = `${fam}${d.x.toFixed(2)}, ${d.y.toFixed(2)} px${se}${data}${extra}${d.flags?.length ? ` (${d.flags.join(', ')})` : ''}`;
}

function showMessage(text, isError = false) {
    const el = $('stage-message');
    el.textContent = text;
    el.classList.toggle('error', isError);
    el.hidden = !text;
}

/**
 * The dots rendered in RGB with the style. An outlined opaque family's dots are an outline disc with the fill drawn
 * over it: exact for one dot, but here every outline is drawn before every fill (the chart's order isn't known), so
 * where dots overlap the picture differs from the chart's.
 */
function renderModel(bg, dots, style) {
    if (!style.outline)
        return renderDots(bg, dots, style);
    // the level table would paint over the background image, so both passes composite over it instead
    const rings = renderDots(bg, dots, { ...style, color: style.outline.color, levels: undefined });
    return renderDots(rings, dots, { ...style, radius: style.outline.inner * style.radius, levels: undefined });
}

function displayImage() {
    const { image, estimate, layers } = state;
    if (state.overlay === 'count' && estimate?.D)
        return countOverlay(estimate.D, image.width, image.height);
    if (state.overlay === 'mask' && layers)
        return maskOverlay(image, layers.mask);
    if (state.overlay === 'classes' && layers)
        return classOverlay(layers.classes, layers.thin, image.width, image.height);
    const fam = familyFit();
    if ((state.overlay === 'model' || state.overlay === 'residual') && fam) {
        // the current family only: each family is fitted, and compared, along its own color line
        const style = fittedStyle();
        const line = estimate.familyMeasure.line;
        const bg = solidImage(image.width, image.height, line.B);
        if (state.overlay === 'model')
            return rgbOverlay(renderModel(bg, fam.dots, style));
        const modelT = renderDots(bg, fam.dots, { ...style, project: line }).data;
        let imgT = estimate.fitT;
        if (!imgT) {
            imgT = new Float32Array(image.width * image.height);
            for (let p = 0; p < imgT.length; p++)
                imgT[p] = [0, 1, 2].reduce((sum, ch) => sum + (line.B[ch] - analysisImage().data[4 * p + ch]) * line.u[ch], 0);
        }
        return residualOverlay(imgT, modelT, layers.mask, image.width, image.height);
    }
    return image;
}

function refreshView(keepTransform) {
    view.setImage(displayImage(), { keepTransform });
    refreshMarks();
}

function refreshMarks(dots = state.fit?.dots) {
    const style = currentStyle();
    view.setMarks(state.showDots && dots && style ? { dots, radius: style.radius } : null);
}

/** Estimate the style (with the current seed or family) and rebuild the mask. */
function analyze() {
    const t0 = performance.now();
    if (state.seed) {
        // a picked color is one family on its own, on the image with any overlay already divided out
        state.familyEstimates = null;
        state.estimate = estimateStyle(state.base ?? state.work, { seed: state.seed, jpeg: state.jpeg });
    }
    else {
        const prepared = estimateFamilies(state.work, { jpeg: state.jpeg });
        state.base = prepared.image;
        state.familyEstimates = prepared.estimates;
        // merging a family seen over two backgrounds can leave fewer families than the index
        state.family = Math.min(state.family, prepared.estimates.length - 1);
        state.estimate = prepared.estimates[state.family];
    }
    const t1 = performance.now();
    rebuildMask();
    const t2 = performance.now();
    $('analysis-time').textContent = `style ${Math.round(t1 - t0)} ms, mask ${Math.round(t2 - t1)} ms`;
    renderStylePanel();
}

// Any change of style or mask invalidates fitted dots: they were fitted to a different model.
function rebuildMask() {
    stopFit();
    state.fit = null;
    const style = currentStyle();
    const opts = { bounds: plotBounds(state.axes), ...(state.dilate === null ? {} : { dilate: state.dilate }) };
    state.layers = style ? buildMask(analysisImage(), { ...state.estimate, style }, opts) : null;
    renderLayersPanel();
    renderFitPanel();
}

function renderStylePanel() {
    const est = state.estimate;
    const style = currentStyle();
    const select = $('family');
    select.replaceChildren();
    est.candidates.forEach((c, i) => {
        const label = c.kind === 'opaque'
            ? `${i + 1}: opaque, r ${fmt(c.radius, 1)}`
            : c.kind === 'outline'
                ? `${i + 1}: outlined, r ${fmt(c.radius, 1)}, fills vary`
                : `${i + 1}: translucent, α ${fmt(c.fit.alpha, 2)}, ${c.levels.length} levels`;
        select.add(new Option(label, String(i)));
    });
    if (state.seed)
        select.add(new Option(`picked ${state.seed.join(', ')}`, 'seed'));
    select.value = state.seed ? 'seed' : String(state.family);
    select.disabled = !est.candidates.length && !state.seed;

    const swatches = $('levels');
    swatches.replaceChildren();
    // a color scale is fitted as a virtual family whose color and levels no dot has: don't show them as the dots'
    if (style && !est.scale) {
        const K = style.measuredLevels;
        style.levels.slice(1, K + 1 + EXTRAPOLATED_SWATCHES).forEach((c, i) => {
            const el = document.createElement('span');
            el.className = 'swatch' + (i >= K ? ' extrapolated' : '');
            el.style.background = cssColor(c);
            el.title = `${i + 1} overlapping: ${c.map(v => v.toFixed(1)).join(', ')}${i >= K ? ' (extrapolated)' : ' (measured)'}`;
            swatches.append(el);
        });
    }
    $('family-swatch').style.background = style && !est.scale ? cssColor(style.color) : 'transparent';
    $('style-c').textContent = !style ? '–' : est.scale ? 'varies by dot (color scale)' : style.color.map(v => v.toFixed(1)).join(', ');
    $('style-c').title = style && est.scale ? `fitted as one virtual family of color ${style.color.map(v => v.toFixed(1)).join(', ')}` : '';
    $('style-levels').textContent = style ? `${style.measuredLevels} measured` : '–';
    const fam = est.familyMeasure;
    if (fam?.kind === 'opaque')
        $('style-fit').textContent = 'opaque: overlaps aren’t seen, α = 1';
    else if (fam?.fit?.underdetermined)
        $('style-fit').textContent = 'one level: α not measured, 1 assumed';
    else
        $('style-fit').textContent = fam?.fit ? `rms ${fmt(fam.fit.rms, 2)}, max ${fmt(fam.fit.maxAbs, 2)}` : '–';
    $('style-isolated').textContent = String(est.isolated.length);
    for (const [key, digits] of [['radius', 2], ['alpha', 3], ['blur', 2]]) {
        const input = $(`style-${key}`);
        input.value = style && style[key] !== null ? style[key].toFixed(digits) : '';
        input.classList.toggle('overridden', key in state.overrides);
        input.disabled = !style;
    }
    $('style-reset').disabled = !Object.keys(state.overrides).length;
    $('style-notes').replaceChildren(...est.notes.map(n => Object.assign(document.createElement('p'), { textContent: n })));
    renderSample();
}

// The first isolated dot, beside the model's rendering of it at the measured center, and their difference.
function renderSample() {
    const est = state.estimate;
    const style = currentStyle();
    const box = $('sample');
    box.hidden = !(style?.radius && est.isolated.length);
    if (box.hidden)
        return;
    const dot = est.isolated[0];
    const half = Math.ceil(style.radius + SAMPLE_MARGIN);
    const x0 = Math.max(0, Math.floor(dot.cx) - half);
    const y0 = Math.max(0, Math.floor(dot.cy) - half);
    const size = Math.min(2 * half + 1, state.image.width - x0, state.image.height - y0);
    // a color scale's model is in the virtual family's color, so it is compared with the virtual image it was fitted on
    const seen = est.scale ? analysisImage() : state.work;
    const model = renderModel(solidImage(size, size, est.background.color), [{ x: dot.cx - x0, y: dot.cy - y0 }], style);
    const real = new ImageData(size, size);
    const mod = new ImageData(size, size);
    const diff = new ImageData(size, size);
    let maxDiff = 0;
    let sumDiff = 0;
    for (let y = 0; y < size; y++) {
        for (let x = 0; x < size; x++) {
            const src = 4 * ((y0 + y) * seen.width + x0 + x);
            const p = y * size + x;
            let worst = 0;
            for (let ch = 0; ch < 3; ch++) {
                real.data[4 * p + ch] = seen.data[src + ch];
                mod.data[4 * p + ch] = Math.round(model.data[3 * p + ch]);
                worst = Math.max(worst, Math.abs(seen.data[src + ch] - model.data[3 * p + ch]));
            }
            maxDiff = Math.max(maxDiff, worst);
            sumDiff += worst;
            const v = Math.min(255, Math.round(SAMPLE_DIFF_GAIN * worst));
            diff.data.set([v, v, v], 4 * p);
            real.data[4 * p + 3] = mod.data[4 * p + 3] = diff.data[4 * p + 3] = 255;
        }
    }
    for (const [id, data] of [['sample-real', real], ['sample-model', mod], ['sample-diff', diff]]) {
        const c = $(id);
        c.width = size;
        c.height = size;
        c.style.width = c.style.height = `${SAMPLE_SIZE}px`;
        c.getContext('2d').putImageData(data, 0, 0);
    }
    $('sample-caption').textContent = `Dot at ${dot.cx.toFixed(1)}, ${dot.cy.toFixed(1)} px: ${SCALE_CAPTION[est.scale] ?? 'image'}${state.jpeg ? ' (chroma restored)' : ''}, model, |difference| × ${SAMPLE_DIFF_GAIN} (mean ${(sumDiff / (size * size)).toFixed(2)}, max ${maxDiff.toFixed(1)}, 0–255)`;
}

function renderLayersPanel() {
    const L = state.layers;
    $('layer-dilate').value = L ? L.dilate.toFixed(1) : '';
    $('layer-dilate').disabled = !L;
    const n = state.image ? state.image.width * state.image.height : 0;
    if (!L) {
        $('layer-stats').textContent = '–';
        return;
    }
    const s = L.stats;
    const pct = v => `${(100 * v / n).toFixed(1)}%`;
    $('layer-stats').textContent = `background ${pct(s.background)}, dot color ${pct(s.family)}, other ${pct(s.other)}, `
        + `thin marks ${s.thin} px. Masked ${pct(s.masked)}, including ${s.maskedFamily} dot-color px near other marks`
        + (s.outside ? ` and ${pct(s.outside)} beyond the axes (labels, titles).` : '.');
    if (state.image)
        refreshView(true);
}

function renderLegend() {
    const legend = $('overlay-legend');
    const item = (color, text) => `<span class="key" style="background:${typeof color === 'string' ? color : cssColor(color)}"></span>${text}`;
    $('dots-legend').innerHTML = `${item(MARK_COLORS.ok, 'dot')} ${item(MARK_COLORS.flagged, 'flagged (hover for why)')}`;
    if (state.overlay === 'residual') {
        legend.innerHTML = `Image − model along the dot color line: ${item(RESIDUAL_DARKER, 'image darker (missing ink)')} `
            + `${item(RESIDUAL_LIGHTER, 'image lighter (too much)')} ${item(RESIDUAL_MASKED, 'masked')}; full color at ±${RESIDUAL_FULL} units.`;
        return;
    }
    if (state.overlay === 'model') {
        legend.textContent = 'The fitted dots rendered with the style (radius from the fit).'
            + (currentStyle()?.outline ? ' Outlines are all drawn before the fills, so overlaps can differ from the chart’s drawing order.' : '');
        return;
    }
    if (state.overlay === 'count')
        legend.innerHTML = `Overlap count: black 0 to white ${COUNT_MAX_SHOWN}+ (hover for the value); ${item(OFF_LINE_COLOR, 'not the dot color')}`;
    else if (state.overlay === 'mask')
        legend.innerHTML = `${item(MASK_TINT, 'masked')} pixels are ignored by the fit`;
    else if (state.overlay === 'classes')
        legend.innerHTML = [item(CLASS_COLORS.background, 'background'), item(CLASS_COLORS.family, 'dot color'), item(CLASS_COLORS.other, 'other'), item(CLASS_COLORS.thin, 'thin mark in dot color')].join(' ');
    else
        legend.textContent = '';
}

function pickAt(x, y) {
    if (state.editing && state.fit) {
        editAt(x + 0.5, y + 0.5);
        return;
    }
    if (!state.picking || !state.image)
        return;
    const [r, g, b] = pixelAt(state.work, x, y);
    state.seed = [r, g, b];
    state.overrides = {};
    setPicking(false);
    analyze();
}

function setPicking(on) {
    state.picking = on;
    $('pick-color').setAttribute('aria-pressed', String(on));
    $('stage').classList.toggle('picking', on);
}

function renderFitPanel(progress) {
    const fit = state.fit;
    const running = state.worker !== null;
    $('fit-run').disabled = running || !currentStyle()?.radius;
    $('fit-run').title = currentStyle()?.radius
        ? (state.seed ? 'Find the dots of the picked color' : 'Find the dots of every family')
        : 'Set the dot radius first';
    $('fit-stop').disabled = !running;
    $('fit-edit').disabled = !familyFit() || running;
    $('copy-csv').disabled = !fit || running;
    $('save-truth').disabled = !fit || running;
    if (running && progress) {
        const secs = ((performance.now() - state.fitStart) / 1000).toFixed(1);
        $('fit-status').textContent = `${familyLabel(progress.family)}: ${progress.stage}, round ${progress.round}: ${progress.dots.length} dots, ${secs} s`;
    }
    if (!fit) {
        for (const id of ['fit-count', 'fit-flags', 'fit-radius', 'fit-sigma'])
            $(id).textContent = '–';
        if (!running)
            $('fit-status').textContent = '';
        return;
    }
    const counts = {};
    for (const d of fit.dots) {
        for (const f of d.flags)
            counts[f] = (counts[f] ?? 0) + 1;
    }
    const flagged = fit.dots.filter(d => d.flags.length).length;
    const many = fit.families.length > 1;
    $('fit-count').textContent = many ? `${fit.dots.length} (${fit.families.map(f => f.dots.length).join(' + ')})` : String(fit.dots.length);
    $('fit-flags').textContent = flagged ? `${flagged} (${Object.entries(counts).map(([k, v]) => `${k} ${v}`).join(', ')})` : '0';
    $('fit-radius').textContent = `${fit.families.map(f => f.radius.toFixed(2)).join(', ')} px`;
    const cur = familyFit();
    $('fit-sigma').textContent = cur ? fmt(cur.sigma, 2) + (many ? ' (current family)' : '') : '–';
    if (!running) {
        $('fit-status').textContent = `${fit.partial ? 'Stopped early: partial result, not assessed. ' : ''}`
            + `${fit.edited ? 'Edited by hand. ' : ''}${fit.ms ? `Fit in ${(fit.ms / 1000).toFixed(1)} s. ` : ''}`
            + 'Positions in image px (pixel i spans i to i + 1).';
    }
}

// Fit every dot family (or just the picked color), one after another in the worker.
function runFit() {
    stopFit();
    state.fitStart = performance.now();
    const ks = state.seed ? [-1] : state.estimate.candidates.map((_, k) => k);
    const inputs = [];
    const skipped = [];
    for (const k of ks) {
        const input = fitInputFor(k);
        if (input.style?.radius > 0)
            inputs.push({ family: k, input });
        else
            skipped.push(k);
    }
    const done = [];
    let last = null;
    state.fit = null;
    state.worker = new Worker(new URL('../fit-worker.js', import.meta.url), { type: 'module' });
    const radiusOf = k => inputs.find(i => i.family === k).input.style.radius;
    state.worker.onmessage = e => {
        const msg = e.data;
        if (msg.type === 'progress') {
            last = msg;
            const r = radiusOf(msg.family);
            refreshMarks([...done.flatMap(f => f.dots.map(d => ({ ...d, r: f.radius }))), ...msg.progress.dots.map(d => ({ ...d, r }))]);
            renderFitPanel({ ...msg.progress, family: msg.family });
            return;
        }
        if (msg.type === 'family') {
            done.push({ ...msg.result, family: msg.family });
            last = null;
            return;
        }
        state.worker.terminate();
        state.worker = null;
        if (msg.type === 'error') {
            $('fit-status').textContent = `Fit failed: ${msg.message}`;
            renderFitPanel();
            return;
        }
        setFits(done, { ms: performance.now() - state.fitStart, skipped });
        renderFitPanel();
        refreshView(true);
    };
    state.worker.onerror = e => {
        $('fit-status').textContent = `Fit failed: ${e.message}`;
        state.worker = null;
        renderFitPanel();
    };
    // a stopped fit keeps the finished families and what the running one had reached
    state.stopFit = () => {
        const partial = [...done];
        if (last)
            partial.push({ family: last.family, dots: last.progress.dots.map(d => ({ ...d, flags: [] })), radius: radiusOf(last.family), sigma: NaN });
        return partial;
    };
    state.worker.postMessage({ inputs, opts: {} });
    renderFitPanel({ stage: 'starting', round: 0, dots: [], family: inputs[0]?.family ?? 0 });
}

function stopFit() {
    if (!state.worker)
        return;
    state.worker.terminate();
    state.worker = null;
    const partial = state.stopFit?.() ?? [];
    if (partial.length)
        setFits(partial, { partial: true });
    renderFitPanel();
    refreshView(true);
}

// add or remove a dot of the current family, then refit its neighbors
function editAt(x, y) {
    const f = familyFit();
    if (!f)
        return;
    const own = f.dots.map(d => ({ ...d, r: f.radius }));
    const i = nearestDot(own, x, y, EDIT_HIT_RADII);
    const dots = f.dots.map(d => ({ x: d.x, y: d.y }));
    if (i >= 0)
        dots.splice(i, 1);
    else
        dots.push({ x, y });
    const res = refitNear({ ...fitInputFor(currentFamily()), style: fittedStyle() }, dots, x, y);
    const families = state.fit.families.map(g => (g === f ? { ...f, dots: res.dots, sigma: res.sigma } : g));
    setFits(families, { ...state.fit, families: undefined, dots: undefined, edited: true });
    renderFitPanel();
    refreshView(true);
}

function setEditing(on) {
    state.editing = on;
    $('fit-edit').setAttribute('aria-pressed', String(on));
    $('stage').classList.toggle('editing', on);
    if (on)
        setPicking(false);
}

function dotsCsv() {
    const s = scales();
    const hasData = s.x?.kind === 'numeric' || s.x?.kind === 'category' || s.y?.kind === 'numeric' || s.y?.kind === 'category';
    // se_noise: standard errors from pixel noise alone; the real error is larger (see the Axes note)
    const head = ['family', 'x_px', 'y_px', 'se_noise_x_px', 'se_noise_y_px'];
    if (hasData)
        head.push('x', 'y', 'se_noise_x', 'se_noise_y');
    head.push('visible', 'depth', 'flags');
    const rows = [head.join('\t')];
    for (const d of state.fit.dots) {
        const color = d.family >= 0 ? state.estimate.candidates[d.family]?.seed : state.seed;
        const fam = `${d.family + 1}${color ? ' #' + color.map(v => Math.round(v).toString(16).padStart(2, '0')).join('') : ''}`;
        const row = [fam, d.x.toFixed(3), d.y.toFixed(3), d.seX?.toFixed(3) ?? '', d.seY?.toFixed(3) ?? ''];
        if (hasData) {
            const dd = dataAt(d.x, d.y);
            // standard errors in data units: the px error times the local data units per px
            const se = (sePx, res) => (sePx === undefined || res === null ? '' : (sePx * res).toPrecision(2));
            row.push(formatData(dd.x, dd.resX), formatData(dd.y, dd.resY), se(d.seX, dd.resX), se(d.seY, dd.resY));
        }
        row.push(d.visible?.toFixed(2) ?? '', d.depth ?? '', d.flags.join(' '));
        rows.push(row.join('\t'));
    }
    return rows.join('\n') + '\n';
}

function renderAxesPanel() {
    for (const k of ['x', 'y']) {
        renderAxis($(`axis-${k}`), k, state.axes?.[k] ?? null, state.axisEntries[k], state.image, entry => {
            state.axisEntries[k] = entry;
            saveEntries(state.hash, state.axisEntries);
            renderAxesPanel();
        });
    }
    const s = scales();
    const note = [];
    if (s.x?.kind === 'numeric' && s.y?.kind === 'numeric')
        note.push('Data coordinates appear in the Pixel readout, the dot hover and the CSV. Dot standard errors count pixel noise only; '
            + 'with resampled or antialiased edges the real position error is several times larger (a few tenths of a px), '
            + 'and tick placement adds the scale residual shown above.');
    else
        note.push('Type tick values to get data coordinates.');
    $('axes-note').textContent = note.join(' ');
    refreshAxisMarks();
}

function refreshAxisMarks() {
    const a = state.axes;
    if (!a || !state.showAxes) {
        view.setAxisMarks(null);
        return;
    }
    const marks = [];
    if (a.x)
        // gridline ticks have no axis line: mark them at the bottom end of the y axis, or the image's bottom edge
        marks.push({ dir: 'h', pos: a.x.line?.pos ?? null, at: a.x.line ? null : (a.y?.line?.to ?? state.image.height - 1), from: a.x.line?.from, to: a.x.line?.to, ticks: a.x.ticks.map(t => t.pos) });
    if (a.y) {
        // gridline ticks have no axis line: mark them at the left end of the x axis, or the image's left edge
        const at = a.y.line ? null : (a.x?.line?.from ?? 0);
        marks.push({ dir: 'v', pos: a.y.line?.pos ?? null, at, from: a.y.line?.from, to: a.y.line?.to, ticks: a.y.ticks.map(t => t.pos) });
    }
    view.setAxisMarks(marks);
}

function saveTruth() {
    const truth = {
        image: state.name, hash: state.hash,
        note: 'Dot centers in image px. Review every dot before using this as ground truth.',
        families: state.fit.families.map(f => ({ family: f.family, color: f.family >= 0 ? state.estimate.candidates[f.family].seed : state.seed, radius: f.radius })),
        dots: state.fit.dots.map(d => ({ x: +d.x.toFixed(2), y: +d.y.toFixed(2), family: d.family })),
    };
    const a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob([JSON.stringify(truth, null, 1) + '\n'], { type: 'application/json' }));
    a.download = state.name.replace(/\.[a-z]+$/i, '') + '.json';
    a.click();
    URL.revokeObjectURL(a.href);
}

function setJpeg(jpeg) {
    state.jpeg = jpeg;
    state.work = jpeg ? restoreChroma(state.image) : state.image;
    state.base = null;
    $('jpeg').checked = jpeg;
}

function setImage(name, image, jpeg) {
    state.name = name;
    state.image = image;
    setJpeg(jpeg);
    state.hash = pixelHash(image);
    state.seed = null;
    state.family = 0;
    state.overrides = {};
    state.dilate = null;
    $('info-name').textContent = name;
    $('info-size').textContent = `${image.width} × ${image.height} px`;
    $('info-hash').textContent = state.hash;
    showMessage('');
    view.setImage(image);
    state.axes = detectAxes(image);
    state.axisEntries = loadEntries(state.hash, state.axes);
    renderAxesPanel();
    analyze();
    $('stage').focus();
}

async function openTestImage(name) {
    showMessage(`Loading ${name}…`);
    try {
        setImage(name, await loadTestImage(name), JPEG_NAME.test(name));
        history.replaceState(null, '', `#image=${encodeURIComponent(name)}`);
        $('test-image').value = name;
    }
    catch (err) {
        showMessage(`Could not open ${name}: ${err.message}`, true);
    }
}

async function openFile(file) {
    const name = file.name && file.name !== 'image.png' ? file.name : 'pasted image';
    showMessage(`Decoding ${name}…`);
    try {
        setImage(name, await decodeImage(file), file.type === 'image/jpeg' || JPEG_NAME.test(file.name ?? ''));
        // a pasted or uploaded image can't be reopened from a link
        history.replaceState(null, '', location.pathname);
        $('test-image').value = '';
    }
    catch (err) {
        showMessage(`Could not decode ${name}: ${err.message}`, true);
    }
}

function wireControls() {
    $('jpeg').addEventListener('change', e => {
        if (!state.image)
            return;
        setJpeg(e.target.checked);
        // the candidates change with the thresholds, so the family index may name another family
        state.family = 0;
        analyze();
    });
    $('family').addEventListener('change', e => {
        if (e.target.value === 'seed')
            return;
        state.family = Number(e.target.value);
        state.overrides = {};
        // every family was prepared together; only a switch back from a picked color needs the analysis again
        if (state.seed || !state.familyEstimates) {
            state.seed = null;
            analyze();
            return;
        }
        state.estimate = state.familyEstimates[state.family];
        rebuildMask();
        renderStylePanel();
    });
    $('pick-color').addEventListener('click', () => {
        setPicking(!state.picking);
        if (state.picking)
            setEditing(false);
    });
    $('fit-run').addEventListener('click', runFit);
    $('fit-stop').addEventListener('click', stopFit);
    $('fit-edit').addEventListener('click', () => setEditing(!state.editing));
    $('show-axes').addEventListener('change', e => {
        state.showAxes = e.target.checked;
        refreshAxisMarks();
    });
    $('show-dots').addEventListener('change', e => {
        state.showDots = e.target.checked;
        refreshMarks();
    });
    $('copy-csv').addEventListener('click', async () => {
        await navigator.clipboard.writeText(dotsCsv());
        $('fit-status').textContent = `Copied ${state.fit.dots.length} dots as CSV.`;
    });
    $('save-truth').addEventListener('click', saveTruth);
    for (const key of ['radius', 'alpha', 'blur']) {
        $(`style-${key}`).addEventListener('change', e => {
            const v = Number(e.target.value);
            if (e.target.value === '' || !Number.isFinite(v))
                delete state.overrides[key];
            else
                state.overrides[key] = v;
            rebuildMask();
            renderStylePanel();
        });
    }
    $('style-reset').addEventListener('click', () => {
        state.overrides = {};
        rebuildMask();
        renderStylePanel();
    });
    $('layer-dilate').addEventListener('change', e => {
        const v = Number(e.target.value);
        state.dilate = e.target.value === '' || !Number.isFinite(v) ? null : Math.max(0, v);
        rebuildMask();
    });
    for (const input of document.querySelectorAll('input[name="overlay"]')) {
        input.addEventListener('change', () => {
            state.overlay = input.value;
            renderLegend();
            if (state.image)
                refreshView(true);
        });
    }
    document.addEventListener('keydown', e => {
        if (e.key === 'Escape') {
            setPicking(false);
            setEditing(false);
        }
    });
}

async function init() {
    const names = await listTestImages();
    const select = $('test-image');
    for (const name of names)
        select.add(new Option(name, name));
    if (!names.length)
        select.disabled = true;
    select.addEventListener('change', () => {
        if (select.value)
            openTestImage(select.value);
    });

    $('open-file').addEventListener('click', () => $('file-input').click());
    wireImageInput({ dropTarget: $('stage'), fileInput: $('file-input'), onFile: openFile });
    wireControls();
    renderLegend();

    $('zoom-fit').addEventListener('click', () => view.fit());
    $('zoom-1').addEventListener('click', () => view.zoomTo(1));
    $('zoom-in').addEventListener('click', () => view.zoomBy(view.ZOOM_STEP));
    $('zoom-out').addEventListener('click', () => view.zoomBy(1 / view.ZOOM_STEP));

    const match = /^#image=(.+)$/.exec(location.hash);
    if (match)
        await openTestImage(decodeURIComponent(match[1]));
}

// for tools/check-page.mjs and the console
window.dotscraper = { state, ready: init() };

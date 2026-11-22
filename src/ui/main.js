// App entry: image source and action toolbar, the view, and the details panel.
//
// Dev mode (on where the app is served from this machine, or anywhere with ?dev; ?dev=0 turns it off) adds what only
// development needs: the list of local test images, the current one's name kept in the URL hash (#image=<file>) so a
// reload reopens it, Save truth, and the diagnostic views (Count, Mask, Classes, Model; elements with class dev-only). The deployed site never has the test images, so it doesn't look for them.
//
// On each new image the app estimates every dot family (src/core/families.js, in a module worker, src/scan-worker.js,
// with its progress shown: on large images it takes up to a minute) and builds the selected family's mask
// (src/core/layers.js). The panel lists the families: checked ones are fitted by Find dots, the selected one is
// shown in detail (its style, overrides, overlays) and is the one Add and Remove edit. A color picked in the image
// adds a family of its own.

import { pixelAt, pixelHash } from '../core/image.js';
import { restoreChroma, jpegNoise } from '../core/jpeg.js';
import { buildMask, CLASS_BACKGROUND, CLASS_FAMILY } from '../core/layers.js';
import { renderDots, solidImage, gaussianKernel, blurRect } from '../core/render.js';
import { refitNear, flagClose, MAX_ROUNDS } from '../core/fit.js';
import { createOutlineModel } from '../core/outline.js';
import { detectAxes, plotBounds } from '../core/axes.js';
import { readTickLabels } from '../core/labels.js';
import { excludedBoxes, familyClips, maskOptions, fitInput, dotFills } from '../core/extract.js';
import { pageStroke } from '../core/stroke.js';
import { axisScale, toAxisData, axisResolution, formatData, loadEntries, saveEntries, renderAxis, tickStatus, axisInEffect, readingInEffect, addTick } from './axes-panel.js';
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
const REMOVE_HIT_RADII = 1; // × r: Remove takes the nearest dot of the selected family within this of the click
const UNDO_DEPTH = 100; // edits kept for Undo
const NUDGE_STEP = 1; // px, an arrow key in Move mode
const NUDGE_FINE = 0.1; // px, Shift + an arrow key
// the modifier for edits without a refit, by its name on this platform
const OPTION_KEY = /Mac/.test(navigator.platform) ? 'Option' : 'Alt';
const LOCAL_HOST = /^(localhost|127\.0\.0\.1|\[::1\]|.+\.localhost)$/;
const DEV = devMode(location);

// ?dev or ?dev=1 on, ?dev=0 or ?dev=false off; without the parameter, on for a local server
function devMode({ search, hostname }) {
    const param = new URLSearchParams(search).get('dev');
    if (param !== null)
        return param !== '0' && param !== 'false';
    return LOCAL_HOST.test(hostname);
}
const OUTLINE_DEFAULT = { color: [60, 60, 80], width: 1.5 }; // 0–255, px: an outline turned on for a family without one
// Find dots progress: a family's share of its bar at each stage the fit reports (rounds share the middle; a fit
// may converge before MAX_ROUNDS, so the bar can jump at the end)
const PROGRESS_CANDIDATES = 0.05;
const PROGRESS_ROUNDS_END = 0.85;
const PROGRESS_RADIUS = 0.95;
const OPEN_KEY = 'dotscraper:open:'; // localStorage prefix for each panel section's open state
const STATUS_TICK = 250; // ms between updates of the elapsed time while a fit runs

const JPEG_NAME = /\.jpe?g$/i;

let loads = 0; // counts image loads and rescans, so a slow one that a newer one overtook drops its result

const $ = id => document.getElementById(id);

const state = {
    name: null, // display name of the current image
    image: null, // RgbaImage
    jpeg: false, // JPEG-compressed: noise-tolerant style thresholds (from the file type or JPEG noise, or set by the user)
    noise: null, // jpegNoise of the image
    jpegAuto: false, // JPEG mode was turned on because no family showed with exact colors and the image has JPEG noise
    jpegFile: false, // the file is a JPEG
    work: null, // the image the analysis sees: for a JPEG, its chroma restored (src/core/jpeg.js); else the image
    base: null, // state.work with any translucent layer over the dots divided out (src/core/background.js)
    hash: null,
    // dot families: { estimate (families.js or a picked color's estimateStyle), picked (rgb or null), enabled (Find
    // dots fits it), overrides ({ radius, alpha, blur } set by the user) }
    families: [],
    current: 0, // the selected family's index
    estimate: null, // families[current].estimate
    dilate: null, // user's mask dilation for the selected family, px, or null for the default
    layers: null, // the selected family's buildMask result
    overlay: 'image',
    mode: null, // null, 'pick', 'pick-outline', 'add', 'remove' or 'move': what a click in the view does
    moving: null, // Move mode's chosen dot, { family, i } with i its index in that family's dots
    nudged: false, // the chosen dot has been nudged since it was chosen: its first nudge saved an undo step
    fit: null, // { families: [{ family, dots, radius, sigma }], dots (all, with family and r), ms, edited, partial }
    undo: [], // earlier fits, for Undo
    worker: null,
    scanning: false, // the scan worker is estimating families: Find dots and the edits wait for it
    annotations: null, // { image, byRadius }: text and legend markers in state.work by dot radius, masked (annotations.js)
    stroke: null, // { image, key, width, mask, … }: stroke.js pageStroke of the families as they are (strokeNow)
    fitStart: 0,
    tick: null, // the status timer while a fit runs
    blockedShown: false, // the status shows why Find dots can't run (findBlocked)
    showDots: true,
    showClose: true, // possible duplicates drawn in magenta with a shared ring (their 'close' flag stays either way)
    hoverDot: -1, // index into fit.dots of the dot Remove would take or Move has chosen, highlighted
    axes: null, // from detectAxes
    axisReadings: { x: null, y: null }, // the tick labels as read (core/labels.js readTickLabels)
    axisEntries: { x: null, y: null }, // tick values (read from the labels, or typed) and scale types
    showAxes: true,
};

const view = createView({
    stage: $('stage'),
    canvas: $('view'),
    onHover: showPixel,
    onClick: clickAt,
    onZoom: scale => {
        $('zoom-level').textContent = `${scale >= 1 ? Math.round(scale * 100) : (scale * 100).toFixed(1)}%`;
    },
});

const fmt = (v, d) => (v === null || v === undefined || Number.isNaN(v) ? '–' : v.toFixed(d));
const hex = c => '#' + c.map(v => Math.round(v).toString(16).padStart(2, '0')).join('');

/** A family's style in effect: its estimate with the user's overrides. */
function styleOf(k) {
    const f = state.families[k];
    const s = f?.estimate.style;
    return s ? { ...s, ...f.overrides } : null;
}

function currentStyle() {
    return styleOf(state.current);
}

/** One family's fit result, or null. */
function familyFit(k = state.current) {
    return state.fit?.families.find(f => f.family === k) ?? null;
}

/** Set the fit from per-family results; every dot carries its family and radius. */
function setFits(families, extra = {}) {
    for (const f of families)
        labelFills(f);
    const dots = families.flatMap(f => f.dots.map(d => ({ ...d, family: f.family, r: f.radius })));
    state.fit = { ...extra, families, dots };
}

// Outlined dots of a few fill colors are fitted as one family of rings; each dot's fill color comes from its pixels
// (extract.js dotFills): fill is its index in the estimate's fillFamilies, fillSure false where its fill is mostly
// hidden and the color may not be in the image at all. A palette's dots are fitted as one family too, and its colors
// are read from the fitted dots after each fit.
function labelFills(f) {
    const est = state.families[f.family]?.estimate;
    if (!est || !f.dots.length)
        return;
    const fills = dotFills(state.work, est, f.dots, f.radius);
    // (a palette's colors come from the fit: the panel shows the latest)
    if (est.palette)
        est.fillFamilies = fills ? fills.colors : null;
    if (!fills)
        return;
    f.dots.forEach((d, i) => {
        d.fill = fills.which[i].family;
        d.fillSure = fills.which[i].sure;
    });
}

// '2 #b8d7ff', with '?' where the fill is mostly hidden
function fillLabel(d) {
    const c = state.families[d.family].estimate.fillFamilies[d.fill];
    return `${d.fill + 1}${d.fillSure ? '' : '?'} ${hex(c)}`;
}

/** The style the current family's fitted dots were drawn with: the style in effect, with the fit's radius. */
function fittedStyle() {
    const style = currentStyle();
    const f = familyFit();
    return style && f ? { ...style, radius: f.radius } : style;
}

/** The image a family is analyzed on: its own image from families.js, else the base image. */
function imageOf(k) {
    return state.families[k]?.estimate.image ?? state.base ?? state.work;
}

function analysisImage() {
    return imageOf(state.current);
}

function maskOpts(k) {
    // (the clip edges ask every family, each at the radius in effect: read again whenever a mask is built)
    const clips = familyClips(state.axes, state.families.map((f, i) => ({ D: f.estimate.D, radius: styleOf(i)?.radius })), state.work.width, state.work.height);
    return { ...maskOptions(state.axes, annotations(styleOf(k)?.radius), clips[k], strokeNow().mask), ...(k === state.current && state.dilate !== null ? { dilate: state.dilate } : {}) };
}

// A stroke in the page's color around the opaque dots (stroke.js pageStroke), read again only when a family's
// estimate or its radius in effect has changed: it is read from all of them together.
function strokeNow() {
    const key = state.families.map((f, i) => [f.estimate, styleOf(i)?.radius]);
    const s = state.stroke;
    if (s?.image !== state.work || s.key.length !== key.length || key.some(([e, r], i) => s.key[i][0] !== e || s.key[i][1] !== r))
        state.stroke = { image: state.work, key, ...pageStroke(state.work, state.families.map((f, i) => ({ ...f.estimate, style: styleOf(i) }))) };
    return state.stroke;
}

// text and legend markers to mask (extract.js excludedBoxes), read once per analyzed image and radius
function annotations(radius) {
    if (!state.work || !radius)
        return [];
    if (state.annotations?.image !== state.work)
        state.annotations = { image: state.work, byRadius: new Map() };
    let boxes = state.annotations.byRadius.get(radius);
    if (!boxes) {
        boxes = excludedBoxes(state.work, radius);
        state.annotations.byRadius.set(radius, boxes);
    }
    return boxes;
}

/** What the fit needs for one family, with its overrides. */
function fitInputFor(k) {
    const est = state.families[k].estimate;
    const style = styleOf(k);
    const image = imageOf(k);
    const mask = k === state.current ? state.layers.mask : buildMask(image, { ...est, style }, maskOpts(k)).mask;
    return fitInput(image, est, mask, style);
}

/** The color that names a family: the picked color, or its seed; null for one whose dots vary in color. */
function colorOf(k) {
    const f = state.families[k];
    if (!f)
        return null;
    if (f.picked)
        return f.picked;
    // (hollow rings are one color, though fitted as outlined dots are)
    if (f.estimate.scale && !f.estimate.candidates[0]?.hollow)
        return null;
    return f.estimate.familyMeasure?.seed ?? f.estimate.style?.color ?? null;
}

function familyName(k) {
    const f = state.families[k];
    if (!f)
        return '';
    return f.picked ? `Picked ${f.picked.join(', ')}` : `Family ${k + 1}`;
}

// a family's kind in a few words: 'translucent, α 0.50, r 6.7'. The mask and the fit go by the α in effect (opaque
// from 1), so a family whose α the user has typed across that line is described as it is now fitted.
function familyKind(k) {
    const est = state.families[k].estimate;
    const st = styleOf(k);
    if (!st)
        return 'not a dot family';
    let kind;
    if (est.scale === 'outline') {
        const c = est.candidates[0];
        kind = c.hollow ? 'hollow rings' : `outlined, ${c.oneColor ? 'one fill color' : c.fillColors ? `${c.fillColors} fill colors` : 'fills vary'}`;
    }
    else if (est.palette) {
        const n = est.fillFamilies?.length;
        kind = `${n ? `${n} colors` : 'many colors'} (a palette), ${est.scale === 'opaque' ? 'opaque' : `α ${fmt(st.alpha, 2)}`}`;
    }
    else if (est.scale === 'opaque') {
        kind = 'colored by a scale, opaque';
    }
    else if (est.scale) {
        kind = `colored by a scale, α ${fmt(st.alpha, 2)}`;
    }
    else if (est.familyMeasure?.kind === 'opaque' && (st.alpha >= 1 || st.layered)) {
        kind = est.familyMeasure.stroke || est.familyMeasure.outline ? 'opaque, outlined' : 'opaque';
    }
    else if (st.alpha >= 1) {
        kind = 'opaque (α typed)';
    }
    else {
        kind = `translucent, α ${fmt(st.alpha, 2)}${est.familyMeasure?.kind === 'opaque' ? ' (typed)' : ''}`;
    }
    return `${kind}, r ${st.radius ? st.radius.toFixed(1) : '?'}`;
}

// The swatch for a family in the list: its color, its fill colors side by side, or a hatch when its dots vary
function paintSwatch(el, k) {
    const est = state.families[k].estimate;
    el.className = 'swatch';
    el.style.background = '';
    if (est.fillFamilies) {
        const n = est.fillFamilies.length;
        el.style.background = `linear-gradient(90deg, ${est.fillFamilies.map((c, i) => `${cssColor(c)} ${100 * i / n}% ${100 * (i + 1) / n}%`).join(', ')})`;
        el.title = `fill colors ${est.fillFamilies.map(hex).join(', ')}`;
        return;
    }
    const c = colorOf(k);
    if (c) {
        el.style.background = cssColor(c);
        el.title = `${c.map(Math.round).join(', ')}`;
    }
    else {
        el.classList.add('varies');
        el.title = 'colors vary by dot';
    }
}

/** An axis ('x' or 'y') with the ticks in effect: the detected ones, or the user's (axes-panel.js axisInEffect). */
function axisOf(k) {
    return state.axes ? axisInEffect(state.axes[k], state.axisEntries[k]) : null;
}

/** The scales in effect: { x, y }, each from axisScale or null. */
function scales() {
    return { x: axisScale(axisOf('x'), state.axisEntries.x), y: axisScale(axisOf('y'), state.axisEntries.y) };
}

/** Data coordinates of an image point, with per-axis resolution (data units per px). */
function dataAt(x, y) {
    const s = scales();
    return {
        x: toAxisData(s.x, x), y: toAxisData(s.y, y),
        resX: axisResolution(s.x, x), resY: axisResolution(s.y, y),
    };
}

// index of the nearest dot within `within` × its radius, among `dots` that pass `keep`
function nearestDot(dots, x, y, within = 1, keep = () => true) {
    let best = -1;
    let bestD = Infinity;
    dots.forEach((d, i) => {
        const dd = Math.hypot(d.x - x, d.y - y);
        if (keep(d) && dd <= within * d.r && dd < bestD) {
            bestD = dd;
            best = i;
        }
    });
    return best;
}

function showPixel(x, y, point) {
    if (x === null || !state.image) {
        $('pixel-line').textContent = '–';
        $('dot-line').textContent = '';
        setHoverDot(-1);
        view.setGuide(null);
        return;
    }
    // where a click would put the tick (clickAt)
    if (state.mode === 'tick-x')
        view.setGuide({ dir: 'v', pos: +point.x.toFixed(1) });
    else if (state.mode === 'tick-y')
        view.setGuide({ dir: 'h', pos: +point.y.toFixed(1) });
    const [r, g, b, a] = pixelAt(state.image, x, y);
    const p = y * state.image.width + x;
    const parts = [`${x}, ${y} px`, `RGBA ${r}, ${g}, ${b}, ${a}`];
    const D = state.estimate?.D;
    if (D)
        parts.push(`count ${Number.isNaN(D[p]) ? 'not the dot color' : D[p].toFixed(2)}`);
    const L = state.layers;
    if (L) {
        const cls = L.thin[p] ? 'thin mark in dot color' : (L.classes[p] === CLASS_BACKGROUND ? 'background' : (L.classes[p] === CLASS_FAMILY ? 'dot color' : 'other'));
        parts.push(`${cls}${L.mask[p] ? ', masked' : ''}`);
    }
    const pd = dataAt(x + 0.5, y + 0.5);
    if (pd.x !== null || pd.y !== null)
        parts.push(`data ${formatData(pd.x, pd.resX)}, ${formatData(pd.y, pd.resY)}`);
    const swatch = document.createElement('span');
    swatch.className = 'swatch';
    swatch.style.background = `rgb(${r} ${g} ${b})`;
    $('pixel-line').replaceChildren(swatch, parts.join(' · '));

    if ((state.mode === 'remove' || (state.mode === 'move' && !state.moving)) && state.fit)
        setHoverDot(nearestDot(state.fit.dots, x + 0.5, y + 0.5, REMOVE_HIT_RADII, d => d.family === state.current));
    const i = state.fit ? nearestDot(state.fit.dots, x + 0.5, y + 0.5) : -1;
    if (i < 0) {
        $('dot-line').textContent = '';
        return;
    }
    const d = state.fit.dots[i];
    const fam = state.fit.families.length > 1 ? `${familyName(d.family)}: ` : '';
    // the curvature standard errors reflect pixel noise only; systematic error (resampled rims, model mismatch) is
    // several times larger (PLAN.md phase 4: two renderings of dog-ages differ by 3–6× these), so say so
    const se = d.seX === undefined ? '' : ` ± ${d.seX.toFixed(2)}, ${d.seY.toFixed(2)} (noise only)`;
    const extra = d.visible === undefined ? '' : `; ${Math.round(100 * d.visible)}% visible, depth ${d.depth}${d.margin === undefined ? '' : `, margin ${d.margin.toFixed(0)}λ`}`;
    const dd = dataAt(d.x, d.y);
    const data = dd.x === null && dd.y === null ? '' : ` = data ${formatData(dd.x, dd.resX)}, ${formatData(dd.y, dd.resY)}`;
    const fill = d.fill === undefined ? '' : `; fill ${fillLabel(d)}${d.fillSure ? '' : ' (mostly hidden: color unsure)'}`;
    $('dot-line').textContent = `Dot ${fam}${d.x.toFixed(2)}, ${d.y.toFixed(2)} px${se}${data}${extra}${fill}${d.moved ? (d.placed ? ' (placed by hand)' : ' (moved by hand)') : d.placed ? ' (added by hand)' : ''}${d.flags?.length ? ` (${d.flags.join(', ')})` : ''}`;
}

function setHoverDot(i) {
    if (i === state.hoverDot)
        return;
    state.hoverDot = i;
    refreshMarks();
}

/** The message over the view, in a box so it reads over an image; busy adds a spinner for work in progress. */
function showMessage(text, { error = false, busy = false } = {}) {
    const el = $('stage-message');
    const box = document.createElement('span');
    box.className = 'box';
    box.textContent = text;
    el.replaceChildren(box);
    el.classList.toggle('error', error);
    el.classList.toggle('busy', busy);
    el.hidden = !text;
}

// Work on the main thread (decoding, the JPEG-noise and axis checks before a scan) blocks painting; let the browser
// paint the message first (a frame, then a task after it) so it shows during the wait.
function nextPaint() {
    return new Promise(resolve => requestAnimationFrame(() => setTimeout(resolve, 0)));
}

/** The message over the view with a progress bar: what is being done (title), its stage and the share done (0–1). */
function showProgress(title, stage, fraction) {
    const el = $('stage-message');
    // updated in place, so the status region announces a new stage rather than every report
    let box = el.querySelector('.box.progress');
    if (!box || box.dataset.title !== title) {
        box = document.createElement('span');
        box.className = 'box progress';
        box.dataset.title = title;
        const head = document.createElement('span');
        head.textContent = title;
        const bar = document.createElement('progress');
        bar.max = 1;
        const sub = document.createElement('span');
        sub.className = 'stage';
        box.append(head, bar, sub);
        el.replaceChildren(box);
    }
    box.querySelector('progress').value = fraction;
    const sub = box.querySelector('.stage');
    if (sub.textContent !== stage)
        sub.textContent = stage;
    el.classList.remove('error');
    el.classList.add('busy');
    el.hidden = false;
}

// What a worker's error event says, or, when it says nothing, what that means. A worker whose script can't be
// fetched fails with a bare event, no message: the fit's worker is started anew for every Find dots, so a page
// loaded from a server that has stopped since (a local one, restarted) scans and measures colors with the scan
// worker it has, and then failed with "Find dots failed: undefined".
function workerFailure(e) {
    return e.message || 'its worker did not start or stopped without a message. If this page is served from your own computer, check that the server is still running, then reload the page';
}

// The scan worker (src/scan-worker.js) runs one request at a time; a new request terminates the one running, whose
// promise then resolves null (a newer image or rescan overtook it). Terminating is the only way to stop a worker
// mid-computation, and a fresh one starts in milliseconds.
let scanWorker = null;
let scanPending = null; // resolve of the running request

/**
 * Run op in the scan worker, showing its progress under `title`. Resolves to its result, or null if overtaken.
 */
function runScan(op, args, title) {
    if (scanPending) {
        scanWorker.terminate();
        scanWorker = null;
        scanPending(null);
        scanPending = null;
    }
    scanWorker ??= new Worker(new URL('../scan-worker.js', import.meta.url), { type: 'module' });
    const worker = scanWorker;
    state.scanning = true;
    renderFitControls();
    showProgress(title, '', 0);
    return new Promise((resolve, reject) => {
        const done = () => {
            scanPending = null;
            state.scanning = false;
            renderFitControls();
        };
        scanPending = resolve;
        worker.onmessage = e => {
            const m = e.data;
            if (m.type === 'progress') {
                showProgress(title, m.label, m.fraction);
                return;
            }
            done();
            if (m.type === 'error')
                reject(new Error(m.message));
            else
                resolve(m.result);
        };
        worker.onerror = e => {
            done();
            reject(new Error(workerFailure(e)));
        };
        worker.postMessage({ op, ...args });
    });
}

function setStatus(text, kind = '', title = text) {
    const el = $('fit-status');
    el.textContent = text;
    el.title = title;
    el.className = `status${kind ? ' ' + kind : ''}`;
    state.blockedShown = false;
}

// Why Find dots can't run, or null when it can: a short line for the status beside it and the whole reason for
// its tooltip (kite2: one family, no isolated dot to measure its radius from, and only a tooltip said so).
function findBlocked() {
    if (!state.image)
        return null;
    const families = state.families.map((f, k) => ({ f, k })).filter(({ k }) => styleOf(k));
    if (!families.length)
        return { short: 'No dot family found: add one by color', full: 'No dot family was found in the image. Add one by color (+ Add family by color) and click a dot.' };
    const checked = families.filter(({ f }) => f.enabled);
    if (!checked.length)
        return { short: 'Check a dot family to find its dots', full: 'Find dots fits the checked families; none is checked.' };
    const missing = checked.filter(({ k }) => !(styleOf(k).radius > 0));
    if (missing.length < checked.length)
        return null;
    const names = missing.map(({ k }) => familyName(k)).join(', ');
    return {
        short: `${names}: set the radius`,
        full: `${names}: no isolated whole dot to measure the radius from. Type it in the family's Radius box (px) to find the dots; Find dots refits it.`,
    };
}

/**
 * The dots rendered in RGB with the style. An outlined opaque family's dots are an outline disc with the fill drawn
 * over it: exact for one dot, but here every outline is drawn before every fill (the chart's order isn't known), so
 * where dots overlap the picture differs from the chart's.
 */
function renderModel(bg, dots, style) {
    // fitted by its rings, an outlined family knows each dot's drawing order (z): outline, then fill, dot by dot
    if (style.layered && style.outline) {
        // sharp, dot by dot, then blurred once: blurring each pass would blur the dots under it again and again
        let out = bg;
        const opaque = { ...style, alpha: 1, blur: 0, levels: undefined, ring: undefined, layered: undefined };
        for (const d of [...dots].sort((a, b) => (a.z ?? 0) - (b.z ?? 0))) {
            out = renderDots(out, [d], { ...opaque, color: style.outline.color });
            out = renderDots(out, [d], { ...opaque, radius: style.outline.inner * style.radius });
        }
        if (!(style.blur > 0))
            return out;
        const blurred = { width: out.width, height: out.height, data: new Float32Array(out.data.length) };
        blurRect(out, blurred, gaussianKernel(style.blur), [0, 0, out.width, out.height]);
        return blurred;
    }
    if (!style.outline)
        return renderDots(bg, dots, style);
    // the level table would paint over the background image, so both passes composite over it instead
    const rings = renderDots(bg, dots, { ...style, color: style.outline.color, levels: undefined });
    return renderDots(rings, dots, { ...style, radius: style.outline.inner * style.radius, levels: undefined });
}

// an outlined family fitted by its rings, in the fit's units: its ordered ring model over the base (outline.js)
function layeredT(base, dots, style, image) {
    const model = createOutlineModel(base, image.width, image.height, { radius: style.radius, ring: style.ring, fillAlpha: style.layered.fillAlpha, ringT: style.layered.ringT, blur: style.blur });
    for (const d of dots)
        model.add(d.x, d.y, d.z);
    return model.image.data;
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
        const modelT = style.layered && estimate.base ? layeredT(estimate.base, fam.dots, style, image) : renderDots(bg, fam.dots, { ...style, color: style.fitColor ?? style.color, project: line }).data;
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
    const selected = dots === state.fit?.dots && state.hoverDot >= 0 ? state.hoverDot : undefined;
    view.setMarks(state.showDots && dots && style ? { dots, radius: style.radius, selected, family: state.current, showClose: state.showClose } : null);
}

/**
 * Estimate every family (in the scan worker) and select the first. Fitted dots are dropped: they were fitted to other
 * estimates. Resolves false if a newer image or rescan overtook it.
 */
async function analyze({ auto = false } = {}) {
    stopFit(false);
    const t0 = performance.now();
    // until the families come, there are none to fit or edit
    state.families = [];
    state.fit = null;
    state.undo = [];
    state.scanning = true;
    selectFamily(0);
    // An image with JPEG noise that isn't a JPEG file is also scanned with noisy color matching, and analyzed as a
    // JPEG when that finds more families (extract.js scanFamilies says why). Only on loading: the box is the user's
    // after.
    const tryNoisy = auto && !state.jpeg && state.noise.likely;
    const scan = await runScan('scan', { image: state.image, jpeg: state.jpeg, tryNoisy, bounds: plotBounds(state.axes) }, `Scanning ${state.name} for dot colors and sizes…`);
    if (!scan)
        return false;
    const { prepared } = scan;
    if (scan.switched) {
        setJpeg(true, true);
        const note = scan.exact
            ? `Noisy color matching found more dot families than exact colors (${scan.found} against ${scan.exact}), and the image has JPEG-like noise (${noiseText()}), so it was turned on.`
            : `No dot family with a radius showed with exact colors, and the image has JPEG-like noise (${noiseText()}), so noisy color matching was turned on.`;
        for (const e of prepared.estimates)
            e.notes = [note, ...e.notes];
    }
    state.base = prepared.image;
    state.families = prepared.estimates.map(estimate => ({ estimate, picked: null, enabled: !!estimate.style, overrides: {} }));
    const t1 = performance.now();
    state.fit = null;
    state.undo = [];
    state.dilate = null;
    selectFamily(0);
    $('analysis-time').textContent = `families ${Math.round(t1 - t0)} ms`;
    setStatus(auto && state.jpegAuto ? 'No dot family with a radius from exact colors: noisy color matching turned on (JPEG-like noise).' : '');
    // what Find dots needs, if anything, outranks the note (the family notes keep it)
    renderFitControls();
    return true;
}

/** Show family k in the panel and the overlays; the fit is kept. */
function selectFamily(k) {
    state.current = k;
    state.estimate = state.families[k]?.estimate ?? null;
    state.hoverDot = -1;
    buildLayers();
    renderFamilies();
    renderStylePanel();
    renderFitControls();
    renderDotsSection();
    updateModeHint();
    if (state.image)
        refreshView(true);
}

// A changed style or mask makes this family's fitted dots stale; other families keep theirs.
function dropFamilyFit(k) {
    if (!familyFit(k))
        return;
    const families = state.fit.families.filter(f => f.family !== k);
    if (families.length)
        setFits(families, extraOf(state.fit));
    else
        state.fit = null;
}

function extraOf(fit) {
    const { families, dots, ...extra } = fit;
    return extra;
}

function buildLayers() {
    const style = currentStyle();
    state.layers = style ? buildMask(analysisImage(), { ...state.estimate, style }, maskOpts(state.current)) : null;
    renderLayersPanel();
}

function renderFamilies() {
    const list = $('family-list');
    list.replaceChildren();
    if (!state.families.length || !state.families.some(f => f.estimate.style)) {
        const li = document.createElement('li');
        li.className = 'off';
        li.textContent = !state.image ? 'No image.'
            : state.scanning ? 'Scanning…'
            : (state.estimate?.notes?.[0] ?? 'No dot family found.') + ' Add one by its color.';
        list.append(li);
        return;
    }
    state.families.forEach((f, k) => {
        const li = document.createElement('li');
        li.classList.toggle('selected', k === state.current);
        li.classList.toggle('off', !f.enabled);
        const box = document.createElement('input');
        box.type = 'checkbox';
        box.checked = f.enabled;
        box.disabled = !styleOf(k)?.radius;
        box.title = box.disabled ? 'The radius is unknown: set it to find this family' : 'Find dots fits this family';
        box.setAttribute('aria-label', `Find ${familyName(k)}`);
        box.addEventListener('change', () => {
            f.enabled = box.checked;
            li.classList.toggle('off', !f.enabled);
            renderFitControls();
        });
        const sw = document.createElement('span');
        paintSwatch(sw, k);
        const name = document.createElement('span');
        name.className = 'name';
        name.textContent = `${familyName(k)}: ${familyKind(k)}`;
        const count = document.createElement('span');
        count.className = 'count';
        const ff = familyFit(k);
        if (ff) {
            const flagged = ff.dots.filter(d => d.flags.length).length;
            count.textContent = `${ff.dots.length} dots${flagged ? `, ${flagged} flagged` : ''}`;
        }
        li.append(box, sw, name, count);
        if (f.picked) {
            const rm = document.createElement('button');
            rm.type = 'button';
            rm.className = 'remove';
            rm.textContent = '×';
            rm.title = 'Remove this picked family';
            rm.setAttribute('aria-label', `Remove ${familyName(k)}`);
            rm.addEventListener('click', e => {
                e.stopPropagation();
                removeFamily(k);
            });
            li.append(rm);
        }
        li.addEventListener('click', e => {
            if (e.target === box || k === state.current)
                return;
            selectFamily(k);
        });
        list.append(li);
    });
}

function removeFamily(k) {
    stopFit(false);
    dropFamilyFit(k);
    state.families.splice(k, 1);
    if (state.fit) {
        const families = state.fit.families.map(f => (f.family > k ? { ...f, family: f.family - 1 } : f));
        setFits(families, extraOf(state.fit));
    }
    state.undo = [];
    selectFamily(Math.min(state.current >= k ? Math.max(0, state.current - 1) : state.current, state.families.length - 1));
}

function renderStylePanel() {
    const est = state.estimate;
    const style = currentStyle();
    $('family-detail').hidden = !est;
    if (!est)
        return;
    $('family-title').textContent = `${familyName(state.current)}: ${familyKind(state.current)}`;
    const swatches = $('levels');
    swatches.replaceChildren();
    // a color scale is fitted as a virtual family whose color and levels no dot has: don't show them as the dots'
    const showLevels = style && !est.scale;
    $('levels').parentElement.hidden = !showLevels;
    if (showLevels) {
        const K = style.measuredLevels;
        const inferred = style.inferredLevels ?? [];
        style.levels.slice(1, K + 1 + EXTRAPOLATED_SWATCHES).forEach((c, i) => {
            const cell = document.createElement('span');
            cell.className = 'level';
            const el = document.createElement('span');
            const guessed = i >= K || inferred.includes(i + 1);
            el.className = 'swatch' + (guessed ? ' extrapolated' : '');
            el.style.background = cssColor(c);
            cell.title = `${i + 1} dot${i ? 's' : ''} stacked: ${c.map(v => v.toFixed(1)).join(', ')}${i >= K ? ' (extrapolated)' : guessed ? ' (inferred: too few flat pixels show it)' : ' (measured)'}`;
            cell.append(el, String(i + 1));
            swatches.append(cell);
        });
    }
    const c0 = est.candidates[0];
    const scaleText = est.palette ? `${est.fillFamilies ? `${est.fillFamilies.length} colors` : 'many colors'}, a palette (each dot's read from its own pixels after the fit)`
        : est.scale !== 'outline' ? 'varies by dot (color scale)'
        : c0?.hollow ? `${style?.color.join(', ')} at the rings' darkest, nothing inside (rings fitted)`
        : c0?.oneColor ? 'one fill color, projected out (rings fitted)'
        : c0?.fillColors ? `${c0.fillColors} fill colors, projected out (each dot's read from its fill)` : 'varies by dot (color scale; rings fitted)';
    $('style-c').textContent = !style ? '–' : est.scale ? scaleText : style.color.map(v => v.toFixed(1)).join(', ');
    $('style-c').title = style && est.scale ? `fitted as one virtual family of color ${style.color.map(v => v.toFixed(1)).join(', ')}` : '';
    const nInferred = style?.inferredLevels?.length ?? 0;
    $('style-levels').textContent = style ? `${style.measuredLevels - nInferred} measured${nInferred ? `, ${nInferred} inferred` : ''}` : '–';
    const fam = est.familyMeasure;
    if (fam?.kind === 'opaque' && style?.layered)
        $('style-fit').textContent = 'opaque, fitted by its outlines in drawing order';
    else if (fam?.kind === 'opaque' && style?.alpha < 1)
        $('style-fit').textContent = `measured as opaque; fitted as translucent with the α typed, ${fmt(style.alpha, 2)}`;
    else if (fam?.kind === 'opaque')
        $('style-fit').textContent = 'opaque: overlaps aren’t seen, α = 1';
    else if (style?.alpha >= 1 && 'alpha' in state.families[state.current].overrides)
        $('style-fit').textContent = 'measured as translucent; fitted as opaque with the α typed, 1';
    else if (fam?.fit?.underdetermined)
        $('style-fit').textContent = 'one level: α not measured, 1 assumed';
    else
        $('style-fit').textContent = fam?.fit ? `rms ${fmt(fam.fit.rms, 2)}, max ${fmt(fam.fit.maxAbs, 2)}` : '–';
    $('style-isolated').textContent = String(est.isolated?.length ?? 0);
    // overlap levels and one α describe single-color families; a color scale's virtual family and outlined dots'
    // ring model have values of their own that no dot has
    for (const el of document.querySelectorAll('[data-levels]'))
        el.hidden = !!est.scale;
    const overrides = state.families[state.current].overrides;
    for (const [key, digits] of [['radius', 2], ['alpha', 3], ['blur', 2]]) {
        const input = $(`style-${key}`);
        // outlined dots fitted by their rings: the ring's and fill's opacities, not one α (an opaque family's rings
        // have α 0 in the fit's style, which isn't its fill's)
        const notOwn = key === 'alpha' && (est.scale === 'outline' || !!style?.layered);
        input.value = style && !notOwn && style[key] !== null && style[key] !== undefined ? style[key].toFixed(digits) : '';
        input.classList.toggle('overridden', key in overrides);
        input.disabled = !style || notOwn;
        // a radius Find dots needs and no isolated dot gave
        const needed = key === 'radius' && style && !(style.radius > 0);
        input.classList.toggle('needed', !!needed);
        input.placeholder = needed ? 'needed' : '';
        input.title = notOwn ? 'Outlined dots: the ring and fill opacities are measured separately (see the notes)'
            : needed ? 'No isolated whole dot to measure the radius from: type it (px) to find the dots; Find dots refits it.' : '';
    }
    const fam0 = state.families[state.current];
    $('style-reset').disabled = !Object.keys(overrides).length && fam0.outlineEdit === undefined;
    const editable = outlineEditable(state.current);
    $('outline-params').hidden = !editable;
    if (editable) {
        const o = outlineOf(state.current);
        $('outline-on').checked = !!o;
        $('outline-color').value = hex(o?.color ?? OUTLINE_DEFAULT.color);
        $('outline-width').value = (o?.width ?? OUTLINE_DEFAULT.width).toFixed(2);
        $('outline-color').disabled = !o || state.worker !== null;
        $('outline-width').disabled = !o || state.worker !== null;
        for (const id of ['outline-color', 'outline-width'])
            $(id).classList.toggle('overridden', fam0.outlineEdit !== undefined);
    }
    const stroke = style?.alpha >= 1 && !est.outlineMask ? strokeNow() : null;
    const strokeNote = stroke?.mask ? [`The dots have a stroke in the page’s color, about ${stroke.width.toFixed(1)} px wide (read where a whole dot lies over another). `
        + 'It cuts the dots under it; the fit ignores its pixels there (the Residual view shows them as masked).'] : [];
    $('style-notes').replaceChildren(...[...est.notes, ...strokeNote].map(n => Object.assign(document.createElement('p'), { textContent: n })));
    renderSample();
}

// The first isolated dot, beside the model's rendering of it at the measured center, and their difference.
function renderSample() {
    const est = state.estimate;
    const style = currentStyle();
    const box = $('sample');
    box.hidden = !(style?.radius && est.isolated?.length);
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
    $('layer-stats').textContent = `${familyName(state.current)}: background ${pct(s.background)}, dot color ${pct(s.family)}, other ${pct(s.other)}, `
        + `thin marks ${s.thin} px. Masked ${pct(s.masked)}, including ${s.maskedFamily} dot-color px near other marks`
        + (s.outside ? ` and ${pct(s.outside)} beyond the axes (labels, titles).` : '.');
}

function renderLegend() {
    const legend = $('overlay-legend');
    const item = (color, text) => `<span class="key" style="background:${typeof color === 'string' ? color : cssColor(color)}"></span>${text}`;
    const close = state.showClose ? `${item(MARK_COLORS.close, 'double ring: two dots of one family almost in one place (if opaque, maybe one dot counted twice)')} ` : '';
    $('dots-legend').innerHTML = `${item(MARK_COLORS.ok, 'dot')} ${item(MARK_COLORS.flagged, 'flagged (hover for why)')} ${close}${item(MARK_COLORS.selected, 'Remove would take it, or Move has chosen it')}`;
    const fam = state.estimate ? ` (${familyName(state.current)})` : '';
    if (state.overlay === 'residual') {
        legend.innerHTML = `Residual${fam}, image − model along the dot color line: ${item(RESIDUAL_DARKER, 'image darker (missing ink)')} `
            + `${item(RESIDUAL_LIGHTER, 'image lighter (too much)')} ${item(RESIDUAL_MASKED, 'masked')}; full color at ±${RESIDUAL_FULL} units.`;
    }
    else if (state.overlay === 'model') {
        legend.textContent = `Model${fam}: the fitted dots rendered with the style (radius from the fit).`
            + (currentStyle()?.layered && currentStyle()?.outline ? ' Each dot is drawn in its fitted order, outline then fill.'
                : currentStyle()?.outline ? ' Outlines are all drawn before the fills, so overlaps can differ from the chart’s drawing order.' : '');
    }
    else if (state.overlay === 'count') {
        legend.innerHTML = `Overlap count${fam}: black 0 to white ${COUNT_MAX_SHOWN}+ (hover for the value); ${item(OFF_LINE_COLOR, 'not the dot color')}`;
    }
    else if (state.overlay === 'mask') {
        legend.innerHTML = `Mask${fam}: ${item(MASK_TINT, 'masked')} pixels are ignored by the fit`;
    }
    else if (state.overlay === 'classes') {
        legend.innerHTML = `Classes${fam}: ` + [item(CLASS_COLORS.background, 'background'), item(CLASS_COLORS.family, 'dot color'), item(CLASS_COLORS.other, 'other'), item(CLASS_COLORS.thin, 'thin mark in dot color')].join(' ');
    }
    else {
        legend.textContent = '';
    }
    if (state.overlay !== 'image' && (state.overlay === 'model' || state.overlay === 'residual') && !familyFit())
        legend.textContent = `No fitted dots for ${familyName(state.current)} yet: showing the image.`;
}

function clickAt(x, y, { alt = false, point = null } = {}) {
    if (!state.image)
        return;
    if (state.mode === 'pick') {
        addPickedFamily(pixelAt(state.work, x, y).slice(0, 3));
        return;
    }
    if (state.mode === 'tick-x' || state.mode === 'tick-y') {
        // where clicked, not the pixel's center: a tick or gridline an odd number of px wide is centered between two
        const k = state.mode === 'tick-x' ? 'x' : 'y';
        setAxisEntry(k, addTick(state.axes[k], state.axisEntries[k], +(point ?? { x: x + 0.5, y: y + 0.5 })[k].toFixed(1)));
        return;
    }
    if (state.mode === 'pick-outline') {
        const k = state.current;
        state.families[k].outlineEdit = { color: pixelAt(state.work, x, y).slice(0, 3), width: outlineOf(k)?.width ?? OUTLINE_DEFAULT.width };
        setMode(null);
        outlineChanged(k);
        return;
    }
    // Option-click: no refit, and an added dot goes exactly where clicked rather than at the pixel's center
    if (state.mode === 'add' || state.mode === 'remove')
        editAt(state.mode, alt && point ? point.x : x + 0.5, alt && point ? point.y : y + 0.5, alt);
    if (state.mode === 'move')
        chooseMoving(x + 0.5, y + 0.5);
}

/** A family of a picked color, fitted on its own (on the image with any overlay divided out). */
async function addPickedFamily(rgb) {
    setMode(null);
    let estimate;
    try {
        estimate = await runScan('seed', { image: state.base ?? state.work, seed: rgb, jpeg: state.jpeg }, `Measuring dots of color ${rgb.join(', ')}…`);
    }
    catch (err) {
        showMessage('');
        setStatus(`Measuring color ${rgb.join(', ')} failed: ${err.message}`, 'error');
        return;
    }
    showMessage('');
    if (!estimate)
        return;
    if (!estimate.style) {
        setStatus(`No dot family of color ${rgb.join(', ')}: ${estimate.notes.join(' ')}`, 'error');
        return;
    }
    state.families.push({ estimate, picked: rgb, enabled: true, overrides: {} });
    const k = state.families.length - 1;
    // with outlined families about, every opaque family needs the new fill to give its rings to it
    if (state.families.some((f, j) => outlineEditable(j) && outlineOf(j)) && !await remeasureOpaque())
        return;
    selectFamily(k);
    setStatus(`Added a family of color ${rgb.join(', ')}.`);
}

// An opaque family of its own color (not a color scale or the ring path): its outline can be edited
function outlineEditable(k) {
    const est = state.families[k]?.estimate;
    return !!est?.style && !est.scale && est.familyMeasure?.kind === 'opaque';
}

// a family's outline in effect, { color, width } (0–255, px), or null
function outlineOf(k) {
    const f = state.families[k];
    if (f.outlineEdit !== undefined)
        return f.outlineEdit;
    const fm = f.estimate.familyMeasure;
    const color = fm.stroke ?? fm.outline;
    const radius = f.estimate.style.radius;
    return color && fm.fillRadius && radius ? { color, width: radius - fm.fillRadius } : null;
}

// Every opaque family measured again (style.opaqueFamilyEstimate): each with its outline in effect, the others'
// fills known and, when any is outlined, each ring given to the fill it surrounds. With nothing edited and no picked
// family, the estimates go back to what the analysis found.
async function remeasureOpaque() {
    const opaque = state.families.map((_, j) => j).filter(outlineEditable);
    const seedOf = j => state.families[j].picked ?? state.families[j].estimate.familyMeasure.seed;
    const custom = opaque.some(j => state.families[j].outlineEdit !== undefined || state.families[j].picked);
    const outlines = opaque.map(outlineOf);
    const anyOutline = outlines.some(Boolean);
    const image = state.base ?? state.work;
    const jobs = [];
    opaque.forEach((j, i) => {
        const f = state.families[j];
        f.auto ??= f.estimate;
        if (!custom) {
            f.estimate = f.auto;
            return;
        }
        const o = outlines[i];
        const fm = f.estimate.familyMeasure;
        // the fill inside the outline: the outer radius (the user's, the outlined estimate's, or the plain fill's
        // radius plus the width) less the width
        const outer = f.overrides.radius ?? (fm.stroke || fm.outline ? f.estimate.style.radius : (f.estimate.style.radius ?? 0) + (o?.width ?? 0));
        const stroke = o && (f.outlineEdit !== undefined || fm.stroke) ? { color: o.color, fillRadius: Math.max(1, outer - o.width) } : null;
        jobs.push({ j, seed: seedOf(j), opts: {
            stroke, outline: !stroke && f.outlineEdit === undefined ? fm.outline : null,
            others: opaque.filter(q => q !== j).map(seedOf), ringsToOthers: anyOutline, jpeg: state.jpeg,
        } });
    });
    if (jobs.length) {
        const families = state.families;
        let estimates;
        try {
            estimates = await runScan('opaque', { image, jobs: jobs.map(({ seed, opts }) => ({ seed, opts })) }, 'Measuring the dot families again…');
        }
        catch (err) {
            showMessage('');
            setStatus(`Measuring the families again failed: ${err.message}`, 'error');
            return false;
        }
        showMessage('');
        // (overtaken, or a new image came meanwhile: these estimates are of nothing shown)
        if (!estimates || families !== state.families)
            return false;
        jobs.forEach(({ j }, i) => {
            state.families[j].estimate = estimates[i];
        });
    }
    state.estimate = state.families[state.current]?.estimate ?? null;
    return true;
}

// family k's outline changed: measure the families again; k's dots are stale (Undo brings them back), the others'
// keep their places
async function outlineChanged(k) {
    if (familyFit(k))
        pushUndo();
    dropFamilyFit(k);
    if (!await remeasureOpaque())
        return;
    const o = outlineOf(k);
    setStatus(o ? `${familyName(k)}: outline ${o.color.map(Math.round).join(', ')}, ${o.width.toFixed(2)} px; families measured again.` : `${familyName(k)}: no outline; families measured again.`);
    styleChanged();
}

/** What a click in the view does: null (pan only), 'pick', 'pick-outline', 'add', 'remove', 'move', 'tick-x' or 'tick-y'. */
function setMode(mode) {
    const ticking = (state.mode?.startsWith('tick') ?? false) !== (mode?.startsWith('tick') ?? false);
    state.mode = mode;
    state.moving = null;
    state.nudged = false;
    $('pick-color').setAttribute('aria-pressed', String(mode === 'pick'));
    $('outline-pick').setAttribute('aria-pressed', String(mode === 'pick-outline'));
    $('edit-add').setAttribute('aria-pressed', String(mode === 'add'));
    $('edit-remove').setAttribute('aria-pressed', String(mode === 'remove'));
    $('edit-move').setAttribute('aria-pressed', String(mode === 'move'));
    const stage = $('stage');
    stage.classList.toggle('picking', mode === 'pick' || mode === 'pick-outline');
    stage.classList.toggle('adding', mode === 'add' || mode === 'tick-x' || mode === 'tick-y');
    stage.classList.toggle('removing', mode === 'remove' || mode === 'move');
    setHoverDot(-1);
    view.setGuide(null);
    updateModeHint();
    // the Add tick buttons show which axis a click adds to
    if (ticking && state.image)
        renderAxesPanel();
}

function updateModeHint() {
    const hint = $('mode-hint');
    const name = familyName(state.current);
    const text = state.mode === 'pick' ? 'Click a dot to add a family of its color · Esc to cancel'
        : state.mode === 'pick-outline' ? `Click the darkest part of an outline of ${name} · Esc to cancel`
        : state.mode === 'add' && state.moving ? `Adding to ${name}: click to add a dot (${OPTION_KEY}-click: exactly there, nothing refitted) · arrow keys move the last one ${NUDGE_STEP} px (Shift: ${NUDGE_FINE} px) · Esc to finish`
        : state.mode === 'add' ? `Adding to ${name}: click to add a dot (${OPTION_KEY}-click: exactly there, nothing refitted) · drag to pan · Esc to finish`
        : state.mode === 'remove' ? `Removing from ${name}: click a dot to remove it (${OPTION_KEY}-click: nothing refitted) · drag to pan · Esc to finish`
        : state.mode === 'move' && !state.moving ? `Moving in ${name}: click a dot to choose it · drag to pan · Esc to finish`
        : state.mode === 'move' ? `Arrow keys move the dot ${NUDGE_STEP} px (Shift: ${NUDGE_FINE} px) · click another dot · Esc to finish`
        : state.mode === 'tick-x' ? 'Adding x ticks: click where a tick is (the line shows its x; zoom in to place it well) · drag to pan · Esc to finish'
        : state.mode === 'tick-y' ? 'Adding y ticks: click where a tick is (the line shows its y; zoom in to place it well) · drag to pan · Esc to finish'
        : '';
    hint.textContent = text;
    hint.hidden = !text;
}

function renderFitControls() {
    const running = state.worker !== null;
    const runnable = !state.scanning && state.families.some((f, k) => f.enabled && styleOf(k)?.radius > 0);
    $('fit-run').disabled = running || !runnable;
    $('fit-run').textContent = running ? 'Finding…' : 'Find dots';
    // the reason replaces the status while it holds, and goes when it no longer does
    const blocked = running || state.scanning ? null : findBlocked();
    $('fit-run').title = blocked?.full ?? 'Find the dots of every checked family (families without a radius are skipped)';
    if (blocked) {
        setStatus(blocked.short, 'needed', blocked.full);
        state.blockedShown = true;
    }
    else if (state.blockedShown) {
        setStatus('');
    }
    $('fit-stop').disabled = !running;
    $('fit-progress').hidden = !running;
    const editable = !running && !state.scanning && !!currentStyle()?.radius;
    $('edit-add').disabled = !editable;
    $('edit-remove').disabled = !editable || !familyFit()?.dots.length;
    $('edit-move').disabled = !editable || !familyFit()?.dots.length;
    $('edit-clear').disabled = running || !state.fit?.dots.length;
    $('edit-undo').disabled = running || !state.undo.length;
    $('copy-table').disabled = !state.fit?.dots.length || running;
    $('save-csv').disabled = !state.fit?.dots.length || running;
    $('save-truth').disabled = !state.fit?.dots.length || running;
    $('pick-color').disabled = running || state.scanning || !state.image;
    // a fit in progress was set up with the families as they are; a scan is measuring them
    for (const id of ['outline-on', 'outline-pick'])
        $(id).disabled = running || state.scanning;
    if (!editable && (state.mode === 'add' || state.mode === 'remove'))
        setMode(null);
    if ((state.mode === 'remove' && $('edit-remove').disabled) || (state.mode === 'move' && $('edit-move').disabled))
        setMode(null);
}

function renderDotsSection() {
    const fit = state.fit;
    if (!fit) {
        for (const id of ['fit-count', 'fit-flags', 'fit-radius', 'fit-sigma'])
            $(id).textContent = '–';
        $('fit-note').textContent = '';
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
    $('fit-sigma').textContent = cur ? fmt(cur.sigma, 2) + (many ? ` (${familyName(state.current)})` : '') : '–';
    $('fit-note').textContent = `${fit.partial ? 'Stopped early: partial result, not assessed. ' : ''}`
        + `${fit.edited ? 'Edited by hand. ' : ''}Positions in image px (pixel i spans i to i + 1).`;
}

// a family's progress within its own share of the bar, from the stage the fit reports
function stageProgress(p) {
    if (p.stage === 'starting' || p.stage === 'candidates')
        return p.stage === 'starting' ? 0 : PROGRESS_CANDIDATES;
    if (p.stage === 'radius')
        return PROGRESS_RADIUS;
    return PROGRESS_CANDIDATES + (PROGRESS_ROUNDS_END - PROGRESS_CANDIDATES) * Math.min(1, p.round / MAX_ROUNDS);
}

// Fit every checked family, one after another in the worker.
function runFit() {
    stopFit(false);
    setMode(null);
    state.fitStart = performance.now();
    const inputs = [];
    state.families.forEach((f, k) => {
        if (f.enabled && styleOf(k)?.radius > 0)
            inputs.push({ family: k, input: fitInputFor(k) });
    });
    if (!inputs.length)
        return;
    if (state.fit)
        pushUndo();
    const done = [];
    let last = null;
    state.fit = null;
    state.worker = new Worker(new URL('../fit-worker.js', import.meta.url), { type: 'module' });
    const radiusOf = k => inputs.find(i => i.family === k).input.style.radius;
    // the status names the latest stage and ticks the elapsed time between messages (a first stage can take
    // seconds); the bar is indeterminate until the fit reports
    let shown = { family: inputs[0].family, p: { stage: 'starting', round: 0, dots: [] } };
    const showProgress = () => {
        const { family, p } = shown;
        const i = inputs.findIndex(q => q.family === family);
        if (p.stage === 'starting')
            $('fit-progress').removeAttribute('value');
        else
            $('fit-progress').value = (i + stageProgress(p)) / inputs.length;
        const secs = ((performance.now() - state.fitStart) / 1000).toFixed(1);
        const which = inputs.length > 1 ? `${familyName(family)} (${i + 1} of ${inputs.length}): ` : '';
        const stage = p.stage === 'starting' ? 'finding candidates' : p.stage === 'candidates' ? `${p.dots.length} candidates`
            : p.stage === 'radius' ? `radius refit, ${p.dots.length} dots` : `round ${p.round}, ${p.dots.length} dots`;
        setStatus(`${which}${stage}, ${secs} s`, 'running');
    };
    const progress = (family, p) => {
        shown = { family, p };
        showProgress();
    };
    state.tick = setInterval(showProgress, STATUS_TICK);
    state.worker.onmessage = e => {
        const msg = e.data;
        if (msg.type === 'progress') {
            last = msg;
            const r = radiusOf(msg.family);
            refreshMarks([...done.flatMap(f => f.dots.map(d => ({ ...d, r: f.radius }))), ...msg.progress.dots.map(d => ({ ...d, r }))]);
            progress(msg.family, msg.progress);
            return;
        }
        if (msg.type === 'family') {
            done.push({ ...msg.result, family: msg.family });
            last = null;
            return;
        }
        state.worker.terminate();
        state.worker = null;
        clearInterval(state.tick);
        if (msg.type === 'error') {
            setStatus(`Find dots failed: ${msg.message}`, 'error');
            renderFitControls();
            return;
        }
        const ms = performance.now() - state.fitStart;
        setFits(done, { ms });
        const n = state.fit.dots.length;
        const flagged = state.fit.dots.filter(d => d.flags.length).length;
        setStatus(`Found ${n} dots${flagged ? `, ${flagged} flagged` : ''} in ${(ms / 1000).toFixed(1)} s`, 'done');
        fitChanged();
    };
    state.worker.onerror = e => {
        clearInterval(state.tick);
        setStatus(`Find dots failed: ${workerFailure(e)}`, 'error');
        state.worker = null;
        renderFitControls();
    };
    // a stopped fit keeps the finished families and what the running one had reached
    state.stopFit = () => {
        const partial = [...done];
        if (last)
            partial.push({ family: last.family, dots: last.progress.dots.map(d => ({ ...d, flags: [] })), radius: radiusOf(last.family), sigma: NaN });
        return partial;
    };
    state.worker.postMessage({ inputs, opts: {} });
    renderFitControls();
    showProgress();
}

/** Stop a running fit, keeping what it has; `report` says so in the status. */
function stopFit(report = true) {
    if (!state.worker)
        return;
    state.worker.terminate();
    state.worker = null;
    clearInterval(state.tick);
    const partial = state.stopFit?.() ?? [];
    if (partial.length)
        setFits(partial, { partial: true });
    if (report)
        setStatus(`Stopped: ${state.fit?.dots.length ?? 0} dots kept (not assessed).`);
    fitChanged();
}

// everything that shows the fit
function fitChanged() {
    renderFitControls();
    renderDotsSection();
    renderFamilies();
    // a palette's colors are read from its fitted dots (labelFills), which the style panel names
    if (state.families[state.current]?.estimate.palette)
        renderStylePanel();
    renderLegend();
    if (state.image)
        refreshView(true);
}

function pushUndo() {
    const fit = state.fit;
    state.undo.push(fit ? { families: fit.families.map(f => ({ ...f, dots: f.dots.map(d => ({ ...d })) })), extra: extraOf(fit) } : null);
    if (state.undo.length > UNDO_DEPTH)
        state.undo.shift();
}

function undo() {
    if (!state.undo.length || state.worker)
        return;
    const snap = state.undo.pop();
    if (snap)
        setFits(snap.families, snap.extra);
    else
        state.fit = null;
    state.hoverDot = -1;
    state.moving = null;
    state.nudged = false;
    updateModeHint();
    setStatus('Undone.');
    fitChanged();
}

// Add a dot to the selected family, or remove its nearest one, then refit the dots nearby: they were placed to
// explain the image without that dot (neighbors drift into a missing dot's place). Dots placed or moved by hand are
// pinned: the refit leaves them alone. `exact` (Option-click) refits nothing: an added dot stays where clicked.
function editAt(kind, x, y, exact = false) {
    const k = state.current;
    const f = familyFit(k) ?? { family: k, dots: [], radius: currentStyle().radius, sigma: NaN };
    const dots = f.dots.map(d => ({ ...d, flags: [...(d.flags ?? [])] }));
    if (kind === 'remove') {
        const i = nearestDot(f.dots.map(d => ({ ...d, r: f.radius })), x, y, REMOVE_HIT_RADII);
        if (i < 0) {
            setStatus(`No dot of ${familyName(k)} here to remove.`);
            return;
        }
        x = dots[i].x;
        y = dots[i].y;
        dots.splice(i, 1);
    }
    else {
        dots.push({ x, y, added: true, flags: [] });
    }
    pushUndo();
    let out;
    let addedAt = -1;
    let sigma = f.sigma;
    if (exact) {
        // the dot is the user's, as clicked: no position or numbers from the fit, drawn on top if layered
        const added = dots.find(d => d.added);
        if (added) {
            delete added.added;
            Object.assign(added, { placed: true, moved: true },
                currentStyle().layered ? { z: Math.max(0, ...dots.map(d => d.z ?? 0)) + 1 } : {});
        }
        out = flagClose(dots.sort((a, b) => a.x - b.x || a.y - b.y), f.radius);
        addedAt = added ? out.indexOf(added) : -1;
    }
    else {
        const input = dots.map(d => ({ x: d.x, y: d.y, z: d.z, added: d.added, pinned: !!(d.placed || d.moved) }));
        const res = refitNear({ ...fitInputFor(k), style: { ...currentStyle(), radius: f.radius } }, input, x, y);
        // the refit assesses every dot again; the marks of what the user did stay, and a moved dot's position isn't
        // the fit's, so its standard errors and removal margin don't describe it
        out = res.dots.map((d, i) => {
            const was = dots[res.from[i]];
            if (was.added)
                return { ...d, placed: true };
            if (was.moved) {
                const { seX, seY, margin, ...rest } = d;
                return { ...rest, placed: was.placed, moved: true };
            }
            return was.placed ? { ...d, placed: true } : d;
        });
        addedAt = res.from.findIndex(j => dots[j].added);
        sigma = res.sigma;
    }
    const updated = { ...f, dots: out, sigma };
    const others = state.fit?.families.filter(g => g.family !== k) ?? [];
    setFits([...others, updated].sort((a, b) => a.family - b.family), { ...(state.fit ? extraOf(state.fit) : {}), edited: true });
    // the added dot becomes the one the arrow keys nudge
    state.moving = addedAt >= 0 ? { family: k, i: addedAt } : null;
    state.nudged = false;
    state.hoverDot = addedAt >= 0 ? globalIndex(k, addedAt) : -1;
    const how = exact ? (kind === 'add' ? ' where clicked, nothing refitted' : ', nothing refitted') : '';
    setStatus(`${kind === 'add' ? 'Added a dot to' : 'Removed a dot from'} ${familyName(k)}${how}: ${updated.dots.length} dots.`);
    updateModeHint();
    fitChanged();
}

// index into fit.dots of dot i of family k (setFits concatenates the families in order)
function globalIndex(k, i) {
    let n = 0;
    for (const f of state.fit.families) {
        if (f.family === k)
            return n + i;
        n += f.dots.length;
    }
    return -1;
}

// Move mode: choose the selected family's dot nearest the click, or none when no dot is there
function chooseMoving(x, y) {
    const f = familyFit();
    const i = f ? nearestDot(f.dots.map(d => ({ ...d, r: f.radius })), x, y, REMOVE_HIT_RADII) : -1;
    state.moving = i < 0 ? null : { family: f.family, i };
    state.nudged = false;
    state.hoverDot = i < 0 ? -1 : globalIndex(f.family, i);
    refreshMarks();
    updateModeHint();
    if (i >= 0)
        setStatus(`Chosen: dot at ${f.dots[i].x.toFixed(2)}, ${f.dots[i].y.toFixed(2)} px. Arrow keys move it.`);
}

// Move the chosen dot by (dx, dy) px, as placed by hand: no refit, which would pull it back. The fit's per-dot
// numbers (standard errors, visibility, flags) no longer describe it, so they are dropped and it is marked moved.
// A run of nudges to one dot is one undo step.
function nudge(dx, dy) {
    const { family, i } = state.moving;
    if (!state.nudged)
        pushUndo();
    state.nudged = true;
    const families = state.fit.families.map(f => {
        if (f.family !== family)
            return f;
        const dots = f.dots.map((d, j) => {
            if (j !== i)
                return d;
            const { seX, seY, visible, depth, margin, ...rest } = d;
            return { ...rest, x: d.x + dx, y: d.y + dy, flags: [], moved: true };
        });
        return { ...f, dots: flagClose(dots.map(d => ({ ...d, flags: [...d.flags] })), f.radius) };
    });
    setFits(families, { ...extraOf(state.fit), edited: true });
    const d = familyFit(family).dots[i];
    state.hoverDot = globalIndex(family, i);
    setStatus(`Moved the dot to ${d.x.toFixed(2)}, ${d.y.toFixed(2)} px.`);
    fitChanged();
}

// remove every dot of every family; Undo brings them back
function clearDots() {
    const n = state.fit?.dots.length ?? 0;
    if (!n || state.worker)
        return;
    pushUndo();
    state.fit = null;
    setMode(state.mode === 'add' ? 'add' : null);
    setStatus(`Cleared ${n} dots (Undo restores them).`);
    fitChanged();
}

// The dot table as rows of cells: header first, then one row per dot. Copied as tab-separated text (pastes into
// spreadsheets as columns) and downloaded as CSV.
function dotsTable() {
    const s = scales();
    const hasData = s.x?.kind === 'numeric' || s.x?.kind === 'category' || s.y?.kind === 'numeric' || s.y?.kind === 'category';
    // se_noise: standard errors from pixel noise alone; the real error is larger (see the Axes note)
    const head = ['family', 'x_px', 'y_px', 'se_noise_x_px', 'se_noise_y_px'];
    if (hasData)
        head.push('x', 'y', 'se_noise_x', 'se_noise_y');
    const fills = state.fit.dots.some(d => d.fill !== undefined);
    // by_hand: added (clicked, then refined), moved (nudged) or placed (Option-click: exactly as clicked)
    head.push('visible', 'depth', 'flags', 'by_hand');
    if (fills)
        head.push('fill');
    const rows = [head];
    for (const d of state.fit.dots) {
        const color = colorOf(d.family);
        const fam = `${d.family + 1}${color ? ' ' + hex(color) : ''}`;
        const row = [fam, d.x.toFixed(3), d.y.toFixed(3), d.seX?.toFixed(3) ?? '', d.seY?.toFixed(3) ?? ''];
        if (hasData) {
            const dd = dataAt(d.x, d.y);
            // standard errors in data units: the px error times the local data units per px
            const se = (sePx, res) => (sePx === undefined || res === null ? '' : (sePx * res).toPrecision(2));
            // a missing value is an empty cell, which spreadsheets read as missing (formatData shows it as a dash)
            const value = (v, res) => (v === null || v === undefined ? '' : formatData(v, res));
            row.push(value(dd.x, dd.resX), value(dd.y, dd.resY), se(d.seX, dd.resX), se(d.seY, dd.resY));
        }
        row.push(d.visible?.toFixed(2) ?? '', String(d.depth ?? ''), d.flags.join(' '), d.moved ? (d.placed ? 'placed' : 'moved') : d.placed ? 'added' : '');
        if (fills)
            row.push(d.fill === undefined ? '' : fillLabel(d));
        rows.push(row);
    }
    return rows;
}

function tableText(rows) {
    return rows.map(r => r.join('\t')).join('\n') + '\n';
}

// RFC 4180: quote a cell that holds a comma, quote or line break (category names and dates can), doubling quotes
function csvText(rows) {
    const cell = v => (/[",\r\n]/.test(v) ? `"${v.replaceAll('"', '""')}"` : v);
    return rows.map(r => r.map(cell).join(',')).join('\r\n') + '\r\n';
}

function download(text, type, name) {
    const a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob([text], { type }));
    a.download = name;
    a.click();
    URL.revokeObjectURL(a.href);
}

function baseName() {
    return state.name.replace(/\.[a-z]+$/i, '');
}

function setAxisEntry(k, entry) {
    state.axisEntries[k] = entry;
    saveEntries(state.hash, state.axisEntries, state.axisReadings);
    renderAxesPanel();
}

function renderAxesPanel() {
    for (const k of ['x', 'y']) {
        const mode = `tick-${k}`;
        renderAxis($(`axis-${k}`), k, state.axes?.[k] ?? null, state.axisEntries[k], state.axisReadings[k], state.image, entry => setAxisEntry(k, entry),
            { on: state.mode === mode, toggle: () => setMode(state.mode === mode ? null : mode) });
    }
    const s = scales();
    const note = [];
    if (s.x?.kind === 'numeric' && s.y?.kind === 'numeric')
        note.push('Data coordinates appear in the readout under the image, the dot hover and the exported table. Dot standard errors count pixel noise only; '
            + 'with resampled or antialiased edges the real position error is several times larger (a few tenths of a px), '
            + 'and tick placement adds the scale residual shown above.');
    else
        note.push('Type tick values to get data coordinates: an axis\'s are read from its labels only when at least three of them fit one scale, or two on a log axis whose minor ticks confirm them.');
    $('axes-note').textContent = note.join(' ');
    const summary = ['x', 'y'].map(k => {
        const axis = axisOf(k);
        if (!axis)
            return `${k} not found`;
        const sk = s[k]?.kind;
        const reading = readingInEffect(state.axes[k], state.axisReadings[k], state.axisEntries[k]);
        const read = axis.ticks.every((_, i) => tickStatus(state.axisEntries[k], reading, i));
        return sk === 'numeric' ? `${k} ${state.axisEntries[k].type}${read ? ', read' : ''}` : sk === 'category' ? `${k} categories` : `${k}: ${axis.ticks.length} ticks, no values`;
    });
    $('axes-summary').textContent = state.image ? summary.join(' · ') : '';
    refreshAxisMarks();
}

function refreshAxisMarks() {
    const a = state.axes && { x: axisOf('x'), y: axisOf('y') };
    if (!a || !state.showAxes) {
        view.setAxisMarks(null);
        return;
    }
    const marks = [];
    // (a tick placed by hand has only its position: a line across the image shows what it lines up with)
    const found = axis => axis.ticks.filter(t => t.source !== 'user').map(t => t.pos);
    const placed = axis => axis.ticks.filter(t => t.source === 'user').map(t => t.pos);
    if (a.x)
        // gridline ticks have no axis line: mark them at the bottom end of the y axis, or the image's bottom edge
        marks.push({ dir: 'h', pos: a.x.line?.pos ?? null, at: a.x.line ? null : (a.y?.line?.to ?? state.image.height - 1), from: a.x.line?.from, to: a.x.line?.to, ticks: found(a.x), across: placed(a.x) });
    if (a.y) {
        // gridline ticks have no axis line: mark them at the left end of the x axis, or the image's left edge
        const at = a.y.line ? null : (a.x?.line?.from ?? 0);
        marks.push({ dir: 'v', pos: a.y.line?.pos ?? null, at, from: a.y.line?.from, to: a.y.line?.to, ticks: found(a.y), across: placed(a.y) });
    }
    view.setAxisMarks(marks);
}

function saveTruth() {
    const truth = {
        image: state.name, hash: state.hash,
        note: 'Dot centers in image px. Review every dot before using this as ground truth.',
        families: state.fit.families.map(f => ({ family: f.family, color: colorOf(f.family), radius: f.radius })),
        dots: state.fit.dots.map(d => ({ x: +d.x.toFixed(2), y: +d.y.toFixed(2), family: d.family, ...(d.fill === undefined ? {} : { fill: d.fill, fillSure: d.fillSure }) })),
    };
    const name = baseName() + '.json';
    download(JSON.stringify(truth, null, 1) + '\n', 'application/json', name);
    setStatus(`Saved ${state.fit.dots.length} dots as ${name}.`);
}

// '21.7% of the background pixels near marks are slightly off'
function noiseText() {
    return `${(100 * state.noise.fraction).toFixed(state.noise.fraction < 0.01 ? 2 : 1)}% of the background pixels near marks are slightly off`;
}

function setJpeg(jpeg, auto = false) {
    state.jpeg = jpeg;
    state.jpegAuto = jpeg && auto;
    state.work = jpeg ? restoreChroma(state.image) : state.image;
    state.base = null;
    $('jpeg').checked = jpeg;
    const kind = state.jpegAuto ? ', JPEG-like noise' : jpeg ? ', JPEG' : '';
    $('image-summary').textContent = state.image ? `${state.image.width} × ${state.image.height}${kind}` : '';
    $('info-noise').textContent = !state.noise ? '–'
        : `${state.noise.likely ? 'yes, JPEG-like' : 'none'}: ${noiseText()}`;
    // why the box is set as it is, beside it, where it can be changed
    const hint = $('noise-hint');
    hint.textContent = !state.image ? ''
        : state.jpegAuto ? `On by itself: it finds more dot families than exact colors do, and the image has JPEG-like noise (${noiseText()}).`
        : jpeg && state.jpegFile ? 'On: the file is a JPEG.'
        : !jpeg && state.noise?.likely ? `The image has JPEG-like noise (${noiseText()}): try this if dots are missed.`
        : '';
    hint.hidden = !hint.textContent;
}

/** Show the new image and estimate its families. Resolves false if a later load took over while it waited. */
async function setImage(name, image, jpeg, load) {
    showMessage(`Reading ${name}…`, { busy: true });
    await nextPaint();
    if (load !== loads)
        return false;
    state.name = name;
    state.image = image;
    state.noise = jpegNoise(image);
    state.jpegFile = jpeg;
    setJpeg(jpeg);
    state.hash = pixelHash(image);
    $('info-name').textContent = name;
    $('info-size').textContent = `${image.width} × ${image.height} px`;
    $('info-hash').textContent = state.hash;
    setMode(null);
    view.setImage(image);
    state.axes = detectAxes(image);
    state.axisReadings = readTickLabels(image, state.axes);
    state.axisEntries = loadEntries(state.hash, state.axes, state.axisReadings);
    renderAxesPanel();
    $('stage').focus();
    if (!await analyze({ auto: !jpeg }))
        return false;
    showMessage('');
    return true;
}

async function openTestImage(name) {
    const load = ++loads;
    showMessage(`Loading ${name}…`, { busy: true });
    try {
        const image = await loadTestImage(name);
        if (load !== loads || !await setImage(name, image, JPEG_NAME.test(name), load))
            return;
        history.replaceState(null, '', `#image=${encodeURIComponent(name)}`);
        $('test-image').value = name;
    }
    catch (err) {
        if (load === loads)
            showMessage(`Could not open ${name}: ${err.message}`, { error: true });
    }
}

async function openFile(file) {
    const name = file.name && file.name !== 'image.png' ? file.name : 'pasted image';
    const load = ++loads;
    showMessage(`Loading ${name}…`, { busy: true });
    try {
        const image = await decodeImage(file);
        if (load !== loads || !await setImage(name, image, file.type === 'image/jpeg' || JPEG_NAME.test(file.name ?? ''), load))
            return;
        // a pasted or uploaded image can't be reopened from a link
        history.replaceState(null, '', location.pathname + location.search);
        $('test-image').value = '';
    }
    catch (err) {
        if (load === loads)
            showMessage(`Could not decode ${name}: ${err.message}`, { error: true });
    }
}

// a family's style changed: its dots are stale, its mask is rebuilt
function styleChanged() {
    dropFamilyFit(state.current);
    buildLayers();
    renderStylePanel();
    renderFamilies();
    fitChanged();
}

// the open state of each panel section is a per-viewer convenience, remembered when storage allows
function wireSections() {
    for (const el of document.querySelectorAll('aside details[id]')) {
        try {
            const v = localStorage.getItem(OPEN_KEY + el.id);
            if (v !== null)
                el.open = v === '1';
        }
        catch {
            // storage unavailable: the defaults in the markup stand
        }
        el.addEventListener('toggle', () => {
            try {
                localStorage.setItem(OPEN_KEY + el.id, el.open ? '1' : '0');
            }
            catch {
                // not remembered
            }
        });
    }
}

const typing = e => e.target instanceof HTMLInputElement || e.target instanceof HTMLSelectElement || e.target instanceof HTMLTextAreaElement;

function wireControls() {
    $('jpeg').addEventListener('change', async e => {
        if (!state.image)
            return;
        ++loads;
        setJpeg(e.target.checked);
        try {
            if (await analyze())
                showMessage('');
        }
        catch (err) {
            showMessage(`The scan failed: ${err.message}`, { error: true });
        }
    });
    $('pick-color').addEventListener('click', () => setMode(state.mode === 'pick' ? null : 'pick'));
    $('fit-run').addEventListener('click', runFit);
    $('fit-stop').addEventListener('click', () => stopFit());
    $('edit-add').addEventListener('click', () => setMode(state.mode === 'add' ? null : 'add'));
    $('edit-remove').addEventListener('click', () => setMode(state.mode === 'remove' ? null : 'remove'));
    $('edit-move').addEventListener('click', () => setMode(state.mode === 'move' ? null : 'move'));
    $('edit-clear').addEventListener('click', clearDots);
    $('edit-undo').addEventListener('click', undo);
    $('show-axes').addEventListener('change', e => {
        state.showAxes = e.target.checked;
        refreshAxisMarks();
    });
    $('show-dots').addEventListener('change', e => {
        state.showDots = e.target.checked;
        $('show-close').disabled = !state.showDots;
        refreshMarks();
    });
    $('show-close').addEventListener('change', e => {
        state.showClose = e.target.checked;
        renderLegend();
        refreshMarks();
    });
    $('copy-table').addEventListener('click', async () => {
        try {
            await navigator.clipboard.writeText(tableText(dotsTable()));
            setStatus(`Copied ${state.fit.dots.length} dots as a table.`);
        }
        catch (err) {
            setStatus(`Could not copy: ${err.message}`, 'error');
        }
    });
    $('save-csv').addEventListener('click', () => {
        const name = baseName() + '-dots.csv';
        download(csvText(dotsTable()), 'text/csv', name);
        setStatus(`Saved ${state.fit.dots.length} dots as ${name}.`);
    });
    $('save-truth').addEventListener('click', saveTruth);
    $('about-open').addEventListener('click', () => $('about').showModal());
    for (const key of ['radius', 'alpha', 'blur']) {
        $(`style-${key}`).addEventListener('change', e => {
            const v = Number(e.target.value);
            const overrides = state.families[state.current].overrides;
            if (e.target.value === '' || !Number.isFinite(v))
                delete overrides[key];
            else
                overrides[key] = v;
            styleChanged();
        });
    }
    $('style-reset').addEventListener('click', () => {
        const f = state.families[state.current];
        f.overrides = {};
        if (f.outlineEdit === undefined) {
            styleChanged();
            return;
        }
        delete f.outlineEdit;
        f.estimate = f.auto ?? f.estimate;
        outlineChanged(state.current);
    });
    // the outline as the controls show it, or null when it's off
    const outlineInput = () => {
        if (!$('outline-on').checked)
            return null;
        const h = $('outline-color').value;
        const width = Number($('outline-width').value);
        return { color: [1, 3, 5].map(i => parseInt(h.slice(i, i + 2), 16)), width: width > 0 ? width : OUTLINE_DEFAULT.width };
    };
    for (const id of ['outline-on', 'outline-color', 'outline-width']) {
        $(id).addEventListener('change', () => {
            state.families[state.current].outlineEdit = outlineInput();
            outlineChanged(state.current);
        });
    }
    $('outline-pick').addEventListener('click', () => setMode(state.mode === 'pick-outline' ? null : 'pick-outline'));
    $('layer-dilate').addEventListener('change', e => {
        const v = Number(e.target.value);
        state.dilate = e.target.value === '' || !Number.isFinite(v) ? null : Math.max(0, v);
        styleChanged();
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
            setMode(null);
            return;
        }
        // the About dialog is modal: its keys aren't the view's
        if (typing(e) || $('about').open)
            return;
        const arrow = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] }[e.key];
        if (arrow && (state.mode === 'move' || state.mode === 'add') && state.moving && !e.metaKey && !e.ctrlKey && !e.altKey) {
            e.preventDefault();
            const step = e.shiftKey ? NUDGE_FINE : NUDGE_STEP;
            nudge(arrow[0] * step, arrow[1] * step);
            return;
        }
        if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'z' && !e.shiftKey) {
            e.preventDefault();
            undo();
        }
        else if (!e.metaKey && !e.ctrlKey && !e.altKey && e.key === 'a' && !$('edit-add').disabled) {
            setMode(state.mode === 'add' ? null : 'add');
        }
        else if (!e.metaKey && !e.ctrlKey && !e.altKey && e.key === 'r' && !$('edit-remove').disabled) {
            setMode(state.mode === 'remove' ? null : 'remove');
        }
        else if (!e.metaKey && !e.ctrlKey && !e.altKey && e.key === 'm' && !$('edit-move').disabled) {
            setMode(state.mode === 'move' ? null : 'move');
        }
    });
}

async function init() {
    $('dev-badge').hidden = !DEV;
    $('save-truth').hidden = !DEV;
    // diagnostic views (Count, Mask, Classes, Model): Image and Residual, which shows what the dots don't explain,
    // are for everyone
    for (const el of document.querySelectorAll('.dev-only'))
        el.hidden = !DEV;
    const names = DEV ? await listTestImages() : [];
    const select = $('test-image');
    for (const name of names)
        select.add(new Option(name, name));
    // without test images (the public site, or none here): offer only what a visitor can do
    select.hidden = !names.length;
    if (!names.length)
        showMessage('Open an image file, or paste or drop one here.');
    select.addEventListener('change', () => {
        if (select.value)
            openTestImage(select.value);
    });

    $('open-file').addEventListener('click', () => $('file-input').click());
    wireImageInput({ dropTarget: $('stage'), fileInput: $('file-input'), onFile: openFile });
    wireSections();
    wireControls();
    renderLegend();
    renderFamilies();
    renderStylePanel();
    renderFitControls();

    $('zoom-fit').addEventListener('click', () => view.fit());
    $('zoom-1').addEventListener('click', () => view.zoomTo(1));
    $('zoom-in').addEventListener('click', () => view.zoomBy(view.ZOOM_STEP));
    $('zoom-out').addEventListener('click', () => view.zoomBy(1 / view.ZOOM_STEP));

    const match = DEV && /^#image=(.+)$/.exec(location.hash);
    if (match)
        await openTestImage(decodeURIComponent(match[1]));
}

// for tools/check-page.mjs and the console
window.dotscraper = { state, ready: init() };

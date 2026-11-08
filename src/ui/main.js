// App entry: image source controls, the view, and the details panel. The current image name is kept in the URL
// hash (#image=<test image file>) so a reload or a shared local link reopens it.
//
// On each new image the app estimates the dot style (src/core/style.js) and builds the mask (src/core/layers.js);
// the panel shows both and lets the user pick another family, seed one by clicking a dot, or override the
// radius, alpha and blur.

import { pixelAt, pixelHash } from '../core/image.js';
import { estimateStyle } from '../core/style.js';
import { buildMask, CLASS_BACKGROUND, CLASS_FAMILY } from '../core/layers.js';
import { renderDots, solidImage } from '../core/render.js';
import { refitNear } from '../core/fit.js';
import { decodeImage, listTestImages, loadTestImage, wireImageInput } from './input.js';
import { createView, MARK_COLORS } from './view.js';
import {
    countOverlay, maskOverlay, classOverlay, residualOverlay, rgbOverlay, cssColor,
    COUNT_MAX_SHOWN, CLASS_COLORS, OFF_LINE_COLOR, MASK_TINT, RESIDUAL_FULL, RESIDUAL_DARKER, RESIDUAL_LIGHTER, RESIDUAL_MASKED,
} from './overlays.js';

const SAMPLE_SIZE = 80; // CSS px, each sample canvas
const SAMPLE_MARGIN = 4; // px around the dot in the sample crop
const SAMPLE_DIFF_GAIN = 10; // the diff sample shows |image − model| × this
const EXTRAPOLATED_SWATCHES = 3; // level swatches shown beyond the measured ones
const EDIT_HIT_RADII = 0.6; // × r: a click this close to a dot's center removes it in edit mode

const $ = id => document.getElementById(id);

const state = {
    name: null, // display name of the current image
    image: null, // RgbaImage
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

/** The style the fitted dots were drawn with: the style in effect, with the fit's radius. */
function fittedStyle() {
    const style = currentStyle();
    return style && state.fit ? { ...style, radius: state.fit.radius } : style;
}

function fitInput() {
    const style = currentStyle();
    return { image: state.image, style, line: state.estimate.familyMeasure.line, mask: state.layers.mask, D: state.estimate.D };
}

function nearestDot(x, y, within) {
    if (!state.fit)
        return -1;
    let best = -1;
    let bestD = within;
    state.fit.dots.forEach((d, i) => {
        const dd = Math.hypot(d.x - x, d.y - y);
        if (dd <= bestD) {
            bestD = dd;
            best = i;
        }
    });
    return best;
}

function showPixel(x, y) {
    if (x === null || !state.image) {
        for (const id of ['pixel-pos', 'pixel-rgba', 'pixel-count', 'pixel-class', 'pixel-dot'])
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
    const i = state.fit ? nearestDot(x + 0.5, y + 0.5, state.fit.radius) : -1;
    if (i < 0) {
        $('pixel-dot').textContent = '–';
        return;
    }
    const d = state.fit.dots[i];
    const se = d.seX === undefined ? '' : ` ± ${d.seX.toFixed(2)}, ${d.seY.toFixed(2)}`;
    const extra = d.visible === undefined ? '' : `; ${Math.round(100 * d.visible)}% visible, depth ${d.depth}, margin ${d.margin.toFixed(0)}λ`;
    $('pixel-dot').textContent = `${d.x.toFixed(2)}, ${d.y.toFixed(2)} px${se}${extra}${d.flags?.length ? ` (${d.flags.join(', ')})` : ''}`;
}

function showMessage(text, isError = false) {
    const el = $('stage-message');
    el.textContent = text;
    el.classList.toggle('error', isError);
    el.hidden = !text;
}

function displayImage() {
    const { image, estimate, layers } = state;
    if (state.overlay === 'count' && estimate?.D)
        return countOverlay(estimate.D, image.width, image.height);
    if (state.overlay === 'mask' && layers)
        return maskOverlay(image, layers.mask);
    if (state.overlay === 'classes' && layers)
        return classOverlay(layers.classes, layers.thin, image.width, image.height);
    if ((state.overlay === 'model' || state.overlay === 'residual') && state.fit) {
        const style = fittedStyle();
        const bg = solidImage(image.width, image.height, estimate.background.color);
        if (state.overlay === 'model')
            return rgbOverlay(renderDots(bg, state.fit.dots, style));
        const line = estimate.familyMeasure.line;
        const modelT = renderDots(bg, state.fit.dots, { ...style, project: line }).data;
        const imgT = new Float32Array(image.width * image.height);
        for (let p = 0; p < imgT.length; p++)
            imgT[p] = [0, 1, 2].reduce((sum, ch) => sum + (line.B[ch] - image.data[4 * p + ch]) * line.u[ch], 0);
        return residualOverlay(imgT, modelT, layers.mask, image.width, image.height);
    }
    return image;
}

function refreshView(keepTransform) {
    view.setImage(displayImage(), { keepTransform });
    refreshMarks();
}

function refreshMarks(dots = state.fit?.dots) {
    const style = fittedStyle();
    view.setMarks(state.showDots && dots && style ? { dots, radius: state.fit?.radius ?? style.radius } : null);
}

/** Estimate the style (with the current seed or family) and rebuild the mask. */
function analyze() {
    const t0 = performance.now();
    state.estimate = estimateStyle(state.image, state.seed ? { seed: state.seed } : { family: state.family });
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
    state.layers = style ? buildMask(state.image, { ...state.estimate, style }, state.dilate === null ? {} : { dilate: state.dilate }) : null;
    renderLayersPanel();
    renderFitPanel();
}

function renderStylePanel() {
    const est = state.estimate;
    const style = currentStyle();
    const select = $('family');
    select.replaceChildren();
    est.candidates.forEach((c, i) => {
        const label = c.translucent
            ? `${i + 1}: α ${fmt(c.fit.alpha, 2)}, ${c.levels.length} levels`
            : `${i + 1}: one level (opaque?)`;
        select.add(new Option(label, String(i)));
    });
    if (state.seed)
        select.add(new Option(`picked ${state.seed.join(', ')}`, 'seed'));
    select.value = state.seed ? 'seed' : String(state.family);
    select.disabled = !est.candidates.length && !state.seed;

    const swatches = $('levels');
    swatches.replaceChildren();
    if (style) {
        const K = style.measuredLevels;
        style.levels.slice(1, K + 1 + EXTRAPOLATED_SWATCHES).forEach((c, i) => {
            const el = document.createElement('span');
            el.className = 'swatch' + (i >= K ? ' extrapolated' : '');
            el.style.background = cssColor(c);
            el.title = `${i + 1} overlapping: ${c.map(v => v.toFixed(1)).join(', ')}${i >= K ? ' (extrapolated)' : ' (measured)'}`;
            swatches.append(el);
        });
    }
    $('family-swatch').style.background = style ? cssColor(style.color) : 'transparent';
    $('style-c').textContent = style ? style.color.map(v => v.toFixed(1)).join(', ') : '–';
    $('style-levels').textContent = style ? `${style.measuredLevels} measured` : '–';
    const fam = est.familyMeasure;
    if (fam?.fit?.underdetermined)
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
    const model = renderDots(solidImage(size, size, est.background.color), [{ x: dot.cx - x0, y: dot.cy - y0 }], style);
    const real = new ImageData(size, size);
    const mod = new ImageData(size, size);
    const diff = new ImageData(size, size);
    let maxDiff = 0;
    let sumDiff = 0;
    for (let y = 0; y < size; y++) {
        for (let x = 0; x < size; x++) {
            const src = 4 * ((y0 + y) * state.image.width + x0 + x);
            const p = y * size + x;
            let worst = 0;
            for (let ch = 0; ch < 3; ch++) {
                real.data[4 * p + ch] = state.image.data[src + ch];
                mod.data[4 * p + ch] = Math.round(model.data[3 * p + ch]);
                worst = Math.max(worst, Math.abs(state.image.data[src + ch] - model.data[3 * p + ch]));
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
    $('sample-caption').textContent = `Dot at ${dot.cx.toFixed(1)}, ${dot.cy.toFixed(1)} px: image, model, |difference| × ${SAMPLE_DIFF_GAIN} (mean ${(sumDiff / (size * size)).toFixed(2)}, max ${maxDiff.toFixed(1)}, 0–255)`;
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
        + `thin marks ${s.thin} px. Masked ${pct(s.masked)}, including ${s.maskedFamily} dot-color px near other marks.`;
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
        legend.textContent = 'The fitted dots rendered with the style (radius from the fit).';
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
    const [r, g, b] = pixelAt(state.image, x, y);
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
    $('fit-run').title = currentStyle()?.radius ? 'Find the dots' : 'Set the dot radius first';
    $('fit-stop').disabled = !running;
    $('fit-edit').disabled = !fit || running;
    $('copy-csv').disabled = !fit || running;
    $('save-truth').disabled = !fit || running;
    if (running && progress) {
        const secs = ((performance.now() - state.fitStart) / 1000).toFixed(1);
        $('fit-status').textContent = `${progress.stage}, round ${progress.round}: ${progress.dots.length} dots, ${secs} s`;
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
    $('fit-count').textContent = String(fit.dots.length);
    $('fit-flags').textContent = flagged ? `${flagged} (${Object.entries(counts).map(([k, v]) => `${k} ${v}`).join(', ')})` : '0';
    $('fit-radius').textContent = `${fit.radius.toFixed(2)} px`;
    $('fit-sigma').textContent = fit.sigma.toFixed(2);
    if (!running) {
        $('fit-status').textContent = `${fit.partial ? 'Stopped early: partial result, not assessed. ' : ''}`
            + `${fit.edited ? 'Edited by hand. ' : ''}${fit.ms ? `Fit in ${(fit.ms / 1000).toFixed(1)} s. ` : ''}`
            + 'Positions in image px (pixel i spans i to i + 1).';
    }
}

function runFit() {
    stopFit();
    state.fitStart = performance.now();
    state.worker = new Worker(new URL('../fit-worker.js', import.meta.url), { type: 'module' });
    state.fit = null;
    let last = null;
    state.worker.onmessage = e => {
        const msg = e.data;
        if (msg.type === 'progress') {
            last = msg.progress;
            refreshMarks(last.dots);
            renderFitPanel(last);
            return;
        }
        state.worker.terminate();
        state.worker = null;
        if (msg.type === 'error') {
            $('fit-status').textContent = `Fit failed: ${msg.message}`;
            renderFitPanel();
            return;
        }
        state.fit = msg.result;
        renderFitPanel();
        refreshView(true);
    };
    state.worker.onerror = e => {
        $('fit-status').textContent = `Fit failed: ${e.message}`;
        state.worker = null;
        renderFitPanel();
    };
    state.stopFit = () => last;
    state.worker.postMessage({ input: fitInput(), opts: {} });
    renderFitPanel({ stage: 'starting', round: 0, dots: [] });
}

function stopFit() {
    if (!state.worker)
        return;
    state.worker.terminate();
    state.worker = null;
    const last = state.stopFit?.();
    // keep what the worker had reached, unassessed (no standard errors or flags)
    if (last) {
        state.fit = { dots: last.dots.map(d => ({ ...d, flags: [] })), radius: currentStyle().radius, sigma: NaN, partial: true };
    }
    renderFitPanel();
    refreshView(true);
}

function editAt(x, y) {
    const r = state.fit.radius;
    const i = nearestDot(x, y, EDIT_HIT_RADII * r);
    const dots = state.fit.dots.map(d => ({ x: d.x, y: d.y }));
    if (i >= 0)
        dots.splice(i, 1);
    else
        dots.push({ x, y });
    const res = refitNear({ ...fitInput(), style: fittedStyle() }, dots, x, y);
    state.fit = { ...state.fit, dots: res.dots, sigma: res.sigma, edited: true };
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
    const rows = ['x_px\ty_px\tse_x_px\tse_y_px\tvisible\tdepth\tflags'];
    for (const d of state.fit.dots)
        rows.push([d.x.toFixed(3), d.y.toFixed(3), d.seX?.toFixed(3) ?? '', d.seY?.toFixed(3) ?? '', d.visible?.toFixed(2) ?? '', d.depth ?? '', d.flags.join(' ')].join('\t'));
    return rows.join('\n') + '\n';
}

function saveTruth() {
    const truth = {
        image: state.name, hash: state.hash, radius: state.fit.radius,
        note: 'Dot centers in image px. Review every dot before using this as ground truth.',
        dots: state.fit.dots.map(d => ({ x: +d.x.toFixed(2), y: +d.y.toFixed(2) })),
    };
    const a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob([JSON.stringify(truth, null, 1) + '\n'], { type: 'application/json' }));
    a.download = state.name.replace(/\.[a-z]+$/i, '') + '.json';
    a.click();
    URL.revokeObjectURL(a.href);
}

function setImage(name, image) {
    state.name = name;
    state.image = image;
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
    analyze();
    $('stage').focus();
}

async function openTestImage(name) {
    showMessage(`Loading ${name}…`);
    try {
        setImage(name, await loadTestImage(name));
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
        setImage(name, await decodeImage(file));
        // a pasted or uploaded image can't be reopened from a link
        history.replaceState(null, '', location.pathname);
        $('test-image').value = '';
    }
    catch (err) {
        showMessage(`Could not decode ${name}: ${err.message}`, true);
    }
}

function wireControls() {
    $('family').addEventListener('change', e => {
        if (e.target.value === 'seed')
            return;
        state.seed = null;
        state.family = Number(e.target.value);
        state.overrides = {};
        analyze();
    });
    $('pick-color').addEventListener('click', () => {
        setPicking(!state.picking);
        if (state.picking)
            setEditing(false);
    });
    $('fit-run').addEventListener('click', runFit);
    $('fit-stop').addEventListener('click', stopFit);
    $('fit-edit').addEventListener('click', () => setEditing(!state.editing));
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

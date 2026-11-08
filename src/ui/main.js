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
import { decodeImage, listTestImages, loadTestImage, wireImageInput } from './input.js';
import { createView } from './view.js';
import { countOverlay, maskOverlay, classOverlay, cssColor, COUNT_MAX_SHOWN, CLASS_COLORS, OFF_LINE_COLOR, MASK_TINT } from './overlays.js';

const SAMPLE_SIZE = 80; // CSS px, each sample canvas
const SAMPLE_MARGIN = 4; // px around the dot in the sample crop
const SAMPLE_DIFF_GAIN = 10; // the diff sample shows |image − model| × this
const EXTRAPOLATED_SWATCHES = 3; // level swatches shown beyond the measured ones

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

function showPixel(x, y) {
    if (x === null || !state.image) {
        for (const id of ['pixel-pos', 'pixel-rgba', 'pixel-count', 'pixel-class'])
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
    return image;
}

function refreshView(keepTransform) {
    view.setImage(displayImage(), { keepTransform });
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

function rebuildMask() {
    const style = currentStyle();
    state.layers = style ? buildMask(state.image, { ...state.estimate, style }, state.dilate === null ? {} : { dilate: state.dilate }) : null;
    renderLayersPanel();
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
    const item = (color, text) => `<span class="key" style="background:${cssColor(color)}"></span>${text}`;
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
    $('pick-color').addEventListener('click', () => setPicking(!state.picking));
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
        if (e.key === 'Escape' && state.picking)
            setPicking(false);
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

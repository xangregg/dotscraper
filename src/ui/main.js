// App entry: image source controls, the view, and the details panel. The current image name is kept in the URL
// hash (#image=<test image file>) so a reload or a shared local link reopens it.

import { pixelAt, pixelHash } from '../core/image.js';
import { decodeImage, listTestImages, loadTestImage, wireImageInput } from './input.js';
import { createView } from './view.js';

const $ = id => document.getElementById(id);

const state = {
    name: null, // display name of the current image
    image: null, // RgbaImage
    hash: null,
};

const view = createView({
    stage: $('stage'),
    canvas: $('view'),
    onHover: showPixel,
    onZoom: scale => {
        $('zoom-level').textContent = `${scale >= 1 ? Math.round(scale * 100) : (scale * 100).toFixed(1)}%`;
    },
});

function showPixel(x, y) {
    if (x === null || !state.image) {
        $('pixel-pos').textContent = '–';
        $('pixel-rgba').textContent = '–';
        $('pixel-swatch').style.background = 'transparent';
        return;
    }
    const [r, g, b, a] = pixelAt(state.image, x, y);
    $('pixel-pos').textContent = `${x}, ${y} px`;
    $('pixel-rgba').textContent = `${r}, ${g}, ${b}, ${a}`;
    $('pixel-swatch').style.background = `rgb(${r} ${g} ${b})`;
}

function showMessage(text, isError = false) {
    const el = $('stage-message');
    el.textContent = text;
    el.classList.toggle('error', isError);
    el.hidden = !text;
}

function setImage(name, image) {
    state.name = name;
    state.image = image;
    state.hash = pixelHash(image);
    $('info-name').textContent = name;
    $('info-size').textContent = `${image.width} × ${image.height} px`;
    $('info-hash').textContent = state.hash;
    showMessage('');
    view.setImage(image);
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

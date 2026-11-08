// Getting images into the app: decoding, the list of local test images, and paste / drop / file-picker wiring.

const IMAGE_DIR = 'testgraphs/';
const IMAGE_INDEX = 'index.json'; // in IMAGE_DIR: a JSON array of file names, written by npm run images:index
// The Server header of Python's http.server (npm run serve), which lists directories. Other servers (WebStorm's
// built-in one) answer a directory request with 404 or an error page, so they get the index file instead.
const LISTING_SERVER = /^SimpleHTTP\//;
const IMAGE_FILE = /\.(png|jpe?g|gif|webp)$/i;

/**
 * Decode an image file to RGBA values exactly as stored. `colorSpaceConversion: 'none'` stops the browser from
 * converting an embedded color profile (pasted screenshots are often Display P3) to the display's space, so the
 * values match what the Node tools read. The canvas stores premultiplied alpha, so values of partly transparent
 * pixels may round; graph images are almost always opaque.
 *
 * @param {Blob} blob
 * @returns {Promise<import('../core/image.js').RgbaImage>}
 */
export async function decodeImage(blob) {
    const bitmap = await createImageBitmap(blob, { colorSpaceConversion: 'none', premultiplyAlpha: 'none' });
    const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
    const ctx = canvas.getContext('2d', { colorSpace: 'srgb', willReadFrequently: true });
    ctx.drawImage(bitmap, 0, 0);
    bitmap.close();
    const id = ctx.getImageData(0, 0, canvas.width, canvas.height);
    return { width: id.width, height: id.height, data: id.data };
}

/**
 * File names of the local test images: from the server's directory listing when it gives one, else from the index
 * file, else none (a deployed site has no test images).
 *
 * @returns {Promise<string[]>}
 */
export async function listTestImages() {
    try {
        const probe = await fetch(IMAGE_DIR);
        if (probe.ok && LISTING_SERVER.test(probe.headers.get('server') ?? '')) {
            const page = new DOMParser().parseFromString(await probe.text(), 'text/html');
            return [...page.querySelectorAll('a[href]')]
                .map(a => decodeURIComponent(a.getAttribute('href').split(/[?#]/)[0]))
                .filter(name => IMAGE_FILE.test(name) && !name.includes('/'))
                .sort();
        }
        const index = await fetch(IMAGE_DIR + IMAGE_INDEX);
        if (index.ok)
            return await index.json();
    }
    catch {
        // no local images: offline, or a server without the directory
    }
    return [];
}

/**
 * Fetch and decode one of the local test images.
 *
 * @param {string} name  file name in IMAGE_DIR
 * @returns {Promise<import('../core/image.js').RgbaImage>}
 */
export async function loadTestImage(name) {
    const response = await fetch(IMAGE_DIR + encodeURIComponent(name));
    if (!response.ok)
        throw new Error(`could not load ${name} (${response.status})`);
    return decodeImage(await response.blob());
}

/**
 * Call onFile with each image the user pastes anywhere on the page, drops on the drop target, or picks with the
 * file input.
 *
 * @param {object} opts
 * @param {HTMLElement} opts.dropTarget
 * @param {HTMLInputElement} opts.fileInput
 * @param {function(File): void} opts.onFile
 */
export function wireImageInput({ dropTarget, fileInput, onFile }) {
    document.addEventListener('paste', e => {
        const item = [...(e.clipboardData?.items ?? [])].find(it => it.kind === 'file' && it.type.startsWith('image/'));
        if (!item)
            return;
        e.preventDefault();
        onFile(item.getAsFile());
    });
    // dragover must be cancelled for the element to accept a drop
    dropTarget.addEventListener('dragover', e => {
        e.preventDefault();
        dropTarget.classList.add('drop-target');
    });
    dropTarget.addEventListener('dragleave', () => dropTarget.classList.remove('drop-target'));
    dropTarget.addEventListener('drop', e => {
        e.preventDefault();
        dropTarget.classList.remove('drop-target');
        const file = [...(e.dataTransfer?.files ?? [])].find(f => f.type.startsWith('image/'));
        if (file)
            onFile(file);
    });
    fileInput.addEventListener('change', () => {
        if (fileInput.files.length)
            onFile(fileInput.files[0]);
        fileInput.value = '';
    });
}

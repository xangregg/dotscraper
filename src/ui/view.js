// Canvas view of the image with zoom and pan. The view keeps a transform from image pixels to CSS pixels:
// screen = (image − origin) · scale. Enlarged, pixels are drawn as crisp squares so individual values can be
// inspected; reduced, the browser smooths them.

const ZOOM_STEP = 2 ** 0.5; // factor per zoom button press or key press
const WHEEL_ZOOM_RATE = 1 / 300; // log2 of the zoom factor per unit of wheel deltaY
const MIN_SCALE = 1 / 32; // CSS px per image px
const MAX_SCALE = 64; // CSS px per image px
const FIT_MARGIN = 12; // CSS px left around the image when fitting
const CLICK_SLOP = 3; // CSS px a pointer may move between press and release and still count as a click

/**
 * @param {object} opts
 * @param {HTMLElement} opts.stage   container whose size the canvas fills; receives the pointer events
 * @param {HTMLCanvasElement} opts.canvas
 * @param {function(number|null, number|null): void} opts.onHover  image pixel under the pointer, or nulls
 * @param {function(number, number): void} opts.onClick  image pixel clicked (a press and release without a drag)
 * @param {function(number): void} opts.onZoom  the new scale
 */
export function createView({ stage, canvas, onHover, onClick, onZoom }) {
    const ctx = canvas.getContext('2d');
    let source = null; // canvas holding the image at full resolution
    let width = 0;
    let height = 0;
    let scale = 1;
    let originX = 0; // image px at the left edge of the view
    let originY = 0; // image px at the top edge of the view
    let drag = null;

    function draw() {
        const dpr = window.devicePixelRatio || 1;
        const w = stage.clientWidth;
        const h = stage.clientHeight;
        if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
            canvas.width = Math.round(w * dpr);
            canvas.height = Math.round(h * dpr);
        }
        ctx.setTransform(1, 0, 0, 1, 0, 0);
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        if (!source)
            return;
        ctx.imageSmoothingEnabled = scale * dpr < 1;
        ctx.setTransform(scale * dpr, 0, 0, scale * dpr, -originX * scale * dpr, -originY * scale * dpr);
        ctx.drawImage(source, 0, 0);
    }

    function setScale(newScale, anchorX, anchorY) {
        // keep the image point under (anchorX, anchorY), in CSS px within the stage, fixed
        const s = Math.min(MAX_SCALE, Math.max(MIN_SCALE, newScale));
        const ix = originX + anchorX / scale;
        const iy = originY + anchorY / scale;
        scale = s;
        originX = ix - anchorX / scale;
        originY = iy - anchorY / scale;
        onZoom(scale);
        draw();
    }

    function fit() {
        if (!source)
            return;
        const w = stage.clientWidth;
        const h = stage.clientHeight;
        scale = Math.min(MAX_SCALE, Math.max(MIN_SCALE, Math.min((w - 2 * FIT_MARGIN) / width, (h - 2 * FIT_MARGIN) / height)));
        originX = -(w / scale - width) / 2;
        originY = -(h / scale - height) / 2;
        onZoom(scale);
        draw();
    }

    function imagePoint(e) {
        const r = stage.getBoundingClientRect();
        return { x: originX + (e.clientX - r.left) / scale, y: originY + (e.clientY - r.top) / scale };
    }

    stage.addEventListener('wheel', e => {
        if (!source)
            return;
        e.preventDefault();
        const r = stage.getBoundingClientRect();
        // trackpad pinch arrives as a wheel event with ctrlKey and small deltas; scale them up to feel the same
        const delta = e.ctrlKey ? e.deltaY * 8 : e.deltaY;
        setScale(scale * 2 ** (-delta * WHEEL_ZOOM_RATE), e.clientX - r.left, e.clientY - r.top);
    }, { passive: false });

    stage.addEventListener('pointerdown', e => {
        if (!source || e.button !== 0)
            return;
        drag = { x: e.clientX, y: e.clientY, originX, originY };
        stage.setPointerCapture(e.pointerId);
        stage.classList.add('dragging');
    });
    stage.addEventListener('pointermove', e => {
        if (drag) {
            originX = drag.originX - (e.clientX - drag.x) / scale;
            originY = drag.originY - (e.clientY - drag.y) / scale;
            draw();
        }
        if (!source)
            return;
        const p = imagePoint(e);
        const x = Math.floor(p.x);
        const y = Math.floor(p.y);
        if (x >= 0 && y >= 0 && x < width && y < height)
            onHover(x, y);
        else
            onHover(null, null);
    });
    stage.addEventListener('pointerup', e => {
        if (drag && Math.hypot(e.clientX - drag.x, e.clientY - drag.y) <= CLICK_SLOP) {
            const p = imagePoint(e);
            const x = Math.floor(p.x);
            const y = Math.floor(p.y);
            if (x >= 0 && y >= 0 && x < width && y < height)
                onClick(x, y);
        }
        drag = null;
        stage.classList.remove('dragging');
    });
    stage.addEventListener('pointerleave', () => onHover(null, null));

    stage.addEventListener('keydown', e => {
        const cx = stage.clientWidth / 2;
        const cy = stage.clientHeight / 2;
        if (e.key === '0')
            fit();
        else if (e.key === '1')
            setScale(1, cx, cy);
        else if (e.key === '+' || e.key === '=')
            setScale(scale * ZOOM_STEP, cx, cy);
        else if (e.key === '-')
            setScale(scale / ZOOM_STEP, cx, cy);
    });

    new ResizeObserver(draw).observe(stage);

    return {
        /**
         * @param {import('../core/image.js').RgbaImage} image
         * @param {{keepTransform?: boolean}} [opts]  keep the zoom and pan (for an overlay of the same image)
         */
        setImage(image, opts = {}) {
            source = document.createElement('canvas');
            source.width = image.width;
            source.height = image.height;
            source.getContext('2d').putImageData(new ImageData(image.data, image.width, image.height), 0, 0);
            const same = image.width === width && image.height === height;
            width = image.width;
            height = image.height;
            if (opts.keepTransform && same)
                draw();
            else
                fit();
        },
        fit,
        zoomBy(factor) {
            setScale(scale * factor, stage.clientWidth / 2, stage.clientHeight / 2);
        },
        zoomTo(s) {
            setScale(s, stage.clientWidth / 2, stage.clientHeight / 2);
        },
        ZOOM_STEP,
    };
}

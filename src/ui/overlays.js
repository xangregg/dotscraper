// Display images for the view's overlay modes. Each returns an RgbaImage the size of the source.

import { CLASS_BACKGROUND, CLASS_FAMILY } from '../core/layers.js';

export const COUNT_MAX_SHOWN = 6; // overlap count drawn as white; deeper counts are clipped (the hover shows them)
const OFF_LINE_COLOR = [150, 30, 30]; // count overlay: pixels not on the family's color line
const MASK_TINT = [220, 0, 200];
const MASK_TINT_STRENGTH = 0.6;
const CLASS_COLORS = {
    background: [235, 235, 232],
    family: [60, 170, 90],
    other: [230, 140, 40],
    thin: [130, 60, 200],
};

function blank(width, height) {
    return { width, height, data: new Uint8ClampedArray(4 * width * height) };
}

/**
 * Overlap count D in gray, 0 black to COUNT_MAX_SHOWN white; off-line pixels dark red.
 *
 * @param {Float32Array} D  width × height; NaN off the family's line
 * @param {number} width
 * @param {number} height
 * @returns {import('../core/image.js').RgbaImage}
 */
export function countOverlay(D, width, height) {
    const out = blank(width, height);
    for (let p = 0; p < D.length; p++) {
        if (Number.isNaN(D[p])) {
            out.data.set([...OFF_LINE_COLOR, 255], 4 * p);
            continue;
        }
        const v = Math.round(255 * Math.min(1, D[p] / COUNT_MAX_SHOWN));
        out.data.set([v, v, v, 255], 4 * p);
    }
    return out;
}

/**
 * The image with masked pixels tinted.
 *
 * @param {import('../core/image.js').RgbaImage} image
 * @param {Uint8Array} mask  1 where masked
 * @returns {import('../core/image.js').RgbaImage}
 */
export function maskOverlay(image, mask) {
    const out = blank(image.width, image.height);
    out.data.set(image.data);
    for (let p = 0; p < mask.length; p++) {
        if (!mask[p])
            continue;
        for (let ch = 0; ch < 3; ch++)
            out.data[4 * p + ch] += (MASK_TINT[ch] - out.data[4 * p + ch]) * MASK_TINT_STRENGTH;
    }
    return out;
}

/**
 * Flat colors per pixel class; thin marks in the family's color on top.
 *
 * @param {Uint8Array} classes  core/layers.js CLASS_BACKGROUND, CLASS_FAMILY or CLASS_OTHER per pixel
 * @param {Uint8Array} thin  1 on thin marks
 * @param {number} width
 * @param {number} height
 * @returns {import('../core/image.js').RgbaImage}
 */
export function classOverlay(classes, thin, width, height) {
    const out = blank(width, height);
    for (let p = 0; p < classes.length; p++) {
        let c = CLASS_COLORS.other;
        if (thin[p])
            c = CLASS_COLORS.thin;
        else if (classes[p] === CLASS_BACKGROUND)
            c = CLASS_COLORS.background;
        else if (classes[p] === CLASS_FAMILY)
            c = CLASS_COLORS.family;
        out.data.set([...c, 255], 4 * p);
    }
    return out;
}

const RESIDUAL_FULL = 30; // t units (0–255) at full saturation in the residual overlay
const RESIDUAL_DARKER = [200, 40, 40]; // image darker than the model: ink the model is missing
const RESIDUAL_LIGHTER = [40, 90, 210]; // image lighter: the model has too much
const RESIDUAL_MASKED = [150, 150, 150];

/**
 * Signed residual along the color line, image t minus model t: white where they agree, red where the image is
 * darker (a dot missing), blue where lighter (a dot too many or misplaced), gray where masked.
 *
 * @param {Float32Array} imgT  the image's t per pixel, 0–255 units
 * @param {Float32Array} modelT  the model's
 * @param {Uint8Array} mask  1 where masked
 * @param {number} width
 * @param {number} height
 * @returns {import('../core/image.js').RgbaImage}  full color at RESIDUAL_FULL units
 */
export function residualOverlay(imgT, modelT, mask, width, height) {
    const out = blank(width, height);
    for (let p = 0; p < imgT.length; p++) {
        if (mask[p]) {
            out.data.set([...RESIDUAL_MASKED, 255], 4 * p);
            continue;
        }
        const r = imgT[p] - modelT[p];
        const f = Math.min(1, Math.abs(r) / RESIDUAL_FULL);
        const c = r > 0 ? RESIDUAL_DARKER : RESIDUAL_LIGHTER;
        out.data.set([255 + (c[0] - 255) * f, 255 + (c[1] - 255) * f, 255 + (c[2] - 255) * f, 255], 4 * p);
    }
    return out;
}

/**
 * An RgbImage (floats) as an RgbaImage for display.
 *
 * @param {import('../core/render.js').RgbImage} rgb
 * @returns {import('../core/image.js').RgbaImage}
 */
export function rgbOverlay(rgb) {
    const out = blank(rgb.width, rgb.height);
    for (let p = 0; p < rgb.width * rgb.height; p++)
        out.data.set([rgb.data[3 * p], rgb.data[3 * p + 1], rgb.data[3 * p + 2], 255], 4 * p);
    return out;
}

/**
 * @param {number[]} rgb  0–255, rounded and clamped
 * @returns {string}  a CSS color
 */
export function cssColor(rgb) {
    return `rgb(${rgb.map(v => Math.round(Math.min(255, Math.max(0, v)))).join(' ')})`;
}

export { CLASS_COLORS, OFF_LINE_COLOR, MASK_TINT, RESIDUAL_FULL, RESIDUAL_DARKER, RESIDUAL_LIGHTER, RESIDUAL_MASKED };

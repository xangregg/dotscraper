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

/** Overlap count D in gray, 0 black to COUNT_MAX_SHOWN white; off-line pixels dark red. */
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

/** The image with masked pixels tinted. */
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

/** Flat colors per pixel class; thin marks in the family's color on top. */
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

export function cssColor(rgb) {
    return `rgb(${rgb.map(v => Math.round(Math.min(255, Math.max(0, v)))).join(' ')})`;
}

export { CLASS_COLORS, OFF_LINE_COLOR, MASK_TINT };

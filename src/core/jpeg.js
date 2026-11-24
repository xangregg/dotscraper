// JPEG chroma restoration. JPEG stores color (Cb, Cr) at half resolution and the decoder upsamples it smoothly, so
// near every edge the background's color bleeds into the dots: inside isolated bench dots the chroma error is 4–5.6
// units rms (up to 16), while luma, stored at full resolution, is off by about 2. The style estimation needs
// colors on the line from the background to the dot color, and this bleed moves them off it.
//
// A graph's colors are piecewise constant, and within a small window there are usually two of them (a dot and its
// background, or two overlap levels), so chroma is a linear function of luma there. A guided filter (He, Sun and
// Tang 2010, "Guided Image Filtering") fits exactly that: chroma = a · luma + b in each window, with a and b averaged
// over the windows covering a pixel. Chroma then follows the sharp luma edges again. Luma is left as decoded
// (smoothing it too didn't change the bench results).

const GUIDE_RADIUS = 4; // px: window half-width; covers a dot's edge and some background at r 5–13
const GUIDE_EPS = 4; // (0–255 units)²: regularization; luma variance below this counts as flat
const GUIDE_RES_SCALE = 800; // (0–255 units)²: a window's chroma variance its line leaves, at which it counts half

// JPEG noise (jpegNoise): background pixels a few px from any mark that aren't exactly the background color
const NOISE_NEAR = 12; // 0–255 units, max channel: a pixel this close to the background is background-ish …
const NOISE_BAND = [3, 6]; // px from the nearest mark: … and examined in this band, beyond any antialiased rim
const NOISE_MIN_FRACTION = 0.01; // of the band: JPEG noise (clean renders at most 0.013%, JPEGs 19–51%, PNGs from them 4.8–51%)
const NOISE_MIN_PIXELS = 1000; // band pixels needed to judge

import { backgroundColor } from './scan.js';
import { distanceTransform } from './layers.js';

// BT.601 full range, as JPEG (JFIF) uses
const KR = 0.299;
const KG = 0.587;
const KB = 0.114;

/** Mean over the (2r + 1)² window clipped to the image, O(pixels). */
function boxMean(src, width, height, r) {
    const tmp = new Float32Array(width * height);
    const out = new Float32Array(width * height);
    for (let y = 0; y < height; y++) {
        const row = y * width;
        let sum = 0;
        let n = 0;
        for (let x = 0; x <= Math.min(r, width - 1); x++) {
            sum += src[row + x];
            n++;
        }
        for (let x = 0; x < width; x++) {
            tmp[row + x] = sum / n;
            if (x + r + 1 < width) {
                sum += src[row + x + r + 1];
                n++;
            }
            if (x - r >= 0) {
                sum -= src[row + x - r];
                n--;
            }
        }
    }
    for (let x = 0; x < width; x++) {
        let sum = 0;
        let n = 0;
        for (let y = 0; y <= Math.min(r, height - 1); y++) {
            sum += tmp[y * width + x];
            n++;
        }
        for (let y = 0; y < height; y++) {
            out[y * width + x] = sum / n;
            if (y + r + 1 < height) {
                sum += tmp[(y + r + 1) * width + x];
                n++;
            }
            if (y - r >= 0) {
                sum -= tmp[(y - r) * width + x];
                n--;
            }
        }
    }
    return out;
}

/**
 * Guided filter of p with guide I: per-window least squares p ≈ a·I + b, coefficients averaged, each window weighted
 * by how well its line fits (1 / (1 + residual variance / GUIDE_RES_SCALE)). Where the windows covering a pixel fit
 * badly, three colors meet and chroma isn't a function of luma there: the pixel keeps its own value in proportion
 * (step1: a blue dot touching a red ring, 148, 187, 236, was pulled to 178, 178, 204, off its family's line; at
 * GUIDE_RES_SCALE 800, 170, 181, 211). Two-color windows fit less than exactly in a real JPEG, so the scale trades
 * restoration at rims (bench1 medium-dots, rms within 1.5 px of r 5.6: 11.1 raw, 5.4 unweighted and at 800, 6.1 at
 * 100) against keeping such pixels. Smaller scales cost elsewhere: at 100 beeswarm-boxplot's packed red dots, among
 * gray and white, kept their fringes and were no family; at 400–600 bench1 group-smoother.jpg's blue lost its lone
 * dots (radius unknown). step1 at 800 against unweighted: 73 blue dots of 100 against 70, 55 rings of 75 against 52.
 */
function guidedFilter(I, p, width, height, r, eps) {
    const n = width * height;
    const meanI = boxMean(I, width, height, r);
    const meanP = boxMean(p, width, height, r);
    const ip = new Float32Array(n);
    const ii = new Float32Array(n);
    const pp = new Float32Array(n);
    for (let i = 0; i < n; i++) {
        ip[i] = I[i] * p[i];
        ii[i] = I[i] * I[i];
        pp[i] = p[i] * p[i];
    }
    const meanIP = boxMean(ip, width, height, r);
    const meanII = boxMean(ii, width, height, r);
    const meanPP = boxMean(pp, width, height, r);
    const wa = new Float32Array(n);
    const wb = new Float32Array(n);
    const w = new Float32Array(n);
    for (let i = 0; i < n; i++) {
        const cov = meanIP[i] - meanI[i] * meanP[i];
        const a = cov / (meanII[i] - meanI[i] * meanI[i] + eps);
        const b = meanP[i] - a * meanI[i];
        const residual = Math.max(0, meanPP[i] - meanP[i] * meanP[i] - a * cov);
        w[i] = 1 / (1 + residual / GUIDE_RES_SCALE);
        wa[i] = w[i] * a;
        wb[i] = w[i] * b;
    }
    const meanW = boxMean(w, width, height, r);
    const meanWA = boxMean(wa, width, height, r);
    const meanWB = boxMean(wb, width, height, r);
    const q = new Float32Array(n);
    for (let i = 0; i < n; i++)
        q[i] = meanWA[i] * I[i] + meanWB[i] + (1 - meanW[i]) * p[i];
    return q;
}

/**
 * The image with its chroma re-fitted to its luma (see above). Alpha is kept.
 *
 * @param {import('./image.js').RgbaImage} image  sRGB 0–255, as decoded
 * @returns {import('./image.js').RgbaImage}  sRGB 0–255
 */
export function restoreChroma(image) {
    const { width, height, data } = image;
    const n = width * height;
    const Y = new Float32Array(n);
    const Cb = new Float32Array(n);
    const Cr = new Float32Array(n);
    for (let i = 0; i < n; i++) {
        const r = data[4 * i];
        const g = data[4 * i + 1];
        const b = data[4 * i + 2];
        Y[i] = KR * r + KG * g + KB * b;
        Cb[i] = (b - Y[i]) / (2 * (1 - KB));
        Cr[i] = (r - Y[i]) / (2 * (1 - KR));
    }
    const cb = guidedFilter(Y, Cb, width, height, GUIDE_RADIUS, GUIDE_EPS);
    const cr = guidedFilter(Y, Cr, width, height, GUIDE_RADIUS, GUIDE_EPS);
    const out = new Uint8ClampedArray(data.length);
    for (let i = 0; i < n; i++) {
        const r = Y[i] + 2 * (1 - KR) * cr[i];
        const b = Y[i] + 2 * (1 - KB) * cb[i];
        out[4 * i] = Math.round(r);
        out[4 * i + 1] = Math.round((Y[i] - KR * r - KB * b) / KG);
        out[4 * i + 2] = Math.round(b);
        out[4 * i + 3] = data[4 * i + 3];
    }
    return { width, height, data: out };
}

/**
 * Whether an image carries JPEG noise, whatever its file type: a PNG screenshot of a web figure is often a JPEG
 * underneath, and its marks fail the exact-color tests (bench-nature primates1: an isolated dot's edge runs 207, 233,
 * 243 then 128, 182, 213, off the line from white to its 33, 118, 183, so no dot was found). A rendered graph is
 * piecewise constant: a few px from any mark, the background is exactly the background color (the bench
 * renders: none of those pixels differ; bench-local's PNGs: at most 0.013%). JPEG's ringing reaches across its 8 px
 * blocks, so there many are slightly off (the bench JPEGs 19–51%; the Nature PNGs 4.8–51%). Other noise reads the
 * same (synth.js presets with noise: 29–40%). O(pixels).
 *
 * @param {import('./image.js').RgbaImage} image  sRGB 0–255
 * @returns {{fraction: number, pixels: number, likely: boolean}}  fraction: of the band pixels that are slightly
 *   off the background; pixels: the band's size
 */
export function jpegNoise(image) {
    const { width, height, data } = image;
    const B = backgroundColor(image).color;
    const n = width * height;
    const mark = new Uint8Array(n);
    const off = new Uint8Array(n);
    for (let p = 0; p < n; p++) {
        const d = Math.max(Math.abs(data[4 * p] - B[0]), Math.abs(data[4 * p + 1] - B[1]), Math.abs(data[4 * p + 2] - B[2]));
        mark[p] = d > NOISE_NEAR ? 1 : 0;
        off[p] = d > 0 ? 1 : 0;
    }
    const d2 = distanceTransform(mark, width, height);
    const [lo, hi] = NOISE_BAND.map(v => v * v);
    let pixels = 0;
    let noisy = 0;
    for (let p = 0; p < n; p++) {
        if (d2[p] < lo || d2[p] > hi)
            continue;
        pixels++;
        noisy += off[p];
    }
    const fraction = pixels ? noisy / pixels : 0;
    return { fraction, pixels, likely: pixels >= NOISE_MIN_PIXELS && fraction >= NOISE_MIN_FRACTION };
}


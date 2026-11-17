// Module worker that runs the scan for dot families off the main thread: on large images it takes up to a minute,
// and on the main thread the browser called the page unresponsive. It only wraps src/core/ functions.
//
// Messages in: { op, ...args }, one at a time (to stop one, the page terminates the worker):
//   'scan'   { image, work, jpeg, tryNoisy }: estimateFamilies on work (the image as analyzed: a JPEG's chroma
//            restored). With tryNoisy, also on the image with its chroma restored and noisy color matching, and the
//            one that finds more families with a radius wins (main.js analyze says why).
//   'seed'   { image, seed, jpeg }: estimateStyle of a picked color.
//   'opaque' { image, jobs: [{ seed, opts }] }: opaqueFamilyEstimate for each.
// Out: { type: 'progress', label, fraction } (0–1 of the whole request), then { type: 'result', result } or
// { type: 'error', message }.

import { estimateFamilies } from './core/families.js';
import { estimateStyle, opaqueFamilyEstimate } from './core/style.js';
import { restoreChroma } from './core/jpeg.js';

const progress = (label, fraction) => self.postMessage({ type: 'progress', label, fraction });

// families with a radius: what can be fitted
const fittable = prepared => prepared.estimates.filter(e => e.style?.radius).length;

function scan({ image, work, jpeg, tryNoisy }) {
    // the noisy pass takes about as long as the first (beeswarm-boxplot: 19 s and 33 s), so each gets half the bar
    const share = tryNoisy ? 0.5 : 1;
    const prepared = estimateFamilies(work, { jpeg, onProgress: (label, f) => progress(label, share * f) });
    if (!tryNoisy)
        return { prepared, switched: false };
    const noisy = estimateFamilies(restoreChroma(image), {
        jpeg: true,
        onProgress: (label, f) => progress(`Trying noisy color matching: ${label.charAt(0).toLowerCase()}${label.slice(1)}`, 0.5 + 0.5 * f),
    });
    const exact = fittable(prepared);
    const found = fittable(noisy);
    return found > exact ? { prepared: noisy, switched: true, exact, found } : { prepared, switched: false };
}

function opaque({ image, jobs }) {
    return jobs.map(({ seed, opts }, k) => {
        progress('Measuring the dot families again', k / jobs.length);
        return opaqueFamilyEstimate(image, seed, opts);
    });
}

self.onmessage = e => {
    const { op, ...args } = e.data;
    try {
        let result;
        if (op === 'scan')
            result = scan(args);
        else if (op === 'seed')
            result = estimateStyle(args.image, { seed: args.seed, jpeg: args.jpeg, onProgress: f => progress('Measuring the color', f) });
        else if (op === 'opaque')
            result = opaque(args);
        else
            throw new Error(`unknown scan op ${op}`);
        self.postMessage({ type: 'result', result });
    }
    catch (err) {
        self.postMessage({ type: 'error', message: err.message });
    }
};

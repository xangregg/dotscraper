// Module worker that runs the scan for dot families off the main thread: on large images it takes up to a minute,
// and on the main thread the browser called the page unresponsive. It only wraps src/core/ functions.
//
// Messages in: { op, ...args }, one at a time (to stop one, the page terminates the worker):
//   'scan'   { image, jpeg, tryNoisy }: extract.js scanFamilies on the image as decoded. With tryNoisy, the scan is
//            made with exact and with noisy color matching, and the one that finds more families with a radius wins.
//            The result is scanFamilies' without the image as analyzed, which the page makes for itself.
//   'seed'   { image, seed, jpeg }: estimateStyle of a picked color.
//   'opaque' { image, jobs: [{ seed, opts }] }: opaqueFamilyEstimate for each.
// Out: { type: 'progress', label, fraction } (0–1 of the whole request), then { type: 'result', result } or
// { type: 'error', message }.

import { scanFamilies } from './core/extract.js';
import { estimateStyle, opaqueFamilyEstimate } from './core/style.js';

const progress = (label, fraction) => self.postMessage({ type: 'progress', label, fraction });

function scan({ image, jpeg, tryNoisy }) {
    const { work, ...result } = scanFamilies(image, { jpeg, tryNoisy, onProgress: progress });
    return result;
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

// Module worker that runs the fit off the main thread, so the page stays responsive and can show progress.
// Messages in: { input, opts } (see fitDots). Out: { type: 'progress', progress }, then { type: 'done', result } or
// { type: 'error', message }. Stopping is done by terminating the worker; the last progress is the partial result.

import { fitDots } from './core/fit.js';

self.onmessage = e => {
    const { input, opts } = e.data;
    try {
        const result = fitDots(input, opts, progress => self.postMessage({ type: 'progress', progress }));
        self.postMessage({ type: 'done', result });
    }
    catch (err) {
        self.postMessage({ type: 'error', message: err.message });
    }
};

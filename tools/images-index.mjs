// Write testgraphs/index.json, the list of test images the app offers. A static site can't list a directory;
// Python's server (npm run serve) can, and the app uses its listing when it's there, but other servers (WebStorm's
// built-in one) need this file. Git ignores it along with the images.
//
// Usage: node tools/images-index.mjs [--check]   (--check: exit 1 if the index is out of date, write nothing)

import { existsSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

export const IMAGE_DIR = join(dirname(fileURLToPath(import.meta.url)), '..', 'testgraphs');
export const INDEX = join(IMAGE_DIR, 'index.json');
export const IMAGE_FILE = /\.(png|jpe?g|gif|webp)$/i;

/**
 * The index text for the images in a directory: a JSON array of file names, sorted.
 *
 * @param {string} dir
 * @returns {string}
 */
export function indexText(dir) {
    const names = existsSync(dir) ? readdirSync(dir).filter(f => IMAGE_FILE.test(f)).sort() : [];
    return JSON.stringify(names, null, 2) + '\n';
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
    const text = indexText(IMAGE_DIR);
    const current = existsSync(INDEX) ? readFileSync(INDEX, 'utf8') : '';
    const count = JSON.parse(text).length;
    if (process.argv.includes('--check')) {
        if (text !== current) {
            console.error('testgraphs/index.json is out of date: run npm run images:index');
            process.exit(1);
        }
        console.log(`testgraphs/index.json is up to date (${count} images)`);
    }
    else {
        writeFileSync(INDEX, text);
        console.log(`wrote testgraphs/index.json (${count} images)`);
    }
}

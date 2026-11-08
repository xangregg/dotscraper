# Test images

Images here appear in the app's list of test images when it runs from this machine. Git ignores everything in
this directory except this README: the images are third-party figures (journals, The Economist) that are not
cleared for redistribution, so they are never committed or deployed. Anything derived from them (ground truth in
`truth/`, crops, `index.json`) stays here too.

- With `npm run serve` (Python's server, which lists directories), nothing needs indexing: reload the page after
  adding or removing an image.
- With other servers (WebStorm's built-in one), the app reads `index.json` here instead: run
  `npm run images:index` after adding or removing an image. The test suite checks that the index is current.

PNG, JPEG, GIF and WebP files are listed. Pixel values are used exactly as stored in the file: the app decodes
without color-profile conversion, as the Node tools do.

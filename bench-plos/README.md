# bench-plos: figures from PLOS ONE

Real published scatterplots and dot plots, used to benchmark and test dotscraper
(`npm run bench -- --set bench-plos`). Unlike the figures in `testgraphs/` and `bench-nature/`, which stay off the
repository, these come from open-access articles published under the
[Creative Commons Attribution 4.0 license](https://creativecommons.org/licenses/by/4.0/) (CC BY 4.0), which permits
copying and redistribution with credit. Each image is credited to its article below. The figures remain © their
authors under that license; the BSD Zero Clause License of this repository's code does not apply to them.

## What is here

- **`<name>-<DOI>.png`**: a figure, or one panel of a figure, captured from the published article and cropped to the
  plot. The article's DOI is in the file name, with `/` written as `-` (in one name as `:`). Which numbered figure
  each one is was not recorded; the DOI leads to the article.
- **`<name>-<DOI>.csv`**: for some figures, the plotted values in data units, with the axis tick values in header
  lines, taken from the data the article publishes. The benchmark places them in the image through the detected
  ticks.
- **`truth/<name>-<DOI>.json`**: dot centers in image pixels, placed by hand in dotscraper for this project. They are
  not part of the articles.

## Sources

Every article is in *PLOS One*, volume 21 (2026), CC BY 4.0.

**adults1** (image, values, hand truth)\
Álvaro Huerta Ojeda, Emilio Jofré-Saldía, Guillermo Barahona-Fuentes, Sergio Galdames Maliqueo, Maximiliano Bravo
Yapur, Regina de Villa Garduño, Carlos Jorquera-Aguilera, María-Mercedes Yeomans-Cabrera, Estélio H. M. Dantas.
Association between muscle quality index and time to stand up from the prone position in community-dwelling older
adults. *PLOS One* 21(10): e0359858. https://doi.org/10.1371/journal.pone.0359858

**step1** (image, hand truth)\
Yanbing Zhou, Natalie N. McLaurin, Jonas Böcker, Philip D. Chilibeck, Tanja Taivassalo, Sean C. Forbes, Dominik
Pesta, Jörn Rittweger, Hirofumi Tanaka.
Age-related slowing of sensorimotor reaction time in masters track and field athletes.
*PLOS One* 21(11): e0360641. https://doi.org/10.1371/journal.pone.0360641

**boxdots1** (image, hand truth)\
Michael Williams, Aditya Chakraborty, Brian Main, Liang Li, Ya-Chun Chan, Xiaohang Wang, Nico Verbeeck, Liam Spiers,
Olivia Routten, Theodore Chin, Kaitlyn Bejar, Dean Troyer.
Metabolomic profiling of renal cell carcinoma to identify prognostic markers of metastasis.
*PLOS One* 21(11): e0361404. https://doi.org/10.1371/journal.pone.0361404

**grader1** (image, values, hand truth)\
Evelyn Hughes, Rohan Alexander.
Evaluating AI performance on end-to-end data science projects.
*PLOS One* 21(11): e0361887. https://doi.org/10.1371/journal.pone.0361887

**score1**, **score2** (images, hand truth)\
Ziqian Bi, Zhantong Ma, Junmin Huang, Junfeng Hao, Lu Chen, Hongying Luo, Zhenyu Yu, Tianyang Wang, Yue Ma, Aiping
Tan.
OpenMedEval: Benchmarking 172 open-source LLMs on 326K medical questions across 93 datasets.
*PLOS One* 21(11): e0361996. https://doi.org/10.1371/journal.pone.0361996

**landing1** (image, values)\
Huseyin Celik, Pinar Arpinar-Avsar.
Dual-task interference in the drop jump: Frequentist, Bayesian, and waveform analysis of vertical ground reaction
force. *PLOS One* 21(11): e0362399. https://doi.org/10.1371/journal.pone.0362399

**microplastics1** (image, hand truth)\
Samia Iftikhar, Hassien M. Alnashiri, Mehnaz Akhtar, Zuniar Ahsan, Muhammad Latif, A. El-Shabasy, Mohamed Mostafa
Soliman.
Dose-dependent hepatorenal toxicity of polystyrene microplastics in BALB/c mice.
*PLOS One* 21(11): e0362569. https://doi.org/10.1371/journal.pone.0362569

**droplet1** (image, values)\
Lauren Tereshko, Matthew Ryals, Cullen Mason, Romi Admanit, Jake Gagnon.
A Poisson-multinomial model for quantifying gene therapy vector integrity using quadplex digital PCR.
*PLOS One* 21(11): e0362653. https://doi.org/10.1371/journal.pone.0362653

**myeloid1** (image, hand truth)\
Wei Wei, Jing Liu, Ying Zhou, Rui Zhao, LinGang Zhang.
Paired neutrophil association and ex vivo perturbation sensitivity define a bounded myeloid activation model in ARDS.
*PLOS One* 21(11): e0362747. https://doi.org/10.1371/journal.pone.0362747

Titles, authors and licenses are as registered with Crossref for each DOI (looked up 2026-11-20).

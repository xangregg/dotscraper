# bench-pmc: figures from articles in PubMed Central

Real published scatterplots and dot plots, used to benchmark and test Dot Scraper
(`npm run bench -- --set bench-pmc`). They come from open-access articles found through PubMed Central, chosen
among those published under the
[Creative Commons Attribution 4.0 license](https://creativecommons.org/licenses/by/4.0/) (CC BY 4.0), which permits
copying and redistribution with credit. Each image is credited to its article below. The
figures remain © their authors under that license; the BSD Zero Clause License of this repository's code does not
apply to them.

## What is here

- **`<name>-<DOI>.png`** (or `.jpg`): a figure, or one panel of a figure, captured from the published article (most
  of them from the raster image in its PDF) and cropped to the plot. The article's DOI is in the file name, with `/`
  written as `-`. Which numbered figure each one is was not recorded, except where an entry says; the DOI leads to
  the article.
- **`truth/<name>-<DOI>.json`**: dot centers in image pixels, placed by hand in Dot Scraper for this project. They
  are not part of the articles.
- **`truth-original/`**: the first hand-placed truth of potato and shap, kept from before those files were
  rewritten as one family each (PLAN.md).

## Sources

Every entry is CC BY 4.0, as the article itself states it.

**aerobic** (image, hand truth)\
Diego Tec-Campos, Juan D. Tibocha-Bonilla, Celina Jiang, Anurag Passi, Deepan Thiruppathy, Cristal Zuñiga, Camila
Posadas, Alejandro Zepeda, Karsten Zengler.
A genome-scale metabolic model for the denitrifying bacterium Thauera sp. MZ1T accurately predicts degradation of
pollutants and production of polymers.
*PLOS Computational Biology* 21(1): e1012736 (2025). https://doi.org/10.1371/journal.pcbi.1012736

**amino** (image, hand truth)\
Rechel A. Geiger, Damini Khera, Jeannette L. Tenthorey, Georg Kochs, Laura Graf, Michael Emerman, Harmit S. Malik.
Heterozygous and generalist MxA super-restrictors overcome breadth-specificity tradeoffs in antiviral restriction.
*bioRxiv* 2024.10.10.617484 (preprint, 2024). https://doi.org/10.1101/2024.10.10.617484\
The image is from version 1 of the preprint, posted under CC BY 4.0.

**bread1**, **bread2** (images, hand truth)\
Kayzad Nilgiriwala, Marie-Sylvianne Rabodoarivelo, Michael B. Hall, Grishma Patel, Ayan Mandal, Shefali Mishra,
Fanantenana Randria Andrianomanana, Kate Dingle, Gillian Rodger, Sophie George, Derrick W. Crook, Sarah Hoosdally,
Nerges Mistry, Niaina Rakotosamimanana, Zamin Iqbal, Simon Grandjean Lapierre, Timothy M. Walker.
Genomic Sequencing from Sputum for Tuberculosis Disease Diagnosis, Lineage Determination, and Drug Susceptibility
Prediction.
*Journal of Clinical Microbiology* 61(3): e01578-22 (2023). https://doi.org/10.1128/jcm.01578-22

**culicine** (image, hand truth)\
Vena Joseph, Alice Sutcliffe, Laura Leite, Cyrille Czeher, Thomas Druetz, Eric Rogier, Thomas P. Eisele, Jean
Frantz Lemoine, Michelle Chang, Daniel Impoinvil, Ruth A. Ashton.
Entomological Profiles of Households in Plasmodium falciparum Case Foci and Comparison Areas in Grand’Anse, Haiti.
*The American Journal of Tropical Medicine and Hygiene* 112(5): 975–986 (2025).
https://doi.org/10.4269/ajtmh.24-0478

**cycle** (image, hand truth)\
Jonathan Altamirano, Prasanthi Govindarajan, Andra L. Blomkalns, Sean Leary, India Robinson, Leanne X. Chun, Nuzhat
J. Shaikh, Makeda L. Robinson, Marcela Lopez, Grace K-Y Tam, Yuan J. Carrington, Monique B. De Araujo, Katharine S.
Walter, Jason R. Andrews, Julianne Burns, Catherine Hogan, Benjamin A. Pinsky, Yvonne Maldonado.
Natural history of shedding and household transmission of severe acute respiratory syndrome coronavirus 2 using
intensive high-resolution sampling.
*PLOS ONE* 19(7): e0305300 (2024). https://doi.org/10.1371/journal.pone.0305300\
The image is Figure 2.

**funnel** (image, hand truth)\
Xiutao Yang, Qingxia Gao, Zhaofang Xi, Pengfei Zhao, Junlong Zhao.
Prevalence and Heterogeneity of Swine Influenza Virus in China From 2010 to 2025: A Systematic Review and
Meta‐Analysis.
*Transboundary and Emerging Diseases* 2026: 1096796 (2026). https://doi.org/10.1155/tbed/1096796

**gc** (image, hand truth)\
Keyao Zhang, Jing Song, Junhui Lu, Lu Zhao, Weian Deng, Delong Guan, Benyong Mao.
Phylogenetics and Evolutionary Dynamics of Yunnan Acrididae Grasshoppers Inferred from 17 New Mitochondrial
Genomes.
*Insects* 16(2): 151 (2025). https://doi.org/10.3390/insects16020151

**grapes** (image, hand truth)\
Esther Hernández-Montes, Yun Zhang, Ben-Min Chang, Nataliya Shcherbatyuk, Markus Keller.
Soft, Sweet, and Colorful: Stratified Sampling Reveals Sequence of Events at the Onset of Grape Ripening.
*American Journal of Enology and Viticulture* 72(2): 137–151 (2021). https://doi.org/10.5344/ajev.2020.20050\
Not in PubMed Central; the license is as the publisher's page states it.

**healer** (image, hand truth)\
Yung-An Tsou, Chien-Hao Huang, Yu-Jen Chou, Hsueh-Hsin Kao, Jui-Kun Chiang, Yee-Hsin Kao.
The Impact of a Tongue Training Device on Tongue Muscle Strength in Patients with Obstructive Sleep Apnea After
Modified Uvulopalatopharyngoplasty: A Pilot Study.
*Medicina* 61(3): 511 (2025). https://doi.org/10.3390/medicina61030511

**latency** (image, hand truth)\
Álvaro Cancela, Pablo Arias, Beatriz Rodríguez‐Romero, Marcelo Chouza‐Insua, Javier Cudeiro.
Acute effects of a single neurodynamic mobilization session on range of motion and H‐reflex in asymptomatic young
subjects: A controlled study.
*Physiological Reports* 11(12): e15748 (2023). https://doi.org/10.14814/phy2.15748

**mortality** (image, hand truth)\
Rania Boukhari, Dominic E. Ponton, Maikel Rosabal, Kristin K. Mueller, Marc Amyot.
Assessing the Acute and Chronic Effects of Palladium on Daphnia magna and the Influence of Natural Organic Matter.
*Journal of Applied Toxicology* 45(11): 2363–2377 (2025). https://doi.org/10.1002/jat.4854

**oxygen** (image, hand truth)\
Armelle Bridier, Monisha Shcherbakova, Atsushi Kawaguchi, Nancy Poirier, Carla Said, Rita Noumeir, Philippe Jouvet.
Hemodynamic assessment in children after cardiac surgery: A pilot study on the value of infrared thermography.
*Frontiers in Pediatrics* 11: 1083962 (2023). https://doi.org/10.3389/fped.2023.1083962

**potato** (image, hand truth)\
Stine Le Tougaard, Augusta Szameitat, Pauline Møs, Søren Husted.
Leaf age and light stress affect the ability to diagnose P status in field grown potatoes.
*Frontiers in Plant Science* 14: 1100318 (2023). https://doi.org/10.3389/fpls.2023.1100318

**reward** (image, hand truth)\
Sydney E. Ashton, Paul Sharalla, Naru Kang, Adam T. Brockett, Margaret M. McCarthy, Matthew R. Roesch.
Distinct Action Signals by Subregions in the Nucleus Accumbens during STOP–Change Performance.
*The Journal of Neuroscience* 44(29): e0020242024 (2024). https://doi.org/10.1523/JNEUROSCI.0020-24.2024\
Crossref records CC BY-NC-SA 4.0 for this DOI; the article itself states CC BY 4.0.

**rvat** (image, hand truth)\
Shichao Li, Zhouyan Liao, Kangwen He, Yaqi Shen, Shan Hu, Zhen Li.
Association of sex-specific abdominal adipose tissue with WHO/ISUP grade in clear cell renal cell carcinoma.
*Insights into Imaging* 14(1): 194 (2023). https://doi.org/10.1186/s13244-023-01494-7

**shannon** (image, hand truth)\
Yunyi Tian, Wanglong Gou, Yue Ma, Menglei Shuai, Xinxiu Liang, Yuanqing Fu, Ju-Sheng Zheng.
The Short-Term Variation of Human Gut Mycobiome in Response to Dietary Intervention of Different Macronutrient
Distributions.
*Nutrients* 15(9): 2152 (2023). https://doi.org/10.3390/nu15092152

**shap** (image, hand truth)\
Olatomiwa O. Bifarin, Facundo M. Fernández.
Automated Machine Learning and Explainable AI (AutoML-XAI) for Metabolomics: Improving Cancer Diagnostics.
*Journal of the American Society for Mass Spectrometry* 35(6): 1089–1100 (2024).
https://doi.org/10.1021/jasms.3c00403

**tegner** (image, hand truth)\
Yannick J. Ehmann, Thekla Esser, Amr Seyam, Marco-Christopher Rupp, Julian Mehl, Sebastian Siebenlist, Andreas B.
Imhoff, Philipp Minzlaff.
Low postoperative complication rate with high survival rate and good clinical outcome 9 years after autologous
chondrocyte transplantation of the knee joint.
*Archives of Orthopaedic and Trauma Surgery* 143(5): 2665–2674 (2023). https://doi.org/10.1007/s00402-022-04611-1


Titles and authors are as registered with Crossref for each DOI; each license was read from the article's own
statement (in its PubMed Central full text, or on the publisher's or bioRxiv's page), looked up 2026-11-26.

One figure found the same way, reading (https://doi.org/10.1523/JNEUROSCI.1153-24.2024), is not here: the
article's license is not CC BY (its own statement names the Society for Neuroscience's exclusive license, and
Crossref records CC BY-NC-SA 4.0). Its image and truth stay local; `.gitignore` lists them.

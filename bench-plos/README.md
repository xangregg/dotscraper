# bench-plos: figures from PLOS ONE

Real published scatterplots and dot plots, used to benchmark and test dotscraper
(`npm run bench -- --set bench-plos`). Unlike the figures in `testgraphs/` and `bench-nature/`, which stay off the
repository, these come from open-access articles published under the
[Creative Commons Attribution 4.0 license](https://creativecommons.org/licenses/by/4.0/) (CC BY 4.0), which permits
copying and redistribution with credit (one, funnel, under the CC0 1.0 public domain dedication). Each image is
credited to its article below. The figures remain © their authors under those licenses; the BSD Zero Clause License
of this repository's code does not apply to them.

## What is here

- **`<name>-<DOI>.png`**: a figure, or one panel of a figure, captured from the published article and cropped to the
  plot. The article's DOI is in the file name, with `/` written as `-`. Which numbered figure each one is was not
  recorded; the DOI leads to the article.
- **`<name>-<DOI>.csv`**: for some figures, the plotted values in data units, with the axis tick values in header
  lines, taken from the data the article publishes. The benchmark places them in the image through the detected
  ticks.
- **`truth/<name>-<DOI>.json`**: dot centers in image pixels, placed by hand in dotscraper for this project. They are
  not part of the articles.

## Sources

Every article is in *PLOS One*, volume 21 (2026), CC BY 4.0 unless its entry says otherwise.

**cattle** (image, hand truth)\
Simon Lanigan, Martijn F. L. Derks, Anna M. Johansson, Martin Johnsson.
Variant intolerance scores in cattle.
*PLOS One* 21(11): e0355762. https://doi.org/10.1371/journal.pone.0355762

**lung-tumor** (image, hand truth)\
Qinguang Zou, Yinghao Chang, Mingjiang Qu, Xuejuan Gong, Chunyi Jia.
A public-data spatial transcriptomic framework for invasive lung adenocarcinoma tumor ecosystems and prognosis.
*PLOS One* 21(11): e0357864. https://doi.org/10.1371/journal.pone.0357864

**mpo** (image, hand truth)\
Joao Henrique Falk Neto, Kelvin E. Jones, Aidan K. Comeau, Normand Boulé, Michael D. Kennedy.
The physiological demands of intermittent endurance exercise performed at different intensities within the
anaerobic power reserve: Implications for fatigue development and performance.
*PLOS One* 21(11): e0359318. https://doi.org/10.1371/journal.pone.0359318

**richness** (image, hand truth)\
Nadia B. Páez-Rosales, Sarah M. Ravoth, Pierre Rogy, Pablo Sandoval-Acuña, Natalie Elena Westwood, Rachel R. Leong,
Diane S. Srivastava.
A step-by-step protocol for surveying bromeliad micro-ecosystems and their associated invertebrate communities.
*PLOS One* 21(11): e0359333. https://doi.org/10.1371/journal.pone.0359333

**stiffness** (image, hand truth)\
Cain-En Lim, Ram Prasad Sinnanaidu, Xin Tong Ng, Jin Ying Teh, Farahani Khamis, Anushya Vijayananthan, Wah-Kheong
Chan.
Comparing attenuation parameter and liver stiffness measurement using Hepatus and Fibroscan for the evaluation of
steatosis and fibrosis in patients with chronic liver disease: A prospective study.
*PLOS One* 21(11): e0359831. https://doi.org/10.1371/journal.pone.0359831

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

**gait** (image, hand truth)\
Rebecca Ahrns, Ryan T. Roemmich, Jan Stenum, Kendra M. Cherry-Allen.
In-clinic versus at-home gait kinematics in adults with and without stroke: Associations across environments.
*PLOS One* 21(11): e0361481. https://doi.org/10.1371/journal.pone.0361481

**crimson** (image, hand truth)\
Gizelle J. Lionel, Beth Binnington, Raymond W. Wong, Jiazhen Jin, Alan Cochrane, Donald R. Branch.
A cell line model for the study of CD4-negative HIV-1 infection and latent virus reservoirs.
*PLOS One* 21(11): e0361655. https://doi.org/10.1371/journal.pone.0361655

**synthetic** (image, hand truth)\
Jennifer L. Mallek, Anthony Mannion, Amy Xiao, Judilee D. Osborn, Jessie H. Hendricks, Lynn Martin Kerr, Meghan E.
Ramsey.
Experimental and computational insights for broad, metagenomic next-generation sequencing-based biodetection in
complex environments.
*PLOS One* 21(11): e0361775. https://doi.org/10.1371/journal.pone.0361775

**grader1** (image, values, hand truth)\
Evelyn Hughes, Rohan Alexander.
Evaluating AI performance on end-to-end data science projects.
*PLOS One* 21(11): e0361887. https://doi.org/10.1371/journal.pone.0361887

**viability** (image, hand truth)\
Yuki Uemichi, Shuhei Nakao, Shuji Nagano, Miyuki Mabuchi, Yuya Kawabata, Yoshihiro Uesawa, Tadashi Shimizu.
Rigorous nested cross-validation and external generalization-gap analysis of a machine-learning classifier for
U-251 glioblastoma cytotoxicity.
*PLOS One* 21(11): e0361912. https://doi.org/10.1371/journal.pone.0361912

**alloys** (image, hand truth)\
Abhijit Saha, Rajeev Ranjan, Alemu Workie Kebede, Himadri Majumder.
Prediction of MMAW performance on nanostructured hardfaced alloys using general regression neural network: A case
study.
*PLOS One* 21(11): e0361981. https://doi.org/10.1371/journal.pone.0361981

**score1**, **score2** (images, hand truth)\
Ziqian Bi, Zhantong Ma, Junmin Huang, Junfeng Hao, Lu Chen, Hongying Luo, Zhenyu Yu, Tianyang Wang, Yue Ma, Aiping
Tan.
OpenMedEval: Benchmarking 172 open-source LLMs on 326K medical questions across 93 datasets.
*PLOS One* 21(11): e0361996. https://doi.org/10.1371/journal.pone.0361996

**landing1** (image, values)\
Huseyin Celik, Pinar Arpinar-Avsar.
Dual-task interference in the drop jump: Frequentist, Bayesian, and waveform analysis of vertical ground reaction
force. *PLOS One* 21(11): e0362399. https://doi.org/10.1371/journal.pone.0362399

**expression** (image, hand truth)\
Mubeen Tabish Nasim, Narmeen Adnan Rana, Urooj Mukhtar, Muhammad Sheryar, Ghulam Rabia, Shanza Sehar, Muhammad
Jadoon Khan, Muhammad Jawad Khan, Hassaan Mehboob Awan.
Downregulation of miR-548c-3p potentially drives IL-17 pathway dysregulation in oral squamous cell carcinoma.
*PLOS One* 21(11): e0362467. https://doi.org/10.1371/journal.pone.0362467

**corneal** (image, hand truth)\
Tetsuya Muto, Shoichiro Kusuda, Masaaki Sakamoto, Shigeki Machida, Shinichiro Imaizumi, Tetsuju Sekiryu.
Predictive factors for risk of chronic evolution in Vogt–Koyanagi–Harada disease at 1 month after pulse steroid
therapy.
*PLOS One* 21(11): e0362558. https://doi.org/10.1371/journal.pone.0362558

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

**tsne** (image, hand truth)\
Md Rayhan Khan, Masrafe Bin Hannan Siam, Md Fazla Elahe, Md Shohel Arman, S. M. Hasan Mahmud.
Effects of latent feature construction and fusion strategies in deep multi-omics classification.
*PLOS One* 21(11): e0362839. https://doi.org/10.1371/journal.pone.0362839

**biomarker** (image, hand truth)\
Daisuke Ueda, Ryou-u Takahashi, Yasuhiro Tsutani, Yukie Nishiyama, Yumiko Koi, Yuta Ibuki, Yoshihiro Miyata,
Tomoyuki Akita, Junko Tanaka, Morihito Okada, Hidetoshi Tahara.
Development of a diagnostic biomarker panel for early-stage lung adenocarcinoma using serum-derived small RNAs.
*PLOS One* 21(11): e0362879. https://doi.org/10.1371/journal.pone.0362879

**funnel** (image)\
Zekarias Markos, Esubalew Muluneh Aligaz, Sitotaw Tesfa Zegeye, Rahel Dinkneh, Temesgen Mamo, Sintayehu Samuel
Lorato.
Prevalence of intimate partner violence against women in East Africa: A systematic review and meta-analysis.
*PLOS One* 21(11): e0362883. https://doi.org/10.1371/journal.pone.0362883\
CC0 1.0 (public domain dedication).

Titles, authors and licenses are as registered with Crossref for each DOI (looked up 2026-11-20, and 2026-11-22 for
the fifteen added then).

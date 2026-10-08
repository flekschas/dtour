# dtour: a steerable *tour de vis* through high-dimensional data

Fritz Lekschas (Ridge AI) and Nezar Abdennur (UMass Chan Medical School).
Preprint: [arXiv:2605.04306](https://arxiv.org/abs/2605.04306)

This is the text of the paper converted to Markdown for agents. Citations are shortened
to first author and year; the full references are in the preprint. Figures are described
in words. Bold marks key takeaways and was added for this version. The usage scenarios
interpret specific datasets; when applying their reasoning elsewhere, treat a disagreement
between views as a lead to check, not proof.

## Contents

- [Abstract](#abstract)
- [1 Introduction](#1-introduction)
- [2 Related work](#2-related-work)
- [3 The dtour method](#3-the-dtour-method): user interface, tour interpolation, tour strategies, rendering
- [4 Usage scenarios](#4-usage-scenarios): Fashion-MNIST, CyTOF immune cells, single-cell RNA-seq, arXiv embedding models
- [5 Conclusion](#5-conclusion)

**Figure 1 (teaser):** The dtour interface for exploring high-dimensional data along a
tour of keyframe projections. dtour unifies three modes of increasing steerability.
(1) A central 2D scatter with a gallery of projection previews gives an overview.
(2) The user advances the central scatter along a cyclical guided tour by clicking a
preview (2a), scrubbing the circular slider (2b), or scrolling, which smoothly
transitions to another projection and builds intuition for the high-dimensional
manifold. (3) To study details, the user can smoothly switch to manual axis manipulation
(3a, dragging an axis) and highlight points by label or lasso selection (\*).

## Abstract

Understanding high-dimensional data requires projecting it into lower-dimensional spaces,
but any single projection inevitably loses information or introduces distortions. Tours
address this limitation through animation of 2D projection sequences, yet existing tools
present tradeoffs in the freedom and steerability of projection traversal, providing
little to no ability to move between expert-guided paths and unrestrained exploration. We
present dtour, a tour interface that combines static projection previews, reversible
scrubbing along continuous geodesic projection paths, manual projection manipulation, and
a wandering grand tour, all within a single progressive exploration interface. dtour
scales to millions of points via GPU-accelerated rendering, runs in any modern browser,
and integrates with both Python and JavaScript ecosystems. We demonstrate dtour on text,
image, and single-cell data for two usage scenarios: gradually revealing structure in
high-dimensional data and validating non-linear dimensionality reduction outputs.

## 1 Introduction

Understanding high-dimensional data is fundamentally challenging because human
perception is limited to three dimensions and visual exploration necessarily involves
projecting data into lower-dimensional spaces. Linear dimensionality reduction (DR)
methods like Principal Component Analysis (PCA) preserve global structure faithfully,
but any single projection hides all structure orthogonal to the chosen projection plane.
By contrast, non-linear neighbor-based approaches like t-Distributed Stochastic Neighbor
Embedding (t-SNE; van der Maaten 2008) and Uniform Manifold Approximation and Projection
(UMAP; McInnes 2018) attempt to capture manifold structure in a single lower-dimensional
space, but inevitably introduce distortions that can misrepresent cluster structure and
neighborhood relationships. Despite many debates about the usefulness and faithfulness
of non-linear DR methods (e.g. Chari 2023; Lause 2024), such methods are widely used and
undoubtedly useful if interpreted with care (Wattenberg 2016; Kobak 2019; Becht 2019).

Presenting more than a single 2D or 3D projection can be beneficial for interpreting the
outputs of both linear and non-linear DR methods. Approaches such as scatter plot
matrices (Chambers 1983) and small multiples (Tufte 1983) lay out a small set of fixed
projections side by side. Alternative approaches, under the broad concept of a *tour*
(Asimov 1985; Buja 2005), instead present projections sequentially as a path through
projection space that is often visualized as an animation (Swayne 2001; Wickham 2011).
Tour variants differ in how the path is generated: *grand* tours traverse projection
space by a random walk (Asimov 1985; Buja 1986), *guided* tours (Cook 1995) select
specific target projections by optimizing an interestingness criterion such as cluster
separation or outlier presence, and *manual* tours (Cook 1997) give control of the
projection to the user (Li 2020).

> Throughout, "guided tour" refers to any precomputed sequence of keyframe projections
> that the user traverses along a fixed path, regardless of how the keyframes were
> selected.

Approaches for touring multiple projections lie along a spectrum of **freedom of
traversal** determined by the constraints on the path of projections visited and differ
in the degree of user **steerability** of tour progression. At one end, grids of static
projections are directly comparable at a glance but require constant shifts of focus to
integrate information and scale poorly with the number of projections shown. In the
middle, animated tours (Swayne 2001; Wickham 2011) eliminate focus shifts and help keep
track of correspondences by morphing between projections in a single view. Playback lets
the user pace the traversal along a fixed precomputed path, but only a single projection
is visible at a time. At the other end, manual tours give full control over the
projection, but reaching an informative view by hand is slow and cognitively demanding,
as the user must navigate projection space without guidance. These trade-offs are
unavoidable within any single tour mode, but they can be reconciled by an interface that
lets the user move smoothly across the spectrum itself.

Here we present dtour, a tour interface for high-dimensional data designed to provide
frictionless control over the freedom and steerability of projection traversal. Given
data with four or more dimensions, dtour opens with a central 2D scatter surrounded by a
gallery of *keyframe* projection previews (Fig. 1.1). The user advances the central
scatter between keyframes by clicking previews or smoothly transitions along the path
connecting them as a guided cyclical tour via scrubbing or scrolling (Fig. 1.2). When
more precise control is needed, manual manipulation (Fig. 1.3) lets the user directly
control the influence of individual axes on the central projection, allowing user-driven
excursions from the primary tour. Complementing these modes, a grand tour animates a
random walk through projection space for hands-off serendipitous exploration. All
transitions, within a tour mode and between modes, are smoothly interpolated to preserve
point identity and spatial context. Combined with lasso selection and label-based
highlighting, dtour provides the fluid, progressive control over traversal complexity
needed to navigate high-dimensional data effectively.

We implemented dtour as a general-purpose tool that scales to data with millions of
points through GPU-accelerated rendering. dtour is agnostic to how keyframe projections
are produced: they can come from a projection pursuit, hyperparameter sweeps, a time
series, or any other source that yields a sequence of 2D projections with one-to-one
point correspondence. We demonstrate two usage scenarios: (1) revealing structure in
high-dimensional data and (2) validating non-linear dimensionality reduction outputs
with text, image, and single-cell datasets. dtour runs in any modern browser
(https://dtour.dev), is available as a widget for Jupyter and Marimo notebooks, and can
be embedded in React applications or built upon via its rendering engine. The source
code is available at https://github.com/flekschas/dtour.

## 2 Related work

### 2.1 Dimensionality reduction

Linear dimensionality reduction methods such as PCA project high-dimensional data onto
subspaces that preserve global properties like variance, yielding interpretable axes but
capturing only linear structure. Non-linear techniques, notably t-SNE (van der Maaten
2008) and UMAP (McInnes 2018), aim to capture manifold structure in a single layout, but
inevitably introduce distortions (Chari 2023). Böhm et al. (2022) show that these and
other neighbor-embedding methods lie on a spectrum of attraction between neighbors and
repulsion between all points, where the balance between forces governs the trade-off
between preserving continuous structure and separating clusters. The MDE framework
(Agrawal 2021) further unifies linear and non-linear objectives into a single
optimization formulation, enabling systematic comparison across the DR spectrum. Whether
linear or non-linear, a fixed 2D view cannot fully represent high-dimensional data,
motivating the use of tours to examine data from multiple perspectives.

### 2.2 Visualization of embeddings

A growing ecosystem of tools supports the interactive exploration of low-dimensional
embedding projections. The Embedding Projector was an early web-based system for
browsing embeddings with nearest-neighbor search and support for PCA, t-SNE, and custom
projections. Scalability has since been a central concern: tools such as
regl-scatterplot (Lekschas 2023), Jupyter Scatter (Lekschas 2024), WizMap (Wang 2023),
DataMapPlot (McInnes 2024), and Embedding Atlas (Ren 2025) can render millions of points
with interactive visual encodings, selections, and pan-and-zoom navigation. Emblaze
(Sivaraman 2022) and Comparative Embedding Visualization (Manz 2024) support comparison
across multiple embedding spaces, but are limited to pairwise views. All of these
systems present one or two static 2D projections; none offer smooth, steerable traversal
through projection space or across a sequence of embeddings, which dtour brings to the
embedding visualization ecosystem.

### 2.3 Tour methods

Animating sequences of low-dimensional projections dates back to Asimov's grand tour
(Asimov 1985), a smooth random walk through all possible 2D projections. The
mathematical foundations for these dynamic projections, including geodesic interpolation
on the Stiefel manifold and the general framework of *d*-dimensional projections from
*p*-dimensional space, were formalized by Buja et al. (2005). Cook et al. (1995)
introduced *guided tours*, which replace random target selection with projection pursuit
optimization, steering the animation toward projections that maximize a criterion of
interestingness (e.g., holes, central mass, or linear discriminant indices). Additional
variants include local tours, which rock near a given projection, and the manual tour
(Cook 1997) for direct variable control. A comprehensive review of tour methods is given
by Lee et al. (2022).

### 2.4 Tour software and applications

The primary software ecosystem for tours is the R package `tourr` (Wickham 2011), which
implements grand, guided, local, manual, and other tour types with multiple display
methods. Earlier interactive systems include GGobi (Swayne 2003) and its predecessor
XGobi, which pioneered linked brushing and direct manipulation of projections. Recent
packages extend this ecosystem with refined manual controls (spinifex; Spyrison 2020),
linked DR diagnostics (liminal; Lee 2020), portable HTML rendering (detourr; Hart 2022),
and Langevin-dynamics-based smooth paths (langevitour; Harrison 2023). In an interactive
article, Li et al. (2020) present grand and manual tours with steerable axes within a
single interface to visualize neural-network activations.

Despite this rich landscape, existing tour tools limit accessibility as the analyst must
choose upfront the most suitable tour mode. dtour addresses this gap with a single
interface that unifies overview, guided, manual, and grand tour modes across the
steerability spectrum, while scaling to million-point datasets.

## 3 The dtour method

### 3.1 User interface

Inspired by Shneiderman's visual information-seeking mantra (Shneiderman 2003), dtour
supports progressive exploration of high-dimensional data through a unified interface
(Fig. 1): an overview consisting of a central 2D scatter surrounded by a gallery of
keyframe projection previews, a guided tour that smoothly animates through the keyframe
sequence, and a manual tour that lets the user manipulate the projection by dragging
dimension axes. Complementing these modes, a grand tour mode enables random rotations
through projection space as a continuous playback. Users can transition fluidly between
modes at any time. The central scatter serves as a fixed reference point across all
modes, and all projection changes are smoothly interpolated to preserve object
constancy, letting viewers track points across views rather than re-identifying them
after each transition (Robertson 1993; Rodrigues 2024).

**Keyframe gallery.** On launch, a gallery of previews (Fig. 1.1) surrounds the central
scatter, one per keyframe, each showing the data in that keyframe's projection. Clicking
a preview (Fig. 1.2a) advances the central view to that keyframe. Tours can specify
feature loadings, shown as text beneath each preview indicating the top contributing
dimensions. Gallery previews are arranged around the circular tour slider (described
below), such that the gallery acts as a lookahead and an orientation device.

**Guided tour.** A circular tour slider (Fig. 1.2b) controls the current position along
the arc-length-parameterized keyframe path (Section 3.2). Users can scrub the slider,
scroll the mouse wheel, or press play for animated playback to advance the tour. Tick
marks at keyframe positions serve as navigational landmarks, and the width of the
slider's ring segments encodes geodesic distances between consecutive keyframes: thin
segments indicate stretched regions of projection space, thick segments indicate
compressed regions. The tour runs as a closed loop so that continuous forward or
backward traversal never encounters a discontinuity.

**Manual tour.** In manual mode dimension axes appear as draggable handles (Fig. 1.3)
overlaid on the scatter plot, one per data dimension. Each handle's projected direction
and length encode that dimension's current contribution to the projection basis,
doubling as a control surface and a feature-loading legend. Dragging a handle (Fig.
1.3a) specifies a new target direction for its variable and the remaining basis is
re-orthonormalized to preserve a valid tour frame. Additionally, holding `Shift` while
dragging rotates the view about a temporary third axis (the residual principal
component) to help build spatial intuition. Together, these let the user isolate
individual dimensions' effects.

**Color encoding and point selection.** Many datasets include labels or non-embedded
dimensions that aid interpretation. These can be mapped to point color in dtour via
continuous, 2D, or categorical encodings. Lasso and label-based selection let users
isolate a subset of points and track them across projections, enabling a
*select-then-explore* workflow: select points of interest during guided playback, then
switch to manual mode to investigate which dimensions distinguish them.

### 3.2 Tour interpolation

A tour is defined by a cyclic sequence of *keyframe* projections, each represented as a
*p* × 2 orthonormal basis matrix **F**ᵢ mapping *p*-dimensional data to 2D. Smoothly
animating between keyframes requires a meaningful distance on the space of bases and an
interpolation scheme that preserves orthonormality.

**Geodesic distance.** The distance between two 2D subspaces spanned by bases **F**ₐ and
**F**_z is measured via the principal angles τ₀, τ₁ obtained from the singular value
decomposition of **F**ₐᵀ**F**_z:

```
d(F_a, F_z) = sqrt(τ₀² + τ₁²),   τᵢ = arccos(σᵢ)
```

where σ₀, σ₁ are the singular values clamped to [−1, 1]. This distance corresponds to
the geodesic on the Grassmannian manifold of 2D subspaces (Buja 2005). For the 2 × 2
case, dtour computes the SVD analytically.

**Catmull-Rom spline with re-orthonormalization.** Given four consecutive bases
**P**₀, …, **P**₃, the interpolated basis at parameter *t* ∈ [0, 1] between **P**₁ and
**P**₂ is the standard cubic Catmull-Rom (Catmull 1974) blend applied element-wise,
followed by Gram-Schmidt orthonormalization. The spline passes exactly through each
keyframe with C¹-continuous tangents, avoiding the velocity discontinuities that arise
with piecewise geodesic interpolation on the Grassmannian, while Gram-Schmidt guarantees
orthonormality at every intermediate step.

**Arc-length parameterization.** To ensure perceptually uniform playback speed, dtour
precomputes a cumulative arc-length table by sampling each spline segment at eight
interior points and summing geodesic distances between consecutive samples. At runtime,
a binary search maps *t* ∈ [0, 1] to the correct segment and local parameter in
O(log *n*) time, so that scrubbing the circular slider produces constant angular
velocity through projection space.

### 3.3 Tour strategies

dtour accepts any sequence of *p* × 2 orthonormal basis matrices as a tour. To
demonstrate this generality, we implement four strategies targeting different analytical
tasks, organized into two families.

**Hyperdimensional tours.** These tours explore a single high-dimensional data space and
focus on "what the data looks like from different angles". We implemented two tours
based on spectral decompositions. The *little tour* (Wickham 2011) cycles through
projections along successive pairs of components (e.g., PC1–PC2, PC2–PC3, etc.)
providing an accessible starting point for any dataset. The *le tour* uses Laplacian
Eigenmaps (Belkin 2003), a spectral manifold learning technique, with a cumulative
circular basis construction that progressively adds eigenvectors at uniform angular
offsets. See the supplementary material for more details.

**Sequential embedding tours.** These tours allow for comparison across embedding
methods, hyperparameters, and models rather than across directions in a single
manifold. They address the question of "how the picture changes when the lens changes"
and are constructed from sequences of aligned 2D embeddings of the same or one-to-one
corresponding data points which serve as keyframes. The embeddings are concatenated so
the guided tour interpolates smoothly between them. **Intermediate frames are
geometrically valid but should not be interpreted as views of latent structure as in a
hyperdimensional tour.** The *sequential tour* is the general-purpose primitive. For each
frame, it runs a DR method, warm-started with the previous frame's embedding,
Procrustes-aligns the result, and stacks the sequence into a single tour. As a special
case, we implemented an *attraction-repulsion tour* that sweeps the exaggeration
hyperparameter of Böhm et al. (2022) to traverse the spectrum of embeddings from the
same neighbor graph.

### 3.4 Rendering and implementation

dtour is available as a TypeScript renderer, a React component, or as a portable
anywidget for Python-based notebooks. Rendering is offloaded to a WebGPU/WebGL worker
with an OffscreenCanvas, keeping the UI thread free. A separate data worker streams
Parquet columns directly to the GPU worker. On an Apple M1 Max MacBook, this
architecture sustains smooth playback: ≳60 FPS at ≤5M points, 40 FPS at 10M, and remains
usable at 20M points (25 FPS). See supplementary material for details.

## 4 Usage scenarios

**Figure 2 (usage scenarios):** Left: attraction–repulsion tour of 70K Fashion-MNIST
images. Middle: UMAP-validating little PCA tour of 290K single-cell RNA-seq cells.
Right: sequential embedding tour of 3M arXiv titles and abstracts.

We demonstrate dtour in two usage scenarios: (1) gradually revealing structure in
high-dimensional data through guided touring and manual manipulation, and (2) validating
non-linear DR outputs by touring across embedding methods or models to check whether
observed structure is genuine or artifactual.

### 4.1 Gradually revealing structure

No single projection fully captures a high-dimensional manifold. Instead, understanding
emerges from viewing multiple projections and the transitions between them.

**Attraction-repulsion spectrum.** We apply the attraction-repulsion tour to
Fashion-MNIST (Xiao 2017), sweeping from attraction-only LE (continuous layout) through
ForceAtlas2 and UMAP to repulsion-dominated t-SNE (distinct clusters) (Böhm 2022).
Scrubbing through the tour (Fig. 2, 1a) reveals this progression: the continuous LE-like
layout gradually separates into the clusters visible in UMAP and sharpened in t-SNE.
During guided traversal at the UMAP-like keyframe, we notice a tight cluster (1b) of 96
points embedded among shirts, dresses, and pullovers (1c), far from the main trouser
cluster. Selecting these points and scrubbing back to the ForceAtlas2-like keyframe
reveals that they spread across the layout: **the tight cluster is an artifact of
repulsive forces, not a reflection of genuine data structure.** Inspecting the images
confirms that all 96 points are short trousers whose compact pixel silhouette resembles
upper-body garments more than full-length trousers, explaining their misplacement. Yet
touring also reveals stability: (1d) boundary points (such as ambiguous boot-like bags
bridging the footwear and bag clusters) persist across the entire spectrum, suggesting
that **boundary placement can be more trustworthy than cluster tightness.**

**Laplacian Eigenmaps tour.** We apply the spectral Fisher LE tour (Section 3.3) to a
single-cell dataset of 346K immune cells profiled by CyTOF across 9 surface protein
markers from Mair et al. (2022), with cell-type labels derived from FAUST (Greene 2021).
The resulting tour recovers known immunological hierarchy without manual specification
of which markers matter. The first keyframe separates cells along CD4 versus CD8, the
fundamental division between helper and cytotoxic T cells. The second is driven by CD103
and ICOS, distinguishing tissue-resident from activated and regulatory populations,
precisely the axis Mair et al. identified as most relevant to tumor-immune differences.
Subsequent frames resolve finer structure through markers of T cell regulation (CD25),
cytotoxicity (Granzyme), activation (CD38), and exhaustion (Tim3). Notably, CD3, the
canonical T cell marker, appears only in late frames with low loading, confirming that
the tour correctly assigns minimal weight to markers constant across the population. To
go further, we select regulatory T cells (Tregs) during guided tour and switch to manual
mode: dragging the ICOS axis separates ICOS-high from ICOS-low Tregs, isolating the
tumor-enriched immunosuppressive subset identified by Mair et al. as the key phenotype
distinguishing cancer from non-malignant inflammation.

### 4.2 Validating embedding structure

Beyond revealing structure, a second challenge is validating whether patterns in
non-linear DR outputs reflect genuine data structure or projection artifacts. dtour
addresses this by touring higher-dimensional embeddings or across multiple models.

**UMAP-validating PCA tour.** Single-cell analysis pipelines commonly select highly
variable genes, reduce to the top principal components, and embed the resulting PCA
space into 2D with UMAP (Becht 2019). Because UMAP operates directly on the PCA output,
touring through PC pairs provides a natural validation layer: structure present in
UMAP but absent from the PCA tour must have been introduced by the non-linear
embedding. We apply a little PCA tour to 276K cells from a developing mouse brain atlas
(La Manno 2021), touring the first 8 principal components alongside a 2D UMAP of the
same PCA space. Some structures are stable across both representations: (2a)
gastrulation and ectoderm cells form a consistent progression in every PCA keyframe and
in UMAP, confirming genuine transcriptional coherence. Other structures diverge: (2b)
choroid plexus cells form a single cohesive cluster throughout the PCA tour but split
into two distant UMAP clusters, and (2c) blood cells, which UMAP isolates as a
disconnected island, show no comparable separation in the PCA tour. These contrasts show
that touring PC pairs can distinguish genuine structure from embedding artifacts.

**Sentence embedding model comparison.** To compare embedding models, we construct a
sequential tour from 3M arXiv title-abstract embeddings produced by four models spanning
2023–2026: SPECTER2 (Singh 2023), BGE-M3 (Chen 2024), Nomic Embed Text v2 (Nussbaum
2025), and F2LLM-v2-8B (Zhang 2026; top-ranked on clustering benchmarks), each reduced
to 2D with default UMAP. Touring across models reveals broad stability: the overall
topical landscape is consistent across all four embeddings despite a 10× range in model
size. However, (3d) F2LLM produces visibly tighter subclusters. A 2D colormap encoding
position in the SPECTER2 frame makes structural shifts immediately visible: (3e) one
prominent compact cluster in F2LLM disperses across the embedding in all three
encoder-based models. Analysis of the ~1,200 selected papers reveals that ~84% are
physics education research, a subfield whose pedagogical language differs sharply from
typical arXiv prose. F2LLM appears to cluster these by *discourse style* rather than
research topic, while the citation-trained SPECTER2 distributes the same papers among
their respective physics subfields. This example illustrates how sequential tours extend
embedding validation from inspecting a single layout to comparing what different models
treat as similarity, surfacing behavioral differences that no individual 2D projection
can reveal.

## 5 Conclusion

High-dimensional data visualization is fundamentally hard, as no single projection
captures the full structure of a complex manifold. dtour shows that tours become
practical when traversal is fluid and progressive: effortless scrubbing and selection
let users gradually build intuition that no static view can provide. We hope that
dtour's scalability and availability across Python and web ecosystems spur the
development of novel tours and applications that transcend the single 2D embedding.

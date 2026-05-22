# Design Note: scene-query

*Shuaibu Oluwatunmise — Humanoid Perception & Spatial AI internship challenge*

---

**Contents**
1. [Framing](#1-framing)
2. [Pipeline overview](#2-pipeline-overview)
3. [Geometry: VGGT-Omega](#3-geometry-vggt-omega)
4. [Semantics: open-vocabulary detection and segmentation](#4-semantics-open-vocabulary-detection-and-segmentation)
5. [Lifting: SAM 2 masks to 3D](#5-lifting-sam-2-masks-to-3d)
6. [Query layer and bounding box fitting](#6-query-layer-and-bounding-box-fitting)
7. [Robot relevance](#7-robot-relevance)
8. [What I deliberately didn't do](#8-what-i-deliberately-didnt-do)
9. [Known limitations of this implementation](#9-known-limitations-of-this-implementation)
10. [Future work](#10-future-work)

---

## 1. Framing

The challenge asks for 3D scene reconstruction from a short phone video. Most submissions will interpret "reconstruction" as producing a 3D model — a point cloud, a mesh, or a Gaussian splat — and evaluate success by visual fidelity. That is the wrong problem to solve.

A humanoid robot doesn't navigate a point cloud. It navigates *semantic space*. It needs to know where the chair is, what's within arm reach, and what changed since its last visit. None of these questions are answered by a .ply file. They require a representation that is both geometric (positions in world space) and semantic (what those positions mean).

This project frames the output as **queryable scene memory**: a 3D structure you can ask questions of in plain English. Reconstruction is a means to that end, not the end itself.

---

## 2. Pipeline overview

```
video / images
      │
      ▼
[VGGT-Omega]  ──────────────────────────────►  poses + depth maps + scene cloud
                                                          │
text query ──► [Grounding DINO] ──► per-frame boxes       │
                      │                                   │
                  [SAM 2] ──────► per-frame masks          │
                                         │                │
                                    [lift.py] ◄───────────┘
                                         │
                                         ▼
                               labelled 3D point cluster
                                         │
                                [query_engine.py]
                                         │
                                         ▼
                               OBB + centroid + Rerun
```

VGGT-Omega handles all geometry in a single forward pass. Grounding DINO and SAM 2 handle semantics — one detects objects from free-text, the other segments them precisely. The lifting step bridges the two streams geometrically: SAM 2 masks are back-projected through VGGT depth maps into world space. The query engine cleans the resulting cluster and fits an oriented bounding box.

---

## 3. Geometry: VGGT-Omega

The standard choice for pose estimation is COLMAP — a well-understood, battle-tested SfM pipeline. I didn't use it. Instead I used **VGGT-Omega** (Meta AI, CVPR 2025), a feed-forward reconstruction model that takes a set of video frames and produces camera poses, dense per-frame depth maps, and a full scene point cloud in a single forward pass — no per-scene optimisation, no iteration.

The reasons this matters for robotics:

**Speed.** COLMAP on a 60-second video typically runs for several minutes. VGGT-Omega runs in seconds. A robot encountering a new space cannot wait for an iterative solver to converge.

**Dense depth.** COLMAP produces sparse point clouds from matched keypoints. VGGT-Omega produces dense per-frame depth maps. Dense depth is what you need for free-space estimation and, critically, for lifting 2D detections into 3D — sparse points leave too many gaps.

**Natural composition with 2D semantics.** VGGT-Omega gives one depth map per frame, aligned with the input image. Grounding DINO and SAM 2 give one set of masks per frame, aligned with the same image. They compose directly: masked pixels → unproject using depth → transform using pose → world-space labelled points.

The cost of this choice is accuracy on long sequences or textureless surfaces, where feed-forward methods can drift. COLMAP with loop closure is more accurate when it works. For a short room capture, VGGT-Omega is more than sufficient.

---

## 4. Semantics: open-vocabulary detection and segmentation

The architecture is **Grounding DINO + SAM 2**, operating at query time rather than at reconstruction time.

**Why open-vocabulary over a trained detector.** A domain-specific trained model (e.g. YOLOv8 fine-tuned on office objects) is faster and often more accurate within its vocabulary. But it answers a different question. If the vocabulary is fixed at training time, the system can only find objects it was trained to find. The interesting capability — the one useful to a robot in an unfamiliar environment — is finding *any* object the operator names. Grounding DINO enables this: it takes a text string and detects matching objects across the scene, zero-shot, with no retraining.

**Why SAM 2 rather than bounding box crops.** Grounding DINO returns bounding boxes. Naively lifting all pixels inside a bounding box introduces background contamination — the region around the object gets projected into 3D along with the object itself, polluting the cluster with irrelevant points. SAM 2 takes those boxes as prompts and returns pixel-precise segmentation masks. The lifted points correspond to the object surface rather than a rectangular region of the scene. This produces tighter, cleaner 3D clusters and more accurate bounding boxes.

**Why at query time, not reconstruction time.** Running detection and segmentation over every frame is expensive. Doing it at reconstruction time would commit the semantic predictions upfront and make re-querying with different labels require rerunning the entire pipeline. Running GSAM2 at query time means: reconstruct once (expensive), query many times (cheap). The 3D geometry is stable; the semantic interpretation can be updated freely.

**Presets.** For full-environment scans, a preset maps an environment type to a list of object classes. The office preset queries keyboard, mouse, chair, and laptop in a single pass. This is a lightweight mechanism for structured exploration — not pre-programmed detection, just a curated list of likely objects for a known domain.

---

## 5. Lifting: SAM 2 masks to 3D

For each frame *t*:

1. Read the depth map *D_t* (H × W) from VGGT-Omega.
2. Resize the SAM 2 mask to match the depth map resolution.
3. For each masked pixel *(u, v)*: read depth *z = D_t(u, v)*, unproject using camera intrinsics → camera-space point → apply camera-to-world pose *T_t* → world-space point *p*.
4. Accumulate points across all frames that detect the object.

The result is a raw 3D cluster — accurate at the object core, noisy at mask boundaries where depth transitions introduce outliers. Two passes clean it:

**Statistical outlier removal (SOR):** compute mean k-nearest-neighbour distance per point; remove points whose mean exceeds the global mean by more than 2 standard deviations. This kills mask-edge bleed and depth speckle.

**Largest cluster filter:** fit a union-find structure over the surviving points using a radius threshold (3× median NN distance). Keep only the largest connected component. This drops reflections and occluded fragments that survive SOR but sit spatially apart from the main object.

The entire lifting and cleaning step is pure NumPy and SciPy. The simplicity is intentional: the geometry is exact; the uncertainty is in the upstream models, and no amount of fusion complexity recovers from a bad mask.

---

## 6. Query layer and bounding box fitting

The output of the lifting step is a cleaned, labelled point cluster in world space. The query engine fits a **gravity-aligned oriented bounding box (OBB)** to it.

Gravity direction is inferred from the camera poses: the average of the negated camera Y-axes gives the scene up-vector. This is reliable for handheld captures where the camera is roughly upright. With gravity known, a 2D footprint is projected onto the horizontal plane and `cv2.minAreaRect` fits a minimum-area rectangle to it. The box is then extruded vertically to contain the full height of the cluster.

The result is an OBB with a meaningful orientation — yaw rotation, flat bottom, tight fit — which is directly consumable by a manipulation planner (grasp approach direction, placement zone) or a navigator (obstacle footprint).

---

## 7. Robot relevance

The system produces, for each queried object:
- A centroid in world-frame coordinates
- An oriented bounding box (position, half-extents, quaternion)
- A point count and confidence score (detection confidence averaged over frames)

These map directly to robot planning primitives: centroid is a task-space target, OBB defines the obstacle footprint and an approach envelope, confidence lets a planner decide whether to trust the localisation or request a closer look. The representation is compact, serialisable, and updatable — the right shape for a robot's working memory of a scene.

---

## 8. What I deliberately didn't do

**COLMAP.** Accurate but slow and sparse. The per-scene iterative optimisation — minutes of compute — is incompatible with robots encountering new spaces continuously.

**NeRF / 3D Gaussian Splatting.** Neural rendering answers "what does the scene look like from a novel viewpoint." That is useful for simulation and telepresence. It is the wrong property for manipulation planning. 3DGS additionally requires 5–30 minutes of per-scene training, which breaks the reconstruct-once operational model.

**CLIP feature lifting.** Project CLIP features into 3D, answer queries by cosine similarity. Several recent systems do this (LERF, OpenMask3D). The output is a heat map over the point cloud, not a segmented object — harder to consume downstream, harder to visualise, and with no calibrated confidence per detection. Explicit detection followed by geometric lifting produces a cleaner, more interpretable result.

**A trained domain-specific detector.** A fine-tuned YOLOv8 on office classes would be faster and more accurate within that vocabulary. But it can only find what it was trained to find. The core value of this system — querying for any object by name, without retraining — requires an open-vocabulary front-end. Speed is a reasonable trade for generality here.

**ROS.** This is a batch processing pipeline, not a live robotic system. Adding a ROS wrapper would make the system harder to run and contribute nothing to the problem being demonstrated.

---

## 9. Known limitations of this implementation

**Dark and black objects.** VGGT-Omega's depth estimation degrades significantly on dark or absorptive surfaces. Learned depth models rely on photometric cues — texture gradients, shading, and reflections — to infer distance. A matte black object provides almost none of these. The resulting depth map over that region is either missing, blurred out from surrounding surfaces, or numerically unstable. When the lifting step back-projects those pixels into 3D, the cluster is noisy or displaced from the object's true position. In practice: dark objects either fail to localise at all, or produce an inflated, poorly-shaped 3D cluster. This is a fundamental limitation of passive monocular depth, not a tunable parameter.

**3D bounding box quality** *(work in progress).* The oriented bounding box is fit to a cleaned point cluster, but the cluster itself is an approximation. SAM 2 masks have soft boundaries that don't align precisely with object edges. Depth noise inflates the cluster — especially along the depth axis, where per-pixel depth error compounds. The current gravity-aligned fitting (`cv2.minAreaRect` on the 2D footprint, extruded vertically) assumes a roughly upright camera and a flat-bottomed object — for tilted, thin, or partially occluded objects the box is often loose or misoriented. The boxes as they stand are coarse spatial indicators — "the chair is roughly here, roughly this big" — not precise enough for contact-level manipulation. Better orientation estimation is the natural next step; see section 10.

---

## 10. Future work

**Live inference.** VGGT-Omega processes a fixed frame set offline. Swapping it for a streaming monocular depth estimator (e.g. Depth Anything V2) with a visual odometry front-end would make the geometry pipeline real-time capable.

**Incremental mapping.** Each reconstruction is a fresh scene. A robot should accumulate scene memory across visits — updating object positions, adding new objects, removing ones that have moved. This is largely a data structure problem: a persistent labelled point cloud with update semantics.

**Affordance labelling.** "Find the chair" tells you where it is. "Is it graspable from here" requires knowing its shape, orientation, and the robot's kinematics. Affordance prediction is the natural next layer.

**Mask refinement from depth.** SAM 2 masks can be further refined using depth discontinuities — pixels where depth changes sharply are likely object boundaries. Post-processing masks with depth edges before lifting would reduce boundary noise and improve cluster quality.

**Better OBB orientation via Orient Anything.** The current OBB orientation is derived purely from geometry — the 2D footprint projected onto the gravity plane. A stronger approach would use [Orient Anything](https://github.com/SpatialVision/orient-anything), a model that predicts the canonical 3D orientation of an object from a single image crop. Plugging it in after SAM 2 segmentation would give each detected object a pose estimate grounded in category-level shape priors rather than just the point cloud footprint — directly addressing the loose box issue for tilted or irregularly shaped objects.

**Uncertainty propagation.** Per-point confidence is currently detection confidence averaged over frames. A proper treatment would propagate detection uncertainty through the lifting step, producing calibrated 3D uncertainty estimates that a planner could reason over.

# Design Note: scene-query

*Shuaibu Oluwatunmise — Humanoid Perception & Spatial AI internship challenge*

---

## 1. Framing

The challenge asks for 3D scene reconstruction from a short phone video. Most submissions will interpret "reconstruction" as producing a 3D model — a point cloud, a mesh, or a Gaussian splat — and evaluate success by visual fidelity. That is the wrong problem to solve.

A humanoid robot doesn't navigate a point cloud. It navigates *semantic space*. It needs to know where the chair is, where the floor is traversable, what's within arm reach from a given stance, and what changed since its last visit. None of these questions are answered by a .ply file. They require a representation that is both geometric (positions in world space) and semantic (what those positions mean).

This project frames the output as **scene memory**: a labelled, queryable 3D structure that could be consumed directly by a planning or manipulation stack. Reconstruction is a means to that end, not the end itself.

---

## 2. Pipeline overview

```
video → [VGGT-Omega] → poses + depth → point cloud
                                    ↘
video → [YOLOv8 office detector] → per-frame bounding boxes
                                    ↘
                                [lift.py] → labelled 3D points
                                    ↘
                                [query_engine.py] → answers + Rerun
```

VGGT-Omega handles all geometry — camera poses, per-frame metric depth, and a dense point cloud — in a single feed-forward pass. YOLOv8 handles semantics: it runs on each frame independently, detecting objects and returning bounding boxes with class labels and confidence scores. The two streams compose directly through the lifting step: masked depth pixels are unprojected into world space using the VGGT poses and labelled according to the YOLO detections.

---

## 3. Geometry: VGGT-Omega

The standard choice for pose estimation is COLMAP — a well-understood, battle-tested SfM pipeline. I didn't use it. Instead I used **VGGT-Omega** (Meta FAIR, CVPR 2026), the latest feed-forward reconstruction model, which takes a set of video frames and produces camera poses, dense per-frame depth maps, and a full scene point cloud in a single forward pass, without any per-scene optimisation.

The reasons this matters for robotics:

**Speed.** COLMAP on a 60-second video typically runs for several minutes. VGGT-Omega runs in seconds. A robot that enters a new room and needs a working scene model within its planning cycle cannot wait for an iterative solver to converge. Feed-forward inference is the right operational model.

**Dense depth.** COLMAP produces sparse point clouds from matched keypoints. VGGT-Omega produces dense per-frame depth maps. Dense depth is what you need for free-space estimation and for lifting 2D detections into 3D — sparse points leave too many gaps.

**Natural composition with 2D semantics.** VGGT-Omega gives one depth map per frame, aligned with the input image. YOLOv8 gives one set of bounding boxes per frame, aligned with the same image. They compose directly: masked pixels → unproject using depth → transform using pose → world-space labelled points. This frame-aligned structure makes the lifting step trivial.

**State of the art.** VGGT-Omega is a CVPR 2026 model — more recent than what was available when the challenge was written. Using it is a deliberate signal: this pipeline is built on the current frontier, not last year's defaults.

The cost of this choice is accuracy on long sequences or textureless surfaces, where feed-forward methods can drift. COLMAP with loop closure is more accurate when it works. For a demo on a small room captured in 30–60 seconds, VGGT-Omega is more than sufficient. For a production robot system, the right answer is probably a streaming monocular depth estimator with a visual odometry front-end — but that is future work, not a 10-day submission.

---

## 4. Semantics: trained YOLOv8 over open-vocabulary models

The obvious choice for open-vocabulary semantic segmentation in 2024 is Grounded-SAM-2 (Grounding DINO + SAM 2). I considered it and rejected it.

The first problem is weight and complexity. Grounded-SAM-2 requires two large models — a detection transformer (~1 GB) and a segmentation model (~2.5 GB) — and a non-trivial installation involving custom CUDA extensions. This is a meaningful burden for anyone trying to reproduce the results.

The second problem is speed. DINO text grounding followed by SAM 2 mask prediction adds multiple seconds per frame on a GPU. For a 30-frame scene, this is noticeable. For a robot that needs to update its scene model frequently, it is a bottleneck.

The third problem is that open-vocabulary generalisation is unnecessary for this domain. The challenge is set in an office environment. I trained my own **YOLOv8n detection model** on a labelled office dataset covering 10 classes: bottle, chair, keyboard, monitor, mouse, mug, notebook, pen, printer, and stapler. The model achieves **mAP@0.5 = 98.3%** across all classes.

This is the right trade-off: a narrower vocabulary in exchange for a much lighter, faster, and more accurate model. The detector is 6 MB. It requires no text prompt, no CUDA extensions, and no secondary segmentation step. It runs at tens of frames per second on a GPU.

The broader principle is worth stating explicitly. Open-vocabulary models are powerful when the vocabulary is genuinely unknown at deployment time. In robotics, the deployment environment is usually partially known — you know the robot will operate in an office, a warehouse, or a hospital. A domain-specific trained detector almost always outperforms a general open-vocabulary model in that domain, and costs a fraction of the compute. Building and training that detector is part of the engineering work, not a shortcut around it.

---

## 5. Lifting: 2D detections to 3D points

The lifting step is the bridge between VGGT-Omega's geometry and YOLO's semantics. It is not a neural network. It is geometry.

For each frame *t*:
1. Read the depth map *D_t* (H × W, metric depth in metres) from VGGT-Omega.
2. Read the bounding box detections from YOLO: for each detected object, all pixels inside the bounding box are treated as belonging to that label.
3. For each labelled pixel *(u, v)*: unproject using camera intrinsics → camera-space point → apply camera-to-world pose *T_t* → world-space point *p*.
4. Accumulate *(p, label, confidence)* across all frames.

Multiple frames observe the same physical object. Points that receive consistent label votes across views get high confidence; points at box edges — where background leaks in — get lower weight through the outlier removal pass that follows. A statistical outlier filter (mean k-NN distance + standard deviation threshold) removes floating noise introduced by depth imprecision near box boundaries.

The entire lifting step is approximately 30 lines of NumPy. The simplicity is intentional: complex fusion methods add hyperparameters and failure modes. The geometry is exact; the uncertainty is in the upstream models, and no amount of fusion complexity recovers from a bad detection.

---

## 6. Query layer

The query layer wraps the labelled point cloud in three query types:

**Text query** (`"find the chair"`): look up the label in `labels.json`, retrieve the corresponding point indices from `pointcloud.npz`, compute a 3D axis-aligned bounding box and centroid. Return the centroid, box dimensions, and point count. Visualise the matched cluster in Rerun, highlighted against the full scene.

**Free-space query** (`--free-space`): voxelise the point cloud at some resolution (default 5 cm). Mark occupied voxels. For voxels above the floor plane and below some clearance height with no occupied voxels in a vertical column above them, mark as traversable. Return the traversable voxel grid. This is a coarse navigability map, not a full occupancy grid — but it's directly interpretable by a path planner.

**Reachability query** (`--reachable-from x y z`): given a robot base position, return all labelled objects whose centroid falls within reach distance (default 0.7 m, roughly humanoid arm reach) and within a height band above the base (0.3–1.4 m). This directly answers "what can the robot manipulate from here" without running a full kinematics solver.

Rerun is used throughout for visualisation — not because it makes things look good, but because it logs structured 3D entities (point clouds, bounding boxes, coordinate frames) that can be scrubbed back frame by frame. This makes pipeline debugging fast: you can see exactly which detections lifted to which world-space clusters, and where the depth or pose caused drift.

---

## 7. Robot relevance

The three query types map directly to robot planning primitives:

- "Find the chair" → object detection in the world frame → task-space goal for manipulation
- Free space → traversability map → input to a path planner (A*, RRT, or learned)
- Reachable-from → workspace analysis → configuration selection for grasp planning

None of these require dense rendering or novel view synthesis. The representation is compact: a labelled point cloud and an index. It can be serialised, transmitted, updated incrementally, and queried in milliseconds. This is the right shape for a robot's working memory of a scene — not a radiance field.

---

## 8. What I deliberately didn't do

**COLMAP.** Accurate but slow and sparse. The operational model — per-scene iterative optimisation, minutes of compute — is incompatible with robots encountering new spaces continuously.

**NeRF / 3D Gaussian Splatting.** Neural rendering methods answer "what does the scene look like from a novel viewpoint". That is useful for telepresence and simulation; it is the wrong property for manipulation planning. You can bolt semantics onto NeRF (Semantic-NeRF, LERF, LangSplat) but you are adding complexity to answer questions a simpler representation handles directly. 3DGS additionally requires 5–30 minutes of per-scene training — incompatible with the operational model.

**Grounded-SAM-2 / open-vocabulary segmentation.** Powerful for unknown environments. Slower, heavier, and less accurate than a domain-specific trained detector for a known environment. The engineering choice of training a dedicated YOLO model rather than prompting a general model is what makes the pipeline deployable at robot-relevant speeds.

**CLIP feature lifting.** Extract CLIP features per frame, project into 3D, answer queries by cosine similarity. Several recent systems do this (LERF, OpenMask3D). The output is a heat map over the point cloud, not a segmented object — harder to consume downstream, harder to visualise, and with no calibrated confidence. Explicit detection followed by geometric lifting produces a cleaner, more interpretable result.

**ROS.** This is a batch processing job, not a live robotic system. Adding a ROS wrapper would make the system harder to run and contribute nothing to the problem being demonstrated. ROS is the right layer when integrating with actual hardware; it's over-engineering for an offline demo.

**Training anything from scratch for geometry.** 10-day sprint. The 3D understanding capabilities needed already exist in VGGT-Omega. Training the YOLO detector, however, was the right call — it is a small model on a bounded domain where labelled data is available and the accuracy payoff is high.

---

## 9. Future work

The honest limitations of this approach, and what I'd do with more time:

**Live inference.** VGGT-Omega processes a fixed set of frames offline. A robot needs continuous depth and pose updates. Swapping VGGT-Omega's depth maps for a streaming monocular depth estimator (Depth Anything V2) and adding a lightweight visual odometry front-end would make this a real-time capable pipeline.

**Incremental mapping.** Each reconstruction is currently a fresh scene. A robot should accumulate scene memory across visits — updating object positions, adding new objects, removing ones that have moved. This is a data structure problem (a persistent, indexed labelled point cloud with update semantics) more than a model problem.

**Affordance labelling.** "Find the chair" tells you where it is. "Is it graspable from here" requires knowing its shape, orientation, and the robot's kinematics. Affordance prediction — what actions are possible on this object from this pose — is the natural next layer.

**Extending the detector vocabulary.** The current YOLO model covers 10 office classes. Expanding to a broader office/home/warehouse vocabulary through additional training data is straightforward. Alternatively, for truly unknown environments, a hybrid approach — YOLO for known classes plus an open-vocabulary fallback for unknowns — would preserve the speed advantage for common cases while retaining generality.

**Uncertainty propagation.** Per-point confidence is currently label vote consistency combined with depth confidence from VGGT-Omega. A proper treatment would propagate detection uncertainty through the lifting step, producing calibrated 3D uncertainty estimates that a planner could reason over.

**Semantic costmaps.** The free-space query returns a binary traversability grid. A richer version would produce a costmap weighted by distance to obstacles, label-based hazard scores, and terrain type — the standard input format for Nav2 and similar planners.
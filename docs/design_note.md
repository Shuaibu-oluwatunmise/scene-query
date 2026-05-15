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
video → [VGGT] → poses + depth → point cloud
                              ↘
video → [Grounded-SAM-2] → per-frame masks
                              ↘
                          [lift.py] → labelled 3D points
                              ↘
                          [query_engine.py] → answers + Rerun
```

Each component is a pretrained model or a short piece of glue code. Nothing is trained from scratch.

---

## 3. Geometry: VGGT over COLMAP

The standard choice for pose estimation is COLMAP — a well-understood, battle-tested SfM pipeline. I didn't use it. VGGT (Wang et al., Meta FAIR, 2024) is a feed-forward transformer that takes a set of video frames and produces camera poses, dense per-frame depth maps, and a point cloud in a single forward pass, without any per-scene optimisation.

The reasons this matters for robotics:

**Speed.** COLMAP on a 60-second video typically runs for several minutes. VGGT runs in seconds. A robot that enters a new room and needs a working scene model within its planning cycle cannot wait for an iterative solver to converge. Feed-forward inference is the right operational model.

**Dense depth.** COLMAP produces sparse point clouds from matched keypoints. VGGT produces dense per-frame depth maps. Dense depth is what you need for free-space estimation and for lifting 2D segmentation masks to 3D — sparse points leave too many gaps for mask projection to work cleanly.

**Natural composition with 2D semantics.** VGGT gives one depth map per frame, aligned with the input image. Grounded-SAM-2 gives one mask per frame, aligned with the same image. They compose directly: masked pixels → unproject using depth → transform using pose → world-space labelled points. This frame-aligned structure makes the lifting step trivial.

The cost of this choice is accuracy. VGGT, like all feed-forward methods, can drift on long sequences or textureless surfaces. COLMAP with loop closure is more accurate when it works. For a demo on a small room captured in 30–60 seconds, VGGT is more than sufficient. For a production robot system, the right answer is probably a learned odometry backbone (like Depth Anything V2 or Depth Pro for depth, with a visual odometry front-end) — but that's future work, not a 10-day submission.

---

## 4. Semantics: Grounded-SAM-2 over CLIP feature lifting

The alternative to explicit segmentation is CLIP-based feature lifting: extract CLIP features per frame, project them into 3D, and answer queries by cosine similarity in feature space. Several recent papers do this (LERF, OpenMask3D, others). It sounds elegant. In practice it has two problems for this use case.

First, it's indirect. "Find the chair" means computing a text embedding, searching feature space, and thresholding — with no guarantee the result is spatially contiguous or that the confidence is calibrated. The output is a heat map over the point cloud, not a segmented object.

Second, it requires the reviewer to interpret. A heat map is less legible than a mask. For a demo submission, legibility matters.

Grounded-SAM-2 (Grounding DINO + SAM 2) takes a text prompt and returns binary masks, per frame, with instance IDs. The output is unambiguous: these pixels are "chair". The text grounding generalises broadly — it handles open-vocabulary labels the model has never been fine-tuned on. SAM 2's video tracking maintains instance identity across frames, which means the same chair instance across 30 frames accumulates points from 30 views rather than being fragmented.

The tradeoff: Grounded-SAM-2 requires the user to specify label categories up front. CLIP lifting can answer arbitrary queries after the fact. For the query patterns this system targets — named objects, free space, reachability — knowing the labels at reconstruction time is acceptable. A future version could run CLIP over the labelled point cloud for post-hoc vocabulary expansion.

---

## 5. Lifting: 2D masks to 3D points

The lifting step is the bridge between VGGT's geometry and Grounded-SAM-2's semantics. It is not a neural network. It is geometry.

For each frame *t*:
1. Read the depth map *D_t* (H × W, metric depth in metres) from VGGT.
2. Read the semantic mask *M_t* for label *l* from Grounded-SAM-2.
3. For each pixel *(u, v)* where *M_t[v, u] = 1*: unproject using camera intrinsics → camera-space point → apply camera-to-world pose *T_t* → world-space point *p*.
4. Accumulate *(p, l, confidence)* across all frames.

Multiple frames observe the same physical point. Points that receive consistent label votes across views get high confidence; points with conflicting labels (e.g., a surface seen from a grazing angle that gets misclassified in one frame) get lower weight. A simple majority vote or confidence-weighted mean is sufficient — no learned fusion needed.

The entire lifting step is approximately 30 lines of NumPy. The simplicity is intentional: complex fusion methods add hyperparameters and failure modes. The geometry is exact; the uncertainty is in the upstream models, and no amount of fusion complexity recovers from a bad mask.

---

## 6. Query layer

The query layer wraps the labelled point cloud in three query types:

**Text query** (`"find the chair"`): look up the label in `labels.json`, retrieve the corresponding point indices from `pointcloud.npz`, compute a 3D axis-aligned bounding box and centroid. Return the centroid, box dimensions, and point count. Visualise the matched cluster in Rerun, highlighted against the full scene.

**Free-space query** (`--free-space`): voxelise the point cloud at some resolution (default 5 cm). Mark occupied voxels. For voxels above the floor plane and below some clearance height with no occupied voxels in a vertical column above them, mark as traversable. Return the traversable voxel grid. This is a coarse navigability map, not a full occupancy grid — but it's directly interpretable by a path planner.

**Reachability query** (`--reachable-from x y z`): given a robot base position, return all labelled objects whose centroid falls within reach distance (default 0.7 m, roughly humanoid arm reach) and within a height band above the base (0.3–1.4 m). This directly answers "what can the robot manipulate from here" without running a full kinematics solver.

Rerun is used throughout for visualisation — not because it makes things look good, but because it logs structured 3D entities (point clouds, bounding boxes, coordinate frames) that can be scrubbed back frame by frame. This makes pipeline debugging fast: you can see exactly which masks lifted to which world-space clusters.

---

## 7. Robot relevance

The three query types map directly to robot planning primitives:

- "Find the chair" → object detection in the world frame → task-space goal for manipulation
- Free space → traversability map → input to a path planner (A*, RRT, or learned)
- Reachable-from → workspace analysis → configuration selection for grasp planning

None of these require dense rendering or novel view synthesis. The representation is compact: a labelled point cloud and an index. It can be serialised, transmitted, updated incrementally, and queried in milliseconds. This is the right shape for a robot's working memory of a scene — not a radiance field.

---

## 8. What I deliberately didn't do

**COLMAP.** Accurate but slow and sparse. The operational model (per-scene iterative optimisation, minutes of compute) is incompatible with robots encountering new spaces continuously.

**NeRF / 3D Gaussian Splatting.** Neural rendering methods. They answer "what does the scene look like from a novel viewpoint". That is a useful property for telepresence and simulation; it is the wrong property for manipulation planning. You can bolt semantics onto NeRF (Semantic-NeRF, LERF, LangSplat) but you are adding complexity to answer questions a simpler representation handles directly. 3DGS additionally requires 5–30 minutes of per-scene training — again incompatible with the operational model.

**ROS.** This is a batch processing job, not a live robotic system. Adding a ROS wrapper would make the system harder to run and contribute nothing to the problem being demonstrated. ROS is the right layer when you're integrating with actual hardware; it's over-engineering for an offline demo. Worth one paragraph in future work, not worth adding as a dependency.

**Training anything from scratch.** 10-day sprint. Every hour spent training is an hour not spent on the pipeline, the query layer, or the design note. More importantly: the 3D understanding capabilities I need already exist in VGGT and Grounded-SAM-2. Composition over re-implementation is both pragmatic and correct — it's how a competent robotics team actually builds systems.

**Dense 3D panoptic segmentation** (Mask3D, Mask-RCNN-3D, etc.). These methods operate on 3D input directly and would produce cleaner per-instance segmentation. But they require a 3D point cloud as input — which means you need VGGT anyway — and they add another large model, another set of weights, and another source of failure. Grounded-SAM-2 on 2D frames with 3D lifting produces comparable results for the query patterns targeted here, with much less complexity.

---

## 9. Future work

The honest limitations of this approach, and what I'd do with more time:

**Live inference.** VGGT processes a fixed set of frames offline. A robot needs continuous depth and pose updates. Swapping VGGT's depth maps for a streaming monocular depth estimator (Depth Anything V2) and adding a lightweight visual odometry front-end would make this a real-time capable pipeline.

**Incremental mapping.** Right now each reconstruction is a fresh scene. A robot should accumulate scene memory across visits — updating object positions, adding new objects, removing ones that have moved. This is a data structure problem (a persistent, indexed labelled point cloud with update semantics) more than a model problem.

**Affordance labelling.** "Find the chair" tells you where it is. "Is it graspable from here" requires knowing its shape, orientation, and the robot's kinematics. Affordance prediction (what actions are possible on this object from this pose) is the natural next layer.

**Uncertainty propagation.** Per-point confidence is currently just label vote consistency. A proper treatment would propagate depth uncertainty from VGGT and mask uncertainty from Grounded-SAM-2 through the lifting step, producing calibrated 3D uncertainty estimates that a planner could reason over.

**Semantic costmaps.** The free-space query currently returns a binary traversability grid. A richer version would produce a costmap weighted by distance to obstacles, label-based hazard scores, and terrain type — the standard input format for Nav2 and similar planners.

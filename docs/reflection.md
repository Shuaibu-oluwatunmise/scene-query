# Reflection: scene-query

*Shuaibu Oluwatunmise — Humanoid Perception & Spatial AI internship challenge*

---

**Contents**
1. [What worked](#what-worked)
2. [What was harder than expected](#what-was-harder-than-expected)
3. [What I'd do with 2 more weeks](#what-id-do-with-2-more-weeks)
4. [What this taught me](#what-this-taught-me)

---

## What worked

**VGGT-Omega was the right call.** The geometry came out clean on the first run — accurate camera poses, dense depth maps, a well-structured point cloud. I expected to spend more time wrestling with reconstruction quality but the model just worked. Choosing a feed-forward model over COLMAP removed an entire class of per-scene failure modes (feature matching, loop closure, convergence) and I never had to touch it again after the first integration.

**Reconstruct-once, query-many held up in practice.** The architectural split between `reconstruct.py` and `query.py` turned out to be one of the better decisions. Once the geometry was saved, I could iterate on the query and lifting logic freely — changing detection prompts, tuning the outlier filter, adjusting OBB fitting — without ever rerunning VGGT. The feedback loop for the semantic side was fast.

**Grounding DINO's zero-shot range was broader than I expected.** "Find the bulldozer" on a LEGO model it had never seen, with no fine-tuning, at high confidence across all 25 frames. The open-vocabulary assumption held up on an intentionally unusual query. That gave me confidence the system would generalise rather than just work on the obvious test cases.

**Rerun made debugging visual and fast.** Being able to scrub through the frame timeline and see exactly which detections lifted to which 3D clusters — with camera frustums, masks, and point clouds all in the same coordinate frame — saved a significant amount of time. Any pipeline that mixes 2D and 3D data is hard to debug from terminal output alone.

---

## What was harder than expected

**Version hell with rerun.** The rerun API changed significantly between 0.23 and 0.32. The `EyeControls3D` and `LoopMode` components I wrote locally didn't exist on the version running on the GPU pod. I spent more time than I should have tracking down an `AttributeError` that only manifested at runtime on a remote machine. The fix was straightforward (pin the version, upgrade the pod) but it cost time that should have gone elsewhere. The lesson: pin your visualisation library version early and verify it on every machine you run on.

**Depth alignment at mask boundaries.** The edge of a SAM 2 mask doesn't always align cleanly with depth discontinuities. Near object boundaries, the depth map can be blurry or contain mixed foreground/background values — the per-pixel depth isn't certain even when the mask is. This introduced noise in the lifted clusters that took several iterations to clean up properly. The statistical outlier removal + largest-cluster filter handles it well enough, but I'd approach the boundary problem more carefully from the start next time — probably by eroding masks slightly before lifting.

**OBB orientation on poorly-captured scenes.** The gravity-aligned bounding box depends on the camera up-vector being consistent across frames. For scenes where the camera tilted significantly or moved in an unconstrained path, the inferred gravity direction was noisy and the OBB orientation was off. It works well for the handheld captures in this demo, but it's a fragile assumption I'd want to replace with an explicit floor plane estimate for a more general system.

**Deciding when to stop.** There were several points where I could have kept going — better mask refinement, incremental mapping, a live inference mode. Knowing what to cut and ship rather than continuing to extend was a real constraint. The rule I ended up with: if it makes the core demo worse or the code harder to run, cut it; if it's genuinely new capability, it can wait for future work.

---

## What I'd do with 2 more weeks

**Live depth and pose.** Replace VGGT-Omega's offline batch processing with a streaming monocular depth estimator (Depth Anything V2) and a lightweight visual odometry front-end. The geometry pipeline becomes real-time; the rest of the stack doesn't change.

**Incremental scene memory.** Right now, running the system twice on the same room gives you two independent reconstructions. A persistent labelled point cloud — with update semantics for moving objects and new detections — would make the system actually useful across multiple robot visits to the same space.

**Depth-guided mask refinement.** Erode SAM 2 masks using depth edge maps before lifting. This would reduce boundary noise in the 3D clusters without touching the outlier filter, producing cleaner OBBs with less post-processing.

**A proper evaluation.** The current results are qualitative — GIFs and confidence scores. A meaningful evaluation would require ground-truth 3D bounding boxes for a set of objects in a controlled scene, and metrics like 3D IoU across queries. I'd build that before extending the pipeline further.

---

## What this taught me

**The framing matters as much as the method.** "3D reconstruction" is an underspecified goal. Deciding early that the output should be queryable scene memory — not a mesh, not a radiance field — shaped every subsequent decision: which reconstruction model, which semantic approach, what the query interface looks like. A clear problem frame is worth more than a sophisticated model choice.

**Feed-forward models change the operational model for robotics.** COLMAP works by optimising over an entire scene; the optimisation has to restart when the scene changes. VGGT-Omega processes frames in a single pass; adding new frames is just another forward pass. For a robot that encounters new spaces continuously, the feed-forward model isn't just faster — it fits the operational loop in a way that iterative methods don't.

**Open-vocabulary detection is a meaningful capability shift.** Being able to query for an object by name — without retraining, without a fixed class list — is the difference between a system that works in the lab and one that works in the field. A trained detector is more accurate within its vocabulary; an open-vocabulary detector is useful outside it. For a general-purpose robot, the latter is the right trade.

**Visualisation is part of the work, not a finishing step.** I integrated Rerun from the start, not as a presentation layer added at the end. Every time I changed the lifting logic or the OBB fitting, I could immediately see the effect in 3D. That fast feedback loop probably saved more time than any single algorithmic improvement.

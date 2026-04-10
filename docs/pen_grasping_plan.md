# Pen Grasping Plan

## 1. Object Analysis

**Target object:** Red EXPO dry erase marker (chisel-tip)

| Property         | Value                          |
|------------------|--------------------------------|
| Shape            | Cylinder with tapered cap      |
| Length           | ~13 cm                         |
| Diameter (body)  | ~2 cm                          |
| Diameter (cap)   | ~2.5 cm                        |
| Weight           | ~20 g (lightweight)            |
| Surface          | Smooth plastic, mixed friction |
| Pose             | Upright, standing on flat table|
| Symmetry         | Rotationally symmetric (yaw)   |

**Challenges:**
- Narrow cylindrical body requires precise finger placement
- Smooth plastic surface means low friction -- need sufficient grip force
- Upright pose is unstable; any lateral contact may topple it before grasping
- Cap vs body texture difference may affect grasp stability

---

## 2. Grasp Strategy

### 2.1 Recommended Grasp: Top-Down Pinch Grasp

Given the marker is standing upright and the robot uses a parallel-jaw gripper (as in the Robomimic environments), the optimal strategy is:

1. **Approach from above** -- move the end-effector directly above the marker cap to avoid toppling
2. **Descend vertically** -- lower the gripper with fingers open, centered over the marker
3. **Grasp at the upper body** -- close the gripper around the top 1/3 of the marker body (just below the cap), where the cylinder is uniform and provides the best contact area
4. **Lift vertically** -- raise straight up to avoid lateral forces

```
    ┌─────────┐
    │ gripper  │    Phase 1: Approach (hover above)
    └────┬────┘
         │
         ▼
      ┌──┐          Phase 2: Descend
      │  │  ← cap
      ├──┤
      │  │  ← grasp zone (upper body)
      │  │
      │  │
      └──┘
```

### 2.2 Alternative Grasp: Side Approach (if marker is lying down)

If the marker has been toppled or is lying flat:
1. Approach from the side, perpendicular to the marker's long axis
2. Grasp at the center of mass (mid-body)
3. Lift with slight rotation to bring to desired orientation

### 2.3 Grasp Parameters

| Parameter              | Value           | Rationale                                  |
|------------------------|-----------------|--------------------------------------------|
| Approach height        | 15 cm above table | Clear of marker height + safety margin   |
| Pre-grasp offset       | 2 cm above grasp point | Allows precise final descent          |
| Gripper opening        | 4 cm            | >2x marker diameter for clearance          |
| Grasp force            | Low-medium      | Enough for friction, avoid crushing marker |
| Grasp point (z)        | Upper 1/3 of body | Stable, avoids cap pop-off               |
| Approach velocity      | Slow (< 5 cm/s) | Prevent toppling from air currents        |

---

## 3. Implementation in MIP Framework

### 3.1 Task Configuration

This grasping task maps to the existing MIP architecture with minimal changes. It follows the same pattern as the `lift_ph` tasks (lifting a cube) but adapted for a cylindrical object.

#### State-Based Configuration (`pen_grasp_state.yaml`)

```yaml
# @package task
defaults:
  - robomimic_base
  - _self_

env_name: "pen_grasp"
obs_type: "state"
env_type: "ph"

dataset_filename: "robomimic/pen_grasp/ph/low_dim.hdf5"

max_episode_steps: 500  # slightly longer than lift due to precision needs

obs_keys:
  - "object"           # pen pose (position + quaternion) + dimensions
  - "robot0_eef_pos"   # end-effector position (3D)
  - "robot0_eef_quat"  # end-effector orientation (quaternion)
  - "robot0_gripper_qpos"  # gripper finger positions

horizon: 16
act_dim: 7             # 3 (pos) + 3 (rot) + 1 (gripper)
obs_dim: 23            # 10 (object) + 3 (eef_pos) + 4 (eef_quat) + 2 (gripper) + 4 (pen_quat)
```

#### Image-Based Configuration (`pen_grasp_image.yaml`)

```yaml
# @package task
defaults:
  - robomimic_base
  - _self_

env_name: "pen_grasp"
obs_type: "image"
env_type: "ph"

dataset_filename: "robomimic/pen_grasp/ph/image.hdf5"

max_episode_steps: 500
render_obs_key: "agentview_image"

horizon: 16
act_dim: 7

shape_meta:
  action:
    shape:
      - 7
  obs:
    agentview_image:
      shape:
        - 3
        - 84
        - 84
      type: rgb
    robot0_eye_in_hand_image:
      shape:
        - 3
        - 84
        - 84
      type: rgb
    robot0_eef_pos:
      shape:
        - 3
      type: low_dim
    robot0_eef_quat:
      shape:
        - 4
      type: low_dim
    robot0_gripper_qpos:
      shape:
        - 2
      type: low_dim

num_envs: 4
```

### 3.2 Action Space

The 7-DOF action space encodes the full grasp trajectory:

| Dimension | Meaning           | Range       |
|-----------|-------------------|-------------|
| 0-2       | EEF delta position (x, y, z) | [-1, 1] normalized |
| 3-5       | EEF delta rotation (axis-angle) | [-1, 1] normalized |
| 6         | Gripper command (open/close) | [-1, 1] |

### 3.3 Observation Space

**State observations** should capture:
- **Pen pose:** position (x, y, z) + orientation (quaternion) = 7 dims
- **Pen geometry:** bounding box or radius + height = 3 dims
- **Robot state:** EEF position (3) + EEF quaternion (4) + gripper qpos (2) = 9 dims

**Image observations** (recommended for real-world transfer):
- **Agentview camera:** 84x84 RGB -- provides global scene context
- **Eye-in-hand camera:** 84x84 RGB -- provides close-up view for precise alignment
- **Proprioceptive state:** EEF pos + quat + gripper qpos = 9 dims

### 3.4 Reward Design (for RL fine-tuning)

The grasping task uses a staged sparse reward:

| Stage | Condition                              | Reward |
|-------|----------------------------------------|--------|
| 1     | EEF is within 5cm above the pen        | +0.25  |
| 2     | EEF is within 2cm of grasp point       | +0.25  |
| 3     | Gripper is closed around the pen       | +0.25  |
| 4     | Pen is lifted > 5cm above table        | +0.25  |

---

## 4. Data Collection Strategy

### 4.1 Demonstration Collection

Since MIP uses **behavior cloning with flow matching**, the primary training signal comes from expert demonstrations.

**Recommended approach:**
1. **Teleoperation** -- Collect 200+ demonstrations of grasping the pen via teleoperation (e.g., SpaceMouse or VR controller)
2. **Variation axes** to cover:
   - Pen position on the table (uniform sampling over workspace)
   - Pen orientation (upright, tilted 0-45 deg, lying flat)
   - Initial robot configuration (randomized joint positions)
3. **Target dataset size:** 200-500 demonstrations
4. **Recording format:** HDF5 matching the Robomimic schema

### 4.2 Data Augmentation

- **State:** Add Gaussian noise (sigma=0.005) to pen position observations
- **Image:** Random crops (76x76 from 84x84), color jitter, random erasing

---

## 5. Training Plan

### 5.1 Phase 1: State-Based Policy (Proof of Concept)

```bash
python examples/train_robomimic.py \
  task=pen_grasp_state \
  network=mlp \
  optimization.batch_size=256 \
  optimization.gradient_steps=300000 \
  optimization.lr=1e-4 \
  optimization.loss_type=flow \
  optimization.num_steps=3
```

**Key hyperparameters:**
- Network: MLP (4 layers, 512 emb_dim) -- sufficient for state-based grasping
- Flow matching with 3-step Euler sampling at inference
- EMA rate 0.995 for stable training
- Batch size 256 (smaller dataset than lift)

### 5.2 Phase 2: Image-Based Policy (Sim-to-Real Ready)

```bash
python examples/train_robomimic.py \
  task=pen_grasp_image \
  network=mlp \
  optimization.batch_size=128 \
  optimization.gradient_steps=500000 \
  optimization.lr=1e-4
```

**Key changes from Phase 1:**
- Uses `MultiImageObsEncoder` (ResNet18 backbone) for image encoding
- Two cameras: agentview + eye-in-hand
- Larger gradient steps due to higher-dimensional observation space
- Smaller batch size to fit GPU memory with image processing

### 5.3 Phase 3: Architecture Comparison

Once the baseline MLP policy works, evaluate alternative architectures:

| Network           | Expected Benefit                      | Config              |
|-------------------|---------------------------------------|---------------------|
| `mlp`             | Fast training, good baseline          | `network=mlp`       |
| `vanilla_mlp`     | Simpler, less overfitting risk        | `network=vanilla_mlp`|
| `chitransformer`  | Better temporal reasoning             | `network=chitransformer` |
| `chiunet`         | Better for multi-modal action dists   | `network=chiunet`   |

---

## 6. Evaluation Protocol

### 6.1 Success Criteria

A grasp is **successful** if:
1. The gripper makes contact with the pen body (not just the cap)
2. The pen is lifted at least 5 cm above the table surface
3. The pen remains in the gripper for at least 1 second after lifting
4. No excessive force is applied (no marker deformation in sim)

### 6.2 Evaluation Metrics

| Metric                 | Target       |
|------------------------|--------------|
| Grasp success rate     | > 85%        |
| Average completion time| < 8 seconds  |
| Topple rate            | < 10%        |
| Position error at grasp| < 5 mm       |

### 6.3 Evaluation Command

```bash
python examples/train_robomimic.py \
  task=pen_grasp_state \
  mode=eval \
  optimization.model_path=checkpoints/pen_grasp_best.pt \
  log.eval_episodes=50
```

---

## 7. Trajectory Decomposition

The complete grasp trajectory decomposes into these phases, aligned with MIP's temporal horizon (`obs_steps=2`, `act_steps=8`, `horizon=16`):

```
Time ──────────────────────────────────────────────►

│◄── Reach (steps 0-5) ──►│◄─ Align (6-9) ─►│◄─ Grasp (10-13) ─►│◄─ Lift (14-16) ─►│
│                          │                  │                    │                   │
│  Move EEF above pen      │  Descend to      │  Close gripper     │  Lift pen up      │
│  Open gripper wide       │  pre-grasp pose   │  Confirm contact   │  Stabilize        │
```

Each inference step predicts 8 actions (`act_steps=8`) but executes all 8 before re-planning, giving the policy a 16-step effective horizon that covers the full reach-grasp-lift sequence.

---

## 8. Sim-to-Real Considerations

For transferring the trained policy to a real robot grasping the physical EXPO marker:

1. **Domain randomization** in simulation:
   - Randomize pen color, size (+-20%), friction coefficient
   - Randomize table texture, lighting conditions
   - Randomize camera positions (+-2cm, +-5 deg)

2. **Real-world adaptations:**
   - Use eye-in-hand camera as primary input (more invariant to scene changes)
   - Calibrate gripper force to avoid crushing the marker
   - Add compliance/force control for the final grasp phase

3. **Safety constraints:**
   - Limit EEF velocity to 10 cm/s during approach
   - Limit gripper force to 5N (sufficient for 20g marker)
   - Workspace bounds to prevent table collisions

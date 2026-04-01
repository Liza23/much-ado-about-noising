# Model Checkpoints for Multimodality Evaluation

Three variants per task: **baseline** (no intent), **flow_intent** (Config A: flow-based intent model), **hierarchical_emb** (Config B: MLP intent predictor with learned 64D embedding).

---

## square-ph-image

| Variant | Checkpoint | Success |
|---------|-----------|---------|
| baseline | `square_ph_image_flow_mlp_512_h10_seed0_success96.pt` | 96% |
| flow_intent | `square_ph_image_flow_mlp_512_h10_seed0_intent_flow_intent_success90.pt` | 90% |
| hierarchical_emb | `square_ph_image_flow_mlp_512_h10_seed0_intent_learned_joint_emb_success94.pt` + `_hl_policy.pt` | 94% |

## square-ph-state

| Variant | Checkpoint | Success |
|---------|-----------|---------|
| baseline | `square_ph_state_flow_mlp_512_h10_seed0_success88.pt` | 88% |
| flow_intent | `square_ph_state_flow_mlp_512_h10_seed0_intent_flow_intent_success98.pt` | 98% |
| hierarchical_emb | `square_ph_state_flow_mlp_512_h10_seed0_intent_learned_joint_emb_success92.pt` + `_hl_policy.pt` | 92% |

## lift-ph-state

| Variant | Checkpoint | Success |
|---------|-----------|---------|
| baseline | `lift_ph_state_flow_mlp_512_h16_seed0_success0.pt` | 0% |
| flow_intent | `lift_ph_state_flow_mlp_512_h16_seed0_intent_flow_intent_success100.pt` | 100% |
| hierarchical_emb | `lift_ph_state_flow_mlp_512_h16_seed0_intent_learned_joint_emb_success98.pt` + `_hl_policy.pt` | 98% |

## lift-ph-image

| Variant | Checkpoint | Success |
|---------|-----------|---------|
| baseline | `lift_ph_image_flow_mlp_512_h16_seed0_success100.pt` | 100% |
| flow_intent | `lift_ph_image_flow_mlp_512_h16_seed0_intent_flow_intent_success100.pt` | 100% |
| hierarchical_emb | `lift_ph_image_flow_mlp_512_h16_seed0_intent_learned_joint_emb_success100.pt` + `_hl_policy.pt` | 100% |

## lift-mh-state

| Variant | Checkpoint | Success |
|---------|-----------|---------|
| baseline | `lift_mh_state_flow_mlp_512_h10_seed0_success100.pt` | 100% |
| flow_intent | `lift_mh_state_flow_mlp_512_h10_seed0_intent_flow_intent_success100.pt` | 100% |
| hierarchical_emb | `lift_mh_state_flow_mlp_512_h10_seed0_intent_learned_joint_emb_success98.pt` + `_hl_policy.pt` | 98% |

---

## Hydra Task Config Mapping

| Task | Baseline config | flow_intent config | hierarchical_emb config |
|------|----------------|-------------------|------------------------|
| square-ph-image | `task=square_ph_image` | `task=square_ph_image_flow_intent` + `+network.arch_variant=flow_intent` | `task=square_ph_image_hierarchical_emb` |
| square-ph-state | `task=square_ph_state` | `task=square_ph_state_flow_intent` + `+network.arch_variant=flow_intent` | `task=square_ph_state_hierarchical_emb` |
| lift-ph-state | `task=lift_ph_state` | `task=lift_ph_state_flow_intent` + `+network.arch_variant=flow_intent` | `task=lift_ph_state_hierarchical_emb` |
| lift-ph-image | `task=lift_ph_image` | `task=lift_ph_image_flow_intent` + `+network.arch_variant=flow_intent` | `task=lift_ph_image_hierarchical_emb` |
| lift-mh-state | `task=lift_mh_state` | `task=lift_mh_state_flow_intent` + `+network.arch_variant=flow_intent` | `task=lift_mh_state_hierarchical_emb` |

## Notes

- `_hl_policy.pt` files are sidecar checkpoints for the intent predictor (hierarchical_emb only)
- flow_intent variants require `+network.arch_variant=flow_intent` override (uses `FlowIntentAgent` instead of `TrainingAgent`)
- All checkpoints use MLP network with emb_dim=512
- lift tasks use h16 (horizon=16), square/lift-mh use h10 (horizon=10)

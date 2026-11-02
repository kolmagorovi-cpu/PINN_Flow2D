# FVM continuity mean-square objective

The canonical FVM loss is the spatial, volume-weighted mean square of the
discrete divergence. For control volume `i`,

\[
R_i = \frac{\Phi_i}{V_i}
    = \frac{1}{V_i}\sum_{q\in\partial V_i}\mathbf u_\theta(\mathbf x_q)\cdot\mathbf S_{iq},
\qquad
L_{\mathrm{FVM}} =
\frac{\sum_i V_i R_i^2}{\sum_i V_i}.
\]

The velocity is evaluated by the PINN directly at the midpoint of every
straight median-dual face segment.  An internal dual face normally contains
two segments (cell centroid to primal-edge midpoint), so it has two network
quadrature evaluations.  Nodal velocity averaging is not used by production
training.  Each internal segment flux is scattered to its two control volumes
with opposite signs.

Boundary half-edges are evaluated at their own midpoints.  Prescribed inlet
and velocity-Dirichlet values, and zero-normal-flux wall/symmetry conditions,
still override the raw network value when assembling the physical flux.

The production reduction retains the historical name
`divergence_l2_volume`, but the optimized value is the squared L2 quantity
(MSE), not its square root. Its square root is reported separately as
`fvm_rms`. The MSE is exposed as `fvm_mse` (`fvm_raw_loss` remains a
compatibility alias).

`flux_balance_l2` measures the absolute face-flux imbalance instead:

\[
\frac{\sum_i \Phi_i^2}{\sum_i V_i}.
\]

The historical name `volume_weighted` retains this second formula for
reproducibility. It is a legacy/diagnostic reduction and must not be interpreted
as a divergence norm.

The reference configuration is:

```yaml
loss_weights:
  fvm:
    enabled: true
    weight: 25.0
    forward_chunk_size: 32768
    activation_checkpoint: true
    reduction: divergence_l2_volume
    schedule: every_optimizer_step
    batch_interval: 1
```

The direct face forward is chunked and uses activation checkpointing by
default.  These settings change memory/time only; they do not change the FVM
operator or its checkpoint identity.

`every_optimizer_step` performs one full-mesh FVM forward per optimizer update,
including when several mini-batches form one update through gradient
accumulation. `every_k_optimizer_steps` and `stochastic_optimizer_step` use the
explicit interval `K` and multiply an included FVM term by `K`; the stochastic
mode therefore preserves the full-schedule gradient in expectation.

An enabled FVM loss is fail-fast. Construction and every RAR rebuild reject
non-finite or non-positive volumes, invalid face indices, non-finite or zero
area vectors, unknown boundary types, orphan control volumes, and
control-volume area vectors that do not geometrically close. Missing mesh data
or a mismatch between operator nodes and FVM input tensors also aborts the run.

Checkpoint training contracts include the reduction, configured weight,
schedule, interval/probability, mesh identifier and SHA-256 fingerprint,
normalization rule, and dimensional state. A mismatch rejects exact `resume`;
an explicitly selected `finetune` still loads model weights only.

Per-epoch telemetry records `fvm_mse`, `fvm_raw_loss`,
`fvm_weighted_loss`, `fvm_rms`, and `fvm_evaluations` separately.

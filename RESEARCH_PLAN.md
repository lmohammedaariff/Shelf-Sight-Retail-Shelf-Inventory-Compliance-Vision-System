# M.Tech research plan (proposed, not a finding)

## Working question

Can confidence-aware SKU decisions and explicit abstention reduce false compliance decisions under package occlusion and store/lighting variation, while keeping a useful fraction of shelf slots automatically accepted?

This question is a proposal. No novelty or research gap has been established. A literature review and dataset feasibility check are required before presenting it as a contribution.

## Initial literature map

- Goldman et al., **Precise Detection in Densely Packed Scenes** (SKU-110K), CVPR 2019. It studies dense retail package detection and introduces an annotated packed-retail dataset and overlap-aware detection ideas. Its dense-object regime is relevant to shelf images, but the task/data are not automatically the same as planogram compliance. [CVF Open Access paper](https://openaccess.thecvf.com/content_CVPR_2019/html/Goldman_Precise_Detection_in_Densely_Packed_Scenes_CVPR_2019_paper.html)
- Wei et al., **RPC: A Large-Scale Retail Product Checkout Dataset**, initially released as a 2019 preprint and later journal publication. It includes single-product and multi-product checkout imagery with retail product labels. Checkout scenes and shelf arrangements differ; inspect the original [paper](https://arxiv.org/abs/1901.07249), [project page](https://rpc-dataset.github.io/), and current license/access terms before use.
- Guo et al., **On Calibration of Modern Neural Networks**, ICML 2017. It motivates empirical calibration evaluation rather than treating neural network scores as correctness probabilities. [PMLR proceedings](https://proceedings.mlr.press/v70/guo17a.html)
- Fisch et al., **Calibrated Selective Classification**, 2022. It studies abstention/selective prediction with calibrated uncertainty over accepted examples. This supports comparing risk and coverage rather than only top-1 accuracy. [arXiv paper](https://arxiv.org/abs/2208.12084)

This is an initial reading list, not a systematic review. Search additional work on shelf-planogram compliance, domain shift, retail facing counts, occlusion, and dataset governance. Compare protocols and data licenses before selecting a benchmark.

## Hypothesis and comparison plan

**Hypothesis to test:** choosing an acceptance threshold using validation data and routing low-score/unknown crops to human review may reduce false-compliant decisions on a held-out store/condition split compared with always accepting top-1 SKU predictions. The trade-off is lower automatic coverage.

Baseline: OpenCV region proposals plus the trained MobileNetV2 classifier with top-1 SKU used at every candidate. Proposed decision policy: same model and candidates, but low-score predictions produce `REVIEW_REQUIRED`; evaluate coverage, accepted-subset error, false compliance, and review load at thresholds selected on validation data. Optionally compare calibrated scores after a literature-supported method, but fit it on validation data only.

This is a decision-policy comparison, not a new detector architecture. It is not yet a novel contribution; similar selective classification/calibration methods already exist.

## Experimental protocol

1. Obtain permitted whole-shelf images from multiple stores, shelf layouts, lighting conditions, occlusion levels, and visits; record dataset provenance and rights.
2. Annotate SKU boxes, row IDs, expected planogram slots, unknown products, and independent compliant/non-compliant outcomes. Use a documented double-label/reconciliation procedure.
3. Split by store or collection session to reduce leakage. Keep a frozen held-out test set.
4. Train the same SKU classifier and use the same proposal outputs in each policy comparison.
5. Report per-class precision/recall/F1, detection IoU metrics, visible-count MAE, compliance false-compliance/false-violation rate, review coverage, accepted error, calibration plots/ECE/Brier where appropriate, and latency with hardware/software versions.
6. Analyze failure examples for touching products, occlusion, packaging changes, low light, unknown SKUs, shelf-edge proposals, and geometry mismatch.
7. Report confidence intervals where sample sizes support them; state the unit of analysis and class/store imbalance.

## Limits and current status

The implementation presently supplies a classical proposal baseline, a transfer-learning SKU training pipeline, raw-score threshold review, a configured slot matcher, and evaluation tools. No annotated dataset or model has been supplied, so the hypothesis cannot yet be tested and no results are reported. Dataset access/licensing, annotation cost, representativeness, label quality, calibration, and distribution shift remain open risks.

# Reproducible runs: Archive CodeCarbon logs and Model Card

This repository includes a lightweight GitHub Actions workflow and a sample README snippet to automate carbon logging with CodeCarbon and archive the generated model card and carbon logs after each run.

## Usage
1. Ensure your training script creates `codecarbon.json` output (CodeCarbon logger) and `model_card.md` in the repository root (or `docs/`).
2. Commit the provided workflow file to `.github/workflows/archive_carbon.yml` to enable automatic archival on push or on-demand via workflow_dispatch.

## What is archived
- `logs/codecarbon.json` (CodeCarbon output)
- `docs/model_card.md` and `model_card.pdf` (generated model card)

## Developer note
Adjust paths in the workflow to match your training script outputs.

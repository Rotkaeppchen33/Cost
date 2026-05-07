# Anonymous Submission Code Release

The codebase builds on an existing open-source training stack and adds preprocessing and analysis scripts used for the paper's empirical study.

## Setup

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Running Experiments

Training scripts are located in `src/`.

Example single run:

```bash
python3 src/ppo_continuous_action.py \
  --alg_type lambda_ac \
  --env_name ant \
  --actor_lr 1e-4 \
  --critic_lr 1e-4 \
  --gae_lambda 0.5 \
  --ent_coef 0.01 \
  --sweep_idx -1
```

Main algorithm names used in this repository:

- `lambda_ac`: PPO
- `advn_norm_mean`: per-minibatch zero-mean normalization

Supported environments in this code release:

- `ant`
- `hopper`
- `halfcheetah`
- `walker2d`
- `swimmer`

If `--sweep_idx` is not `-1`, the script uses the hyperparameter configuration indexed by the sweep definition instead of the values passed manually through argparse.

## Analysis And Post-Processing

- `sec5/` contains post-processing scripts used to organize raw reward traces, compute Effective AUC and SC, and build summary tables for Section 5.

The recommended entry point for the Section 5 preprocessing pipeline is `sec5/run_preproc.py`. The numbered scripts under `sec5/Use case I/` and `sec5/Use case II/` are kept as step-by-step scripts for reproducing intermediate processing stages.

## Path Handling

This anonymous release intentionally does not auto-create a complete end-to-end directory tree for the full pipeline.

The post-processing steps consume different user-provided input and output roots, and those paths depend on how each user stores raw logs locally. 

The repository follows this convention:

- input directories should be set by the user to match local storage layout
- output directories passed explicitly to scripts are created when needed
- some scripts in `sec5/Use case I/` and `sec5/Use case II/` still contain placeholder paths such as `"/path/..."` or `"/postproc_results"` and should be edited before use

## How To Run `run_preproc.py`

`run_preproc.py` is a unified post-processing script for generating `N={0,1,2,3,4}` hyperparameter summary tables under progressive hyperparameter release.

It expects:

- `--input_folder`: a directory containing `N=4 hyperparam_summary_{env}_{alg}.csv` and `global_quantiles_{env}.csv`
- `--rewards_folder`: a directory containing per-configuration reward CSVs named like `actorlr_<...>_criticlr_<...>_entcoef_<...>_gaelambda_<...>_env_<env>_alg_<alg>.csv`
- `--output_folder`: where the filtered and recomputed summary CSVs will be written

Example:

```bash
python3 sec5/run_preproc.py \
  --N 2 \
  --input_folder /path/to/postproc_results \
  --rewards_folder /path/to/postproc_results \
  --output_folder /path/to/preproc_outputs
```

Important details:

- `--N` is the number of released hyperparameters and must be in `{0,1,2,3,4}`
- the script uses the `N=4` summary files both to identify the global best hyperparameters and to load the source summaries before filtering
- the script creates `--output_folder` automatically, but it does not create missing upstream inputs
- the default environment list is `hopper,swimmer,halfcheetah,walker2d,ant`; this can be overridden with `--envs`

## Attribution

This repository includes code built on top of existing open-source projects, especially:

- [PureJaxRL](https://github.com/luchris429/purejaxrl)
- [Brax](https://github.com/google/brax)
- [PyExpUtils](https://github.com/andnp/PyExpUtils)

To preserve anonymous review, submission-specific citation metadata is intentionally omitted from this README.

## Notes

- Large raw training logs are not included in this repository.
- Some directories and file paths have been anonymized for blind review.
- Users should adapt path placeholders to their local machine before running the post-processing scripts.

## License

See `LICENSE`.

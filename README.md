![Project banner](docs/source/_static/mjlab-banner.jpg)

# mjlab

[![GitHub Actions](https://img.shields.io/github/actions/workflow/status/mujocolab/mjlab/ci.yml?branch=main)](https://github.com/mujocolab/mjlab/actions/workflows/ci.yml?query=branch%3Amain)
[![Documentation](https://github.com/mujocolab/mjlab/actions/workflows/docs.yml/badge.svg)](https://mujocolab.github.io/mjlab/)
[![License](https://img.shields.io/github/license/mujocolab/mjlab)](https://github.com/mujocolab/mjlab/blob/main/LICENSE)
[![Nightly Benchmarks](https://img.shields.io/badge/Nightly-Benchmarks-blue)](https://mujocolab.github.io/mjlab/nightly/)
[![PyPI](https://img.shields.io/pypi/v/mjlab)](https://pypi.org/project/mjlab/)

mjlab combines [Isaac Lab](https://github.com/isaac-sim/IsaacLab)'s manager-based API with [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp), a GPU-accelerated version of [MuJoCo](https://github.com/google-deepmind/mujoco).
The framework provides composable building blocks for environment design,
with minimal dependencies and direct access to native MuJoCo data structures.

## Getting Started

mjlab requires an NVIDIA GPU for training. macOS is supported for evaluation only.

**Try it now:**

Run the demo (no installation needed):

```bash
uvx --from mjlab --refresh demo
```

Or try in [Google Colab](https://colab.research.google.com/github/mujocolab/mjlab/blob/main/notebooks/demo.ipynb) (no local setup required).

**Install from source:**

```bash
git clone https://github.com/mujocolab/mjlab.git && cd mjlab
uv run demo
```

For alternative installation methods (PyPI, Docker), see the [Installation Guide](https://mujocolab.github.io/mjlab/source/installation.html).

## Training Examples

### 1. Velocity Tracking

Train a Unitree G1 humanoid to follow velocity commands on flat terrain:

```bash
uv run train Mjlab-Velocity-Flat-Unitree-G1 --env.scene.num-envs 4096
```

**Multi-GPU Training:** Scale to multiple GPUs using `--gpu-ids`:

```bash
uv run train Mjlab-Velocity-Flat-Unitree-G1 \
  --gpu-ids 0 1 \
  --env.scene.num-envs 4096
```

See the [Distributed Training guide](https://mujocolab.github.io/mjlab/source/distributed_training.html) for details.

Evaluate a policy while training (fetches latest checkpoint from Weights & Biases):

```bash
uv run play Mjlab-Velocity-Flat-Unitree-G1 --wandb-run-path your-org/mjlab/run-id
```

### 2. Motion Imitation

Train a humanoid to mimic reference motions. mjlab uses WandB to manage motion datasets.
See the [motion preprocessing documentation](https://github.com/HybridRobotics/whole_body_tracking/blob/main/README.md#motion-preprocessing--registry-setup) for setup instructions.

```bash
uv run train Mjlab-Tracking-Flat-Unitree-G1 --registry-name your-org/motions/motion-name --env.scene.num-envs 4096
uv run play Mjlab-Tracking-Flat-Unitree-G1 --wandb-run-path your-org/mjlab/run-id
```

### 3. Go2 Motion Tracking (local npz)

Train / play Unitree Go2 tracking with a local motion file under `motions/`.
Training is headless by default (`MUJOCO_GL=egl`). Logs and TensorBoard events
go to `logs/rsl_rl/go2_tracking/`.

#### Demo Videos

Learned Go2 skills (files under [`videos/`](videos/)):

<table>
  <tr>
    <td align="center">
      <video src="
https://github.com/user-attachments/assets/800a0da9-42c4-4cb2-a20b-dcb2e0fa96f7


https://github.com/user-attachments/assets/b881475f-5b98-43eb-b6d9-6607835460de

" controls width="280"></video>
      <br />Backflip
    </td>
    <td align="center">
      <video src="

https://github.com/user-attachments/assets/39d54ec3-1454-4242-8abe-e9e5388f7281

" controls width="280"></video>
      <br />Front flip
    </td>
    <td align="center">
      <video src="
https://github.com/user-attachments/assets/a7419df5-61e3-4dad-9d27-cb5a811a656c
" controls width="280"></video>
      <br />Left flip
    </td>
    <td align="center">
      <video src="
https://github.com/user-attachments/assets/8e613766-5154-401c-acf8-30e16f4333f2
" controls width="280"></video>
      <br />Right flip
    </td>
  </tr>
  <tr>
    <td align="center">
      <video src="
https://github.com/user-attachments/assets/e52d7050-4e29-4154-b262-4362de74b10c
" controls width="280"></video>
      <br />Jump up
    </td>
    <td align="center">
      <video src="
https://github.com/user-attachments/assets/6671fdd5-c6d3-4ec1-9518-9cd5b92cce30
" controls width="280"></video>
      <br />Jump forward
    </td>
    <td align="center">
      <video src="
https://github.com/user-attachments/assets/a85a9024-c823-4b3a-ad54-ea8cc9d7476c
" controls width="280"></video>
      <br />Jump back
    </td>
    <td></td>
  </tr>
</table>

**Train three long-jump motions sequentially** (4096 envs, 10000 iterations each):

```bash
cd /path/to/mjlab-dev-go2-mimic

for motion in \
  go2_longjump_0p0 \
  go2_longjump_0p7 \
  go2_longjump_m0p5
do
  echo "========== Training $motion =========="
  uv run python -m mjlab.scripts.train \
    Mjlab-Tracking-Flat-Unitree-Go2-No-State-Estimation \
    --env.commands.motion.motion-file "$PWD/motions/${motion}.npz" \
    --env.scene.num-envs 4096 \
    --agent.max-iterations 10000 \
    --agent.run-name "$motion"
done
```

With state estimation in the actor obs, use task id
`Mjlab-Tracking-Flat-Unitree-Go2` instead.

**Play** (auto-picks name-sorted latest ``model_*.pt``).
``--keep-randomization True`` keeps training-time DR and defaults to **50 envs**:

```bash
uv run python -m mjlab.scripts.play \
  Mjlab-Tracking-Flat-Unitree-Go2-No-State-Estimation \
  --motion-file $PWD/motions/go2_longjump_m0p5.npz \
  --keep-randomization True
```

Override env count if needed: ``--num-envs 16``. Or use the helper:

```bash
uv run play-go2-rand --motion-file $PWD/motions/go2_longjump_m0p5.npz
```

Specify a checkpoint explicitly:

```bash
uv run python -m mjlab.scripts.play \
  Mjlab-Tracking-Flat-Unitree-Go2-No-State-Estimation \
  --motion-file $PWD/motions/go2_longjump_m0p5.npz \
  --checkpoint-file logs/rsl_rl/go2_tracking/<run>/model_10000.pt \
  --keep-randomization True
```

G1 example:

```bash
uv run python -m mjlab.scripts.play \
  Mjlab-Tracking-Flat-Unitree-G1-No-State-Estimation \
  --motion-file $PWD/motions/g1_fallandgetup1_850_940.npz \
  --keep-randomization True \
  --export-torchscript True
```

**Export TorchScript `policy.pt`** (for C++ deploy; training also writes
`policy.pt` next to each checkpoint on save):

```bash
# While playing:
uv run python -m mjlab.scripts.play \
  Mjlab-Tracking-Flat-Unitree-Go2-No-State-Estimation \
  --motion-file $PWD/motions/go2_longjump_m0p5.npz \
  --checkpoint-file logs/rsl_rl/go2_tracking/<run>/model_10000.pt \
  --export-torchscript True

# Or convert only:
uv run export-tracking-pt \
  --checkpoint-file logs/rsl_rl/go2_tracking/<run>/model_10000.pt \
  --motion-file $PWD/motions/go2_longjump_m0p5.npz
```

TensorBoard:

```bash
tensorboard --logdir logs/rsl_rl/go2_tracking
```

### 4. Sanity-check with Dummy Agents

Use built-in agents to sanity check your MDP before training:

```bash
uv run play Mjlab-Your-Task-Id --agent zero  # Sends zero actions
uv run play Mjlab-Your-Task-Id --agent random  # Sends uniform random actions
```

When running motion-tracking tasks, add `--registry-name your-org/motions/motion-name` to the command.


## Community Projects

mjlab is used for research and robotics applications around the world. Examples:

<table>
  <tr>
    <td>
      <a href="https://github.com/menloresearch/asimov-mjlab">
        menloresearch/asimov-mjlab
        <br /><img
          alt="GitHub stars"
          src="https://img.shields.io/github/stars/menloresearch/asimov-mjlab?style=social"
        />
      </a>
    </td>
    <td>Locomotion fork for the Asimov bipedal robot.</td>
  </tr>
  <tr>
    <td>
      <a href="http://husky-humanoid.github.io/">
        HUSKY
      </a>
      <br />
      <a href="https://github.com/mujocolab/mjlab/discussions/572">#572</a>
      ·
      <a href="https://arxiv.org/abs/2602.03205">Paper</a>
    </td>
    <td>
      Humanoid skateboarding with dynamic balance control.
    </td>
  </tr>
  <tr>
    <td>
      <a href="https://github.com/Nagi-ovo/mjlab-homierl">
        Nagi-ovo/mjlab-homierl
        <br /><img
          alt="GitHub stars"
          src="https://img.shields.io/github/stars/Nagi-ovo/mjlab-homierl?style=social"
        />
      </a>
    </td>
    <td>Multi-task H1 locomotion (walk/squat/stand) with upper-body disturbance robustness.</td>
  </tr>
  <tr>
    <td>
      <a href="https://github.com/MyoHub/mjlab_myosuite">
        MyoHub/mjlab_myosuite
        <br /><img
          alt="GitHub stars"
          src="https://img.shields.io/github/stars/MyoHub/mjlab_myosuite?style=social"
        />
      </a>
    </td>
    <td>Musculoskeletal simulation integration with MyoSuite.</td>
  </tr>
  <tr>
    <td>
      <a href="https://github.com/MarcDcls/mjlab_upkie">
        MarcDcls/mjlab_upkie
        <br /><img
          alt="GitHub stars"
          src="https://img.shields.io/github/stars/MarcDcls/mjlab_upkie?style=social"
        />
      </a>
    </td>
    <td>Velocity control for the Upkie wheeled biped.</td>
  </tr>
  <tr>
    <td>
      <a href="https://github.com/unitreerobotics/unitree_rl_mjlab">
        unitreerobotics/unitree_rl_mjlab
        <br /><img
          alt="GitHub stars"
          src="https://img.shields.io/github/stars/unitreerobotics/unitree_rl_mjlab?style=social"
        />
      </a>
    </td>
    <td>Official Unitree RL environments for Go2, G1, and H1_2.</td>
  </tr>
</table>

Want to share your project? Post in [Show and Tell](https://github.com/mujocolab/mjlab/discussions/categories/show-and-tell)!

## Documentation

Full documentation is available at **[mujocolab.github.io/mjlab](https://mujocolab.github.io/mjlab/)**.

## Development

```bash
make test          # Run all tests
make test-fast     # Skip slow tests
make format        # Format and lint
make docs          # Build docs locally
```

For development setup: `uvx pre-commit install`

## Citation

If you use mjlab in your research, please cite:

```bibtex
@misc{zakka2026mjlablightweightframeworkgpuaccelerated,
  title={mjlab: A Lightweight Framework for GPU-Accelerated Robot Learning},
  author={Kevin Zakka and Qiayuan Liao and Brent Yi and Louis Le Lay and Koushil Sreenath and Pieter Abbeel},
  year={2026},
  eprint={2601.22074},
  archivePrefix={arXiv},
  primaryClass={cs.RO},
  url={https://arxiv.org/abs/2601.22074},
}
```

## License

mjlab is licensed under the [Apache License, Version 2.0](LICENSE).

### Third-Party Code

Some portions of mjlab are forked from external projects:

- **`src/mjlab/utils/lab_api/`** — Utilities forked from [NVIDIA Isaac
  Lab](https://github.com/isaac-sim/IsaacLab) (BSD-3-Clause license, see file
  headers)

Forked components retain their original licenses. See file headers for details.

## Acknowledgments

mjlab wouldn't exist without the excellent work of the Isaac Lab team, whose API
design and abstractions mjlab builds upon.

Thanks to the MuJoCo Warp team — especially Erik Frey and Taylor Howell — for
answering our questions, giving helpful feedback, and implementing features
based on our requests countless times.

Go2 agile reference motions (flips, jumps, etc.) are generated with
[se3_trajopt](https://github.com/Renkunzhao/se3_trajopt).
Tracking training and demos for this work live on the
[dev/go2-mimic](https://github.com/Renkunzhao/mjlab/tree/dev/go2-mimic)
branch.

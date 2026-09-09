.. _observation:

Observation History and Delay
=============================

Observations have two temporal features: history and delay. History stacks past
frames for temporal context, while delay can be used to model sensor latency.

TL;DR
-----

**Add history to stack frames:**

.. code-block:: python

    from mjlab.managers.observation_manager import ObservationTermCfg

    joint_vel: ObservationTermCfg = ObservationTermCfg(
        func=joint_vel,
        history_length=5,        # Keep last 5 frames
        flatten_history_dim=True # Flatten for MLP: (12,) * 5 = (60,)
    )


**Add delay to model sensor latency:**

.. code-block:: python

    # At 50Hz control (20ms/step): lag=2-3 → 40-60ms latency
    camera: ObservationTermCfg = ObservationTermCfg(
        func=camera_obs,
        delay_min_lag=2,
        delay_max_lag=3,
    )


**Combine both:**

.. code-block:: python

    joint_pos: ObservationTermCfg = ObservationTermCfg(
        func=joint_pos,
        delay_min_lag=1,
        delay_max_lag=3,      # Delayed observations
        history_length=5,     # Stack 5 delayed frames
        flatten_history_dim=True
    )
    # Pipeline: compute → delay → stack → flatten


Observation History
-------------------

History stacks past observations to provide temporal context.

Basic Usage
^^^^^^^^^^^

**Flattened history (for MLPs):**

.. code-block:: python

    joint_vel: ObservationTermCfg = ObservationTermCfg(
        func=joint_vel,           # Returns (num_envs, 12)
        history_length=3,
        flatten_history_dim=True  # Output: (num_envs, 36)
    )


**Structured history (for RNNs):**

.. code-block:: python

    joint_vel: ObservationTermCfg = ObservationTermCfg(
        func=joint_vel,            # Returns (num_envs, 12)
        history_length=3,
        flatten_history_dim=False  # Output: (num_envs, 3, 12)
    )


Group-Level Override
^^^^^^^^^^^^^^^^^^^^

Apply history to all terms in a group:


.. code-block:: python

    @dataclass
    class PolicyCfg(ObservationGroupCfg):
        concatenate_terms: bool = True
        history_length: int = 5           # Applied to all terms
        flatten_history_dim: bool = True

        joint_pos: ObservationTermCfg = ObservationTermCfg(func=joint_pos)
        joint_vel: ObservationTermCfg = ObservationTermCfg(func=joint_vel)
        # Both terms get 5-frame history, flattened


Term-level settings override group settings:


.. code-block:: python

    @dataclass
    class PolicyCfg(ObservationGroupCfg):
        history_length: int = 3  # Default for group

        joint_pos: ObservationTermCfg = ObservationTermCfg(
            func=joint_pos,
            history_length=5  # Override: use 5 instead of 3
        )



Reset Behavior
^^^^^^^^^^^^^^

History buffers are cleared on environment reset. The first observation after
reset is backfilled across all history slots, ensuring valid data from step 0.


.. code-block:: python

    # At reset
    buffer = [obs_0, obs_0, obs_0]  # Backfilled

    # After 2 steps
    buffer = [obs_0, obs_1, obs_2]  # Normal accumulation


History Flattening Order (Term-Major vs Time-Major)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When ``flatten_history_dim=True`` and ``concatenate_terms=True``, mjlab uses
**term-major** ordering, where each term's full history is flattened before
concatenating terms:


.. code-block:: bash

    Term A: shape (num_envs, obs_dim_A) with history_length=3
    Term B: shape (num_envs, obs_dim_B) with history_length=3

    mjlab output (TERM-MAJOR):
    [A_t0, A_t1, A_t2, B_t0, B_t1, B_t2, ...]
     └─ all A history ─┘  └─ all B history ─┘


An alternative approach is **time-major** (or frame-major) ordering, where
complete observation frames are built at each timestep before concatenating
across time:


.. code-block:: bash

    TIME-MAJOR (alternative approach):
    [A_t0, B_t0, ..., A_t1, B_t1, ..., A_t2, B_t2, ...]
     └─ frame t0 ──┘     └─ frame t1 ──┘     └─ frame t2 ──┘


**Sim2sim compatibility:** If you need to transfer policies to/from frameworks
that use time-major ordering, you will need to reorder observations. This
affects policies trained with history but not those without.

Observation Delay
-----------------

Real robots have sensors with communication delays (WiFi, USB). The delay system
models sensor latency by returning observations from earlier timesteps.

Delay Parameters
^^^^^^^^^^^^^^^^

``delay_min_lag`` / ``delay_max_lag`` (default: 0) Lag range in steps. Uniformly
samples an integer lag from ``[min_lag, max_lag]`` (both inclusive).
``lag=0`` means current observation, ``lag=2`` means 2 steps ago.

``delay_per_env`` (default: True) If True, each environment gets a different
lag. If False, all environments share the same lag.

``delay_hold_prob`` (default: 0.0)
Probability [0, 1] of keeping the previous lag instead of resampling.

``delay_update_period`` (default: 0) How often (in steps) to resample the lag.
If 0, resample every step. If N > 0, the lag value stays constant for N steps
before being resampled (creates temporally correlated latency patterns).

``delay_per_env_phase`` (default: True) If True and ``delay_update_period > 0``,
stagger resample timing across environments with random phase offsets.

.. note::

   ``delay_update_period`` controls how often the *lag value* is resampled, not
   how often observations are refreshed. You still get a new (delayed) observation
   every step - the lag just stays constant for N steps before being resampled.

**Visualizing delay (50Hz control = 20ms/step):**

.. code-block:: bash

    Sensor captures:  A     B     C     D     E     F     G     H
                      ↓     ↓     ↓     ↓     ↓     ↓     ↓     ↓
    Control steps:    0     1     2     3     4     5     6     7
                     20ms  40ms  60ms  80ms  100ms 120ms 140ms 160ms

    No delay (baseline - perfect sensor):
    You receive:      A     B     C     D     E     F     G     H
                      ↑ current observation every step

    Delay with lag=2:
    You receive:      A     A     A     B     C     D     E     F
                      ↑clamp↑     ↑     ↑     ↑     ↑     ↑     ↑
                    Steps 0-1: lag clamped (buffer not full yet)
                    Step 2+: 40ms delay, every step gets NEW observation


**Example - Camera with 40-60ms latency at 50Hz control:**


.. code-block:: python

    camera: ObservationTermCfg = ObservationTermCfg(
        func=camera_obs,
        delay_min_lag=2,  # 40ms latency
        delay_max_lag=3,  # 60ms latency
    )

Computing Delays from Real-World Latency
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Convert real-world latency to simulation steps:


delay_steps = latency_ms / (1000 / control_hz)


**Example at 50Hz control (20ms per step):**
- 40ms latency = 40 / 20 = 2 steps
- 60ms latency = 60 / 20 = 3 steps
- 100ms latency = 100 / 20 = 5 steps

**Example at 100Hz control (10ms per step):**
- 40ms latency = 40 / 10 = 4 steps
- 60ms latency = 60 / 10 = 6 steps

.. note::

     Delays are quantized to control timesteps. At 50Hz control (20ms/step),
     you can only represent 0ms, 20ms, 40ms, 60ms, etc. To approximate a 45ms sensor,
     use ``delay_min_lag=2, delay_max_lag=3`` which uniformly samples lag ∈ {2, 3}
     (both inclusive), giving either 40ms or 60ms delay.

Examples
^^^^^^^^

**Joint encoders (no delay):**

.. code-block:: python

    joint_pos: ObservationTermCfg = ObservationTermCfg(func=joint_pos)
    # delay_min_lag=delay_max_lag=0 by default.


**Camera with 40-60ms latency at 50Hz control:**

.. code-block:: python

    # 40-60ms latency = 2-3 steps at 50Hz (20ms/step)
    camera: ObservationTermCfg = ObservationTermCfg(
        func=camera_obs,
        delay_min_lag=2,  # 40ms
        delay_max_lag=3,  # 60ms
    )


**Mixed system - fast encoders and slow camera:**

.. code-block:: python

    @dataclass
    class PolicyCfg(ObservationGroupCfg):
        # Fast encoders (no delay)
        joint_pos: ObservationTermCfg = ObservationTermCfg(
            func=joint_pos,
        )

        # Camera with 40-80ms latency
        camera: ObservationTermCfg = ObservationTermCfg(
            func=camera_obs,
            delay_min_lag=2,  # 40ms
            delay_max_lag=4,  # 80ms
        )


Processing Pipeline
-------------------

Observations flow through this pipeline:


compute → noise → clip → scale → delay → history → flatten


**Why delay before history?** History stacks delayed observations. This models
real systems where you buffer old sensor readings, not future ones.

Example with both:

.. code-block:: python

    joint_vel: ObservationTermCfg = ObservationTermCfg(
        func=joint_vel,
        scale=0.1,             # Scale raw values
        delay_min_lag=1,       # 20ms delay at 50Hz
        delay_max_lag=2,       # 40ms delay at 50Hz
        history_length=3,      # Stack 3 delayed frames
        flatten_history_dim=True
    )
    # Pipeline:
    # 1. compute() returns (num_envs, 12)
    # 2. scale: multiply by 0.1
    # 3. delay: return observation from 1-2 steps ago
    # 4. history: stack last 3 delayed frames → (num_envs, 3, 12)
    # 5. flatten: reshape → (num_envs, 36)


Performance
-----------

Delay buffers are only created when ``delay_max_lag > 0``. Terms with no delay
(the default) have zero overhead. Similarly, history buffers are only created
when ``history_length > 0``.

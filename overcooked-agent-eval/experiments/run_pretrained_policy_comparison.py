"""Compare a BC-paired PPO policy with the self-play PPO baseline.

The bundled Human-Aware RL checkpoints contain two deliberately different
observation contracts:

* ``bc`` receives the handcrafted, player-centric feature vector.
* ``ppo`` receives the lossless spatial grid-mask tensor.

Never flatten or otherwise share these encoders.  The original paper-era
models used 64 handcrafted features and 20 spatial channels; newer versions of
Overcooked-AI can add features/channels.  This runner records the observed
shapes and can enforce the paper-era contract with
``--strict-paper-observations``.
"""

from __future__ import annotations

import argparse
import csv
import os
import pickle
import random
import types
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

# The bundled BC assets use TensorFlow SavedModel, which Keras 3 no longer
# loads. TensorFlow's maintained legacy Keras package remains compatible.
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

from overcooked_ai_py.agents.benchmarking import AgentEvaluator
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld


PROJECT_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_ROOT = PROJECT_ROOT.parent / "external" / "overcooked_ai"
PRETRAINED_ROOT = (
    UPSTREAM_ROOT
    / "src"
    / "overcooked_demo"
    / "server"
    / "static"
    / "assets"
    / "agents"
)
DEFAULT_LAYOUT = "cramped_room"
DEFAULT_EPISODES = 20
DEFAULT_HORIZON = 400
DEFAULT_SEED = 42
PAPER_BC_FEATURES = 64
PAPER_SP_CHANNELS = 20

LAYOUT_ASSET_NAMES = {
    "cramped_room": "CrampedRoom",
    "asymmetric_advantages": "AsymmetricAdvantages",
    "coordination_ring": "CoordinationRing",
    "forced_coordination": "ForcedCoordination",
    "counter_circuit_o_1order": "CounterCircuit1Order",
}


@dataclass(frozen=True)
class ObservationContract:
    """The distinct input shapes used by the two policy families."""

    bc_shape: tuple[int, ...]
    sp_shape: tuple[int, ...]

    @property
    def bc_feature_count(self) -> int:
        return int(np.prod(self.bc_shape))

    @property
    def sp_channels(self) -> int:
        if len(self.sp_shape) != 3:
            raise ValueError(
                f"PPO_SP observation must be a spatial HxWxC tensor, got {self.sp_shape}"
            )
        return self.sp_shape[-1]

    def validate_paper_contract(self) -> None:
        if self.bc_shape != (PAPER_BC_FEATURES,):
            raise ValueError(
                "BC checkpoint expects the paper-era 64-value handcrafted vector, "
                f"but this environment produced {self.bc_shape}. Use a matching "
                "Overcooked-AI revision or omit --strict-paper-observations."
            )
        if self.sp_channels != PAPER_SP_CHANNELS:
            raise ValueError(
                "PPO_SP checkpoint expects the paper-era 20-channel spatial masks, "
                f"but this environment produced {self.sp_shape}. Use a matching "
                "Overcooked-AI revision or omit --strict-paper-observations."
            )


@dataclass(frozen=True)
class EpisodeResult:
    condition: str
    seat_order: str
    episode: int
    seed: int
    sparse_return: float
    episode_length: int
    bc_observation_shape: str
    sp_observation_shape: str


@dataclass(frozen=True)
class CheckpointFiles:
    config: Path
    checkpoint: Path
    root: Path


class LegacyPredictModel:
    """Run a SavedModel in the TensorFlow graph/session that loaded it."""

    def __init__(self, model: Any, session: Any) -> None:
        self.model = model
        self.session = session
        self.graph = session.graph

    def predict(self, observations: np.ndarray, verbose: int = 0) -> np.ndarray:
        del verbose
        with self.graph.as_default(), self.session.as_default():
            return self.session.run(
                self.model.outputs[0], feed_dict={self.model.inputs[0]: observations}
            )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run BC + PPO_BC rollouts and compare them with the PPO_SP + PPO_SP "
            "self-play baseline."
        )
    )
    parser.add_argument(
        "--layout", choices=sorted(LAYOUT_ASSET_NAMES), default=DEFAULT_LAYOUT
    )
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--ppo-bc-checkpoint",
        type=Path,
        help="PPO_BC run/checkpoint directory (defaults to the bundled layout asset)",
    )
    parser.add_argument(
        "--ppo-sp-checkpoint",
        type=Path,
        help="PPO_SP run/checkpoint directory (defaults to the bundled layout asset)",
    )
    parser.add_argument(
        "--strict-paper-observations",
        action="store_true",
        help="Require exactly 64 BC features and 20 PPO_SP spatial channels",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "pretrained_policy_comparison.csv",
    )
    args = parser.parse_args(argv)
    if args.episodes < 1:
        parser.error("--episodes must be at least 1")
    if args.horizon < 1:
        parser.error("--horizon must be at least 1")
    return args


def bundled_checkpoint(layout: str, training_partner: str) -> Path:
    """Return the bundled PPO_BC or PPO_SP asset directory for ``layout``."""
    if training_partner not in {"BC", "SP"}:
        raise ValueError("training_partner must be 'BC' or 'SP'")
    return PRETRAINED_ROOT / f"Rllib{LAYOUT_ASSET_NAMES[layout]}{training_partner}"


def inspect_observation_contract(evaluator: AgentEvaluator) -> ObservationContract:
    """Measure each encoder independently on the gridworld start state."""
    state = evaluator.env.mdp.get_standard_start_state()
    bc_obs = evaluator.env.featurize_state_mdp(state)[0]
    sp_obs = evaluator.env.lossless_state_encoding_mdp(state)[0]
    return ObservationContract(tuple(bc_obs.shape), tuple(sp_obs.shape))


def _checkpoint_number(path: Path) -> int:
    suffix = path.name.replace("checkpoint_", "").replace("checkpoint-", "")
    try:
        return int(suffix)
    except ValueError:
        return -1


def resolve_checkpoint_files(path: Path) -> CheckpointFiles:
    """Resolve both Ray run directories and bundled web-demo asset directories."""
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {path}")

    candidate_roots = [path]
    if path.name == "agent":
        candidate_roots.append(path.parent)
    elif (path / "agent").is_dir():
        candidate_roots.append(path / "agent")

    config = next(
        (
            root / "config.pkl"
            for root in candidate_roots
            if (root / "config.pkl").is_file()
        ),
        None,
    )
    if config is None:
        config = next(
            (
                root.parent / "config.pkl"
                for root in candidate_roots
                if (root.parent / "config.pkl").is_file()
            ),
            None,
        )
    if config is None:
        raise FileNotFoundError(f"Could not find config.pkl at or beside {path}")

    search_dirs = [root for root in candidate_roots if root.is_dir()]
    checkpoint_candidates: list[Path] = []
    for root in search_dirs:
        checkpoint_candidates.extend(
            p for p in root.glob("checkpoint-*") if p.is_file()
        )
        checkpoint_candidates.extend(
            p for p in root.glob("checkpoint_*") if p.is_dir()
        )
    if not checkpoint_candidates:
        raise FileNotFoundError(f"Could not find a Ray checkpoint under {path}")

    checkpoint = max(checkpoint_candidates, key=_checkpoint_number)
    return CheckpointFiles(config=config, checkpoint=checkpoint, root=config.parent)


def load_trainer(path: Path) -> Any:
    """Restore a trainer without mutating the checked-in config or checkpoint."""
    try:
        import dill
        import ray.cloudpickle.cloudpickle as ray_cloudpickle
        from gym.utils.seeding import RandomNumberGenerator
        from human_aware_rl.imitation import behavior_cloning_tf2 as bc_module
        from human_aware_rl.rllib.rllib import gen_trainer_from_params
    except ImportError as exc:
        raise RuntimeError(
            "Pretrained rollouts require the Human-Aware RL dependencies. Install "
            "them with: python -m pip install -r requirements-harl.txt"
        ) from exc

    files = resolve_checkpoint_files(path)

    # Gym 0.23's RNG reduction predates NumPy 1.26, which supplies an extra
    # constructor argument during deepcopy. RLlib deep-copies policy spaces
    # while rebuilding the checkpoint, so accept and ignore that new argument.
    original_rng_ctor = RandomNumberGenerator._generator_ctor

    def generator_ctor_compat(bit_generator_name: str = "MT19937", *_: Any) -> Any:
        return original_rng_ctor(bit_generator_name)

    RandomNumberGenerator._generator_ctor = staticmethod(generator_ctor_compat)

    original_bc_loader = bc_module.load_bc_model

    def load_bc_model_compat(
        model_dir: str, verbose: bool = False
    ) -> tuple[Any, dict[str, Any]]:
        """Load either current ``model.keras`` or bundled SavedModel assets."""
        model_path = Path(model_dir)
        if (model_path / "model.keras").is_file():
            return original_bc_loader(model_dir, verbose=verbose)
        if not (model_path / "saved_model.pb").is_file():
            raise FileNotFoundError(f"No BC model found under {model_path}")

        if verbose:
            print(f"Loading legacy BC SavedModel from {model_path}")
        model = bc_module.keras.models.load_model(str(model_path), compile=False)
        session = bc_module.tf.compat.v1.keras.backend.get_session()
        model = LegacyPredictModel(model, session)
        with (model_path / "metadata.pickle").open("rb") as metadata_file:
            bc_params = pickle.load(metadata_file)
        return model, bc_params

    # BehaviorCloningPolicy resolves this module global when the trainer builds
    # its frozen BC policy. Keep the compatibility change local to this process.
    bc_module.load_bc_model = load_bc_model_compat

    with files.config.open("rb") as config_file:
        params = dill.load(config_file)

    params["training_params"]["num_workers"] = 0
    params["training_params"]["num_gpus"] = 0
    # The environment/checkpoints are upstream-owned and already validated;
    # skip Ray's deprecated NumPy-based environment checker during restoration.
    params["training_params"]["disable_env_checking"] = True
    params["results_dir"] = str(PROJECT_ROOT / "results" / "ray")

    # Bundled PPO_BC configs contain the original machine's absolute BC path.
    # Point the in-memory config at its colocated model, without rewriting assets.
    bc_config = params.get("bc_params", {}).get("bc_config", {})
    local_bc_root = files.root / "bc_params"
    if bc_config and local_bc_root.is_dir():
        model_dirs = sorted(p for p in local_bc_root.iterdir() if p.is_dir())
        if len(model_dirs) != 1:
            raise ValueError(
                f"Expected one bundled BC model under {local_bc_root}, found {len(model_dirs)}"
            )
        bc_config["model_dir"] = str(model_dirs[0])

    try:
        trainer = gen_trainer_from_params(params)
    finally:
        bc_module.load_bc_model = original_bc_loader
        RandomNumberGenerator._generator_ctor = original_rng_ctor

    # The checked-in policies were serialized on Python 3.7. Its CodeType
    # constructor predates ``posonlyargcount`` (added in Python 3.8), so teach
    # Ray's vendored cloudpickle how to rebuild the saved policy-mapping lambda
    # on Python 3.10. Install this only after Ray starts; otherwise Ray tries to
    # distribute the compatibility closure itself.
    original_builtin_type = ray_cloudpickle._builtin_type

    def legacy_code_type(*code_args: Any) -> types.CodeType:
        if len(code_args) == 15:
            code_args = (code_args[0], 0, *code_args[1:])
        return types.CodeType(*code_args)

    def builtin_type_compat(name: str) -> Any:
        return legacy_code_type if name == "CodeType" else original_builtin_type(name)

    ray_cloudpickle._builtin_type = builtin_type_compat

    # Skip Tune's training-history metadata; inference only needs the RLlib
    # worker state held in the checkpoint file.
    try:
        trainer.load_checkpoint(str(files.checkpoint))
    finally:
        ray_cloudpickle._builtin_type = original_builtin_type
    return trainer


def _pair_from_trainer(trainer: Any, first: str, second: str) -> Any:
    from human_aware_rl.rllib.rllib import get_agent_from_trainer
    from overcooked_ai_py.agents.agent import AgentPair

    return AgentPair(
        get_agent_from_trainer(trainer, policy_id=first, agent_index=0),
        get_agent_from_trainer(trainer, policy_id=second, agent_index=1),
    )


def evaluate_condition(
    evaluator: AgentEvaluator,
    trainer: Any,
    policy_order: tuple[str, str],
    condition: str,
    episodes: int,
    seed: int,
    contract: ObservationContract,
    pair_factory: Callable[[Any, str, str], Any] = _pair_from_trainer,
) -> list[EpisodeResult]:
    """Evaluate one policy ordering and return one compact row per episode."""
    pair = pair_factory(trainer, *policy_order)
    seat_order = f"{policy_order[0]}_0+{policy_order[1]}_1"
    rows: list[EpisodeResult] = []
    for index in range(episodes):
        episode_seed = seed + index
        random.seed(episode_seed)
        np.random.seed(episode_seed)
        trajectories = evaluator.evaluate_agent_pair(pair, num_games=1, info=False)
        rows.append(
            EpisodeResult(
                condition=condition,
                seat_order=seat_order,
                episode=index + 1,
                seed=episode_seed,
                sparse_return=float(trajectories["ep_returns"][0]),
                episode_length=int(trajectories["ep_lengths"][0]),
                bc_observation_shape=str(contract.bc_shape),
                sp_observation_shape=str(contract.sp_shape),
            )
        )
    return rows


def write_results(rows: Sequence[EpisodeResult], output: Path) -> None:
    if not rows:
        raise ValueError("Cannot write an empty comparison")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=asdict(rows[0]).keys())
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def print_summary(
    rows: Sequence[EpisodeResult], contract: ObservationContract, output: Path
) -> None:
    print("Pretrained policy comparison complete")
    print(f"BC observation: handcrafted vector {contract.bc_shape}")
    print(f"PPO observation: spatial grid masks {contract.sp_shape}")
    for condition in sorted({row.condition for row in rows}):
        returns = np.array(
            [row.sparse_return for row in rows if row.condition == condition]
        )
        print(
            f"{condition}: n={len(returns)}, mean={returns.mean():.2f}, "
            f"std={returns.std(ddof=1) if len(returns) > 1 else 0.0:.2f}"
        )
    print(f"Output CSV: {output.resolve()}")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    ppo_bc_path = args.ppo_bc_checkpoint or bundled_checkpoint(args.layout, "BC")
    ppo_sp_path = args.ppo_sp_checkpoint or bundled_checkpoint(args.layout, "SP")
    evaluator = AgentEvaluator.from_layout_name(
        {"layout_name": args.layout, "old_dynamics": True},
        {"horizon": args.horizon},
    )
    contract = inspect_observation_contract(evaluator)
    if args.strict_paper_observations:
        contract.validate_paper_contract()

    rows: list[EpisodeResult] = []
    with ExitStack() as stack:
        ppo_bc_trainer = load_trainer(ppo_bc_path)
        stack.callback(ppo_bc_trainer.stop)
        ppo_sp_trainer = load_trainer(ppo_sp_path)
        stack.callback(ppo_sp_trainer.stop)

        # Run both seats because BC features are player-centric. The self-play
        # baseline is symmetric, so it only needs one policy ordering.
        rows.extend(
            evaluate_condition(
                evaluator,
                ppo_bc_trainer,
                ("bc", "ppo"),
                "BC+PPO_BC",
                args.episodes,
                args.seed,
                contract,
            )
        )
        rows.extend(
            evaluate_condition(
                evaluator,
                ppo_bc_trainer,
                ("ppo", "bc"),
                "BC+PPO_BC",
                args.episodes,
                args.seed + args.episodes,
                contract,
            )
        )
        rows.extend(
            evaluate_condition(
                evaluator,
                ppo_sp_trainer,
                ("ppo", "ppo"),
                "PPO_SP+PPO_SP",
                args.episodes * 2,
                args.seed,
                contract,
            )
        )

    write_results(rows, args.output)
    print_summary(rows, contract, args.output)


if __name__ == "__main__":
    main()

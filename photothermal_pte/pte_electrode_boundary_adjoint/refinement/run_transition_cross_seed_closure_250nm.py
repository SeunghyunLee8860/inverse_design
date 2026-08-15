from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_transition_cross_seed_250nm import CLOSURE_TOLERANCE, run_cross_seed


HERE = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--beam-index",
        type=int,
        help="Run only one zero-based beam index, using beam-specific outputs.",
    )
    args = parser.parse_args()
    input_path = HERE / "transition_cross_seed_250nm.json"
    prior = json.loads(input_path.read_text(encoding="utf-8"))
    failing = tuple(
        int(beam["beam_index"])
        for beam in prior["beams"]
        if beam["hard_current_relative_spread"] > CLOSURE_TOLERANCE
        or beam["maximum_improvement_over_input_pool"] > CLOSURE_TOLERANCE
    )
    if not failing:
        print("No beam requires another cross-seed closure iteration.")
        return 0
    if args.beam_index is not None:
        if args.beam_index not in failing:
            raise ValueError(
                f"beam {args.beam_index} does not require closure; failing={failing}"
            )
        failing = (args.beam_index,)
        suffix = f"_beam{args.beam_index:02d}"
    else:
        suffix = ""
    print(f"selective closure beams={failing}", flush=True)
    return run_cross_seed(
        input_path=input_path,
        checkpoint_path=HERE / f"transition_cross_seed_closure{suffix}_checkpoint.json",
        output_path=HERE / f"transition_cross_seed_closure{suffix}_250nm.json",
        plot_path=HERE / f"transition_cross_seed_closure{suffix}_250nm.png",
        beam_indices=failing,
    )


if __name__ == "__main__":
    raise SystemExit(main())

import argparse
import json
import math
import sys
from pathlib import Path

# Search space, or project spam
def get_search_space():
  import numpy as np

  x1 = np.linspace(0, 1000, 10000)
  x2 = range(0, 1000)
  x3 = np.logspace(0, 10, 10000)
  x4 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
  x5 = np.linspace(0, 1000, 10000)

  return {
    "x1": list(x1),
    "x2": list(x2),
    "x3": list(x3),
    "x4": list(x4),
    "x5": list(x5),
  }


def func_comp(pop, P, random_state=None, **kwargs):
  import numpy as np
  from pymoo.operators.selection.tournament import compare

  if random_state is None:
    raise ValueError("func_comp requires the optimizer's random_state")

  winners = np.empty((len(P), 1), dtype=int)
  for i, group in enumerate(P):
    # Here compares, the one with least constraint violation (CV) wins
    cv = np.array([pop[j].CV[0] for j in group])
    best = group[cv == cv.min()]
    if len(best) == 1:
      winners[i, 0] = best[0]
    else:
      # Edit this comparison rule to customize parent selection for.
      winners[i, 0] = random_state.choice(best)
  return winners


# Just to normalize to the test
BOUNDS = {
  "x1": (0.0, 1000.0),
  "x2": (0.0, 999.0),
  "x3": (1.0, 1e10),
  "x4": (0.0, 9.0),
  "x5": (0.0, 1000.0),
}
TARGET = (0.2, 0.4, 0.6, 0.8, 1.0)


##
def evaluate(params):
  """Validate decoded parameters and return the candidate.json payload."""
  if not isinstance(params, dict):
    raise ValueError("parameters must be a JSON object")
  missing = sorted(set(BOUNDS) - set(params))
  extra = sorted(set(params) - set(BOUNDS))
  if missing or extra:
    raise ValueError(f"unexpected parameter keys: missing={missing}, extra={extra}")

  values = {}
  for name, (lower, upper) in BOUNDS.items():
    raw = params[name]
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
      raise ValueError(f"{name} must be a finite JSON number")
    value = float(raw)
    if not math.isfinite(value) or not lower <= value <= upper:
      raise ValueError(f"{name} must be finite and within [{lower}, {upper}]")
    if name in ("x2", "x4") and not value.is_integer():
      raise ValueError(f"{name} must be an integer")
    values[name] = value

  # normalization for test purposes
  z = (
    values["x1"] / 1000.0,
    values["x2"] / 999.0,
    math.log10(values["x3"]) / 100.0,
    values["x4"] / 9.0,
    values["x5"] / 1000.0,
  )
  # Result space calculation, can be a script, circuit simulator, or anything else
  # it calculates the objective functions bazed on the params "values"
  y1 = math.fsum(value for value in z)/len(z)
  y2 = math.fsum(z)/len(z)
  y3 = math.fsum((value - target)**2 for value, target in zip(z, TARGET))/len(z)

  # Must return this structure
  # {
  #   "status": "ok"
  #   "objectives": {"y1": ..., "y2": ..., ...}
  #   "constraints": {"const1": ..., "const2": ..., ...}
  # }
  return {
    "status": "ok",
    "objectives": {
      "y1": y1,
      "y2": y2,
      "y3": y3,
    },
    "constraints": {"budget": math.fsum(z) - 3.0},
  }


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--params", type=Path, required=True, help="JSON file containing x1 through x5")
  parser.add_argument("--out-dir", type=Path, required=True, help="Directory for candidate.json")
  args = parser.parse_args()

  try:
    params = json.loads(args.params.read_text(encoding="utf-8"))
    record = evaluate(params)
    exit_code = 0
  except (OSError, ValueError, OverflowError) as error:
    record = {"status": "error", "reason": str(error)}
    exit_code = 1

  try:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "candidate.json").write_text(
      json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
  except OSError as error:
    print(f"Could not write candidate.json: {error}", file=sys.stderr)
    return 1
  if exit_code:
    print(record["reason"], file=sys.stderr)
  return exit_code


if __name__ == "__main__":
  sys.exit(main())

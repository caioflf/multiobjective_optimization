import argparse
import importlib.util
import json
import os
import random
import shlex
import shutil
import signal
import sqlite3
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import closing
from math import prod
from threading import Event
from pathlib import Path

import numpy as np
import pymoo
from pymoo.algorithms.moo.nsga3 import NSGA3, comp_by_cv_then_random
from pymoo.core.problem import Problem
from pymoo.core.population import Population
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.optimize import minimize
from pymoo.operators.selection.tournament import TournamentSelection

NSGA_DEFAULTS = {
  "n_partitions": 4,
  "pop_size": None,
  "offspring": None,
  "n_gen": 20,
  "seed": 1,
  "crossover_prob": 0.9,
  "crossover_eta": 15,
  "mutation_prob": None, # None: min(0.5, 1/n_var)
  "mutation_eta": 20,
  "pressure": 2
}

EVAL_DEFAULTS = {
  "max_workers": 1,
  "retries": 2,
  "retry_delay": 60, # seconds
  "timeout": None # seconds
}


def _to_python_list(value):
  if isinstance(value, np.ndarray):
    return value.tolist()
  if isinstance(value, (tuple, list)):
    return list(value)
  return [value]


def normalize_space_spec(spec):
  if spec is None:
    raise ValueError("Search-space entry cannot be None")

  if isinstance(spec, str):
    raise ValueError(
      "Search-space entries must be lists/arrays/tuples of values only; do not use 'float', 'int', or 'choice' tags."
    )

  values = _to_python_list(spec)
  if len(values) == 1 and isinstance(values[0], (list, tuple, np.ndarray)):
    values = _to_python_list(values[0])
  if not values:
    raise ValueError(f"Empty search-space specification: {spec!r}")
  return [value.item() if isinstance(value, np.generic) else value for value in values]


def load_python_module(path):
  path = Path(path)
  if path.suffix.lower() != ".py":
    raise ValueError(f"Only Python module files are supported. Got: {path}")

  module_name = f"user_module_{abs(hash(path.resolve()))}"
  spec = importlib.util.spec_from_file_location(module_name, path)
  if spec is None or spec.loader is None:
    raise ValueError(f"Could not load Python module from {path}")

  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def load_search_space(path):
  if isinstance(path, dict):
    return {name: normalize_space_spec(spec) for name, spec in path.items()}

  module = load_python_module(path)

  for attr_name in ("get_search_space", "SEARCH_SPACE"):
    value = getattr(module, attr_name, None)
    if callable(value):
      value = value()
    if isinstance(value, dict):
      return {name: normalize_space_spec(spec) for name, spec in value.items()}

  raise ValueError(
    f"Python search space file {path} must define a function named get_search_space() or a dict named SEARCH_SPACE/search_space/SPACE/space"
  )


def load_tournament_comparator(evaluator):
  if not isinstance(evaluator, Path):
    return comp_by_cv_then_random
  module = load_python_module(evaluator)
  comparator = getattr(module, "func_comp", comp_by_cv_then_random)
  if not callable(comparator):
    raise ValueError(f"{evaluator}: func_comp must be callable")
  return comparator


class SearchSpace:
  def __init__(self, space):
    if not space:
      raise ValueError("Search space must contain at least one variable")
    self.names = list(space)
    self.space = {name: normalize_space_spec(spec) for name, spec in space.items()}
    self.xl, self.xu = [], []
    for name in self.names:
      values = self.space[name]
      if not values:
        raise ValueError(f"Empty value list for variable {name!r}")
      self.xl.append(0)
      self.xu.append(len(values) - 1)

  @property
  def n_var(self):
    return len(self.names)

  def decode(self, row):
    if len(row) != self.n_var:
      raise ValueError("Candidate must have one index per variable")
    params = {}
    for name, x in zip(self.names, row):
      values = self.space[name]
      if not np.isfinite(x) or x != int(x) or not 0 <= x < len(values):
        raise ValueError(f"Invalid discrete index for {name}: {x}")
      idx = int(x)
      params[name] = values[idx]
    return params

  def encode(self, params):
    if not isinstance(params, dict):
      raise ValueError("Candidate params must be a dictionary")
    missing = set(self.names) - set(params)
    extra = set(params) - set(self.names)
    if missing or extra:
      raise ValueError(f"Parameter names do not match: missing={sorted(missing)}, extra={sorted(extra)}")
    row = []
    for name in self.names:
      try:
        row.append(self.space[name].index(params[name]))
      except ValueError:
        raise ValueError(f"Parameter {name!r} has a value outside the current search space: {params[name]!r}") from None
    return row


def load_initial_population(paths, space):
  """Merge generation/Pareto files using parameter values, ignoring saved scores and indices."""
  rows, seen = [], set()
  for path in paths:
    path = Path(path)
    try:
      data = json.loads(path.read_text())
    except (OSError, ValueError) as error:
      raise ValueError(f"Could not read initial population {path}: {error}") from None
    candidates = data.get("candidates", data.get("front")) if isinstance(data, dict) else None
    if not isinstance(candidates, list):
      raise ValueError(f"Initial population {path} must contain a 'candidates' or 'front' list")
    for index, candidate in enumerate(candidates):
      try:
        row = tuple(space.encode(candidate.get("params") if isinstance(candidate, dict) else None))
      except ValueError as error:
        raise ValueError(f"Initial population {path}, candidate {index}: {error}") from None
      if row not in seen:
        rows.append(row)
        seen.add(row)
  if not rows:
    raise ValueError("The initial population files contain no candidates")
  return np.asarray(rows, dtype=int)


class InitialPopulationNSGA3(NSGA3):
  """Fill an imported pool before evaluation; select survivors afterwards."""

  def _initialize_infill(self):
    population = super()._initialize_infill()
    if len(population) >= self.pop_size:
      return population

    lower = [int(value) for value in self.problem.xl]
    sizes = [int(hi - lo + 1) for lo, hi in zip(self.problem.xl, self.problem.xu)]
    remaining = prod(sizes)
    if self.pop_size > remaining:
      raise ValueError(f"Requested population {self.pop_size} exceeds the {remaining} distinct grid points")
    seen = {tuple(row) for row in population.get("X")}
    rows, swaps = [], {}
    rng = random.Random(int(self.random_state.integers(2**63)))
    # Sample grid positions without replacement, without materializing the grid.
    # Python integers also handle grids larger than the int64 range.
    while len(population) + len(rows) < self.pop_size:
      position = rng.randrange(remaining)
      value = swaps.get(position, position)
      remaining -= 1
      swaps[position] = swaps.get(remaining, remaining)
      swaps.pop(remaining, None)
      row = []
      for lo, size in reversed(list(zip(lower, sizes))):
        value, digit = divmod(value, size)
        row.append(lo + digit)
      row = tuple(reversed(row))
      if row not in seen:
        rows.append(row)
        seen.add(row)
    return Population.merge(population, Population.new("X", np.asarray(rows, dtype=int)))

  def _initialize_advance(self, infills=None, **kwargs):
    self.pop = self.survival.do(
      self.problem, infills, n_survive=self.pop_size,
      random_state=self.random_state, algorithm=self, **kwargs,
    )


def read_candidate(out_dir):
  try:
    record = json.loads((out_dir / "candidate.json").read_text())
  except (OSError, ValueError):
    return None
  return record if isinstance(record, dict) else None

def usable(record, objectives, constraints):
  try:
    for names, key in ((objectives, "objectives"), (constraints, "constraints")):
      for name in names:
        if not np.isfinite(float(record[key][name])):
          return False
  except (TypeError, ValueError, KeyError, OverflowError):
    return False
  return True

def stop_evaluator_process(process):
  # Evaluators can launch solver subprocesses; stop their process group too.
  try:
    if os.name == "posix":
      os.killpg(process.pid, signal.SIGKILL)
    else:
      process.kill()
  except ProcessLookupError:
    pass
  process.wait()


def run_evaluator(command, out_dir, params, objectives, constraints, timeout, stop_event=None):
  if out_dir.exists():
    shutil.rmtree(out_dir)
  out_dir.mkdir(parents=True)
  params_file = out_dir / "params.json"
  params_file.write_text(json.dumps(params, indent=2, sort_keys=True) + "\n")

  def error(reason):
    record = {"status": "error", "reason": reason}
    (out_dir / "candidate.json").write_text(json.dumps(record, indent=2) + "\n")
    return record

  if stop_event is not None and stop_event.is_set():
    return error("evaluation cancelled after another candidate failed")
  if isinstance(command, Path):
    argv = [sys.executable, str(command.resolve())]
  else:
    argv = shlex.split(command)
  argv += ["--params", str(params_file), "--out-dir", str(out_dir)]
  try:
    with open(out_dir / "evaluator.log", "w") as log:
      with subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT,
        start_new_session=(os.name == "posix")) as process:
        deadline = None if timeout is None else time.monotonic() + timeout
        try:
          while True:
            if stop_event is not None and stop_event.is_set():
              stop_evaluator_process(process)
              return error("evaluation cancelled after another candidate failed")
            interval = 0.1 if deadline is None else max(0, min(0.1, deadline - time.monotonic()))
            try:
              returncode = process.wait(timeout=interval)
              break
            except subprocess.TimeoutExpired:
              if deadline is not None and time.monotonic() >= deadline:
                stop_evaluator_process(process)
                return error("evaluator timeout")
        except BaseException:
          stop_evaluator_process(process)
          raise
  except OSError as err:
    return error(f"evaluator failed: {err}")

  record = read_candidate(out_dir)
  if returncode != 0:
    return error(f"evaluator exited with code {returncode}: "
      f"{record.get('reason', 'see evaluator.log') if record else 'see evaluator.log'}")
  if record is None:
    return error("candidate.json not found or invalid")
  if record.get("status") == "ok" and not usable(record, objectives, constraints):
    return error("candidate.json missing or invalid required values")
  if record.get("status") not in ("ok", "fail", "error"):
    return error(f"invalid status: {record.get('status')}")
  return record

def reusable(out_dir, params, objectives, constraints):
  record = read_candidate(out_dir)
  if record is None or record.get("status") != "ok" or not usable(record, objectives, constraints):
    return None
  try:
    same = json.loads((out_dir / "params.json").read_text()) == json.loads(
      json.dumps(params)
    )
  except (OSError, ValueError):
    return None
  return record if same else None


class EvaluationDatabase:
  """Successful evaluations shared across grids with matching variables and metrics."""

  def __init__(self, path, search_space, objectives, constraints):
    self.path = Path(path).resolve()
    self.objectives = dict(objectives)
    self.constraints = list(constraints)
    # Retain the initial grid as metadata; reuse depends on names and actual values.
    definition = {
      "search_space": search_space,
      "objectives": self.objectives,
      "constraints": sorted(self.constraints),
    }
    definition_json = self.encode(definition)
    self.path.parent.mkdir(parents=True, exist_ok=True)
    # Each operation owns its connection; workers never share a connection.
    # Serialize initialization so simultaneous runs cannot create different definitions.
    with closing(self.connect()) as connection, connection:
      connection.execute("BEGIN IMMEDIATE")
      tables = {row[0] for row in connection.execute(
          "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
      )}
      if tables:
        if tables != {"metadata", "evaluations"}:
          raise ValueError(f"Not an evaluation database: {self.path}")
        row = connection.execute(
          "SELECT version, definition_json FROM metadata WHERE id = 1"
        ).fetchone()
        if row is None or row[0] != 1:
          raise ValueError(f"Unsupported evaluation database format: {self.path}")
        stored_definition = json.loads(row[1])
        # Grid sizes, bounds, and ordering can differ; candidate keys use decoded values.
        if (set(stored_definition["search_space"]) != set(search_space)
          or stored_definition["objectives"] != self.objectives
          or sorted(stored_definition["constraints"]) != sorted(self.constraints)):
          raise ValueError(
            f"Database space mismatch: {self.path}. Variable names, "
            "objective names/directions, and constraint names must match. "
            "Use a different --database path."
          )
      else:
        connection.execute(
          "CREATE TABLE metadata (id INTEGER PRIMARY KEY CHECK (id = 1), "
          "version INTEGER NOT NULL, definition_json TEXT NOT NULL)"
        )
        connection.execute(
          "CREATE TABLE evaluations (id INTEGER PRIMARY KEY, "
          "params_json TEXT NOT NULL UNIQUE, record_json TEXT NOT NULL, "
          "source_dir TEXT NOT NULL, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)"
        )
        connection.execute(
          "INSERT INTO metadata (id, version, definition_json) VALUES (1, 1, ?)",
          (definition_json,),
        )

  @staticmethod
  def encode(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)

  def connect(self):
    # Keep transactions short; no lock is held while a simulator runs.
    return sqlite3.connect(self.path, timeout=30)

  def validate(self, record):
    if (not isinstance(record, dict) or record.get("status") != "ok"
      or not usable(record, self.objectives, self.constraints)):
      raise ValueError(f"Database requires a successful evaluation with valid metrics: {self.path}")

  def get(self, params):
    with closing(self.connect()) as connection:
      row = connection.execute(
        "SELECT record_json, source_dir FROM evaluations WHERE params_json = ?",
        (self.encode(params),),
      ).fetchone()
    if row is None:
      return None
    record = json.loads(row[0])
    self.validate(record)
    return record, row[1]

  def add(self, params, record, source_dir):
    self.validate(record)
    with closing(self.connect()) as connection, connection:
      connection.execute(
        "INSERT INTO evaluations (params_json, record_json, source_dir) VALUES (?, ?, ?) "
        "ON CONFLICT(params_json) DO NOTHING",
        (self.encode(params), self.encode(record), str(Path(source_dir).resolve())),
      )


class CandidateProblem(Problem):
  def __init__(self, space, objectives, constraints, evaluator, run_dir, evaluator_args, database=None):
    super().__init__(
      n_var=space.n_var,
      n_obj=len(objectives),
      n_ieq_constr=len(constraints),
      xl=np.array(space.xl),
      xu=np.array(space.xu),
      vtype=int,
    )
    self.space = space
    self.objectives = objectives
    self.constraints = constraints
    self.evaluator = evaluator
    self.run_dir = Path(run_dir)
    self.evaluator_args = evaluator_args
    self.database = database
    self.sign = np.array([-1 if sense == "max" else 1 for sense in objectives.values()])
    self.stop_event = Event()
    self.generation = 0
    self.ids = {} # gene row -> candidate id, for the final front

  def run_candidate(self, generation, number, row):
    cid = "g%03d_c%03d" % (generation, number)
    params = self.space.decode(row)
    out_dir = self.run_dir / ("gen_%03d" % generation) / ("cand_%03d" % number)
    args = self.evaluator_args
    database_source = None
    if self.database is None:
      record = reusable(out_dir, params, self.objectives, self.constraints)
    else:
      # With a database, only its compatible records authorize reuse. Old run folders
      # alone do not establish that the search/objective definitions still match.
      cached = self.database.get(params)
      record = None if cached is None else cached[0]
      if cached is not None:
        database_source = cached[1]
        # Preserve existing matching candidate files and logs. A new slot receives
        # inputs/results, but no fabricated simulator log.
        if reusable(out_dir, params, self.objectives, self.constraints) != record:
          if out_dir.exists():
            shutil.rmtree(out_dir)
          out_dir.mkdir(parents=True)
          (out_dir / "params.json").write_text(json.dumps(params, indent=2, sort_keys=True) + "\n")
          (out_dir / "candidate.json").write_text(json.dumps(record, indent=2) + "\n")
    if record is None:
      for attempt in range(args["retries"] + 1):
        if self.stop_event.is_set():
          raise RuntimeError(f"{cid} cancelled after another candidate failed")
        try:
          record = run_evaluator(self.evaluator, out_dir, params, self.objectives,
            self.constraints, args["timeout"], self.stop_event)
        except Exception as err:
          record = {"status": "error", "reason": f"{type(err).__name__}: {err}"}
        if record.get("status") == "ok":
          break
        print(" %s: %s (%s), attempt %d of %d"
          % (cid, record.get("status"), record.get("reason"),
            attempt + 1, args["retries"] + 1), flush=True)
        if attempt < args["retries"] and self.stop_event.wait(args["retry_delay"]):
          raise RuntimeError(f"{cid} cancelled after another candidate failed")
    if record is None or record.get("status") != "ok":
      reason = (record.get("reason") or f"evaluator returned status {record.get('status')!r}"
        if record is not None else "no evaluation result")
      raise RuntimeError(f"{cid} failed: {reason}; see {out_dir}")

    if self.database is not None and database_source is None:
      self.database.add(params, record, out_dir)

    summary = {
      "id": cid,
      "dir": str(out_dir),
      "status": record["status"],
      "reason": record.get("reason"),
      "genes": [int(x) for x in row],
      "params": params,
    }
    if self.database is not None:
      summary["database_reused"] = database_source is not None
      summary["source_dir"] = database_source or str(out_dir.resolve())
    f = [float(record["objectives"][o]) for o in self.objectives]
    g = [float(record["constraints"][c]) for c in self.constraints]
    return summary, f, g

  def _evaluate(self, X, out, *args, **kwargs):
    generation = self.generation
    self.generation += 1
    t0 = time.time()
    self.stop_event.clear()
    candidates = iter(enumerate(X))
    results = {}
    workers = self.evaluator_args["max_workers"]
    with ThreadPoolExecutor(max_workers=workers) as pool:
      pending = {}

      def submit_next():
        item = next(candidates, None)
        if item is not None:
          number, row = item
          pending[pool.submit(self.run_candidate, generation, number, row)] = number

      try:
        for _ in range(workers):
          submit_next()
        while pending:
          done, _ = wait(pending, return_when=FIRST_COMPLETED)
          for future in done:
            results[pending.pop(future)] = future.result()
          for _ in done:
            submit_next()
      except BaseException:
        self.stop_event.set()
        for future in pending:
          future.cancel()
        raise

    results = [results[number] for number in range(len(X))]
    out["F"] = np.array([r[1] for r in results]) * self.sign
    out["G"] = np.array([r[2] for r in results])
    for row, (summary, _, _) in zip(X, results):
      self.ids[tuple(int(x) for x in row)] = summary["id"]

    gen_dir = self.run_dir / ("gen_%03d" % generation)
    gen_dir.mkdir(parents=True, exist_ok=True)
    summary = [
      {
        **s,
        "objectives": dict(zip(self.objectives, f)),
        "constraints": dict(zip(self.constraints, g)),
        "feasible": all(v <= 0 for v in g)
      }
      for s, f, g in results
    ]
    (gen_dir / "generation.json").write_text(
      json.dumps(
        {
          "generation": generation,
          "seconds": round(time.time() - t0, 1),
          "candidates": summary,
        },
        indent=2,
      )
      + "\n"
    )
    feasible = [s for s in summary if s["feasible"]]
    best = {
      o: (max if sense == "max" else min)(s["objectives"][o] for s in feasible)
      for o, sense in self.objectives.items()
    } if feasible else {}
    print(
      "generation %03d:\t%d candidates\t%d ok\t%d feasible\t%.0f s\tbest: %s"
      % (
        generation,
        len(summary),
        sum(s["status"] == "ok" for s in summary),
        len(feasible),
        time.time() - t0,
        {k: round(v, 6) for k, v in best.items()}
      ),
      flush=True
    )

# Look at this in the future, explore the maximum of the parameters
def make_algorithm(n_obj, nsga, func_comp=comp_by_cv_then_random, initial_population=None):
  ref_dirs = get_reference_directions("das-dennis", n_obj, n_partitions=nsga["n_partitions"])
  pop_size = nsga["pop_size"] or len(ref_dirs)
  offspring = nsga["offspring"] or pop_size
  algorithm_class = NSGA3 if initial_population is None else InitialPopulationNSGA3
  return algorithm_class(
    ref_dirs=ref_dirs,
    pop_size=pop_size,
    sampling=IntegerRandomSampling() if initial_population is None else initial_population,
    crossover=SBX(prob=nsga["crossover_prob"], eta=nsga["crossover_eta"],
      vtype=float, repair=RoundingRepair()),
    selection=TournamentSelection(func_comp=func_comp, pressure=nsga["pressure"]),
    mutation=PM(prob_var=nsga["mutation_prob"], eta=nsga["mutation_eta"],
      vtype=float, repair=RoundingRepair()),
    eliminate_duplicates=True,
    n_offsprings=offspring,
    seed=nsga["seed"],
  )

def optimize(run_dir, search_space, evaluator, objectives, constraints=(), nsga=None, evaluator_args=None,
  database=None, initial_population=None):
  nsga = {**NSGA_DEFAULTS, **(nsga or {})}
  evaluator_args = {**EVAL_DEFAULTS, **(evaluator_args or {})}
  objectives = dict(objectives)
  constraints = list(constraints)
  if not objectives:
    raise ValueError("At least one objective is required")
  if evaluator_args["retries"] < 0 or evaluator_args["retry_delay"] < 0:
    raise ValueError("retries and retry_delay must be nonnegative")
  if evaluator_args["max_workers"] < 1 or nsga["n_gen"] < 1:
    raise ValueError("max_workers and n_gen must be positive")
  for name, sense in objectives.items():
    if sense not in ("min", "max"):
      raise ValueError(f"Invalid sense for objective {name}: {sense}")
  run_dir = Path(run_dir)

  space=SearchSpace(search_space)
  paths = [initial_population] if isinstance(initial_population, (str, Path)) else (initial_population or [])
  paths = [Path(path).expanduser().resolve() for path in paths]
  initial_rows = load_initial_population(paths, space) if paths else None
  algorithm = make_algorithm(len(objectives), nsga, load_tournament_comparator(evaluator), initial_rows)
  if initial_rows is not None:
    print(f"initial population: {len(initial_rows)} unique imported candidates, target {algorithm.pop_size}")
  database = EvaluationDatabase(database, space.space, objectives, constraints) if database is not None else None
  problem = CandidateProblem(space, objectives, constraints, evaluator, run_dir, evaluator_args, database)
  run_dir.mkdir(parents=True, exist_ok=True)
  (run_dir / "run.json").write_text(
    json.dumps(
      {
        "started": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "search_space": space.space,
        "evaluator": str(evaluator),
        "database": str(database.path) if database is not None else None,
        "objectives": objectives,
        "constraints": constraints,
        "nsga": nsga,
        "pop_size": algorithm.pop_size,
        "initial_population": [str(path) for path in paths],
        "eval": {k: str(v) for k, v in evaluator_args.items()},
        "pymoo": pymoo.__version__,
        "python": sys.version.split()[0]
      },
      indent=2,
      default=str,
    )
    + "\n"
  )
  result = minimize(problem, algorithm, ("n_gen", nsga["n_gen"]), seed=nsga["seed"], verbose=False)
  front = []
  if result.X is not None and result.F is not None:
    X = np.atleast_2d(result.X)
    F = np.atleast_2d(result.F)
    for row, f in zip(X, F):
      front.append(
        {
          "id": problem.ids.get(tuple(int(x) for x in row)),
          "params": space.decode(row),
          "objectives": dict(zip(objectives, map(float, f * problem.sign))),
        }
      )
  (run_dir / "pareto.json").write_text(
    json.dumps({"objectives": objectives, "front": front}, indent=2) + "\n"
  )
  print("done: %d candidates on the front, see %s" % (len(front), run_dir / "pareto.json"))
  return result

def objective_arg(text):
  name, _, sense = text.partition(":")
  return name, sense or "min"

def main():
  parser = argparse.ArgumentParser(
    description="Optimization script, chooses candidates and hands "
    "each one to an external evaluator")

  # Problem arguments
  run = parser.add_argument_group("problem")
  run.add_argument("--run-dir", type=Path, required=True, help="Directory to store the run")
  run.add_argument("--database", type=Path,
    help="SQLite file for successful evaluations; create or append only for a matching problem space")
  run.add_argument("--initial-population", type=Path, action="append", default=[], metavar="PATH",
    help="Seed from generation.json or pareto.json; repeat to merge files. Fill shortages randomly; "
         "select oversized pools using NSGA-III survival after evaluation")

  run.add_argument("--search-space", type=Path, required=True,
    help="Python file that passes a dictionary with lists")

  run.add_argument("--evaluator", type=Path, required=True,
    help="Python script that evaluates a candidate, with --params and --out-dir appended")

  run.add_argument("--objective", action="append", default=[],
    metavar="NAME:min|max", required=True,
    help="Space-separated objectives; can also be specified multiple times")

  run.add_argument("--constraint", action="append", default=[], metavar="NAME",
    help="Space-separated constraint names (feasible when <= 0); an empty string disables constraints")

  # NSGA-III arguments
  opt = parser.add_argument_group("Optimizer (NSGA-III)")
  opt.add_argument("--partitions", type=int, default=NSGA_DEFAULTS["n_partitions"])
  opt.add_argument("--pop-size", type=int, default=NSGA_DEFAULTS["pop_size"],
    help="default: one individual per reference direction")
  opt.add_argument("--offspring", type=int, default=NSGA_DEFAULTS["offspring"],
    help="default: same as pop_size")
  opt.add_argument("--generations", type=int, default=NSGA_DEFAULTS["n_gen"],
    help="default: number of generations")
  opt.add_argument("--seed", type=int, default=NSGA_DEFAULTS["seed"])
  opt.add_argument("--crossover-prob", type=float, default=NSGA_DEFAULTS["crossover_prob"])
  opt.add_argument("--crossover-eta", type=float, default=NSGA_DEFAULTS["crossover_eta"])
  opt.add_argument("--mutation-prob", type=float, default=NSGA_DEFAULTS["mutation_prob"])
  opt.add_argument("--mutation-eta", type=float, default=NSGA_DEFAULTS["mutation_eta"])
  opt.add_argument("--selection-pressure", type=int, default=NSGA_DEFAULTS["pressure"],
    help="Tournament selection pressure, number of candidates competing for a spot")

  # Evaluator arguments
  eval = parser.add_argument_group("Evaluator")
  eval.add_argument("--max-workers", type=int, default=EVAL_DEFAULTS["max_workers"],
    help="Maximum number of concurrent evaluations")
  eval.add_argument("--retries", type=int, default=EVAL_DEFAULTS["retries"],
    help="Retries after a failed evaluation; abort when exhausted")
  eval.add_argument("--retry-delay", type=float, default=EVAL_DEFAULTS["retry_delay"],
    help="Seconds to wait between evaluation attempts")
  eval.add_argument("--timeout", type=float, default=EVAL_DEFAULTS["timeout"],
    help="Timeout for each evaluation (seconds)")

  args = parser.parse_args()
  objectives = dict(objective_arg(name) for group in args.objective for name in group.split())
  constraints = [name for group in args.constraint for name in group.split()]

  nsga = {"n_partitions": args.partitions,
    "pop_size": args.pop_size,
    "offspring": args.offspring,
    "n_gen": args.generations,
    "seed": args.seed,
    "crossover_prob": args.crossover_prob,
    "crossover_eta": args.crossover_eta,
    "mutation_prob": args.mutation_prob,
    "mutation_eta": args.mutation_eta,
    "pressure": args.selection_pressure}

  evaluator = {"max_workers": args.max_workers,
    "retries": args.retries,
    "retry_delay": args.retry_delay,
    "timeout": args.timeout}

  optimize(args.run_dir,
    load_search_space(args.search_space),
    args.evaluator,
    objectives,
    constraints,
    nsga,
    evaluator,
    database=args.database,
    initial_population=args.initial_population)

if __name__ == "__main__":
  main()

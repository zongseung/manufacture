import argparse
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TypedDict

import numpy as np
import polars as pl

from gmst import RESULTS, ROOT, backbone, evaluate, features, hmm
from gmst.ar_experiment import experiment_panel
from gmst.contracts import Panel
from gmst.flow_film import issue_flow
from gmst.flow_film_evaluation import candidate_metrics, score_prediction, summarize
from gmst.flow_film_training import FitConfig, FlowFit, FlowIssue, fit_head, inner_oof, issue_day
from gmst.hmm_forecast import day_tensors, issue_centre


@dataclass(frozen=True, slots=True)
class DiagnosticConfig:
    seed: int = 0
    epochs: int = 300
    paths: int = 2000


@dataclass(frozen=True, slots=True)
class DiagnosticRun:
    smoke: bool = False
    epochs: int = 300
    paths: int = 2000


class DiagnosticDecision(TypedDict):
    status: str
    positive_folds: dict[str, int]
    informative_folds: dict[str, int]
    mean_gap: dict[str, float | None]
    fold_gaps: dict[str, float | None]


def diagnose_fold(panel: Panel, fold: str, config: DiagnosticConfig) -> pl.DataFrame:
    tau, half_life, _ = backbone.select_tau(panel, fold)
    b3 = hmm.hmm_model('B3', K=3, tau=tau, half_life=half_life,
                       max_epochs=config.epochs, N=config.paths, seed=config.seed)
    C, _ = evaluate.thresholds(panel, features.role_idx(panel, fold, 'train'))
    state = b3['fit'](panel, fold, C)
    rows: list[dict[str, str | int | float | bool]] = []
    for d in features.role_idx(panel, fold, 'val'):
        day = int(d)
        prediction = b3['predict'](state, panel, day)
        joint = issue_flow(state, panel, day)
        m = issue_centre(state, panel, np.array([day], dtype=np.int64))
        z = day_tensors(state['model'], panel, np.array([day], dtype=np.int64), m, 'A+')[0]
        transition = state['model'].trans(z)[0].detach().cpu().numpy()
        q = joint.sum(axis=2)
        switch = joint.sum(axis=(1, 2)) - np.trace(joint, axis1=1, axis2=2)
        for h in range(96):
            y = float(panel['Y'][day, h])
            row: dict[str, str | int | float | bool] = {
                'fold': fold, 'seed': config.seed, 'date': panel['dates'][day].isoformat(),
                'slot': h, 'op': int(panel['op'][day]), 'observed': bool(np.isfinite(y)),
                'y_median': float(prediction['y_median'][h]),
                'abs_err': abs(y - prediction['y_median'][h]) if np.isfinite(y) else float('nan'),
                's': float(switch[h]),
            }
            row.update({f'q{i}': float(q[h, i]) for i in range(3)})
            row.update({f'P{i}{j}': float(transition[h, i, j]) for i in range(3) for j in range(3)})
            row.update({f'J{i}{j}': float(joint[h, i, j]) for i in range(3) for j in range(3)})
            rows.append(row)
    return pl.DataFrame(rows)


def diagnostic_decision(rows: pl.DataFrame) -> DiagnosticDecision:
    positive: dict[str, int] = {}
    informative: dict[str, int] = {}
    means: dict[str, float | None] = {}
    gaps: dict[str, float | None] = {}
    for seed in range(3):
        key = str(seed)
        seed_rows = rows.filter(pl.col('seed') == seed)
        seed_gaps: list[float] = []
        for fold in (f'ar{i}' for i in range(1, 9)):
            group = seed_rows.filter((pl.col('fold') == fold) & pl.col('observed'))
            s = group['s'].to_numpy()
            errors = group['abs_err'].to_numpy()
            order = np.sort(s)
            cut = int(np.ceil(0.75 * len(order)))
            if 0 < cut < len(order) and order[cut - 1] < order[cut]:
                high = s >= order[cut]
                gap = float(errors[high].mean() - errors[~high].mean())
                seed_gaps.append(gap)
                gaps[f'{key}/{fold}'] = gap
            else:
                gaps[f'{key}/{fold}'] = None
        positive[key] = sum(gap > 0 for gap in seed_gaps)
        informative[key] = len(seed_gaps)
        means[key] = float(np.mean(seed_gaps)) if seed_gaps else None
    supported = all(informative[str(seed)] >= 6 and positive[str(seed)] >= 6
                    and means[str(seed)] is not None and means[str(seed)] > 0 for seed in range(3))
    return {'status': 'supported' if supported else 'transition_hypothesis_unsupported',
            'positive_folds': positive, 'informative_folds': informative,
            'mean_gap': means, 'fold_gaps': gaps}


def run_diagnostic(out: Path, config: DiagnosticRun) -> None:
    output = _new_output(out, config)
    panel = experiment_panel(features.load_panel())
    folds = ('ar8',) if config.smoke else tuple(f'ar{i}' for i in range(1, 9))
    seeds = (0,) if config.smoke else (0, 1, 2)
    rows = pl.concat([diagnose_fold(panel, fold, DiagnosticConfig(seed, config.epochs, config.paths))
                      for fold in folds for seed in seeds])
    decision: dict[str, str | int | list[str] | list[int] | DiagnosticDecision] = {
        'status': 'smoke_only' if config.smoke else 'development_diagnostic',
        'epochs': config.epochs, 'paths': config.paths, 'folds': list(folds), 'seeds': list(seeds),
    }
    if not config.smoke:
        verdict = diagnostic_decision(rows)
        decision['diagnostic'] = verdict
        decision['status'] = verdict['status']
    output.mkdir(parents=True)
    rows.write_csv(output / 'diagnostic.csv')
    (output / 'decision.json').write_text(json.dumps(decision, indent=2, allow_nan=False) + '\n')


def _new_output(out: Path, config: DiagnosticRun) -> Path:
    output = out.resolve()
    if output.is_relative_to(RESULTS.resolve()) or output.is_relative_to((ROOT / 'results_ar').resolve()):
        raise ValueError('Flow experiment cannot write to protected results')
    if output.exists():
        raise ValueError(f'Flow experiment output already exists: {output}')
    if config.epochs < 1 or config.paths < 1:
        raise ValueError('Experiment budgets must be positive')
    return output


def _diagnostic_source(source: Path, config: DiagnosticRun) -> pl.DataFrame:
    metadata = json.loads((source / 'decision.json').read_text())
    if metadata.get('epochs') != config.epochs or metadata.get('paths') != config.paths:
        raise ValueError('Diagnostic source budget does not match full experiment')
    rows = pl.read_csv(source / 'diagnostic.csv')
    expected = {(f'ar{i}', seed) for i in range(1, 9) for seed in range(3)}
    actual = {(row['fold'], row['seed']) for row in rows.select('fold', 'seed').unique().iter_rows(named=True)}
    if actual != expected or rows.height != 8 * 3 * 14 * 96 or any(date >= '2021-09-01' for date in rows['date']):
        raise ValueError('Diagnostic source is incomplete or contains sealed dates')
    if diagnostic_decision(rows)['status'] != 'supported':
        raise ValueError('Diagnostic source did not pass the transition gate')
    return rows


def _comparison_fold(panel: Panel, fold: str, config: FitConfig
                     ) -> tuple[list[dict[str, str | int | float]], list[FlowIssue], FlowFit, np.ndarray]:
    blocks = inner_oof(panel, fold, config)
    tau, half_life, _ = backbone.select_tau(panel, fold)
    b3 = hmm.hmm_model('B3', K=3, tau=tau, half_life=half_life,
                       max_epochs=config.epochs, N=config.paths, seed=config.seed)
    state = b3['fit'](panel, fold, blocks.C)
    issues = [issue_day(state, panel, int(day), config.paths)
              for day in features.role_idx(panel, fold, 'val')]
    rows = [candidate_metrics(panel, fold, config.seed, 'B3',
                              [(issue, issue.base) for issue in issues],
                              [(record, record.issue.base) for record in blocks.cal],
                              blocks.C, config.paths)]
    gated: FlowFit | None = None
    for mode in ('mlp', 'q', 'j', 'gated_j'):
        fitted = fit_head(mode, blocks, config)
        outer = [(issue, fitted.predict(issue, blocks.C)) for issue in issues]
        cal = [(record, fitted.predict(record.issue, blocks.C)) for record in blocks.cal]
        rows.append(candidate_metrics(panel, fold, config.seed, mode, outer, cal, blocks.C, config.paths))
        if mode == 'gated_j':
            gated = fitted
    assert gated is not None
    return rows, issues, gated, blocks.C


def _shuffled_control(panel: Panel, runs: dict[tuple[str, int], tuple[list[FlowIssue], FlowFit, np.ndarray]],
                      summary: pl.DataFrame) -> dict[str, object]:
    shuffled: dict[str, float] = {}
    true: dict[str, float] = {}
    for seed in range(3):
        fold_maes: list[float] = []
        for fold in (f'ar{i}' for i in range(1, 9)):
            issues, fitted, C = runs[fold, seed]
            permutation = np.random.default_rng(0).permutation(len(issues))
            predictions = [fitted.predict(replace(issue, flow=issues[int(j)].flow), C)
                           for issue, j in zip(issues, permutation, strict=True)]
            scores = [score_prediction(pred, panel, issue.day, C)
                      for issue, pred in zip(issues, predictions, strict=True)]
            n_slots = sum(score['n_slots'] for score in scores)
            fold_maes.append(sum(score['mae'] * score['n_slots'] for score in scores) / n_slots)
        shuffled[str(seed)] = float(np.mean(fold_maes))
        true[str(seed)] = float(summary.filter((pl.col('seed') == seed) &
                                               (pl.col('candidate') == 'gated_j'))['mae_mean'][0])
    return {'shuffled_mae_by_seed': shuffled, 'true_mae_by_seed': true,
            'direction_contribution_supported': all(shuffled[str(seed)] > true[str(seed)] for seed in range(3))}


def _stability(runs: dict[tuple[str, int], tuple[list[FlowIssue], FlowFit, np.ndarray]]) -> dict[str, float]:
    q_sd: list[float] = []
    j_sd: list[float] = []
    for fold in (f'ar{i}' for i in range(1, 9)):
        flow = np.stack([np.stack([issue.flow for issue in runs[fold, seed][0]]) for seed in range(3)])
        q_sd.append(float(flow.sum(axis=-1).std(axis=0).mean()))
        j_sd.append(float(flow.std(axis=0).mean()))
    return {'q_mean_slotwise_seed_sd': float(np.mean(q_sd)),
            'J_mean_slotwise_seed_sd': float(np.mean(j_sd))}


def run_experiment(out: Path, config: DiagnosticRun, diagnostic_source: Path | None = None) -> None:
    output = _new_output(out, config)
    if not config.smoke:
        panel = experiment_panel(features.load_panel())
        diagnostic = (_diagnostic_source(diagnostic_source, config) if diagnostic_source is not None
                      else pl.concat([diagnose_fold(panel, fold, DiagnosticConfig(seed, config.epochs, config.paths))
                                      for fold in (f'ar{i}' for i in range(1, 9)) for seed in range(3)]))
        verdict = diagnostic_decision(diagnostic)
        output.mkdir(parents=True)
        diagnostic.write_csv(output / 'diagnostic.csv')
        if verdict['status'] != 'supported':
            (output / 'decision.json').write_text(json.dumps({'status': verdict['status'],
                'diagnostic': verdict}, indent=2, allow_nan=False) + '\n')
            return
    else:
        panel = experiment_panel(features.load_panel())
        verdict = None
        output.mkdir(parents=True)
    folds = ('ar8',) if config.smoke else tuple(f'ar{i}' for i in range(1, 9))
    seeds = (0,) if config.smoke else (0, 1, 2)
    metrics: list[dict[str, str | int | float]] = []
    runs: dict[tuple[str, int], tuple[list[FlowIssue], FlowFit, np.ndarray]] = {}
    for fold in folds:
        for seed in seeds:
            rows, issues, fitted, C = _comparison_fold(panel, fold, FitConfig(seed, config.epochs, config.paths))
            metrics.extend(rows)
            runs[fold, seed] = issues, fitted, C
            pl.DataFrame(metrics).write_csv(output / 'metrics.csv')
            print(f'{fold} seed={seed}: five candidates scored', flush=True)
    frame = pl.DataFrame(metrics)
    decision: dict[str, object] = {
        'status': 'smoke_only', 'epochs': config.epochs, 'head_max_epochs': min(config.epochs, 50),
        'paths': config.paths, 'folds': list(folds), 'seeds': list(seeds),
        'formula': 'J[h,i,j]=q_pre[h,i]*P[h,i,j]; gated_delta=switch_mass*FiLM(J_offdiag,x)',
        'features': 'A+ issue-known 14 features plus B3 median; six directed off-diagonal J cells',
        'seed_rule': 'B3 fit and neural initialisation by candidate seed; paths by issue date',
        'scope': 'pre_September_development_only',
        'caveat': 'A+ operating flags are proxies reconstructed from realized production, not confirmed future schedules',
    }
    if not config.smoke:
        summary, selection = summarize(frame)
        summary.write_csv(output / 'summary.csv')
        decision.update(selection)
        decision['diagnostic'] = verdict
        decision['stability'] = _stability(runs)
        if selection['qualified']:
            decision['negative_control'] = _shuffled_control(panel, runs, summary)
        else:
            decision['negative_control'] = {'status': 'not_run_candidate_not_qualified'}
    (output / 'decision.json').write_text(json.dumps(decision, indent=2, allow_nan=False) + '\n')


def main() -> None:
    parser = argparse.ArgumentParser(description='AR-free B3 transition-flow experiment')
    parser.add_argument('--diagnostic', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--out', type=Path, default=ROOT / 'results_flow')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--paths', type=int)
    parser.add_argument('--diagnostic-source', type=Path)
    args = parser.parse_args()
    if args.diagnostic and args.smoke:
        parser.error('Choose one mode')
    if args.diagnostic_source is not None and (args.diagnostic or args.smoke):
        parser.error('--diagnostic-source requires full mode')
    try:
        epochs = args.epochs if args.epochs is not None else (2 if args.smoke else 300)
        paths = args.paths if args.paths is not None else (50 if args.smoke else 2000)
        config = DiagnosticRun(args.smoke, epochs, paths)
        if args.diagnostic:
            run_diagnostic(args.out, config)
        else:
            run_experiment(args.out, config, args.diagnostic_source)
    except ValueError as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()

# Session Notes - Judge Optimization

## Status: Running Judge Experiments (3 of 4 verticals complete)

### IMPORTANT REMINDER

**Domino Jobs will NOT run successfully** because the workspace is syncing to branch `6.2` instead of the main repo. The job environment pulls from main, so any uncommitted or branch-only changes (like `scripts/register_best_judges.py`) won't be available to the job.

**To fix:** Either merge branch `6.2` to main, or run experiments locally without `--submit-job`.

### Bug Fix Applied

**Fixed Domino Job syntax error** in `scripts/run_judge_experiment.py`:
- The `--submit-job` command was generating inline Python with quoting issues
- Created dedicated `scripts/register_best_judges.py` script to avoid quoting problems
- Job now calls: `python scripts/register_best_judges.py --vertical {vertical} --batch-id {batch_id}`

### Completed Judge Experiments

| Vertical | Best Config | Agreement | Consistency |
|----------|-------------|-----------|-------------|
| financial_services | gpt-4o-mini-t0.0-direct-binary | 100% | 0.000 |
| healthcare | gpt-4o-mini-t0.0-direct-binary | 100% | 0.000 |
| energy | gpt-4o-mini-t0.0-direct-binary | 100% | 0.000 |

**Key finding:** Binary scale significantly outperforms three_point scale across all verticals tested. All binary configs achieve 100% agreement.

### Next: Run Last Judge Experiment

```bash
python scripts/run_judge_experiment.py --vertical public_sector --model gpt-4o-mini --runs-per-config 2
```

(Run without `--submit-job` until branch is merged to main)

### After Judge Experiments: Local Model Optimization

Once all 4 verticals have validated judges, run agent optimization:

```bash
python scripts/run_local_optimization.py --vertical healthcare --submit-job
```

**Note:** Local Qwen endpoint may need verification - was returning 404 previously.

### Local Model Endpoint Issue

The Qwen endpoint (`https://genai-llm.domino-eval.com/endpoints/bf209962-1bd0-4524-87c8-2d0ac662a022/`) returns 404 for all tested paths:
- `/v1/chat/completions`
- `/v2/chat/completions`
- `/tool-calling/v1/chat/completions`

**Action needed**: Verify the endpoint is running and check the correct API path in the Domino Model Endpoints UI.

### Key Files

| File | Purpose |
|------|---------|
| `scripts/run_judge_experiment.py` | Judge optimization script |
| `scripts/run_local_optimization.py` | Agent optimization script |
| `configs/judges.yaml` | Judge configs (stores optimized results) |
| `configs/agents.yaml` | Agent configs |
| `example-data/ground_truth_judgments.yaml` | Human labels for all verticals |

### Domino Jobs Submitted

| Job | Vertical | Job ID | Status |
|-----|----------|--------|--------|
| BEST-JUDGES-healthcare | healthcare | 694ac80741d3de61e5838fb6 | Failed (quoting bug) |
| BEST-JUDGES-energy | energy | 694ad48941d3de61e5838fe2 | Failed (quoting bug) |
| BEST-JUDGES-financial_services | financial_services | 694c1969b1957a560829dd85 | Pending (will fail - branch issue) |

Note: Jobs fail because workspace syncs to branch `6.2` instead of main.

### Temperature Recommendations

Judges use 0.0 temperature for consistency.

| Agent | Baseline | Local Range |
|-------|----------|-------------|
| Classifier | 0.2 | 0.1-0.3 |
| ImpactAssessor | 0.3 | 0.2-0.4 |
| ResourceMatcher | 0.1 | 0.0-0.2 |
| ResponseDrafter | 0.6 | 0.5-0.7 |

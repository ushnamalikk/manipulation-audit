"""build_df.py — turn Inspect .eval logs into results/scored.csv + a gold-labeling template.
Run after `inspect eval`. Requires inspect-ai.
"""
import pandas as pd
from pathlib import Path
from inspect_ai.log import list_eval_logs, read_eval_log

SCORER = "multi_judge"


def build(logs_dir="logs", out="results/scored.csv"):
    rows = []
    for info in list_eval_logs(logs_dir):
        log = read_eval_log(info.name)
        target = log.eval.model
        for s in (log.samples or []):
            md = s.metadata or {}
            score = (s.scores or {}).get(SCORER) or next(iter((s.scores or {}).values()), None)
            if score is None:
                continue
            cell = f"{target}|{md.get('family')}|{md.get('paraphrase')}|{md.get('persona')}"
            rid = f"{cell}|ep{s.epoch}"
            text = s.output.completion if s.output else ""
            base = dict(response_id=rid, cell_id=cell, model=target, epoch=s.epoch,
                        family=md.get("family"), paraphrase=md.get("paraphrase"),
                        persona=md.get("persona"), pattern=md.get("pattern"),
                        response_text=text)
            verdicts = (score.metadata or {}).get("verdicts")
            if verdicts:
                for j, v in verdicts.items():
                    rows.append({**base, "judge": j, "present": int(bool(v.get("present"))),
                                 "severity": int(v.get("severity", 0)), "evidence": v.get("evidence")})
            else:
                m = score.metadata or {}
                rows.append({**base, "judge": m.get("judge", "single"),
                             "present": int(bool(m.get("present"))),
                             "severity": int(m.get("severity", 0)), "evidence": m.get("evidence")})
    df = pd.DataFrame(rows)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"wrote {len(df)} rows -> {out} | {df.model.nunique()} targets, "
          f"{df.judge.nunique()} judges, {df.response_id.nunique()} responses")
    return df


def gold_template(scored="results/scored.csv", out="results/gold_template.csv", n=120, seed=42):
    df = pd.read_csv(scored)
    resp = (df.drop_duplicates("response_id")
              [["response_id", "pattern", "persona", "family", "response_text"]]
              .sample(min(n, df.response_id.nunique()), random_state=seed))
    resp["human_present"] = ""
    resp["human_severity"] = ""
    resp.to_csv(out, index=False)
    print(f"label these {len(resp)} responses -> {out} (then save as results/gold.csv)")


if __name__ == "__main__":
    build()
    gold_template()

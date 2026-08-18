"""task.py — persona-conditional manipulation audit (Inspect harness).
Requires Python 3.10+ and `pip install inspect-ai`. Run:
    inspect eval manipulation_audit/task.py --model ollama/qwen2.5:7b --epochs 3 --temperature 0.4
"""
import numpy as np
from inspect_ai import Task, task
from inspect_ai.dataset import Sample, MemoryDataset
from inspect_ai.solver import generate
from inspect_ai.scorer import scorer, Score, mean, stderr
from inspect_ai.model import get_model

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from manipulation_audit.rubric import build_judge_prompt, _parse, iter_scenarios
from manipulation_audit.personas import PERSONAS

# The judge model(s) you actually have configured.
# MVP: one judge is fine. Add a second (different) model later for inter-judge reliability.
JUDGES = ["ollama/qwen2.5:7b", "ollama/qwen2.5:3b"]   # two judges -> inter-judge reliability


def build_dataset():
    samples = []
    for sc in iter_scenarios():
        for pid, prefix in PERSONAS.items():
            samples.append(Sample(
                input=prefix + sc["text"],
                metadata={"family": sc["family"], "paraphrase": sc["paraphrase"],
                          "persona": pid, "pattern": sc["pattern"]}))
    return MemoryDataset(samples)


@scorer(metrics=[mean(), stderr()])
def multi_judge():
    models = {j: get_model(j) for j in JUDGES}

    async def score(state, target):
        md = state.metadata
        prompt = build_judge_prompt(md["pattern"], state.input_text, state.output.completion)
        verdicts = {}
        for jname, jm in models.items():
            out = await jm.generate(prompt)
            verdicts[jname] = _parse(out.completion)
        sev = float(np.mean([v["severity"] for v in verdicts.values()])) / 3
        return Score(value=sev, metadata={**md, "verdicts": verdicts})

    return score


@task
def manipulation_audit():
    return Task(dataset=build_dataset(), solver=generate(), scorer=multi_judge())

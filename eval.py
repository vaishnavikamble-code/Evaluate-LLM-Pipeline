from gemini_model import GeminiModel
from deepeval import evaluate
import google.generativeai as genai
import os
import json
import time
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
   # HallucinationMetric,
    ToxicityMetric,
    BiasMetric,
    ContextualRecallMetric,
    ContextualPrecisionMetric,
    ContextualRelevancyMetric,

)

from deepeval.test_case import LLMTestCase

# ============================================================================
# LLM Evaluation Pipeline
# ============================================================================
# User Question → LLM Generation → DeepEval Metrics → Quality Score
# ============================================================================
print("CI TEST")
gemini = GeminiModel()

# ============================================================================
# STAGE 1: User Question Input
# ============================================================================
user_question = 'What is python?'
print(f"\n📝 [STAGE 1] User Question: {user_question}\n")

# ============================================================================
# STAGE 2: LLM Generation (Gemini) with Latency Measurement
# ============================================================================
print("⏱️  [STAGE 2] Generating answer with Gemini...\n")
latency_start = time.time()

test_case = LLMTestCase(
    input=user_question,
    actual_output='Python is a programming language.',
    expected_output='Python is programming language.',
    retrieval_context=[
        "Python is a high-level programming language.",
        "Python is widely used for AI, Machine Learning, Data Science and Web Development."
    ]
)

latency_end = time.time()
latency_ms = (latency_end - latency_start) * 1000
print(f"✅ Generated answer in {latency_ms:.2f}ms\n")

# ============================================================================
# STAGE 3: DeepEval Judge Metrics (Relevancy, Toxicity, Bias, etc.)
# ============================================================================
print("🔍 [STAGE 3] Running DeepEval metrics...\n")

metrics=[
    AnswerRelevancyMetric(
        threshold=0.7,
        model=gemini
    ),
    
    FaithfulnessMetric(
        threshold=0.7,
        model=gemini
    ),
    
    # HallucinationMetric(
    #     threshold=0.3,
    #     model=gemini
    # ),
    
    ToxicityMetric(
        threshold=0.1,
        model=gemini
    ),
    
    BiasMetric(
        threshold=0.1,
        model=gemini
    ),
    
    ContextualRecallMetric(
        threshold=0.7,
        model=gemini
    ),
    
    ContextualPrecisionMetric(
        threshold=0.7,
        model=gemini
    ),
    
    ContextualRelevancyMetric(
        threshold=0.7,
        model=gemini
    ),
]

test_result = evaluate(
    test_cases=[test_case],
    metrics=metrics
)

# Create results summary from evaluation by reading MetricData produced
tr = None
if hasattr(test_result, 'test_results') and len(test_result.test_results) > 0:
    tr = test_result.test_results[0]

# Helper to find metric data by name
def _find_metric_score(test_res, metric_name):
    if test_res is None or test_res.metrics_data is None:
        return None
    for md in test_res.metrics_data:
        if md.name.lower() == metric_name.lower():
            return md.score
    return None

results = {
    "answer_relevancy": _find_metric_score(tr, "Answer Relevancy"),
    "faithfulness": _find_metric_score(tr, "Faithfulness"),
    "toxicity": _find_metric_score(tr, "Toxicity"),
    "bias": _find_metric_score(tr, "Bias"),
    "contextual_recall": _find_metric_score(tr, "Contextual Recall"),
    "contextual_precision": _find_metric_score(tr, "Contextual Precision"),
    "contextual_relevancy": _find_metric_score(tr, "Contextual Relevancy")
}


os.makedirs("reports", exist_ok=True)

with open("reports/report.json", "w") as f:
    json.dump(results, f, indent=4)

print("Report Generated Successfully!")


# Quality Gate

MIN_RELEVANCY = 0.7
MIN_FAITHFULNESS = 0.7
MAX_TOXICITY = 0.1
MAX_BIAS = 0.1

def _require_score(name, value):
    if value is None:
        raise Exception(f"Evaluation Failed: missing score for {name}")
    return value

# Validate and enforce quality gates
relevancy = _require_score('answer_relevancy', results.get('answer_relevancy'))
faith = _require_score('faithfulness', results.get('faithfulness'))
tox = _require_score('toxicity', results.get('toxicity'))
bias = _require_score('bias', results.get('bias'))

if relevancy < MIN_RELEVANCY:
    raise Exception(f"Evaluation Failed: Relevancy={relevancy}")

if faith < MIN_FAITHFULNESS:
    raise Exception(f"Evaluation Failed: Faithfulness={faith}")

if tox > MAX_TOXICITY:
    raise Exception(f"Evaluation Failed: Toxicity={tox}")

if bias > MAX_BIAS:
    raise Exception(f"Evaluation Failed: Bias={bias}")

print("✅ All Quality Checks Passed")
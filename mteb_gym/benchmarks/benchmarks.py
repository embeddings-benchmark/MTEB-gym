from __future__ import annotations

from .benchmark import Benchmark

PAPER_TASKS = (
    "ArguAna",
    "ClimateFEVERHardNegatives",
    "CQADupstackGamingRetrieval",
    "CQADupstackUnixRetrieval",
    "FEVERHardNegatives",
    "FiQA2018",
    "HotpotQAHardNegatives",
    "SCIDOCS",
    "Touche2020Retrieval.v3",
    "TRECCOVID",
    "BrightAopsRetrieval",
    "BrightBiologyRetrieval",
    "BrightEarthScienceRetrieval",
    "BrightEconomicsRetrieval",
    "BrightLeetcodeRetrieval",
    "BrightPonyRetrieval",
    "BrightPsychologyRetrieval",
    "BrightRoboticsRetrieval",
    "BrightStackoverflowRetrieval",
    "BrightSustainableLivingRetrieval",
    "BrightTheoremQAQuestionsRetrieval",
    "BrightTheoremQATheoremsRetrieval",
    "AILACasedocs",
    "AppsRetrieval",
    "CUREv1",
    "FinQARetrieval",
    "NFCorpus",
    "SciFact",
)

GYM_ENG_V1 = Benchmark(
    name="Gym(eng, v1)",
    tasks=PAPER_TASKS,
    judge="autotrust/JEV-27B",
    judge_revision="14b425269ed24452e0a5ad5a8f3e51d139647a88",
    judging="pointwise",
    query_source="synthetic",
    n_queries=100,
    generator="deepseek-ai/DeepSeek-V4-Flash+0e4b4485586e",
    description="28 English retrieval tasks, each judged on 100 queries written from its documents by DeepSeek-V4-Flash, "
    "with JEV-27B grading every retrieved document.",
    reference="https://github.com/embeddings-benchmark/MTEB-gym",
)

GYM_ENG_V1_PAIRWISE = Benchmark(
    name="Gym(eng, v1, pairwise)",
    tasks=PAPER_TASKS,
    judge="autotrust/JEV-27B",
    judge_revision="14b425269ed24452e0a5ad5a8f3e51d139647a88",
    judging="pairwise",
    query_source="synthetic",
    n_queries=100,
    generator="deepseek-ai/DeepSeek-V4-Flash+0e4b4485586e",
    description="Gym(eng, v1) with JEV-27B comparing two models' result lists at a time.",
    reference="https://github.com/embeddings-benchmark/MTEB-gym",
)

GYM_ENG_V1_VALIDATION = Benchmark(
    name="Gym(eng, v1, original queries)",
    tasks=PAPER_TASKS,
    judge="autotrust/JEV-27B",
    judge_revision="14b425269ed24452e0a5ad5a8f3e51d139647a88",
    judging="pointwise",
    query_source="original",
    n_queries=100,
    description="The same tasks judged on their own labelled queries, to compare the judge's ranking with the labels'.",
    reference="https://github.com/embeddings-benchmark/MTEB-gym",
)

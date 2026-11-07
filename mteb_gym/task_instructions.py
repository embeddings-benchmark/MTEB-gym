"""A sentence describing the retrieval task, for a set of MTEB tasks: pass one as `task_instruction` to use it."""

_STACKEXCHANGE = "Given a {} question, retrieve web documents that provide the information needed to answer it"

TASK_INSTRUCTIONS = {
    "ArguAna": "Given an argument, retrieve its best counterargument",
    "ClimateFEVERHardNegatives": "Given a claim about climate change, retrieve Wikipedia passages that support or refute the claim",
    "CQADupstackGamingRetrieval": "Given a question from a gaming forum, retrieve earlier forum questions that ask the same thing",
    "CQADupstackUnixRetrieval": "Given a question from a Unix forum, retrieve earlier forum questions that ask the same thing",
    "FEVERHardNegatives": "Given a claim, retrieve Wikipedia passages that support or refute the claim",
    "FiQA2018": "Given a financial question, retrieve user replies that best answer the question",
    "HotpotQAHardNegatives": "Given a question that requires combining several facts, retrieve the Wikipedia passages needed to answer it",
    "SCIDOCS": "Given a scientific paper title, retrieve paper abstracts that are cited by the given paper",
    "Touche2020Retrieval.v3": "Given a question on a controversial topic, retrieve arguments from either side that help answer it",
    "TRECCOVID": "Given a query on COVID-19, retrieve documents that answer the query",
    "BrightAopsRetrieval": "Given a math problem, retrieve solved problems that use the same technique",
    "BrightBiologyRetrieval": _STACKEXCHANGE.format("biology"),
    "BrightEarthScienceRetrieval": _STACKEXCHANGE.format("earth science"),
    "BrightEconomicsRetrieval": _STACKEXCHANGE.format("economics"),
    "BrightLeetcodeRetrieval": "Given a coding problem, retrieve solved problems that use the same algorithm",
    "BrightPonyRetrieval": "Given a coding problem in the Pony language, retrieve the Pony syntax documentation needed to solve it",
    "BrightPsychologyRetrieval": _STACKEXCHANGE.format("psychology"),
    "BrightRoboticsRetrieval": _STACKEXCHANGE.format("robotics"),
    "BrightStackoverflowRetrieval": _STACKEXCHANGE.format("programming"),
    "BrightSustainableLivingRetrieval": _STACKEXCHANGE.format("sustainable living"),
    "BrightTheoremQAQuestionsRetrieval": "Given a problem, retrieve solved problems that use the same theorem",
    "BrightTheoremQATheoremsRetrieval": "Given a problem, retrieve the theorems needed to solve it",
    "AILACasedocs": "Given a description of a legal situation, retrieve relevant prior judgments of the Supreme Court of India",
    "AppsRetrieval": "Given a programming problem, retrieve code that solves it",
    "CUREv1": "Given a question by a medical professional, retrieve relevant passages that best answer the question",
    "FinQARetrieval": "Given a financial numerical reasoning question, retrieve relevant financial data that helps answer the question",
    "NFCorpus": "Given a health or nutrition topic, retrieve medical research documents relevant to it",
    "SciFact": "Given a scientific claim, retrieve documents that support or refute the claim",
}

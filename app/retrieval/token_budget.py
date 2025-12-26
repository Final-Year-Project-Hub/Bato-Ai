from dataclasses import dataclass


@dataclass
class TokenBudget:
    system_tokens: int
    query_tokens: int
    retrieval_tokens: int
    answer_tokens: int

    @property
    def total(self) -> int:
        return (
            self.system_tokens
            + self.query_tokens
            + self.retrieval_tokens
            + self.answer_tokens
        )


class TokenBudgetPlanner:
    """
    Token planning based on ONLY two intents:
    - learn
    - build
    """

    def __init__(self, model_max_tokens: int = 4096):
        self.model_max_tokens = model_max_tokens

    def plan(self, intent: str, depth: str) -> TokenBudget:
        system_tokens = 500
        query_tokens = 100

        # Base defaults
        if intent == "learn":
            retrieval_tokens = 1400
            answer_tokens = 1000
        else:  # build
            retrieval_tokens = 2000
            answer_tokens = 700

        # Depth adjustment
        if depth == "conceptual":
            retrieval_tokens -= 300
            answer_tokens += 300

        if depth == "practical":
            retrieval_tokens += 200
            answer_tokens -= 200

        # Safety clamps
        retrieval_tokens = max(800, retrieval_tokens)
        answer_tokens = max(500, answer_tokens)

        total = system_tokens + query_tokens + retrieval_tokens + answer_tokens

        if total > self.model_max_tokens:
            overflow = total - self.model_max_tokens
            retrieval_tokens -= overflow

        return TokenBudget(
            system_tokens=system_tokens,
            query_tokens=query_tokens,
            retrieval_tokens=retrieval_tokens,
            answer_tokens=answer_tokens
        )

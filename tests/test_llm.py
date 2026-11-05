import types


def test_llm_drops_rejected_params():
    """A model that refuses a parameter (400 naming it) still answers."""
    from mteb_gym.llm import LLM

    class Refused(Exception):
        status_code = 400

    calls = []

    class Completions:
        def create(self, **kw):
            calls.append(sorted(k for k in kw if k in ("temperature", "max_completion_tokens")))
            for k in ("temperature", "max_completion_tokens"):
                if k in kw:
                    raise Refused(f"Unsupported parameter: '{k}' is not supported with this model.")
            return types.SimpleNamespace(
                model="m-2026-01-01", choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="ok"))]
            )

    llm = LLM("m", api_key="x", max_tokens=512, temperature=0.0)
    llm.client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=Completions()))
    assert llm.chat([{"role": "user", "content": "hi"}]) == "ok"
    assert llm.chat([{"role": "user", "content": "hi"}]) == "ok"
    assert calls == [["max_completion_tokens", "temperature"], ["max_completion_tokens"], [], []]  # learned once
    from mteb_gym.llm import llm_settings

    assert llm.served_model == "m-2026-01-01"
    assert llm_settings(llm)["refused"] == ["max_completion_tokens", "temperature"]  # configured, but refused

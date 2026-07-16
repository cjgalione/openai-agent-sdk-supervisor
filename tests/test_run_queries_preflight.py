from scripts import run_queries


def test_preflight_classifies_known_failures() -> None:
    assert run_queries._preflight_failure_category(RuntimeError("Incorrect API key")) == "authentication"
    assert run_queries._preflight_failure_category(RuntimeError("insufficient_quota")) == "quota"
    assert run_queries._preflight_failure_category(RuntimeError("connection timed out")) == "transient"


def test_preflight_error_does_not_echo_key(monkeypatch) -> None:
    monkeypatch.setenv("BRAINTRUST_API_KEY", "test-braintrust")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai")
    monkeypatch.setenv("EXA_API_KEY", "test-exa")
    monkeypatch.setattr(
        run_queries,
        "_openai_client",
        lambda: (_ for _ in ()).throw(RuntimeError("invalid API key sk-test-secret")),
    )

    try:
        run_queries._run_preflight()
    except RuntimeError as exc:
        assert str(exc) == "Provider preflight failed (authentication)."
        assert "sk-test-secret" not in str(exc)
    else:
        raise AssertionError("Expected preflight failure")

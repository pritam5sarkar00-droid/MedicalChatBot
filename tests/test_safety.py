from src.safety import detect_emergency


def test_detects_chest_pain():
    assert detect_emergency("I have really bad chest pain and can't breathe") is True


def test_detects_suicidal_language():
    assert detect_emergency("I want to end my life") is True


def test_detects_unconsciousness():
    assert detect_emergency("My friend just fainted and is unresponsive") is True


def test_ignores_normal_question():
    assert detect_emergency("What are the symptoms of a common cold?") is False


def test_ignores_unrelated_pain_mention():
    # "pain" alone shouldn't trip it -- only specific emergency phrases should
    assert detect_emergency("I have some mild knee pain after running") is False


def test_case_insensitive():
    assert detect_emergency("CHEST PAIN right now, what do I do") is True


def test_empty_string():
    assert detect_emergency("") is False


def test_none_input_is_safe():
    assert detect_emergency(None) is False

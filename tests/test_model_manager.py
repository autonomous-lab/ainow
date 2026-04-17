from src.services.model_manager import _should_load_mmproj


def test_mmproj_respects_vision_flag(monkeypatch):
    monkeypatch.delenv("AUDIO_LLM", raising=False)
    assert _should_load_mmproj({"mmproj": "vision.gguf"}, True) is True
    assert _should_load_mmproj({"mmproj": "vision.gguf"}, False) is False


def test_mmproj_loads_for_audio_llm(monkeypatch):
    monkeypatch.setenv("AUDIO_LLM", "1")
    assert _should_load_mmproj({"mmproj": "vision.gguf"}, False) is True

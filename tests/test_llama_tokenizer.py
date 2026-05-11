import importlib.util
import json
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "llama_tokenizer.py"
SPEC = importlib.util.spec_from_file_location("llama_tokenizer", MODULE_PATH)
llama_tokenizer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(llama_tokenizer)

LlamaTokenizer = llama_tokenizer.LlamaTokenizer
SPACE = "\u2581"


def make_tokenizer(pieces):
    tok = LlamaTokenizer()
    for idx, piece in enumerate(pieces):
        tok.vocab[piece] = idx
        tok.vocab_inv[idx] = piece
        tok.scores[idx] = float(idx)
    return tok


def test_load_vocab_txt_parses_tab_space_and_plain_tokens(tmp_path, capsys):
    vocab_path = tmp_path / "toy.vocab"
    vocab_path.write_text(
        "<unk>\t-1.5\n"
        "<s> -2.0\n"
        "plain token\n"
        "scoreless\n"
        "not_float nope\n",
        encoding="utf-8",
    )

    tok = LlamaTokenizer(vocab_path)

    assert tok.vocab["<unk>"] == 0
    assert tok.scores[0] == -1.5
    assert tok.vocab["<s>"] == 1
    assert tok.scores[1] == -2.0
    assert tok.vocab["plain token"] == 2
    assert tok.scores[2] == 0.0
    assert tok.vocab["scoreless"] == 3
    assert tok.vocab["not_float nope"] == 4
    assert "Loaded 5 tokens" in capsys.readouterr().out


def test_encode_decode_uses_space_marker_and_byte_fallback():
    tok = make_tokenizer(["<unk>", "<s>", "</s>", SPACE, "h", "i", "<0x21>"])

    encoded = tok.encode("hi!", add_bos=True, add_eos=True)

    assert encoded == [tok.bos_id, 3, 4, 5, 6, tok.eos_id]
    assert tok.decode(encoded) == "hi!"


def test_encode_prefers_lowest_score_bpe_pair():
    tok = make_tokenizer(["<unk>", "<s>", "</s>", SPACE, "a", "b", "c", "ab", "bc"])
    tok.scores[tok.vocab["ab"]] = 10.0
    tok.scores[tok.vocab["bc"]] = -1.0

    assert tok.encode("abc", add_bos=False) == [3, 4, 8]


def test_encode_and_decode_require_loaded_vocabulary():
    tok = LlamaTokenizer()

    with pytest.raises(ValueError, match="Tokenizer not loaded"):
        tok.encode("anything")

    with pytest.raises(ValueError, match="Tokenizer not loaded"):
        tok.decode([1, 2, 3])


def test_decode_skips_special_tokens_and_labels_unknown_ids():
    tok = make_tokenizer(["<unk>", "<s>", "</s>", SPACE, "x"])

    assert tok.decode([tok.bos_id, 99, tok.eos_id, tok.pad_id]) == "[99]"


def test_load_json_simple_vocab_builds_forward_and_reverse_maps(tmp_path, capsys):
    vocab_path = tmp_path / "tokenizer.json"
    vocab_path.write_text(
        json.dumps({"vocab": {"<unk>": 0, "<s>": 1, "</s>": 2, SPACE: 3, "x": 4}}),
        encoding="utf-8",
    )

    tok = LlamaTokenizer(vocab_path)

    assert tok.vocab_size == 5
    assert len(tok) == 5
    assert tok.vocab[SPACE] == 3
    assert tok.vocab_inv[4] == "x"
    assert "Loaded 5 tokens from JSON" in capsys.readouterr().out


def test_varint_field_and_skip_field_helpers_cover_wire_types():
    tok = LlamaTokenizer()

    value, pos = tok._read_varint(bytes([0xAC, 0x02]), 0)
    assert (value, pos) == (300, 2)

    field_num, wire_type, pos = tok._read_varint_field(bytes([(5 << 3) | 2]), 0)
    assert (field_num, wire_type, pos) == (5, 2, 1)

    assert tok._skip_field(b"\x96\x01tail", 0, 0) == 2
    assert tok._skip_field(b"abcdefghTAIL", 0, 1) == 8
    assert tok._skip_field(b"\x03abcTAIL", 0, 2) == 4
    assert tok._skip_field(b"abcdTAIL", 0, 5) == 4

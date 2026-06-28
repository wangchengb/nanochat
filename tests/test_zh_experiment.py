import pyarrow as pa
import pyarrow.parquet as pq
import os
import regex

from nanochat.dataloader import _document_batches
from nanochat.tokenizer import (
    IncrementalTextDecoder,
    get_split_pattern,
    resolve_tokenizer_dir,
)
from scripts.prepare_zh_experiment_data import cjk_language_ratio, normalize_messages
from scripts.prepare_bilingual_pretrain_data import chunk_document
from scripts.zh_eval import response_metrics, summarize


class ByteTokenizer:
    enc = None

    def __init__(self):
        self.enc = self

    def encode(self, text):
        return list(text.encode("utf-8"))

    def decode_bytes(self, token_ids):
        return bytes(token_ids)

    def decode(self, token_ids):
        return bytes(token_ids).decode("utf-8", errors="replace")


def test_incremental_decoder_hides_partial_utf8():
    decoder = IncrementalTextDecoder(ByteTokenizer())
    encoded = list("你".encode("utf-8"))
    assert decoder.push(encoded[0]) == ""
    assert decoder.push(encoded[1]) == ""
    assert decoder.push(encoded[2]) == "你"


def test_normalize_alpaca_messages():
    messages = normalize_messages({
        "instruction": "介绍机器学习",
        "input": "面向初学者",
        "output": "机器学习让计算机从数据中学习。",
    })
    assert messages == [
        {"role": "user", "content": "介绍机器学习\n\n面向初学者"},
        {"role": "assistant", "content": "机器学习让计算机从数据中学习。"},
    ]


def test_normalize_conversations_messages():
    messages = normalize_messages({
        "conversations": [
            {"from": "human", "value": "你好"},
            {"from": "gpt", "value": "你好！"},
        ]
    })
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"


def test_response_metrics_language_ratio():
    metrics = response_metrics("你好，this is a test")
    assert metrics["cjk_chars"] == 2
    assert metrics["latin_chars"] == 11
    assert not metrics["contains_replacement_character"]
    assert cjk_language_ratio("这是中文回答") == 1.0


def test_empty_response_does_not_pass_language_checks():
    records = [
        {"language": "zh", "metrics": response_metrics("")},
        {"language": "en", "metrics": response_metrics("")},
    ]
    summary = summarize(records)
    assert summary["zh_response_rate"] == 0.0
    assert summary["en_response_rate"] == 0.0


def test_document_batches_accept_custom_data_dir(tmp_path):
    train_path = tmp_path / "shard_00000.parquet"
    val_path = tmp_path / "shard_99999.parquet"
    pq.write_table(pa.table({"text": ["train-a", "train-b"]}), train_path)
    pq.write_table(pa.table({"text": ["val-a"]}), val_path)

    train_batches = _document_batches("train", None, tokenizer_batch_size=8, data_dir=tmp_path)
    val_batches = _document_batches("val", None, tokenizer_batch_size=8, data_dir=tmp_path)
    assert next(train_batches)[0] == ["train-a", "train-b"]
    assert next(val_batches)[0] == ["val-a"]


def test_custom_data_dir_requires_train_and_validation_shards(tmp_path):
    pq.write_table(pa.table({"text": ["only-shard"]}), tmp_path / "shard_00000.parquet")

    try:
        next(_document_batches("train", None, tokenizer_batch_size=8, data_dir=tmp_path))
    except AssertionError as error:
        assert "Custom data directory" in str(error)
    else:
        raise AssertionError("Expected a custom data directory with one shard to be rejected")


def test_tagged_tokenizer_directory_is_isolated(tmp_path):
    original = os.environ.get("NANOCHAT_BASE_DIR")
    os.environ["NANOCHAT_BASE_DIR"] = str(tmp_path)
    try:
        assert resolve_tokenizer_dir() == str(tmp_path / "tokenizer")
        assert resolve_tokenizer_dir(tokenizer_tag="bilingual-32k-v1") == str(
            tmp_path / "tokenizers" / "bilingual-32k-v1"
        )
    finally:
        if original is None:
            os.environ.pop("NANOCHAT_BASE_DIR", None)
        else:
            os.environ["NANOCHAT_BASE_DIR"] = original


def test_bilingual_split_pattern_bounds_han_and_separates_latin():
    text = "机器学习让计算机处理数据。Hello world! 中英文mixed测试。"
    pieces = regex.findall(
        get_split_pattern("bilingual-han4"),
        text,
        flags=regex.VERSION1,
    )
    han_pieces = [
        piece for piece in pieces
        if any(0x3400 <= ord(char) <= 0x9FFF for char in piece)
    ]
    assert all(
        sum(0x3400 <= ord(char) <= 0x9FFF for char in piece) <= 4
        for piece in han_pieces
    )
    assert "mixed" in pieces
    assert get_split_pattern("default") != get_split_pattern("bilingual-han4")


def test_han1_split_pattern_separates_each_chinese_character():
    pieces = regex.findall(
        get_split_pattern("bilingual-han1"),
        "机器学习 English",
        flags=regex.VERSION1,
    )
    assert pieces[:4] == ["机", "器", "学", "习"]
    assert pieces[-1] == " English"


def test_pretrain_chunking_preserves_utf8_and_token_limit():
    tokenizer = ByteTokenizer()
    text = "中文English混合文本" * 50
    chunks = list(
        chunk_document(
            tokenizer,
            text,
            max_chunk_tokens=31,
            min_chunk_tokens=1,
        )
    )
    assert "".join(chunk["text"] for chunk in chunks) == text
    assert all(chunk["content_tokens"] <= 31 for chunk in chunks)
    assert all(
        chunk["training_tokens"] == chunk["content_tokens"] + 1
        for chunk in chunks
    )

from __future__ import annotations

from pathlib import Path

from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, processors, trainers
from transformers import PreTrainedTokenizerFast

SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[BOS]", "[EOS]"]


def opus_books_text_iterator(dataset, batch_size: int = 1000):
    for start in range(0, len(dataset), batch_size):
        batch = dataset[start : start + batch_size]["translation"]
        yield [text for pair in batch for text in pair.values()]


def build_bpe_tokenizer(dataset, vocab_size: int = 16384, batch_size: int = 1000) -> PreTrainedTokenizerFast:
    tokenizer = Tokenizer(models.BPE(continuing_subword_prefix="##"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.normalizer = normalizers.NFKC()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        continuing_subword_prefix="##",
    )
    tokenizer.train_from_iterator(opus_books_text_iterator(dataset, batch_size), trainer=trainer)
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[BOS] $A [EOS]",
        special_tokens=[
            ("[BOS]", tokenizer.token_to_id("[BOS]")),
            ("[EOS]", tokenizer.token_to_id("[EOS]")),
        ],
    )
    tokenizer.decoder = decoders.WordPiece(prefix="##")
    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="[BOS]",
        eos_token="[EOS]",
        pad_token="[PAD]",
        unk_token="[UNK]",
    )


def save_tokenizer(tokenizer: PreTrainedTokenizerFast, path: str | Path) -> None:
    tokenizer.save_pretrained(path)


def load_tokenizer(path: str | Path) -> PreTrainedTokenizerFast:
    return PreTrainedTokenizerFast.from_pretrained(path)

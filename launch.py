FASTTEXT_EXECUTABLE = 'fasttext'
PRETRAINED_MODEL_FILE = 'wiki.en.bin'
VOCAB_FILE = 'quora.vocab'
OUTPUT_FILE = 'quora.vec'
EMBEDDING_DIM = 300

with open(OUTPUT_FILE, 'w') as f:
    print(f'{len(vocab)} {EMBEDDING_DIM}', file=f)

with open(VOCAB_FILE) as f_vocab:
    with open(OUTPUT_FILE, 'a') as f_output:
        subprocess.run(
            [FASTTEXT_EXECUTABLE, 'print-word-vectors', PRETRAINED_MODEL_FILE],
            stdin=f_vocab,
            stdout=f_output,
        )
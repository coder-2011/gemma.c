from tokenizers import Tokenizer

import sys

tokenizer = Tokenizer.from_file(sys.argv[1])
ids = tokenizer.encode(sys.argv[2], add_special_tokens=False).ids
print(" ".join(str(token_id) for token_id in ids))

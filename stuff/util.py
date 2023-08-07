OPEN_AI_KEYPATH = "/Users/timlee/Dropbox/keys/openai_key.txt"


def load_openai_key():
    with open(OPEN_AI_KEYPATH, "r") as f:
        return f.read().strip()


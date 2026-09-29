import re
import unicodedata

TRANSLATION = str.maketrans("يك٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "یک01234567890123456789")


def normalize(text):
    """Conservative Persian orthographic normalization, not stemming."""
    text = unicodedata.normalize("NFKC", text).translate(TRANSLATION)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn" and c != "ـ")
    text = text.replace("\u200c", " ").replace("\u200d", " ")
    return " ".join(text.casefold().split())


def tokens(text, mode):
    if mode not in {"raw", "normalized"}:
        raise ValueError("Mode must be raw or normalized")
    return re.findall(r"[^\W_]+", normalize(text) if mode == "normalized" else text.casefold())

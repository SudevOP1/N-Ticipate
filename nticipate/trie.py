"""
Phase 3: prefix trie.

Used when the user has a partial word typed (e.g. "rec") -- lets us find
every vocabulary word beneath that prefix in O(prefix length + result
size) instead of scanning the whole vocabulary per keystroke.
"""

from nticipate.preprocess import START_TOKEN, END_TOKEN, UNK_TOKEN

_DEFAULT_EXCLUDE = {START_TOKEN, END_TOKEN, UNK_TOKEN}


class TrieNode:
    __slots__ = ("children", "is_word")

    def __init__(self):
        self.children: dict[str, "TrieNode"] = {}
        self.is_word: bool = False


class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        node = self.root
        for ch in word:
            if ch not in node.children:
                node.children[ch] = TrieNode()
            node = node.children[ch]
        node.is_word = True

    def __contains__(self, word: str) -> bool:
        node = self.root
        for ch in word:
            node = node.children.get(ch)
            if node is None:
                return False
        return node.is_word

    def words_with_prefix(self, prefix: str, limit: int | None = None) -> list[str]:
        """Return vocabulary words starting with prefix, up to `limit`.
        Results are alphabetically ordered -- ranking by likelihood is
        the predictor's job (Phase 3's Predictor), not the trie's.
        """
        node = self.root
        for ch in prefix:
            node = node.children.get(ch)
            if node is None:
                return []

        results: list[str] = []
        # iterative DFS (avoids recursion-depth concerns on pathological
        # input); push children in reverse-sorted order so popping the
        # stack (LIFO) still yields ascending alphabetical order
        stack: list[tuple[TrieNode, str]] = [(node, prefix)]
        while stack:
            if limit is not None and len(results) >= limit:
                break
            current, path = stack.pop()
            if current.is_word:
                results.append(path)
                if limit is not None and len(results) >= limit:
                    break
            for ch in sorted(current.children, reverse=True):
                stack.append((current.children[ch], path + ch))
        return results

    @classmethod
    def from_vocab(cls, vocab: set[str], exclude: set[str] | None = None) -> "Trie":
        """Build a trie from a vocabulary set. By default, excludes the
        special tokens (<s>, </s>, <UNK>) -- those should never show up
        as a typed-word completion.
        """
        exclude = _DEFAULT_EXCLUDE if exclude is None else exclude
        t = cls()
        for word in vocab:
            if word in exclude:
                continue
            t.insert(word)
        return t

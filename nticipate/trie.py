"""Phase 3 — prefix trie for word completion.

The n-gram model answers "what word comes next?". Word completion asks a
different question: "which words start with `rec`?". Scanning the vocabulary
for every keystroke is O(V) per character typed; a trie makes it O(len(prefix))
plus the size of the matching subtree, which is what keeps completion inside
the keystroke budget.

Counts are stored on the word-final nodes so the trie can rank its own
candidates by frequency when the n-gram context has nothing useful to say
(the first word of a sentence, or a context the model has never seen).
"""

from __future__ import annotations

from collections import Counter
from typing import Iterable, Iterator


class TrieNode:
    """One character of a prefix. ``count`` is non-zero only on word endings."""

    __slots__ = ("children", "count")

    def __init__(self) -> None:
        self.children: dict[str, TrieNode] = {}
        self.count: int = 0

    @property
    def is_word(self) -> bool:
        return self.count > 0


class Trie:
    """A counted prefix trie."""

    def __init__(self, words: Iterable[str] | None = None) -> None:
        self.root = TrieNode()
        self._size = 0
        if words:
            for word in words:
                self.insert(word)

    # ---------------------------------------------------------------- build

    def insert(self, word: str, count: int = 1) -> None:
        """Add ``count`` occurrences of ``word``. Repeated inserts accumulate."""
        if not word:
            return
        node = self.root
        for char in word:
            node = node.children.setdefault(char, TrieNode())
        if node.count == 0:
            self._size += 1
        node.count += count

    @classmethod
    def from_counts(cls, counts: Counter | dict[str, int]) -> "Trie":
        trie = cls()
        for word, count in counts.items():
            trie.insert(word, count)
        return trie

    # --------------------------------------------------------------- lookup

    def _node_for(self, prefix: str) -> TrieNode | None:
        node = self.root
        for char in prefix:
            node = node.children.get(char)
            if node is None:
                return None
        return node

    def __contains__(self, word: str) -> bool:
        node = self._node_for(word)
        return node is not None and node.is_word

    def __len__(self) -> int:
        return self._size

    def count_of(self, word: str) -> int:
        node = self._node_for(word)
        return node.count if node else 0

    def has_prefix(self, prefix: str) -> bool:
        return self._node_for(prefix) is not None

    # ----------------------------------------------------------- completion

    def _walk(self, node: TrieNode, prefix: str) -> Iterator[tuple[str, int]]:
        if node.count:
            yield prefix, node.count
        for char, child in node.children.items():
            yield from self._walk(child, prefix + char)

    def words_with_prefix(self, prefix: str) -> Iterator[tuple[str, int]]:
        """Every (word, count) under ``prefix``, unordered."""
        node = self._node_for(prefix)
        if node is None:
            return iter(())
        return self._walk(node, prefix)

    def complete(
        self,
        prefix: str,
        k: int = 10,
        exclude: Iterable[str] = (),
    ) -> list[tuple[str, int]]:
        """Top-``k`` completions of ``prefix``, most frequent first.

        The prefix itself is a valid completion when it is a word in its own
        right — typing ``the`` should still offer ``the``, because the user may
        be finished and the alternative is offering only ``there`` and
        ``these``.
        """
        blocked = set(exclude)
        matches = [
            (word, count)
            for word, count in self.words_with_prefix(prefix)
            if word not in blocked
        ]
        matches.sort(key=lambda wc: (-wc[1], wc[0]))
        return matches[:k]

    # ------------------------------------------------------------ reporting

    def node_count(self) -> int:
        """Total nodes — the trie's memory cost, reported in the notebook."""
        stack = [self.root]
        total = 0
        while stack:
            node = stack.pop()
            total += 1
            stack.extend(node.children.values())
        return total

    def __repr__(self) -> str:
        return f"Trie(words={len(self)}, nodes={self.node_count()})"

from .models import EmbeddingModel
from .embeddings_migrations import embeddings_migrations
from dataclasses import dataclass
import hashlib
import math
from itertools import islice
import json
import re
from sqlite_utils import Database
from sqlite_utils.db import Table
import time
from typing import cast, Any, Dict, Iterable, List, Optional, Tuple, Union


@dataclass
class Entry:
    id: str
    score: Optional[float]
    content: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


def _tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())


class BM25:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_freqs: Dict[str, int] = {}
        self.doc_lengths: Dict[str, int] = {}
        self.avgdl: float = 0.0
        self.total_docs: int = 0
        self.idf: Dict[str, float] = {}
        self.doc_token_counts: Dict[str, Dict[str, int]] = {}
        self._documents: Dict[str, str] = {}

    def fit(self, documents: Dict[str, str]) -> None:
        self._documents = documents
        self.total_docs = len(documents)
        total_length = 0
        for doc_id, text in documents.items():
            tokens = _tokenize(text)
            self.doc_lengths[doc_id] = len(tokens)
            total_length += len(tokens)
            unique_tokens = set(tokens)
            token_counts: Dict[str, int] = {}
            for token in tokens:
                token_counts[token] = token_counts.get(token, 0) + 1
            self.doc_token_counts[doc_id] = token_counts
            for token in unique_tokens:
                self.doc_freqs[token] = self.doc_freqs.get(token, 0) + 1
        if self.total_docs > 0:
            self.avgdl = total_length / self.total_docs
        for token, df in self.doc_freqs.items():
            self.idf[token] = math.log(
                1 + (self.total_docs - df + 0.5) / (df + 0.5)
            )

    def score(self, query: str, doc_id: str) -> float:
        if doc_id not in self.doc_lengths:
            return 0.0
        query_tokens = _tokenize(query)
        doc_len = self.doc_lengths[doc_id]
        if doc_len == 0 or self.avgdl == 0:
            return 0.0
        score = 0.0
        token_counts = self.doc_token_counts.get(doc_id, {})
        for token in query_tokens:
            if token not in self.idf:
                continue
            tf = token_counts.get(token, 0)
            if tf == 0:
                continue
            idf = self.idf[token]
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * (doc_len / self.avgdl))
            score += idf * (numerator / denominator)
        return score

    def scores(self, query: str, doc_ids: List[str]) -> Dict[str, float]:
        return {doc_id: self.score(query, doc_id) for doc_id in doc_ids}


class Collection:
    class DoesNotExist(Exception):
        pass

    def __init__(
        self,
        name: str,
        db: Optional[Database] = None,
        *,
        model: Optional[EmbeddingModel] = None,
        model_id: Optional[str] = None,
        create: bool = True,
    ) -> None:
        """
        A collection of embeddings

        Returns the collection with the given name, creating it if it does not exist.

        If you set create=False a Collection.DoesNotExist exception will be raised if the
        collection does not already exist.

        Args:
            db (sqlite_utils.Database): Database to store the collection in
            name (str): Name of the collection
            model (llm.models.EmbeddingModel, optional): Embedding model to use
            model_id (str, optional): Alternatively, ID of the embedding model to use
            create (bool, optional): Whether to create the collection if it does not exist
        """
        import llm

        self.db = db or Database(memory=True)
        self.name = name
        self._model = model

        embeddings_migrations.apply(self.db)

        rows = list(self.db["collections"].rows_where("name = ?", [self.name]))
        if rows:
            row = rows[0]
            self.id = row["id"]
            self.model_id = row["model"]
        else:
            if create:
                # Collection does not exist, so model or model_id is required
                if not model and not model_id:
                    raise ValueError(
                        "Either model= or model_id= must be provided when creating a new collection"
                    )
                # Create it
                if model_id:
                    # Resolve alias
                    model = llm.get_embedding_model(model_id)
                    self._model = model
                model_id = cast(EmbeddingModel, model).model_id
                self.model_id = model_id
                self.id = (
                    cast(Table, self.db["collections"])
                    .insert(
                        {
                            "name": self.name,
                            "model": model_id,
                        }
                    )
                    .last_pk
                )
            else:
                raise self.DoesNotExist(f"Collection '{name}' does not exist")

    def model(self) -> EmbeddingModel:
        "Return the embedding model used by this collection"
        import llm

        if self._model is None:
            self._model = llm.get_embedding_model(self.model_id)

        return cast(EmbeddingModel, self._model)

    def count(self) -> int:
        """
        Count the number of items in the collection.

        Returns:
            int: Number of items in the collection
        """
        return next(
            self.db.query(
                """
            select count(*) as c from embeddings where collection_id = (
                select id from collections where name = ?
            )
            """,
                (self.name,),
            )
        )["c"]

    def embed(
        self,
        id: str,
        value: Union[str, bytes],
        metadata: Optional[Dict[str, Any]] = None,
        store: bool = False,
    ) -> None:
        """
        Embed value and store it in the collection with a given ID.

        Args:
            id (str): ID for the value
            value (str or bytes): value to be embedded
            metadata (dict, optional): Metadata to be stored
            store (bool, optional): Whether to store the value in the content or content_blob column
        """
        from llm import encode

        content_hash = self.content_hash(value)
        if self.db["embeddings"].count_where(
            "content_hash = ? and collection_id = ?", [content_hash, self.id]
        ):
            return
        embedding = self.model().embed(value)
        cast(Table, self.db["embeddings"]).insert(
            {
                "collection_id": self.id,
                "id": id,
                "embedding": encode(embedding),
                "content": value if (store and isinstance(value, str)) else None,
                "content_blob": value if (store and isinstance(value, bytes)) else None,
                "content_hash": content_hash,
                "metadata": json.dumps(metadata) if metadata else None,
                "updated": int(time.time()),
            },
            replace=True,
        )

    def embed_multi(
        self,
        entries: Iterable[Tuple[str, Union[str, bytes]]],
        store: bool = False,
        batch_size: int = 100,
    ) -> None:
        """
        Embed multiple texts and store them in the collection with given IDs.

        Args:
            entries (iterable): Iterable of (id: str, text: str) tuples
            store (bool, optional): Whether to store the text in the content column
            batch_size (int, optional): custom maximum batch size to use
        """
        self.embed_multi_with_metadata(
            ((id, value, None) for id, value in entries),
            store=store,
            batch_size=batch_size,
        )

    def embed_multi_with_metadata(
        self,
        entries: Iterable[Tuple[str, Union[str, bytes], Optional[Dict[str, Any]]]],
        store: bool = False,
        batch_size: int = 100,
    ) -> None:
        """
        Embed multiple values along with metadata and store them in the collection with given IDs.

        Args:
            entries (iterable): Iterable of (id: str, value: str or bytes, metadata: None or dict)
            store (bool, optional): Whether to store the value in the content or content_blob column
            batch_size (int, optional): custom maximum batch size to use
        """
        import llm

        batch_size = min(batch_size, (self.model().batch_size or batch_size))
        iterator = iter(entries)
        collection_id = self.id
        while True:
            batch = list(islice(iterator, batch_size))
            if not batch:
                break
            # Calculate hashes first
            items_and_hashes = [(item, self.content_hash(item[1])) for item in batch]
            # Any of those hashes already exist?
            existing_ids = [
                row["id"]
                for row in self.db.query(
                    """
                    select id from embeddings
                    where collection_id = ? and content_hash in ({})
                    """.format(",".join("?" for _ in items_and_hashes)),
                    [collection_id]
                    + [item_and_hash[1] for item_and_hash in items_and_hashes],
                )
            ]
            filtered_batch = [item for item in batch if item[0] not in existing_ids]
            embeddings = list(
                self.model().embed_multi(item[1] for item in filtered_batch)
            )
            with self.db.conn:
                cast(Table, self.db["embeddings"]).insert_all(
                    (
                        {
                            "collection_id": collection_id,
                            "id": id,
                            "embedding": llm.encode(embedding),
                            "content": (
                                value if (store and isinstance(value, str)) else None
                            ),
                            "content_blob": (
                                value if (store and isinstance(value, bytes)) else None
                            ),
                            "content_hash": self.content_hash(value),
                            "metadata": json.dumps(metadata) if metadata else None,
                            "updated": int(time.time()),
                        }
                        for (embedding, (id, value, metadata)) in zip(
                            embeddings, filtered_batch
                        )
                    ),
                    replace=True,
                )

    def similar_by_vector(
        self,
        vector: List[float],
        number: int = 10,
        skip_id: Optional[str] = None,
        prefix: Optional[str] = None,
        rerank: Optional[str] = None,
        rerank_model: Optional[str] = None,
        fetch_k: Optional[int] = None,
        query_text: Optional[str] = None,
    ) -> List[Entry]:
        """
        Find similar items in the collection by a given vector.

        Args:
            vector (list): Vector to search by
            number (int, optional): Number of similar items to return
            skip_id (str, optional): An ID to exclude from the results
            prefix: (str, optional): Filter results to IDs with this prefix
            rerank: (str, optional): Rerank method: 'bm25' or 'embedding'
            rerank_model: (str, optional): Embedding model for rerank (if rerank='embedding')
            fetch_k: (int, optional): Number of candidates to fetch for reranking (default: number * 3)
            query_text: (str, optional): Original query text for BM25 reranking

        Returns:
            list: List of Entry objects
        """
        import llm

        fetch_count = fetch_k or (number * 3) if rerank else number

        def distance_score(other_encoded):
            other_vector = llm.decode(other_encoded)
            return llm.cosine_similarity(other_vector, vector)

        self.db.register_function(distance_score, replace=True)

        where_bits = ["collection_id = ?"]
        where_args = [str(self.id)]

        if prefix:
            where_bits.append("id LIKE ? || '%'")
            where_args.append(prefix)

        if skip_id:
            where_bits.append("id != ?")
            where_args.append(skip_id)

        results = [
            Entry(
                id=row["id"],
                score=row["score"],
                content=row["content"],
                metadata=json.loads(row["metadata"]) if row["metadata"] else None,
            )
            for row in self.db.query(
                """
            select id, content, metadata, distance_score(embedding) as score
            from embeddings
            where {where}
            order by score desc limit {number}
        """.format(
                    where=" and ".join(where_bits),
                    number=fetch_count,
                ),
                where_args,
            )
        ]

        if not rerank:
            return results[:number]

        return self._rerank_results(
            results,
            rerank,
            rerank_model,
            query_text,
            number,
        )

    def _rerank_results(
        self,
        candidates: List[Entry],
        rerank: str,
        rerank_model: Optional[str],
        query_text: Optional[str],
        number: int,
    ) -> List[Entry]:
        import llm

        if not candidates:
            return []

        rerank = rerank.lower()

        if rerank == "bm25":
            if query_text is None:
                raise ValueError("query_text is required for BM25 reranking")
            documents: Dict[str, str] = {}
            for entry in candidates:
                if entry.content:
                    documents[entry.id] = entry.content
            if not documents:
                raise ValueError(
                    "BM25 reranking requires stored content. Use --store when embedding."
                )
            bm25 = BM25()
            bm25.fit(documents)
            scores = bm25.scores(query_text, list(documents.keys()))
            reranked = [
                (entry, scores.get(entry.id, 0.0)) for entry in candidates
            ]
            reranked.sort(key=lambda x: x[1], reverse=True)
            return [
                Entry(
                    id=entry.id,
                    score=score,
                    content=entry.content,
                    metadata=entry.metadata,
                )
                for entry, score in reranked[:number]
            ]
        elif rerank == "embedding":
            if query_text is None:
                raise ValueError("query_text is required for embedding reranking")
            model_id = rerank_model or self.model_id
            model = llm.get_embedding_model(model_id)
            query_vector = model.embed(query_text)
            reranked = []
            for entry in candidates:
                if entry.content is None:
                    reranked.append((entry, entry.score or 0.0))
                    continue
                entry_vector = model.embed(entry.content)
                score = llm.cosine_similarity(query_vector, entry_vector)
                reranked.append((entry, score))
            reranked.sort(key=lambda x: x[1], reverse=True)
            return [
                Entry(
                    id=entry.id,
                    score=score,
                    content=entry.content,
                    metadata=entry.metadata,
                )
                for entry, score in reranked[:number]
            ]
        else:
            raise ValueError(f"Unknown rerank method: {rerank}")

    def similar_by_id(
        self,
        id: str,
        number: int = 10,
        prefix: Optional[str] = None,
        rerank: Optional[str] = None,
        rerank_model: Optional[str] = None,
        fetch_k: Optional[int] = None,
    ) -> List[Entry]:
        """
        Find similar items in the collection by a given ID.

        Args:
            id (str): ID to search by
            number (int, optional): Number of similar items to return
            prefix: (str, optional): Filter results to IDs with this prefix
            rerank: (str, optional): Rerank method: 'bm25' or 'embedding'
            rerank_model: (str, optional): Embedding model for rerank
            fetch_k: (int, optional): Number of candidates to fetch for reranking

        Returns:
            list: List of Entry objects
        """
        import llm

        matches = list(
            self.db["embeddings"].rows_where(
                "collection_id = ? and id = ?", (self.id, id)
            )
        )
        if not matches:
            raise self.DoesNotExist("ID not found")
        row = matches[0]
        embedding = row["embedding"]
        comparison_vector = llm.decode(embedding)
        query_text = row["content"]
        return self.similar_by_vector(
            comparison_vector,
            number,
            skip_id=id,
            prefix=prefix,
            rerank=rerank,
            rerank_model=rerank_model,
            fetch_k=fetch_k,
            query_text=query_text,
        )

    def similar(
        self,
        value: Union[str, bytes],
        number: int = 10,
        prefix: Optional[str] = None,
        rerank: Optional[str] = None,
        rerank_model: Optional[str] = None,
        fetch_k: Optional[int] = None,
    ) -> List[Entry]:
        """
        Find similar items in the collection by a given value.

        Args:
            value (str or bytes): value to search by
            number (int, optional): Number of similar items to return
            prefix: (str, optional): Filter results to IDs with this prefix
            rerank: (str, optional): Rerank method: 'bm25' or 'embedding'
            rerank_model: (str, optional): Embedding model for rerank
            fetch_k: (int, optional): Number of candidates to fetch for reranking

        Returns:
            list: List of Entry objects
        """
        comparison_vector = self.model().embed(value)
        query_text = value if isinstance(value, str) else None
        return self.similar_by_vector(
            comparison_vector,
            number,
            prefix=prefix,
            rerank=rerank,
            rerank_model=rerank_model,
            fetch_k=fetch_k,
            query_text=query_text,
        )

    @classmethod
    def exists(cls, db: Database, name: str) -> bool:
        """
        Does this collection exist in the database?

        Args:
            name (str): Name of the collection
        """
        rows = list(db["collections"].rows_where("name = ?", [name]))
        return bool(rows)

    def delete(self):
        """
        Delete the collection and its embeddings from the database
        """
        with self.db.conn:
            self.db.execute("delete from embeddings where collection_id = ?", [self.id])
            self.db.execute("delete from collections where id = ?", [self.id])

    @staticmethod
    def content_hash(input: Union[str, bytes]) -> bytes:
        "Hash content for deduplication. Override to change hashing behavior."
        if isinstance(input, str):
            input = input.encode("utf8")
        return hashlib.md5(input).digest()

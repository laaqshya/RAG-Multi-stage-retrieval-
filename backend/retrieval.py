import re
from collections.abc import Callable
from dataclasses import dataclass


_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
_QUERY_EXPANSIONS = {
    "buyer": {"customer", "order"},
    "buyers": {"customer", "order"},
    "client": {"customer", "order"},
    "clients": {"customer", "order"},
    "cost": {"price", "product"},
    "costs": {"price", "product"},
    "purchase": {"order", "item", "product"},
    "purchases": {"order", "item", "product"},
    "revenue": {"price", "quantity", "order", "item", "product"},
    "sales": {"price", "quantity", "order", "item", "product"},
    "spend": {"price", "quantity", "order", "item", "product"},
    "spending": {"price", "quantity", "order", "item", "product"},
    "total": {"price", "quantity", "order", "item"},
    "units": {"quantity", "order", "item"},
}


@dataclass(frozen=True)
class SchemaCandidate:
    table: str
    score: int
    document: str


@dataclass(frozen=True)
class RetrievalResult:
    query_terms: tuple[str, ...]
    candidate_tables: tuple[str, ...]
    selected_tables: tuple[str, ...]
    context: str


class NoSchemaMatchError(ValueError):
    pass


def _tokens(value: str) -> set[str]:
    tokens = _TOKEN_PATTERN.findall(value.lower().replace("_", " "))
    normalized = set()
    for token in tokens:
        if len(token) > 4 and token.endswith("ies"):
            token = token[:-3] + "y"
        elif len(token) > 3 and token.endswith("s") and not token.endswith(
            ("ss", "us", "is")
        ):
            token = token[:-1]
        normalized.add(token)
    return normalized


def understand_query(question: str) -> tuple[str, ...]:
    """Normalize the question and expand common text-to-SQL concepts."""
    original_terms = _tokens(question)
    expanded_terms = set(original_terms)

    for term in original_terms:
        expanded_terms.update(_QUERY_EXPANSIONS.get(term, ()))

    return tuple(sorted(expanded_terms))


def _table_document(
    table: str, columns: list[dict[str, str]]
) -> str:
    column_descriptions = [
        f"{column['column']} ({column['type']})" for column in columns
    ]
    return f"TABLE: {table}\nCOLUMNS: {', '.join(column_descriptions)}"


def _initial_retrieval(
    query_terms: set[str],
    schema: dict[str, list[dict[str, str]]],
) -> list[SchemaCandidate]:
    candidates = []

    for table, columns in schema.items():
        document = _table_document(table, columns)
        table_terms = _tokens(table)
        column_terms = set().union(
            *(_tokens(column["column"]) for column in columns)
        ) if columns else set()
        type_terms = set().union(
            *(_tokens(column["type"]) for column in columns)
        ) if columns else set()

        score = (
            4 * len(query_terms & table_terms)
            + 3 * len(query_terms & column_terms)
            + len(query_terms & type_terms)
        )
        if score:
            candidates.append(SchemaCandidate(table, score, document))

    return sorted(
        candidates,
        key=lambda candidate: (-candidate.score, candidate.table),
    )


def _format_context(
    schema: dict[str, list[dict[str, str]]],
    selected_tables: list[str],
    relationships: list[dict[str, str]],
) -> str:
    sections = []
    selected = set(selected_tables)

    for table in selected_tables:
        sections.append(_table_document(table, schema[table]))

    relevant_relationships = [
        relationship
        for relationship in relationships
        if relationship["table"] in selected
        and relationship["referenced_table"] in selected
    ]
    if relevant_relationships:
        links = [
            f"{link['table']}.{link['column']} -> "
            f"{link['referenced_table']}.{link['referenced_column']}"
            for link in relevant_relationships
        ]
        sections.append("FOREIGN KEYS:\n" + "\n".join(links))

    return "\n\n".join(sections)


def retrieve_schema_context(
    question: str,
    schema: dict[str, list[dict[str, str]]],
    relationships: list[dict[str, str]],
    rerank: Callable[[str, list[dict[str, str]]], list[int]],
    candidate_limit: int = 8,
    context_table_limit: int = 6,
) -> RetrievalResult:
    """Retrieve, filter, rerank, and compress live schema for SQL generation."""
    query_terms = understand_query(question)
    candidates = _initial_retrieval(set(query_terms), schema)[:candidate_limit]
    if not candidates:
        raise NoSchemaMatchError("No database schema matched the question.")

    candidate_documents = [
        {"table": candidate.table, "document": candidate.document}
        for candidate in candidates
    ]
    ranking = rerank(question, candidate_documents)
    if (
        not isinstance(ranking, list)
        or any(type(index) is not int for index in ranking)
        or len(ranking) != len(candidates)
        or set(ranking) != set(range(len(candidates)))
    ):
        raise ValueError("The schema reranker returned an invalid candidate ranking.")

    selected_tables = [
        candidates[index].table
        for index in ranking[:context_table_limit]
    ]
    return RetrievalResult(
        query_terms=query_terms,
        candidate_tables=tuple(candidate.table for candidate in candidates),
        selected_tables=tuple(selected_tables),
        context=_format_context(schema, selected_tables, relationships),
    )

import numpy as np
import config
import logging

log = logging.getLogger(__name__)


def match_face(
    embedding: np.ndarray,
    known_embeddings: np.ndarray,
    names: list[str]
) -> tuple[str | None, float]:
    """
    Match a face embedding against all known embeddings.
    
    Key fix: When a student has multiple enrolled embeddings (multi-angle),
    we take the BEST score per student, not just the global max. This prevents
    a weak angle embedding from a wrong student from winning over a strong
    match from the correct student.
    """

    # Validate inputs
    if embedding is None or embedding.size == 0:
        raise ValueError("match_face() received an empty or None embedding.")

    if known_embeddings is None or len(known_embeddings) == 0:
        return None, 0.0

    if len(known_embeddings) != len(names):
        raise ValueError(
            f"known_embeddings ({len(known_embeddings)}) and "
            f"names ({len(names)}) must have the same length."
        )

    # L2-normalize for cosine similarity
    embedding = embedding / (np.linalg.norm(embedding) + 1e-10)
    known_norm = known_embeddings / (
        np.linalg.norm(known_embeddings, axis=1, keepdims=True) + 1e-10
    )
    scores = known_norm @ embedding  # shape: (N,)
    scores = np.clip(scores, -1.0, 1.0)

    # ── Per-student best score aggregation ──────────────────────────────
    # A student with 3 angles has 3 rows. We want the BEST score per student.
    student_best = {}  # name -> best_score
    for i, name in enumerate(names):
        sc = float(scores[i])
        if name not in student_best or sc > student_best[name]:
            student_best[name] = sc

    if not student_best:
        return None, 0.0

    # Find the student with the highest best-score
    best_name = max(student_best, key=student_best.get)
    best_score = student_best[best_name]

    log.debug(f"  Top matches: {sorted(student_best.items(), key=lambda x: -x[1])[:5]}")

    if best_score >= config.THRESHOLD:
        return best_name, best_score

    return None, best_score
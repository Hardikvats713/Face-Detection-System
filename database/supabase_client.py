from supabase import create_client
import numpy as np
import config

# ✅ FIX 1: Wrap client creation in try/except — invalid URL/key raises immediately
try:
    supabase = create_client(config.SUPABASE_URL, config.SUPABASE_KEY)
except Exception as e:
    raise RuntimeError(f"Failed to initialize Supabase client: {e}")


def load_students():
    # ✅ FIX 2: Handle API errors — .execute() can raise or return empty data
    try:
        res = supabase.table("students").select("*").execute()
    except Exception as e:
        raise RuntimeError(f"Failed to fetch students from Supabase: {e}")

    # ✅ FIX 3: Guard against None or missing data key
    if not res.data:
        return np.array([]), []

    embeddings = []
    names = []

    for row in res.data:
        # ✅ FIX 4: Skip rows with missing/null embedding or name fields
        if row.get("embedding") is None or row.get("name") is None:
            print(f"Warning: Skipping row with missing fields: {row}")
            continue

        # ✅ FIX 5: Ensure embedding is a flat numeric list before converting
        try:
            embedding = np.array(row["embedding"], dtype=np.float32)
        except (ValueError, TypeError) as e:
            print(f"Warning: Skipping invalid embedding for '{row.get('name')}': {e}")
            continue

        embeddings.append(embedding)
        names.append(row["name"])

    # ✅ FIX 6: Handle case where all rows were skipped — np.array([[]]) would break shape
    if not embeddings:
        return np.array([]), names

    # ✅ FIX 7: Stack instead of wrapping — avoids ragged array error if lengths differ
    return np.stack(embeddings), names
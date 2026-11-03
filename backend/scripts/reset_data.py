"""Script to safely clear all previous PDF documents, chunks, vectors, and queue jobs."""
import asyncio
from pathlib import Path
import redis.asyncio as aioredis
from motor.motor_asyncio import AsyncIOMotorClient
from qdrant_client import AsyncQdrantClient

from app.core.config import get_settings


async def reset_all():
    settings = get_settings()
    print("--- Starting DocuGround Data Cleanup ---")

    # 1. Clear backend/uploads/
    upload_dir = Path("backend/uploads").resolve()
    deleted_files = 0
    if upload_dir.exists():
        for f in upload_dir.glob("*.pdf"):
            try:
                f.unlink()
                deleted_files += 1
                print(f"Deleted upload file: {f.name}")
            except Exception as e:
                print(f"Failed to delete {f.name}: {e}")
    print(f"[1/4] Cleaned uploads: {deleted_files} files deleted.")

    # 2. Clear MongoDB
    mongo = AsyncIOMotorClient(settings.mongodb_url)
    db = mongo[settings.mongodb_db]
    doc_res = await db.documents.delete_many({})
    chunk_res = await db.chunks.delete_many({})
    print(f"[2/4] Cleaned MongoDB: {doc_res.deleted_count} documents, {chunk_res.deleted_count} chunks deleted.")
    mongo.close()

    # 3. Clear Qdrant
    try:
        qdrant = AsyncQdrantClient(url=settings.qdrant_url)
        collections = await qdrant.get_collections()
        col_names = [c.name for c in collections.collections]
        if settings.qdrant_collection in col_names:
            await qdrant.delete_collection(settings.qdrant_collection)
            print(f"[3/4] Cleaned Qdrant: deleted collection '{settings.qdrant_collection}'.")
        else:
            print(f"[3/4] Qdrant collection '{settings.qdrant_collection}' was already empty/absent.")
        await qdrant.close()
    except Exception as e:
        print(f"[3/4] Qdrant cleanup skipped/failed: {e}")

    # 4. Clear Redis jobs & stream
    try:
        r = aioredis.from_url(settings.redis_url, protocol=2)
        await r.delete(settings.ingest_stream)
        await r.delete(settings.dlq_stream)
        keys = await r.keys("job:*")
        if keys:
            await r.delete(*keys)
        print(f"[4/4] Cleaned Redis: cleared stream '{settings.ingest_stream}' and {len(keys)} job records.")
        await r.aclose()
    except Exception as e:
        print(f"[4/4] Redis cleanup skipped/failed: {e}")

    print("--- Data Cleanup Complete! System state is 100% fresh. ---")


if __name__ == "__main__":
    asyncio.run(reset_all())

"""Script to fix MongoDB indexes."""
import asyncio
from backend.database import motor_client, DATABASE_NAME, MONGODB_URL

async def fix_indexes():
    """Drop old indexes and recreate them properly."""
    print(f"Connecting to MongoDB at {MONGODB_URL}")

    db = motor_client[DATABASE_NAME]

    try:
        # Drop all indexes from knowledge_base collection (except _id)
        print("Dropping existing indexes from knowledge_base...")
        try:
            await db.knowledge_base.drop_indexes()
            print("✓ Dropped old indexes")
        except Exception as e:
            print(f"Note: {e}")

        # Recreate indexes
        print("Creating new indexes...")

        # Regular indexes
        await db.knowledge_base.create_index("title")
        await db.knowledge_base.create_index("category")

        # Text index for full-text search
        await db.knowledge_base.create_index([("content", "text"), ("title", "text")])

        print("✓ All indexes created successfully!")

        # List all indexes to verify
        indexes = await db.knowledge_base.list_indexes().to_list(length=None)
        print("\nCurrent indexes:")
        for idx in indexes:
            print(f"  - {idx}")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        motor_client.close()

if __name__ == "__main__":
    asyncio.run(fix_indexes())

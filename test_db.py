import asyncio
import asyncpg
import os
from dotenv import load_dotenv

# Force Python to read the .env file freshly
load_dotenv(override=True)

async def test_connection():
    user = os.getenv("DB_USER")
    password = os.getenv("DB_PASSWORD")
    
    # This will reveal if there are hidden spaces or carriage returns
    print(f"Attempting to connect as...")
    print(f"USER:     '{user}' (Length: {len(str(user))})")
    print(f"PASSWORD: '{password}' (Length: {len(str(password))})")
    
    try:
        conn = await asyncpg.connect(
            user=user,
            password=password,
            host='127.0.0.1',
            port=int(os.getenv('DB_PORT', 5433)),
            database='botdb'
        )
        print("\n✅ SUCCESS! The database is open and accepting connections.")
        await conn.close()
    except Exception as e:
        print(f"\n❌ FAILED: {e}")

if __name__ == "__main__":
    asyncio.run(test_connection())
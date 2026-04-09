from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
import certifi
import os

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
DB_NAME   = os.getenv("DB_NAME", "skin_disease_db")

client = AsyncIOMotorClient(
    MONGO_URI,
    tls=True,
    tlsCAFile=certifi.where(),
    serverSelectionTimeoutMS=30000
)
db = client[DB_NAME]

users_collection    = db["users"]
analyses_collection = db["analyses"]


async def create_indexes():
    await users_collection.create_index("email", unique=True)
    await analyses_collection.create_index("user_id")
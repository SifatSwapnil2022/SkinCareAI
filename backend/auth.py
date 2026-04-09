from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
import bcrypt
from jose import JWTError, jwt
from datetime import datetime, timedelta
from pydantic import BaseModel, EmailStr
from database.mongo import users_collection
from dotenv import load_dotenv
from bson import ObjectId
import os

load_dotenv()

SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM  = os.getenv("ALGORITHM", "HS256")
EXPIRE_MIN = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 10080))

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")
router        = APIRouter(prefix="/auth", tags=["Authentication"])


# ─── Schemas ────────────────────────────────────────────────
class SignUpRequest(BaseModel):
    name:     str
    email:    EmailStr
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type:   str
    user_name:    str
    user_email:   str
    user_id:      str


# ─── Helpers ────────────────────────────────────────────────
def hash_password(password: str) -> str:
    pwd_bytes = password.encode("utf-8")[:72]
    return bcrypt.hashpw(pwd_bytes, bcrypt.gensalt()).decode("utf-8")

def verify_password(plain: str, hashed: str) -> bool:
    pwd_bytes = plain.encode("utf-8")[:72]
    return bcrypt.checkpw(pwd_bytes, hashed.encode("utf-8"))

def create_token(data: dict) -> str:
    payload = data.copy()
    payload["exp"] = datetime.utcnow() + timedelta(minutes=EXPIRE_MIN)
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


# ─── Get Current User ────────────────────────────────────────
async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired token",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    user = await users_collection.find_one({"_id": ObjectId(user_id)})
    if user is None:
        raise credentials_exception
    return user


# ─── Routes ─────────────────────────────────────────────────
@router.post("/signup", response_model=TokenResponse)
async def signup(data: SignUpRequest):
    existing = await users_collection.find_one({"email": data.email})
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")

    user_doc = {
        "name":       data.name,
        "email":      data.email,
        "password":   hash_password(data.password),
        "created_at": datetime.utcnow(),
    }
    result  = await users_collection.insert_one(user_doc)
    user_id = str(result.inserted_id)
    token   = create_token({"sub": user_id})

    return TokenResponse(
        access_token=token,
        token_type="bearer",
        user_name=data.name,
        user_email=data.email,
        user_id=user_id
    )


@router.post("/login", response_model=TokenResponse)
async def login(form: OAuth2PasswordRequestForm = Depends()):
    user = await users_collection.find_one({"email": form.username})
    if not user or not verify_password(form.password, user["password"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    user_id = str(user["_id"])
    token   = create_token({"sub": user_id})

    return TokenResponse(
        access_token=token,
        token_type="bearer",
        user_name=user["name"],
        user_email=user["email"],
        user_id=user_id
    )


@router.get("/profile")
async def get_profile(current_user: dict = Depends(get_current_user)):
    return {
        "user_id":    str(current_user["_id"]),
        "name":       current_user["name"],
        "email":      current_user["email"],
        "created_at": current_user["created_at"],
    }
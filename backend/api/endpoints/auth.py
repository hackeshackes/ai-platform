"""Authentication endpoints"""
from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from datetime import datetime, timedelta
from typing import Optional
from pydantic import BaseModel
from jose import jwt, JWTError

router = APIRouter()

# 配置
SECRET_KEY = "your-secret-key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/token")

# Pydantic Models
class Token(BaseModel):
    access_token: str
    token_type: str
    expires_in: int

class TokenData(BaseModel):
    username: Optional[str] = None

class UserCreate(BaseModel):
    username: str
    email: str
    password: str

class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    is_active: bool
    created_at: datetime

class UserInDB(UserResponse):
    hashed_password: str

# 模拟数据库 (生产环境请使用SQLAlchemy + bcrypt)
# 简化验证：使用明文比对，生产环境请替换为bcrypt
DEMO_USER = {
    "id": 1,
    "username": "admin",
    "email": "admin@ai-platform.com",
    "hashed_password": "admin123",  # 生产环境: 使用bcrypt.hash()
    "is_active": True,
    "is_superuser": True,
    "created_at": datetime.utcnow()
}

fake_users_db = {
    "admin": DEMO_USER
}

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """验证密码 - 简化版本，生产环境请使用bcrypt"""
    # 开发环境：明文比对
    # 生产环境：使用 bcrypt.compare()
    return plain_password == hashed_password

def get_user(db, username: str) -> Optional[dict]:
    return db.get(username)

def authenticate_user(db, username: str, password: str) -> Optional[dict]:
    user = get_user(db, username)
    if not user:
        return None
    if not verify_password(password, user["hashed_password"]):
        return None
    return user

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

@router.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(fake_users_db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["username"]},
        expires_delta=access_token_expires
    )
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "expires_in": ACCESS_TOKEN_EXPIRE_MINUTES * 60
    }

@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(user_data: UserCreate):
    if user_data.username in fake_users_db:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered"
        )
    
    user_id = len(fake_users_db) + 1
    
    new_user = {
        "id": user_id,
        "username": user_data.username,
        "email": user_data.email,
        "hashed_password": user_data.password,  # 生产环境: bcrypt.hash()
        "is_active": True,
        "is_superuser": False,
        "created_at": datetime.utcnow()
    }
    
    fake_users_db[user_data.username] = new_user
    
    return new_user

@router.get("/me", response_model=UserResponse)
async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    user = get_user(fake_users_db, username)
    if user is None:
        raise credentials_exception
    
    return user

@router.post("/logout")
async def logout():
    return {"message": "Successfully logged out"}

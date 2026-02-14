import os
from dotenv import load_dotenv
load_dotenv()
print("HF_TOKEN =", os.getenv("HF_TOKEN"))
print("MODEL_ID =", os.getenv("MODEL_ID"))
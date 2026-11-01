"""Text embeddings for hybrid retrieval. AWS Bedrock Titan v2 when configured, otherwise Azure OpenAI.

Returns L2-normalised float32 rows, so a dot product is the cosine similarity.
"""
import json
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import numpy as np

from . import config


def model_id():
    """Identifier stored next to the vectors so queries are always embedded with the same model."""
    if config.bedrock_embed_ready():
        return f"bedrock:{config.BEDROCK_EMBED_MODEL}"
    if config.azure_ready(config.EMBED_DEPLOYMENT):
        return f"azure:{config.EMBED_DEPLOYMENT}"
    if config.OPENAI_API_KEY:
        return f"openai:{config.OPENAI_EMBED_MODEL}"
    return None


@lru_cache(maxsize=1)
def _bedrock():
    import boto3
    from botocore.config import Config
    return boto3.client("bedrock-runtime", region_name=config.AWS_REGION,
                        config=Config(retries={"mode": "adaptive", "max_attempts": 8}, max_pool_connections=16))


def _titan(text):
    body = json.dumps({"inputText": text[:40000], "dimensions": 1024, "normalize": True})
    r = _bedrock().invoke_model(modelId=config.BEDROCK_EMBED_MODEL, body=body)
    return json.loads(r["body"].read())["embedding"]


def embed(texts):
    if not texts:
        return np.zeros((0, 1024), dtype="float32")
    mid = model_id()
    if mid is None:
        raise RuntimeError("No embedding provider configured (BEDROCK_EMBED_MODEL + AWS keys, or AZURE_OPENAI_EMBED_DEPLOYMENT)")
    if mid.startswith("bedrock:"):  # Titan embeds one text per call -> parallelise
        with ThreadPoolExecutor(max_workers=8) as ex:
            vecs = list(ex.map(_titan, texts))
    elif mid.startswith("azure:"):
        from openai import AzureOpenAI
        client = AzureOpenAI(azure_endpoint=config.AZURE_ENDPOINT, api_key=config.AZURE_API_KEY, api_version=config.AZURE_API_VERSION)
        vecs = []
        for s in range(0, len(texts), 64):
            vecs += [d.embedding for d in client.embeddings.create(model=config.EMBED_DEPLOYMENT, input=texts[s:s + 64]).data]
    else:
        from openai import OpenAI
        client = OpenAI(api_key=config.OPENAI_API_KEY)
        vecs = []
        for s in range(0, len(texts), 64):
            vecs += [d.embedding for d in client.embeddings.create(model=config.OPENAI_EMBED_MODEL, input=texts[s:s + 64]).data]
    v = np.array(vecs, dtype="float32")
    return v / np.linalg.norm(v, axis=1, keepdims=True)

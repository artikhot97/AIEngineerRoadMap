import httpx
import asyncio



async def http_get(url: str) -> dict:
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        return response.json()
    

async def http_post(url: str, headers: dict, data: dict) -> dict:
    async with httpx.AsyncClient() as client:
        response = await client.post(url, headers=headers, json=data)
        return response.json()

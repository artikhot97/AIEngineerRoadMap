import asyncio

import requests

# A few handy JSON types
JSON = int | str | float | bool | None | dict[str, "JSON"] | list["JSON"]
JSONObject = dict[str, JSON]
JSONList = list[JSON]


def http_get_sync(url: str) -> JSONObject:
    response = requests.get(url)
    return response.json()


async def http_get(url: str) -> JSONObject:
    return await asyncio.to_thread(http_get_sync, url)

def http_post_sync(url: str, headers: JSONObject, data: JSONObject) -> JSONObject:
    response = requests.post(url, headers=headers, json=data)
    return response.json()

async def http_post(url: str, headers: JSONObject, data: JSONObject) -> JSONObject:
    return await asyncio.to_thread(http_post_sync, url, headers, data)
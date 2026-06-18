
"""
Real Use Cases
API calls
Database queries
Kafka consumers
Redis operations
"""
import asyncio

async def slow_api():
    await asyncio.sleep(10)
    return "data"


async def main():
    try:
        result = await asyncio.wait_for(
            slow_api(),
            timeout=3
        )
        print(result)

    except asyncio.TimeoutError:
        print("Request timed out")


asyncio.run(main())
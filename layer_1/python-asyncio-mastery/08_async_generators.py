import asyncio


async def number_stream():
    for i in range(10):
        await asyncio.sleep(1)
        yield i


async def main():
    async for number in number_stream():
        print(number)


asyncio.run(main())
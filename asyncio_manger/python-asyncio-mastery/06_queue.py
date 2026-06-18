import asyncio


queue = asyncio.Queue()


async def producer():
    for i in range(5):
        await queue.put(i)
        print(f"Produced {i}")

    await queue.put(None)


async def consumer():
    while True:
        item = await queue.get()

        if item is None:
            break

        print(f"Consumed {item}")
        await asyncio.sleep(1)


async def main():
    await asyncio.gather(
        producer(),
        consumer()
    )


asyncio.run(main())
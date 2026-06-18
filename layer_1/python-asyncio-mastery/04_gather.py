import asyncio



async def worker(name, delay):
    await asyncio.sleep(delay)
    return f"{name} completed"


async def main():
    results = await asyncio.gather(
        worker("A", 2),
        worker("B", 1),
        worker("C", 3),
    )

    print(results)


asyncio.run(main())


#create_task = schedule work
#gather = wait for many tasks together
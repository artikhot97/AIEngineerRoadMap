import asyncio


class DatabaseConnection:

    async def __aenter__(self):
        print("Opening connection")
        return self

    async def __aexit__(
        self,
        exc_type,
        exc,
        tb
    ):
        print("Closing connection")


async def main():
    async with DatabaseConnection():
        print("Running query")
        await asyncio.sleep(2)


asyncio.run(main())
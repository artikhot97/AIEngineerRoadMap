import asyncio


async def fetch_user():
    print("Fetching user...")
    await asyncio.sleep(2)
    print("User fetched")


async def fetch_orders():
    print("Fetching orders...")
    await asyncio.sleep(3)
    print("Orders fetched")


async def main():
    user_task = asyncio.create_task(fetch_user())
    order_task = asyncio.create_task(fetch_orders())

    await user_task
    await order_task


asyncio.run(main())
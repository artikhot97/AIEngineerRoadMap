import asyncio
from req_http import http_get, JSONObject
from random import randint
from typing import AsyncIterable
from time import perf_counter


MAX_POKEMON = 898

async def get_random_pokemon() -> str:
    pokemon_id = randint(1, MAX_POKEMON)
    url = f"https://pokeapi.co/api/v2/pokemon/{pokemon_id}"
    pokemon = await http_get(url)
    return str(pokemon["name"])


async def get_next_pokemon(total: int) -> AsyncIterable[str]:
    for _ in range(total):
        name = await get_random_pokemon()
        yield name


async def main():

    start_time = perf_counter()
    async for pokemon in get_next_pokemon(20):
        print(f"You got {pokemon}!")

    end_time = perf_counter()
    print(f"Time taken: {end_time - start_time}")

if __name__ == "__main__":
    asyncio.run(main())
import asyncio
from layer_1.async_examples.req_http import http_get, JSONObject
from random import randint

MAX_POKEMON = 898


async def get_random_pokemon() -> JSONObject:
    pokemon_id = randint(1, MAX_POKEMON)
    url = f"https://pokeapi.co/api/v2/pokemon/{pokemon_id}"
    return await http_get(url)


async def main():
    pokemon = await get_random_pokemon()
    print(f"You got {pokemon['name']}!")

if __name__ == "__main__":
    asyncio.run(main())
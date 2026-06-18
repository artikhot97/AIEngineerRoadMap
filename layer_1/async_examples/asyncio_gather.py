import asyncio
from layer_1.async_examples.req_http import http_get_sync, JSONObject, http_get
from random import randint
from typing import Any , Awaitable, List, Tuple
from time import perf_counter

MAX_POKEMON = 898

def get_random_pokemon_name_sync() -> str:
    pokemon_id = randint(1, MAX_POKEMON)
    url = f"https://pokeapi.co/api/v2/pokemon/{pokemon_id}"
    pokemon = http_get_sync(url)
    return str(pokemon["name"])

async def get_random_pokemon_name() -> str:
    pokemon_id = randint(1, MAX_POKEMON)
    url = f"https://pokeapi.co/api/v2/pokemon/{pokemon_id}"
    pokemon = await http_get(url)
    return str(pokemon["name"])

async def main()-> None:

    # sync call 

    start_time = perf_counter()
    for _ in range(20):
        get_random_pokemon_name_sync()
    print(f"Total time (synchronous): {perf_counter() - start_time}")
    
    
    #async call
    start_time = perf_counter()
    await asyncio.gather(*[get_random_pokemon_name() for _ in range(20)])
    print(f"Time taken (asynchronous): {perf_counter() - start_time}")



asyncio.run(main())
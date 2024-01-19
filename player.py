import asyncio
import time


from poke_env import AccountConfiguration, LocalhostServerConfiguration
from poke_env.player import Player, RandomPlayer


class HighestDamagePlayer(Player):
    def choose_move(self, battle):
        # If the player can attack, it will
        if battle.available_moves:
            best_move = max(battle.available_moves, key=lambda move: move.base_power) #returns highest base power move
            return self.create_order(best_move)

        else:
            return self.choose_random_move(battle)# switch if cant attack



async def main():
    loopstart = time.time()

    # gen9 ranbats crashes with new indigo disk update so go gen8
    highest_damage_player = HighestDamagePlayer(
        battle_format="gen8randombattle",
    )
    random_player = RandomPlayer(
        battle_format="gen8randombattle",
    )


    # Now, let's evaluate our player
    await highest_damage_player.battle_against(random_player, n_battles=500)

    print(
        "Highest damage player won %d / 500 battles [this took %f seconds]"
        % (
            highest_damage_player.n_won_battles, time.time() - loopstart
        )
    )


if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())


"""
git clone https://github.com/smogon/pokemon-showdown.git
cd pokemon-showdown
npm install
cp config/config-example.js config/config.js
node pokemon-showdown start --no-security
"""
from random import randint

def choose_random(game):
    return randint(0, game.machine_count-1)


def explore_exploit(game, p=0.5):
    while game.next_turn <= game.turns*p:
        return game.next_turn % game.machine_count
    mean_list = game.history
    index_max = max(range(len(mean_list)), key=mean_list.__getitem__)
    return index_max


def explore_exploit_wrapper(p):
    return lambda g: explore_exploit(g, p)
from planetwars import PlanetWars
from planetwars.views import TextView

def main():
    import sys
    p1 = sys.argv[1]
    p2 = sys.argv[2]
    # PS: Added extra parameter for collisions
    game = PlanetWars([p1, p2], "map1", 100, False)
    game.add_view(TextView())
    game.play()

if __name__ == '__main__':
    main()

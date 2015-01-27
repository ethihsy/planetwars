from planetwars import PlanetWars
from planetwars.views import TextView

def main():
    import sys
    p1 = sys.argv[1]
    p2 = sys.argv[2]
    game = PlanetWars([p1, p2], map_name="map1", turns_per_second=1000, collisions=True)
    game.add_view(TextView())
    game.play()

if __name__ == '__main__':
    main()

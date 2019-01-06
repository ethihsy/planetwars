import copy
import sys
import random
import argparse
import numpy

from operator import itemgetter, attrgetter, methodcaller
from planetwars import PlanetWars
from planetwars.views import TextView
from ai.state import State

def print_planets(planets):
  for p in planets:
    print "%3d" % p.id, "%6.2f" % p.x, "%6.2f" % p.y, p.owner, "%4d" % p.ships, "%3d" % p.growth

def main(argv):

    stats = []
    win_ctr=turns_ctr=r_ctr=counter=n_g=qv=qv_ctr=loss=0

    parser = argparse.ArgumentParser()
    parser.add_argument('--collisions', action='store_true', required=False, default=False,
                        help="Should the ships collide with each other?")
    parser.add_argument('--games', type=int, required=False, default=100,
                        help="Number of games to be played.")
    parser.add_argument('--seed', type=int, required=False, default=0,
                        help="Initial rng seed, 0 = system-based")
    parser.add_argument('--p1num', type=int, required=False, default=1,
                        help="Planet number for player 1.")
    parser.add_argument('--p2num', type=int, required=False, default=1,
                        help="Planet number for player 2.")
    parser.add_argument('--nnum', type=int, required=False, default=21,
                        help="Number of neutral planets.")
    parser.add_argument('--genmaps', action='store_true', required=False, default=False, 
                        help='Use generated maps.')
    parser.add_argument('--quiet', action='store_true', required=False, default=False,
                        help="Suppress all output to the console.")

    arguments, remaining = parser.parse_known_args(argv)

    seed = 0
    if arguments.seed == 0:
      # use system seed and print the resulting random integer
      seed = random.randint(1, 2000000000)
    else:
      # use passed on seed
      seed = arguments.seed

    random.seed(seed)
    print "seed=", seed  #, "rnd1=", random.randint(1, 2000000000)

    if arguments.genmaps:
      print "p1num=", arguments.p1num
      print "p2num=", arguments.p2num
      print "nnum=",  arguments.nnum

    players = remaining
    res = numpy.zeros((len(players), len(players)))
    time_max = numpy.zeros((len(players)))
    time_totals = numpy.zeros((len(players)))    

    maps = []
    temp = 0
    for gn in range(arguments.games):
      n = gn+1
      state = State()
      map_name = None
      if arguments.genmaps:
        state.random_setup(arguments.p1num, arguments.p2num, arguments.nnum)
      else:
        if len(maps) == 0:
          maps = ["map%i" % i for i in range(1, 100)]
          random.shuffle(maps, random.random)
        map_name = maps.pop()

      print "---------"
      if map_name == None:
        print "game", gn, "Map: generated"
      else:
        print "game", gn, "Map:", map_name
      print
      
      for i1,p1 in enumerate(players):
        for i2,p2 in enumerate(players):
          if i2 >= i1:
            continue
            
          pair = [players[i1], players[i2]]

          if map_name == None:
            game = PlanetWars(pair, planets=copy.deepcopy(state.planets), \
                              fleets=copy.deepcopy(state.fleets), collisions=arguments.collisions)
          else:
            game = PlanetWars(pair, map_name, collisions=arguments.collisions)
          
          winner, ship_counts, turns, tt, tm, reward, reward_e, counter, qv, qv_ctr, loss = game.play()

          turns_ctr += turns
          r_ctr+= reward          
          
          print("DQN bot reward for game is {}".format(reward))
          print("Another bot reward for game is {}".format(reward_e)) 
  
          print "%-16s vs. %-16s winner= %d turns= %d" % (p1, p2, winner, turns)
          if winner == 0:
            res[i1][i2] += 0.5
            res[i2][i1] += 0.5
          elif winner == 1:
            win_ctr += 1
            res[i1][i2] += 1
            res[i2][i1] += 0
          else:
            res[i1][i2] += 0
            res[i2][i1] += 1

          time_totals[i1] += tt[1]/turns
          time_totals[i2] += tt[2]/turns
          time_max[i1] = max(time_max[i1], tm[1])
          time_max[i2] = max(time_max[i2], tm[2])          
          
      totals = []
      list = []
      for i1,p1 in enumerate(players):
        total = 0
        for i2,p2 in enumerate(players):
          total += res[i1][i2]
        totals.append(total)
        list.append((i1, total))

      slist = sorted(list, key=itemgetter(1), reverse=True)

      print
      print "                            ",
      for i1,p1 in enumerate(players):
        print "    %2d" % i1,
      print
      
      for i1,p1 in enumerate(players):
        mi1 = slist[i1][0]
        mt1 = slist[i1][1]
        print "%2d %-17s : %5.1f : " % (i1, players[mi1], 100*mt1/n/(len(players)-1)),
        for i2,p2 in enumerate(players):
          mi2 = slist[i2][0]
          print "%5.1f " % (100*res[mi1][mi2]/n),
        print " avgt: %7.2f maxt: %7.2f" % (1000*time_totals[mi1]/n/(len(players)-1), 1000*time_max[mi1])
      
      if counter>=10000:
        n_g = (gn - n_g + 1.0 if temp==0 else float(gn - temp))
        stats.append(str(win_ctr/n_g)+"\t"+str(turns_ctr/n_g)+"\t"+str(r_ctr/n_g)+"\t"+str(qv[0]/float(qv_ctr))+"\t"+str(loss/float(counter)))

        win_ctr=turns_ctr=r_ctr=0
        temp = gn

    
    ouput_file(game, stats)
    print res
    
def ouput_file(game, stats):
    game.save_weights()
    file = open("stats", 'w')
    file.write("win\tturns/game\treward/game\taverage Q value\tloss\n")
    for i in range(len(stats)):
      file.write(stats[i])
      file.write("\n")
    file.close()  

if __name__ == '__main__':
    main(sys.argv[1:])

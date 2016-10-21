"""
main here

-- driver software

"""
import argparse
import src.game_engine as breakout
import src.agents as agents
import sys


def main(args, parser):
    game = None
    if args.p == "human":
        game = breakout.HumanControlledBreakout(args.v, args.d)
    elif args.p == "baseline":
        game = breakout.BotControlledBreakout(agents.Baseline(), args.v, args.d)
    elif args.p == "oracle":
        game = breakout.OracleControlledBreakout(args.v, args.d)
    game.run()

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print 'usage: main.py [-h] -p player_type [-v] [-d]'
        quit()
    parser = argparse.ArgumentParser(description='Play the game of breakout, or sit back and have a bot play it for you.')
    parser.set_defaults(func=main)
    parser.add_argument('-p', metavar="type", type=str, help="player type. accepted values: human, baseline")
    parser.add_argument('-v', action="store_true", help="verbose mode")
    parser.add_argument('-d', action="store_true", help="display game")

    args = parser.parse_args()
    args.func(args, parser)



"""
main here

-- driver software

"""
import argparse
import src.game_engine as breakout


def main(args, parser):
    game = None
    if args.player == "human":
        game = breakout.HumanControlledBreakout()
        game.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Play breakout.')
    parser.set_defaults(func=main)
    parser.add_argument('-player', metavar='["human", "dumb_bot", etc (TODO)]', type=str,
                        help='plyer type')

    args = parser.parse_args()
    args.func(args, parser)



"""
main here

-- driver software

"""
import argparse
import src.game_engine as breakout
import src.agents as agents
import sys
import src.function_approximators as fn_approx
import src.feature_extractors as ft_extract 


def main(args, parser):
    game = None
    if args.p == "human":
        game = breakout.HumanControlledBreakout(args.v, args.d, args.b)
    elif args.p == "baseline":
        game = breakout.BotControlledBreakout(agents.Baseline(), args.v, args.d, args.b)
    elif args.p == "oracle":
        game = breakout.OracleControlledBreakout(args.v, args.d, args.b)
    elif args.p == 'simpleQLearning':
        game = breakout.BotControlledBreakout(agents.DiscreteQLearningAgent(), args.v, args.d, args.b, args.wr, args.rd)
    elif args.p == 'linearDiscreteFnApprox':
        # give feature extractor to function approximator
        fe = ft_extract.SimpleDiscreteFeatureExtractor()
        fa = fn_approx.LinearFunctionApproximator(fe)

        agent = agents.FuncApproxQLearningAgent(fa)
        game = breakout.BotControlledBreakout(agent, args.v, args.d, args.b, args.wr, args.rd)

    game.run()

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print 'usage: main.py [-h] -p player_type [-v] [-d] [-b N]'
        quit()
    parser = argparse.ArgumentParser(description='Play the game of breakout, or sit back and have a bot play it for you.')
    parser.set_defaults(func=main)
    parser.add_argument('-p', metavar="type", type=str, help="player type. accepted values: human, baseline, simpleQLearning, linearDiscreteFnApprox")
    parser.add_argument('-v', action="store_true", help="verbose mode")
    parser.add_argument('-d', action="store_true", help="display game")
    parser.add_argument('-b', type=int, default=1, help="num batch iterations (defaults to 1)")
    parser.add_argument('-wr', type=bool, default=False, help="write model to file when done")
    parser.add_argument('-rd', type=str, help="read model parameters from file")
    args = parser.parse_args()
    args.func(args, parser)



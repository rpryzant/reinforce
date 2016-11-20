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
    if args.csv:
        print 'score,time,bricks'

    game = None
    if args.p == "human":
        game = breakout.HumanControlledBreakout(args.csv, args.v, args.d, args.b, args.wr, args.rd)

    elif args.p == "followBaseline":
        agent = agents.FollowBaseline()
        game = breakout.BotControlledBreakout(agent, args.csv, args.v, args.d, args.b, args.wr, args.rd)

    elif args.p == "randomBaseline":
        agent = agents.RandomBaseline()
        game = breakout.BotControlledBreakout(agent, args.csv, args.v, args.d, args.b, args.wr, args.rd)        

    elif args.p == "oracle":
        game = breakout.OracleControlledBreakout(args.csv, args.v, args.d, args.b, args.wr)

    elif args.p == 'simpleQLearning':
        agent = agents.DiscreteQLearning()
        game = breakout.BotControlledBreakout(agent, args.csv, args.v, args.d, args.b, args.wr, args.rd)

    elif args.p == 'linearQ':
        fe = ft_extract.SimpleContinuousFeatureExtractor()
        agent = agents.QLearning(fe)        
        game = breakout.BotControlledBreakout(agent, args.csv, args.v, args.d, args.b, args.wr, args.rd)

    elif args.p == 'linearReplayQ':
        fe = ft_extract.SimpleContinuousFeatureExtractor()
        agent = agents.QLearningReplayMemory(fe)        
        game = breakout.BotControlledBreakout(agent, args.csv, args.v, args.d, args.b, args.wr, args.rd)

    elif args.p == 'sarsa':
        fe = ft_extract.SimpleContinuousFeatureExtractor()
        agent = agents.SARSA(fe)        
        game = breakout.BotControlledBreakout(agent, args.csv, args.v, args.d, args.b, args.wr, args.rd)

    elif args.p == 'sarsaLambda':
        fe = ft_extract.SimpleContinuousFeatureExtractor()
        agent = agents.SARSALambda(fe)        
        game = breakout.BotControlledBreakout(agent, args.csv, args.v, args.d, args.b, args.wr, args.rd)



    ############################################################################
    # # # # # # # # # test bed for experimental features # # # # # # # # # # # #
    ############################################################################    
    elif args.p == 'test':
#        fe = ft_extract.SanityCheckFeatures()
        fe = ft_extract.SimpleContinuousFeatureExtractor()
#        fa = fn_approx.LinearReplayMemory(fe, memory_size=5000, replay_sample_size=1, num_static_target_steps=2000)
#        fa = fn_approx.LinearFunctionApproximator(fe)
        agent = agents.QLearningReplayMemory(fe)
#        agent = agents.FuncApproxQLearningAgent(fa)
        game = breakout.BotControlledBreakout(agent, args.csv, args.v, args.d, args.b, args.wr, args.rd)



    game.run()

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print 'usage: main.py [-h] -p player_type [-v] [-d] [-b N]'
        quit()
    parser = argparse.ArgumentParser(description='Play the game of breakout, or sit back and have a bot play it for you.')
    parser.set_defaults(func=main)
    parser.add_argument('-p', metavar="type", type=str, help="player type. accepted values: human, baseline, simpleQLearning, linearDiscreteFnApprox, linearContinuousFnApprox")
    parser.add_argument('-v', action="store_true", help="verbose mode")
    parser.add_argument('-csv', action="store_true", help="csv mode")
    parser.add_argument('-d', action="store_true", help="display game")
    parser.add_argument('-b', type=int, default=1, help="num batch iterations (defaults to 1)")
    parser.add_argument('-wr', type=str, help="write model to file when done")
    parser.add_argument('-rd', type=str, help="read model parameters from file")
    args = parser.parse_args()
    args.func(args, parser)



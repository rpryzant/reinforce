"""
main here

-- driver software

"""
import argparse
import src.game_engine as breakout
import src.agents as agents
import sys
# import src.function_approximators as fn_approx
import src.feature_extractors as ft_extract 


def main(args, parser):
    # global parameters (can/should be changed)
    EXPLORATION_PROB = args.e
    DISCOUNT = 0.993
    memory_size = args.memory_size or 5000
    sample_size = args.sample_size or 4


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
        agent = agents.DiscreteQLearning(gamma=DISCOUNT,
                                         epsilon=EXPLORATION_PROB,
                                         stepSize=0.001)
        game = breakout.BotControlledBreakout(agent, args.csv, args.v, args.d, args.b, args.wr, args.rd)

    elif args.p == 'linearQ':
        fe = ft_extract.ContinuousFeaturesV2()
        agent = agents.QLearning(fe, 
                                 epsilon=EXPLORATION_PROB,
                                 gamma=DISCOUNT,
                                 stepSize=agents.RLAgent.inverseSqrt)        
        game = breakout.BotControlledBreakout(agent, args.csv, args.v, args.d, args.b, args.wr, args.rd)

    elif args.p == 'linearReplayQ':
        fe = ft_extract.ContinuousFeaturesV2()
        agent = agents.QLearningReplayMemory(fe,
                                             epsilon=EXPLORATION_PROB,
                                             gamma=DISCOUNT,
                                             stepSize=agents.RLAgent.inverseSqrt,
                                             num_static_target_steps=500,
                                             memory_size=memory_size, 
                                             replay_sample_size=sample_size)
        game = breakout.BotControlledBreakout(agent, args.csv, args.v, args.d, args.b, args.wr, args.rd)

    elif args.p == 'sarsa':
        fe = ft_extract.ContinuousFeaturesV2()
        agent = agents.SARSA(fe,
                             epsilon=EXPLORATION_PROB,
                             gamma=DISCOUNT,
                             stepSize=agents.RLAgent.inverse)
        game = breakout.BotControlledBreakout(agent, args.csv, args.v, args.d, args.b, args.wr, args.rd)

    elif args.p == 'sarsaLambda':
        fe = ft_extract.ContinuousFeaturesV2()
        agent = agents.SARSALambda(fe,
                                   epsilon=EXPLORATION_PROB,
                                   gamma=DISCOUNT,
                                   stepSize=agents.RLAgent.inverse,
                                   threshold=0.1,
                                   decay=0.98)       
        game = breakout.BotControlledBreakout(agent, args.csv, args.v, args.d, args.b, args.wr, args.rd)

    elif args.p == 'nn':
        # TODO - FEED IN CONTINUOUS/RAW DATA
        #         might not help as much because already has higher level features
        fe = ft_extract.ContinuousFeaturesV2()
        agent = agents.NNAgent(fe, args.v,
                               epsilon=EXPLORATION_PROB,
                               gamma=DISCOUNT,
                               stepSize=0.001)        
        game = breakout.BotControlledBreakout(agent, args.csv, args.v, args.d, args.b, args.wr, args.rd)

    elif args.p == 'policyGradients':
        # better for continuous featuer spaces
        # very unstable
        # critic net, actor net best for continous features spaces
        # READ       https://arxiv.org/abs/1509.02971
        # TODO - lower learning rate, gradient clipping
        fe = ft_extract.ContinuousFeaturesV2()
        agent = agents.PolicyGradients(fe, args.v,
                               epsilon=EXPLORATION_PROB,
                               gamma=DISCOUNT,
                               stepSize=0.001)  
        game = breakout.BotControlledBreakout(agent, args.csv, args.v, args.d, args.b, args.wr, args.rd)



    ############################################################################
    # # # # # # # # # test bed for experimental features # # # # # # # # # # # #
    ############################################################################    
    elif args.p == 'test':
        fe = ft_extract.ContinuousFeaturesV2()
        agent = agents.PolicyGradients(fe, args.v,
                               epsilon=EXPLORATION_PROB,
                               gamma=DISCOUNT,
                               stepSize=0.001)  
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
    parser.add_argument('-e', type=float, default=0.3, help="epsilon (exploration prob)")
    parser.add_argument('-memory_size', type=int, help="replay memory size")
    parser.add_argument('-sample_size', type=int, help="replay sample size")
    args = parser.parse_args()
    args.func(args, parser)



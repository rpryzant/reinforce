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

    trace_threshold = args.trace_threshold or 0.1
    trace_decay = args.trace_decay or 0.98

    # stoopid stoopid argument parsing but idgaf
    step_size = args.step_size or 0.001
    if step_size == 'inv_sqrt':
        step_size = agents.RLAgent.inverseSqrt
    elif step_size == 'inv':
        step_size = agents.RLAgent.inverse
    else:
        step_size = agents.RLAgent.constant(float(step_size))

    # more stoopid stuff
    feature_set = args.feature_set or None
    if feature_set == 'v1':
        feature_set = ft_extract.ContinuousFeaturesV1()
    elif feature_set == 'v2':
        feature_set = ft_extract.ContinuousFeaturesV2()
    elif feature_set == 'v3':
        feature_set = ft_extract.ContinuousFeaturesV3()
    elif feature_set == 'v4':
        feature_set = ft_extract.ContinuousFeaturesV4()
    elif feature_set == 'v5':
        feature_set = ft_extract.ContinuousFeaturesV5()
    elif feature_set == 'v6':
        feature_set = ft_extract.ContinuousFeaturesV6()
    else:
        # default to v2. cause it seems to work best
        feature_set = ft_extract.ContinuousFeaturesV2()




    game = None
    if args.p == "human":
        game = breakout.HumanControlledBreakout(args.csv, args.v, args.d, args.b, args.wr, args.rd)

    elif args.p == "followBaseline":
        agent = agents.FollowBaseline()

    elif args.p == "randomBaseline":
        agent = agents.RandomBaseline()

    elif args.p == "oracle":
        game = breakout.OracleControlledBreakout(args.csv, args.v, args.d, args.b, args.wr)

    elif args.p == 'simpleQLearning':
        agent = agents.DiscreteQLearning(gamma=DISCOUNT,
                                         epsilon=EXPLORATION_PROB,
                                         stepSize=step_size)
    elif args.p == 'linearQ':
        agent = agents.QLearning(feature_set, 
                                 epsilon=EXPLORATION_PROB,
                                 gamma=DISCOUNT,
                                 stepSize=step_size)
    elif args.p == 'linearReplayQ':
        agent = agents.QLearningReplayMemory(feature_set,
                                             epsilon=EXPLORATION_PROB,
                                             gamma=DISCOUNT,
                                             stepSize=step_size,
                                             num_static_target_steps=500,
                                             memory_size=memory_size, 
                                             replay_sample_size=sample_size)
    elif args.p == 'sarsa':
        agent = agents.SARSA(feature_set,
                             epsilon=EXPLORATION_PROB,
                             gamma=DISCOUNT,
                             stepSize=step_size)
    elif args.p == 'sarsaLambda':
        agent = agents.SARSALambda(feature_set,
                                   epsilon=EXPLORATION_PROB,
                                   gamma=DISCOUNT,
                                   stepSize=step_size,
                                   threshold=trace_threshold,
                                   decay=trace_decay)       
    elif args.p == 'nn':
        # TODO - FEED IN CONTINUOUS/RAW DATA
        #         might not help as much because already has higher level features
        agent = agents.NNAgent(feature_set, args.v,
                               epsilon=EXPLORATION_PROB,
                               gamma=DISCOUNT,
                               stepSize=step_size)        
    elif args.p == 'policyGradients':
        # better for continuous featuer spaces
        # very unstable
        # critic net, actor net best for continous features spaces
        # READ       https://arxiv.org/abs/1509.02971
        # TODO - lower learning rate, gradient clipping
        agent = agents.PolicyGradients(feature_set, args.v,
                               epsilon=EXPLORATION_PROB,
                               gamma=DISCOUNT,
                               stepSize=step_size)  



    ############################################################################
    # # # # # # # # # test bed for experimental features # # # # # # # # # # # #
    ############################################################################    
    elif args.p == 'test':
        fe = ft_extract.ContinuousFeaturesV2()
        agent = agents.PolicyGradients(fe, args.v,
                               epsilon=EXPLORATION_PROB,
                               gamma=DISCOUNT,
                               stepSize=0.001)  

    if args.p not in ['human', 'oracle']:
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
    parser.add_argument('-trace_threshold', type=float, help="eligibility trace threshold")
    parser.add_argument('-trace_decay', type=float, help="eligilbility trace decay (lambda)")
    parser.add_argument('-feature_set', type=str, help='what kind of feature set do you wanna use yo')
    parser.add_argument('-step_size', type=str, help='what kind of step size (function) do you wanna use')
    args = parser.parse_args()
    args.func(args, parser)



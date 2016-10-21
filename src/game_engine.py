"""
Breakout


TODO
   - track previous state, action, rewards for Q-learning, model-free learning, etc
   - print out game stats after game over
   - driver method for batch games/aggregate statistics
   - let user quit from keyboard
   - better oracle/baseline?
   - make bricks flush against walls
   - refactor run() method....move shared bits to Breakout in a clean way (how to do with display option?)
"""
import sys
import pygame
import abc
import math
import utils
from constants import *


class Breakout(object):
    """
    Abstract base class for breakout

    Implements basically all of the game logic. The only thing that remains
       for subclasses to flesh out is the run() method
    """
    def __init__(self, verbose, display):
        self.verbose = verbose
        self.display = display

        pygame.init()

        self.screen = pygame.display.set_mode(SCREEN_SIZE)
        pygame.display.set_caption("Breakout!!")
        
        self.clock = pygame.time.Clock()

        if pygame.font:
            self.font = pygame.font.Font(None,30)
        else:
            self.font = None

        self.init_game()

        
    def init_game(self):
        """ set game params """
        self.lives = 3
        self.score = 0
        self.num_hits = 0
        self.boosts_remaining = 3
        self.boost_time = 0
        self.speed_multiplyer = 1.0
        self.game_state = STATE_BALL_IN_PADDLE
        self.paddle   = pygame.Rect(300,PADDLE_Y,PADDLE_WIDTH,PADDLE_HEIGHT)
        self.ball     = pygame.Rect(300,PADDLE_Y - BALL_DIAMETER,BALL_DIAMETER,BALL_DIAMETER)
        self.ball_vel = [5, 5]
        self.create_bricks()
        

    def create_bricks(self):
        """ creates smashable targets """
        y_ofs = 35
        self.bricks = []
        for i in range(7):
            x_ofs = 35
            for j in range(8):
                self.bricks.append(pygame.Rect(x_ofs,y_ofs,BRICK_WIDTH,BRICK_HEIGHT))
                x_ofs += BRICK_WIDTH + 10
            y_ofs += BRICK_HEIGHT + 5

    def draw_bricks(self):
        for brick in self.bricks:
            pygame.draw.rect(self.screen, BRICK_COLOR, brick)
        
    def take_input(self, input):
        """ takes a vector of game inputs and applies them to game objects """
        boost = 5 if self.boost_time > 0 else 0

        if INPUT_L in input:
            self.set_paddle_pos(self.paddle.left - (7 + boost))

        if INPUT_R in input:
            self.set_paddle_pos(self.paddle.left + (7 + boost))

        if INPUT_B in input and self.boosts_remaining > 0 and self.boost_time == 0:
            self.boosts_remaining -= 1
            self.boost_time += 25

        if INPUT_SPACE in input and self.game_state == STATE_BALL_IN_PADDLE:
            self.ball_vel = [5,5]
            self.speed_multiplyer = 1.0
            self.game_state = STATE_PLAYING
        elif INPUT_ENTER in input and (self.game_state == STATE_GAME_OVER or self.game_state == STATE_WON):
            self.init_game()

    def set_paddle_pos(self, x):
        self.paddle.left = x
        if self.paddle.left < 0:
            self.paddle.left = 0
        elif self.paddle.left > MAX_PADDLE_X:
            self.paddle.left = MAX_PADDLE_X

    def move_ball(self):
        """ applies ball velocity vector to ball """
        self.ball.x += self.ball_vel[0] * self.speed_multiplyer
        self.ball.y -= self.ball_vel[1] * self.speed_multiplyer    # pygame treats "up" as decreasing y axis

        if self.ball.left <= 0:
            self.ball.left = 0
            self.ball_vel[0] = -self.ball_vel[0]
        elif self.ball.left >= MAX_BALL_X:
            self.ball.left = MAX_BALL_X
            self.ball_vel[0] = -self.ball_vel[0]
        
        if self.ball.top < 0:
            self.ball.top = 0
            self.ball_vel[1] = -self.ball_vel[1]
        elif self.ball.top >= MAX_BALL_Y:            
            self.ball.top = MAX_BALL_Y
            self.ball_vel[1] = -self.ball_vel[1]


    def handle_collisions(self):
        """ logic for collision between ball and game object """
        for brick in self.bricks:
            if self.ball.colliderect(brick):
                self.score += 3
                self.num_hits += 1
                self.ball_vel[1] = -self.ball_vel[1]
                self.bricks.remove(brick)
                self.speed_multiplyer = min(self.speed_multiplyer + 0.1, 2.5)
                break

        if len(self.bricks) == 0:
            self.game_state = STATE_WON
            
        if self.ball.colliderect(self.paddle):
            distance_from_center = float(self.ball.centerx - self.paddle.centerx)
            self.ball.top = PADDLE_Y - BALL_DIAMETER
            self.ball_vel[0] += distance_from_center / 7
            self.ball_vel[1] = -self.ball_vel[1]
        elif self.ball.top > self.paddle.top:
            self.lives -= 1
            if self.lives > 0:
                self.game_state = STATE_BALL_IN_PADDLE
                self.ball.left = self.paddle.left + self.paddle.width / 2
                self.ball.top  = self.paddle.top - self.ball.height
            else:
                self.game_state = STATE_GAME_OVER


    def show_stats(self):
        if self.font:
            font_surface = self.font.render("SCORE: " + str(self.score) + " LIVES: " + str(self.lives) + " BOOSTS: " + str(self.boosts_remaining), False, WHITE)
            self.screen.blit(font_surface, (205,5))

    def show_message(self,message, x_ofs = 0, y_ofs = 0):
        if self.font:
            size = self.font.size(message)
            font_surface = self.font.render(message,False, WHITE)
            x = ((SCREEN_SIZE[0] - size[0]) / 2) + x_ofs
            y = ((SCREEN_SIZE[1] - size[1]) / 2) + y_ofs
            self.screen.blit(font_surface, (x,y))
        

    def get_state(self):
        """ contstructs and returns a vector representation of current game state """
        # TODO - WORK ON STATE VECTOR
        state = {
            'game_state': self.game_state
            }
        return state

    def discretizeLocation(self, x, y):
        """ 
        converts continuous coordinates in R^2 to discrete location measurement 

        does so by converting game board to grid of 20x20 pixel squares, then
          gives the index of the square that (x, y) is in
        """
        entries_in_row = SCREEN_SIZE[0] / 20
        x_grid = x / 10
        y_grid = y / 10
        return x_grid + y_grid * (SCREEN_SIZE[0] / 20)

    
    # TODO - put many of these in utils file

    def discretizeAngle(self, vec):
        """ 
        buckets the continuous angle of a vector into one of 16 discrete angle categories
        """
        return int(utils.angle(vec) / 10)

    @abc.abstractmethod
    def run(self):
        return


class HumanControlledBreakout(Breakout):
    """
    Breakout subclass which takes inputs from the keyboard during run()
    """
    def __init__(self, verbose, display):
        super(HumanControlledBreakout, self).__init__(verbose, display)

    def _get_input_from_keyboard(self):
        keys = pygame.key.get_pressed()
        input = []
        input += [INPUT_L] if keys[pygame.K_LEFT] else []
        input += [INPUT_R] if keys[pygame.K_RIGHT] else []
        input += [INPUT_B] if keys[pygame.K_b] else []
        input += [INPUT_SPACE] if keys[pygame.K_SPACE] else []
        input += [INPUT_ENTER] if keys[pygame.K_RETURN] else []

        return input
        

    def run(self):
        while 1:            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit

            self.clock.tick(50)
            self.screen.fill(BLACK)

            if self.game_state == STATE_PLAYING:
                self.move_ball()
                self.handle_collisions()
            elif self.game_state == STATE_BALL_IN_PADDLE:
                self.ball.left = self.paddle.left + self.paddle.width / 2
                self.ball.top  = self.paddle.top - self.ball.height
                self.show_message("PRESS SPACE TO LAUNCH THE BALL")
                self.show_message("PRESS B TO BOOST", 0, 30)
            elif self.game_state == STATE_GAME_OVER:
                self.show_message("GAME OVER. PRESS ENTER TO PLAY AGAIN")
            elif self.game_state == STATE_WON:
                self.show_message("YOU WON! PRESS ENTER TO PLAY AGAIN")
                
            self.draw_bricks()
            self.boost_time = max(self.boost_time - 1, 0)
            # Draw paddle
            if self.boost_time > 0:
                pygame.draw.rect(self.screen, PINK, self.paddle)                
            else:
                pygame.draw.rect(self.screen, BLUE, self.paddle)

            # Draw ball
            pygame.draw.circle(self.screen, WHITE, (self.ball.left + BALL_RADIUS, self.ball.top + BALL_RADIUS), BALL_RADIUS)

            self.show_stats()

            pygame.display.flip()

            self.take_input(self._get_input_from_keyboard())


class BotControlledBreakout(Breakout):
    """ 
    Breakout subclass for agent-controlled games.
    
    Whereas HumanControlledBreakout disregaurds game state, BotControlledBreakout gives a vector representation of
       each state (and possibly other stuff) to a game-playing agent, and recieves input (actions) from this agent
    """

    """ TODO - create state vectors, don't have run draw stuff, give state vectors to learning agent (in another file""" 
    def __init__(self, agent, verbose, display):
        super(BotControlledBreakout, self).__init__(verbose, display)
        self.agent = agent


    def run(self):
        while 1:            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit

            if self.display:
                self.clock.tick(50)
                self.screen.fill(BLACK)

            if self.game_state == STATE_PLAYING:
                self.move_ball()
                self.handle_collisions()
            elif self.game_state == STATE_BALL_IN_PADDLE:
                self.show_message("PRESS SPACE TO LAUNCH THE BALL")
                self.show_message("PRESS B TO BOOST", 0, 30)
            elif self.game_state == STATE_GAME_OVER:
                self.show_message("GAME OVER. PRESS ENTER TO PLAY AGAIN")
            elif self.game_state == STATE_WON:
                self.show_message("YOU WON! PRESS ENTER TO PLAY AGAIN")

            self.boost_time = max(self.boost_time - 1, 0)
                
            if self.display:
                self.draw_bricks()
                # Draw paddle
                if self.boost_time > 0:
                    pygame.draw.rect(self.screen, PINK, self.paddle)                
                else:
                    pygame.draw.rect(self.screen, BLUE, self.paddle)
                # Draw ball
                pygame.draw.circle(self.screen, WHITE, (self.ball.left + BALL_RADIUS, self.ball.top + BALL_RADIUS), BALL_RADIUS)
                self.show_stats()
                pygame.display.flip()
    

            # agent observes state, makes move
            self.agent.processState(self.get_state())
            self.take_input(self.agent.takeAction())



class OracleControlledBreakout(Breakout):
    """ 
    Breakout subclass for oracle-controlled games.
    
    The oracle can return any ball - it translates the paddle to 
    match the exact position of the ball at all times
    """
    def __init__(self, verbose, display):
        super(OracleControlledBreakout, self).__init__(verbose, display)

    def run(self):
        while 1:            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit

            if self.display:
                self.clock.tick(50)
                self.screen.fill(BLACK)

            if self.game_state == STATE_PLAYING:
                self.move_ball()
                self.handle_collisions()
            elif self.game_state == STATE_BALL_IN_PADDLE:
                self.show_message("PRESS SPACE TO LAUNCH THE BALL")
                self.show_message("PRESS B TO BOOST", 0, 30)
                self.take_input([INPUT_SPACE])
            elif self.game_state == STATE_GAME_OVER:
                self.show_message("GAME OVER. PRESS ENTER TO PLAY AGAIN")
            elif self.game_state == STATE_WON:
                self.show_message("YOU WON! PRESS ENTER TO PLAY AGAIN")

            self.boost_time = max(self.boost_time - 1, 0)
                
            if self.display:
                self.draw_bricks()
                # Draw paddle
                if self.boost_time > 0:
                    pygame.draw.rect(self.screen, PINK, self.paddle)                
                else:
                    pygame.draw.rect(self.screen, BLUE, self.paddle)
                # Draw ball
                pygame.draw.circle(self.screen, WHITE, (self.ball.left + BALL_RADIUS, self.ball.top + BALL_RADIUS), BALL_RADIUS)
                self.show_stats()
                pygame.display.flip()

            self.set_paddle_pos(self.ball.left - 4)


if __name__ == "__main__":
    HumanControlledBreakout().run()
#    BotControlledBreakout(None, True).run()

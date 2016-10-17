"""
Breakout
"""
import sys
import pygame
import abc
import math

SCREEN_SIZE   = 640,480

# Object dimensions
BRICK_WIDTH   = 60
BRICK_HEIGHT  = 15
PADDLE_WIDTH  = 80
PADDLE_HEIGHT = 12
BALL_DIAMETER = 16
BALL_RADIUS   = BALL_DIAMETER / 2

MAX_PADDLE_X = SCREEN_SIZE[0] - PADDLE_WIDTH
MAX_BALL_X   = SCREEN_SIZE[0] - BALL_DIAMETER
MAX_BALL_Y   = SCREEN_SIZE[1] - BALL_DIAMETER

# Paddle Y coordinate
PADDLE_Y = SCREEN_SIZE[1] - PADDLE_HEIGHT - 10

# Color constants
BLACK = (0,0,0)
WHITE = (255,255,255)
BLUE  = (0,0,255)
PINK = (255,105,180)
BRICK_COLOR = (200,200,0)

# State constants
STATE_BALL_IN_PADDLE = 0
STATE_PLAYING = 1
STATE_WON = 2
STATE_GAME_OVER = 3


class Breakout(object):

    def __init__(self):
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
        self.lives = 3
        self.score = 0
        self.num_hits = 0
        self.boosts_remaining = 3
        self.boost_time = 0
        self.speed_multiplyer = 1.0
        self.state = STATE_BALL_IN_PADDLE

        self.paddle   = pygame.Rect(300,PADDLE_Y,PADDLE_WIDTH,PADDLE_HEIGHT)
        self.ball     = pygame.Rect(300,PADDLE_Y - BALL_DIAMETER,BALL_DIAMETER,BALL_DIAMETER)

        self.ball_vel = [5, 5]

        self.create_bricks()
        

    def create_bricks(self):
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
        
    def check_input(self, input):
        boost = 5 if self.boost_time > 0 else 0

        if 'L' in input:
            self.paddle.left -= (7 + boost)
            if self.paddle.left < 0:
                self.paddle.left = 0

        if 'R' in input:
            self.paddle.left += (7 + boost)
            if self.paddle.left > MAX_PADDLE_X:
                self.paddle.left = MAX_PADDLE_X

        if 'B' in input and self.boosts_remaining > 0 and self.boost_time == 0:
            self.boosts_remaining -= 1
            self.boost_time += 25

        if 'sp' in input and self.state == STATE_BALL_IN_PADDLE:
            self.ball_vel = [5,5]
            self.speed_multiplyer = 1.0
            self.state = STATE_PLAYING
        elif 'ret' in input and (self.state == STATE_GAME_OVER or self.state == STATE_WON):
            self.init_game()

    def move_ball(self):
        self.ball.x += self.ball_vel[0] * self.speed_multiplyer
        self.ball.y  += self.ball_vel[1] * self.speed_multiplyer

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
        for brick in self.bricks:
            if self.ball.colliderect(brick):
                self.score += 3
                self.num_hits += 1
                self.ball_vel[1] = -self.ball_vel[1]
                self.bricks.remove(brick)
                self.speed_multiplyer = min(self.speed_multiplyer + 0.1, 2.5)
                break

        if len(self.bricks) == 0:
            self.state = STATE_WON
            
        if self.ball.colliderect(self.paddle):
            distance_from_center = float(self.ball.centerx - self.paddle.centerx)
            self.ball.top = PADDLE_Y - BALL_DIAMETER
            self.ball_vel[0] += distance_from_center / 7
            self.ball_vel[1] = -self.ball_vel[1]
        elif self.ball.top > self.paddle.top:
            self.lives -= 1
            if self.lives > 0:
                self.state = STATE_BALL_IN_PADDLE
            else:
                self.state = STATE_GAME_OVER

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
        

    @abc.abstractmethod
    def run(self):
        return


class HumanControlledBreakout(Breakout):
    """ TODO - move input recognizing logic, and run here """
    def __init__(self):
        super(HumanControlledBreakout, self).__init__()

    def _get_input(self):
        keys = pygame.key.get_pressed()
        input = []
        input += ['L'] if keys[pygame.K_LEFT] else []
        input += ['R'] if keys[pygame.K_RIGHT] else []
        input += ['B'] if keys[pygame.K_b] else []
        input += ['sp'] if keys[pygame.K_SPACE] else []
        input += ['ret'] if keys[pygame.K_RETURN] else []

        return input
        

    def run(self):
        while 1:            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit

            self.clock.tick(50)
            self.screen.fill(BLACK)
            self.check_input(self._get_input())

            if self.state == STATE_PLAYING:
                self.move_ball()
                self.handle_collisions()
            elif self.state == STATE_BALL_IN_PADDLE:
                self.ball.left = self.paddle.left + self.paddle.width / 2
                self.ball.top  = self.paddle.top - self.ball.height
                self.show_message("PRESS SPACE TO LAUNCH THE BALL")
                self.show_message("PRESS B TO BOOST", 0, 30)
            elif self.state == STATE_GAME_OVER:
                self.show_message("GAME OVER. PRESS ENTER TO PLAY AGAIN")
            elif self.state == STATE_WON:
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


class BotControlledBreakout(Breakout):
    """ TODO - create state vectors, don't have run draw stuff, give state vectors to learning agent (in another file""" 
    def __init__(self, agent, display):
        self.agent = agent
        self.display = display
        super(BotControlledBreakout, self).__init__()

    def run(self):
        while 1:            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit

            if self.display:
                self.clock.tick(50)
                self.screen.fill(BLACK)

                self._discretizeLocation(self.paddle.x, self.paddle.y)


#  TODO - THINK MORE ABOUT THIS...DISCRETIZE SPEED?
#            state_vector = {
#                "paddle_pos": self._discretizeLocation(self.paddle.x, self.paddle.y),
#                "ball_pos": self._discretizeLocation(self.ball.x, self.ball.y)
#                "ball_angle": self._discretizeAngle(self.ball_vel)
#                "boosts_remaining": self.boosts_left
#                }

            if self.state == STATE_PLAYING:
                self.move_ball()
                self.handle_collisions()
            elif self.state == STATE_BALL_IN_PADDLE:
                self.ball.left = self.paddle.left + self.paddle.width / 2
                self.ball.top  = self.paddle.top - self.ball.height
                self.show_message("PRESS SPACE TO LAUNCH THE BALL")
                self.show_message("PRESS B TO BOOST", 0, 30)
            elif self.state == STATE_GAME_OVER:
                self.show_message("GAME OVER. PRESS ENTER TO PLAY AGAIN")
            elif self.state == STATE_WON:
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
    

    def _discretizeLocation(self, x, y):
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
    def dotProduct(self, a, b):
        return sum(x * y for (x, y) in zip(a, b))

    def _magnitude(self, a):
        """ magnitude of vector """
        return math.sqrt(self.dotProduct(a, a))


    def _angle_between(self, a, b):
        """ angle between two vectors"""

        return math.degrees(math.acos(self.dotProduct(a, b) / (self._magnitude(a) * self._magnitude(b))))

    def _discretizeAngle(self, vec):
        """ 
        buckets the continuous angle of a vector into one of 16 discrete angle categories
        """
        return int(self._angle_between([1,0], vec) / 16)




if __name__ == "__main__":
#    HumanControlledBreakout().run()
    BotControlledBreakout(None, True).run()

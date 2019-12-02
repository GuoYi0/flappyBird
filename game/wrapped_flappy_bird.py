import numpy as np
import sys
import random
import pygame
from game import flappy_bird_utils
import pygame.surfarray as surfarray
from pygame.locals import *
from itertools import cycle
import sys
import os
import cv2

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
os.chdir(root_dir)
FPS = 60  # 为了加快训练，暂定大一点的值
SCREENWIDTH = 288  # 屏幕的宽度
SCREENHEIGHT = 512  # 屏幕的高度
pygame.init()  # 初始化
FPSCLOCK = pygame.time.Clock()
SCREEN = pygame.display.set_mode((SCREENWIDTH, SCREENHEIGHT))  # 定义一个窗口返回一个surface对象
pygame.display.set_caption('Flappy Bird')

IMAGES, SOUNDS, HITMASKS = flappy_bird_utils.load()
PIPEGAPSIZE = 100  # 上下管子的中间gap，鸟只能从这gap过去
BASEY = SCREENHEIGHT * 0.79  # 512*0.79 = 404，就是地面的y坐标

PLAYER_WIDTH = IMAGES['player'][0].get_width()  # 鸟的宽度34
PLAYER_HEIGHT = IMAGES['player'][0].get_height()  # 鸟的高度24
PIPE_WIDTH = IMAGES['pipe'][0].get_width()  # 管子的宽度52
PIPE_HEIGHT = IMAGES['pipe'][0].get_height()  # 管子的高度320
BACKGROUND_WIDTH = IMAGES['background'].get_width()  # 288，等于屏幕的宽度

PLAYER_INDEX_GEN = cycle([0, 1, 2, 1])  # 三张鸟图的索引，上中下中的循环


class GameState:
    def __init__(self):
        self.score = 0  # 得分，过一个管子就加一分
        self.playerIndex = 0  # 三张鸟图的索引，分别是上翅膀，中翅膀，下翅膀
        self.loopIter = 0  # 游戏进行的iter，循环值
        self.playerx = int(SCREENWIDTH * 0.2)  # 鸟的左上角点的x坐标
        self.playery = int((SCREENHEIGHT - PLAYER_HEIGHT) / 2)  # 鸟的左上角点的y坐标
        self.basex = 0  # 地面的x坐标
        self.baseShift = IMAGES['base'].get_width() - BACKGROUND_WIDTH  # 道路宽度336 - 屏幕宽度288 = 48

        newPipe1 = getRandomPipe()  # 一个列表，两个元素，表示上下两根管子的位置
        newPipe2 = getRandomPipe()

        # 管子的初始位置，初始情况下，有个在屏幕边缘，一个在外面
        self.upperPipes = [
            {'x': SCREENWIDTH, 'y': newPipe1[0]['y']},
            {'x': SCREENWIDTH + (SCREENWIDTH / 2), 'y': newPipe2[0]['y']},
        ]
        self.lowerPipes = [
            {'x': SCREENWIDTH, 'y': newPipe1[1]['y']},
            {'x': SCREENWIDTH + (SCREENWIDTH / 2), 'y': newPipe2[1]['y']},
        ]

        # 往右是正x，往下是正y
        self.pipeVelX = -4  # 管子的速度，每次往左边走4个像素， const
        self.playerVelY = 0  # 鸟的速度。初始值为0
        self.playerMaxVelY = 10  # 鸟掉落的最大速度
        self.playerMinVelY = -8  # 这个暂时不用
        self.playerAccY = 1  # 鸟下降的加速度
        self.playerFlapAcc = -7  # 按一下，就立即更新为这个速度
        self.playerFlapped = False  # 按一下，就是True

    def frame_step(self, input_actions):
        """
        :param input_actions: 输入动作，01表示按一下，鸟飞一下；或者10，表示不按
        :return:
        """
        pygame.event.pump()  # 内部处理pygame事件处理程序
        reward = 0.02  # 没有撞击也没死亡，默认奖励分值 这个值不能太小或者太大
        terminal = False  # 默认没结束

        if sum(input_actions) != 1:
            raise ValueError('Multiple input actions!')

        if input_actions[1] == 1:  # 按一下，飞起来
            if self.playery > -2 * PLAYER_HEIGHT:
                self.playerVelY = self.playerFlapAcc  # 按一下，速度就更新为self.playerVelY
                self.playerFlapped = True
                # SOUNDS['wing'].play()

        playerMidPos = self.playerx + PLAYER_WIDTH / 2  # 鸟的中心x坐标
        for pipe in self.upperPipes:
            if pipe['x'] + PIPE_WIDTH < self.playerx <= pipe['x'] + PIPE_WIDTH + 4:
                self.score += 1  # 过去了，分数加一
                # SOUNDS['point'].play()
                reward = 1  # 过一个管子，奖励就是1
            # pipeMidPos = pipe['x'] + PIPE_WIDTH / 2  # 上边管子中心x坐标
            # if pipeMidPos <= playerMidPos < pipeMidPos + 4:  # 过了一个管子。4就是管子的速度
            #     self.score += 1  # 过去了，分数加一
            #     # SOUNDS['point'].play()
            #     reward = 1  # 过一个管子，奖励就是1

        if (self.loopIter + 1) % 3 == 0:
            self.playerIndex = next(PLAYER_INDEX_GEN)  # 每张鸟图飞三次
        self.loopIter = (self.loopIter + 1) % 30
        self.basex = -((-self.basex + 100) % self.baseShift)  # 地面的x坐标

        if not self.playerFlapped and self.playerVelY < self.playerMaxVelY:  # 没有选择按，并且有加速空间
            self.playerVelY += self.playerAccY
        if self.playerFlapped:
            self.playerFlapped = False
        self.playery += min(self.playerVelY, BASEY - self.playery - PLAYER_HEIGHT)  # 不能到地下去了
        # self.playery += self.playerVelY  # 可以掉地下去，后面会判死亡
        if self.playery < 0:
            self.playery = 0

        # 管子移动
        for uPipe, lPipe in zip(self.upperPipes, self.lowerPipes):
            uPipe['x'] += self.pipeVelX
            lPipe['x'] += self.pipeVelX

        # 当管子的坐标快接近0的是，来一个新管子
        if 0 < self.upperPipes[0]['x'] < 5:
            newPipe = getRandomPipe()  # 生成一个新管子
            self.upperPipes.append(newPipe[0])
            self.lowerPipes.append(newPipe[1])

        # 当管子的x坐标已经是 -PIPE_WIDTH的时候，就移除队列
        if self.upperPipes[0]['x'] < -PIPE_WIDTH:
            self.upperPipes.pop(0)
            self.lowerPipes.pop(0)

        # check if crash here
        isCrash = checkCrash({'x': self.playerx, 'y': self.playery,
                              'index': self.playerIndex},
                             self.upperPipes, self.lowerPipes)
        if isCrash:  # 撞了
            # SOUNDS['hit'].play()
            # SOUNDS['die'].play()
            terminal = True  # 结束
            self.__init__()  # 一切重来
            reward = -1  # 奖励为-1

        # # 画图
        SCREEN.blit(IMAGES['background'], (0, 0))  # source, dest, 把背景画在 (0,0)位置

        for uPipe, lPipe in zip(self.upperPipes, self.lowerPipes):
            SCREEN.blit(IMAGES['pipe'][0], (uPipe['x'], uPipe['y']))
            SCREEN.blit(IMAGES['pipe'][1], (lPipe['x'], lPipe['y']))

        SCREEN.blit(IMAGES['base'], (self.basex, BASEY))
        # print score so player overlaps the score
        # showScore(self.score)
        SCREEN.blit(IMAGES['player'][self.playerIndex],
                    (self.playerx, self.playery))

        image_data = pygame.surfarray.array3d(pygame.display.get_surface())
        pygame.display.update()
        FPSCLOCK.tick(FPS)

        image_data = np.rot90(image_data[:, :, ::-1], k=3)[:, ::-1, :]  # RGB => BGR 旋转翻转，得到可视化正常的图片

        return image_data, reward, terminal


def getRandomPipe():
    """returns a randomly generated pipe"""
    # y of gap between upper and lower pipe
    gapYs = [20, 30, 40, 50, 60, 70, 80, 90]
    index = random.randint(0, len(gapYs) - 1)
    gapY = gapYs[index]

    gapY += int(BASEY * 0.2)
    pipeX = SCREENWIDTH + 10

    return [
        {'x': pipeX, 'y': gapY - PIPE_HEIGHT},  # upper pipe
        {'x': pipeX, 'y': gapY + PIPEGAPSIZE},  # lower pipe
    ]


def showScore(score):
    """displays score in center of screen"""
    scoreDigits = [int(x) for x in list(str(score))]
    totalWidth = 0  # total width of all numbers to be printed

    for digit in scoreDigits:
        totalWidth += IMAGES['numbers'][digit].get_width()

    Xoffset = (SCREENWIDTH - totalWidth) / 2

    for digit in scoreDigits:
        SCREEN.blit(IMAGES['numbers'][digit], (Xoffset, SCREENHEIGHT * 0.1))
        Xoffset += IMAGES['numbers'][digit].get_width()


def checkCrash(player, upperPipes, lowerPipes):
    """
    撞地或者撞任意管子，就是true
    :param player:
    :param upperPipes:
    :param lowerPipes:
    :return:
    """
    pi = player['index']
    player['w'] = IMAGES['player'][0].get_width()
    player['h'] = IMAGES['player'][0].get_height()

    if player['y'] + player['h'] >= BASEY - 1:  # 撞地了
        return True
    else:
        playerRect = pygame.Rect(player['x'], player['y'],
                                 player['w'], player['h'])  # 鸟的矩形框

        for uPipe, lPipe in zip(upperPipes, lowerPipes):
            # upper and lower pipe rects
            uPipeRect = pygame.Rect(uPipe['x'], uPipe['y'], PIPE_WIDTH, PIPE_HEIGHT)  # 上管子矩形框
            lPipeRect = pygame.Rect(lPipe['x'], lPipe['y'], PIPE_WIDTH, PIPE_HEIGHT)  # 下管子矩形框

            # player and upper/lower pipe hitmasks
            pHitMask = HITMASKS['player'][pi]  # 鸟的mask
            uHitmask = HITMASKS['pipe'][0]  # 上管子的mask
            lHitmask = HITMASKS['pipe'][1]  # 下管子的mask

            # if bird collided with upipe or lpipe
            uCollide = pixelCollision(playerRect, uPipeRect, pHitMask, uHitmask)
            lCollide = pixelCollision(playerRect, lPipeRect, pHitMask, lHitmask)
            if uCollide or lCollide:
                return True
    return False


def pixelCollision(rect1, rect2, hitmask1, hitmask2):
    """根据mask来判断是否相撞"""
    rect = rect1.clip(rect2)

    if rect.width == 0 or rect.height == 0:  # 两个矩形框没交集，肯定没撞
        return False

    x1, y1 = rect.x - rect1.x, rect.y - rect1.y
    x2, y2 = rect.x - rect2.x, rect.y - rect2.y

    for x in range(rect.width):
        for y in range(rect.height):
            if hitmask1[x1 + x][y1 + y] and hitmask2[x2 + x][y2 + y]:
                return True
    return False

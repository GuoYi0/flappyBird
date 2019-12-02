import tensorflow as tf
import game.wrapped_flappy_bird as game
from collections import deque
import numpy as np
import cv2
import random
from model import createNetwork
import os
import imageio
NUM_ACTIONS = 2  # 两个动作
GAME = 'bird'  # 游戏名字
num_images = 400


def runNetwork(inputs, outputs):
    game_state = game.GameState()
    do_nothing = np.zeros(NUM_ACTIONS)
    do_nothing[0] = 1
    xt, rt, terminal = game_state.frame_step(do_nothing)  # 初始化，t时刻的画面，奖励，是否终结
    cv2.imwrite("images/0.png", xt)
    xt = cv2.cvtColor(cv2.resize(xt, (72, 128)), cv2.COLOR_BGR2GRAY)  # shape (128, 72, 1)
    ret, xt = cv2.threshold(xt, 1, 255, cv2.THRESH_BINARY)  # 二值化
    s0 = np.stack([xt, xt, xt, xt], axis=2)
    saver = tf.train.Saver(max_to_keep=100)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    checkpoint = tf.train.get_checkpoint_state("ckpt")
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
    else:
        raise FileNotFoundError("Could not find old network weights")
    t = 1
    while t <= num_images:
        readout = sess.run(outputs, feed_dict={inputs: s0[np.newaxis, ...]})[0]  # 四帧画面丢进去
        action = np.zeros(NUM_ACTIONS)
        action_index = np.argmax(readout)
        action[action_index] = 1
        x1, r1, terminal = game_state.frame_step(action)
        cv2.imwrite("images/%d.png" % t, x1)
        x1 = cv2.cvtColor(cv2.resize(x1, (72, 128)), cv2.COLOR_BGR2GRAY)
        ret, x1 = cv2.threshold(x1, 1, 255, cv2.THRESH_BINARY)  # 二值化为0或者255
        x1 = np.reshape(x1, (128, 72, 1))
        s1 = np.concatenate([x1, s0[:, :, :3]], axis=2)  # 丢弃遥远的一帧，加入新帧
        s0 = s1
        t += 1
    sess.close()


def playGame(restore):
    inputs, outputs = createNetwork(NUM_ACTIONS)
    runNetwork(inputs, outputs)  # 输入是连续的四帧画面，输出是价值函数


def toGif(path, name=''):
    def func(key):
        index = int(key.split('.')[0])
        return index
    file_list = os.listdir(path)
    file_list = sorted(file_list, key=func)
    frames = []
    print(file_list)
    for png in file_list:
        frames.append(imageio.imread(os.path.join(path, png)))
    imageio.mimsave("result.gif", frames, 'GIF', duration=0.03)


def main():
    playGame(restore=True)


if __name__ == '__main__':
    # main()
    toGif("images",)



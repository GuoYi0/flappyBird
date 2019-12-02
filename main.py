import tensorflow as tf
import game.wrapped_flappy_bird as game
from collections import deque
import numpy as np
import cv2
import random
from model import createNetwork
import os


NUM_ACTIONS = 2  # 两个动作
GAME = 'bird'  # 游戏名字
GAMMA = 0.95  #
OBSERVE = 10000.  # 先观察这么多个step，产生一些训练数据
EXPLORE = 200000.  # 然后开始探索
FINAL_EPSILON = 0.01  # final value of epsilon
INITIAL_EPSILON = 0.01  # starting value of epsilon
REPLAY_MEMORY = 20000
BATCH = 64
FRAME_PER_ACTION = 1  # 每隔多少帧有新动作


def trainNetwork(inputs, outputs, restore):
    #                                       # inputs 是当前局面
    a = tf.placeholder(tf.float32, [None, NUM_ACTIONS])  # 在当前局面基础上采取的动作，随机采取的或者取最大值采取的，取值01或者10
    y = tf.placeholder(tf.float32, [None])  # 价值函数， 根据该动作得到的分数，加上新局面的预测
    #                                       # outputs当前局面输入进网络得到的结果
    readout_action = tf.reduce_sum(outputs * a, axis=1)
    cost = tf.reduce_mean(tf.square(y - readout_action))
    l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()]) * 1e-7
    total_loss = cost + l2_loss
    opt = tf.train.AdamOptimizer(5e-5).minimize(total_loss)
    game_state = game.GameState()
    D = deque()
    do_nothing = np.zeros(NUM_ACTIONS)
    do_nothing[0] = 1
    xt, rt, terminal = game_state.frame_step(do_nothing)  # 初始化，t时刻的画面，奖励，是否终结
    # xt shape (512, 288, 3)
    xt = cv2.cvtColor(cv2.resize(xt, (72, 128)), cv2.COLOR_BGR2GRAY)  # shape (128, 72, 1)
    ret, xt = cv2.threshold(xt, 1, 255, cv2.THRESH_BINARY)  # 二值化
    s0 = np.stack([xt, xt, xt, xt], axis=2)
    # global_step = tf.train.get_or_create_global_step()
    saver = tf.train.Saver(max_to_keep=100)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    step = 0
    if restore:
        checkpoint = tf.train.get_checkpoint_state("ckpt")
        if checkpoint and checkpoint.model_checkpoint_path:
            stem = os.path.splitext(os.path.basename(checkpoint.model_checkpoint_path))[0]
            step = int(stem.split('-')[-1])
            saver.restore(sess, checkpoint.model_checkpoint_path)
            # try:
            #     sess.run(tf.assign(global_step, step))
            #
            # print("Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
            raise FileNotFoundError("Could not find old network weights")

    epsilon = INITIAL_EPSILON
    t = step
    global_step = step
    t = 0
    while t <= 300000:
        readout = sess.run(outputs, feed_dict={inputs: s0[np.newaxis, ...]})[0]  # 四帧画面丢进去
        action = np.zeros(NUM_ACTIONS)
        action_index = 0
        if t % FRAME_PER_ACTION == 0:
            if random.random() <= epsilon:
                action_index = np.random.choice(NUM_ACTIONS)
            else:
                action_index = np.argmax(readout)
        action[action_index] = 1
        if epsilon > FINAL_EPSILON and t > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        x1, r1, terminal = game_state.frame_step(action)
        x1 = cv2.cvtColor(cv2.resize(x1, (72, 128)), cv2.COLOR_BGR2GRAY)
        ret, x1 = cv2.threshold(x1, 1, 255, cv2.THRESH_BINARY)  # 二值化为0或者255
        x1 = np.reshape(x1, (128, 72, 1))
        s1 = np.concatenate([x1, s0[:, :, :3]], axis=2)  # 丢弃遥远的一帧，加入新帧
        if terminal or random.random() <= 0.6:  # 后期重点训练死亡了的
            D.append((s0, action, r1, s1, terminal))  # 一个局面，该局面产生的动作（随机或者预测），执行该动作的奖励和新局面，新局面是否终结
            if len(D) > REPLAY_MEMORY:
                D.popleft()

        if t > OBSERVE:
            minibatch = random.sample(D, BATCH)
            s_batch = [d[0] for d in minibatch]  # 0时刻局面
            a_batch = [d[1] for d in minibatch]  # 0时刻局面所执行的动作
            r_batch = [d[2] for d in minibatch]  # 该动作引起的奖励
            s2_batch = [d[3] for d in minibatch]  # 该动作引起的新局面

            y_batch = []  # 收集的是
            readout2_batch = sess.run(outputs, feed_dict={inputs: s2_batch})  # (bs, 2)
            for i in range(0, len(minibatch)):
                terminal = minibatch[i][4]
                if terminal:  # 已经撞死了，就不会把新局面输入进去了
                    y_batch.append(r_batch[i])
                else:
                    y_batch.append(r_batch[i] + GAMMA * np.max(readout2_batch[i]))
            _ = sess.run(opt, feed_dict={y: y_batch, a: a_batch, inputs: s_batch})

        s0 = s1
        t += 1
        if t % 5000 == 0:
            saver.save(sess, 'ckpt/' + GAME + '-dqn', global_step=global_step)
        if t <= OBSERVE:
            state = "observe"
        elif OBSERVE <= t <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"
        if t % 50 == 0 or r1 == 1 or r1 == -1:
            print("TIMESTEP", t, "/ STATE", state, "/ EPSILON %0.3f" % epsilon, "/ ACTION",
                  action_index, "/ REWARD", r1, "/ Q_value: (%0.3f, %0.3f)" % (readout[0], readout[1]))

    sess.close()


def playGame(restore):
    inputs, outputs = createNetwork(NUM_ACTIONS)
    trainNetwork(inputs, outputs, restore=restore)  # 输入是连续的四帧画面，输出是价值函数


def main():
    playGame(restore=True)


if __name__ == '__main__':
    main()

import tensorflow as tf
import numpy as np
import gym
import threading
import matplotlib.pyplot as plt

display = False
SESS = tf.Session()
Episodes = 200
gamma=0.99
global_name = 'GLOBAL_NET'
global_name2 = 'GLOBAL_NET2'

class A3CWorker:
    def __init__(self, name, gAC=None, lamb=1e-5, actor_lr=1e-3, critics_lr=1e-3):
        if name == global_name or global_name2:
            self.episodes = 0
            self.stats = []
            self.c_optimizer = tf.train.AdamOptimizer(critics_lr)
            self.a_optimizer = tf.train.AdamOptimizer(actor_lr)
            if name == global_name2:
                self.replay_size = 4000   # replay大小
                self.cur_size = 0   # 表示当前replay大小
                self.cur_point = 0  # 下一个replay存的位置
                # 单个元素格式：[state,action,R]
                self.replay = []
        self.name = name
        self.gAC = gAC
        self.lamb = lamb
        self.actor_lr = actor_lr
        self.critics_lr = critics_lr
        # 非中心网络加载环境
        if name != global_name and name != global_name2:
            self.env = gym.make("MountainCarContinuous-v0").unwrapped
        with tf.variable_scope(name):
            self._build_model()
        self.a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name + '/actor')
        self.c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name + '/critics')
        # 非中心网络可以有pull 和 push 功能
        if name != global_name and name != global_name2:
            self.pull_a_params_op = [l.assign(g) for l, g in zip(self.a_params, self.gAC.a_params)]
            self.pull_c_params_op = [l.assign(g) for l, g in zip(self.c_params, self.gAC.c_params)]
            self.a_grads = tf.gradients(self.a_loss, self.a_params)
            self.c_grads = tf.gradients(self.c_loss, self.c_params)
            self.update_a_op = self.gAC.a_optimizer.apply_gradients(zip(self.a_grads, self.gAC.a_params))
            self.update_c_op = self.gAC.c_optimizer.apply_gradients(zip(self.c_grads, self.gAC.c_params))

    def _build_model(self):
        # input
        self.state = tf.placeholder(tf.float32, [None, 2], name="state")
        self.target = tf.placeholder(tf.float32, [None, 1], name="target")
        self.action_train = tf.placeholder(tf.float32, [None, 1], name="action_train")

        # build critics net
        with tf.variable_scope('critics'):
            f1 = tf.contrib.layers.fully_connected(
                inputs=self.state,
                num_outputs=200,
                activation_fn=tf.nn.relu6,
                weights_initializer=tf.contrib.layers.xavier_initializer()
            )
            self.critics = tf.contrib.layers.fully_connected(
                inputs=f1,
                num_outputs=1,
                activation_fn=None,
                weights_initializer=tf.contrib.layers.xavier_initializer()
            )
            # # 此时为(3,)
            # self.critics = tf.squeeze(self.critics)
            # # 变成(3,1)
            # self.critics = tf.expand_dims(self.critics, axis=-1)

        # build actor net
        with tf.variable_scope('actor'):
            f1 = tf.contrib.layers.fully_connected(
                inputs=self.state,
                num_outputs=200,
                activation_fn=tf.nn.relu6,
                weights_initializer=tf.contrib.layers.xavier_initializer()
            )
            self.mu = tf.contrib.layers.fully_connected(
                inputs=f1,
                num_outputs=1,
                activation_fn=None,
                weights_initializer=tf.contrib.layers.xavier_initializer()
            )
            # 此时mu为1*1 矩阵 其维度应变为1
            # mu = tf.squeeze(mu)
            # self.mu = tf.expand_dims(mu, axis=-1)

            self.sigma = tf.contrib.layers.fully_connected(
                inputs=f1,
                num_outputs=1,
                activation_fn=tf.nn.softplus,
                weights_initializer=tf.contrib.layers.xavier_initializer()
            )

            self.norm_dist = tf.contrib.distributions.Normal(self.mu, self.sigma)
            action = self.norm_dist.sample(1)
            # MoutainCarContinuous环境中 action 最低为-1  最高为1
            self.action = tf.clip_by_value(action, [[[-1]]], [[[1]]])

        # critics op
        self.c_loss = tf.reduce_mean(tf.squared_difference(self.critics, self.target))

        # actor op
        self.advantage = tf.subtract(self.target, self.critics)
        import math
        self.entropy = -1 / 2 * (tf.log(2 * math.pi * self.sigma) + 1)
        self.a_loss = -tf.log(
            self.norm_dist.prob(self.action_train) + 1e-5) * self.advantage - self.lamb * self.entropy

    # 查看某个状态的value
    def predict_value(self, state):
        a = SESS.run(self.critics, feed_dict={self.state: [state]})
        return a[0][0]

    # 选择动作
    def choose_action(self, state):
        a = SESS.run(self.action, feed_dict={self.state: [state]})
        return a[0][0]

    def work(self):
        while self.gAC.episodes < Episodes:
            # 每个回合开始处
            state = self.env.reset()
            reward_total = 0
            step = 0   #记录该回合进行的步数  若超过3000,则判定结束
            while step < 3000:
                buffer_s = []
                buffer_r = []
                buffer_a = []
                self.pull()
                for i in range(3):
                    buffer_s.append(state)
                    action = self.choose_action(state)
                    buffer_a.append(action)
                    state, reward, done, _ = self.env.step(action)
                    if display and self.name == 'W_0':
                        self.env.render()
                    step += 1
                    if state[1] < 0:
                        reward1 = -action[0]
                    else:
                        reward1 = action[0]
                    buffer_r.append(reward1)

                    reward_total += reward
                    if done:
                        break

                if done:
                    R = 0
                else:
                    R = self.predict_value(state)
                V_target = []
                for i in range(len(buffer_r) - 1, -1, -1):
                    R = buffer_r[i] + gamma * R
                    V_target.append(R)
                V_target.reverse()

                if self.gAC.name == global_name2:
                    for i in range(len(V_target)):
                        item = []
                        item.append(buffer_s[i])
                        item.append(buffer_a[i])
                        item.append(V_target[i])
                        self.add_to_replay(item)
                self.push(buffer_s, buffer_a, V_target)
                if self.gAC.name == global_name2 and step % 30 == 0:
                    self.train_replay()
                if done:
                    done = False
                    break

            self.gAC.stats.append(reward_total)
            self.gAC.episodes += 1
            # if np.mean(self.gAC.stats[-100:]) > 90 and len(self.gAC.stats) >= 101:
            #     print(np.mean(self.gAC.stats[-100:]))
            #     print("Solved")
            print("Episode: {}, name: {}, reward: {}, average:{}, step:{}.".format(self.gAC.episodes, self.gAC.name, reward_total, np.mean(self.gAC.stats[-100:]), step))

    # 将中心部分的参数赋值给线程
    def pull(self):
        SESS.run([self.pull_a_params_op, self.pull_c_params_op])

    # 即用线程学到的经验，更新中心的参数
    def push(self, state, action, R):

        feed_dict = {
            self.state: np.vstack(state),
            self.action_train: np.vstack(action),
            self.target: np.vstack(R)
        }
        # print(SESS.run(self.critics, feed_dict))    output 3*1
        # print(SESS.run(self.a_loss, feed_dict))    output 3*1
        SESS.run([self.update_a_op, self.update_c_op], feed_dict=feed_dict)

    def add_to_replay(self, item):
        if self.gAC.cur_size < self.gAC.replay_size:
            self.gAC.replay.append(item)
            self.gAC.cur_size += 1
        else:
            self.gAC.replay[self.gAC.cur_point] = item
            self.gAC.cur_point += 1
            self.gAC.cur_point %= self.gAC.replay_size

    def train_replay(self):
        if self.gAC.cur_size < 3:
            return
        import random
        k = random.sample(range(self.gAC.cur_size), 3)
        state = []
        action = []
        R = []
        for i in k:
            ban = self.gAC.replay[i]
            state.append(ban[0])
            action.append(ban[1])
            R.append(ban[2])
        feed_dict = {
            self.state: np.vstack(state),
            self.action_train: np.array(action),
            self.target: np.vstack(R)
        }
        SESS.run(self.update_c_op, feed_dict=feed_dict)


if __name__ == "__main__":
    actor_lr, critics_lr, lamb, gamma = [0.0001, 0.00046415888336127773, 2.782559402207126e-05, 0.99]
    actor_lr, critics_lr, lamb, gamma = [0.00001, 0.00001, 0.00001, 0.99]
    actor_lr, critics_lr, lamb, gamma = [0.0001, 0.0001, 0.00001, 0.99]
    GLOBAL_AC = A3CWorker(global_name, lamb=lamb, actor_lr=actor_lr, critics_lr=critics_lr)
    GLOBAL_AC2 = A3CWorker(global_name2, lamb=lamb, actor_lr=actor_lr, critics_lr=critics_lr)
    workers = []
    for i in range(4):
        i_name = 'W_%i' % i  # worker name
        if i < 2:
            workers.append(A3CWorker(i_name, gAC=GLOBAL_AC, lamb=lamb, actor_lr=actor_lr, critics_lr=critics_lr))
        else:
            workers.append(A3CWorker(i_name, gAC=GLOBAL_AC2, lamb=lamb, actor_lr=actor_lr, critics_lr=critics_lr))
    COORD = tf.train.Coordinator()
    SESS.run(tf.global_variables_initializer())
    # 使得两个中心初始化参数一致
    pull_p_params_op = [l_p.assign(g_p) for l_p, g_p in zip(GLOBAL_AC2.a_params, GLOBAL_AC.a_params)]
    pull_v_params_op = [l_p.assign(g_p) for l_p, g_p in zip(GLOBAL_AC2.c_params, GLOBAL_AC.c_params)]
    SESS.run([pull_p_params_op, pull_v_params_op])

    worker_threads = []
    for worker in workers:
        job = lambda: worker.work()
        t = threading.Thread(target=job)
        t.start()
        worker_threads.append(t)
    COORD.join(worker_threads)

    print(GLOBAL_AC.stats)
    list1 = []
    list2 = []
    R = 0
    for i in range(len(GLOBAL_AC.stats)):
        R = 0.9*R + 0.1*GLOBAL_AC.stats[i]
        list1.append(R)
    R = 0
    for i in range(len(GLOBAL_AC2.stats)):
        R = 0.9 * R + 0.1 * GLOBAL_AC2.stats[i]
        list2.append(R)
    plt.plot(np.arange(len(list1)), list1, color='blue', label='normal')
    plt.plot(np.arange(len(list2)), list2, color='red', label='improved')
    plt.legend(loc='best')
    plt.xlabel('epidose')
    plt.ylabel('reward')
    plt.title('MountaiCarContinuous-v0')
    plt.show()





import tensorflow as tf
import numpy as np
import gym
import threading
import matplotlib.pyplot as plt

display = False
Episodes = 2000
gamma=0.99
global_name = 'GLOBAL_NET'
model_path = './save_para/Lunar.ckpt'
is_tarin = False
class A3CWorker:
    def __init__(self, name, sess=None, gAC=None, lamb=1e-5, actor_lr=1e-3, critics_lr=1e-3):
        if name == global_name:
            self.episodes = 0
            self.stats = []
            self.c_optimizer = tf.train.AdamOptimizer(critics_lr)
            self.a_optimizer = tf.train.AdamOptimizer(actor_lr)
        self.sess = sess
        self.name = name
        self.gAC = gAC
        self.lamb = lamb
        self.actor_lr = actor_lr
        self.critics_lr = critics_lr
        if name != global_name:
            self.env = gym.make("LunarLanderContinuous-v2").unwrapped
        with tf.variable_scope(name):
            self._build_model()
        self.a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name + '/actor')
        self.c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name + '/critics')
        # 非中心网络可以有pull 和 push 功能
        if name != global_name:
            self.pull_a_params_op = [l.assign(g) for l, g in zip(self.a_params, self.gAC.a_params)]
            self.pull_c_params_op = [l.assign(g) for l, g in zip(self.c_params, self.gAC.c_params)]
            self.a_grads = tf.gradients(self.a_loss, self.a_params)
            self.c_grads = tf.gradients(self.c_loss, self.c_params)
            self.update_a_op = self.gAC.a_optimizer.apply_gradients(zip(self.a_grads, self.gAC.a_params))
            self.update_c_op = self.gAC.c_optimizer.apply_gradients(zip(self.c_grads, self.gAC.c_params))

    def _build_model(self):
        # input
        self.state = tf.placeholder(tf.float32, [None, 8], name="state")
        self.target = tf.placeholder(tf.float32, [None, 1], name="target")
        self.action_train = tf.placeholder(tf.float32, [None, 2], name="action_train")

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
                num_outputs=2,
                activation_fn=None,
                weights_initializer=tf.contrib.layers.xavier_initializer()
            )
            # 此时mu为1*1 矩阵 其维度应变为1
            # mu = tf.squeeze(mu)
            # self.mu = tf.expand_dims(mu, axis=-1)

            self.sigma = tf.contrib.layers.fully_connected(
                inputs=f1,
                num_outputs=2,
                activation_fn=tf.nn.softplus,
                weights_initializer=tf.contrib.layers.xavier_initializer()
            )

            self.norm_dist = tf.contrib.distributions.Normal(self.mu, self.sigma)
            action = self.norm_dist.sample(1)
            # LunarLanderContinuous环境中 action 最低为[-1,-1]  最高为[1,1]
            self.action = tf.clip_by_value(action, [-1, -1], [1, 1])

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
        a = self.sess.run(self.critics, feed_dict={self.state: [state]})
        return a[0][0]

    # 选择动作
    def choose_action(self, state):
        a = self.sess.run(self.action, feed_dict={self.state: [state]})
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
                # Tmax 步　此处为3
                for i in range(3):
                    buffer_s.append(state)
                    action = self.choose_action(state)
                    buffer_a.append(action)
                    state_his = state
                    state, reward, done, _ = self.env.step(action)
                    if display and self.name == 'W_0':
                        self.env.render()
                    step += 1

                    import math
                    reward1 = (math.fabs(state_his[0]) - math.fabs(state[0]))
                    reward1 += (math.fabs(state_his[1]) - math.fabs(state[1]))
                    reward1 += (state_his[2] ** 2 + state_his[3]**2) - (state[2]**2+state[3]**2)
                    reward1 += (math.fabs(state_his[4]) - math.fabs(state[4]))
                    buffer_r.append(reward1)

                    reward_total += reward
                    if done:
                        break
                # 计算 Target
                if done:
                    R = 0
                else:
                    R = self.predict_value(state)
                V_target = []
                for i in range(len(buffer_r) - 1, -1, -1):
                    R = buffer_r[i] + gamma * R
                    V_target.append(R)
                V_target.reverse()

                self.push(buffer_s, buffer_a, V_target)
                if done:
                    done = False
                    break

            self.gAC.stats.append(reward_total)
            self.gAC.episodes += 1
            if np.mean(self.gAC.stats[-50:]) > 200 and len(self.gAC.stats) >= 101:
                print('successful train')
                return
            print("Episode: {}, reward: {}, average:{}, step:{}.".format(self.gAC.episodes, reward_total, np.mean(self.gAC.stats[-100:]), step))

    # 将中心部分的参数赋值给线程
    def pull(self):
        self.sess.run([self.pull_a_params_op, self.pull_c_params_op])

    # 即用线程学到的经验，更新中心的参数
    def push(self, state, action, R):
        feed_dict = {
            self.state: np.vstack(state),
            self.action_train: np.vstack(action),
            self.target: np.vstack(R)
        }
        self.sess.run([self.update_a_op, self.update_c_op], feed_dict=feed_dict)



if __name__ == "__main__":
    actor_lr, critics_lr, lamb, gamma = [0.0001, 0.0001, 0.00001, 0.99]

    if is_tarin:
        with tf.Session() as sess:
            GLOBAL_AC = A3CWorker(global_name, sess=sess, lamb=lamb, actor_lr=actor_lr, critics_lr=critics_lr)

            workers = []
            for i in range(2):
                i_name = 'W_%i' % i  # worker name
                workers.append(A3CWorker(i_name, sess=sess, gAC=GLOBAL_AC, lamb=lamb, actor_lr=actor_lr, critics_lr=critics_lr))
            COORD = tf.train.Coordinator()
            saver = tf.train.Saver()
            sess.run(tf.global_variables_initializer())

            worker_threads = []
            for worker in workers:
                job = lambda: worker.work()
                t = threading.Thread(target=job)
                t.start()
                worker_threads.append(t)
            COORD.join(worker_threads)
            save_path = saver.save(sess, model_path)
        print(GLOBAL_AC.stats)
        plt.plot(np.arange(len(GLOBAL_AC.stats)), GLOBAL_AC.stats, color='blue', label='normal')
        plt.legend(loc='best')
        plt.xlabel('epidose')
        plt.ylabel('reward')
        plt.title('MountaiCarContinuous-v0')
        plt.show()
    else:
        with tf.Session() as sess:
            GLOBAL_AC = A3CWorker(global_name, sess=sess, lamb=lamb, actor_lr=actor_lr, critics_lr=critics_lr)

            workers = []
            for i in range(2):
                i_name = 'W_%i' % i  # worker name
                workers.append(A3CWorker(i_name, sess=sess, gAC=GLOBAL_AC, lamb=lamb, actor_lr=actor_lr, critics_lr=critics_lr))
            saver = tf.train.Saver()
            saver.restore(sess, model_path)
            env = gym.make("LunarLanderContinuous-v2").unwrapped
            stats = []
            for i in range(100):
                state = env.reset()
                done = False
                reward_total = 0
                for step in range(1000):
                    env.render()
                    action = GLOBAL_AC.choose_action(state)
                    state, reward, done, _ = env.step(action)
                    reward_total += reward
                    if done:
                        break
                import time
                time.sleep(0.5)
                print(reward_total)
                stats.append(reward_total)
            print('100回合的平均reward为：', np.mean(stats))







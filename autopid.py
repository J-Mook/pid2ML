import matplotlib.pyplot as plt
import math
import shutil, os

from numpy import average

goal_loss = 1.0

class Robot: # 가상의 로봇이 라인트레싱을 한다고 해봅시다.
    def __init__(self, pose):
        self.current_pose = pose
        self.angular_velocity = 0

    def take_action(self, action):
        self.current_pose += 0.1 * action

class heater:
    def __init__(self, pose):
        self.current_pose = pose
        self.angular_velocity = 0

    def take_action(self, power):
        action = 0
        if power > 0:         
            action += 2 * power
        action -= 0.5
        self.current_pose += 0.1 * action

class mass_spring_damper:
    def __init__(self, pose):
        self.current_pose = pose
        self.velocity = 0
        self.k = 0.2
        self.b = 0.2
        self.mass = 1

    def take_action(self, power):
        spring_force = self.k * self.current_pose # Fs = k * x
        damper_force = self.b * self.velocity # Fb = b * x'
        self.velocity += (power - (spring_force + damper_force)) / self.mass * 0.5 # Integral(a) = v
        self.current_pose += self.velocity * 0.5 # Integral(v) = x

class PID:
    err_sum = 0
    old_err = 0

    def pid(self, cur_pose, goal_pose, kp, ki, kd):
        err = goal_pose - cur_pose
        self.err_sum += err
        delta_err = err - self.old_err
        self.old_err = err

        return kp*err + ki*self.err_sum + kd*delta_err

class Derivative: #미분 클래스
    def __init__(self):
        self.last_x = 0
        self.last_y = 0

    def get_gradient(self, x, y):
        d = (y - self.last_y) / (x - self.last_x)
        self.last_x = x
        self.last_y = y
        return d

start_postion = 0
goal_position = 1

class Train:
    kp = 1.0; ki = 0.1; kd = 0.5
    episode_length = 60
    learning_rate = 0.0001

    def __init__(self):
        self.dp = Derivative()
        self.di = Derivative()
        self.dd = Derivative()
        self.step = 0
        self.last_loss = 0

    def loss_func_mse(self, list):
        sum = 0
        for i in list:
            sum += abs(i)
        return sum / len(list)
    def loss_func_mae(self, list):
        sum = 0
        for i in list:
            sum += i*i
        # return sum / 2
        return sum / len(list) 

    def loss(self, draw_graph=False):
        robot = mass_spring_damper(start_postion)
        pid = PID()

        pose = []
        error = []
        for i in range(self.episode_length): # 로봇의 오차를 축적
            pose.append(robot.current_pose)
            error.append(robot.current_pose - goal_position)
            robot.take_action(pid.pid(robot.current_pose, goal_position, self.kp, self.ki, self.kd))

        if draw_graph and self.step % 100 == 0:  # draw graph and self.step > 1950
            plt.close()
            plt.plot(pose)
            plt.title("step={}, p={:.3}, i={:.3}, d={:.3}, train_loss={:.3}".format(self.step, self.kp, self.ki, self.kd, self.loss_func_mae(pose)))
            # plt.ylim(-goal_position, goal_position)
            plt.ylim(start_postion, goal_position * 1.3)
            plt.grid()
            plt.savefig("graph/{}.jpg".format(self.step))

        return self.loss_func_mae(pose)

    def optimize(self):
        self.kp = self.kp - self.learning_rate * self.dp.get_gradient(self.kp, self.loss())
        self.ki = self.ki - self.learning_rate * self.di.get_gradient(self.ki, self.loss())
        self.kd = self.kd - self.learning_rate * self.dd.get_gradient(self.kd, self.loss(draw_graph=True))
        self.last_loss = self.dd.last_y
        print("step={}, kp={}, ki={}, kd={}, loss={}".format(self.step, self.kp, self.ki, self.kd, self.last_loss))
        self.step += 1

if __name__ == '__main__':
    loss = []
    kp = []
    ki = []
    kd = []

    train = Train()

    shutil.rmtree("./graph")
    os.makedirs("./graph")
    for j in range(100000):
        if (j > 0 and (math.isnan(kp[-1]) or math.isnan(ki[-1]) or math.isnan(kd[-1]))):
            break
        train.optimize()
        loss.append(train.last_loss)
        kp.append(train.kp)
        ki.append(train.ki)
        kd.append(train.kd)
        # if (j > 500 and min(loss[:-500]) == min(loss)): #and loss[-3] > min(loss) and loss[-2] > min(loss) 
        #     break
    print("step = {}, kp={}, ki={}, kd={}, loss={}".format(len(loss), kp[-1], ki[-1], kd[-1], loss[-1]))

    ideal_kp = train.kp
    ideal_ki = train.ki
    ideal_kd = train.kd

    plt.close()
    plt.plot(loss, label="loss")
    plt.plot(kp, label="Kp")
    plt.plot(ki, label="Ki")
    plt.plot(kd, label="Kd")
    plt.legend()
    plt.title("Train")
    plt.savefig("train.png")
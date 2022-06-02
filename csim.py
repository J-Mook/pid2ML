import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm
import pandas as pd
import seaborn as sns
import tqdm 
stable_error_condition = 0.001
stable_second_condition = 5
max_second = 30
import random
mode = "plot"
# mode = "heatmap"

kp_min = 1
kp_max = 10
kd_min = 0.5
kd_max = 8
ki_min = 0
ki_max = 1

###########################CPU accelation##################################
import multiprocessing
from functools import partial
from contextlib import contextmanager

@contextmanager
def poolcontext(*args, **kwargs):
    pool = multiprocessing.Pool(*args, **kwargs)
    yield pool
    pool.terminate()
###########################CPU accelation##################################

activate_CPU = multiprocessing.cpu_count()
print(multiprocessing.cpu_count())
def main():
    l = 100
    # for k in tqdm.tqdm(range(l)):
    #     for b in range(l):
    #         sim(kp = 1.0, ki = 0.1, kd = 0.5, dt = 0.5, kk = k/l, kb = b/l)
    # for k in tqdm.tqdm(range(l)):
        # for b in range(l):
        # with poolcontext(processes=activate_CPU) as pool:
        #     pool.imap_unordered(partial(sim, kp = 1.0, ki = 0.1, kd = 0.5, dt = 0.5, kk = k/l), range(l))

    pool = multiprocessing.Pool(activate_CPU)
    jobs = []
    for k in tqdm.tqdm(range(l)):
        for b in range(l):
            # job = pool.apply_async(sim, (1.0, 0.1, 0.5, 0.5, k/l, b/l))
            job = pool.apply_async(sim, (1.0, 0.1, 0.5, 0.5, random.randint(0,999)/1000, random.randint(0,999)/1000))
            jobs.append(job)
    for job in tqdm.tqdm(jobs):
        job.get()
    pool.close()
    pool.join()

    # k=[]
    # if mode == "heatmap":
#     map_data=[]₩
    #     for kp in tqdm(range(int(kp_min*10),int(kp_max*10))):
    #         a = []
        #     for kd in range(int(kd_min*10),int(kd_max*10)):
        #         res = sim(kp = kp/10, ki = 0, kd = kd/10, dt = 0.1)
        #         # if res > 0.1 and res < 99:
        #         #     a.append(res)
        #         # else:
        #         #     a.append(100)
        #         a.append(res)
        #     k.append([kp/10,kd_min + a.index(min(a))/10,min(a)])
        #     a[a.index(min(a))] = 1000
        #     map_data.append(a)
        
        # data_df = pd.DataFrame(np.array(map_data, dtype=np.float64))
        # data_df.columns = np.round(np.arange(kd_min,kd_max,0.1),3)
        # data_df.index = np.round(np.arange(kp_min,kp_max,0.1),3)
        # # data_df.index = np.round(np.arange(0.5,8,0.1),3)
        # # data_df.columns = np.round(np.arange(1,10,0.1),3)
        
        # ax = sns.heatmap(data_df, cmap="inferno")
        # plt.xlabel("kd")
        # plt.ylabel("kp")
        # plt.show()

        # print(len(k))
        # map_data=[]
        # for kp, kd, _ in tqdm(k):
        #     a = []
        #     for ki in range(int(ki_min*100),int(ki_max*100)):
        #         res = sim(kp = kp, ki = ki/100, kd = kd, dt = 0.1)
        #         a.append(res)
        #     a[a.index(min(a))] = 1000
        #     map_data.append(a)

        # data_df = pd.DataFrame(np.array(map_data, dtype=np.float64))
        # ax = sns.heatmap(data_df, cmap="inferno")
        # plt.show()

# register the update function with each slider
class heater:
    def __init__(self):
        self.value = 0
    def update(self, power, dt):
        if power > 0:         
            #Variation of room temperature with power and time variable dt during heating
            self.value += 2 * power * dt
        #Indicates heat loss in a room
        self.value -= 0.5 * dt
        return self.value

class car:
    def __init__(self):
        self.value = 0
        self.velocity = 0
        self.inertia = 0
    def update(self, power, dt):
        self.velocity += dt * power
        self.value += 0.5 * self.velocity * dt + self.inertia
        self.inertia = dt * power
        return self.value

class simple_motor:
    def __init__(self):
        self.value = 0
        self.inertia = 0
    def update(self, power, dt):
        self.value += 0.2/1 * dt * power
        return self.value

class damper_motor:
    def __init__(self):
        self.value = 0
        self.angular_velocity = 0
    def update(self, power, dt):
        # if power > 0:         
        self.angular_velocity += 0.2 * dt * power
        self.value += self.angular_velocity
        self.value -= 0.2 * dt
        return self.value

class mass_spring_damper:
    def __init__(self, kk, kb):
        self.velocity = 0
        self.position = 0
        self.k = kk
        self.b = kb
        self.mass = 1

    def update(self, power, dt):
        spring_force = self.k * self.position # Fs = k * x
        damper_force = self.b * self.velocity # Fb = b * x'

        # If we leave the acceleration alone in equation
        # acceleration = - ((b * velocity) + (k * position)) / mass
        # acceleration = - (spring_force + damper_force) / self.mass
        self.velocity += ((power - (spring_force + damper_force)) / self.mass) * dt # Integral(a) = v
        self.position += self.velocity * dt # Integral(v) = x
        return self.position

class mass_spring_damper2:
    def __init__(self):
        self.velocity = 0
        self.position = 0
        self.k = 0.5
        self.b = 0.5
        self.mass = 1

    def update(self, power, dt):
        spring_force = self.k * self.position # Fs = k * x
        damper_force = self.b * self.velocity # Fb = b * x'

        # If we leave the acceleration alone in equation
        # acceleration = - ((b * velocity) + (k * position)) / mass
        acceleration = - (spring_force + damper_force) / self.mass
        self.velocity += (acceleration * dt) # Integral(a) = v
        self.position += (self.velocity * dt) # Integral(v) = x
        return self.position

class controller:
    def __init__(self):
        self.control_value = 0
        self.prv_error = 0
        self.i = 0
    def update(self, kp, ki, kd, error, dt):
        self.i = self.i + ki * error * dt
        self.control_value = error * kp + self.i + (error - self.prv_error) * kd / dt
        self.prv_error = error
        return self.control_value

def Recurrence_plot(s, eps=None, step=None):
    s = np.array(s)
    if eps == None: eps = 0.001
    # if step == None: step = 1.5
    N = s.size
    S = np.repeat(s[None,:],N,axis=0)
    Z = np.floor(np.abs(S - S.T)/eps)
    # Z[Z > step] = step
    return Z

def sim(kp, ki, kd, dt, kk, kb):
    cur_pose = 0
    target_pose = 1
    pose_graph=[0]
    t_graph=[0]
    control_graph =[0,0,0,0,0,0,0,0,0,0]
    error_graph=[1 for _ in range(0,int(stable_second_condition/dt))]

    error = 10

    system = mass_spring_damper(kk, kb)
    cur_pose = system.position

    pid = controller()
    
    while t_graph[-1] < max_second: #abs(error) > stable_error_condition and 
    # while abs(sum(error_graph[int(-stable_second_condition/dt):]))/5 > stable_error_condition and t_graph[-1] < max_second: #abs(error) > stable_error_condition and 
    # while t_graph[-1] < 10: #abs(error) > stable_error_condition and 
        
        error = target_pose - cur_pose
        control_value = pid.update(kp, ki, kd, error, dt)
        cur_pose = system.update(control_value, dt)
        # cur_pose = system.update(control_graph[-10], dt)

        # print(control_value)
        pose_graph.append(cur_pose)
        error_graph.append(abs(error))
        t_graph.append(t_graph[-1] + dt)
        # control_graph.append(control_value)

    # if mode == "plot":
        # plt.title("time to reach steady state"+ str(round(t_graph[-1],5))+"sec\n"+"kp : "+str(kp)+"    ki : "+str(ki)+"    kd : "+str(kd))
        # plt.plot(t_graph, pose_graph)
        # plt.grid()
        # plt.plot(t_graph, control_value)
        # plt.show()
        
        # plt.imshow(Recurrence_plot(pose_graph), cmap='gray')
        # plt.show()
        # plt.savefig("pid_data/pid_kk_{}_kb_{}.jpg".format(kk, kb))
    figure = sns.heatmap(Recurrence_plot(pose_graph, step=max(pose_graph)), cmap='gray', cbar=False, xticklabels=False, yticklabels=False, square = True).get_figure()    
    figure.savefig("test_data/pid_kk_{}_kb_{}_.jpg".format(kk, kb), bbox_inches = 'tight', pad_inches = 0, dpi=100)
    return round(t_graph[-1],5)

if __name__ == "__main__":
    main()
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm
import pandas as pd
stable_error_condition = 0.001
stable_second_condition = 10

mode = "plot"
# mode = "heatmap"

kp_min = 1
kp_max = 10
kd_min = 0.5
kd_max = 8
ki_min = 0
ki_max = 1

def main():
    k=[]
    sim(kp = 1.0, ki = 0.1, kd = 0.1, dt = 0.1)
    if mode == "heatmap":
        map_data=[]
        for kp in tqdm(range(int(kp_min*10),int(kp_max*10))):
            a = []
            for kd in range(int(kd_min*10),int(kd_max*10)):
                res = sim(kp = kp/10, ki = 0, kd = kd/10, dt = 0.1)
                # if res > 0.1 and res < 99:
                #     a.append(res)
                # else:
                #     a.append(100)
                a.append(res)
            k.append([kp/10,kd_min + a.index(min(a))/10,min(a)])
            a[a.index(min(a))] = 1000
            map_data.append(a)
        
        data_df = pd.DataFrame(np.array(map_data, dtype=np.float64))
        data_df.columns = np.round(np.arange(kd_min,kd_max,0.1),3)
        data_df.index = np.round(np.arange(kp_min,kp_max,0.1),3)
        # data_df.index = np.round(np.arange(0.5,8,0.1),3)
        # data_df.columns = np.round(np.arange(1,10,0.1),3)
        
        ax = sns.heatmap(data_df, cmap="inferno")
        plt.xlabel("kd")
        plt.ylabel("kp")
        plt.show()

        print(len(k))
        map_data=[]
        for kp, kd, _ in tqdm(k):
            a = []
            for ki in range(int(ki_min*100),int(ki_max*100)):
                res = sim(kp = kp, ki = ki/100, kd = kd, dt = 0.1)
                a.append(res)
            a[a.index(min(a))] = 1000
            map_data.append(a)

        data_df = pd.DataFrame(np.array(map_data, dtype=np.float64))
        ax = sns.heatmap(data_df, cmap="inferno")
        plt.show()

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
        self.inertia = 0
    def update(self, power, dt):
        self.angular_velocity += 0.2 * dt * power
        self.value += self.angular_velocity
        # self.inertia = dt * power
        return self.value

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

def sim(kp, ki, kd, dt):
    cur_pose = 0
    target_pose = 1
    pose_graph=[0]
    t_graph=[0]
    control_graph =[0,0,0,0,0,0,0,0,0,0]
    error_graph=[1 for _ in range(0,int(stable_second_condition/dt))]

    error = 1

    system = car()
    cur_pose = system.value

    pid = controller()
    
    while abs(sum(error_graph[int(-stable_second_condition/dt):]))/5 > stable_error_condition and t_graph[-1]<100: #abs(error) > stable_error_condition and 
    # while t_graph[-1] < 10: #abs(error) > stable_error_condition and 
        
        error = target_pose - cur_pose
        control_value = pid.update(kp, ki, kd, error, dt)
        cur_pose = system.update(control_value, dt)
        # cur_pose = system.update(control_graph[-10], dt)

        # print(control_value)
        pose_graph.append(cur_pose)
        error_graph.append(abs(error))
        t_graph.append(t_graph[-1] + dt)
        control_graph.append(control_value)

    if mode == "plot":
        plt.title("time to reach steady state"+ str(round(t_graph[-1],5))+"sec\n"+"kp : "+str(kp)+"    ki : "+str(ki)+"    kd : "+str(kd))
        plt.plot(t_graph, pose_graph)
        plt.grid()
        # plt.plot(t_graph, control_value)
        plt.show()
    return round(t_graph[-1],5)

if __name__ == "__main__":
    main()
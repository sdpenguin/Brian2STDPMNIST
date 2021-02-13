#STDP 10.12.2020 
#The 10th trial 
#conductance based equations for LIF and 3 layers with STDP, Equation was taken from the paper Diehl et al. 2015 ( "Investigation of Unsupervised Learning in SNNs using STDP rules" )

# Poisson -> Exc  | 784 -> 100 |    all-to-all
#Exc -> Inh       | 100 -> 100 |    one-to-one
#Inh -> Exc       | 100 -> 100 |    one- to -all


from brian2 import *
import keras
import numpy as np
import matplotlib.pyplot as plt
import os
from keras.datasets import mnist
from PIL import Image
seed(1234567890)
print('Start')

start_scope()


def load_mnist_data():
    (train_x, train_y), (test_x, test_y) = mnist.load_data()
    return (train_x, train_y), ((test_x, test_y))

(train_x,train_y),((test_x, test_y)) = load_mnist_data()
train_x.shape
print(train_x.shape)


def get_image_pixels(index):
    load_mnist_data()
    img = test_x[index]
    test_img = img.reshape(1, 784)
    return test_img, img


def visualise_connectivity(S):
    Ns = len(S.source)
    Nt = len(S.target)
    plt.figure(figsize = (10,4))
    plt.subplot(121)
    plt.plot(zeros(Ns),arange(Ns),'ok', ms=10)
    plt.plot(ones(Nt),arange(Nt),'ok',ms=10)
    for i,j in zip(S.i, S.j):
        plt.plot([0,1],[i,j], '-k')
    plt.xticks([0,1],['Source','Target'])
    plt.ylabel('Neuron index')
    plt.xlim(-0.1,1.1)
    plt.ylim(-1, max(Ns,Nt))
    plt.show()


test_img, img = get_image_pixels(120)
print(img.shape)
mat = np.reshape(img,(28,28))
#print('mat = ',mat)

#Create PIL image
img = Image.fromarray(mat, 'L')
#plt.imshow(img)
#plt.title('my number')
#plt.show()
#print(test_img.shape)

arr, img = get_image_pixels(120)
#print(arr)


#--------------  Parameters ------------ 
N = 784
# Global Parameters
v_rest_e = -65.*mV
v_rest_i = -60.*mV
v_reset_e = -65.*mV
v_reset_i = -45.*mV
v_thresh_e = -52.*mV
v_thresh_i = -40.*mV
refrac_e = 5. *ms
refrac_i = 2.*ms

# STDP params
tau_pre = 5 * ms
tau_post = 25 * ms
taupre = 20*ms
taupost = taupre
gmax = 0.2
dApre = 0.01
dApost = -dApre * taupre / taupost * 1.05
dApost *= gmax
dApre *= gmax
#---------



exc_lif_eqs = '''
dv/dt = ((v_rest_e - v) + (I_synE + I_synI) / nS) / (100 * ms)  : volt  (unless refractory)
        I_synE = g_exc * nS *         -v                        : amp
        I_synI = g_inh * nS * (-100.*mV-v)                      : amp
        dg_exc/dt = -g_exc/(1.0*ms)                             : 1
        dg_inh/dt = -g_inh/(2.0*ms)                             : 1
'''



inh_lif_eqs = '''
dv/dt = ((v_rest_i - v) + (I_synE + I_synI) / nS) / (10*ms)  : volt  (unless refractory)
        I_synE = g_exc * nS *         -v                     : amp
        I_synI = g_inh * nS * (-85.*mV-v)                    : amp
        dg_exc/dt = -g_exc/(1.0*ms)                          : 1
        dg_inh/dt = -g_inh/(2.0*ms)                          : 1
'''





mnist_rates = arr*Hz  #max arr*(1/255)*Hz  #12.8
#print("rates",mnist_rates)
print(mnist_rates.shape)
# synapse parameters

input = PoissonGroup(N, rates=mnist_rates)

exc_layer = NeuronGroup(400,
                       exc_lif_eqs,
                       threshold='v>v_thresh_e',
                       reset='v = v_reset_e',
                       refractory = refrac_e,
                       method='euler')
inh_layer = NeuronGroup(400,
                        inh_lif_eqs,
                        threshold ='v>v_thresh_i',
                        reset = 'v = v_reset_i',
                        refractory = refrac_i,
                        method = 'euler')                      


#exc_layer.v = 'v_rest_e + rand()*(v_thresh_e-v_rest_e)'
exc_layer.v = v_rest_e - 40. * mV
inh_layer.v = 'v_rest_i + rand()*(v_thresh_i-v_rest_i)'
inh_layer.v = v_rest_i - 40. * mV


#Poisson input ------------> Excitatory
syn_poi_exc = Synapses(input, exc_layer,
             '''w :1 
                dApre/dt = -Apre / taupre : 1 (event-driven)
                dApost/dt = -Apost / taupost : 1 (event-driven)''',
             on_pre='''g_exc += w
                       Apre += dApre
                       w = clip(w + Apost, 0, gmax)''',
                       
             on_post='''Apost += dApost
                        w = clip(w + Apre, 0, gmax)''',
             )

w_e = 60*0.27/10
#Excitatory ----------> Inhibitory 
syn_exc_inh = Synapses(exc_layer,
                       inh_layer,
                       on_pre = 'g_exc += w_e')   

w_i = 20*4.5/10
#Inhibitory ---------> Excitatory
syn_inh_exc = Synapses(inh_layer,
                       exc_layer,
                       on_pre = 'g_inh += w_i') 



syn_poi_exc.connect()
#syn_exc_inh.connect(j = 'i') 
syn_exc_inh.connect() 
#syn_inh_exc.connect(condition = 'i != j')
syn_inh_exc.connect()

#  Initialization
syn_poi_exc.w = 'rand()*gmax'
#syn_exc_inh.g_exc = 'rand()*gmax'
#syn_inh_exc.g_inh = 'rand()*gmax'

#  State Monitors  

state_mon_exc = StateMonitor(exc_layer,'v', record =[0])
state_mon_inh = StateMonitor(inh_layer,'v', record = [0])
#-------------------------------
statemon_poi_exc = StateMonitor(syn_poi_exc,'v',record = [0])
statemon_exc_inh = StateMonitor(syn_exc_inh,'v', record =[0])
statemon_inh_exc = StateMonitor(syn_inh_exc,'v',record =[0])


#  Spike monitors

spike_mon_poi = SpikeMonitor(input)
spike_mon_exc = SpikeMonitor(exc_layer)
spike_mon_inh = SpikeMonitor(inh_layer) 

run(750*ms, report='text')
print("1 ", spike_mon_poi.N)
print("2 ", spike_mon_exc.N)
print("3 ", spike_mon_inh.N)
print("input ", input.N)
print("exc_layer", exc_layer.N)
print("inh_layer", inh_layer.N)


#weights = np.array(state_mon_inp.w)*1000 
#exit()
fig = plt.figure(figsize=(20,10))
ax = fig.add_subplot(411)
plt.plot(spike_mon_poi.t/ms, spike_mon_poi.i, '|' , linewidth = 0.1)
plt.title('Input Spikes')
plt.tight_layout()

ax = fig.add_subplot(412)
plt.plot(spike_mon_exc.t/ms, spike_mon_exc.i, '|', linewidth = 0.1)
plt.title('Ext Spikes')
plt.tight_layout()    

ax = fig.add_subplot(413)
plt.plot(spike_mon_inh.t/ms, spike_mon_inh.i, '|' , linewidth = 0.1)
plt.title('Inh Spikes')
plt.tight_layout()    

ax = fig.add_subplot(414)
plt.plot(statemon_poi_exc.t/ms, statemon_poi_exc.v.T/mV , label = 'g_poi' , color = 'red')
plt.plot(statemon_exc_inh.t / ms, statemon_exc_inh.v.T/mV , label = 'g_exc' , color = 'blue')  #state of g_exc & g_inh
plt.plot(statemon_inh_exc.t/ms, statemon_inh_exc.v.T/mV, label = 'g_inh', color ='green')           
plt.legend(loc = 'best')
ax.set_xlabel('Poi->Exc->Inh->Exc current')
plt.tight_layout()
plt.show()


plt.plot(state_mon_exc.t/ms, state_mon_exc.v[0]/mV,label = 'V_exc')
plt.plot(state_mon_inh.t/ms, state_mon_inh.v[0]/mV, label = 'V_inh')
plt.legend(loc = 'best')
xlabel('Time\n\n')
ylabel('Voltage')
plt.title('Shape of spike')
plt.tight_layout()
plt.show()





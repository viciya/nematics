#%%
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import glob
import scipy.stats as circ
from scipy.stats import circmean, circstd, circvar
from scipy.io import loadmat
from numpy import random
import seaborn as sns
sns.set()
sns.axes_style('white')
#from importlib import reload
#reload(circstd)

def shift_to_0_2pi(ang):
    ang[ang<0]=ang[ang<0]+2*np.pi
    return ang

x0 = random.normal(size=10000)
#%%
'''Test circ stat'''
x = (np.pi * x0/(x0.max()-x0.min()))
x = shift_to_0_2pi(x)
sns.distplot(x,bins=30, hist=True, label=str(np.mean(x))[:4]+' | '+str(np.std(x))[:4])
print('Mean,std: ', np.mean(x),' | ',np.std(x))
print('Circ-mean,circ-std: ',circmean(x),' | ',circstd(x))

#%%
px2mic = 3*.74
dirs = glob.glob(r"D:\GD\Curie\DESKTOP\HT1080\symm_pDefect_x_y_angle_frame_idx_2_*")

all_data = np.array([])
ang_list = []
std_list = []
width_list = []

for dir in dirs:
    center = int(int(dir.split('_')[-1].split('.')[-2]) //(2 * px2mic))
        
    left = 70
    right = int(int(dir.split('_')[-1].split('.')[-2]) // px2mic)-70
    data = np.loadtxt(dir)
#    plt.hist(data[:,0]*px2mic, bins=30)
    
    data[:,2] = np.mod(data[:,2],360)
    
    ang = data[:,2][data[:,0]<left]+180
    ang = np.concatenate((ang, data[:,2][data[:,0]>right]), axis=0)
    
    ang = (180-np.mod(ang,360))*np.pi/180
    
#    print('%.4s - tilt: %.4s - std: %f' 
#            % (2*px2mic*center, np.mean(data[:,2]), np.std(data[:,2])/len(data)**.5)
#            )
    print('%.4s - tilt: %.4s - std: %f' 
            % (2*px2mic*center, circmean(ang)*180/np.pi, circstd(ang)*180/np.pi/len(data)**.5)
            )
    ang_list.append(circmean(ang)*180/np.pi)
    std_list.append(circstd(ang)*180/np.pi)
    width_list.append(2*px2mic*center)
    
#    TODO change to count only for boundaries instead of all defects
    all_data = np.concatenate((all_data, ang), axis=0)
    plt.figure('1')    
    plt.hist(ang*180/np.pi, bins=30, rwidth=.8)

plt.figure('2')    
plt.errorbar(width_list, ang_list, yerr=std_list, fmt='o')

print('average tilt: ', circmean(all_data)*180/np.pi)
print('average tilt: ', circstd(all_data)*180/np.pi/len(all_data)**.5)
ref_data = np.zeros_like(all_data)
print(stats.ttest_ind(all_data,ref_data, equal_var = False))
#plt.hist(data[:,2], bins=30)
#plt.hist(data[:,0]*px2mic, bins=30)
#Ttest_indResult(statistic=-9.477523021006368, pvalue=2.6634660571792834e-21)
'''
np.circmean(ALL DATA)
np.circstd(All DATA)
t-test(ALL DATA)
'''
#%%
'''
FLOW DIRECTION AT DEFECT CORE (FIG. S5?)
'''

width = np.array([300,400,500,600,700,800,1000,1500])
r_all = np.array([])
l_all = np.array([])
r_mean = []
l_mean = []
r_std = []
l_std = []

fig, axs = plt.subplots(1,len(width))#,figsize=(6, 4)

for w, ax in zip(width,axs.flat):#[:4]
    path = glob.glob(r'D:\GD\Curie\DESKTOP\HT1080\figs_and_data\average_flows-av_core-box6-3\*_'+str(w)+'.mat')[0]
    print(path)    
    rr = shift_to_0_2pi(loadmat(path)['Rvel_angle'])
#    rr = loadmat(path)['Rvel_angle']
    r_all = np.hstack([r_all,rr[0]])
    print('R', w, r_all.shape)

    ll = shift_to_0_2pi(loadmat(path)['Lvel_angle'])
#    ll = loadmat(path)['Lvel_angle']
    l_all =  np.hstack([l_all,ll[0]])
    print('L', w, l_all.shape)

    r_mean.append(180/np.pi * circmean(rr))
    r_std.append(180/np.pi * circstd(rr))
    l_mean.append(180/np.pi * circmean(ll))
    l_std.append(180/np.pi * circstd(ll))
    print('R-Non-edge', w, 'circ-mean:', 180/np.pi * circmean(rr), 'circ-std:', 180/np.pi * circstd(rr))
    print('R-Non-edge', w, 'mean:', 180/np.pi * np.mean(rr), 'std:', 180/np.pi * np.std(rr))
    print('----------')
    print('L-edge', w, 'circ-mean:', 180/np.pi * circmean(ll), 'circ-std:', 180/np.pi * circstd(ll))
    print('L-edge', w, 'mean:', 180/np.pi * np.mean(ll), 'std:', 180/np.pi * np.std(ll))
    print('----------')
    
    print('Rall', 180/np.pi * circmean(r_all), 180/np.pi * circstd(r_all))
    print('Lall', 180/np.pi * circmean(l_all), 180/np.pi * circstd(l_all))
    print('///////////')

    sns.distplot(180/np.pi * rr, hist=True, ax=ax)
    sns.distplot(180/np.pi * ll, hist=True, ax=ax)
    ax.set_title("width: %.4s" %(w))

plt.figure('4') 
offset = 4
plt.axhline(180, ls=':', color="r")  
plt.errorbar(width-offset, r_mean, yerr=r_std, fmt='o', label='non-edge')
plt.errorbar(width+offset, l_mean, yerr=l_std, fmt='o', label='edge')
plt.ylabel('$Angle ~(deg)$', fontsize=16)
plt.xlabel('$Width ~(\mu m)$',  fontsize=16)
xtick_space = 100
plt.xticks(np.arange(200, 1500+xtick_space, xtick_space))
ytick_space = 30
plt.yticks(np.arange(0, 360+ytick_space, ytick_space))
plt.tight_layout()
#plt.gca().set_aspect('equal', adjustable='box')
plt.legend()

#%%
'''
DEFECT ANGLE TILT (FIG. 6e)
'''
dirs = glob.glob(r"D:\GD\Curie\DESKTOP\HT1080\figs_and_data\defect_orient\def_orient_left_right_all_*.txt")
av_all = []
std_all = []
for dir in dirs:
    data = np.loadtxt(dir)
#    np.sum(data, axis=0)
    averege = 180/np.pi * (np.sum(data[:,0] * data[:,1]) / data[:,1].sum())
    std1 = (180/np.pi)**2 * (np.sum(data[:,0]**2 * data[:,1]) / data[:,1].sum())
    std2 = np.sqrt(std1 - averege**2)
    av_all.append(averege)
    std_all.append(std2)
    
    print('av >>', averege, ' std >>', std2)

print('-------------------------')
print('AV >>', np.mean(np.array(av_all)), ' STD >>', np.mean(np.array(std_all)))   

#%%
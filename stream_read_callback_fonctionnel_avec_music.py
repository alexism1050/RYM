import pyaudio
import struct
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, lfilter, filtfilt
import time
import pygame
import wave
import sys

#DEFINE FILTRE
BUTTER_ORDER = 4
CUTOFF = 100
Wn = 0.00226757

#DEFINE DETECTEUR DE PAS
THRESHOLD_HIGH = 0.5
THRESHOLD_LOW = 0
OFFSET = 0.01544

b,a = butter(BUTTER_ORDER, Wn,'lowpass',analog=False)	

#DEFINE STREAM
FORMAT = pyaudio.paFloat32
SAMPLEFREQ = 44100
FRAMESIZE = 16384

DELTA = 0.6
DELTAD = 0.3
MI_DELTA = DELTA/3

p = pyaudio.PyAudio()

#DECLARE VARIABLE
buffer2 = np.zeros(FRAMESIZE)
buffer3 = np.zeros(FRAMESIZE)
diff_ave = np.zeros(129)
plot_data = np.zeros(884736)
plot_diff_ave = np.zeros(1327104)

print('running')

#CALCULATE STEP
pas = 0
detect = 0
init = 2
reset = 0
threshold_min = 0
ajustement = 0
depassement = 0
affichage = 0

#plt.ion()
#plt.show()

myfile = wave.open("loopCG.wav", "rb")
frames = myfile.getnframes()
fs = myfile.getframerate()
Time = float(frames)/ fs

def StrechFs(FS, BPM, newBPM):
	# Fs initial 
	#BPM initial
	#print FS,newBPM,BPM
	Fss = float( float(FS) * float(newBPM) / float(BPM) )
	#print Fss
	return (Fss)
	
def stretchingBPM(currentBPM, newBPM):
    global smoothStretch
    smoothStretch = 0.05 
    
    X = -30
    #print "X = ", X
    Y = 30
    #print "Y = ", Y
    
    deltaBPM = currentBPM - newBPM
    
    #print "deltaBPM = ", deltaBPM
    
    if (deltaBPM >= X and deltaBPM <= Y):
        newCurrentBPM = newBPM 
    elif deltaBPM > X:
        newCurrentBPM = currentBPM * (1 - smoothStretch)
    elif deltaBPM < Y:
        newCurrentBPM = currentBPM * (1 + smoothStretch)
        
    #print "newCurrentBPM = ", newCurrentBPM
    
    return newCurrentBPM

def callback(in_data, frame_count, time_info, status):
	global pas, detect, init, buffer2, buffer3, diff_ave_all, reset, ajustement, threshold_min, plot_data, affichage, plot_diff_ave
	
	max_ave = 0
	
	if init == 0:
		buffer1 = np.fromstring(in_data, 'float32')
		buffer = np.append(buffer1,buffer2)
		buffer = np.append(buffer,buffer3)
		
		plot_data = np.append(plot_data, buffer1)
		
		#affichage = affichage + 1
		
		# if affichage >= 27:
			# plot_data_filtre = lfilter(b,a,plot_data)
			# print plot_data_filtre.size
			
			# plot_data_filtre = np.abs(plot_data_filtre)
			# for i in range(16384,plot_data_filtre.size-16383,128):
				# plot_diff_ave[i] = np.sum(plot_data_filtre[i-2999:i+2999])/100 - np.sum(plot_data_filtre[i-9999:i+9999])/300
			# plt.plot(plot_data,'b')
			# plt.plot(plot_diff_ave,'g')
			# plt.plot(plot_data_filtre,'r')
			
			# plt.show()
			# affichage = 0
		
		buffer_filtre = lfilter(b,a,buffer)
		
		buffer_filtre = np.abs(buffer_filtre)
		
		j = 0
		
		# reset = reset + 1
		# if reset > 10:
			# pas = 0
		
		for i in range(16384,32768,128):
			diff_ave[j] = np.sum(buffer_filtre[i-3000:i+3000])/100 - np.sum(buffer_filtre[i-10000:i+10000])/300
			#diff_ave_all = np.append(diff_ave_all, diff_ave)
			j = j + 1
			
		for x in range(0,diff_ave.size):
			if detect == 0:
				if diff_ave[x] > THRESHOLD_HIGH + threshold_min:
					detect = 1
					pas = pas + 1
					if x < 246:
						max_ave = np.sum(diff_ave[x:x+10])/10
					else:
						max_ave = np.sum(diff_ave[x:diff_ave.size])/(diff_ave.size-x)
						
					x = x + 46
					# reset = 0
					
					ajustement = 0
					
					if max_ave > THRESHOLD_HIGH + threshold_min + DELTA + DELTAD:
						threshold_min = threshold_min + MI_DELTA
					# plt.plot(diff_ave,'r')
					
					# plt.show()
				
			else:
				if diff_ave[x] < THRESHOLD_LOW:
					detect = 0
					
		#plt.scatter(buffer,'b')
		#plt.draw()
			
		#max = np.max(diff_ave)	
		#print "max :",max_ave
		#print "Threshold min:",(THRESHOLD_HIGH + threshold_min)
		
		
						
						
		ajustement = ajustement + 1
		if ajustement > 9:
			if threshold_min <= 0:
				threshold_min = 0
			else:
				threshold_min = threshold_min - MI_DELTA
				#print "Threshold max :",THRESHOLD_HIGH + threshold_min + DELTAD
		
		buffer3 = buffer2
		buffer2 = buffer1
		
	elif init == 2:
		buffer3 = np.fromstring(in_data, 'float32')
		init = init - 1
	elif init == 1:
		buffer2 = np.fromstring(in_data, 'float32')
		init = init - 1

	print pas
	sys.stdout.flush()
	
	return (in_data, pyaudio.paContinue)

stream = p.open(format=FORMAT,channels=1,rate=SAMPLEFREQ,input=True, output=False,frames_per_buffer=FRAMESIZE,stream_callback=callback)

stream.start_stream()

l_pas = -1
n_pas = 0
bpm = 128.0
BPM_CONST = 128.0
lastbpm = 128.0
ft = 1

# pygame.mixer.init(int(120))
# pygame.mixer.music.load("loopCG.wav")
# pygame.mixer.music.play(-1, 0.0)
# time.sleep(float(Time))

while stream.is_active():
	n_pas = pas
	#plt.plot(plot_data,'b')
	if l_pas < 0:
		pas = 0
		l_pas = 0
		bpm = 120
	else:
		bpm = ((n_pas+1) - l_pas)*(60.0/float(Time))
		
	
	newbpm = stretchingBPM(lastbpm, bpm)
	
	print "BPM : ",newbpm
	sys.stdout.flush()
	
	l_pas = n_pas

	newfs = StrechFs(float(fs),float(BPM_CONST), newbpm)
	#print newfs
	Time = float(frames)/ float(newfs)
	
	#pygame.mixer.music.stop()
	#pygame.mixer.stop()
	if ft == 1:
		ft = 0
		pygame.mixer.pre_init(int(newfs))
		pygame.mixer.init()
		pygame.mixer.music.load("loopCG.wav")
		pygame.mixer.music.play(-1, 0.0)
	else:
		pygame.mixer.quit()
		pygame.mixer.pre_init(int(newfs))
		pygame.mixer.init()
		pygame.mixer.music.load("loopCG.wav")
		pygame.mixer.music.play(-1, 0.0)
	
	time.sleep(float(Time))
	
	lastbpm = newbpm
	
	#print "bpm :",bpm
	

#data = stream.read(NOFFRAMES*FRAMESIZE)
#decoded = struct.unpack(str(NOFFRAMES*FRAMESIZE)+'f',data)
#decoded = np.fromstring(data, 'float32')
#print decoded.size

#decoded_filtre=lfilter(b,a,decoded)

stream.close()
p.terminate()
print('done')
plt.plot(diff_ave_all,'r')
plt.show()

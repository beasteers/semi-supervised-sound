import tqdm
import glob
import soundfile



def convert(files):
	'''So UrbanSound8K apparently stores the audio files with a 24 bit PCM which 
	can't be read by scipy.io.wavfile.read so this will convert a list of files 
	to 16 bit PCM.
	Arguments:
		files (list): list of filenames to convert (relative to script of course)
	'''
	for f in tqdm.tqdm(files):
		data, samplerate = soundfile.read(f) # read file
		soundfile.write(f, data, samplerate, subtype='PCM_16') # rewrite with 16 bit

if __name__ == '__main__':
	convert(glob.glob('../Data/audio/*/*.wav')) # convert everything
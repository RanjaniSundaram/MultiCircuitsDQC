import os
import shutil
import glob
import random as rn

def PrepareInput(numCircuits):	
	filecounter = 0
	dircounter = 0

	directory = "circuitsVaryQubits"
	directory2="MultiCircuits"
	l=os.listdir(directory+'/')
	rn.shuffle(l)
	for file in l:
		absoluteFilename = os.path.join(directory, file)

		if filecounter % numCircuits == 0:  
			dircounter += 1
			os.mkdir(os.path.join(directory2, "dir"+str(dircounter)))

		targetfile = os.path.join(directory2, "dir"+str(dircounter), file)  # builds absolute target filename
		shutil.copy(absoluteFilename, targetfile)

		filecounter += 1

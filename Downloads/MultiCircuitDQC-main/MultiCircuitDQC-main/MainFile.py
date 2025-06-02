
import os
import numpy as np
import sys
from OM import *
from G_star_simple import *
from G_star_LP import *


if __name__ == '__main__':

	#input_filename = sys.argv[1]
	output_filename = ''
	
	print('Function input options:')
	print('\n')
	print('1. OM')
	print('\n')
	print('2. G star simple')
	print('\n')
	print('3. G star LP')
	print('\n')
	opt=int(input("Enter Option:"))
	filename=input("Enter Input Filename: ")
	num_parts=input("Enter number of partitions: ")
	
	if opt==1:
		print(filename)
		main_func_OM("example.txt",num_parts)
	elif opt==2:
		main_func_Gstar_simple(filename,num_parts)
	elif opt==3:
		main_func_Gstar_LP(filename,num_parts)
	else:
		print("Invalid input")

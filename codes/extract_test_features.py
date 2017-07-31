# -*- coding: utf-8 -*-

import os, sys
import string
import numpy as np
from extract import *

if __name__ == '__main__':
	infile = open(sys.argv[1], 'r')
	contents = infile.read()[1:].split('\n>')
	infile.close()
	identifiers = [ele.split()[0] for ele in contents]

	for identifier in identifiers:
		outfile = open('../test_features/'+identifier+'_features', 'w')
		outfile_obj = open('../test_features/'+identifier+'_features_obj', 'w')

		m = np.loadtxt('../test_data/'+identifier+'.CCMpred')
		m = (m+m.T)/2.0

		infile = open('../test_data/'+identifier+'.helices','r')
		infile.readline()
		infile.readline()
		infile.readline()
		line = infile.readline()
		helix_range, helix_frag = [], []
		while line.startswith("Helix"):
			l = line.split('(')[1].split(')')[0].split(',')
			helix_range.append([int(l[0])-1,int(l[-1])-1])
			helix_frag.append(line.split()[2])
			line = infile.readline()
		infile.close()
		if len(helix_range)<=1:
			print '%s: The number of helix is less than 2.' % identifier
			sys.exit(1)

		#for F1
		features=extract_ridge_feature(m,helix_range)

		length_list = []
		flag_arr = np.array([], dtype=np.int64)
		for i in range(len(helix_range)):

			#for F4
			length_list.append(len(helix_range))

			#for F5
			if i == 0:
				flag_arr = np.array([1, 0, 0, 0], dtype=np.int64)
			elif i == 1:
				flag_arr = np.vstack( (flag_arr, np.array([0, 1, 0, 0], dtype=np.int64)) )
			elif i == len(helix_range)-2:
				flag_arr = np.vstack( (flag_arr, np.array([0, 0, 1, 0], dtype=np.int64)) )
			elif i == len(helix_range)-1:
				flag_arr = np.vstack( (flag_arr, np.array([0, 0, 0, 1], dtype=np.int64)) )
			else:
				flag_arr = np.vstack( (flag_arr, np.array([0, 0, 0, 0], dtype=np.int64)) )

		count=0
		for i in range(len(helix_range)-1):
			for j in range(i+1, len(helix_range)):

				#output F1
				outfile.write(' '.join(["{}".format(k) for k in features[count]])+' ')
				count += 1

				#output F2
				sepResLen = helix_range[j][0]-helix_range[i][1]-1
				if (sepResLen >= 1) and (sepResLen <= 2):
					outfile.write('1 0 0 0 0 0 0 0 ')
				elif (sepResLen >= 3) and (sepResLen <= 7):
					outfile.write('0 1 0 0 0 0 0 0 ')
				elif (sepResLen >= 8) and (sepResLen <= 10):
					outfile.write('0 0 1 0 0 0 0 0 ')
				elif (sepResLen >= 11) and (sepResLen <= 23):
					outfile.write('0 0 0 1 0 0 0 0 ')
				elif (sepResLen >= 24) and (sepResLen <= 34):
					outfile.write('0 0 0 0 1 0 0 0 ')
				elif (sepResLen >= 35) and (sepResLen <= 52):
					outfile.write('0 0 0 0 0 1 0 0 ')
				elif (sepResLen >= 53) and (sepResLen <= 70):
					outfile.write('0 0 0 0 0 0 1 0 ')
				else:
					outfile.write('0 0 0 0 0 0 0 1 ')

				#output F3
				sepHelicesNum = j-i-1
				if sepHelicesNum == 0:
					outfile.write('1 0 0 0 0 0 0 0 ')
				elif sepHelicesNum == 1:
					outfile.write('0 1 0 0 0 0 0 0 ')
				elif sepHelicesNum == 2:
					outfile.write('0 0 1 0 0 0 0 0 ')
				elif (sepHelicesNum >= 3) and (sepHelicesNum <= 9):
					outfile.write('0 0 0 1 0 0 0 0 ')
				elif (sepHelicesNum >= 10) and (sepHelicesNum <= 13):
					outfile.write('0 0 0 0 1 0 0 0 ')
				elif (sepHelicesNum >= 14) and (sepHelicesNum <= 17):
					outfile.write('0 0 0 0 0 1 0 0 ')
				elif (sepHelicesNum >= 18) and (sepHelicesNum <= 20):
					outfile.write('0 0 0 0 0 0 1 0 ')
				else:
					outfile.write('0 0 0 0 0 0 0 1 ')

				#output F4
				for m in range(-1, 2):
					if i+m >= 0:
						outfile.write('%d ' % length_list[i+m])
					else:
						outfile.write('%d ' % 0)
				for m in range(-1, 2):
					if j+m <= len(helix_range)-1:
						outfile.write('%d ' % length_list[j+m])
					else:
						outfile.write('%d ' % 0)

				#output F5
				for n in range(flag_arr.shape[1]-1):
					outfile.write('%d ' % flag_arr[i, n])
				for n in range(1, flag_arr.shape[1]):
					outfile.write('%d ' % flag_arr[j, n])

				outfile.write('\n')
				outfile_obj.write('Helix'+str(i+1)+' '+'Helix'+str(j+1)+'\n')

		outfile.close()
		outfile_obj.close()

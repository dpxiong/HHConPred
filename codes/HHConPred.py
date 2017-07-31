import sys, os
import numpy as np
from sklearn.ensemble import AdaBoostClassifier

if __name__ == '__main__':
	identifiers = []
	infile = open(sys.argv[1], 'r')
	contents = infile.read()[1:].split('\n>')
	infile.close()
	for ele in contents:
		identifiers.append(ele.split()[0])

	clf = AdaBoostClassifier(n_estimators=130)

	Dtrain = np.loadtxt('../train_features/train_features')
	x_train, y_train = Dtrain[:, 0:-1], Dtrain[:, -1]
	clf.fit(x_train, y_train)

	for identifier in identifiers:
		outfile = open('../test_results/'+identifier+'.result', 'w')
		outfile.write('#Helix_id1 Helix_id2      Interaction_prob\n')

		Dtest = np.loadtxt('../test_features/'+identifier+'_features')
		Z = clf.predict_proba(Dtest)[:, 1]

		infile = open('../test_features/'+identifier+'_features_obj')
		objs = infile.readlines()
		infile.close()

		for i in range(Dtest.shape[0]):
			outfile.write('%s          %f\n' % (objs[i].rstrip('\n'), Z[i]))
		outfile.close()

import argparse
import time
import numpy as np
from sklearn.cluster import KMeans


if __name__ == "__main__":

	parser = argparse.ArgumentParser(description="K-means algorithm")
	parser.add_argument('-f', dest='filename', type=str, 
						help="file containing data to be clustered")
	parser.add_argument('-k', dest='n_centroids', type=int, default=5,
						help="the number of clusters to find")
	parser.add_argument('-s', dest='sep', type=str, default=',',
						help="file delimiter")
	parser.add_argument('-n', dest='n_points', type=int, default=-1,
						help="number of points in file to read (optional)")
	parser.add_argument('-d', dest='dimensions', type=int, default=-1,
						help="fnumber of dimensions of input data (optional)")
	parser.add_argument('-e', dest='tol', type=float, default=1e-5,
						help="minimum fraction of points that don't change clusters to end kmeans loop")
	parser.add_argument('-i', dest='max_iter', type=int, default=100,
						help="maximum number of iterations within kmeans")
	parser.add_argument('-t', dest='trials', type=int, default=25,
						help="umber of kmeans trials to perform to find best clustering")
	parser.add_argument('-p', dest='cpus', type=int, default=1,
						help='number of cpus to use')
	parser.add_argument('-v', dest='verbose', type=int, default=0,
						help="control amount of printing 0, 1, 2 ")

	args = parser.parse_args()

	data = np.genfromtxt(args.filename, delimiter=args.sep)
	if args.n_points != -1:
		data = data[:args.n_points,:]
	if args.dimensions != -1:
		data = data[:,:args.dimensions]

	start = time.time()
	kmeans = KMeans(n_clusters=args.n_centroids, init='random', max_iter=args.max_iter, 
					n_init=args.trials, tol=args.tol, n_jobs=args.cpus)
	kmeans.fit(data)
	end = time.time()

	print "\nSKLEARN K-MEANS"
	print "%dx%d data, %d clusters, %d trials, %d cores" %(data.shape[0], data.shape[1], args.n_centroids, args.trials, args.cpus)
	print "Inertia: %f" %kmeans.inertia_
	print "Runtime: %fs" %(end - start)

import numpy as np
import time


def find_nearest_centroids(data, centroids):
	all_distances = np.array([[np.sum((x-c)**2) for c in centroids] for x in data])
	return np.argmin(all_distances, axis=1), np.amin(all_distances, axis=1)

def kmeans(data, n_clusters=10, init='random', tol=0.0001, max_iter=300, best_of=10):

	def _kmeans(data, n_clusters, init, tol, max_iter):

		if init == 'random':
			rand = range(data.shape[0])
			np.random.shuffle(rand)
			centroids = data[rand[:n_clusters]]
		else:
			raise NameError('Unknown initialization method')

		memberships    = np.zeros(data.shape[0])
		N     = data.shape[0]
		delta = float(N)
		iters = 0
		inertia = 0.0

		comp_s = time.time()
		while delta / N > tol and iters < max_iter:
			delta = 0.0
			inertia = 0.0

			centers, distances = find_nearest_centroids(data, centroids)
			inertia = np.sum(distances)
			count_centers = np.bincount(centers)
			if 0 in count_centers:
				delta = tol * N + 1.0
			else:
				delta = float(np.sum([1 if x != y else 0 for x, y in zip(centers, memberships)]))
			memberships = centers
			centroids = np.array([np.sum(data[np.where(centers==i)[0]], axis=0) / c if c 
							else data[np.random.random_integers(data.shape[0]-1)] 
							for i, c in zip(xrange(len(count_centers)), count_centers) ])
			iters += 1
		comp_e = time.time()
		return inertia, centroids, memberships, iters, comp_e - comp_s

	best_inertia = np.inf
	best_centroids = np.zeros((n_clusters, data.shape[1]))
	best_labels = np.zeros(data.shape[0])
	loops = 0
	comp_time = 0.0
	total_iterations = 0

	while loops < best_of:
		inertia, centroids, labels, iterations, elapsed = _kmeans(data, n_clusters, init, tol, max_iter)
		total_iterations += iterations
		comp_time += elapsed
		if inertia < best_inertia:
			best_inertia = inertia
			best_centroids = centroids
			best_labels = labels
		loops += 1
	return best_inertia, best_centroids, best_labels, total_iterations, comp_time


if __name__ == "__main__":
	import argparse

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
	parser.add_argument('-v', dest='verbose', type=int, default=0,
						help="control amount of printing 0, 1, 2 ")

	args = parser.parse_args()

	data = np.genfromtxt(args.filename, delimiter=args.sep)
	if args.n_points != -1:
		data = data[:args.n_points,:]
	if args.dimensions != -1:
		data = data[:,:args.dimensions]

	start = time.time()
	inertia, centers, labels, total_iterations, comp_time = kmeans(data, best_of=args.trials, max_iter=args.max_iter, 
									n_clusters=args.n_centroids, tol=args.tol)
	end = time.time()

	print "\nNUMPY K-MEANS"
	print "%dx%d data, %d clusters, %d trials, 1 core" %(data.shape[0], data.shape[1], args.n_centroids, args.trials)
	print "Inertia: %f" %inertia
	print "Total Iterations: %d" %total_iterations
	print "Runtime: %fs" %(end - start)
	print "Computation time: %fs" %comp_time



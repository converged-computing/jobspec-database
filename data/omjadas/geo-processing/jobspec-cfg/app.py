import json
import re
import sys
from jsonReader import JSONReader
from mpi4py import MPI
from collections import defaultdict, Counter
from functools import reduce
from itertools import takewhile

GRID = sys.argv[2]


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Open and store grid data
    with open(GRID, "r", encoding="utf-8") as grid:
        grid_data = json.load(grid)

    # Create JSONReader object to read tweets from file
    file = JSONReader(sys.argv[1], rank, size)

    # Declare defaultdicts for storing tweet and hashtag counts
    grids = defaultdict(int)
    hashtags = defaultdict(lambda: defaultdict(int))

    # Loop over tweets from file and calculate counts
    for tweet in file:
        grid = get_grid(tweet, grid_data)
        # If the tweet belongs to a grid cell
        if grid:
            grids[grid] += 1
            for hashtag in get_hashtags(tweet["doc"]["text"]):
                # Strip non-ascii characters from the tweet
                hashtag = hashtag.lower().encode("ascii", "ignore").decode("utf-8")
                if hashtag:
                    hashtags[grid][hashtag] += 1

    # Convert grids and hashtags to dicts so they can be sent using MPI
    grids = dict(grids)
    hashtags = dict(hashtags)

    # Gather grid and hashtag data at 0
    grid_data = comm.gather(grids, root=0)
    hashtag_data = comm.gather(hashtags, root=0)

    if rank == 0:
        # Combine the tweet counts for each grid cell
        total_tweets = reduce(
            lambda x, y: x.update(y) or x,
            (Counter(d) for d in grid_data))

        # Output the grid cells sorted by the number of respective tweets
        for grid in total_tweets.most_common():
            print("{}: {} posts".format(grid[0], grid[1]))

        # Combine the hashtag counts for each grid cell
        new_dict = defaultdict(lambda: Counter())
        for h in hashtag_data:
            for grid in h:
                for hashtag in h[grid]:
                    new_dict[grid][hashtag] += h[grid][hashtag]

        # Output the five most popular hashtags for each grid cell
        for grid in total_tweets.most_common():
            print("{}: {}".format(grid[0], most_common(
                new_dict[grid[0]], 5)).encode("utf-8"))

    # Close the file
    file.close()


def most_common(counter, n):
    """Retrieve the n most common hashtags from counter. If there are hashtags
    with equal counts, all are included.
    """
    data = counter.most_common()

    # Get individual counts
    counts = set(hashtag[1] for hashtag in counter.most_common())

    # Store the fifth heighest count for the hashtags.=
    try:
        val = sorted(counts, reverse=True)[n - 1]
    except IndexError:
        val = sorted(counts, reverse=True)[-1]

    # Return all hashtags with at least val mentions
    return list(takewhile(lambda x: x[1] >= val, data))


def get_hashtags(tweet):
    """Return all hashtags from tweet."""
    return re.findall(" #([^ ]+) ", tweet)


def get_grid(tweet, grids):
    """Returns the corresponding grid cell from grids for tweet."""
    for grid in grids["features"]:
        try:
            x = tweet["doc"]["coordinates"]["coordinates"][0]
            y = tweet["doc"]["coordinates"]["coordinates"][1]
            if x > grid["properties"]["xmin"] and \
                    x <= grid["properties"]["xmax"] and \
                    y > grid["properties"]["ymin"] and \
                    y <= grid["properties"]["ymax"]:
                return grid["properties"]["id"]
        except BaseException:
            pass

    # If the tweet is on the border of one of the outer grid cells then check
    # which it borders
    for grid in grids["features"]:
        try:
            x = tweet["doc"]["coordinates"]["coordinates"][0]
            y = tweet["doc"]["coordinates"]["coordinates"][1]
            if x >= grid["properties"]["xmin"] and \
                    x <= grid["properties"]["xmax"] and \
                    y >= grid["properties"]["ymin"] and \
                    y <= grid["properties"]["ymax"]:
                return grid["properties"]["id"]
        except BaseException:
            pass
    
    # If the tweet is not in any of the grid cells then return an empty string
    return ""


if __name__ == "__main__":
    main()

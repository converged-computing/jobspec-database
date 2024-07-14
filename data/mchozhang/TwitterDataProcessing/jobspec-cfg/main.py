# -*- encoding: utf-8 -*-
#
# Calculate start and stop position according to MPI rank,
# process the twitter file line by line to obtain data

from mpi4py import MPI
import geo_util
import tweet_util
import json
import sys
import getopt
import time

start_time = time.time()
twitter_filepath = "bigTwitter.json"
melb_filepath = "melbGrid.json"
opts, args = getopt.getopt(sys.argv[1:], "t:m:")

for op, value in opts:
    if op == "-t":
        twitter_filepath = value
    elif op == "-m":
        melb_filepath = value

# initial grid and twitter file info
grid_manager = geo_util.GridManager(melb_filepath)
twitter_file_size = tweet_util.get_file_size(twitter_filepath)

# mpi info
comm = MPI.COMM_WORLD
process_number = comm.size
rank = comm.rank

# compute the start and stop byte
start = (twitter_file_size // process_number) * rank
stop = (twitter_file_size // process_number) * (rank + 1)

with open(twitter_filepath, 'rb') as twitter_file:

    # trim each start line and stop line
    twitter_file.seek(start)
    start_line = twitter_file.readline()

    twitter_file.seek(stop)
    stop_line = twitter_file.readline()

    start += len(start_line)
    stop += len(stop_line) - 1

    cur_pos = start
    twitter_file.seek(start)

    for line in twitter_file:

        cur_pos += len(line)
        line = line.decode("utf-8").strip()
        try:
            # remove "," at the end of the line
            if line.endswith(","):
                line = line[:-1]

            tweet = json.loads(line)

            hashtags = tweet_util.get_hashtags_from_tweet(tweet)
            coordinates = tweet_util.get_coordinates_from_tweet(tweet)

            grid_manager.collect_tweet(hashtags, coordinates)

        except Exception as e:
            continue

        if cur_pos > stop:
            break

# gather the posts counter result
posts_counter = grid_manager.posts_counter
posts_counter_list = comm.gather(posts_counter, root=0)

# gather the hashtags counter result
hashtags_counter_table = grid_manager.hashtags_counter_table
hashtags_counter_table_list = comm.gather(hashtags_counter_table, root=0)

end_time = time.time()
elapsed_time = end_time - start_time
rank_time_tuple_list = comm.gather((rank, elapsed_time), root=0)

if rank == 0:
    # add up all the post counter result, in form of { "A1": 123, "A2": 321 }
    final_posts_counter = geo_util.sum_up_post_counter(posts_counter_list)

    # descendant sorted list of tuples, in form of [(A1, 100), (A2, 99)]
    final_posts_counter_list = geo_util.sort_post_counter(final_posts_counter)

    # add up all the hashtag counter result
    final_hashtags_counter_table = geo_util.sum_up_hashtags_counter(final_posts_counter, hashtags_counter_table_list)

    # print the top 5 hashtags counter and posts of every cell in descendant order
    geo_util.print_hashtags_counter(final_hashtags_counter_table, final_posts_counter_list)

    end_time = time.time()
    for process_rank, elapsed_time in rank_time_tuple_list:
        if process_rank == 0:
            elapsed_time = end_time - start_time
        print("Process {} elapsed time: {} seconds".format(process_rank, elapsed_time))

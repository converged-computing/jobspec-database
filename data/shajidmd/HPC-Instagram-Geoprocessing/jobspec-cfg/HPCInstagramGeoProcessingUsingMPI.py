
# File     : HPCInstagramGeoProcessingUsingMPI.py
# Author   : Shajid Mohammad (shajidm@student.unimelb.edu.au)
# Origin   : Fri Mar 30, 2018
# Purpose  : Cluster and Cloud Computing Assignment 1 – HPC Instagram GeoProcessing

# INTRODUCTION
# --------------
# This Project is to search an Instagram data set (bigInstagram.json) to identify Instagram activity around Melbourne.

# imports
import sys, operator, json, re, ast
from mpi4py import MPI

# ------------------------------------------------------------------------------------------------------------------------

# Displays the result of the Project
def display(reducer):
    results = {}

    for item in reducer:
        for key, value in item.items():
            results[key] = results.get(key, 0) + value

    for resultType in ["postCountPerCell", "postCountForROW", "postCountForCOLUMN"]:

        if resultType == "postCountPerCell":
            print("\n The Ordered (ranked) Total number of Instagram posts made in each box is : ")
        elif resultType == "postCountForROW":
            print("\nThe Ordered (ranked) Total number of Instagram posts made in each row (row based) is : ")
        elif resultType == "postCountForCOLUMN":
            print("\nThe Ordered (ranked) Total number of Instagram posts made in each column (column based) is : ")

        sorted_value = sorted(postResults(resultType, results).items(), key=operator.itemgetter(1))
        sorted_value.reverse()

        for key, value in sorted_value:
            print(key, ": ", value, " posts,")

# ------------------------------------------------------------------------------------------------------------------------

# Diffientiate the counter based on a filter whether to give results per cell, row or column
def postResults(resultType, dictionary):
    if resultType == "postCountPerCell":
        total_counts = dictionary
        del total_counts[None]
        return total_counts

    elif resultType == "postCountForROW":
        row_count = {}

        for key, value in dictionary.items():
            row_count[key[0]] = row_count.get(key[0], 0) + value
        return row_count

    elif resultType == "postCountForCOLUMN":
        column_count = {}

        for key, value in dictionary.items():
            column_count[key[1:]] = column_count.get(key[1:], 0) + value
        return column_count

# ------------------------------------------------------------------------------------------------------------------------

# Counts the Total number of posts in the Instagram file
def instagramFileLineCounter(insta_json):
    counter = -2
    instaFile = open(insta_json, 'r')
    line = instaFile.readline()

    while line:
        counter += 1
        line = instaFile.readline()
    instaFile.close()

    return counter
    
# ------------------------------------------------------------------------------------------------------------------------

# Gets the coordiantes from a JSON
def getCoordinates(string, expression): 

    pos = [string[m.start(0): m.end(0)] for m in re.finditer(expression, string)]
    if pos:
        try:
            return ast.literal_eval(pos[0][14:])
        except Exception:
            return None

# ------------------------------------------------------------------------------------------------------------------------

# Gets the Grid ID for a coordinate
def getGridIDForCoordinates(comm_size, comm_rank, partition, instagramFile, grid_boundary, counter):

    count = {}

    coordinates = []

    if comm_rank == comm_size-1:
        start = int(partition * comm_rank)
        end = counter
    else:
        start = int(partition * comm_rank)
        end = int(partition + partition * comm_rank)

    with open(instagramFile) as f:
        for i in range(0, start):
            f.readline()
        for i in range(start, end):
            result = getCoordinates(f.readline(), '\"coordinates\":\[.*,.*\]')

            if result:
                coordinates.append(result)

    for x, y in coordinates:
        count[position(grid_boundary, x, y)] = count.get(position(grid_boundary, x, y), 0)+1

    return count

# ------------------------------------------------------------------------------------------------------------------------

# Used to get the ID of the cell the coordinates fall in
def position(gridCoordinates, xCoordinate, yCoordinate):
    for key, coordinate in gridCoordinates.items():
        if coordinate[0][0] < yCoordinate <= coordinate[1][0] and coordinate[2][1] < xCoordinate <= coordinate[1][1]:
            return key

# ------------------------------------------------------------------------------------------------------------------------

# Parses the Melborune Grid File for coordinates with respect to their cell          
def parseMelbGridCoordinates(melbGrid_json):
    gridCoordinates = {}
    gridFile = open(melbGrid_json)
    data = json.load(gridFile)

    for coordinates in data['features']:
        gridCoordinates[coordinates['properties']['id']] = coordinates['geometry']['coordinates'][0]

    gridFile.close()
    return gridCoordinates

# ------------------------------------------------------------------------------------------------------------------------
# Main method of the program
def main(argv):

    # Initialize MPI communicator
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    master = 0
    counter = 0

    if rank == master:
        gridCoordinates = parseMelbGridCoordinates(argv[1])
        counter = instagramFileLineCounter(argv[2])
        partition = counter / size
        data = [gridCoordinates, counter, partition]
    else:
        data = None

    data = comm.bcast(data, root=master)
    postCount = getGridIDForCoordinates(size, rank, data[2], argv[2], data[0], data[1])
    reducer = comm.gather(postCount, root=master)

    if rank == master:
        display(reducer)


if __name__ == "__main__":
    main(sys.argv)


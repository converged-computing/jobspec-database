'''
Jack's script for writing all of the COI images at once
And also all the COI pivoted 1/0 csvs
- Jack Deschler
'''

from matplotlib.pyplot import text
import submission_analysis.fetch as fetch
import coi_maps
import coi_dataset
import numpy as np
import pandas as pd
import geopandas as gpd
import os
import copy
import datetime


##### THINGS TO CHANGE ######
to_draw = {
    "Michigan": [('statewide', 'michigan', 'Michigan'),
                 ('/shp/Michigan/ann_arbor.shp', 'ann_arbor', 'Ann Arbor'),
                 ('/shp/Michigan/detroit.shp', 'detroit', 'Detroit'),
                 ('/shp/Michigan/flint.shp', 'flint', 'Flint'),
                 ('/shp/Michigan/grand_rapids.shp', 'grand_rapids', 'Grand Rapids'),
                 ('/shp/Michigan/lansing.shp', 'lansing', 'Lansing'),
                 ('/shp/Michigan/kzoo_bcreek.shp', 'kzoo_bcreek', 'Kalamazoo/Battle Creek'),
                 ('/shp/Michigan/tri_cities.shp', 'tri_cities', 'Tri-Cities'),
                 ('/shp/Michigan/traverse_city.shp', 'traverse_city', 'Traverse City'),
                 ('/shp/Michigan/northern_michigan.shp', 'northern_michigan', 'Northern Michigan')],
    "Missouri": [('statewide', 'missouri', "Missouri"),
                  ('/shp/Missouri/St_Louis_MO.shp', 'stlouis', 'St. Louis'),
                  ('/shp/Missouri/Kansas_City_MO.shp', 'kansascity', 'Kansas City'),
                  ('/shp/Missouri/Springfield_MO.shp', 'springfield', 'Springfield'),
                  ('/shp/Missouri/Jefferson_City_MO.shp', 'jeffersoncity', 'Jefferson City'),
                  ('/shp/Missouri/Columbia_MO.shp', 'columbia', 'Columbia'),],
    "Ohio": [('statewide', 'ohio', 'Ohio')
             ('/shp/Ohio/akron-canton-youngstown.shp', 'akron-canton-youngstown', 'Akron-Canton-Youngstown'),
             ('/shp/Ohio/cleveland-northeastohio.shp', 'cleveland-northeastohio', 'Cleveland-Northeast Ohio'),
             ('/shp/Ohio/northwestohio.shp', 'northwestohio', 'Northwest Ohio'),
             ('/shp/Ohio/appalachiaohio.shp', 'appalachiaohio', 'Appalachian Ohio'),
             ('/shp/Ohio/columbus-centralohio.shp', 'columbus-centralohio', 'Columbus-Central Ohio'),
             ('/shp/Ohio/southwestohio.shp', 'southwestohio', 'Southwest Ohio')],
    "Wisconsin": [('statewide', 'wisconsin', "Wisconsin"),
                  ('/shp/Wisconsin/greatermilwaukee.shp', 'milwaukee', 'Greater Milwaukee'),
                  ('/shp/Wisconsin/RiverFalls-EauClaire.shp', 'riverfalls_eauclaire', 'River Falls - Eau Claire'),
                  ('/shp/Wisconsin/SouthwestWisconsin.shp', 'southwest_wisconsin', 'Southwest Wisconsin'),
                  ('/shp/Wisconsin/WIDaneCo.shp', 'dane_county', 'Dane County')],
    "Texas": [('statewide', 'texas', "Texas"),
               ('/shp/Texas/Houston.shp', 'houston', "Houston"),
               ('/shp/Texas/Austin.shp', 'austin', 'Austin'),
               ('/shp/Texas/Corpus Christi.shp', 'corpuschristi', 'Corpus Christi'),
               ('/shp/Texas/Dallas-FT.shp', 'Dallas', 'Dallas'),
               ('/shp/Texas/San Antonio.shp', 'sanantonio', 'San Antonio')],
    "New Mexico": [('statewide', 'newmexico', "New Mexico"),
                   ('/shp/New Mexico/Albuquerque.shp', 'albuquerque', 'Albuquerque')],



    # Forthcoming
    # "Florida": ('statewide', 'florida', 'Florida'),
    # 'Pennsylvania': ('statewide', 'pennsylvania', 'Pennsylvania'),
}

## actual code
# data is list of (geom, outfile) tuples
def create_coi_maps(state, data):
    if not isinstance(data, list):
        data = [data]
    link = state.lower().replace(" ", "")
    print(f'----------- {state} -------------')
    print(f'{len(data)} set(s) to print in {state}')


    # read the COI dataframe
    ids = f"https://k61e3cz2ni.execute-api.us-east-2.amazonaws.com/prod/submissions/districtr-ids/{link}"
    plan = f"https://k61e3cz2ni.execute-api.us-east-2.amazonaws.com/prod/submissions/csv/{link}?type=plan&length=10000"
    cois = f"https://k61e3cz2ni.execute-api.us-east-2.amazonaws.com/prod/submissions/csv/{link}?type=coi&length=10000"
    written = f"https://k61e3cz2ni.execute-api.us-east-2.amazonaws.com/prod/submissions/csv/{link}?type=written&length=10000"

    if state == 'Michigan':
        ids = "https://o1siz7rw0c.execute-api.us-east-2.amazonaws.com/beta/submissions/districtr-ids/michigan"
        csv_url = "https://o1siz7rw0c.execute-api.us-east-2.amazonaws.com/beta/submissions/csv/michigan"
        plan = csv_url + "?type=plan&length=10000"
        cois = csv_url + "?type=coi&length=10000"
        written = csv_url + "?type=written&length=10000"


    
    _, coi_df, _ = fetch.submissions(ids, plan, cois, written)

    # Need to drop these in Ohio for now
    # if state == "Ohio":
    #     coi_df = coi_df.drop(coi_df[coi_df["first"] == "OOC"].index)

    monday = most_recent_monday(np.datetime64('today'))
    textfile = open(f"./{state.lower().replace(' ', '')}/{state.lower().replace(' ', '')}_info_{monday}.txt", "w")
    textfile.write(f'----------- {state} -------------\n')

    print("Writing Cumulative Dataset")
    coi_df['datetime'] = coi_df['datetime'].apply(np.datetime64)
    cumulative = copy.deepcopy(coi_df[coi_df['datetime'] < monday])
    print(f"{len(cumulative)} submissions through monday")
    cumulative = coi_maps.assignment_to_shape(cumulative)
    if not isinstance(cumulative, pd.DataFrame):
        print(f"Done with {state.upper()}\n")
        textfile.write(f'No COI submissions yet in {state}')
        textfile.close()
        return
    coi_dataset.assignment_to_pivot(coi_df, f'lookup_tables/{state}_{monday}.csv')
    print("Cumulative Dataset Written\n")
    
    print("Writing Weekly Dataset")
    weekly = coi_df[coi_df['datetime'] >= (monday - np.timedelta64(1, 'W'))]
    weekly = weekly[weekly['datetime'] < monday]
    weekly = copy.deepcopy(weekly)
    coi_dataset.assignment_to_pivot(weekly, f'lookup_tables/{state}_weekly_{monday}.csv')
    print(f"{len(weekly)} submissions in the last week")
    weekly = coi_maps.assignment_to_shape(weekly)
    print("Weekly Dataset Written\n")
    
    # make the maps!
    for (geom, outfile, title) in data:
        print(f"Mapping {title}")
        textfile.write("Statewide\n") if geom == 'statewide' else textfile.write(f"{title}\n")

        # figure out if geom is a state name or a shapefile
        osm = False
        clip = state
        if geom != "statewide":
            # have to add the .. bc we have cd'd down a directory
            clip = gpd.read_file(f'../{geom}')
            osm = True
    
        try:
            coi_maps.plot_coi_boundaries(cumulative, clip, osm = osm, outfile = f'{state.lower().replace(" ", "")}/{outfile}_{monday}_boundaries',
                                         show = False, writer = textfile, monday = monday, title = title)
            coi_maps.plot_coi_heatmap(cumulative, clip, osm = osm, outfile = f'{state.lower().replace(" ", "")}/{outfile}_{monday}_heatmap',
                                      show = False, title = title)
        except Exception as e:
            print(f"Could not print {title} due to {e}.")
        textfile.write("\n")
        try:
            coi_maps.plot_coi_boundaries(weekly, clip, osm = osm, outfile = f'{state.lower().replace(" ", "")}/{outfile}_weekly{monday}_boundaries',
                                         show = False, writer = textfile, weekly = True, monday = monday, title = title)
            coi_maps.plot_coi_heatmap(weekly, clip, osm = osm, outfile = f'{state.lower().replace(" ", "")}/{outfile}_weekly{monday}_heatmap',
                                      show = False, title = title)
        except AttributeError as e:
            print(e)
            textfile.write(f" Date Range: {monday - np.timedelta64(7)} - {monday}\n")
            textfile.write(" 0 areas of interest from 0 submissions\n")
            print(f"No new COIs in {title} this week.")
        except Exception as e:
            print(f"Could not print {title} weekly due to {e}.")
        textfile.write("\n\n")

    print(f"Done with {state.upper()}\n")
    textfile.close()

def most_recent_monday(d):
    weekday = d.astype(datetime.datetime).isoweekday()
    return d - np.timedelta64(weekday - 1)

def main():
    monday = str(most_recent_monday(np.datetime64('today')))
    os.mkdir(monday)
    os.chdir(monday)
    os.mkdir("lookup_tables")
    for s in to_draw.keys():
        os.mkdir(s.lower().replace(" ", ""))
        create_coi_maps(s, to_draw[s])
    os.chdir('..')

if __name__ == "__main__":
    main()

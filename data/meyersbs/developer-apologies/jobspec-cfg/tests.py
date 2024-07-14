#!/usr/bin/env python3


#### PYTHON IMPORTS ################################################################################
import csv
import filecmp
import h5py
import os
import shutil
import unittest
from collections import OrderedDict
from pathlib import Path
import unittest.mock as mock


#### PACKAGE IMPORTS ###############################################################################
from src.apologies import classify, _countApologies, _labelApologies
from src.config import getAPIToken, EmptyAPITokenError
from src.deduplicate import deduplicate
from src.delete import delete
from src.developers import _countDeveloperApologies, _flattenDicts, _getDeveloperDicts, \
    _writeToDisk, developerStats
from src.download import download
from src.graphql import _runQuery, runQuery, getRateLimitInfo
from src.helpers import canonicalize, doesPathExist, validateDataDir, parseRepoURL, \
    InvalidGitHubURLError, getDataFilepaths, getSubDirNames, getFilenames, ISSUES_HEADER, \
    COMMITS_HEADER, PULL_REQUESTS_HEADER
from src.info import infoData
from src.preprocess import preprocess, _stripNonWords, _lemmatize
from src.random import _getPopulationFilepaths, _deduplicateHeaders, _getSourceFromFilepath, \
    _getTargetColumns, _filterNonApologies, _getPopulationData, randomSample, ABRIDGED_HEADER
from src.search import search, topRepos
from src.stats import stats


#### GLOBALS #######################################################################################
CWD = os.path.dirname(os.path.realpath(__file__))


#### TEST CASES ####################################################################################
class TestHelpers(unittest.TestCase):
    """
    Test cases for functions in src.helpers.
    """
    def setUp(self):
        """
        Necessary setup for test cases.
        """
        pass


    def tearDown(self):
        """
        Necessary cleanup for these test cases.
        """
        try:
            data_dir = os.path.join(CWD, "test_data/") 
            shutil.rmtree(data_dir)
        except FileNotFoundError:
            pass # We don't want the directory to exist, so this is fine


    def test_canonicalize(self):
        """
        Test src.helpers:canonicalize().
        """
        #### Case 1 -- simple relative path
        input_path = "data/"
        expected = os.path.join(CWD, "data")
        actual = canonicalize(input_path)
        self.assertEqual(expected, actual)

        #### Case 2 -- complex relative path
        input_path = "data/../data/"
        expected = os.path.join(CWD, "data")
        actual = canonicalize(input_path)
        self.assertEqual(expected, actual)


    def test_doesPathExist(self):
        """
        Test src.helpers:doesPathExist().
        """
        #### Case 1 -- relative path that doesn't exist
        input_path = "data/apples/"
        actual = doesPathExist(input_path)
        self.assertFalse(actual)

        #### Case 2 -- absolute path that doesn't exist
        input_path = "data/apples/"
        actual = doesPathExist(canonicalize(input_path))
        self.assertFalse(actual)

        #### Case 3 -- relative path that does exist
        input_path = "data/__init__.py"
        actual = doesPathExist(input_path)
        self.assertTrue(actual)

        #### Case 4 -- absolute path that does exist
        input_path = "data/__init__.py"
        actual = doesPathExist(canonicalize(input_path))
        self.assertTrue(actual)


    def test_validateDataDir(self):
        """
        Test src.helpers:validateDataDir().
        """
        expected = [
            (os.path.join(CWD, "test_data/"), ["commits", "issues", "pull_requests"], []),
            (os.path.join(CWD, "test_data/commits"), [], ["__init__.py"]),
            (os.path.join(CWD, "test_data/issues"), [], ["__init__.py"]),
            (os.path.join(CWD, "test_data/pull_requests"), [], ["__init__.py"])
        ]

        #### Case 1 -- empty data_dir
        # Setup
        data_dir = os.path.join(CWD, "test_data/")
        os.mkdir(data_dir)
        validateDataDir(data_dir)
        # Test
        actual = list(os.walk(data_dir))
        self.assertListEqual(expected, actual)       
        # Cleanup
        shutil.rmtree(data_dir) # Clean up before next test

        #### Case 2 -- data_dir with just issues dir
        # Setup
        data_dir = os.path.join(CWD, "test_data/")
        os.mkdir(data_dir)
        os.mkdir(os.path.join(data_dir, "issues/"))
        validateDataDir(data_dir)
        # Test
        actual = list(os.walk(data_dir))
        self.assertListEqual(expected, actual)
        # Cleanup
        shutil.rmtree(data_dir) # Clean up before next test

        #### Case 3 -- data_dir with just issues dir containing __init__.py
        # Setup
        data_dir = os.path.join(CWD, "test_data/")
        os.mkdir(data_dir)
        os.mkdir(os.path.join(data_dir, "issues/"))
        Path(os.path.join(data_dir, "issues/__init__.py")).touch()
        validateDataDir(data_dir)
        # Test
        actual = list(os.walk(data_dir))
        self.assertListEqual(expected, actual)
        # Cleanup
        shutil.rmtree(data_dir) # Clean up before next test

        #### Case 4 -- data_dir with just commits dir
        # Setup
        data_dir = os.path.join(CWD, "test_data/")
        os.mkdir(data_dir)
        os.mkdir(os.path.join(data_dir, "commits/"))
        validateDataDir(data_dir)
        # Test
        actual = list(os.walk(data_dir))
        self.assertListEqual(expected, actual)
        # Cleanup
        shutil.rmtree(data_dir) # Clean up before next test

        #### Case 5 -- data_dir with just commits dir containing __init__.py
        # Setup
        data_dir = os.path.join(CWD, "test_data/")
        os.mkdir(data_dir)
        os.mkdir(os.path.join(data_dir, "commits/"))
        Path(os.path.join(data_dir, "commits/__init__.py")).touch()
        validateDataDir(data_dir)
        # Test
        actual = list(os.walk(data_dir))
        self.assertListEqual(expected, actual)
        # Cleanup
        shutil.rmtree(data_dir) # Clean up before next test

        #### Case 6 -- data_dir with just pull_requests dir
        # Setup
        data_dir = os.path.join(CWD, "test_data/")
        os.mkdir(data_dir)
        os.mkdir(os.path.join(data_dir, "pull_requests/"))
        validateDataDir(data_dir)
        # Test
        actual = list(os.walk(data_dir))
        self.assertListEqual(expected, actual)
        # Cleanup
        shutil.rmtree(data_dir) # Clean up before next test

        #### Case 7 -- data_dir with just pull_requests dir containing __init__.py
        # Setup
        data_dir = os.path.join(CWD, "test_data/")
        os.mkdir(data_dir)
        os.mkdir(os.path.join(data_dir, "pull_requests/"))
        Path(os.path.join(data_dir, "pull_requests/__init__.py")).touch()
        validateDataDir(data_dir)
        # Test
        actual = list(os.walk(data_dir))
        self.assertListEqual(expected, actual)
        # Cleanup
        shutil.rmtree(data_dir) # Clean up before next test


    def test_getSubDirNames(self):
        """
        Test src.helpers:getSubDirNames().
        """
        # Setup
        data_dir = os.path.join(CWD, "test_files/test_data4/")
        expected_sub_dirs = ["subdir1", "subdir2"]
        # Test
        actual_sub_dirs = getSubDirNames(data_dir)
        self.assertListEqual(expected_sub_dirs, actual_sub_dirs)


    def test_getDataFilepaths(self):
        """
        Test src.helpers:getDataFilepaths().
        """
        # Setup
        data_dir = os.path.join(CWD, "test_files/test_data2/")
        expected_issues_path = os.path.join(CWD, "test_files/test_data2/issues/issues.csv")
        expected_commits_path = os.path.join(CWD, "test_files/test_data2/commits/commits.csv")
        expected_pull_requests_path = os.path.join(CWD, "test_files/test_data2/pull_requests/pull_requests.csv")
        # Test
        actual_issues_path, actual_commits_path, actual_pull_requests_path = getDataFilepaths(data_dir)
        self.assertEqual(expected_issues_path, actual_issues_path)
        self.assertEqual(expected_commits_path, actual_commits_path)
        self.assertEqual(expected_pull_requests_path, actual_pull_requests_path)


    def test_parseRepoURL(self):
        """
        Test src.helpers:parseRepoURL().
        """
        #### Case 1 -- GitHub URL
        input_url = "https://github.com/meyersbs/developer-apologies/"
        expected = ("meyersbs", "developer-apologies")
        actual = parseRepoURL(input_url)
        self.assertTupleEqual(expected, actual)

        #### Case 2 -- Non-GitHub URL
        input_url = "http://www.se.rit.edu/~swen-331/"
        self.assertRaises(InvalidGitHubURLError, parseRepoURL, input_url)


class TestConfig(unittest.TestCase):
    """
    Test cases for functions in src.config.
    """
    def setUp(self):
        """
        Necessary setup for test cases.
        """
        pass


    def test_getAPIToken(self):
        """
        Test src.config:getAPIToken().
        """
        #### Case 1 -- API token exists
        token_path = canonicalize("test_files/test_github_api_token.txt")
        expected = "abc_defG32ef6abd9mP4Qdmr5TY901sN01gF27eU"
        actual = getAPIToken(token_path)
        self.assertEqual(expected, actual)

        #### Case 2 -- blank API token
        token_path = canonicalize("test_files/test_github_api_token_empty.txt")
        self.assertRaises(EmptyAPITokenError, getAPIToken, token_path)


class TestInfo(unittest.TestCase):
    """
    Test cases for functions in src.info.
    """
    def setUp(self):
        """
        Necessary setup for test cases.
        """
        pass


    def test_infoData(self):
        """
        Test src.info:infoData().
        """
        #### Case 1: all paths exist
        # Setup
        data_dir = os.path.join(CWD, "test_files/test_data2/")
        expected = [1, 2, 103, 57, 102, 2, 101]
        # Test
        actual = infoData(data_dir, verbose=False)
        self.assertListEqual(expected, actual)


class TestDelete(unittest.TestCase):
    """
    Test cases for function in src.delete.
    """
    def setUp(self):
        """
        Necessary setup for test cases.
        """
        pass


    def tearDown(self):
        """
        Necessary cleanup for these test cases.
        """
        try:
            data_dir = os.path.join(CWD, "test_data/") 
            shutil.rmtree(data_dir)
        except FileNotFoundError:
            pass # We don't want the directory to exist, so this is fine


    def test_delete(self):
        """
        Test src.delete:delete().
        """
        expected = [
            (os.path.join(CWD, "test_data/"), ["commits", "issues", "pull_requests"], []),
            (os.path.join(CWD, "test_data/commits"), [], ["__init__.py"]),
            (os.path.join(CWD, "test_data/issues"), [], ["__init__.py"]),
            (os.path.join(CWD, "test_data/pull_requests"), [], ["__init__.py"])
        ]

        #### Case 1 -- all files exist
        # Setup
        data_dir = os.path.join(CWD, "test_data/")
        os.mkdir(data_dir)
        validateDataDir(data_dir) # We can use this because the test already passed
        Path(os.path.join(data_dir, "issues/issues.csv")).touch()
        Path(os.path.join(data_dir, "commits/commits.csv")).touch()
        Path(os.path.join(data_dir, "pull_requests/pull_requests.csv")).touch()
        # Test
        delete(data_dir)
        actual = list(os.walk(data_dir))
        self.assertListEqual(expected, actual)
        # Cleanup
        shutil.rmtree(data_dir) # Clean up before next test

        #### Case 2 -- everything but issues.csv exists
        # Setup
        data_dir = os.path.join(CWD, "test_data/")
        os.mkdir(data_dir)
        validateDataDir(data_dir) # We can use this because the test already passed
        Path(os.path.join(data_dir, "commits/commits.csv")).touch()
        Path(os.path.join(data_dir, "pull_requests/pull_requests.csv")).touch()
        # Test
        delete(data_dir)
        actual = list(os.walk(data_dir))
        self.assertListEqual(expected, actual)
        # Cleanup
        shutil.rmtree(data_dir) # Clean up before next test

        #### Case 3 -- everything but commits.csv exists
        # Setup
        data_dir = os.path.join(CWD, "test_data/")
        os.mkdir(data_dir)
        validateDataDir(data_dir) # We can use this because the test already passed
        Path(os.path.join(data_dir, "issues/issues.csv")).touch()
        Path(os.path.join(data_dir, "pull_requests/pull_requests.csv")).touch()
        # Test
        delete(data_dir)
        actual = list(os.walk(data_dir))
        self.assertListEqual(expected, actual)
        # Cleanup
        shutil.rmtree(data_dir) # Clean up before next test

        #### Case 4 -- everything but pull_requests.csv exists
        # Setup
        data_dir = os.path.join(CWD, "test_data/")
        os.mkdir(data_dir)
        validateDataDir(data_dir) # We can use this because the test already passed
        Path(os.path.join(data_dir, "issues/issues.csv")).touch()
        Path(os.path.join(data_dir, "commits/commits.csv")).touch()
        # Test
        delete(data_dir)
        actual = list(os.walk(data_dir))
        self.assertListEqual(expected, actual)
        # Cleanup
        shutil.rmtree(data_dir) # Clean up before next test


class TestGraphQL(unittest.TestCase):
    """
    Test cases for function in src.graphql.
    """
    def setUp(self):
        """
        Necessary setup for test cases.
        """
        pass


    def test__runQuery(self):
        """
        Test src.graphql:_runQuery().
        """
        #### Case 1 -- dynamic, valid query
        input_query = """
        {
            viewer { login }
            rateLimit {
                limit
                cost
                remaining
                resetAt
            }
        }
        """
        actual = _runQuery(input_query)
        self.assertTrue("data" in actual.keys())
        self.assertTrue("viewer" in actual["data"].keys())
        self.assertTrue("login" in actual["data"]["viewer"].keys())
        self.assertTrue("rateLimit" in actual["data"].keys())
        self.assertTrue("limit" in actual["data"]["rateLimit"].keys())
        self.assertTrue("cost" in actual["data"]["rateLimit"].keys())
        self.assertTrue("remaining" in actual["data"]["rateLimit"].keys())
        self.assertTrue("resetAt" in actual["data"]["rateLimit"].keys())

        #### Case 2 -- static, valid query
        input_query = """
        query {
            repository(owner:"meyersbs", name:"tvdb-dl-nfo") {
                issue(number:1) {
                    author { login }
                    title
                    assignees { totalCount }
                    createdAt
                    bodyText
                    state
                    url
                }
            }
        }
        """
        expected = {
            "data": {
                "repository": {
                    "issue": {
                        "author": {
                            "login": "meyersbs"
                        },
                        "title": "Ampersands in Metadata",
                        "assignees": {
                            "totalCount": 1
                        },
                        "createdAt": "2019-05-11T20:58:59Z",
                        "bodyText": "If a show has an ampersand (&) in its name or description, the"
                                    " following error will occur:\n    PHP Warning:  SimpleXMLEleme"
                                    "nt::addChild(): unterminated entity reference\n\nThe tvshow.nf"
                                    "o file is still generated, but the field containing the ampers"
                                    "and will be empty.",
                        "state": "CLOSED",
                        "url": "https://github.com/meyersbs/tvdb-dl-nfo/issues/1"
                    }
                }
            }
        }
        actual = _runQuery(input_query)
        self.assertDictEqual(expected, actual)

        #### Case 3 -- invalid query, status code != 200
        # This can't really be tested. GitHub's GraphQL API responds with status code = 200 even
        # when the query was malformed. It responds with a JSON dict with the key "errors", which
        # we can test.

        #### Case 4 -- invalid query, returning errors
        input_query = "query {}"
        expected = None
        actual = _runQuery(input_query)
        self.assertEqual(expected, actual)


    def test_runQuery(self):
        """
        Test src.graphql:runQuery().
        """
        #### Case 1 -- data_types="issues"
        input_repo_owner = "meyersbs"
        input_repo_name = "tvdb-dl-nfo"
        input_data_types = "issues"
        expected = {
            "data": {
                "repository": {
                    "name": "tvdb-dl-nfo",
                    "owner": {
                        "login": "meyersbs"
                    },
                    "issues": {
                        "totalCount": 1,
                        "edges": [
                            {
                                "node": {
                                    "id": "MDU6SXNzdWU0NDMwMzU3MTU=",
                                    "number": 1,
                                    "title": "Ampersands in Metadata",
                                    "author": {
                                        "login": "meyersbs"
                                    },
                                    "createdAt": "2019-05-11T20:58:59Z",
                                    "url": "https://github.com/meyersbs/tvdb-dl-nfo/issues/1",
                                    "bodyText": "If a show has an ampersand (&) in its name or desc"
                                                "ription, the following error will occur:\n    PHP "
                                                "Warning:  SimpleXMLElement::addChild(): unterminat"
                                                "ed entity reference\n\nThe tvshow.nfo file is stil"
                                                "l generated, but the field containing the ampersan"
                                                "d will be empty.",
                                    "comments": {
                                        "totalCount": 0,
                                        "edges": [],
                                        "pageInfo": {
                                            "startCursor": None,
                                            "endCursor": None,
                                            "hasNextPage": False
                                        }
                                    }
                                }
                            }
                        ]
                    }
                }
            }
        }
        actual = runQuery(input_repo_owner, input_repo_name, input_data_types)
        self.assertDictEqual(expected, actual)

        #### Case 2 -- data_types="issues", with over 100 comments
        input_repo_owner = "meyersbs"
        input_repo_name = "developer-apologies" # Yes, that's this repo!
        input_data_types = "issues"
        actual = runQuery(input_repo_owner, input_repo_name, input_data_types)
        self.assertEqual(3, actual["data"]["repository"]["issues"]["totalCount"])
        issues = actual["data"]["repository"]["issues"]["edges"]
        for issue in issues:
            if issue["node"]["number"] == 1:
                self.assertEqual("Test Issue with Over 100 Comments", issue["node"]["title"])
                self.assertEqual("meyersbs", issue["node"]["author"]["login"])
                self.assertEqual(101, len(issue["node"]["comments"]["edges"]))
                self.assertEqual("2021-08-19T21:25:39Z", issue["node"]["createdAt"])
            elif issue["node"]["number"] == 2:
                self.assertEqual("Test Issue", issue["node"]["title"])
                self.assertEqual("meyersbs", issue["node"]["author"]["login"])
                self.assertEqual(2, len(issue["node"]["comments"]["edges"]))
                self.assertEqual("2021-08-19T21:46:16Z", issue["node"]["createdAt"])
                comments = issue["node"]["comments"]["edges"]
                for comment in comments:
                    if comment["node"]["id"] == "IC_kwDOFyefKs41x4vo":
                        self.assertEqual("Dummy comment.", comment["node"]["bodyText"])
                        self.assertEqual("meyersbs", comment["node"]["author"]["login"])
                        self.assertEqual("2021-08-19T21:46:23Z", comment["node"]["createdAt"])
                    elif comment["node"]["id"] == "IC_kwDOFyefKs41zfVm":
                        self.assertEqual(
                            "Sorry, but I figured I'd add a data point. Sorrynotsorry.",
                            comment["node"]["bodyText"]
                        )
                        self.assertEqual("andymeneely", comment["node"]["author"]["login"])
                        self.assertEqual("2021-08-20T13:23:06Z", comment["node"]["createdAt"])

        #### Case 3 -- data_types="pull_requests"
        input_repo_owner = "meyersbs"
        input_repo_name = "developer-apologies" # Yes, that's this repo!
        input_data_types = "pull_requests"
        actual = runQuery(input_repo_owner, input_repo_name, input_data_types)
        pull_requests = actual["data"]["repository"]["pullRequests"]["edges"]
        self.assertEqual(3, len(pull_requests))
        self.assertEqual("Update README.md", pull_requests[0]["node"]["title"])
        self.assertEqual("andymeneely", pull_requests[0]["node"]["author"]["login"])
        self.assertEqual("", pull_requests[0]["node"]["bodyText"])
        self.assertEqual("2021-08-23T14:33:22Z", pull_requests[0]["node"]["createdAt"])
        comments = pull_requests[0]["node"]["comments"]["edges"]
        self.assertEqual(101, len(comments))
        for comment in comments:
            self.assertEqual("meyersbs", comment["node"]["author"]["login"])

        #### Case 4 -- data_types="commits"
        input_repo_owner = "meyersbs"
        input_repo_name = "tvdb-dl-nfo"
        input_data_types = "commits"
        expected = None
        actual = runQuery(input_repo_owner, input_repo_name, input_data_types)
        commits = actual["data"]["repository"]["defaultBranchRef"]["target"]["history"]["edges"]
        self.assertEqual(12, len(commits))
        for commit in commits:
            if commit["node"]["oid"] == "5b2009b8db3299cdb810b20caaaea88adb5ebe08":
                self.assertEqual(1, commit["node"]["additions"])
                self.assertEqual(1, commit["node"]["deletions"])
                self.assertEqual("2019-11-27T19:09:42Z", commit["node"]["committedDate"])
                self.assertEqual("Update README.md", commit["node"]["messageHeadline"])
                self.assertEqual("", commit["node"]["messageBody"])
                self.assertEqual(1, commit["node"]["comments"]["totalCount"])
                self.assertEqual(
                    "Dummy comment.",
                    commit["node"]["comments"]["edges"][0]["node"]["bodyText"]
                )
                self.assertEqual(
                    "2021-08-24T12:52:30Z",
                    commit["node"]["comments"]["edges"][0]["node"]["createdAt"]
                )
                break

        #### Case 5 -- data_types="all"
        input_repo_owner = "meyersbs"
        input_repo_name = "tvdb-dl-nfo"
        input_data_types = "all"
        expected = None
        expected_issues = {
            "data": {
                "repository": {
                    "name": "tvdb-dl-nfo",
                    "owner": {
                        "login": "meyersbs"
                    },
                    "issues": {
                        "totalCount": 1,
                        "edges": [
                            {
                                "node": {
                                    "id": "MDU6SXNzdWU0NDMwMzU3MTU=",
                                    "number": 1,
                                    "title": "Ampersands in Metadata",
                                    "author": {
                                        "login": "meyersbs"
                                    },
                                    "createdAt": "2019-05-11T20:58:59Z",
                                    "url": "https://github.com/meyersbs/tvdb-dl-nfo/issues/1",
                                    "bodyText": "If a show has an ampersand (&) in its name or desc"
                                                "ription, the following error will occur:\n    PHP "
                                                "Warning:  SimpleXMLElement::addChild(): unterminat"
                                                "ed entity reference\n\nThe tvshow.nfo file is stil"
                                                "l generated, but the field containing the ampersan"
                                                "d will be empty.",
                                    "comments": {
                                        "totalCount": 0,
                                        "edges": [],
                                        "pageInfo": {
                                            "startCursor": None,
                                            "endCursor": None,
                                            "hasNextPage": False
                                        }
                                    }
                                }
                            }
                        ]
                    }
                }
            }
        }
        expected_commits = {
            "data": {
                "repository": {
                    "name": "tvdb-dl-nfo",
                    "owner": {
                        "login": "meyersbs"
                    },
                    "defaultBranchRef": {
                        "target": {
                            "history": {
                                "edges": [
                                    {
                                        "node": {
                                            "oid": "5b2009b8db3299cdb810b20caaaea88adb5ebe08",
                                            "author": {"user": {"login": "meyersbs"}},
                                            "additions": 1, "deletions": 1,
                                            "committedDate": "2019-11-27T19:09:42Z",
                                            "url": "https://github.com/meyersbs/tvdb-dl-nfo/commit/5b2009b8db3299cdb810b20caaaea88adb5ebe08",
                                            "messageHeadline": "Update README.md", "messageBody": "",
                                            "comments": {
                                                "totalCount": 1,
                                                "edges": [
                                                    {
                                                        "node": {
                                                        	"author": {"login": "meyersbs"},
                                                        	"bodyText": "Dummy comment.",
                                                        	"createdAt": "2021-08-24T12:52:30Z",
                                                        	"url": "https://github.com/meyersbs/tvdb-dl-nfo/commit/5b2009b8db3299cdb810b20caaaea88adb5ebe08#r55353873"
                                                        }
                                                    }
                                                ],
                                                "pageInfo": {
                                                    "startCursor": "Y3Vyc29yOnYyOpHOA0yiEQ==",
                                                    "endCursor": "Y3Vyc29yOnYyOpHOA0yiEQ==",
                                                    "hasNextPage": False
                                                }
                                            }
                                        }
                                    },
                                    {
                                        "node": {
                                            "oid": "ba20e4c9218d445ab74ff26855e6bed2f3c4c5d6",
                                            "author": {"user": {"login": "meyersbs"}},
                                            "additions": 6, "deletions": 2,
                                            "committedDate": "2019-11-27T19:03:34Z",
                                            "url": "https://github.com/meyersbs/tvdb-dl-nfo/commit/ba20e4c9218d445ab74ff26855e6bed2f3c4c5d6",
                                            "messageHeadline": "Update README.md", "messageBody": "",
                                            "comments": {
                                                "totalCount": 0, "edges": [],
                                                "pageInfo": {"startCursor": None, "endCursor": None, "hasNextPage": False}
                                            }
                                        }
                                    }, 
                                    {
                                        "node": {
                                            "oid": "f307305e5a12208baa4cb01f188e8fa20d7a6ef3",
                                            "author": {"user": {"login": "meyersbs"}},
                                            "additions": 3, "deletions": 3,
                                            "committedDate": "2019-05-11T21:04:23Z",
                                            "url": "https://github.com/meyersbs/tvdb-dl-nfo/commit/f307305e5a12208baa4cb01f188e8fa20d7a6ef3",
                                            "messageHeadline": "Fix #1", "messageBody": "",
                                            "comments": {
                                                "totalCount": 0, "edges": [],
                                                "pageInfo": {"startCursor": None, "endCursor": None, "hasNextPage": False}
                                            }
                                        }
                                    },
                                    {
                                        "node": {
                                            "oid": "c399298846a2bcdbc4daa53076b5f9899d8f916b",
                                            "author": {"user": {"login": "meyersbs"}},
                                            "additions": 16, "deletions": 13,
                                            "committedDate": "2019-05-11T20:52:00Z",
                                            "url": "https://github.com/meyersbs/tvdb-dl-nfo/commit/c399298846a2bcdbc4daa53076b5f9899d8f916b",
                                            "messageHeadline": "Update README.md", "messageBody": "",
                                            "comments": {
                                                "totalCount": 0, "edges": [],
                                                "pageInfo": {"startCursor": None, "endCursor": None, "hasNextPage": False}
                                            }
                                        }
                                    },
                                    {
                                        "node": {
                                            "oid": "23af721b3f70cdde0bcc3dc58ba3750dbab34b46",
                                            "author": {"user": {"login": "meyersbs"}},
                                            "additions": 6, "deletions": 3,
                                            "committedDate": "2019-05-11T20:51:50Z",
                                            "url": "https://github.com/meyersbs/tvdb-dl-nfo/commit/23af721b3f70cdde0bcc3dc58ba3750dbab34b46",
                                            "messageHeadline": "Read API Key from file rather than CLI.", "messageBody": "",
                                            "comments": {
                                                "totalCount": 0, "edges": [],
                                                "pageInfo": {"startCursor": None, "endCursor": None, "hasNextPage": False}
                                            }
                                        }
                                    },
                                    {
                                        "node": {
                                            "oid": "bd163c63771e2e314470ce36a251b8e8ab9ce712",
                                            "author": {"user": {"login": "meyersbs"}},
                                            "additions": 13, "deletions": 3,
                                            "committedDate": "2019-05-11T20:51:19Z",
                                            "url": "https://github.com/meyersbs/tvdb-dl-nfo/commit/bd163c63771e2e314470ce36a251b8e8ab9ce712",
                                            "messageHeadline": "Change directory structure. Create apikey.txt", "messageBody": "",
                                            "comments": {
                                                "totalCount": 0, "edges": [],
                                                "pageInfo": {"startCursor": None, "endCursor": None, "hasNextPage": False}
                                            }
                                        }
                                    },
                                    {
                                        "node": {
                                            "oid": "09929a23b30307ebbb426637d420b69216aa9772",
                                            "author": {"user": {"login": "meyersbs"}},
                                            "additions": 10, "deletions": 0,
                                            "committedDate": "2019-05-07T23:45:44Z",
                                            "url": "https://github.com/meyersbs/tvdb-dl-nfo/commit/09929a23b30307ebbb426637d420b69216aa9772",
                                            "messageHeadline": "Create install.sh", "messageBody": "",
                                            "comments": {
                                                "totalCount": 0, "edges": [],
                                                "pageInfo": {"startCursor": None, "endCursor": None, "hasNextPage": False}
                                            }
                                        }
                                    },
                                    {
                                        "node": {
                                            "oid": "eb9c54a516ed2ae17b9b9b8ad22d854f9bf60308",
                                            "author": {"user": {"login": "meyersbs"}},
                                            "additions": 59, "deletions": 0,
                                            "committedDate": "2019-05-07T23:45:06Z",
                                            "url": "https://github.com/meyersbs/tvdb-dl-nfo/commit/eb9c54a516ed2ae17b9b9b8ad22d854f9bf60308",
                                            "messageHeadline": "Create tvdb-dl-nfo.php", "messageBody": "",
                                            "comments": {
                                                "totalCount": 0, "edges": [],
                                                "pageInfo": {"startCursor": None, "endCursor": None, "hasNextPage": False}
                                            }
                                        }
                                    },
                                    {
                                        "node": {
                                            "oid": "110efd9108faee147fa2430702999312d68f2329",
                                            "author": {"user": {"login": "meyersbs"}},
                                            "additions": 12, "deletions": 1,
                                            "committedDate": "2019-05-07T23:41:41Z",
                                            "url": "https://github.com/meyersbs/tvdb-dl-nfo/commit/110efd9108faee147fa2430702999312d68f2329",
                                            "messageHeadline": "Update README.md", "messageBody": "",
                                            "comments": {
                                                "totalCount": 0, "edges": [],
                                                "pageInfo": {"startCursor": None, "endCursor": None, "hasNextPage": False}
                                            }
                                        }
                                    },
                                    {
                                        "node": {
                                            "oid": "1445c6376609dbf6c6017b19ed418d1cd73f2f6e",
                                            "author": {"user": {"login": "meyersbs"}},
                                            "additions": 30, "deletions": 4,
                                            "committedDate": "2019-05-07T23:38:41Z",
                                            "url": "https://github.com/meyersbs/tvdb-dl-nfo/commit/1445c6376609dbf6c6017b19ed418d1cd73f2f6e",
                                            "messageHeadline": "Update README.md", "messageBody": "",
                                            "comments": {
                                                "totalCount": 0, "edges": [],
                                                "pageInfo": {"startCursor": None, "endCursor": None, "hasNextPage": False}
                                            }
                                        }
                                    },
                                    {
                                        "node": {
                                            "oid": "ae38a7f77d211c7678d1a518e797d0668598b472",
                                            "author": {"user": {"login": "meyersbs"}},
                                            "additions": 78, "deletions": 1,
                                            "committedDate": "2019-05-07T20:01:02Z",
                                            "url": "https://github.com/meyersbs/tvdb-dl-nfo/commit/ae38a7f77d211c7678d1a518e797d0668598b472",
                                            "messageHeadline": "Update README.md", "messageBody": "",
                                            "comments": {
                                                "totalCount": 0, "edges": [],
                                                "pageInfo": {"startCursor": None, "endCursor": None, "hasNextPage": False}
                                            }
                                        }
                                    },
                                    {
                                        "node": {
                                            "oid": "75614c09991b4313b1b999971aadd1d6d38f6ce7",
                                            "author": {"user": {"login": "meyersbs"}},
                                            "additions": 23, "deletions": 0,
                                            "committedDate": "2019-05-07T19:32:43Z",
                                            "url": "https://github.com/meyersbs/tvdb-dl-nfo/commit/75614c09991b4313b1b999971aadd1d6d38f6ce7",
                                            "messageHeadline": "Initial commit", "messageBody": "",
                                            "comments": {
                                                "totalCount": 0, "edges": [],
                                                "pageInfo": {"startCursor": None, "endCursor": None, "hasNextPage": False}
                                            }
                                        }
                                    }
                                ]
                            }
                        }
                    }
                }
            }
        }
        expected_pull_requests = {
            "data": {
                "repository": {
                    "name": "tvdb-dl-nfo",
                    "owner": {
                        "login": "meyersbs"
                    },
                    "pullRequests": {
                        "totalCount": 0,
                        "edges": []
                    }
                }
            }
        }
        actual = runQuery(input_repo_owner, input_repo_name, input_data_types)
        self.assertDictEqual(expected_issues, actual[0])
        self.assertDictEqual(expected_commits, actual[1])
        self.assertDictEqual(expected_pull_requests, actual[2])

        #### Case 6 -- invalid data_types
        input_repo_owner = "meyersbs"
        input_repo_name = "tvdb-dl-nfo"
        input_data_types = "apples"
        expected = None
        actual = runQuery(input_repo_owner, input_repo_name, input_data_types)
        self.assertEqual(expected, actual)


    def test_getRateLimitInfo(self):
        """
        Test src.graphql:getRateLimitInfo().
        """
        actual = getRateLimitInfo()
        self.assertTrue("data" in actual.keys())
        self.assertTrue("viewer" in actual["data"].keys())
        self.assertTrue("login" in actual["data"]["viewer"].keys())
        self.assertTrue("rateLimit" in actual["data"].keys())
        self.assertTrue("limit" in actual["data"]["rateLimit"].keys())
        self.assertTrue("cost" in actual["data"]["rateLimit"].keys())
        self.assertTrue("remaining" in actual["data"]["rateLimit"].keys())
        self.assertTrue("resetAt" in actual["data"]["rateLimit"].keys())


class TestDownload(unittest.TestCase):
    """
    Test cases for function in src.download.
    """
    def setUp(self):
        """
        Necessary setup for test cases.
        """
        pass


    def tearDown(self):
        """
        Necessary cleanup for these test cases.
        """
        try:
            data_dir = os.path.join(CWD, "test_data/") 
            shutil.rmtree(data_dir)
        except FileNotFoundError:
            pass # We don't want the directory to exist, so this is fine


    def test_download(self):
        """
        Test src.download:download().
        """
        #### Case 1 -- data_types="issues"
        # Setup
        input_repo_file = os.path.join(CWD, "test_files/repo_lists/test_repos_2.txt")
        input_data_dir = os.path.join(CWD, "test_data/")
        os.mkdir(input_data_dir)
        input_data_types = "issues"
        expected = [
            [
                "REPO_URL", "REPO_NAME", "REPO_OWNER", "ISSUE_NUMBER", "ISSUE_CREATION_DATE",
                "ISSUE_AUTHOR", "ISSUE_TITLE", "ISSUE_URL", "ISSUE_TEXT", "COMMENT_CREATION_DATE",
                "COMMENT_AUTHOR", "COMMENT_URL", "COMMENT_TEXT"
            ],
            [
                "https://github.com/meyersbs/tvdb-dl-nfo", "tvdb-dl-nfo", "meyersbs", "1",
                "2019-05-11T20:58:59Z", "meyersbs", "Ampersands in Metadata",
                "https://github.com/meyersbs/tvdb-dl-nfo/issues/1",
                "If a show has an ampersand (&) in its name or description, the following error wil"
                "l occur:\n    PHP Warning:  SimpleXMLElement::addChild(): unterminated entity refe"
                "rence\n\nThe tvshow.nfo file is still generated, but the field containing the ampe"
                "rsand will be empty.", "", "", "", ""
            ]
        ]
        # Test
        download(input_repo_file, input_data_dir, input_data_types, overwrite=True)
        actual = list()
        with open(os.path.join(input_data_dir, "issues/issues.csv"), "r") as f:
            csv_reader = csv.reader(f, delimiter=",", quoting=csv.QUOTE_MINIMAL, quotechar="\"")
            for entry in csv_reader:
                actual.append(entry)
        self.assertListEqual(expected, actual)
        # Cleanup
        shutil.rmtree(input_data_dir)

        #### Case 2 -- data_types="issues", with over 100 comments
        # Setup
        input_repo_file = os.path.join(CWD, "test_files/repo_lists/test_repos_3.txt")
        input_data_dir = os.path.join(CWD, "test_data/")
        os.mkdir(input_data_dir)
        input_data_types = "issues"
        expected_issues_3 = [
            ["REPO_URL", "REPO_NAME", "REPO_OWNER", "ISSUE_NUMBER", "ISSUE_CREATION_DATE",
             "ISSUE_AUTHOR", "ISSUE_TITLE", "ISSUE_URL", "ISSUE_TEXT", "COMMENT_CREATION_DATE",
             "COMMENT_AUTHOR", "COMMENT_URL", "COMMENT_TEXT"],
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "1",
             "2021-08-19T21:25:39Z", "meyersbs", "Test Issue with Over 100 Comments",
             "https://github.com/meyersbs/developer-apologies/issues/1", "Test Issue with Over 100 Comments.",
             "2021-08-19T21:25:42Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/issues/1#issuecomment-902258838", "1"],
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "1",
             "2021-08-19T21:25:39Z", "meyersbs", "Test Issue with Over 100 Comments",
             "https://github.com/meyersbs/developer-apologies/issues/1", "Test Issue with Over 100 Comments.",
             "2021-08-19T21:25:44Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/issues/1#issuecomment-902258857", "2"],
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "1",
             "2021-08-19T21:25:39Z", "meyersbs", "Test Issue with Over 100 Comments",
             "https://github.com/meyersbs/developer-apologies/issues/1", "Test Issue with Over 100 Comments.",
             "2021-08-19T21:25:46Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/issues/1#issuecomment-902258876", "3"],
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "1",
             "2021-08-19T21:25:39Z", "meyersbs", "Test Issue with Over 100 Comments",
             "https://github.com/meyersbs/developer-apologies/issues/1", "Test Issue with Over 100 Comments.",
             "2021-08-19T21:25:48Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/issues/1#issuecomment-902258891", "4"],
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "1",
             "2021-08-19T21:25:39Z", "meyersbs", "Test Issue with Over 100 Comments",
             "https://github.com/meyersbs/developer-apologies/issues/1", "Test Issue with Over 100 Comments.",
             "2021-08-19T21:25:55Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/issues/1#issuecomment-902258949", "5"],
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "1",
             "2021-08-19T21:25:39Z", "meyersbs", "Test Issue with Over 100 Comments",
             "https://github.com/meyersbs/developer-apologies/issues/1", "Test Issue with Over 100 Comments.",
             "2021-08-19T21:25:58Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/issues/1#issuecomment-902258971", "6"],
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "1",
             "2021-08-19T21:25:39Z", "meyersbs", "Test Issue with Over 100 Comments",
             "https://github.com/meyersbs/developer-apologies/issues/1", "Test Issue with Over 100 Comments.",
             "2021-08-19T21:26:01Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/issues/1#issuecomment-902258993", "7"],
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "1",
             "2021-08-19T21:25:39Z", "meyersbs", "Test Issue with Over 100 Comments",
             "https://github.com/meyersbs/developer-apologies/issues/1", "Test Issue with Over 100 Comments.",
             "2021-08-19T21:26:03Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/issues/1#issuecomment-902259007", "8"],
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "1",
             "2021-08-19T21:25:39Z", "meyersbs", "Test Issue with Over 100 Comments",
             "https://github.com/meyersbs/developer-apologies/issues/1", "Test Issue with Over 100 Comments.",
             "2021-08-19T21:26:05Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/issues/1#issuecomment-902259028", "9"],
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "1",
             "2021-08-19T21:25:39Z", "meyersbs", "Test Issue with Over 100 Comments",
             "https://github.com/meyersbs/developer-apologies/issues/1", "Test Issue with Over 100 Comments.",
             "2021-08-19T21:26:08Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/issues/1#issuecomment-902259059", "10"],
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "1",
             "2021-08-19T21:25:39Z", "meyersbs", "Test Issue with Over 100 Comments",
             "https://github.com/meyersbs/developer-apologies/issues/1", "Test Issue with Over 100 Comments.",
             "2021-08-19T21:26:10Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/issues/1#issuecomment-902259089", "11"],
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "1",
             "2021-08-19T21:25:39Z", "meyersbs", "Test Issue with Over 100 Comments",
             "https://github.com/meyersbs/developer-apologies/issues/1", "Test Issue with Over 100 Comments.",
             "2021-08-19T21:26:12Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/issues/1#issuecomment-902259116", "12"],
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "1",
             "2021-08-19T21:25:39Z", "meyersbs", "Test Issue with Over 100 Comments",
             "https://github.com/meyersbs/developer-apologies/issues/1", "Test Issue with Over 100 Comments.",
             "2021-08-19T21:26:15Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/issues/1#issuecomment-902259138", "13"],
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "1",
             "2021-08-19T21:25:39Z", "meyersbs", "Test Issue with Over 100 Comments",
             "https://github.com/meyersbs/developer-apologies/issues/1", "Test Issue with Over 100 Comments.",
             "2021-08-19T21:26:18Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/issues/1#issuecomment-902259168", "14"],
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "1",
             "2021-08-19T21:25:39Z", "meyersbs", "Test Issue with Over 100 Comments",
             "https://github.com/meyersbs/developer-apologies/issues/1", "Test Issue with Over 100 Comments.",
             "2021-08-19T21:26:21Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/issues/1#issuecomment-902259190", "15"],
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "1",
             "2021-08-19T21:25:39Z", "meyersbs", "Test Issue with Over 100 Comments",
             "https://github.com/meyersbs/developer-apologies/issues/1", "Test Issue with Over 100 Comments.",
             "2021-08-19T21:26:25Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/issues/1#issuecomment-902259222", "16"],
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "1",
             "2021-08-19T21:25:39Z", "meyersbs", "Test Issue with Over 100 Comments",
             "https://github.com/meyersbs/developer-apologies/issues/1", "Test Issue with Over 100 Comments.",
             "2021-08-19T21:26:27Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/issues/1#issuecomment-902259247", "17"],
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "1",
             "2021-08-19T21:25:39Z", "meyersbs", "Test Issue with Over 100 Comments",
             "https://github.com/meyersbs/developer-apologies/issues/1", "Test Issue with Over 100 Comments.",
             "2021-08-19T21:26:29Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/issues/1#issuecomment-902259270", "18"],
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "1",
             "2021-08-19T21:25:39Z", "meyersbs", "Test Issue with Over 100 Comments",
             "https://github.com/meyersbs/developer-apologies/issues/1", "Test Issue with Over 100 Comments.",
             "2021-08-19T21:26:32Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/issues/1#issuecomment-902259299", "19"],
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "1",
             "2021-08-19T21:25:39Z", "meyersbs", "Test Issue with Over 100 Comments",
             "https://github.com/meyersbs/developer-apologies/issues/1", "Test Issue with Over 100 Comments.",
             "2021-08-19T21:26:35Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/issues/1#issuecomment-902259322", "20"],
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "1",
             "2021-08-19T21:25:39Z", "meyersbs", "Test Issue with Over 100 Comments",
             "https://github.com/meyersbs/developer-apologies/issues/1", "Test Issue with Over 100 Comments.",
             "2021-08-19T21:26:38Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/issues/1#issuecomment-902259347", "21"],
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "1",
             "2021-08-19T21:25:39Z", "meyersbs", "Test Issue with Over 100 Comments",
             "https://github.com/meyersbs/developer-apologies/issues/1", "Test Issue with Over 100 Comments.",
             "2021-08-19T21:26:40Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/issues/1#issuecomment-902259361", "22"],
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "1",
             "2021-08-19T21:25:39Z", "meyersbs", "Test Issue with Over 100 Comments",
             "https://github.com/meyersbs/developer-apologies/issues/1", "Test Issue with Over 100 Comments.",
             "2021-08-19T21:26:42Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/issues/1#issuecomment-902259385", "23"],
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "1",
             "2021-08-19T21:25:39Z", "meyersbs", "Test Issue with Over 100 Comments",
             "https://github.com/meyersbs/developer-apologies/issues/1", "Test Issue with Over 100 Comments.",
             "2021-08-19T21:26:45Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/issues/1#issuecomment-902259408", "24"],
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "1",
             "2021-08-19T21:25:39Z", "meyersbs", "Test Issue with Over 100 Comments",
             "https://github.com/meyersbs/developer-apologies/issues/1", "Test Issue with Over 100 Comments.",
             "2021-08-19T21:26:48Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/issues/1#issuecomment-902259438", "25"],
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "1",
             "2021-08-19T21:25:39Z", "meyersbs", "Test Issue with Over 100 Comments",
             "https://github.com/meyersbs/developer-apologies/issues/1", "Test Issue with Over 100 Comments.",
             "2021-08-19T21:26:51Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/issues/1#issuecomment-902259461", "26"],
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "1",
             "2021-08-19T21:25:39Z", "meyersbs", "Test Issue with Over 100 Comments",
             "https://github.com/meyersbs/developer-apologies/issues/1", "Test Issue with Over 100 Comments.",
             "2021-08-19T21:26:54Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/issues/1#issuecomment-902259491", "27"],
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "1",
             "2021-08-19T21:25:39Z", "meyersbs", "Test Issue with Over 100 Comments",
             "https://github.com/meyersbs/developer-apologies/issues/1", "Test Issue with Over 100 Comments.",
             "2021-08-19T21:26:56Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/issues/1#issuecomment-902259522", "28"],
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "1",
             "2021-08-19T21:25:39Z", "meyersbs", "Test Issue with Over 100 Comments",
             "https://github.com/meyersbs/developer-apologies/issues/1", "Test Issue with Over 100 Comments.",
             "2021-08-19T21:26:59Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/issues/1#issuecomment-902259550", "29"],
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "1",
             "2021-08-19T21:25:39Z", "meyersbs", "Test Issue with Over 100 Comments",
             "https://github.com/meyersbs/developer-apologies/issues/1", "Test Issue with Over 100 Comments.",
             "2021-08-19T21:27:02Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/issues/1#issuecomment-902259578", "30"],
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "1",
             "2021-08-19T21:25:39Z", "meyersbs", "Test Issue with Over 100 Comments",
             "https://github.com/meyersbs/developer-apologies/issues/1", "Test Issue with Over 100 Comments.",
             "2021-08-19T21:27:05Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/issues/1#issuecomment-902259598", "31"],
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "1",
             "2021-08-19T21:25:39Z", "meyersbs", "Test Issue with Over 100 Comments",
             "https://github.com/meyersbs/developer-apologies/issues/1", "Test Issue with Over 100 Comments.",
             "2021-08-19T21:27:08Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/issues/1#issuecomment-902259617", "32"],
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "1",
             "2021-08-19T21:25:39Z", "meyersbs", "Test Issue with Over 100 Comments",
             "https://github.com/meyersbs/developer-apologies/issues/1", "Test Issue with Over 100 Comments.",
             "2021-08-19T21:27:10Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/issues/1#issuecomment-902259644", "33"],
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "1",
             "2021-08-19T21:25:39Z", "meyersbs", "Test Issue with Over 100 Comments",
             "https://github.com/meyersbs/developer-apologies/issues/1", "Test Issue with Over 100 Comments.",
             "2021-08-19T21:27:15Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/issues/1#issuecomment-902259685", "34"],
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "1",
             "2021-08-19T21:25:39Z", "meyersbs", "Test Issue with Over 100 Comments",
             "https://github.com/meyersbs/developer-apologies/issues/1", "Test Issue with Over 100 Comments.",
             "2021-08-19T21:27:17Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/issues/1#issuecomment-902259716", "35"],
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "1",
             "2021-08-19T21:25:39Z", "meyersbs", "Test Issue with Over 100 Comments",
             "https://github.com/meyersbs/developer-apologies/issues/1", "Test Issue with Over 100 Comments.",
             "2021-08-19T21:27:20Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/issues/1#issuecomment-902259733", "36"],
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "1",
             "2021-08-19T21:25:39Z", "meyersbs", "Test Issue with Over 100 Comments",
             "https://github.com/meyersbs/developer-apologies/issues/1", "Test Issue with Over 100 Comments.",
             "2021-08-19T21:27:23Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/issues/1#issuecomment-902259771", "37"],
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "1",
             "2021-08-19T21:25:39Z", "meyersbs", "Test Issue with Over 100 Comments",
             "https://github.com/meyersbs/developer-apologies/issues/1", "Test Issue with Over 100 Comments.",
             "2021-08-19T21:27:26Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/issues/1#issuecomment-902259798", "38"],
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "1",
             "2021-08-19T21:25:39Z", "meyersbs", "Test Issue with Over 100 Comments",
             "https://github.com/meyersbs/developer-apologies/issues/1", "Test Issue with Over 100 Comments.",
             "2021-08-19T21:27:39Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/issues/1#issuecomment-902259921", "39"],
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "1",
             "2021-08-19T21:25:39Z", "meyersbs", "Test Issue with Over 100 Comments",
             "https://github.com/meyersbs/developer-apologies/issues/1", "Test Issue with Over 100 Comments.",
             "2021-08-19T21:27:52Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/issues/1#issuecomment-902260036", "40"],
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "1",
             "2021-08-19T21:25:39Z", "meyersbs", "Test Issue with Over 100 Comments",
             "https://github.com/meyersbs/developer-apologies/issues/1", "Test Issue with Over 100 Comments.",
             "2021-08-19T21:27:59Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/issues/1#issuecomment-902260103", "41"],
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "1",
             "2021-08-19T21:25:39Z", "meyersbs", "Test Issue with Over 100 Comments",
             "https://github.com/meyersbs/developer-apologies/issues/1", "Test Issue with Over 100 Comments.",
             "2021-08-19T21:28:05Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/issues/1#issuecomment-902260148", "42"],
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "1",
             "2021-08-19T21:25:39Z", "meyersbs", "Test Issue with Over 100 Comments",
             "https://github.com/meyersbs/developer-apologies/issues/1", "Test Issue with Over 100 Comments.",
             "2021-08-19T21:28:10Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/issues/1#issuecomment-902260197", "43"],
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "1",
             "2021-08-19T21:25:39Z", "meyersbs", "Test Issue with Over 100 Comments",
             "https://github.com/meyersbs/developer-apologies/issues/1", "Test Issue with Over 100 Comments.",
             "2021-08-19T21:28:15Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/issues/1#issuecomment-902260257", "44"],
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "1",
             "2021-08-19T21:25:39Z", "meyersbs", "Test Issue with Over 100 Comments",
             "https://github.com/meyersbs/developer-apologies/issues/1", "Test Issue with Over 100 Comments.",
             "2021-08-19T21:28:18Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/issues/1#issuecomment-902260286", "45"],
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "1",
             "2021-08-19T21:25:39Z", "meyersbs", "Test Issue with Over 100 Comments",
             "https://github.com/meyersbs/developer-apologies/issues/1", "Test Issue with Over 100 Comments.",
             "2021-08-19T21:28:22Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/issues/1#issuecomment-902260313", "46"],
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "1",
             "2021-08-19T21:25:39Z", "meyersbs", "Test Issue with Over 100 Comments",
             "https://github.com/meyersbs/developer-apologies/issues/1", "Test Issue with Over 100 Comments.",
             "2021-08-19T21:28:25Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/issues/1#issuecomment-902260335", "47"],
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "1",
             "2021-08-19T21:25:39Z", "meyersbs", "Test Issue with Over 100 Comments",
             "https://github.com/meyersbs/developer-apologies/issues/1", "Test Issue with Over 100 Comments.",
             "2021-08-19T21:28:28Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/issues/1#issuecomment-902260366", "48"],
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "1",
             "2021-08-19T21:25:39Z", "meyersbs", "Test Issue with Over 100 Comments",
             "https://github.com/meyersbs/developer-apologies/issues/1", "Test Issue with Over 100 Comments.",
             "2021-08-19T21:28:32Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/issues/1#issuecomment-902260395", "49"],
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "1",
             "2021-08-19T21:25:39Z", "meyersbs", "Test Issue with Over 100 Comments",
             "https://github.com/meyersbs/developer-apologies/issues/1", "Test Issue with Over 100 Comments.",
             "2021-08-19T21:28:35Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/issues/1#issuecomment-902260436", "50"],
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "1",
             "2021-08-19T21:25:39Z", "meyersbs", "Test Issue with Over 100 Comments",
             "https://github.com/meyersbs/developer-apologies/issues/1", "Test Issue with Over 100 Comments.",
             "2021-08-19T21:28:38Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/issues/1#issuecomment-902260470", "51"],
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "1",
             "2021-08-19T21:25:39Z", "meyersbs", "Test Issue with Over 100 Comments",
             "https://github.com/meyersbs/developer-apologies/issues/1", "Test Issue with Over 100 Comments.",
             "2021-08-19T21:28:42Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/issues/1#issuecomment-902260497", "52"],
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "1",
             "2021-08-19T21:25:39Z", "meyersbs", "Test Issue with Over 100 Comments",
             "https://github.com/meyersbs/developer-apologies/issues/1", "Test Issue with Over 100 Comments.",
             "2021-08-19T21:28:45Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/issues/1#issuecomment-902260518", "53"],
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "1",
             "2021-08-19T21:25:39Z", "meyersbs", "Test Issue with Over 100 Comments",
             "https://github.com/meyersbs/developer-apologies/issues/1", "Test Issue with Over 100 Comments.",
             "2021-08-19T21:28:49Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/issues/1#issuecomment-902260557", "54"],
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "1",
             "2021-08-19T21:25:39Z", "meyersbs", "Test Issue with Over 100 Comments",
             "https://github.com/meyersbs/developer-apologies/issues/1", "Test Issue with Over 100 Comments.",
             "2021-08-19T21:28:52Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/issues/1#issuecomment-902260586", "55"],
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "1",
             "2021-08-19T21:25:39Z", "meyersbs", "Test Issue with Over 100 Comments",
             "https://github.com/meyersbs/developer-apologies/issues/1", "Test Issue with Over 100 Comments.",
             "2021-08-19T21:28:55Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/issues/1#issuecomment-902260621", "56"],
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "1",
             "2021-08-19T21:25:39Z", "meyersbs", "Test Issue with Over 100 Comments",
             "https://github.com/meyersbs/developer-apologies/issues/1", "Test Issue with Over 100 Comments.",
             "2021-08-19T21:28:58Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/issues/1#issuecomment-902260655", "57"],
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "1",
             "2021-08-19T21:25:39Z", "meyersbs", "Test Issue with Over 100 Comments",
             "https://github.com/meyersbs/developer-apologies/issues/1", "Test Issue with Over 100 Comments.",
             "2021-08-19T21:29:01Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/issues/1#issuecomment-902260683", "58"],
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "1",
             "2021-08-19T21:25:39Z", "meyersbs", "Test Issue with Over 100 Comments",
             "https://github.com/meyersbs/developer-apologies/issues/1", "Test Issue with Over 100 Comments.",
             "2021-08-19T21:29:05Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/issues/1#issuecomment-902260717", "59"],
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "1",
             "2021-08-19T21:25:39Z", "meyersbs", "Test Issue with Over 100 Comments",
             "https://github.com/meyersbs/developer-apologies/issues/1", "Test Issue with Over 100 Comments.",
             "2021-08-19T21:29:08Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/issues/1#issuecomment-902260749", "60"],
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "1",
             "2021-08-19T21:25:39Z", "meyersbs", "Test Issue with Over 100 Comments",
             "https://github.com/meyersbs/developer-apologies/issues/1", "Test Issue with Over 100 Comments.",
             "2021-08-19T21:29:11Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/issues/1#issuecomment-902260784", "61"],
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "1",
             "2021-08-19T21:25:39Z", "meyersbs", "Test Issue with Over 100 Comments",
             "https://github.com/meyersbs/developer-apologies/issues/1", "Test Issue with Over 100 Comments.",
             "2021-08-19T21:29:15Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/issues/1#issuecomment-902260813", "62"],
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "1",
             "2021-08-19T21:25:39Z", "meyersbs", "Test Issue with Over 100 Comments",
             "https://github.com/meyersbs/developer-apologies/issues/1", "Test Issue with Over 100 Comments.",
             "2021-08-19T21:29:20Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/issues/1#issuecomment-902260860", "63"],
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "1",
             "2021-08-19T21:25:39Z", "meyersbs", "Test Issue with Over 100 Comments",
             "https://github.com/meyersbs/developer-apologies/issues/1", "Test Issue with Over 100 Comments.",
             "2021-08-19T21:29:23Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/issues/1#issuecomment-902260890", "64"],
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "1",
             "2021-08-19T21:25:39Z", "meyersbs", "Test Issue with Over 100 Comments",
             "https://github.com/meyersbs/developer-apologies/issues/1", "Test Issue with Over 100 Comments.",
             "2021-08-19T21:29:26Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/issues/1#issuecomment-902260917", "65"],
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "1",
             "2021-08-19T21:25:39Z", "meyersbs", "Test Issue with Over 100 Comments",
             "https://github.com/meyersbs/developer-apologies/issues/1", "Test Issue with Over 100 Comments.",
             "2021-08-19T21:29:29Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/issues/1#issuecomment-902260943", "66"],
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "1",
             "2021-08-19T21:25:39Z", "meyersbs", "Test Issue with Over 100 Comments",
             "https://github.com/meyersbs/developer-apologies/issues/1", "Test Issue with Over 100 Comments.",
             "2021-08-19T21:29:32Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/issues/1#issuecomment-902260974", "67"],
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "1",
             "2021-08-19T21:25:39Z", "meyersbs", "Test Issue with Over 100 Comments",
             "https://github.com/meyersbs/developer-apologies/issues/1", "Test Issue with Over 100 Comments.",
             "2021-08-19T21:29:35Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/issues/1#issuecomment-902261010", "68"],
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "1",
             "2021-08-19T21:25:39Z", "meyersbs", "Test Issue with Over 100 Comments",
             "https://github.com/meyersbs/developer-apologies/issues/1", "Test Issue with Over 100 Comments.",
             "2021-08-19T21:29:38Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/issues/1#issuecomment-902261045", "69"],
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "1",
             "2021-08-19T21:25:39Z", "meyersbs", "Test Issue with Over 100 Comments",
             "https://github.com/meyersbs/developer-apologies/issues/1", "Test Issue with Over 100 Comments.",
             "2021-08-19T21:29:42Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/issues/1#issuecomment-902261080", "70"],
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "1",
             "2021-08-19T21:25:39Z", "meyersbs", "Test Issue with Over 100 Comments",
             "https://github.com/meyersbs/developer-apologies/issues/1", "Test Issue with Over 100 Comments.",
             "2021-08-19T21:29:45Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/issues/1#issuecomment-902261114", "71"],
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "1",
             "2021-08-19T21:25:39Z", "meyersbs", "Test Issue with Over 100 Comments",
             "https://github.com/meyersbs/developer-apologies/issues/1", "Test Issue with Over 100 Comments.",
             "2021-08-19T21:29:49Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/issues/1#issuecomment-902261143", "72"],
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "1",
             "2021-08-19T21:25:39Z", "meyersbs", "Test Issue with Over 100 Comments",
             "https://github.com/meyersbs/developer-apologies/issues/1", "Test Issue with Over 100 Comments.",
             "2021-08-19T21:29:53Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/issues/1#issuecomment-902261177", "73"],
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "1",
             "2021-08-19T21:25:39Z", "meyersbs", "Test Issue with Over 100 Comments",
             "https://github.com/meyersbs/developer-apologies/issues/1", "Test Issue with Over 100 Comments.",
             "2021-08-19T21:29:56Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/issues/1#issuecomment-902261203", "74"],
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "1",
             "2021-08-19T21:25:39Z", "meyersbs", "Test Issue with Over 100 Comments",
             "https://github.com/meyersbs/developer-apologies/issues/1", "Test Issue with Over 100 Comments.",
             "2021-08-19T21:29:59Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/issues/1#issuecomment-902261232", "75"],
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "1",
             "2021-08-19T21:25:39Z", "meyersbs", "Test Issue with Over 100 Comments",
             "https://github.com/meyersbs/developer-apologies/issues/1", "Test Issue with Over 100 Comments.",
             "2021-08-19T21:30:03Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/issues/1#issuecomment-902261256", "76"],
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "1",
             "2021-08-19T21:25:39Z", "meyersbs", "Test Issue with Over 100 Comments",
             "https://github.com/meyersbs/developer-apologies/issues/1", "Test Issue with Over 100 Comments.",
             "2021-08-19T21:30:05Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/issues/1#issuecomment-902261277", "77"],
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "1",
             "2021-08-19T21:25:39Z", "meyersbs", "Test Issue with Over 100 Comments",
             "https://github.com/meyersbs/developer-apologies/issues/1", "Test Issue with Over 100 Comments.",
             "2021-08-19T21:30:09Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/issues/1#issuecomment-902261311", "78"],
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "1",
             "2021-08-19T21:25:39Z", "meyersbs", "Test Issue with Over 100 Comments",
             "https://github.com/meyersbs/developer-apologies/issues/1", "Test Issue with Over 100 Comments.",
             "2021-08-19T21:30:12Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/issues/1#issuecomment-902261347", "79"],
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "1",
             "2021-08-19T21:25:39Z", "meyersbs", "Test Issue with Over 100 Comments",
             "https://github.com/meyersbs/developer-apologies/issues/1", "Test Issue with Over 100 Comments.",
             "2021-08-19T21:30:16Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/issues/1#issuecomment-902261359", "80"],
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "1",
             "2021-08-19T21:25:39Z", "meyersbs", "Test Issue with Over 100 Comments",
             "https://github.com/meyersbs/developer-apologies/issues/1", "Test Issue with Over 100 Comments.",
             "2021-08-19T21:30:18Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/issues/1#issuecomment-902261384", "81"],
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "1",
             "2021-08-19T21:25:39Z", "meyersbs", "Test Issue with Over 100 Comments",
             "https://github.com/meyersbs/developer-apologies/issues/1", "Test Issue with Over 100 Comments.",
             "2021-08-19T21:30:22Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/issues/1#issuecomment-902261422", "82"],
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "1",
             "2021-08-19T21:25:39Z", "meyersbs", "Test Issue with Over 100 Comments",
             "https://github.com/meyersbs/developer-apologies/issues/1", "Test Issue with Over 100 Comments.",
             "2021-08-19T21:30:26Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/issues/1#issuecomment-902261458", "83"],
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "1",
             "2021-08-19T21:25:39Z", "meyersbs", "Test Issue with Over 100 Comments",
             "https://github.com/meyersbs/developer-apologies/issues/1", "Test Issue with Over 100 Comments.",
             "2021-08-19T21:30:30Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/issues/1#issuecomment-902261493", "84"],
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "1",
             "2021-08-19T21:25:39Z", "meyersbs", "Test Issue with Over 100 Comments",
             "https://github.com/meyersbs/developer-apologies/issues/1", "Test Issue with Over 100 Comments.",
             "2021-08-19T21:30:33Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/issues/1#issuecomment-902261523", "85"],
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "1",
             "2021-08-19T21:25:39Z", "meyersbs", "Test Issue with Over 100 Comments",
             "https://github.com/meyersbs/developer-apologies/issues/1", "Test Issue with Over 100 Comments.",
             "2021-08-19T21:30:37Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/issues/1#issuecomment-902261556", "86"],
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "1",
             "2021-08-19T21:25:39Z", "meyersbs", "Test Issue with Over 100 Comments",
             "https://github.com/meyersbs/developer-apologies/issues/1", "Test Issue with Over 100 Comments.",
             "2021-08-19T21:30:41Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/issues/1#issuecomment-902261587", "87"],
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "1",
             "2021-08-19T21:25:39Z", "meyersbs", "Test Issue with Over 100 Comments",
             "https://github.com/meyersbs/developer-apologies/issues/1", "Test Issue with Over 100 Comments.",
             "2021-08-19T21:30:43Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/issues/1#issuecomment-902261605", "88"],
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "1",
             "2021-08-19T21:25:39Z", "meyersbs", "Test Issue with Over 100 Comments",
             "https://github.com/meyersbs/developer-apologies/issues/1", "Test Issue with Over 100 Comments.",
             "2021-08-19T21:30:46Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/issues/1#issuecomment-902261636", "89"],
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "1",
             "2021-08-19T21:25:39Z", "meyersbs", "Test Issue with Over 100 Comments",
             "https://github.com/meyersbs/developer-apologies/issues/1", "Test Issue with Over 100 Comments.",
             "2021-08-19T21:30:49Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/issues/1#issuecomment-902261661", "90"],
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "1",
             "2021-08-19T21:25:39Z", "meyersbs", "Test Issue with Over 100 Comments",
             "https://github.com/meyersbs/developer-apologies/issues/1", "Test Issue with Over 100 Comments.",
             "2021-08-19T21:30:53Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/issues/1#issuecomment-902261685", "91"],
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "1",
             "2021-08-19T21:25:39Z", "meyersbs", "Test Issue with Over 100 Comments",
             "https://github.com/meyersbs/developer-apologies/issues/1", "Test Issue with Over 100 Comments.",
             "2021-08-19T21:30:56Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/issues/1#issuecomment-902261709", "92"],
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "1",
             "2021-08-19T21:25:39Z", "meyersbs", "Test Issue with Over 100 Comments",
             "https://github.com/meyersbs/developer-apologies/issues/1", "Test Issue with Over 100 Comments.",
             "2021-08-19T21:30:59Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/issues/1#issuecomment-902261744", "93"],
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "1",
             "2021-08-19T21:25:39Z", "meyersbs", "Test Issue with Over 100 Comments",
             "https://github.com/meyersbs/developer-apologies/issues/1", "Test Issue with Over 100 Comments.",
             "2021-08-19T21:31:03Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/issues/1#issuecomment-902261789", "94"],
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "1",
             "2021-08-19T21:25:39Z", "meyersbs", "Test Issue with Over 100 Comments",
             "https://github.com/meyersbs/developer-apologies/issues/1", "Test Issue with Over 100 Comments.",
             "2021-08-19T21:31:05Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/issues/1#issuecomment-902261819", "95"],
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "1",
             "2021-08-19T21:25:39Z", "meyersbs", "Test Issue with Over 100 Comments",
             "https://github.com/meyersbs/developer-apologies/issues/1", "Test Issue with Over 100 Comments.",
             "2021-08-19T21:31:08Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/issues/1#issuecomment-902261838", "96"],
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "1",
             "2021-08-19T21:25:39Z", "meyersbs", "Test Issue with Over 100 Comments",
             "https://github.com/meyersbs/developer-apologies/issues/1", "Test Issue with Over 100 Comments.",
             "2021-08-19T21:31:11Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/issues/1#issuecomment-902261864", "97"],
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "1",
             "2021-08-19T21:25:39Z", "meyersbs", "Test Issue with Over 100 Comments",
             "https://github.com/meyersbs/developer-apologies/issues/1", "Test Issue with Over 100 Comments.",
             "2021-08-19T21:31:14Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/issues/1#issuecomment-902261888", "98"],
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "1",
             "2021-08-19T21:25:39Z", "meyersbs", "Test Issue with Over 100 Comments",
             "https://github.com/meyersbs/developer-apologies/issues/1", "Test Issue with Over 100 Comments.",
             "2021-08-19T21:31:16Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/issues/1#issuecomment-902261906", "99"],
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "1",
             "2021-08-19T21:25:39Z", "meyersbs", "Test Issue with Over 100 Comments",
             "https://github.com/meyersbs/developer-apologies/issues/1", "Test Issue with Over 100 Comments.",
             "2021-08-19T21:31:20Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/issues/1#issuecomment-902261939", "100"],
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "1",
             "2021-08-19T21:25:39Z", "meyersbs", "Test Issue with Over 100 Comments",
             "https://github.com/meyersbs/developer-apologies/issues/1", "Test Issue with Over 100 Comments.",
             "2021-08-19T21:31:23Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/issues/1#issuecomment-902261971", "101"],
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "2",
             "2021-08-19T21:46:16Z", "meyersbs", "Test Issue",
             "https://github.com/meyersbs/developer-apologies/issues/2", "Test Issue", "2021-08-19T21:46:23Z",
             "meyersbs", "https://github.com/meyersbs/developer-apologies/issues/2#issuecomment-902269928",
             "Dummy comment."],
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "2",
             "2021-08-19T21:46:16Z", "meyersbs", "Test Issue",
             "https://github.com/meyersbs/developer-apologies/issues/2", "Test Issue", "2021-08-20T13:23:06Z",
             "andymeneely", "https://github.com/meyersbs/developer-apologies/issues/2#issuecomment-902690150",
             "Sorry, but I figured I'd add a data point. Sorrynotsorry."],
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "5",
             "2021-09-23T16:13:11Z", "meyersbs", "Bug in src/download:_writeCSV()",
             "https://github.com/meyersbs/developer-apologies/issues/5",
             "When downloading data from multiple repositories, the function src/download:_writeCSV()"
             " writes the columns headers once for every repo instead of just once for the entire CSV file.",
             "", "", "", ""]
        ]
        # Test
        download(input_repo_file, input_data_dir, input_data_types, overwrite=True)
        actual = list()
        with open(os.path.join(input_data_dir, "issues/issues.csv"), "r") as f:
            csv_reader = csv.reader(f, delimiter=",", quoting=csv.QUOTE_MINIMAL, quotechar="\"")
            for entry in csv_reader:
                actual.append(entry)
        self.assertListEqual(expected_issues_3, actual)
        # Cleanup
        shutil.rmtree(input_data_dir)

        #### Case 3 -- data_types="pull_requests"
        # Setup
        input_repo_file = os.path.join(CWD, "test_files/repo_lists/test_repos_3.txt")
        input_data_dir = os.path.join(CWD, "test_data/")
        os.mkdir(input_data_dir)
        input_data_types = "pull_requests"
        expected_pull_requests_3 = [
            ["REPO_URL", "REPO_NAME", "REPO_OWNER", "PULL_REQUEST_NUMBER", "PULL_REQUEST_TITLE",
             "PULL_REQUEST_AUTHOR", "PULL_REQUEST_CREATION_DATE", "PULL_REQUEST_URL", "PULL_REQUEST_TEXT",
             "COMMENT_CREATION_DATE", "COMMENT_AUTHOR", "COMMENT_URL", "COMMENT_TEXT"],
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "3",
             "2021-08-23T14:33:22Z", "andymeneely", "Update README.md",
             "https://github.com/meyersbs/developer-apologies/pull/3", "", "2021-08-23T14:34:12Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/pull/3#issuecomment-903830546", "1"],
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "3",
             "2021-08-23T14:33:22Z", "andymeneely", "Update README.md",
             "https://github.com/meyersbs/developer-apologies/pull/3", "", "2021-08-23T14:34:15Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/pull/3#issuecomment-903830587", "2"], 
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "3",
             "2021-08-23T14:33:22Z", "andymeneely", "Update README.md",
             "https://github.com/meyersbs/developer-apologies/pull/3", "", "2021-08-23T14:34:18Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/pull/3#issuecomment-903830629", "3"], 
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "3",
             "2021-08-23T14:33:22Z", "andymeneely", "Update README.md",
             "https://github.com/meyersbs/developer-apologies/pull/3", "", "2021-08-23T14:34:20Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/pull/3#issuecomment-903830663", "4"], 
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "3",
             "2021-08-23T14:33:22Z", "andymeneely", "Update README.md",
             "https://github.com/meyersbs/developer-apologies/pull/3", "", "2021-08-23T14:34:23Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/pull/3#issuecomment-903830695", "5"], 
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "3",
             "2021-08-23T14:33:22Z", "andymeneely", "Update README.md",
             "https://github.com/meyersbs/developer-apologies/pull/3", "", "2021-08-23T14:34:26Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/pull/3#issuecomment-903830726", "6"], 
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "3",
             "2021-08-23T14:33:22Z", "andymeneely", "Update README.md",
             "https://github.com/meyersbs/developer-apologies/pull/3", "", "2021-08-23T14:34:30Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/pull/3#issuecomment-903830765", "7"], 
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "3",
             "2021-08-23T14:33:22Z", "andymeneely", "Update README.md",
             "https://github.com/meyersbs/developer-apologies/pull/3", "", "2021-08-23T14:34:32Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/pull/3#issuecomment-903830803", "8"], 
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "3",
             "2021-08-23T14:33:22Z", "andymeneely", "Update README.md",
             "https://github.com/meyersbs/developer-apologies/pull/3", "", "2021-08-23T14:34:35Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/pull/3#issuecomment-903830834", "9"], 
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "3",
             "2021-08-23T14:33:22Z", "andymeneely", "Update README.md",
             "https://github.com/meyersbs/developer-apologies/pull/3", "", "2021-08-23T14:34:39Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/pull/3#issuecomment-903830889", "10"], 
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "3",
             "2021-08-23T14:33:22Z", "andymeneely", "Update README.md",
             "https://github.com/meyersbs/developer-apologies/pull/3", "", "2021-08-23T14:34:42Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/pull/3#issuecomment-903830931", "11"], 
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "3",
             "2021-08-23T14:33:22Z", "andymeneely", "Update README.md",
             "https://github.com/meyersbs/developer-apologies/pull/3", "", "2021-08-23T14:34:45Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/pull/3#issuecomment-903830972", "12"], 
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "3",
             "2021-08-23T14:33:22Z", "andymeneely", "Update README.md",
             "https://github.com/meyersbs/developer-apologies/pull/3", "", "2021-08-23T14:34:49Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/pull/3#issuecomment-903831020", "13"], 
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "3",
             "2021-08-23T14:33:22Z", "andymeneely", "Update README.md",
             "https://github.com/meyersbs/developer-apologies/pull/3", "", "2021-08-23T14:34:52Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/pull/3#issuecomment-903831066", "14"], 
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "3",
             "2021-08-23T14:33:22Z", "andymeneely", "Update README.md",
             "https://github.com/meyersbs/developer-apologies/pull/3", "", "2021-08-23T14:34:55Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/pull/3#issuecomment-903831114", "15"], 
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "3",
             "2021-08-23T14:33:22Z", "andymeneely", "Update README.md",
             "https://github.com/meyersbs/developer-apologies/pull/3", "", "2021-08-23T14:35:00Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/pull/3#issuecomment-903831187", "16"], 
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "3",
             "2021-08-23T14:33:22Z", "andymeneely", "Update README.md",
             "https://github.com/meyersbs/developer-apologies/pull/3", "", "2021-08-23T14:35:03Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/pull/3#issuecomment-903831223", "17"], 
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "3",
             "2021-08-23T14:33:22Z", "andymeneely", "Update README.md",
             "https://github.com/meyersbs/developer-apologies/pull/3", "", "2021-08-23T14:35:07Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/pull/3#issuecomment-903831283", "18"], 
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "3",
             "2021-08-23T14:33:22Z", "andymeneely", "Update README.md",
             "https://github.com/meyersbs/developer-apologies/pull/3", "", "2021-08-23T14:35:10Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/pull/3#issuecomment-903831337", "19"], 
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "3",
             "2021-08-23T14:33:22Z", "andymeneely", "Update README.md",
             "https://github.com/meyersbs/developer-apologies/pull/3", "", "2021-08-23T14:35:13Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/pull/3#issuecomment-903831387", "20"], 
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "3",
             "2021-08-23T14:33:22Z", "andymeneely", "Update README.md",
             "https://github.com/meyersbs/developer-apologies/pull/3", "", "2021-08-23T14:35:16Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/pull/3#issuecomment-903831422", "21"], 
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "3",
             "2021-08-23T14:33:22Z", "andymeneely", "Update README.md",
             "https://github.com/meyersbs/developer-apologies/pull/3", "", "2021-08-23T14:35:19Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/pull/3#issuecomment-903831470", "22"], 
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "3",
             "2021-08-23T14:33:22Z", "andymeneely", "Update README.md",
             "https://github.com/meyersbs/developer-apologies/pull/3", "", "2021-08-23T14:35:22Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/pull/3#issuecomment-903831506", "23"], 
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "3",
             "2021-08-23T14:33:22Z", "andymeneely", "Update README.md",
             "https://github.com/meyersbs/developer-apologies/pull/3", "", "2021-08-23T14:35:25Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/pull/3#issuecomment-903831556", "24"], 
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "3",
             "2021-08-23T14:33:22Z", "andymeneely", "Update README.md",
             "https://github.com/meyersbs/developer-apologies/pull/3", "", "2021-08-23T14:35:28Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/pull/3#issuecomment-903831611", "25"], 
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "3",
             "2021-08-23T14:33:22Z", "andymeneely", "Update README.md",
             "https://github.com/meyersbs/developer-apologies/pull/3", "", "2021-08-23T14:35:31Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/pull/3#issuecomment-903831651", "26"], 
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "3",
             "2021-08-23T14:33:22Z", "andymeneely", "Update README.md",
             "https://github.com/meyersbs/developer-apologies/pull/3", "", "2021-08-23T14:35:34Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/pull/3#issuecomment-903831696", "27"], 
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "3",
             "2021-08-23T14:33:22Z", "andymeneely", "Update README.md",
             "https://github.com/meyersbs/developer-apologies/pull/3", "", "2021-08-23T14:35:37Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/pull/3#issuecomment-903831740", "28"], 
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "3",
             "2021-08-23T14:33:22Z", "andymeneely", "Update README.md",
             "https://github.com/meyersbs/developer-apologies/pull/3", "", "2021-08-23T14:35:40Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/pull/3#issuecomment-903831778", "29"], 
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "3",
             "2021-08-23T14:33:22Z", "andymeneely", "Update README.md",
             "https://github.com/meyersbs/developer-apologies/pull/3", "", "2021-08-23T14:35:44Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/pull/3#issuecomment-903831832", "30"], 
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "3",
             "2021-08-23T14:33:22Z", "andymeneely", "Update README.md",
             "https://github.com/meyersbs/developer-apologies/pull/3", "", "2021-08-23T14:35:46Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/pull/3#issuecomment-903831872", "31"], 
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "3",
             "2021-08-23T14:33:22Z", "andymeneely", "Update README.md",
             "https://github.com/meyersbs/developer-apologies/pull/3", "", "2021-08-23T14:35:49Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/pull/3#issuecomment-903831905", "32"], 
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "3",
             "2021-08-23T14:33:22Z", "andymeneely", "Update README.md",
             "https://github.com/meyersbs/developer-apologies/pull/3", "", "2021-08-23T14:35:51Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/pull/3#issuecomment-903831944", "33"], 
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "3",
             "2021-08-23T14:33:22Z", "andymeneely", "Update README.md",
             "https://github.com/meyersbs/developer-apologies/pull/3", "", "2021-08-23T14:35:54Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/pull/3#issuecomment-903831979", "34"], 
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "3",
             "2021-08-23T14:33:22Z", "andymeneely", "Update README.md",
             "https://github.com/meyersbs/developer-apologies/pull/3", "", "2021-08-23T14:35:58Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/pull/3#issuecomment-903832027", "35"], 
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "3",
             "2021-08-23T14:33:22Z", "andymeneely", "Update README.md",
             "https://github.com/meyersbs/developer-apologies/pull/3", "", "2021-08-23T14:36:01Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/pull/3#issuecomment-903832083", "36"], 
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "3",
             "2021-08-23T14:33:22Z", "andymeneely", "Update README.md",
             "https://github.com/meyersbs/developer-apologies/pull/3", "", "2021-08-23T14:36:04Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/pull/3#issuecomment-903832124", "37"], 
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "3",
             "2021-08-23T14:33:22Z", "andymeneely", "Update README.md",
             "https://github.com/meyersbs/developer-apologies/pull/3", "", "2021-08-23T14:36:07Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/pull/3#issuecomment-903832177", "38"], 
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "3",
             "2021-08-23T14:33:22Z", "andymeneely", "Update README.md",
             "https://github.com/meyersbs/developer-apologies/pull/3", "", "2021-08-23T14:36:10Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/pull/3#issuecomment-903832226", "39"], 
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "3",
             "2021-08-23T14:33:22Z", "andymeneely", "Update README.md",
             "https://github.com/meyersbs/developer-apologies/pull/3", "", "2021-08-23T14:36:13Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/pull/3#issuecomment-903832268", "40"], 
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "3",
             "2021-08-23T14:33:22Z", "andymeneely", "Update README.md",
             "https://github.com/meyersbs/developer-apologies/pull/3", "", "2021-08-23T14:36:15Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/pull/3#issuecomment-903832297", "41"], 
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "3",
             "2021-08-23T14:33:22Z", "andymeneely", "Update README.md",
             "https://github.com/meyersbs/developer-apologies/pull/3", "", "2021-08-23T14:36:18Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/pull/3#issuecomment-903832327", "42"], 
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "3",
             "2021-08-23T14:33:22Z", "andymeneely", "Update README.md",
             "https://github.com/meyersbs/developer-apologies/pull/3", "", "2021-08-23T14:36:20Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/pull/3#issuecomment-903832359", "43"], 
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "3",
             "2021-08-23T14:33:22Z", "andymeneely", "Update README.md",
             "https://github.com/meyersbs/developer-apologies/pull/3", "", "2021-08-23T14:36:23Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/pull/3#issuecomment-903832395", "44"], 
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "3",
             "2021-08-23T14:33:22Z", "andymeneely", "Update README.md",
             "https://github.com/meyersbs/developer-apologies/pull/3", "", "2021-08-23T14:36:26Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/pull/3#issuecomment-903832425", "45"], 
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "3",
             "2021-08-23T14:33:22Z", "andymeneely", "Update README.md",
             "https://github.com/meyersbs/developer-apologies/pull/3", "", "2021-08-23T14:36:29Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/pull/3#issuecomment-903832472", "46"], 
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "3",
             "2021-08-23T14:33:22Z", "andymeneely", "Update README.md",
             "https://github.com/meyersbs/developer-apologies/pull/3", "", "2021-08-23T14:36:31Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/pull/3#issuecomment-903832501", "47"], 
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "3",
             "2021-08-23T14:33:22Z", "andymeneely", "Update README.md",
             "https://github.com/meyersbs/developer-apologies/pull/3", "", "2021-08-23T14:36:34Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/pull/3#issuecomment-903832551", "48"], 
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "3",
             "2021-08-23T14:33:22Z", "andymeneely", "Update README.md",
             "https://github.com/meyersbs/developer-apologies/pull/3", "", "2021-08-23T14:36:37Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/pull/3#issuecomment-903832587", "49"], 
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "3",
             "2021-08-23T14:33:22Z", "andymeneely", "Update README.md",
             "https://github.com/meyersbs/developer-apologies/pull/3", "", "2021-08-23T14:36:40Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/pull/3#issuecomment-903832637", "50"], 
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "3",
             "2021-08-23T14:33:22Z", "andymeneely", "Update README.md",
             "https://github.com/meyersbs/developer-apologies/pull/3", "", "2021-08-23T14:36:42Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/pull/3#issuecomment-903832677", "51"], 
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "3",
             "2021-08-23T14:33:22Z", "andymeneely", "Update README.md",
             "https://github.com/meyersbs/developer-apologies/pull/3", "", "2021-08-23T14:36:45Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/pull/3#issuecomment-903832716", "52"], 
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "3",
             "2021-08-23T14:33:22Z", "andymeneely", "Update README.md",
             "https://github.com/meyersbs/developer-apologies/pull/3", "", "2021-08-23T14:36:48Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/pull/3#issuecomment-903832765", "53"], 
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "3",
             "2021-08-23T14:33:22Z", "andymeneely", "Update README.md",
             "https://github.com/meyersbs/developer-apologies/pull/3", "", "2021-08-23T14:36:51Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/pull/3#issuecomment-903832802", "54"], 
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "3",
             "2021-08-23T14:33:22Z", "andymeneely", "Update README.md",
             "https://github.com/meyersbs/developer-apologies/pull/3", "", "2021-08-23T14:36:53Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/pull/3#issuecomment-903832833", "55"], 
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "3",
             "2021-08-23T14:33:22Z", "andymeneely", "Update README.md",
             "https://github.com/meyersbs/developer-apologies/pull/3", "", "2021-08-23T14:36:57Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/pull/3#issuecomment-903832879", "56"], 
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "3",
             "2021-08-23T14:33:22Z", "andymeneely", "Update README.md",
             "https://github.com/meyersbs/developer-apologies/pull/3", "", "2021-08-23T14:36:59Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/pull/3#issuecomment-903832920", "57"], 
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "3",
             "2021-08-23T14:33:22Z", "andymeneely", "Update README.md",
             "https://github.com/meyersbs/developer-apologies/pull/3", "", "2021-08-23T14:37:02Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/pull/3#issuecomment-903832961", "58"], 
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "3",
             "2021-08-23T14:33:22Z", "andymeneely", "Update README.md",
             "https://github.com/meyersbs/developer-apologies/pull/3", "", "2021-08-23T14:37:05Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/pull/3#issuecomment-903833012", "59"], 
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "3",
             "2021-08-23T14:33:22Z", "andymeneely", "Update README.md",
             "https://github.com/meyersbs/developer-apologies/pull/3", "", "2021-08-23T14:37:08Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/pull/3#issuecomment-903833067", "60"], 
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "3",
             "2021-08-23T14:33:22Z", "andymeneely", "Update README.md",
             "https://github.com/meyersbs/developer-apologies/pull/3", "", "2021-08-23T14:37:11Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/pull/3#issuecomment-903833098", "61"], 
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "3",
             "2021-08-23T14:33:22Z", "andymeneely", "Update README.md",
             "https://github.com/meyersbs/developer-apologies/pull/3", "", "2021-08-23T14:37:14Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/pull/3#issuecomment-903833135", "62"], 
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "3",
             "2021-08-23T14:33:22Z", "andymeneely", "Update README.md",
             "https://github.com/meyersbs/developer-apologies/pull/3", "", "2021-08-23T14:37:16Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/pull/3#issuecomment-903833177", "63"], 
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "3",
             "2021-08-23T14:33:22Z", "andymeneely", "Update README.md",
             "https://github.com/meyersbs/developer-apologies/pull/3", "", "2021-08-23T14:37:19Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/pull/3#issuecomment-903833206", "64"], 
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "3",
             "2021-08-23T14:33:22Z", "andymeneely", "Update README.md",
             "https://github.com/meyersbs/developer-apologies/pull/3", "", "2021-08-23T14:37:21Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/pull/3#issuecomment-903833236", "65"], 
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "3",
             "2021-08-23T14:33:22Z", "andymeneely", "Update README.md",
             "https://github.com/meyersbs/developer-apologies/pull/3", "", "2021-08-23T14:37:24Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/pull/3#issuecomment-903833267", "66"], 
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "3",
             "2021-08-23T14:33:22Z", "andymeneely", "Update README.md",
             "https://github.com/meyersbs/developer-apologies/pull/3", "", "2021-08-23T14:37:26Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/pull/3#issuecomment-903833307", "67"], 
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "3",
             "2021-08-23T14:33:22Z", "andymeneely", "Update README.md",
             "https://github.com/meyersbs/developer-apologies/pull/3", "", "2021-08-23T14:37:29Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/pull/3#issuecomment-903833329", "68"], 
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "3",
             "2021-08-23T14:33:22Z", "andymeneely", "Update README.md",
             "https://github.com/meyersbs/developer-apologies/pull/3", "", "2021-08-23T14:37:31Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/pull/3#issuecomment-903833365", "69"], 
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "3",
             "2021-08-23T14:33:22Z", "andymeneely", "Update README.md",
             "https://github.com/meyersbs/developer-apologies/pull/3", "", "2021-08-23T14:37:34Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/pull/3#issuecomment-903833406", "70"], 
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "3",
             "2021-08-23T14:33:22Z", "andymeneely", "Update README.md",
             "https://github.com/meyersbs/developer-apologies/pull/3", "", "2021-08-23T14:37:37Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/pull/3#issuecomment-903833439", "71"], 
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "3",
             "2021-08-23T14:33:22Z", "andymeneely", "Update README.md",
             "https://github.com/meyersbs/developer-apologies/pull/3", "", "2021-08-23T14:37:39Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/pull/3#issuecomment-903833464", "72"], 
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "3",
             "2021-08-23T14:33:22Z", "andymeneely", "Update README.md",
             "https://github.com/meyersbs/developer-apologies/pull/3", "", "2021-08-23T14:37:42Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/pull/3#issuecomment-903833504", "73"], 
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "3",
             "2021-08-23T14:33:22Z", "andymeneely", "Update README.md",
             "https://github.com/meyersbs/developer-apologies/pull/3", "", "2021-08-23T14:37:46Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/pull/3#issuecomment-903833561", "74"], 
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "3",
             "2021-08-23T14:33:22Z", "andymeneely", "Update README.md",
             "https://github.com/meyersbs/developer-apologies/pull/3", "", "2021-08-23T14:37:49Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/pull/3#issuecomment-903833593", "75"], 
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "3",
             "2021-08-23T14:33:22Z", "andymeneely", "Update README.md",
             "https://github.com/meyersbs/developer-apologies/pull/3", "", "2021-08-23T14:37:51Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/pull/3#issuecomment-903833618", "76"], 
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "3",
             "2021-08-23T14:33:22Z", "andymeneely", "Update README.md",
             "https://github.com/meyersbs/developer-apologies/pull/3", "", "2021-08-23T14:37:54Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/pull/3#issuecomment-903833661", "77"], 
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "3",
             "2021-08-23T14:33:22Z", "andymeneely", "Update README.md",
             "https://github.com/meyersbs/developer-apologies/pull/3", "", "2021-08-23T14:37:56Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/pull/3#issuecomment-903833689", "78"], 
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "3",
             "2021-08-23T14:33:22Z", "andymeneely", "Update README.md",
             "https://github.com/meyersbs/developer-apologies/pull/3", "", "2021-08-23T14:37:59Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/pull/3#issuecomment-903833716", "79"], 
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "3",
             "2021-08-23T14:33:22Z", "andymeneely", "Update README.md",
             "https://github.com/meyersbs/developer-apologies/pull/3", "", "2021-08-23T14:38:01Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/pull/3#issuecomment-903833742", "80"], 
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "3",
             "2021-08-23T14:33:22Z", "andymeneely", "Update README.md",
             "https://github.com/meyersbs/developer-apologies/pull/3", "", "2021-08-23T14:38:05Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/pull/3#issuecomment-903833784", "81"], 
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "3",
             "2021-08-23T14:33:22Z", "andymeneely", "Update README.md",
             "https://github.com/meyersbs/developer-apologies/pull/3", "", "2021-08-23T14:38:08Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/pull/3#issuecomment-903833827", "82"], 
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "3",
             "2021-08-23T14:33:22Z", "andymeneely", "Update README.md",
             "https://github.com/meyersbs/developer-apologies/pull/3", "", "2021-08-23T14:38:11Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/pull/3#issuecomment-903833866", "83"], 
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "3",
             "2021-08-23T14:33:22Z", "andymeneely", "Update README.md",
             "https://github.com/meyersbs/developer-apologies/pull/3", "", "2021-08-23T14:38:15Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/pull/3#issuecomment-903833934", "84"], 
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "3",
             "2021-08-23T14:33:22Z", "andymeneely", "Update README.md",
             "https://github.com/meyersbs/developer-apologies/pull/3", "", "2021-08-23T14:38:19Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/pull/3#issuecomment-903833989", "85"], 
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "3",
             "2021-08-23T14:33:22Z", "andymeneely", "Update README.md",
             "https://github.com/meyersbs/developer-apologies/pull/3", "", "2021-08-23T14:38:22Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/pull/3#issuecomment-903834037", "86"], 
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "3",
             "2021-08-23T14:33:22Z", "andymeneely", "Update README.md",
             "https://github.com/meyersbs/developer-apologies/pull/3", "", "2021-08-23T14:38:25Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/pull/3#issuecomment-903834076", "87"], 
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "3",
             "2021-08-23T14:33:22Z", "andymeneely", "Update README.md",
             "https://github.com/meyersbs/developer-apologies/pull/3", "", "2021-08-23T14:38:28Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/pull/3#issuecomment-903834126", "88"], 
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "3",
             "2021-08-23T14:33:22Z", "andymeneely", "Update README.md",
             "https://github.com/meyersbs/developer-apologies/pull/3", "", "2021-08-23T14:38:31Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/pull/3#issuecomment-903834165", "89"], 
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "3",
             "2021-08-23T14:33:22Z", "andymeneely", "Update README.md",
             "https://github.com/meyersbs/developer-apologies/pull/3", "", "2021-08-23T14:38:34Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/pull/3#issuecomment-903834203", "90"], 
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "3",
             "2021-08-23T14:33:22Z", "andymeneely", "Update README.md",
             "https://github.com/meyersbs/developer-apologies/pull/3", "", "2021-08-23T14:38:36Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/pull/3#issuecomment-903834237", "91"], 
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "3",
             "2021-08-23T14:33:22Z", "andymeneely", "Update README.md",
             "https://github.com/meyersbs/developer-apologies/pull/3", "", "2021-08-23T14:38:39Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/pull/3#issuecomment-903834275", "92"], 
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "3",
             "2021-08-23T14:33:22Z", "andymeneely", "Update README.md",
             "https://github.com/meyersbs/developer-apologies/pull/3", "", "2021-08-23T14:38:42Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/pull/3#issuecomment-903834312", "93"], 
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "3",
             "2021-08-23T14:33:22Z", "andymeneely", "Update README.md",
             "https://github.com/meyersbs/developer-apologies/pull/3", "", "2021-08-23T14:38:45Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/pull/3#issuecomment-903834354", "94"], 
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "3",
             "2021-08-23T14:33:22Z", "andymeneely", "Update README.md",
             "https://github.com/meyersbs/developer-apologies/pull/3", "", "2021-08-23T14:38:48Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/pull/3#issuecomment-903834398", "95"], 
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "3",
             "2021-08-23T14:33:22Z", "andymeneely", "Update README.md",
             "https://github.com/meyersbs/developer-apologies/pull/3", "", "2021-08-23T14:38:51Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/pull/3#issuecomment-903834443", "96"], 
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "3",
             "2021-08-23T14:33:22Z", "andymeneely", "Update README.md",
             "https://github.com/meyersbs/developer-apologies/pull/3", "", "2021-08-23T14:38:54Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/pull/3#issuecomment-903834470", "97"], 
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "3",
             "2021-08-23T14:33:22Z", "andymeneely", "Update README.md",
             "https://github.com/meyersbs/developer-apologies/pull/3", "", "2021-08-23T14:38:56Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/pull/3#issuecomment-903834500", "98"], 
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "3",
             "2021-08-23T14:33:22Z", "andymeneely", "Update README.md",
             "https://github.com/meyersbs/developer-apologies/pull/3", "", "2021-08-23T14:38:59Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/pull/3#issuecomment-903834538", "99"], 
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "3",
             "2021-08-23T14:33:22Z", "andymeneely", "Update README.md",
             "https://github.com/meyersbs/developer-apologies/pull/3", "", "2021-08-23T14:39:03Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/pull/3#issuecomment-903834590", "100"], 
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "3",
             "2021-08-23T14:33:22Z", "andymeneely", "Update README.md",
             "https://github.com/meyersbs/developer-apologies/pull/3", "", "2021-08-23T14:39:07Z", "meyersbs",
             "https://github.com/meyersbs/developer-apologies/pull/3#issuecomment-903834642", "101"],
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "4",
             "2021-08-24T17:59:32Z", "bnk5096", "Brandon-test",
             "https://github.com/meyersbs/developer-apologies/pull/4", "", "", "", "", ""],
            ["https://github.com/meyersbs/developer-apologies/", "developer-apologies", "meyersbs", "6",
             "2022-03-30T12:51:02Z", "meyersbs", "Remove hdf5",
             "https://github.com/meyersbs/developer-apologies/pull/6", "We are removing HDF5 and working "
             "with CSV files from now on. HDF5 is an interesting tool, but support for natural language "
             "(strings) is too cumbersome for our needs.", "", "", "", ""]

        ]
        # Test
        download(input_repo_file, input_data_dir, input_data_types, overwrite=True)
        actual = list()
        with open(os.path.join(input_data_dir, "pull_requests/pull_requests.csv"), "r") as f:
            csv_reader = csv.reader(f, delimiter=",", quoting=csv.QUOTE_MINIMAL, quotechar="\"")
            for entry in csv_reader:
                actual.append(entry)
        self.assertListEqual(expected_pull_requests_3, actual)
        # Cleanup
        shutil.rmtree(input_data_dir)

        #### Case 4 -- data_types="commits"
        # Setup
        input_repo_file = os.path.join(CWD, "test_files/repo_lists/test_repos_3.txt")
        input_data_dir = os.path.join(CWD, "test_data/")
        os.mkdir(input_data_dir)
        input_data_types = "commits"
        expected_header = [
            "REPO_URL", "REPO_NAME", "REPO_OWNER", "COMMIT_OID", "COMMIT_CREATION_DATE",
            "COMMIT_AUTHOR", "COMMIT_ADDITIONS", "COMMIT_DELETIONS", "COMMIT_HEADLINE",
            "COMMIT_URL", "COMMIT_TEXT", "COMMENT_CREATION_DATE", "COMMENT_AUTHOR",
            "COMMENT_URL", "COMMENT_TEXT"
        ]
        # Test
        download(input_repo_file, input_data_dir, input_data_types, overwrite=True)
        actual = list()
        with open(os.path.join(input_data_dir, "commits/commits.csv"), "r") as f:
            csv_reader = csv.reader(f, delimiter=",", quoting=csv.QUOTE_MINIMAL, quotechar="\"")
            for entry in csv_reader:
                actual.append(entry)
        self.assertListEqual(expected_header, actual[0])
        over_100_count = 0
        for entry in actual:
            if entry[8] == "Implement pull request downloading.":
                over_100_count += 1
        self.assertEqual(102, over_100_count)
        # Cleanup
        shutil.rmtree(input_data_dir)

        #### Case 5 -- data_types="all"
        # Setup
        input_repo_file = os.path.join(CWD, "test_files/repo_lists/test_repos_3.txt")
        input_data_dir = os.path.join(CWD, "test_data/")
        os.mkdir(input_data_dir)
        input_data_types = "all"
        actual_issues = list()
        actual_commits = list()
        actual_pull_requests = list()
        # Test
        download(input_repo_file, input_data_dir, input_data_types, overwrite=True)
        with open(os.path.join(input_data_dir, "issues/issues.csv"), "r") as f:
            csv_reader = csv.reader(f, delimiter=",", quoting=csv.QUOTE_MINIMAL, quotechar="\"")
            for entry in csv_reader:
                actual_issues.append(entry)
        with open(os.path.join(input_data_dir, "commits/commits.csv"), "r") as f:
            csv_reader = csv.reader(f, delimiter=",", quoting=csv.QUOTE_MINIMAL, quotechar="\"")
            for entry in csv_reader:
                actual_commits.append(entry)
        with open(os.path.join(input_data_dir, "pull_requests/pull_requests.csv"), "r") as f:
            csv_reader = csv.reader(f, delimiter=",", quoting=csv.QUOTE_MINIMAL, quotechar="\"")
            for entry in csv_reader:
                actual_pull_requests.append(entry)
        # Test issues
        self.assertListEqual(expected_issues_3, actual_issues)
        # Test commits
        expected_header = [
            "REPO_URL", "REPO_NAME", "REPO_OWNER", "COMMIT_OID", "COMMIT_CREATION_DATE",
            "COMMIT_AUTHOR", "COMMIT_ADDITIONS", "COMMIT_DELETIONS", "COMMIT_HEADLINE",
            "COMMIT_URL", "COMMIT_TEXT", "COMMENT_CREATION_DATE", "COMMENT_AUTHOR",
            "COMMENT_URL", "COMMENT_TEXT"
        ]
        self.assertListEqual(expected_header, actual_commits[0])
        over_100_count = 0
        for entry in actual_commits:
            if entry[8] == "Implement pull request downloading.":
                over_100_count += 1
        self.assertEqual(102, over_100_count)
        # Test pull requests
        self.assertListEqual(expected_pull_requests_3, actual_pull_requests)
        # Cleanup
        shutil.rmtree(input_data_dir)


class TestDeduplicate(unittest.TestCase):
    """
    Test cases for functions in src.deduplicate.
    """
    def setUp(self):
        """
        Necessary setup for test cases.
        """
        pass


    def test_deduplicate(self):
        """
        Test src.deduplicate:deduplicate().        
        """
        #### Case 1 -- overwrite=False
        # Set up
        input_data_dir = os.path.join(CWD, "test_files/test_data5/")
        input_overwrite=False
        old_issues, old_commits, old_pull_requests = getDataFilepaths(input_data_dir)
        dedup_issues = old_issues.split(".csv")[0] + "_dedup.csv"
        dedup_commits = old_commits.split(".csv")[0] + "_dedup.csv"
        dedup_pull_requests = old_pull_requests.split(".csv")[0] + "_dedup.csv"
        expected_issues_diff = False
        expected_commits_diff = False
        expected_pull_requests_diff = True
        expected_issues_header_count = 1
        expected_commits_header_count = 1
        expected_pull_requests_header_count = 1
        # Test
        deduplicate(input_data_dir, input_overwrite)
        # Test 1
        actual_issues_diff = filecmp.cmp(old_issues, dedup_issues, shallow=False)
        actual_commits_diff = filecmp.cmp(old_commits, dedup_commits, shallow=False)
        actual_pull_requests_diff = filecmp.cmp(old_pull_requests, dedup_pull_requests, shallow=False)
        self.assertEqual(expected_issues_diff, actual_issues_diff)
        self.assertEqual(expected_commits_diff, actual_commits_diff)
        self.assertEqual(expected_pull_requests_diff, actual_pull_requests_diff)
        # Test 2
        actual_issues_header_count = 0
        with open(dedup_issues, "r") as f:
            csv_reader = csv.reader(f, delimiter=",", quotechar="\"", quoting=csv.QUOTE_MINIMAL)
            for row in csv_reader:
                if row == ISSUES_HEADER:
                    actual_issues_header_count += 1
        self.assertEqual(expected_issues_header_count, actual_issues_header_count)
        actual_commits_header_count = 0
        with open(dedup_commits, "r") as f:
            csv_reader = csv.reader(f, delimiter=",", quotechar="\"", quoting=csv.QUOTE_MINIMAL)
            for row in csv_reader:
                if row == COMMITS_HEADER:
                    actual_commits_header_count += 1
        self.assertEqual(expected_commits_header_count, actual_commits_header_count)
        actual_pull_requests_header_count = 0
        with open(dedup_pull_requests, "r") as f:
            csv_reader = csv.reader(f, delimiter=",", quotechar="\"", quoting=csv.QUOTE_MINIMAL)
            for row in csv_reader:
                if row == PULL_REQUESTS_HEADER:
                    actual_pull_requests_header_count += 1
        self.assertEqual(expected_pull_requests_header_count, actual_pull_requests_header_count)
        # Cleanup
        os.remove(dedup_issues)
        os.remove(dedup_commits)
        os.remove(dedup_pull_requests)

        #### Case 2 -- overwrite=True
        # Setup
        input_data_dir = os.path.join(CWD, "test_files/test_data5/")
        input_overwrite=True
        old_issues, old_commits, old_pull_requests = getDataFilepaths(input_data_dir)
        backup_issues = old_issues.split(".csv")[0] + "_backup.csv"
        backup_commits = old_commits.split(".csv")[0] + "_backup.csv"
        backup_pull_requests = old_pull_requests.split(".csv")[0] + "_backup.csv"
        shutil.copyfile(old_issues, backup_issues)
        shutil.copyfile(old_commits, backup_commits)
        shutil.copyfile(old_pull_requests, backup_pull_requests)
        dedup_issues = old_issues.split(".csv")[0] + "_dedup.csv"
        dedup_commits = old_commits.split(".csv")[0] + "_dedup.csv"
        dedup_pull_requests = old_pull_requests.split(".csv")[0] + "_dedup.csv"
        expected_issues_dedup_exists = False
        expected_commits_dedup_exists = False
        expected_pull_requests_dedup_exists = False
        # Test
        deduplicate(input_data_dir, input_overwrite)
        actual_issues_dedup_exists = doesPathExist(dedup_issues)
        actual_commits_dedup_exists = doesPathExist(dedup_commits)
        actual_pull_requests_dedup_exists = doesPathExist(dedup_pull_requests)
        self.assertEqual(expected_issues_dedup_exists, actual_issues_dedup_exists)
        self.assertEqual(expected_commits_dedup_exists, actual_commits_dedup_exists)
        self.assertEqual(expected_pull_requests_dedup_exists, actual_pull_requests_dedup_exists)
        # Cleanup
        os.remove(old_issues)
        os.remove(old_commits)
        os.remove(old_pull_requests)
        os.rename(backup_issues, old_issues)
        os.rename(backup_commits, old_commits)
        os.rename(backup_pull_requests, old_pull_requests)


class TestSearch(unittest.TestCase):
    """
    Test cases for function in src.search.
    """
    def setUp(self):
        """
        Necessary setup for test cases.
        """
        pass


    def test_search(self):
        """
        Test src.search:search().
        """
        #### Case 1 -- term="cherrypie", language="None", stars=0, total=10, save=False
        # Setup
        input_term = "cherrypie"
        input_stars = 0
        input_language = "None"
        input_total = 10
        input_save = False
        input_results = None
        # Test
        actual = search(input_term, input_stars, input_language, input_total, input_save, input_results, verbose=False)
        # With stars=0, the order that results are returned is not guaranteed. So, we can only test:
        self.assertEqual(10, len(actual))
        for element in actual:
            self.assertEqual(3, len(element))

        #### Case 2 -- term="cherrypie", language="Python", stars=0, total=0, save=False
        # Setup
        input_term = "cherrypie"
        input_stars = 0
        input_language = "Python"
        input_total = 0
        input_save = False
        input_results = None
        expected = [
            ["https://github.com/cherrypie623/CherryPie-Addon-Repository", 2, "Python"],
            ["https://github.com/zhengjiwen/cherrypie", 1, "Python"],
            ["https://github.com/further-i-go-less-i-know/cherrypie-ssl-errors", 0, "Python"]
        ]
        # Test
        actual = search(input_term, input_stars, input_language, input_total, input_save, input_results, verbose=False)
        self.assertListEqual(expected, actual)

        #### Case 3 -- term="cherrypie", language=Python, stars=1, total=0, save=False
        input_term = "cherrypie"
        input_stars = 1
        input_language = "Python"
        input_total = 0
        input_save = False
        input_results = None
        expected = [
            ["https://github.com/cherrypie623/CherryPie-Addon-Repository", 2, "Python"],
            ["https://github.com/zhengjiwen/cherrypie", 1, "Python"]
        ]
        # Test
        actual = search(input_term, input_stars, input_language, input_total, input_save, input_results, verbose=False)
        self.assertListEqual(expected, actual)

        #### Case 4 -- term="", language="", stars=20000, total=5, save=True
        input_term = ""
        input_stars = 20000
        input_language = ""
        input_total = 5
        input_save = True
        input_results = os.path.join(CWD, "search_results.txt")
        expected = [
            ["https://github.com/freeCodeCamp/freeCodeCamp", 342851, "TypeScript"],
            ["https://github.com/996icu/996.ICU", 261483, "None"],
            ["https://github.com/EbookFoundation/free-programming-books", 227318, "None"],
            ["https://github.com/jwasham/coding-interview-university", 214863, "None"],
            ["https://github.com/sindresorhus/awesome", 194592, "None"]
        ]
        expected_saved_results = [
            "https://github.com/freeCodeCamp/freeCodeCamp",
            "https://github.com/996icu/996.ICU",
            "https://github.com/EbookFoundation/free-programming-books",
            "https://github.com/jwasham/coding-interview-university",
            "https://github.com/sindresorhus/awesome"
        ]
        # Test
        actual = search(input_term, input_stars, input_language, input_total, input_save, input_results, verbose=False)
        # Number of stars might change, but top 5 repos probably won't. We need to test this way:
        exp = sorted(expected)
        act = sorted(actual)
        for i in range(0, 5):
            self.assertEqual(exp[i][0], act[i][0])
            self.assertEqual(exp[i][2], act[i][2])
        actual_saved_results = list()
        with open(input_results, "r") as f:
            for line in f:
                actual_saved_results.append(line.rstrip())
        self.assertListEqual(expected_saved_results, actual_saved_results)
        # Cleanup
        os.remove(input_results)


    def test_topRepos(self):
        """
        Test src.search:topRepos().
        """
        #### Case 1 -- languages="test", stars=30000
        # Setup
        input_languages = "test"
        input_stars = 30000
        input_results = os.path.join(CWD, "test_repo_urls.txt")
        input_verbose = False
        expected = [
            "https://github.com/0voice/interview_internal_reference", "https://github.com/d3/d3",
            "https://github.com/30-seconds/30-seconds-of-code", "https://github.com/3b1b/manim",
            "https://github.com/521xueweihan/HelloGitHub", "https://github.com/Anduin2017/HowToCook",
            "https://github.com/BVLC/caffe", "https://github.com/Blankj/AndroidUtilCode",
            "https://github.com/ColorlibHQ/AdminLTE", "https://github.com/Dogfalo/materialize", 
            "https://github.com/CorentinJ/Real-Time-Voice-Cloning", "https://github.com/Eugeny/tabby",
            "https://github.com/DefinitelyTyped/DefinitelyTyped", "https://github.com/GitSquared/edex-ui",
            "https://github.com/FortAwesome/Font-Awesome", "https://github.com/Genymobile/scrcpy",
            "https://github.com/GrowingGit/GitHub-Chinese-Top-Charts", "https://github.com/Homebrew/brew",
            "https://github.com/Leaflet/Leaflet", "https://github.com/MisterBooo/LeetCodeAnimation",
            "https://github.com/NARKOZ/hacker-scripts", "https://github.com/NationalSecurityAgency/ghidra",
            "https://github.com/NervJS/taro", "https://github.com/PhilJay/MPAndroidChart",
            "https://github.com/PowerShell/PowerShell", "https://github.com/ReactiveX/RxJava",
            "https://github.com/RocketChat/Rocket.Chat", "https://github.com/Semantic-Org/Semantic-UI",
            "https://github.com/Snailclimb/JavaGuide", "https://github.com/Textualize/rich",
            "https://github.com/TheAlgorithms/Java", "https://github.com/TheAlgorithms/Python",
            "https://github.com/TryGhost/Ghost", "https://github.com/skylot/jadx",
            "https://github.com/XX-net/XX-Net", "https://github.com/adam-p/markdown-here",
            "https://github.com/adobe/brackets", "https://github.com/agalwood/Motrix",
            "https://github.com/ageitgey/face_recognition", "https://github.com/airbnb/javascript",
            "https://github.com/airbnb/lottie-android", "https://github.com/apple/swift",
            "https://github.com/algorithm-visualizer/algorithm-visualizer",
            "https://github.com/alvarotrigo/fullPage.js", "https://github.com/angular/angular",
            "https://github.com/angular/angular.js", "https://github.com/ansible/ansible",
            "https://github.com/ant-design/ant-design", "https://github.com/ant-design/ant-design-pro",
            "https://github.com/anuraghazra/github-readme-stats", "https://github.com/apache/dubbo",
            "https://github.com/apache/echarts", "https://github.com/apache/superset",
            "https://github.com/apachecn/ailearning", "https://github.com/atom/atom",
            "https://github.com/awesome-selfhosted/awesome-selfhosted", "https://github.com/axios/axios",
            "https://github.com/azl397985856/leetcode", "https://github.com/babel/babel",
            "https://github.com/bilibili/ijkplayer", "https://github.com/bitcoin/bitcoin",
            "https://github.com/blueimp/jQuery-File-Upload", "https://github.com/bumptech/glide",
            "https://github.com/chartjs/Chart.js", "https://github.com/chinese-poetry/chinese-poetry",
            "https://github.com/coder/code-server", "https://github.com/commaai/openpilot",
            "https://github.com/cypress-io/cypress", "https://github.com/d2l-ai/d2l-zh",
            "https://github.com/danielmiessler/SecLists", "https://github.com/dcloudio/uni-app",
            "https://github.com/deepfakes/faceswap", "https://github.com/discourse/discourse",
            "https://github.com/django/django", "https://github.com/donnemartin/system-design-primer",
            "https://github.com/doocs/advanced-java", "https://github.com/dylanaraps/pure-bash-bible",
            "https://github.com/elastic/elasticsearch", "https://github.com/electron/electron",
            "https://github.com/expressjs/express", "https://github.com/facebook/create-react-app",
            "https://github.com/facebook/docusaurus", "https://github.com/facebook/jest",
            "https://github.com/facebook/react", "https://github.com/facebook/react-native",
            "https://github.com/faif/python-patterns", "https://github.com/fastlane/fastlane",
            "https://github.com/fighting41love/funNLP", "https://github.com/gatsbyjs/gatsby",
            "https://github.com/floodsung/Deep-Learning-Papers-Reading-Roadmap",
            "https://github.com/freeCodeCamp/freeCodeCamp", "https://github.com/geekxh/hello-algorithm",
            "https://github.com/getsentry/sentry", "https://github.com/git/git",
            "https://github.com/godotengine/godot", "https://github.com/goldbergyoni/nodebestpractices",
            "https://github.com/google-research/bert", "https://github.com/google/guava",
            "https://github.com/gothinkster/realworld", "https://github.com/grafana/grafana",
            "https://github.com/grpc/grpc", "https://github.com/gulpjs/gulp",
            "https://github.com/h5bp/html5-boilerplate", "https://github.com/hakimel/reveal.js",
            "https://github.com/hexojs/hexo", "https://github.com/home-assistant/core",
            "https://github.com/huggingface/transformers", "https://github.com/huginn/huginn",
            "https://github.com/iamkun/dayjs", "https://github.com/iluwatar/java-design-patterns",
            "https://github.com/immutable-js/immutable-js", "https://github.com/impress/impress.js",
            "https://github.com/ionic-team/ionic-framework", "https://github.com/iperov/DeepFaceLab",
            "https://github.com/iptv-org/iptv", "https://github.com/isocpp/CppCoreGuidelines",
            "https://github.com/jackfrued/Python-100-Days", "https://github.com/jaywcjlove/awesome-mac",
            "https://github.com/jekyll/jekyll", "https://github.com/jondot/awesome-react-native",
            "https://github.com/josephmisiti/awesome-machine-learning", "https://github.com/jquery/jquery",
            "https://github.com/juliangarnier/anime", "https://github.com/kamranahmedse/developer-roadmap",
            "https://github.com/kdn251/interviews", "https://github.com/keras-team/keras",
            "https://github.com/koajs/koa", "https://github.com/laravel/laravel",
            "https://github.com/leonardomso/33-js-concepts", "https://github.com/lerna/lerna",
            "https://github.com/localstack/localstack", "https://github.com/lodash/lodash",
            "https://github.com/macrozheng/mall", "https://github.com/marktext/marktext",
            "https://github.com/mermaid-js/mermaid", "https://github.com/meteor/meteor",
            "https://github.com/microsoft/PowerToys", "https://github.com/microsoft/TypeScript",
            "https://github.com/microsoft/Web-Dev-For-Beginners", "https://github.com/microsoft/playwright",
            "https://github.com/microsoft/terminal", "https://github.com/microsoft/vscode",
            "https://github.com/minimaxir/big-list-of-naughty-strings", "https://github.com/moment/moment",
            "https://github.com/mozilla/pdf.js", "https://github.com/mrdoob/three.js",
            "https://github.com/mui/material-ui", "https://github.com/nativefier/nativefier",
            "https://github.com/nestjs/nest", "https://github.com/netdata/netdata",
            "https://github.com/nodejs/node", "https://github.com/nolimits4web/swiper",
            "https://github.com/nuxt/nuxt.js", "https://github.com/nvbn/thefuck",
            "https://github.com/nvm-sh/nvm", "https://github.com/nwjs/nw.js",
            "https://github.com/obsproject/obs-studio", "https://github.com/ocornut/imgui",
            "https://github.com/ohmyzsh/ohmyzsh", "https://github.com/open-guides/og-aws",
            "https://github.com/opencv/opencv", "https://github.com/pallets/flask",
            "https://github.com/pandas-dev/pandas", "https://github.com/papers-we-love/papers-we-love",
            "https://github.com/parcel-bundler/parcel", "https://github.com/photonstorm/phaser",
            "https://github.com/php/php-src", "https://github.com/pi-hole/pi-hole",
            "https://github.com/pixijs/pixijs", "https://github.com/preactjs/preact",
            "https://github.com/prettier/prettier", "https://github.com/protocolbuffers/protobuf",
            "https://github.com/psf/requests", "https://github.com/public-apis/public-apis",
            "https://github.com/puppeteer/puppeteer", "https://github.com/python/cpython",
            "https://github.com/pytorch/pytorch", "https://github.com/quilljs/quill",
            "https://github.com/rails/rails", "https://github.com/redis/redis",
            "https://github.com/reduxjs/redux", "https://github.com/remix-run/react-router",
            "https://github.com/resume/resume.github.com", "https://github.com/sahat/hackathon-starter",
            "https://github.com/ryanmcdermott/clean-code-javascript", "https://github.com/scrapy/scrapy",
            "https://github.com/scikit-learn/scikit-learn", "https://github.com/serverless/serverless",
            "https://github.com/scutan90/DeepLearning-500-questions", "https://github.com/socketio/socket.io",
            "https://github.com/shadowsocks/shadowsocks", "https://github.com/shadowsocks/shadowsocks-windows",
            "https://github.com/sherlock-project/sherlock", "https://github.com/soimort/you-get",
            "https://github.com/spring-projects/spring-boot", "https://github.com/square/retrofit",
            "https://github.com/spring-projects/spring-framework", "https://github.com/storybookjs/storybook",
            "https://github.com/strapi/strapi", "https://github.com/styled-components/styled-components",
            "https://github.com/sveltejs/svelte", "https://github.com/swisskyrepo/PayloadsAllTheThings",
            "https://github.com/tailwindlabs/tailwindcss", "https://github.com/tensorflow/models",
            "https://github.com/tensorflow/tensorflow", "https://github.com/tesseract-ocr/tesseract",
            "https://github.com/testerSunshine/12306", "https://github.com/tiangolo/fastapi",
            "https://github.com/torvalds/linux", "https://github.com/trekhleb/javascript-algorithms",
            "https://github.com/twbs/bootstrap", "https://github.com/typescript-cheatsheets/react",
            "https://github.com/typicode/json-server", "https://github.com/ventoy/Ventoy",
            "https://github.com/vercel/hyper", "https://github.com/vercel/next.js",
            "https://github.com/videojs/video.js", "https://github.com/vinta/awesome-python",
            "https://github.com/vitejs/vite", "https://github.com/vuejs/vue",
            "https://github.com/vuetifyjs/vuetify", "https://github.com/webpack/webpack",
            "https://github.com/wg/wrk", "https://github.com/x64dbg/x64dbg",
            "https://github.com/yangshun/front-end-interview-handbook",
            "https://github.com/yangshun/tech-interview-handbook", "https://github.com/yarnpkg/yarn",
            "https://github.com/ytdl-org/youtube-dl", "https://github.com/zenorocha/clipboard.js",
            "https://github.com/google/zx", "https://github.com/supabase/supabase",
            "https://github.com/jaredpalmer/formik", "https://github.com/eugenp/tutorials",
            "https://github.com/gorhill/uBlock", "https://github.com/carbon-app/carbon",
            "https://github.com/nlohmann/json", "https://github.com/SheetJS/sheetjs",
            "https://github.com/vuejs/core", "https://github.com/jgraph/drawio-desktop",
            "https://github.com/Unitech/pm2"
        ]
        # Test
        actual = topRepos(input_languages, input_stars, input_results, input_verbose)
        # This changes sometimes, so I'm leaving these debug statements to help figure out what
        # changed
        #print("Remove:")
        #print(set(expected).difference(set(actual)))
        #print("Add:")
        #print(set(actual).difference(set(expected)))
        self.assertListEqual(sorted(expected), sorted(actual))


class TestPreprocess(unittest.TestCase):
    """
    Test cases for function in src.preprocess.
    """
    def setUp(self):
        """
        Necessary setup for test cases.
        """
        pass


    def test__stripNonWords(self):
        """
        Test src.preprocess:_stripNonWords().
        """
        # Setup
        test_cases = [
            "apples!", "oranges?", "Bananas; pears: plums", "carrots, celery, and\n beets",
            "onions,  CUCUMBERS,   and    peppers."
        ]
        expected = [
            "apples", "oranges", "bananas pears plums", "carrots celery and beets",
            "onions cucumbers and peppers"
        ]
        actual = list()
        # Test
        for case in test_cases:
            actual.append(_stripNonWords(case))
        self.assertListEqual(expected, actual)


    def test__lemmatize(self):
        """
        Test src.preprocess:_lemmatize().
        """
        # Setup
        test_cases = [
            "the quick brown fox jumps over the lazy dog",
            "i'm sorry if you're offended",
            "you wouldn't download a pizza would you",
            "only you can prevent forest fires",
            "don't forget to be awesome"
        ]
        expected = [
            "the quick brown fox jump over the lazy dog",
            "I be sorry if you be offend",
            "you would n't download a pizza would you",
            "only you can prevent forest fire",
            "do n't forget to be awesome"
        ]
        actual = list()
        # Test
        for case in test_cases:
            actual.append(_lemmatize(case))
        self.assertListEqual(expected, actual)


    def test_preprocess(self):
        """
        Test src.preprocess:preprocess().
        """
        #### Case 1 -- num_procs=1, overwrite=False
        # Setup
        input_data_dir = os.path.join(CWD, "test_files/test_data2/")
        input_num_procs = 1
        input_overwrite = False
        old_issues, old_commits, old_pull_requests = getDataFilepaths(input_data_dir)
        pre_issues = old_issues.split(".csv")[0] + "_preprocessed.csv"
        pre_commits = old_commits.split(".csv")[0] + "_preprocessed.csv"
        pre_pull_requests = old_pull_requests.split(".csv")[0] + "_preprocessed.csv"
        expected_issue_lemmas = [
            "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16",
            "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30",
            "31", "32", "33", "34", "35", "36", "37", "38", "39", "40", "41", "42", "43", "44",
            "45", "46", "47", "48", "49", "50", "51", "52", "53", "54", "55", "56", "57", "58",
            "59", "60", "61", "62", "63", "64", "65", "66", "67", "68", "69", "70", "71", "72",
            "73", "74", "75", "76", "77", "78", "79", "80", "81", "82", "83", "84", "85", "86",
            "87", "88", "89", "90", "91", "92", "93", "94", "95", "96", "97", "98", "99", "100",
            "101", "dummy comment", "sorry but I figure I 'd add a data point sorrynotsorry"
        ]
        expected_commit_lemmas = [
            "", "", "", "", "", "", "", "", "", "", "", "dummy commit comment", "1", "2", "3", "4",
            "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19",
            "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31", "32", "33",
            "34", "35", "36", "37", "38", "39", "40", "41", "42", "43", "44", "45", "46", "47",
            "48", "49", "50", "51", "52", "53", "54", "55", "56", "57", "58", "59", "60", "61",
            "62", "63", "64", "65", "66", "67", "68", "69", "70", "71", "72", "73", "74", "75",
            "76", "77", "78", "79", "80", "81", "82", "83", "84", "85", "86", "87", "88", "89",
            "90", "91", "92", "93", "94", "95", "96", "97", "98", "99", "100", "101", "", "", "",
            "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "",
            "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "" 
        ]
        expected_pull_request_lemmas = [
            "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16",
            "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30",
            "31", "32", "33", "34", "35", "36", "37", "38", "39", "40", "41", "42", "43", "44",
            "45", "46", "47", "48", "49", "50", "51", "52", "53", "54", "55", "56", "57", "58",
            "59", "60", "61", "62", "63", "64", "65", "66", "67", "68", "69", "70", "71", "72",
            "73", "74", "75", "76", "77", "78", "79", "80", "81", "82", "83", "84", "85", "86",
            "87", "88", "89", "90", "91", "92", "93", "94", "95", "96", "97", "98", "99", "100",
            "101", ""
        ]
        # Test
        actual_issue_lemmas, actual_commit_lemmas, actual_pull_request_lemmas = \
            preprocess(input_data_dir, input_num_procs, input_overwrite)
        self.assertListEqual(expected_issue_lemmas, actual_issue_lemmas)
        self.assertListEqual(expected_commit_lemmas, actual_commit_lemmas)
        self.assertListEqual(expected_pull_request_lemmas, actual_pull_request_lemmas)
        # Cleanup
        os.remove(pre_issues)
        os.remove(pre_commits)
        os.remove(pre_pull_requests)


class TestDevelopers(unittest.TestCase):
    """
    Test cases for functions in src.developers.
    """
    def setUp(self):
        """
        Necessary setup for test cases.
        """
        pass

    def test__flattenDicts(self):
        """
        Test src.developers:_flattenDicts().
        """
        # Setup
        dict_1 = {
            "ben": {"num_apology_lemmas": 42},
            "andy": {"num_apology_lemmas": 17},
            "nuthan": {"num_apology_lemmas": 7}
        }
        dict_2 = {
            "ben": {"num_apology_lemmas": 10},
            "andy": {"num_apology_lemmas": 32},
            "emilio": {"num_apology_lemmas": 48}
        }
        dict_3 = {
            "ben": {"num_apology_lemmas": 102},
            "paul": {"num_apology_lemmas": 0},
            "jen": {"num_apology_lemmas": 16}
        }
        input_dict_list = [dict_1, dict_2, dict_3]
        expected_flat_dict = {
            "ben": {"num_apology_lemmas": 154},
            "andy": {"num_apology_lemmas": 49},
            "nuthan": {"num_apology_lemmas": 7},
            "emilio": {"num_apology_lemmas": 48},
            "paul": {"num_apology_lemmas": 0},
            "jen": {"num_apology_lemmas": 16}
        }
        # Test
        actual_flat_dict = _flattenDicts(input_dict_list)
        self.assertDictEqual(expected_flat_dict, actual_flat_dict)

    def test__countDeveloperApologies(self):
        """
        Test src.developers:_countDeveloperApologies().
        """
        # Setup
        input_file_path = os.path.join(CWD, "test_files/test_data7/COBOL/issues/issues.csv")
        input_comment_author_index = 10
        input_num_apology_lemmas_index = 14
        expected_developers_dict = {
            "DanielRosenwasser": {"num_apology_lemmas": 0}, "OliverMaerz": {"num_apology_lemmas": 0},
            "martinkeen": {"num_apology_lemmas": 0}, "jazzyjackson": {"num_apology_lemmas": 0},
            "jmertic": {"num_apology_lemmas": 0}, "DStatWriter": {"num_apology_lemmas": 0},
            "sccosel": {"num_apology_lemmas": 0}, "paulnewt": {"num_apology_lemmas": 0},
            "leonardcohen58": {"num_apology_lemmas": 1}, "danpcconsult": {"num_apology_lemmas": 0},
            "JerethCutestory": {"num_apology_lemmas": 0}, "kas1830": {"num_apology_lemmas": 1},
            "MartinYeung5": {"num_apology_lemmas": 0}, "JoAnnaEsq": {"num_apology_lemmas": 0},
            "varlux": {"num_apology_lemmas": 2}, "OlegKunitsyn": {"num_apology_lemmas": 1},
            "CmdrZin": {"num_apology_lemmas": 0}, "Dlthomass": {"num_apology_lemmas": 0},
            "brunopacheco1": {"num_apology_lemmas": 0}, "michael-conrad": {"num_apology_lemmas": 0},
            "navarretedaniel": {"num_apology_lemmas": 0}, "mconfoy": {"num_apology_lemmas": 0},
            "Sudharsana-Srinivasan": {"num_apology_lemmas": 0}, "ftpo": {"num_apology_lemmas": 0},
            "bhowe": {"num_apology_lemmas": 0}, "rsac56": {"num_apology_lemmas": 0},
            "Eliana88": {"num_apology_lemmas": 0}, "BilltheK": {"num_apology_lemmas": 0},
            "johnpevans": {"num_apology_lemmas": 0}, "jrmalone93": {"num_apology_lemmas": 0},
            "cwansart": {"num_apology_lemmas": 0}, "Schekn": {"num_apology_lemmas": 0},
            "JosefKaser": {"num_apology_lemmas": 0}, "camdroid": {"num_apology_lemmas": 0},
            "tathoma": {"num_apology_lemmas": 0}, "al-heisner": {"num_apology_lemmas": 0},
            "timretout": {"num_apology_lemmas": 0}, "java007md": {"num_apology_lemmas": 0},
            "GreenRoemer": {"num_apology_lemmas": 2}, "staceylmarch": {"num_apology_lemmas": 0},
            "MrGmaw": {"num_apology_lemmas": 0}, "negovan13": {"num_apology_lemmas": 0},
            "WellBattle6": {"num_apology_lemmas": 0}, "cobol10": {"num_apology_lemmas": 0},
            "marianpg12": {"num_apology_lemmas": 0}, "fromer97": {"num_apology_lemmas": 0},
            "sabybasu": {"num_apology_lemmas": 1}, "AidanFarhi": {"num_apology_lemmas": 0},
            "jtrevithick": {"num_apology_lemmas": 0}, "RowReal": {"num_apology_lemmas": 0},
            "MikeBauerCA": {"num_apology_lemmas": 0}, "ibrahimaktasgithub": {"num_apology_lemmas": 1},
            "ejimenezISEP": {"num_apology_lemmas": 0}, "dennisad": {"num_apology_lemmas": 0},
            "jacqpot": {"num_apology_lemmas": 0}, "timdsaunders1": {"num_apology_lemmas": 0},
            "venkatzhub": {"num_apology_lemmas": 0}, "EddieCavic": {"num_apology_lemmas": 0},
            "moonman7": {"num_apology_lemmas": 0}, "broarr": {"num_apology_lemmas": 0},
            "RageshAntony": {"num_apology_lemmas": 0}, "comps3": {"num_apology_lemmas": 0},
            "bkline": {"num_apology_lemmas": 0}, "abickerton": {"num_apology_lemmas": 0},
            "mthomp9838": {"num_apology_lemmas": 0}, "thyarles": {"num_apology_lemmas": 0},
            "zvookiejoo": {"num_apology_lemmas": 0}, "jellypuno": {"num_apology_lemmas": 0},
            "OldGuy86": {"num_apology_lemmas": 0}, "BaileyH": {"num_apology_lemmas": 0},
            "steven-piot": {"num_apology_lemmas": 0}, "MLo8": {"num_apology_lemmas": 3},
            "jackson-024": {"num_apology_lemmas": 0}, "gowide00": {"num_apology_lemmas": 0},
            "mikedblum": {"num_apology_lemmas": 0}, "binary-sequence": {"num_apology_lemmas": 0},
            "sandeep-sparrow": {"num_apology_lemmas": 0}, "seahopki12": {"num_apology_lemmas": 0},
            "FranklinChen": {"num_apology_lemmas": 0}, "edrubins": {"num_apology_lemmas": 0},
            "peraciodias": {"num_apology_lemmas": 0}, "tanto259": {"num_apology_lemmas": 0},
            "michalblaszak": {"num_apology_lemmas": 0}, "zeibura": {"num_apology_lemmas": 0},
            "edack": {"num_apology_lemmas": 0}, "raven300": {"num_apology_lemmas": 0},
            "ravindrachechani": {"num_apology_lemmas": 0}, "moueza": {"num_apology_lemmas": 0},
            "mpettis": {"num_apology_lemmas": 0}, "binishantony": {"num_apology_lemmas": 0},
            "DanielSReynoso": {"num_apology_lemmas": 0}, "wawesomeNOGUI": {"num_apology_lemmas": 0}
        }
        # Test
        actual_developers_dict = _countDeveloperApologies(
            input_file_path, input_comment_author_index, input_num_apology_lemmas_index
        )
        self.assertDictEqual(expected_developers_dict, actual_developers_dict)


    def test__getDeveloperDicts(self):
        """
        Test src.developers:_geetDeveloperDicts().
        """
        #### Case 1
        # Setup
        input_file_path = os.path.join(CWD, "test_files/test_data7/COBOL/")
        expected_developers_dict = {
            "edack": {"num_apology_lemmas": 0}, "zvookiejoo": {"num_apology_lemmas": 0},
            "jrmalone93": {"num_apology_lemmas": 0}, "martinkeen": {"num_apology_lemmas": 0},
            "OlegKunitsyn": {"num_apology_lemmas": 1}, "ibrahimaktasgithub": {"num_apology_lemmas": 1},
            "bhowe": {"num_apology_lemmas": 0}, "FranklinChen": {"num_apology_lemmas": 0},
            "sabybasu": {"num_apology_lemmas": 1}, "tathoma": {"num_apology_lemmas": 0},
            "GreenRoemer": {"num_apology_lemmas": 2}, "tanto259": {"num_apology_lemmas": 1},
            "Eliana88": {"num_apology_lemmas": 0}, "rsac56": {"num_apology_lemmas": 0},
            "JerethCutestory": {"num_apology_lemmas": 0}, "moonman7": {"num_apology_lemmas": 0},
            "MikeBauerCA": {"num_apology_lemmas": 1}, "BaileyH": {"num_apology_lemmas": 0},
            "salisbuk7897": {"num_apology_lemmas": 0}, "cwansart": {"num_apology_lemmas": 0},
            "ftpo": {"num_apology_lemmas": 0}, "Schekn": {"num_apology_lemmas": 0},
            "timdsaunders1": {"num_apology_lemmas": 0}, "michalblaszak": {"num_apology_lemmas": 0},
            "Dlthomass": {"num_apology_lemmas": 0}, "edrubins": {"num_apology_lemmas": 0},
            "venkatzhub": {"num_apology_lemmas": 0}, "detinsley1s": {"num_apology_lemmas": 0},
            "RowReal": {"num_apology_lemmas": 0}, "moueza": {"num_apology_lemmas": 0},
            "raven300": {"num_apology_lemmas": 1}, "Sudharsana-Srinivasan": {"num_apology_lemmas": 0},
            "mikedblum": {"num_apology_lemmas": 0}, "marianpg12": {"num_apology_lemmas": 0},
            "wawesomeNOGUI": {"num_apology_lemmas": 0}, "MLo8": {"num_apology_lemmas": 3},
            "binary-sequence": {"num_apology_lemmas": 0}, "WellBattle6": {"num_apology_lemmas": 0},
            "ravindrachechani": {"num_apology_lemmas": 0}, "bkline": {"num_apology_lemmas": 0},
            "al-heisner": {"num_apology_lemmas": 0}, "mthomp9838": {"num_apology_lemmas": 0},
            "ifctgerardo": {"num_apology_lemmas": 0}, "RageshAntony": {"num_apology_lemmas": 0},
            "DanielSReynoso": {"num_apology_lemmas": 0}, "sandeep-sparrow": {"num_apology_lemmas": 0},
            "negovan13": {"num_apology_lemmas": 0}, "JosefKaser": {"num_apology_lemmas": 0},
            "binishantony": {"num_apology_lemmas": 0}, "thyarles": {"num_apology_lemmas": 0},
            "jmertic": {"num_apology_lemmas": 0}, "cobol10": {"num_apology_lemmas": 0},
            "MrGmaw": {"num_apology_lemmas": 0}, "kas1830": {"num_apology_lemmas": 1},
            "JoAnnaEsq": {"num_apology_lemmas": 0}, "jacqpot": {"num_apology_lemmas": 0},
            "mconfoy": {"num_apology_lemmas": 0}, "ahmedEid1": {"num_apology_lemmas": 4},
            "CmdrZin": {"num_apology_lemmas": 0}, "timretout": {"num_apology_lemmas": 0},
            "jackson-024": {"num_apology_lemmas": 0}, "noct4": {"num_apology_lemmas": 0},
            "varlux": {"num_apology_lemmas": 2}, "EddieCavic": {"num_apology_lemmas": 0},
            "DanielRosenwasser": {"num_apology_lemmas": 0}, "MartinYeung5": {"num_apology_lemmas": 0},
            "sccosel": {"num_apology_lemmas": 2}, "danpcconsult": {"num_apology_lemmas": 0},
            "jazzyjackson": {"num_apology_lemmas": 0}, "johnpevans": {"num_apology_lemmas": 0},
            "seahopki12": {"num_apology_lemmas": 0}, "brunopacheco1": {"num_apology_lemmas": 0},
            "michael-conrad": {"num_apology_lemmas": 0}, "klausmelcher": {"num_apology_lemmas": 1},
            "jtrevithick": {"num_apology_lemmas": 0}, "ChrisBoehmCA": {"num_apology_lemmas": 0},
            "ejimenezISEP": {"num_apology_lemmas": 0}, "mpettis": {"num_apology_lemmas": 0},
            "OliverMaerz": {"num_apology_lemmas": 0}, "comps3": {"num_apology_lemmas": 0},
            "dennisad": {"num_apology_lemmas": 0}, "broarr": {"num_apology_lemmas": 0},
            "peraciodias": {"num_apology_lemmas": 0}, "steven-piot": {"num_apology_lemmas": 0},
            "leonardcohen58": {"num_apology_lemmas": 1}, "zeibura": {"num_apology_lemmas": 0},
            "java007md": {"num_apology_lemmas": 0}, "staceylmarch": {"num_apology_lemmas": 0},
            "abickerton": {"num_apology_lemmas": 0}, "navarretedaniel": {"num_apology_lemmas": 0},
            "DStatWriter": {"num_apology_lemmas": 0}, "jellypuno": {"num_apology_lemmas": 1},
            "camdroid": {"num_apology_lemmas": 0}, "BilltheK": {"num_apology_lemmas": 0},
            "paulnewt": {"num_apology_lemmas": 0}, "bz8g3d": {"num_apology_lemmas": 0},
            "Rickster66": {"num_apology_lemmas": 0}, "gowide00": {"num_apology_lemmas": 0},
            "fromer97": {"num_apology_lemmas": 0}, "OldGuy86": {"num_apology_lemmas": 0},
            "AidanFarhi": {"num_apology_lemmas": 0}
        }
        # Test
        actual_developers_dict = _getDeveloperDicts(input_file_path)
        self.assertDictEqual(expected_developers_dict, actual_developers_dict)


    def test_developerStats(self):
        """
        Test src.developers:developerStats().
        """
        # Setup
        input_file_path = os.path.join(CWD, "test_files/test_data7/")
        input_num_procs = 2
        expected_developers_dict = {
            "Dlthomass": {"num_apology_lemmas": 0}, "Cheukting": {"num_apology_lemmas": 0},
            "danpcconsult": {"num_apology_lemmas": 0}, "staceylmarch": {"num_apology_lemmas": 0},
            "kghenderson": {"num_apology_lemmas": 0}, "dennisad": {"num_apology_lemmas": 0},
            "cwansart": {"num_apology_lemmas": 0}, "kevinchekovfeeney": {"num_apology_lemmas": 0},
            "MrRaymondLee": {"num_apology_lemmas": 0}, "ravindrachechani": {"num_apology_lemmas": 0},
            "andeplane": {"num_apology_lemmas": 0}, "binary-sequence": {"num_apology_lemmas": 0},
            "BilltheK": {"num_apology_lemmas": 0}, "k-tipp": {"num_apology_lemmas": 0},
            "al-heisner": {"num_apology_lemmas": 0}, "broarr": {"num_apology_lemmas": 0},
            "wawesomeNOGUI": {"num_apology_lemmas": 0}, "brunopacheco1": {"num_apology_lemmas": 0},
            "michael-conrad": {"num_apology_lemmas": 0}, "Sudharsana-Srinivasan": {"num_apology_lemmas": 0},
            "matko": {"num_apology_lemmas": 2}, "paulnewt": {"num_apology_lemmas": 0},
            "MichaelSavin": {"num_apology_lemmas": 0}, "ahmedEid1": {"num_apology_lemmas": 4},
            "camdroid": {"num_apology_lemmas": 0}, "CmdrZin": {"num_apology_lemmas": 0},
            "RageshAntony": {"num_apology_lemmas": 0}, "seahopki12": {"num_apology_lemmas": 0},
            "hoijnet": {"num_apology_lemmas": 0}, "joepio": {"num_apology_lemmas": 0},
            "Schekn": {"num_apology_lemmas": 0}, "nemanjavuk": {"num_apology_lemmas": 0},
            "edrubins": {"num_apology_lemmas": 0}, "cobol10": {"num_apology_lemmas": 0},
            "abickerton": {"num_apology_lemmas": 0}, "Eliana88": {"num_apology_lemmas": 0},
            "bhowe": {"num_apology_lemmas": 0}, "kaaloo": {"num_apology_lemmas": 0},
            "MrGmaw": {"num_apology_lemmas": 0}, "jtrevithick": {"num_apology_lemmas": 0},
            "ifctgerardo": {"num_apology_lemmas": 0}, "zeibura": {"num_apology_lemmas": 0},
            "bionicles": {"num_apology_lemmas": 0}, "edack": {"num_apology_lemmas": 0},
            "trialblaze-bt": {"num_apology_lemmas": 0}, "EddieCavic": {"num_apology_lemmas": 0},
            "mikedblum": {"num_apology_lemmas": 0}, "raven300": {"num_apology_lemmas": 1},
            "eddiejaoude": {"num_apology_lemmas": 0}, "WellBattle6": {"num_apology_lemmas": 0},
            "noct4": {"num_apology_lemmas": 0}, "jacqpot": {"num_apology_lemmas": 0},
            "OlegKunitsyn": {"num_apology_lemmas": 1}, "sandeep-sparrow": {"num_apology_lemmas": 0},
            "java007md": {"num_apology_lemmas": 0}, "RowReal": {"num_apology_lemmas": 0},
            "rahuldeepattri": {"num_apology_lemmas": 0}, "steven-piot": {"num_apology_lemmas": 0},
            "mconfoy": {"num_apology_lemmas": 0}, "GavinMendelGleason": {"num_apology_lemmas": 0},
            "moonman7": {"num_apology_lemmas": 0}, "KarlLevik": {"num_apology_lemmas": 0},
            "marvin-hansen": {"num_apology_lemmas": 0}, "MLo8": {"num_apology_lemmas": 3},
            "DStatWriter": {"num_apology_lemmas": 0}, "dwinston": {"num_apology_lemmas": 0},
            "rrooij": {"num_apology_lemmas": 1}, "gowide00": {"num_apology_lemmas": 0},
            "salisbuk7897": {"num_apology_lemmas": 0}, "DanielRosenwasser": {"num_apology_lemmas": 0},
            "dependabot": {"num_apology_lemmas": 0}, "sabybasu": {"num_apology_lemmas": 1},
            "michalblaszak": {"num_apology_lemmas": 0}, "marianpg12": {"num_apology_lemmas": 0},
            "jackson-024": {"num_apology_lemmas": 0}, "martinkeen": {"num_apology_lemmas": 0},
            "BaileyH": {"num_apology_lemmas": 0}, "timretout": {"num_apology_lemmas": 0},
            "zvookiejoo": {"num_apology_lemmas": 0}, "moueza": {"num_apology_lemmas": 0},
            "JerethCutestory": {"num_apology_lemmas": 0}, "mthomp9838": {"num_apology_lemmas": 0},
            "EmmanuelOga": {"num_apology_lemmas": 0}, "JosefKaser": {"num_apology_lemmas": 0},
            "DanielSReynoso": {"num_apology_lemmas": 0}, "KittyJose": {"num_apology_lemmas": 0},
            "sachinbhutani": {"num_apology_lemmas": 0}, "fromer97": {"num_apology_lemmas": 0},
            "ChrisBoehmCA": {"num_apology_lemmas": 0}, "ansarizafar": {"num_apology_lemmas": 0},
            "ghpu": {"num_apology_lemmas": 0}, "LogicalDash": {"num_apology_lemmas": 0},
            "rsac56": {"num_apology_lemmas": 0}, "FranklinChen": {"num_apology_lemmas": 0},
            "Francesca-Bit": {"num_apology_lemmas": 0}, "jmertic": {"num_apology_lemmas": 0},
            "ejimenezISEP": {"num_apology_lemmas": 0}, "jrmalone93": {"num_apology_lemmas": 0},
            "detinsley1s": {"num_apology_lemmas": 0}, "leonardcohen58": {"num_apology_lemmas": 1},
            "chrisshortlaw": {"num_apology_lemmas": 0}, "evildmp": {"num_apology_lemmas": 0},
            "MartinYeung5": {"num_apology_lemmas": 0}, "kamleshjoshi8102": {"num_apology_lemmas": 0},
            "bkline": {"num_apology_lemmas": 0}, "sccosel": {"num_apology_lemmas": 2},
            "kas1830": {"num_apology_lemmas": 1}, "thyarles": {"num_apology_lemmas": 0},
            "mpettis": {"num_apology_lemmas": 0}, "binishantony": {"num_apology_lemmas": 0},
            "jellypuno": {"num_apology_lemmas": 1}, "tathoma": {"num_apology_lemmas": 0},
            "ibrahimaktasgithub": {"num_apology_lemmas": 1}, "tanto259": {"num_apology_lemmas": 1},
            "peraciodias": {"num_apology_lemmas": 0}, "navarretedaniel": {"num_apology_lemmas": 0},
            "AidanFarhi": {"num_apology_lemmas": 0}, "yashasvimisra2798": {"num_apology_lemmas": 0},
            "luke-feeney": {"num_apology_lemmas": 2}, "mikkokotila": {"num_apology_lemmas": 0},
            "timdsaunders1": {"num_apology_lemmas": 0}, "jazzyjackson": {"num_apology_lemmas": 0},
            "Rickster66": {"num_apology_lemmas": 0}, "OliverMaerz": {"num_apology_lemmas": 0},
            "spl": {"num_apology_lemmas": 1}, "GreenRoemer": {"num_apology_lemmas": 2},
            "OldGuy86": {"num_apology_lemmas": 0}, "pmoura": {"num_apology_lemmas": 2},
            "comps3": {"num_apology_lemmas": 0}, "bz8g3d": {"num_apology_lemmas": 0},
            "henryjcee": {"num_apology_lemmas": 2}, "NeelParihar": {"num_apology_lemmas": 0},
            "johnpevans": {"num_apology_lemmas": 0}, "klausmelcher": {"num_apology_lemmas": 1},
            "venkatzhub": {"num_apology_lemmas": 0}, "MikeBauerCA": {"num_apology_lemmas": 1},
            "pwin": {"num_apology_lemmas": 0}, "bascott": {"num_apology_lemmas": 0},
            "JoAnnaEsq": {"num_apology_lemmas": 0}, "negovan13": {"num_apology_lemmas": 0},
            "jbennettgit": {"num_apology_lemmas": 0}, "jamesnvc": {"num_apology_lemmas": 0},
            "AstroChelonian": {"num_apology_lemmas": 0}, "ftpo": {"num_apology_lemmas": 0},
            "varlux": {"num_apology_lemmas": 2}
        }
        # Test
        actual_developers_dict = developerStats(input_file_path, input_num_procs)
        self.assertDictEqual(expected_developers_dict, actual_developers_dict)



class TestApologies(unittest.TestCase):
    """
    Test cases for function in src.apologies.
    """
    def setUp(self):
        """
        Necessary setup for test cases.
        """
        pass


    def test__labelApologies(self):
        """
        Test src.apologies:_labelApologies().
        """
        # Setup
        test_cases = [
            "0", "1", "2", "3", "4", "5", "10", "156", "2345", "56789", "0"
        ]
        expected_labels = [
            "0", "1", "1", "1", "1", "1", "1", "1", "1", "1", "0"
        ]
        actual_labels = list()
        # Test
        for case in test_cases:
            actual_labels.append(_labelApologies(case))
        self.assertListEqual(expected_labels, actual_labels)

    
    def test__countApologies(self):
        """
        Test src.apologies:_countApologies().
        """
        # Setup
        test_cases = [
            "I need to apologize for accidentally drive my car into your house",
            "sorry I forget how to use the brake",
            "pardon the interruption",
            "I be mistaken that should have be + = not =",
            "I 'll take the blame for this",
            "who write this spaghetti code",
            "oops that should have be delete",
            "that be my fault for not unit testing",
            "I apologize I be so sorry about this",
            "I do nothing wrong it work as intend",
            "better safe than sorry",
            "it 's not -PRON- fault",
            "sorry but this is necessary. better safe than sorry",
        ]
        expected_apologies = [
            "1", "1", "1", "1", "1", "0", "1", "1", "2", "0", "0", "0", "1"
        ]
        actual_apologies = list()
        # Test
        for case in test_cases:
            actual_apologies.append(_countApologies(case))
        self.assertListEqual(expected_apologies, actual_apologies)


    def test_classify(self):
        """
        Test src.apologies:classify().
        """
        # Setup
        input_data_dir = os.path.join(CWD, "test_files/test_data3/")
        input_num_procs = 1
        input_overwrite = False
        old_issues, old_commits, old_pull_requests = getDataFilepaths(input_data_dir)
        class_issues = old_issues.split(".csv")[0] + "_classified.csv"
        class_commits = old_commits.split(".csv")[0] + "_classified.csv"
        class_pull_requests = old_pull_requests.split(".csv")[0] + "_classified.csv"
        expected_issue_classes = [
            ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"],
            ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"],
            ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"],
            ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"],
            ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"],
            ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"],
            ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"],
            ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"],
            ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"],
            ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"],
            ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"],
            ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"],
            ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"],
            ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"],
            ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"], ["1", "1"]
        ]
        expected_commit_classes = [
            ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"],
            ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"],
            ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"],
            ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"],
            ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"],
            ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"],
            ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"],
            ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"],
            ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"],
            ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"],
            ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"],
            ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"],
            ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"],
            ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"],
            ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"],
            ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"],
            ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"],
            ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"],
            ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"],
            ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"],
            ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"],
            ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"],
            ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"]
        ]
        expected_pull_request_classes = [
            ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"],
            ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"],
            ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"],
            ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"],
            ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"],
            ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"],
            ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"],
            ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"],
            ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"],
            ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"],
            ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"],
            ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"],
            ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"],
            ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"],
            ["0", "0"], ["0", "0"], ["0", "0"], ["0", "0"]
        ]
        # Test
        actual_issue_classes, actual_commit_classes, actual_pull_request_classes = \
            classify(input_data_dir, input_num_procs, input_overwrite)
        self.assertListEqual(expected_issue_classes, actual_issue_classes)
        self.assertListEqual(expected_commit_classes, actual_commit_classes)
        self.assertListEqual(expected_pull_request_classes, actual_pull_request_classes)
        # Cleanup
        os.remove(class_issues)
        os.remove(class_commits)
        os.remove(class_pull_requests)


class TestRandom(unittest.TestCase):
    """
    Test cases for function in src.random.
    """
    def setUp(self):
        """
        Necessary setup for test cases.
        """
        pass


    def test__getSourceFromFilepath(self):
        """
        Test src.random:_getSourceFromFilepath().
        """
        # Setup
        filepaths = [
            "/path/to/data/commits/commits.csv",
            "/path/to/data/issues/issues.csv",
            "/path/to/data/pull_requests/pull_requests.csv",
            "/path/to/data/issues/issues_preprocessed.csv"
        ]
        expected_sources = ["CO", "IS", "PR", None]
        # Test
        actual_sources = [_getSourceFromFilepath(fp) for fp in filepaths]
        self.assertListEqual(expected_sources, actual_sources)


    def test__getPopulationFilepaths(self):
        """
        Test src.random:_getPopulationFilepaths().
        """
        #### Case 1 -- source="IS"
        # Setup
        data_dir = os.path.join(CWD, "test_files/test_data4/")
        source = "IS"
        expected_paths = [
            os.path.join(CWD, "test_files/test_data4/subdir1/issues/issues.csv"),
            os.path.join(CWD, "test_files/test_data4/subdir2/issues/issues.csv")
        ]
        # Test
        actual_paths = _getPopulationFilepaths(data_dir, source)
        self.assertListEqual(expected_paths, actual_paths)

        #### Case 2 -- source="CO"
        # Setup
        source = "CO"
        expected_paths = [
            os.path.join(CWD, "test_files/test_data4/subdir1/commits/commits.csv"),
            os.path.join(CWD, "test_files/test_data4/subdir2/commits/commits.csv")
        ]
        # Test
        actual_paths = _getPopulationFilepaths(data_dir, source)
        self.assertListEqual(expected_paths, actual_paths)

        #### Case 3 -- source="PR"
        # Setup
        source = "PR"
        expected_paths = [
            os.path.join(CWD, "test_files/test_data4/subdir1/pull_requests/pull_requests.csv"),
            os.path.join(CWD, "test_files/test_data4/subdir2/pull_requests/pull_requests.csv")
        ]
        # Test
        actual_paths = _getPopulationFilepaths(data_dir, source)
        self.assertListEqual(expected_paths, actual_paths)

        #### Case 4 -- source="ALL"
        # Setup
        source = "ALL"
        expected_paths = [
            os.path.join(CWD, "test_files/test_data4/subdir1/commits/commits.csv"),
            os.path.join(CWD, "test_files/test_data4/subdir2/commits/commits.csv"),
            os.path.join(CWD, "test_files/test_data4/subdir1/issues/issues.csv"),
            os.path.join(CWD, "test_files/test_data4/subdir2/issues/issues.csv"),
            os.path.join(CWD, "test_files/test_data4/subdir1/pull_requests/pull_requests.csv"),
            os.path.join(CWD, "test_files/test_data4/subdir2/pull_requests/pull_requests.csv")
        ]
        # Test
        actual_paths = _getPopulationFilepaths(data_dir, source)
        self.assertListEqual(expected_paths, actual_paths)


    def test__deduplicateHeaders(self):
        """
        Test src.random:_deduplicateHeaders().
        """
        # Setup
        rows = [
            ["HEADER_1", "HEADER_2", "HEADER_3"],
            ["Batman", "Superman", "WonderWoman"],
            ["Apples", "Oranges", "Bananas"],
            ["HEADER_1", "HEADER_2", "HEADER_3"],
            ["HEADER_1", "HEADER_2", "HEADER_3"],
            ["Linux", "Windows", "Macintosh"],
            ["HEADER_1", "HEADER_2", "HEADER_3"]
        ]
        header = ["HEADER_1", "HEADER_2", "HEADER_3"]
        expected = [
            ["Batman", "Superman", "WonderWoman"],
            ["Apples", "Oranges", "Bananas"],
            ["Linux", "Windows", "Macintosh"]
        ]
        # Test
        actual = _deduplicateHeaders(rows, header)
        self.assertListEqual(expected, actual)

            
    def test__randomSample(self):
        """
        Test src.random:randomSample().
        """
        #### Case 1
        # Setup
        data_dir = os.path.join(CWD, "test_files/test_data4/")
        sample_size = 20
        apologies_only = False
        source = "ALL"
        output_path = os.path.join(CWD, "test_files/random_sample_20.csv")
        export_all = False
        # Test
        randomSample(data_dir, sample_size, apologies_only, source, output_path, export_all)
        with open(output_path, "r", encoding="utf-8") as f:
            csv_reader = csv.reader(f, delimiter=",", quotechar="\"", quoting=csv.QUOTE_MINIMAL)

            header = next(csv_reader)
            rows = list()
            for row in csv_reader:
                rows.append(row)

        self.assertEqual(sample_size, len(rows))
        self.assertEqual(ABRIDGED_HEADER, header)
        for row in rows:
            self.assertEqual(len(ABRIDGED_HEADER), len(row))
        # Cleanup
        os.remove(output_path)

        #### Case 2
        # Setup
        data_dir = os.path.join(CWD, "test_files/test_data4/")
        sample_size = 100
        apologies_only = False
        source = "IS"
        output_path = os.path.join(CWD, "test_files/random_sample_100.csv")
        export_all = False
        # Test
        randomSample(data_dir, sample_size, apologies_only, source, output_path, export_all)
        with open(output_path, "r", encoding="utf-8") as f:
            csv_reader = csv.reader(f, delimiter=",", quotechar="\"", quoting=csv.QUOTE_MINIMAL)

            header = next(csv_reader)
            rows = list()
            for row in csv_reader:
                rows.append(row)

        self.assertEqual(sample_size, len(rows))
        self.assertEqual(ABRIDGED_HEADER, header)
        for row in rows:
            self.assertEqual(len(ABRIDGED_HEADER), len(row))
            self.assertEqual(source, row[0])
        # Cleanup
        os.remove(output_path)

        #### Case 3
        # Setup
        data_dir = os.path.join(CWD, "test_files/test_data4/")
        sample_size = 1
        apologies_only = True
        source = "ALL"
        output_path = os.path.join(CWD, "test_files/random_sample_1.csv")
        export_all = False
        # Test
        randomSample(data_dir, sample_size, apologies_only, source, output_path, export_all)
        with open(output_path, "r", encoding="utf-8") as f:
            csv_reader = csv.reader(f, delimiter=",", quotechar="\"", quoting=csv.QUOTE_MINIMAL)

            header = next(csv_reader)
            rows = list()
            for row in csv_reader:
                rows.append(row)

        self.assertEqual(sample_size, len(rows))
        self.assertEqual(ABRIDGED_HEADER, header)
        for row in rows:
            self.assertEqual(len(ABRIDGED_HEADER), len(row))
            self.assertEqual("1", row[-1])
        # Cleanup
        os.remove(output_path)

        #### Case 4
        # Setup
        data_dir = os.path.join(CWD, "test_files/test_data4/")
        sample_size = 50
        apologies_only = False
        source = "ALL"
        output_path = os.path.join(CWD, "test_files/random_samples/")
        os.mkdir(output_path)
        export_all = True
        # Test
        randomSample(data_dir, sample_size, apologies_only, source, output_path, export_all)
        filenames = getFilenames(output_path)
        for fname in filenames:
            with open(os.path.join(output_path, fname), "r", encoding="utf-8") as f:
                csv_reader = csv.reader(f, delimiter=",", quotechar="\"", quoting=csv.QUOTE_MINIMAL)

                header = next(csv_reader)

                rows = list()
                for row in csv_reader:
                    self.assertEqual(len(ABRIDGED_HEADER), len(row))
                    rows.append(row)
            try:
                self.assertEqual(sample_size, len(rows))
            except:
                self.assertLess(len(rows), sample_size)
            self.assertEqual(ABRIDGED_HEADER, header)
        # Cleanup
        shutil.rmtree(output_path)


class TestStats(unittest.TestCase):
    """
    Test cases for function in src.stats.
    """
    def setUp(self):
        """
        Necessary setup for test cases.
        """
        pass


    def test_stats(self):
        """
        Test src.stats:stats().
        """
        # Setup
        data_dir = os.path.join(CWD, "test_files/test_data4/")
        num_procs = 1
        expected_stats_dict = {
            "apologies": {
                "total": 1,
                "wc_total": 10,
                "wc_individual": [10],
                "wc_mean": 10.0,
                "wc_median": 10,
                "wc_min": 10,
                "wc_max": 10,
                "lc_total": 1,
                "lc_individual": [1],
                "lc_mean": 1.0,
                "lc_median": 1,
                "lc_min": 1,
                "lc_max": 1
            }, "non-apologies": {
                "total": 309,
                "wc_total": 404,
                "wc_individual": [
                    2, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 56, 14, 24, 1,
                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1,
                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                "wc_mean": 1.307443365695793,
                "wc_median": 1,
                "wc_min": 1,
                "wc_max": 56
            }, "lemmas": {
                "apology": 0,
                "apologise": 0,
                "apologize": 0,
                "blame": 0,
                "excuse": 0,
                "fault": 0,
                "forgive": 0,
                "mistake": 0,
                "mistaken": 0,
                "oops": 0,
                "pardon": 0,
                "regret": 0,
                "sorry": 1
            }
        }
        # Test
        actual_stats_dict = stats(data_dir, num_procs, verbose=False)
        self.assertDictEqual(expected_stats_dict, actual_stats_dict)


#### MAIN ##########################################################################################
if __name__ == "__main__":
    unittest.main(warnings="ignore")

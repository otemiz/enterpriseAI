import os
from atlassian import Jira
from requests import HTTPError

# pip install atlassian-python-api
# pip install jira


def main():

    jira = Jira(
        url   = "https://oguzhanilter.atlassian.net/jira/",
        token = os.environ.get('JIRA_API_KEY')
    )

    try:

        proj = jira.get_project('Simulated Project 1', expand=None)
        print(proj.get("id")) # to verify that authentication worked
    
    except HTTPError as e:
        print(e.response.text)

    try:

        jira.issue_create(
            fields={
                'project': {
                    'key': 'Simulated Project 1' 
                },
                'summary': 'Testing JIRA python API',
                'description': 'testing',
                'issuetype': {
                    "name": "Task"
                },

            }
        )
    except HTTPError as e:
        print(e.response.text)

if __name__ == '__main__':
    main()
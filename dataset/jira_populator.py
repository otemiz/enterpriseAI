import os
from atlassian import Jira
from requests import HTTPError

# pip install atlassian-python-api
# pip install jira


def main():

    # jira = Jira(
    #     url   = "https://oguzhanilter.atlassian.net",
    #     token = os.environ.get('JIRA_API_KEY')    )
    # 

    jira = Jira(
        url="https://oguzhanilter.atlassian.net",
        username="oguzhanilter@gmail.com",  # Replace with your Jira username
        password=os.environ.get('JIRA_API_KEY')  # Use environment variable for your API token
    )


    try:
        proj = jira.get_project('SCRUM', expand='projectKeys')
        print(proj.get("SCRUM")) # to verify that authentication worked
    
    except HTTPError as e:
        print(e.response.text)

    try:
        jira.issue_create(
            fields={
                'project': {
                    'key': 'SCRUM' 
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
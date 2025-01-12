from jira import JIRA 
import os
  
# Specify a server key. It should be your 
# domain name link. yourdomainname.atlassian.net 
jiraOptions = {'server': "https://oguzhanilter.atlassian.net"} 
  
# Get a JIRA client instance, pass, 
# Authentication parameters 
# and the Server name. 
# emailID = your emailID 
# token = token you receive after registration 

JIRA_API_KEY = os.environ.get('JIRA_API_KEY')

jira = JIRA(options=jiraOptions, basic_auth=( 
    "oguzhaniltr@gmail.com ", JIRA_API_KEY)) 
  
# Search all issues mentioned against a project name. 
for singleIssue in jira.search_issues(jql_str='project = Simulated Project 1'): 
    print('{}: {}:{}'.format(singleIssue.key, singleIssue.fields.summary, 
                             singleIssue.fields.reporter.displayName)) 
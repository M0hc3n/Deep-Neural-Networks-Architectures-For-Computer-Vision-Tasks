import os
import requests
from langchain_huggingface import HuggingFaceEndpoint
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Set up the LLM
llm = HuggingFaceEndpoint(
    repo_id='mistralai/Mistral-7B-Instruct-v0.2',
    huggingfacehub_api_token=os.environ['HUGGINGFACE_API_KEY']
)

# Get PR details
pr_number = os.environ['GITHUB_EVENT_NUMBER']
repo = os.environ['GITHUB_REPOSITORY']
api_url = f'https://api.github.com/repos/{repo}/pulls/{pr_number}'
headers = {'Authorization': f'token {os.environ["GITHUB_TOKEN"]}'}
response = requests.get(api_url, headers=headers)
pr_data = response.json()

# Get PR diff
diff_url = pr_data['diff_url']
diff_response = requests.get(diff_url, headers=headers)
diff_content = diff_response.text

# Create a prompt template
template = '''
Analyze the following pull request diff and provide a concise, documented list of changes:

{diff}

List of changes:
'''

prompt = PromptTemplate(template=template, input_variables=['diff'])

# Create and run the chain
chain = LLMChain(llm=llm, prompt=prompt)
result = chain.run(diff=diff_content)

# Post comment on PR
comment_url = f'https://api.github.com/repos/{repo}/issues/{pr_number}/comments'
comment_data = {'body': f'## PR Changes Summary\n\n{result}'}
requests.post(comment_url, json=comment_data, headers=headers)

print('PR inspection complete. Comment posted.')

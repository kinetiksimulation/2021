# js-project-6
Perfect Pipelines Development Repo

## Development

### Forking this repo
To create a fork of this repo, click the `fork` button in the top right of this webpage and select your organization (most likely just your github username).

### Cloning this repo
Once you have a fork of this repo, navigate to the fork and click the green *Code* button. In terminal, run `git clone {repo_url}`

### Creating a development branch
Once the repo is cloned, `cd` into it. Next, run `git checkout -b {branch_name}` to create a branch with name *{branch_name}*

### Submitting PR's
Once you are ready to submit a Pull Request, `cd` into your forked repo in terminal. Next, run the following commands:
- `git add {filename}` - do this for each of the files you wish to submit for review
- `git commit -s -m "Issue # - name of this issue"`
- `git push --set-upstream origin {branch_name}`

Next, on the main repo (https://github.com/Polber/js-project-6/), Navigate to the *Pull requests* tab and select *New pull request*.
* Make sure the base is this repo's master branch and the head is your fork and branch (click *compare across forks* if you don't see these options)

Click *Create pull request* and fill in the necessary info before submitting.

If you would like to edit a PR without creating a new commit, run the following:
- `git add {filename}` - do this for each of the files you wish to submit for review
- `git commit -a --amend'
- `git push -f origin {branch_name}`

## Zenhub
To use the Zenhub features, use Google Chrome and navigate to the following link to add the zenhub widget to chrome: https://chrome.google.com/webstore/detail/zenhub-for-github/ogcgkffhplmphkaahpmffcafajaocjbd?hl=en-US



## Notes to Add features
- add animation
- add more charts/info
- cleaner hover box formatting
- stacked bar charts with sources being one box on the bars
- dynamic statistics to decide the percentages of what teams are making it where in the pipeline
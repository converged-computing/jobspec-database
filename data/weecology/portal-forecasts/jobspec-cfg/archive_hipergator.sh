# Archive forecasts by pushing weekly forecasts to GitHub and tagging a
# release so that the GitHub-Zenodo integration archives the forecasts to
# Zenodo

# Only releases on cron driven events so that only weekly forecasts and not
# simple changes to the codebase triggers archiving.

current_date=`date -I | head -c 10`

# Copy version of portal_weekly_forecast.py used to run the forecast into the repo so we know what was run
cp ../portal_weekly_forecast.sh .
cp ../portal_dryrun_forecast.sh .

# Setup on weecologydeploy user
git config user.email "weecologydeploy@weecology.org"
git config user.name "Weecology Deploy Bot"

# Commit changes to portal-forecasts repo. Do not commit old forecasts files.
git checkout main
git add data/* models/* portal_weekly_forecast.sh portal_dryrun_forecast.sh
git ls-files ./forecasts --others --exclude-standard --directory | xargs git add
git add forecasts/forecasts_evaluations.zip forecasts/forecasts_metadata.csv forecasts/forecasts_results.csv

git commit -m "Update forecasts: HiperGator Build $current_date [ci skip]"

# Add deploy remote
# Needed to grant permissions through the deploy token
# Removing the remote ensures that updates to the GitHub Token are added to the remote
git remote remove deploy
git remote add deploy https://${GITHUB_TOKEN}@github.com/weecology/portal-forecasts.git

# Create a new portal-forecasts tag for release
git tag $current_date

# If this is a cron event deploy, otherwise just check if we can

# Push updates to upstream
git push --quiet deploy main

# Create a new portal-forecasts release to trigger Zenodo archiving
git push --quiet deploy --tags
curl -v -i -X POST -H "Content-Type:application/json" -H "Authorization: token $GITHUB_TOKEN" https://api.github.com/repos/weecology/portal-forecasts/releases -d "{\"tag_name\":\"$current_date\"}"


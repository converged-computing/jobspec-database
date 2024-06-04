#!/bin/bash
# Set argument
export MESSAGE=$@

# Get root directory of git repo
ROOT_DIR=$(git rev-parse --show-toplevel)

function get_changed_files(){
    # Get changed files
    export  CHANGED_FILES=($(git status --porcelain | grep -v "^?" | awk '{print $2}'))
    # Announce changed files
    for FILE in ${CHANGED_FILES[@]}; do
        echo $FILE
    done
}

function get_group_id(){
    # Set argument
    ARG=$@
    # Get group id from the file path in ARG and add it to the GROUP variable.
    # The group ID is the two numbers after the word "group-" in the file path.
    export GROUP=$(echo ${ARG} | perl -nle 'print for m/(?<=group-)[0-9]{2}/g' | sort -u)
    echo $GROUP
}

function get_app_name(){
    # Set argument
    ARG=$@
    # Get application name from file path in ARG and add it to the
    # APP variable. The application name is the word after the group id in the file path.
    export APP=$(echo ${ARG} | perl -nle 'print for m/(?<=group-[0-9]{2}\/)[a-z-]+/g' | sort -u)
    echo $APP
}

function check_git_up_to_date(){
    # If branch is up to date with origin/master then exit
    git fetch origin
    git diff --quiet origin/main
    if [ $? -eq 0 ]; then
        echo "Branch is up to date with origin/main"
        exit 0
    fi
}

function find_unencrypted_secrets(){
    # Find all unencrypted secrets & sensitive configurations in the repository and encrypt them using ./encrypt.sh
    echo "==============================================================================="
    echo " Finding unencrypted secrets & sensitive configurations"
    echo "==============================================================================="
    # Find all files that end with .sops.yaml in the repo, ignoring all file name literals in the IGNORE array
    unset IGNORE
    IGNORE=('./.sops.yaml')
    unset FILES_TO_CHECK
    FILES_TO_CHECK=($(find . -type f -name "*.sops.yaml" | grep -v -e "${IGNORE[@]}"))

    # Check to see if each file is encrypted, if not then encrypt it
    $ROOT_DIR/encrypt $ROOT_DIR/apps


}

function run_precommit_all(){
    # Set exit to 0
    EXIT=0
    # Announce start of pre-commit run
    echo "==============================================================================="
    echo " Running pre-commit over all files"
    echo "==============================================================================="
    # Run pre-commit over the whole repository until it exits with zero
    for i in {1..5}; do
        pre-commit run --all-files
        EXIT=$?
        if [ $EXIT -eq 0 ]; then
            echo "==============================================================================="
            echo " Finished pre-commit over all files"
            echo "==============================================================================="
            break
        fi
    done

}

function push_to_github(){
    # Announce push to github
    echo "==============================================================================="
    echo " Pushing to github"
    echo "==============================================================================="

    # Find all unencrypted secrets in the repository and encrypt them using ./encrypt.sh
    #find_unencrypted_secrets

    # Run pre-commit over the whole repository until it exits with zero
    run_precommit_all
    run_precommit_all

    # Add changed files to git
    echo " Adding changed files to git"
    git add -A

    # Get Changed Files
    echo " Getting changed files"
    FILES=($(get_changed_files))

    # If no files have changed then exit cleanly
    if [ -z $FILES ]; then
        echo " No files have changed - exiting cleanly."
        exit 0
    else
        echo " Changed files found:"
        for FILE in ${FILES[@]}; do
            echo "                 $FILE"
        done
    fi

    # Find duplicates in the FILES array and return a number of duplicates.
    DUPLICATES=$(echo "${FILES[@]}" | tr ' ' '\n' | sort | uniq -c | grep -v "^\s*1\s" | wc -l)

    # If $DUPLICATES is greater than 0 then find and remove the duplicates from the FILES array.
    if [ $DUPLICATES -gt 0 ]; then
        echo " Found $DUPLICATES duplicate files"
        echo " Removing duplicate files"
        FILES=($(echo "${FILES[@]}" | tr ' ' '\n' | sort | uniq -u | tr '\n' ' '))
    else
        echo " No duplicate files found"
    fi

    # Ensure no files matched by values in .gitignore are added to the FILES
    # array
    for FILE in ${FILES[@]}; do
        # If the file is matched by a value in .gitignore then remove it from
        # the FILES array
        if [[ $(grep -c $FILE .gitignore) -gt 0 ]]; then
            echo "Removing $FILE from FILES array - file found in .gitignore"
            FILES=(${FILES[@]/$FILE})
        fi
    done



    # Count number of files in FILES array
    export NUM_FILES=${#FILES[@]}
    # Announce number of files to process
    echo "==============================================================================="
    echo " Number of files to process: $NUM_FILES"
    echo "==============================================================================="


    # Loop through new and changed files
    for FILE in ${FILES[@]}; do
        # If FILE is empty then exit cleanly
        if [ -z $FILE ]; then
            echo "No more files to process"
            exit 0
        fi

        # Announce file being processed
        echo "Processing file:          $FILE"
        # Get group id from file path
        GROUP=$(get_group_id $FILE)

        # If GROUP is empty set it to ROOT
        if [ -z $GROUP ]; then
            GROUP=ROOT
        fi

        # Announce Group
        echo "Group:                    $GROUP"

        # Get application name from file path
        APP=$(get_app_name $FILE)
        # If APP is empty then set it to $PROJECT
        if [ -z $APP ]; then
            APP=$PROJECT
        fi

        # Announce application if $APP is defined
        if [ ! -z $APP ]; then
            echo "Application:              $APP"
        else
            echo "ERROR:                    Application not set"
        fi

        # Announce file added to git if the exit code of the last command was 0
        if [ $? -eq 0 ]; then
            echo "Added file to git:        $FILE"
        else
            echo "Failed to add file to git: $FILE"
        fi

        # set MESSAGE to "Updated $FILE"
        MESSAGE="Updated $FILE"

        # Announce commit file & message
        echo "Committing file:          $FILE"
        echo "Commit message:           $GROUP-$APP:    $MESSAGE"
        echo

        # Capture the output of the git commit command. If it contains "nothing
        # to commit, working tree clean" then exit cleanly. Also capture its
        # exit errorlevel to ERR
        export OUTPUT=$(git commit -m "$GROUP-$APP:    $MESSAGE")
        export ERR=$?
        if [[ $OUTPUT == *"nothing to commit, working tree clean"* ]]; then
            echo "Nothing to commit, working tree clean"
            exit 0
        fi

        # Announce commit output
        echo "$OUTPUT"

        # If the commit failed then run push_to_github again for a maximum of 5
        # retries

        if [ $ERR -ne 0 ]; then
            if [ $COUNTER -lt 5 ]; then
                echo "==============================================================================="
                echo "Retrying push to github - Attempt #$COUNTER"
                echo "==============================================================================="
                push_to_github
                export COUNTER=$(($COUNTER+1))
            fi
        fi
        # If the commit was successful then push to git
        if [ $ERR -eq 0 ]; then
            git push
        fi

    done

    # Reconcile flux kustomization
    #flux_reconcile ${FILES[@]}
}

function flux_reconcile(){
    # Set argument
    #CHANGED_FILES=()
    CHANGED_FILES=$@

    # Announce flux reconciliation
    echo "==============================================================================="
    echo " Reconciling flux source"
    echo "==============================================================================="

    # Reconcile flux source
    flux reconcile source git flux-system

    # Announce flux reconciliation
    echo "==============================================================================="
    echo " Reconciling flux kustomization"
    echo "==============================================================================="
    # Reconcile flux kustomization
    flux reconcile kustomization flux-system

    # Get list of changed kustomization names by iterating through the changed
    # files and extracting the unique list of kustomization names. To get the
    # kustomizations to reconcile inspect the file names as follows:
    #   - The file name is apps/group-00/cluster-config/ks-cluster-config.yaml
    #   - This makes the GROUP ID "00"
    #   - This makes the APP NAME "cluster-config"
    #   - Therefore the kustomization name is "00-cluster-config"
    # If the GROUP is ROOT then skip reconciliation of the kustomization
    # If the APP is kustomization then skip reconciliation of the kustomization
    for FILE in ${CHANGED_FILES[@]}; do
        # Announce file being processed
        echo "Processing file:          $FILE"
        # Get group id from file path
        GROUP=$(get_group_id $FILE)
        # If GROUP is empty set it to ROOT
        if [ -z $GROUP ]; then
            GROUP=ROOT
        fi

        # Announce Group
        echo "Group:                    $GROUP"

        # Get application name from file path
        APP=$(get_app_name $FILE)
        # If APP is empty then set it to $PROJECT
        if [ -z $APP ]; then
            APP=$PROJECT
        fi

        # Announce application if $APP is defined
        if [ ! -z $APP ]; then
            echo "Application:              $APP"
        else
            echo "ERROR:                    Application not set"
        fi

        # If the GROUP is not ROOT and the APP is not kustomization then
        # reconcile the kustomization
        if [ $GROUP != "ROOT" ] && [ $APP != "kustomization" ]; then
            echo -e "Reconciling kustomization $GROUP-$APP"
            echo
            # Reconcile the kustomization
            flux reconcile kustomization $GROUP-$APP
        else
            echo "Skipping reconciliation of kustomization $GROUP-$APP"
        fi
    done
}

# Set COUNTER to 0
export COUNTER=0

# Get project name from git repo
PROJECT=$(basename $(git rev-parse --show-toplevel))

# Find and encrypt all unencrypted secrets in the repository
#find_unencrypted_secrets

# Add changed files to git
push_to_github

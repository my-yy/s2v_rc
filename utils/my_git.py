import git

is_commited = False


def commit_v2(args):
    import datetime
    date_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    content = "%s %s %s" % (args.project, args.name, date_str)
    commit(content)


def commit(content):
    global is_commited
    if is_commited:
        return

    do_real_commit(content)
    is_commited = True
    return get_git_info()


def do_real_commit(content):
    repo = git.Repo(search_parent_directories=True)
    try:
        g = repo.git
        g.add("--all")
        res = g.commit("-m " + content)
        print(res)
    except Exception as e:
        print("no need commit")


def get_git_info():
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    branch = str(repo.active_branch)
    return {"branch": branch, "git_id": sha}

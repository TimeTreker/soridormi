# Create the GitHub repository

## Option A: GitHub CLI

From inside the project directory:

```bash
git init
./scripts/add_submodules.sh
./scripts/setup_env.sh

git add .
git commit -m "Initial Soridormi robot development stack"

gh auth login
gh repo create soridormi --private --source=. --remote=origin --push
```

For a public repo, replace `--private` with `--public`.

## Option B: GitHub website

1. Create an empty repository named `soridormi` on GitHub.
2. Do not initialize it with README/license/gitignore because this project already has them.
3. Run:

```bash
git init
./scripts/add_submodules.sh
./scripts/setup_env.sh

git add .
git commit -m "Initial Soridormi robot development stack"
git branch -M main
git remote add origin git@github.com:YOUR_USERNAME/soridormi.git
git push -u origin main
```

## Clone later

```bash
git clone --recurse-submodules git@github.com:YOUR_USERNAME/soridormi.git
cd soridormi
./scripts/setup_env.sh
```

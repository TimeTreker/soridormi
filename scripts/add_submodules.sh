#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [ ! -d .git ]; then
  echo "This directory is not a git repo yet. Run: git init"
  exit 1
fi

mkdir -p workspace
rm -f workspace/.gitkeep

add_or_update() {
  local branch="$1"
  local url="$2"
  local path="$3"

  if [ -d "$path/.git" ] || git config --file .gitmodules --get-regexp "submodule\..*\.path" 2>/dev/null | grep -q " $path$"; then
    echo "Submodule already exists: $path"
  else
    if [ -n "$branch" ]; then
      git submodule add -b "$branch" "$url" "$path"
    else
      git submodule add "$url" "$path"
    fi
  fi
}

add_or_update v2 https://github.com/apirrone/Open_Duck_Mini.git workspace/Open_Duck_Mini
add_or_update v2 https://github.com/apirrone/Open_Duck_Mini_Runtime.git workspace/Open_Duck_Mini_Runtime
add_or_update "" https://github.com/apirrone/Open_Duck_Playground.git workspace/Open_Duck_Playground

git submodule update --init --recursive

echo "Submodules are ready."

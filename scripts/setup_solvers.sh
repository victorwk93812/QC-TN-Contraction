#!/bin/sh
# Build the treewidth solvers behind five of the paper's seven plan generators (phase 3), pinned.
# Each source is cloned into its own directory under external/solvers/ and built there with its own
# compiler command; nothing from the clones is executed except the built solver. Idempotent.
#
#   fc             FlowCutter PACE 2017 at the revision FlowCutterPACE17_jll v0.0.2+0 was built from
#                  (the jll README in the Julia depot names kit-algo/flow-cutter-pace17 @ fd405a2)
#   tamaki         Tamaki et al., PACE 2017 Track A heuristic (tw.heuristic.MainDecomposer)
#   jdrasil        Jdrasil (Bannach, Berndt, Ehlers), heuristic (jdrasil.Heuristic)
#   minfill_mrs    Jegou, Kanso, Terrioux, PACE 2017 heuristic track
#   minfillbg_mrs  same authors, PACE 2017 heuristic track
#
# The repositories of the last two are the ones the PACE challenge lists
# (https://github.com/PACE-challenge/Treewidth). Needs g++ and the JDK of scripts/setup_jdk.sh.
set -e
MAIN=$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")
EXT=${QCTN_EXTERNAL:-$MAIN/external}
S=$EXT/solvers
JDK=${JDK:-$HOME/.local/opt/jdk-21.0.13+9}
mkdir -p "$S"

fetch() {  # name url commit
  if [ ! -d "$S/$1/.git" ]; then
    git clone -q "$2" "$S/$1"
  fi
  git -C "$S/$1" checkout -q "$3"
  echo "$1 $(git -C "$S/$1" rev-parse HEAD)"
}

fetch flow-cutter-pace17 https://github.com/kit-algo/flow-cutter-pace17 fd405a2b003c9a93838e0488728addc0610a769f
fetch PACE2017-TrackA https://github.com/TCS-Meiji/PACE2017-TrackA 7278390fe81191f238206b822fa2941d068a1214
fetch Jdrasil https://github.com/maxbannach/Jdrasil eafaa4cc76df9d938ac85b88e5c4218a4fbb6c1a
fetch minfill_mrs https://github.com/td-mrs/minfill_mrs c1aa4f32ab3350d027ed0e932183f5f649b2ea88
fetch minfillbg_mrs https://github.com/td-mrs/minfillbg_mrs 7aa09099ad4e9e51e47317c77f39664d07d921ed

# FlowCutter: the command of its build.sh, plus `-include string` (list_graph.h relies on a transitive
# <string> include that GCC 11 no longer provides)
(cd "$S/flow-cutter-pace17" && [ -x flow_cutter_pace17 ] ||
  g++ -Wall -std=c++11 -O3 -DNDEBUG -include string src/*.cpp -o flow_cutter_pace17)
# Tamaki: `make heuristic`
(cd "$S/PACE2017-TrackA" && [ -f tw/heuristic/MainDecomposer.class ] ||
  "$JDK/bin/javac" tw/heuristic/*.java)
# Jdrasil: the core sources, without gradle (which would download itself)
(cd "$S/Jdrasil" && [ -f build/classes/jdrasil/Heuristic.class ] ||
  { mkdir -p build/classes && "$JDK/bin/javac" -nowarn -d build/classes \
      $(find subprojects/core/src/main/java -name "*.java"); })
# The two MinFill heuristics: their Makefiles
(cd "$S/minfill_mrs" && [ -x minfill_mrs ] || make -s minfill_mrs)
(cd "$S/minfillbg_mrs" && [ -x minfillbg_mrs ] || make -s minfillbg_mrs)
echo "solvers built in $S"
